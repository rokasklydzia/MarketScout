"""Universe construction from iShares ETF holdings files.

US  : iShares Russell 1000 ETF (IWB)            ~1,000 names
EU  : iShares STOXX Europe 600 UCITS ETF (EXSA) ~600 names

Each run maps holdings to Yahoo symbols and saves a dated snapshot, so
index membership history accumulates over time (survivorship-bias fix
going forward).
"""
from __future__ import annotations

import csv
import io
import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .config import Settings
from .storage import Store

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (market-scout research)"}

# Exchange name (lower-case substring) -> Yahoo suffix. Checked in order.
EXCHANGE_SUFFIX: list[tuple[str, str]] = [
    ("london", ".L"),
    ("xetra", ".DE"),
    ("deutsche b", ".DE"),
    ("frankfurt", ".DE"),
    ("paris", ".PA"),
    ("amsterdam", ".AS"),
    ("brussels", ".BR"),
    ("lisbon", ".LS"),
    ("dublin", ".IR"),
    ("swiss", ".SW"),
    ("six ", ".SW"),
    ("borsa italiana", ".MI"),
    ("milan", ".MI"),
    ("madrid", ".MC"),
    ("bolsa", ".MC"),
    ("stockholm", ".ST"),
    ("copenhagen", ".CO"),
    ("helsinki", ".HE"),
    ("oslo", ".OL"),
    ("vienna", ".VI"),
    ("wiener", ".VI"),
    ("warsaw", ".WA"),
]

# Country (Location column) -> Yahoo suffix; fallback when exchange is ambiguous
# (e.g. "Nasdaq Omx Nordic" covers Stockholm, Copenhagen and Helsinki).
COUNTRY_SUFFIX = {
    "united kingdom": ".L", "germany": ".DE", "france": ".PA", "netherlands": ".AS",
    "belgium": ".BR", "portugal": ".LS", "ireland": ".IR", "switzerland": ".SW",
    "italy": ".MI", "spain": ".MC", "sweden": ".ST", "denmark": ".CO",
    "finland": ".HE", "norway": ".OL", "austria": ".VI", "poland": ".WA",
    "luxembourg": ".AS",
}

US_EXCHANGE_HINTS = ("nasdaq", "new york", "nyse", "cboe", "bats", "arca")


# --------------------------------------------------------------------------
# Fetch + parse
# --------------------------------------------------------------------------
def fetch_holdings_text(url: str, manual_file: Path | None, timeout: int = 30) -> str:
    """Download a holdings CSV; fall back to a manually saved file."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        text = r.content.decode("utf-8-sig", errors="replace")
        if "Ticker" not in text[:5000]:
            raise ValueError("Response does not look like an iShares holdings CSV")
        if manual_file:  # keep a copy of the last good download
            manual_file.parent.mkdir(parents=True, exist_ok=True)
            manual_file.write_text(text, encoding="utf-8")
        return text
    except Exception as exc:  # noqa: BLE001
        if manual_file and manual_file.exists():
            log.warning("Download failed (%s); using saved file %s", exc, manual_file)
            return manual_file.read_text(encoding="utf-8-sig")
        raise RuntimeError(
            f"Could not download holdings from {url} and no saved file at {manual_file}. "
            "Download the CSV in a browser and save it there."
        ) from exc


def parse_ishares_csv(text: str) -> pd.DataFrame:
    """Parse an iShares holdings CSV, skipping the preamble and footer."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.lstrip('"').startswith("Ticker")), None)
    if start is None:
        raise ValueError("Header row starting with 'Ticker' not found")
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    header = [h.strip() for h in rows[0]]
    body = [r for r in rows[1:] if len(r) == len(header) and r[0].strip() not in ("", "-")]
    df = pd.DataFrame(body, columns=header)
    df.columns = [c.lower().replace(" (%)", "_pct").replace(" ", "_") for c in df.columns]
    return df


# --------------------------------------------------------------------------
# Symbol mapping
# --------------------------------------------------------------------------
def _clean_ticker(t: str) -> str:
    t = t.strip().upper().rstrip(".")
    return re.sub(r"[\s./]+", "-", t)


def yahoo_suffix(exchange: str, country: str) -> str | None:
    ex = (exchange or "").lower()
    for key, suf in EXCHANGE_SUFFIX:
        if key in ex:
            return suf
    return COUNTRY_SUFFIX.get((country or "").strip().lower())


def map_symbol(region: str, ticker: str, exchange: str, country: str) -> str | None:
    if region == "us":
        return _clean_ticker(ticker)
    suf = yahoo_suffix(exchange, country)
    return f"{_clean_ticker(ticker)}{suf}" if suf else None


def load_overrides(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    o = pd.read_csv(path, dtype=str).fillna("")
    return {(r.region, r.source_ticker.strip().upper()): r.yahoo_symbol.strip() for r in o.itertuples()}


def build_region(df: pd.DataFrame, region: str, overrides: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mapped universe rows, unmapped rows)."""
    df = df.copy()
    if "asset_class" in df:
        df = df[df["asset_class"].str.strip().str.lower() == "equity"]
    if region == "us" and "exchange" in df:
        # IWB holds a few non-US listings and placeholders; keep US exchanges.
        ex = df["exchange"].str.lower()
        df = df[ex.apply(lambda e: any(h in e for h in US_EXCHANGE_HINTS))]

    def _map(r) -> str | None:
        key = (region, r["ticker"].strip().upper())
        if key in overrides:
            return overrides[key] or None
        return map_symbol(region, r["ticker"], r.get("exchange", ""), r.get("location", ""))

    df["symbol"] = df.apply(_map, axis=1)
    weight_col = "weight_pct" if "weight_pct" in df else None
    out = pd.DataFrame({
        "region": region,
        "source_ticker": df["ticker"].str.strip(),
        "symbol": df["symbol"],
        "name": df.get("name", ""),
        "sector": df.get("sector", ""),
        "country": df.get("location", ""),
        "exchange": df.get("exchange", ""),
        "currency": df.get("market_currency", df.get("currency", "")),
        "index_weight": pd.to_numeric(df[weight_col].str.replace(",", ""), errors="coerce")
        if weight_col else float("nan"),
    })
    unmapped = out[out["symbol"].isna()]
    out = out.dropna(subset=["symbol"]).drop_duplicates("symbol")
    return out, unmapped


def build_universe(settings: Settings, asof: date | None = None) -> pd.DataFrame:
    asof = asof or date.today()
    store = Store(settings)
    ucfg = settings["universe"]
    overrides = load_overrides(settings.path(ucfg["overrides_file"]))
    parts, unmapped_all = [], []
    for region, src in ucfg["sources"].items():
        text = fetch_holdings_text(src["url"], settings.path(src["manual_file"]))
        mapped, unmapped = build_region(parse_ishares_csv(text), region, overrides)
        log.info("%s: %d symbols mapped, %d unmapped", src["name"], len(mapped), len(unmapped))
        parts.append(mapped)
        unmapped_all.append(unmapped)
    uni = pd.concat(parts, ignore_index=True)
    uni.insert(0, "asof", pd.Timestamp(asof))
    store.write(uni, store.universe_path(asof))
    um = pd.concat(unmapped_all, ignore_index=True)
    if not um.empty:
        um.to_csv(store.p("universe", f"unmapped_{asof:%Y-%m-%d}.csv"), index=False)
        log.warning("%d holdings could not be mapped; see data/universe/unmapped_*.csv "
                    "and add them to config/symbol_overrides.csv", len(um))
    return uni


def benchmark_rows(settings: Settings) -> pd.DataFrame:
    b = settings["universe"]["benchmarks"]
    return pd.DataFrame({
        "region": list(b.keys()),
        "symbol": list(b.values()),
        "currency": ["USD" if r == "us" else "EUR" for r in b],
    })

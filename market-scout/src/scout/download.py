"""Price, FX and currency-metadata download via yfinance.

yfinance is an unofficial scraper: pin the version, and keep all calls
behind `PriceSource` so a paid vendor can replace it later.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Protocol

import pandas as pd

from .config import Settings
from .storage import Store

log = logging.getLogger(__name__)


class PriceSource(Protocol):
    def history(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Return long OHLCV frame: date, symbol, open, high, low, close, adj_close,
        volume, dividends, splits. Missing symbols are simply absent."""

    def currencies(self, symbols: list[str]) -> dict[str, str]:
        """Return quote currency per symbol (e.g. 'USD', 'GBp')."""


# --------------------------------------------------------------------------
# yfinance implementation
# --------------------------------------------------------------------------
class YahooSource:
    FIELD_MAP = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
                 "Adj Close": "adj_close", "Volume": "volume",
                 "Dividends": "dividends", "Stock Splits": "splits"}

    def history(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            tickers=symbols, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=False, actions=True, group_by="ticker", threads=True,
            progress=False, timeout=30,
        )
        return self._to_long(raw, symbols)

    @classmethod
    def _to_long(cls, raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        frames = []
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = set(raw.columns.get_level_values(0))
            for sym in symbols:
                if sym not in lvl0:
                    continue
                part = raw[sym].copy()
                frames.append(cls._tidy(part, sym))
        else:  # single symbol, flat columns
            frames.append(cls._tidy(raw.copy(), symbols[0]))
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @classmethod
    def _tidy(cls, part: pd.DataFrame, sym: str) -> pd.DataFrame:
        part = part.rename(columns=cls.FIELD_MAP)
        for col in cls.FIELD_MAP.values():
            if col not in part:
                part[col] = 0.0 if col in ("dividends", "splits") else float("nan")
        part = part[list(cls.FIELD_MAP.values())].dropna(subset=["close"])
        if part.empty:
            return part
        idx = pd.to_datetime(part.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        part.index = idx.normalize()
        part = part.reset_index(names="date")
        part.insert(1, "symbol", sym)
        return part

    def currencies(self, symbols: list[str]) -> dict[str, str]:
        import yfinance as yf

        out = {}
        for sym in symbols:
            try:
                out[sym] = yf.Ticker(sym).fast_info["currency"]
            except Exception:  # noqa: BLE001
                out[sym] = None
            time.sleep(0.2)
        return out


# --------------------------------------------------------------------------
# Batched download with retries
# --------------------------------------------------------------------------
@dataclass
class DownloadResult:
    prices: pd.DataFrame
    failed: list[str] = field(default_factory=list)
    requested: int = 0

    @property
    def failure_share(self) -> float:
        return len(self.failed) / self.requested if self.requested else 0.0


def _batches(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def download_history(source: PriceSource, symbols: list[str], start: date, end: date,
                     batch_size: int = 50, pause: float = 2.0, max_retries: int = 3) -> DownloadResult:
    got, pending = [], list(dict.fromkeys(symbols))
    for attempt in range(1, max_retries + 1):
        if not pending:
            break
        size = max(1, batch_size // attempt)  # smaller batches on retries
        failed_this_round = []
        for i, batch in enumerate(_batches(pending, size)):
            try:
                df = source.history(batch, start, end)
            except Exception as exc:  # noqa: BLE001
                log.warning("Batch %d failed on attempt %d: %s", i, attempt, exc)
                df = pd.DataFrame()
            have = set(df["symbol"]) if not df.empty else set()
            failed_this_round += [s for s in batch if s not in have]
            if not df.empty:
                got.append(df)
            if pause:
                time.sleep(pause)
        pending = failed_this_round
        if pending:
            log.info("Attempt %d: %d symbols missing, retrying", attempt, len(pending))
            if pause:
                time.sleep(pause * 2 ** attempt)
    prices = pd.concat(got, ignore_index=True) if got else pd.DataFrame()
    return DownloadResult(prices=prices, failed=pending, requested=len(set(symbols)))


# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------
def update_prices(settings: Settings, universe: pd.DataFrame, source: PriceSource | None = None,
                  full: bool = False, end: date | None = None) -> DownloadResult:
    """Full backfill (full=True or empty store) or incremental update."""
    source = source or YahooSource()
    store = Store(settings)
    dcfg = settings["download"]
    end = end or date.today()
    last = None if full else store.last_price_date()
    if last is None:
        start = end - timedelta(days=int(365.25 * dcfg["history_years"]))
        log.info("Full backfill from %s", start)
    else:
        start = (last - pd.tseries.offsets.BDay(dcfg["incremental_overlap_days"])).date()
        log.info("Incremental update from %s", start)

    regions = universe.set_index("symbol")["region"].to_dict()
    res = download_history(source, list(regions), start, end, dcfg["batch_size"],
                           dcfg["pause_seconds"], dcfg["max_retries"])
    if not res.prices.empty:
        res.prices["region"] = res.prices["symbol"].map(regions)
        store.write_raw_prices(res.prices)

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
    pd.DataFrame({"symbol": res.failed, "run": stamp}).to_csv(
        store.p("raw", "logs", f"failures_{stamp}.csv"), index=False)
    status = "partial" if res.failure_share > dcfg["partial_run_threshold"] else "ok"
    log.info("Prices: %d rows, %d/%d symbols failed (%.1f%%) -> run %s",
             len(res.prices), len(res.failed), res.requested, 100 * res.failure_share, status)
    return res


def update_currencies(settings: Settings, symbols: list[str], source: PriceSource | None = None
                      ) -> pd.DataFrame:
    """Cache Yahoo quote currency per symbol (needed for GBp vs GBP)."""
    source = source or YahooSource()
    store = Store(settings)
    path = store.p("reference", "currencies.parquet")
    cur = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["symbol", "currency"])
    missing = [s for s in symbols if s not in set(cur["symbol"])]
    if missing:
        log.info("Fetching quote currency for %d symbols", len(missing))
        new = source.currencies(missing)
        cur = pd.concat([cur, pd.DataFrame({"symbol": list(new), "currency": list(new.values())})],
                        ignore_index=True).drop_duplicates("symbol", keep="last")
        store.write(cur, path)
    return cur


def update_fx(settings: Settings, source: PriceSource | None = None, end: date | None = None
              ) -> pd.DataFrame:
    """Daily EURxxx rates (units of xxx per 1 EUR)."""
    source = source or YahooSource()
    store = Store(settings)
    end = end or date.today()
    start = end - timedelta(days=int(365.25 * settings["download"]["history_years"]) + 10)
    pairs = settings["download"]["fx_pairs"]
    res = download_history(source, pairs, start, end, batch_size=len(pairs), pause=0, max_retries=2)
    if res.prices.empty:
        raise RuntimeError("FX download returned nothing")
    fx = res.prices[["date", "symbol", "close"]].rename(columns={"symbol": "pair", "close": "rate"})
    fx["ccy"] = fx["pair"].str.slice(3, 6)
    store.write(fx, store.p("raw", "fx", "fx_daily.parquet"))
    return fx

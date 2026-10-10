"""Data quality checks, currency normalisation and EUR conversion."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Settings
from .storage import Store

log = logging.getLogger(__name__)

# Yahoo minor-unit currencies -> (major currency, divisor)
MINOR_UNITS = {"GBp": ("GBP", 100.0), "GBX": ("GBP", 100.0), "ZAc": ("ZAR", 100.0),
               "ILA": ("ILS", 100.0)}


def resolve_currency(symbol: str, yahoo_ccy: str | None, universe_ccy: str | None) -> tuple[str, float]:
    """Return (ISO currency, divisor to convert quoted prices into it)."""
    if yahoo_ccy:
        if yahoo_ccy in MINOR_UNITS:
            return MINOR_UNITS[yahoo_ccy]
        return yahoo_ccy.upper(), 1.0
    # Fallback heuristic: London listings on Yahoo are quoted in pence by default.
    if symbol.endswith(".L") and (universe_ccy or "GBP").upper() == "GBP":
        return "GBP", 100.0
    return (universe_ccy or "USD").upper(), 1.0


def eur_rates(fx: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Wide frame (date x ccy) of units-per-EUR, forward-filled onto `dates`."""
    wide = fx.pivot_table(index="date", columns="ccy", values="rate").sort_index()
    wide.index = pd.to_datetime(wide.index)
    idx = wide.index.union(pd.DatetimeIndex(dates)).sort_values()
    wide = wide.reindex(idx).ffill().bfill()
    wide["EUR"] = 1.0
    return wide


def symbol_checks(g: pd.DataFrame, expected_dates: pd.DatetimeIndex, cfg: dict) -> dict:
    g = g.sort_values("date")
    d = pd.DatetimeIndex(g["date"])
    span = expected_dates[(expected_dates >= d.min()) & (expected_dates <= d.max())]
    missing_share = 1 - len(d.intersection(span)) / max(len(span), 1)
    pos = np.searchsorted(span, d)
    max_gap = int(np.diff(pos).max() - 1) if len(pos) > 1 else 0

    close = g["close"].to_numpy()
    vol = g["volume"].fillna(0).to_numpy()
    same = np.r_[False, close[1:] == close[:-1]] & (vol == 0)
    stale_runs = _max_run(same)

    ret = g["adj_close"].pct_change().to_numpy()
    split_days = pd.DatetimeIndex(g.loc[g["splits"].fillna(0) > 0, "date"])
    big = np.where(np.abs(ret) > cfg["split_error_return"])[0]
    split_errors = sum(
        1 for i in big
        if not any(abs((d[i] - s).days) <= 3 for s in split_days)
    )
    return {
        "first_date": d.min(), "last_date": d.max(), "n_days": len(d),
        "missing_share": round(missing_share, 4), "max_gap_days": max_gap,
        "max_stale_run": stale_runs, "split_errors": split_errors,
    }


def _max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def clean_prices(settings: Settings, raw: pd.DataFrame, universe: pd.DataFrame,
                 currencies: pd.DataFrame, fx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean daily prices, per-symbol QA report)."""
    cfg = settings["qa"]
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    n0 = len(df)
    bad = (df["close"] <= 0) | (df["adj_close"] <= 0) | (df["high"] < df["low"])
    df = df[~bad].drop_duplicates(["symbol", "date"], keep="last")
    log.info("QA: dropped %d invalid rows of %d", n0 - len(df), n0)

    # --- currency normalisation --------------------------------------------
    ymap = dict(zip(currencies["symbol"], currencies["currency"]))
    umap = dict(zip(universe["symbol"], universe.get("currency", pd.Series(dtype=str))))
    meta = {s: resolve_currency(s, ymap.get(s), umap.get(s)) for s in df["symbol"].unique()}
    df["ccy"] = df["symbol"].map(lambda s: meta[s][0])
    div = df["symbol"].map(lambda s: meta[s][1])
    for c in ("open", "high", "low", "close", "adj_close", "dividends"):
        df[c] = df[c] / div

    # --- EUR conversion ------------------------------------------------------
    rates = eur_rates(fx, pd.DatetimeIndex(df["date"].unique()))
    missing_ccy = sorted(set(df["ccy"]) - set(rates.columns))
    if missing_ccy:
        log.warning("No FX rate for %s; those symbols are dropped", missing_ccy)
        df = df[~df["ccy"].isin(missing_ccy)]
    stacked = rates.stack().rename("eur_rate").reset_index()
    stacked.columns = ["date", "ccy", "eur_rate"]
    df = df.merge(stacked, on=["date", "ccy"], how="left")
    df["adj_close_eur"] = df["adj_close"] / df["eur_rate"]
    df["close_eur"] = df["close"] / df["eur_rate"]
    df["traded_value_eur"] = df["close_eur"] * df["volume"].fillna(0)

    # --- per-symbol checks ---------------------------------------------------
    reports = []
    for region, rg in df.groupby("region"):
        counts = rg.groupby("date")["symbol"].nunique()
        expected = pd.DatetimeIndex(counts[counts >= 0.5 * counts.max()].index).sort_values()
        for sym, g in rg.groupby("symbol"):
            r = symbol_checks(g, expected, cfg)
            r.update(symbol=sym, region=region, ccy=g["ccy"].iat[0])
            reports.append(r)
    qa = pd.DataFrame(reports)
    if qa.empty:
        return df, qa
    qa["flag_missing"] = qa["missing_share"] > cfg["max_missing_share"]
    qa["flag_gap"] = qa["max_gap_days"] > cfg["max_gap_days"]
    qa["flag_suspended"] = qa["max_stale_run"] >= cfg["stale_days"]
    qa["flag_split_error"] = qa["split_errors"] > 0
    qa["flag_short_history"] = qa["n_days"] < cfg["min_history_days"]
    qa["exclude_backtest"] = qa["flag_missing"] | qa["flag_split_error"]
    log.info("QA: %d symbols, %d excluded from backtests, %d short history",
             len(qa), int(qa["exclude_backtest"].sum()), int(qa["flag_short_history"].sum()))
    keep = ["date", "symbol", "region", "ccy", "open", "high", "low", "close", "adj_close",
            "volume", "dividends", "splits", "eur_rate", "close_eur", "adj_close_eur",
            "traded_value_eur"]
    return df[keep].sort_values(["symbol", "date"]).reset_index(drop=True), qa


def run_qa(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    store = Store(settings)
    raw = store.read_raw_prices()
    universe = store.latest_universe()
    cur_path = settings.data_dir / "reference" / "currencies.parquet"
    currencies = pd.read_parquet(cur_path) if cur_path.exists() else pd.DataFrame(
        columns=["symbol", "currency"])
    fx = pd.read_parquet(settings.data_dir / "raw" / "fx" / "fx_daily.parquet")
    clean, qa = clean_prices(settings, raw, universe, currencies, fx)
    store.write(clean, store.p("clean", "prices_daily.parquet"))
    store.write(qa, store.p("clean", "qa_report.parquet"))
    return clean, qa

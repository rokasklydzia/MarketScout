"""Feature library.

`build_panel` turns clean long prices into wide (date x symbol) frames, used by
the backtester. `snapshot` takes the last row of every feature for screening and
stores it as data/features/asof=YYYY-MM-DD.parquet.

Conventions
- Return-based features (returns, momentum, relative strength, volatility,
  drawdown) use EUR-converted adjusted prices: rankings are for a EUR investor.
- Trend and oscillator features (SMA ratios, RSI, ADX) use local prices.
- Every value at date t uses data up to and including t's close only. The
  backtester trades on t+1, so there is no look-ahead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TD_YEAR = 252


@dataclass
class Panel:
    px_eur: pd.DataFrame      # adjusted close in EUR
    px_local: pd.DataFrame    # adjusted close, local currency
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame       # unadjusted local close (for ADX with high/low)
    traded_value: pd.DataFrame  # EUR
    region: pd.Series         # symbol -> region
    bench: dict[str, str]     # region -> benchmark symbol

    @property
    def rets(self) -> pd.DataFrame:
        return self.px_eur.pct_change(fill_method=None)


def build_panel(clean: pd.DataFrame, benchmarks: dict[str, str]) -> Panel:
    def wide(col: str) -> pd.DataFrame:
        return clean.pivot_table(index="date", columns="symbol", values=col).sort_index()

    region = clean.drop_duplicates("symbol").set_index("symbol")["region"]
    return Panel(px_eur=wide("adj_close_eur"), px_local=wide("adj_close"), high=wide("high"),
                 low=wide("low"), close=wide("close"), traded_value=wide("traded_value_eur"),
                 region=region, bench=benchmarks)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def total_return(px: pd.DataFrame, n: int, skip: int = 0) -> pd.DataFrame:
    return px.shift(skip) / px.shift(n) - 1


def rsi(px: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up, dn = high.diff(), -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    tr = tr.reindex(close.index)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def rolling_beta(rets: pd.DataFrame, bench: pd.Series, n: int = TD_YEAR) -> pd.DataFrame:
    b = bench.reindex(rets.index)
    cov = rets.rolling(n, min_periods=n // 2).cov(b)
    var = b.rolling(n, min_periods=n // 2).var()
    return cov.div(var, axis=0)


def max_drawdown(px: pd.DataFrame, n: int = TD_YEAR) -> pd.DataFrame:
    peak = px.rolling(n, min_periods=20).max()
    dd = px / peak - 1
    return dd.rolling(n, min_periods=20).min()


def bench_series(panel: Panel, frame: pd.DataFrame) -> pd.DataFrame:
    """Benchmark value per symbol's region, broadcast to frame's shape."""
    out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for reg, b in panel.bench.items():
        cols = [c for c in frame.columns if panel.region.get(c) == reg]
        if b in frame and cols:
            out[cols] = np.repeat(frame[b].to_numpy()[:, None], len(cols), axis=1)
    return out


# --------------------------------------------------------------------------
# Feature panel
# --------------------------------------------------------------------------
def compute_features(panel: Panel) -> dict[str, pd.DataFrame]:
    px, loc = panel.px_eur, panel.px_local
    r = panel.rets
    f: dict[str, pd.DataFrame] = {}
    for name, n in {"1d": 1, "1w": 5, "1m": 21, "3m": 63, "6m": 126, "12m": 252}.items():
        f[f"ret_{name}"] = total_return(px, n)
    f["mom_12_1"] = total_return(px, 252, skip=21)
    f["mom_6_1"] = total_return(px, 126, skip=21)

    sma50, sma200 = loc.rolling(50, min_periods=40).mean(), loc.rolling(200, min_periods=160).mean()
    f["px_sma50"] = loc / sma50 - 1
    f["px_sma200"] = loc / sma200 - 1
    f["sma50_sma200"] = sma50 / sma200 - 1
    f["sma50_slope_1m"] = sma50 / sma50.shift(21) - 1
    f["dist_52w_high"] = loc / loc.rolling(252, min_periods=200).max() - 1
    f["adx_14"] = adx(panel.high, panel.low, panel.close)

    for name, n in {"3m": 63, "6m": 126, "12m": 252}.items():
        f[f"rs_{name}"] = f[f"ret_{name}"] - bench_series(panel, f[f"ret_{name}"])

    betas = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for reg, b in panel.bench.items():
        cols = [c for c in r.columns if panel.region.get(c) == reg]
        if b in r and cols:
            betas[cols] = rolling_beta(r[cols], r[b])
    f["beta_252"] = betas
    f["resid_mom_12_1"] = f["mom_12_1"] - betas * bench_series(panel, f["mom_12_1"])

    ann = np.sqrt(TD_YEAR)
    for n in (20, 60, 252):
        f[f"vol_{n}"] = r.rolling(n, min_periods=int(n * 0.8)).std() * ann
    f["downside_dev_60"] = r.clip(upper=0).rolling(60, min_periods=48).std() * ann
    f["max_dd_252"] = max_drawdown(px)

    f["rsi_14"] = rsi(loc)
    m20, s20 = loc.rolling(20).mean(), loc.rolling(20).std()
    f["zscore_20"] = (loc - m20) / s20
    f["reversal_1w"] = -f["ret_1w"]

    tv = panel.traded_value
    f["median_tv_20"] = tv.rolling(20, min_periods=15).median()
    f["volume_surge"] = tv.rolling(20, min_periods=15).mean() / tv.rolling(120, min_periods=90).mean()
    f["amihud_20"] = (r.abs() / tv.replace(0, np.nan)).rolling(20, min_periods=15).mean() * 1e6
    f["history_days"] = px.notna().cumsum()
    return f


def garch_vol(returns: pd.Series) -> float:
    """Annualised 1-day-ahead GARCH(1,1) volatility; NaN if it cannot be fitted."""
    try:
        from arch import arch_model
    except Exception:  # not installed, or blocked by Windows application control
        return np.nan
    x = returns.dropna().iloc[-750:] * 100
    if len(x) < 250:
        return np.nan
    try:
        res = arch_model(x, vol="GARCH", p=1, q=1, mean="Constant", rescale=False).fit(disp="off")
        return float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0]) / 100 * np.sqrt(TD_YEAR))
    except Exception:  # noqa: BLE001
        return np.nan


def snapshot(panel: Panel, feats: dict[str, pd.DataFrame], universe: pd.DataFrame,
             asof: pd.Timestamp | None = None, with_garch: bool = True) -> pd.DataFrame:
    """One row per universe symbol with every feature as of `asof` (default: last date)."""
    asof = asof or panel.px_eur.index.max()
    idx = panel.px_eur.index[panel.px_eur.index <= asof][-1]
    syms = [s for s in universe["symbol"] if s in panel.px_eur.columns]
    snap = pd.DataFrame({k: v.loc[idx, syms] for k, v in feats.items() if not v.empty})
    snap.index.name = "symbol"
    snap = snap.reset_index()
    meta = universe[["symbol", "region", "name", "sector", "country"]].drop_duplicates("symbol")
    snap = meta.merge(snap, on="symbol", how="inner")
    snap.insert(0, "asof", idx)
    peer_med = snap.groupby(["region", "sector"])["ret_6m"].transform("median")
    snap["rs_sector_6m"] = snap["ret_6m"] - peer_med
    if with_garch:
        log.info("Fitting GARCH(1,1) for %d symbols", len(syms))
        r = panel.rets.loc[:idx]
        snap["garch_vol"] = [garch_vol(r[s]) for s in snap["symbol"]]
    return snap

"""Track B: growth screening (hard gates + composite score)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PILLARS = {
    "price_strength": {"mom_12_1": 1, "rs_sector_6m": 1, "dist_52w_high": 1},
    "growth": {"rev_growth_yoy": 1, "eps_growth_yoy": 1, "rev_growth_accel": 1},
    "quality": {"gross_margin_trend": 1, "operating_margin": 1, "fcf_margin": 1},
    # risk: higher = riskier; max_dd_252 is negative, so flip its sign
    "risk_penalty": {"vol_60": 1, "max_dd_252": -1},
}
FACTOR_LABELS = {
    "mom_12_1": "12-1 momentum", "rs_sector_6m": "6m RS vs sector", "dist_52w_high": "near 52w high",
    "rev_growth_yoy": "revenue growth", "eps_growth_yoy": "EPS growth",
    "rev_growth_accel": "revenue acceleration", "gross_margin_trend": "gross margin trend",
    "operating_margin": "operating margin", "fcf_margin": "FCF margin",
    "vol_60": "volatility", "max_dd_252": "drawdown",
}
MIN_GROUP = 5


def apply_gates(snap: pd.DataFrame, gates: dict, qa: pd.DataFrame | None = None) -> pd.DataFrame:
    s = snap.copy()
    s["gate_liquidity"] = s["median_tv_20"] >= gates["min_median_traded_value_eur"]
    s["gate_trend"] = (s["px_sma200"] > 0) & (s["sma50_sma200"] > 0)
    s["gate_history"] = s["history_days"] >= gates["min_history_days"]
    s["gate_drawdown"] = s["dist_52w_high"] >= -gates["max_drawdown_from_high"]
    bad = set(qa.loc[qa["exclude_backtest"], "symbol"]) if qa is not None and not qa.empty else set()
    s["gate_data_quality"] = ~s["symbol"].isin(bad)
    s["passes_gates"] = s[["gate_liquidity", "gate_trend", "gate_history", "gate_drawdown",
                           "gate_data_quality"]].all(axis=1)
    return s


def _z(x: pd.Series) -> pd.Series:
    sd = x.std()
    if not sd or np.isnan(sd):
        return x * np.nan
    return ((x - x.mean()) / sd).clip(-3, 3)


def zscore_within(df: pd.DataFrame, col: str) -> pd.Series:
    """z-score within region x sector; fall back to region when the group is small."""
    by_sec = df.groupby(["region", "sector"])[col]
    z_sec = by_sec.transform(_z)
    n_sec = by_sec.transform("count")
    z_reg = df.groupby("region")[col].transform(_z)
    return z_sec.where(n_sec >= MIN_GROUP, z_reg)


def score(snap_gated: pd.DataFrame, weights: dict) -> pd.DataFrame:
    s = snap_gated[snap_gated["passes_gates"]].copy()
    if s.empty:
        return s
    for pillar, factors in PILLARS.items():
        zs = []
        for f, sign in factors.items():
            if f not in s:
                s[f] = np.nan
            s[f"z_{f}"] = sign * zscore_within(s, f)
            zs.append(f"z_{f}")
        s[f"pillar_{pillar}"] = s[zs].mean(axis=1, skipna=True)
    pill = s[[f"pillar_{p}" for p in PILLARS]].set_axis(list(PILLARS), axis=1)
    avail = pill.notna()
    # A missing pillar counts as neutral (z = 0). Re-weighting the remaining pillars
    # instead would let single-pillar names (no fundamentals) dominate the top ranks.
    sign = pd.Series({p: (-1.0 if p == "risk_penalty" else 1.0) for p in PILLARS})
    w = pd.Series({p: float(weights[p]) for p in PILLARS})
    s["score"] = (pill.fillna(0) * (w * sign)).sum(axis=1)
    s["growth_coverage"] = avail["growth"]
    s["rank"] = s.groupby("region")["score"].rank(ascending=False, method="first").astype(int)

    zcols = [c for c in s.columns if c.startswith("z_")]

    def best_worst(row):
        z = row[zcols].dropna().sort_values()
        name = lambda c: FACTOR_LABELS.get(c[2:], c[2:])  # noqa: E731
        return pd.Series({"strongest": ", ".join(name(c) for c in z.index[::-1][:3]),
                          "weakest": ", ".join(name(c) for c in z.index[:3])})

    s[["strongest", "weakest"]] = s.apply(best_worst, axis=1)
    return s.sort_values(["region", "rank"])


def run_track_b(snap: pd.DataFrame, fund_factors: pd.DataFrame, cfg: dict,
                previous: pd.DataFrame | None = None, qa: pd.DataFrame | None = None) -> pd.DataFrame:
    df = snap.merge(fund_factors, on="symbol", how="left") if not fund_factors.empty else snap
    gated = apply_gates(df, cfg["gates"], qa)
    log.info("Track B: %d of %d pass the gates", int(gated["passes_gates"].sum()), len(gated))
    ranked = score(gated, cfg["weights"])
    if ranked.empty:
        return ranked
    if previous is not None and not previous.empty:
        prev = previous.set_index("symbol")["rank"]
        ranked["prev_rank"] = ranked["symbol"].map(prev)
        ranked["rank_change"] = ranked["prev_rank"] - ranked["rank"]
        top_prev = set(previous.loc[previous["rank"] <= cfg["top_n_per_region"], "symbol"])
        ranked["new_entry"] = (ranked["rank"] <= cfg["top_n_per_region"]) & ~ranked["symbol"].isin(top_prev)
    else:
        ranked["prev_rank"] = np.nan
        ranked["rank_change"] = np.nan
        ranked["new_entry"] = ranked["rank"] <= cfg["top_n_per_region"]
    return ranked


def shortlist(ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cols = ["asof", "region", "rank", "symbol", "name", "sector", "country", "score",
            "pillar_price_strength", "pillar_growth", "pillar_quality", "pillar_risk_penalty",
            "rank_change", "new_entry", "strongest", "weakest", "growth_coverage"]
    top = ranked[ranked["rank"] <= top_n]
    return top[[c for c in cols if c in top]].reset_index(drop=True)


def exits(ranked: pd.DataFrame, previous: pd.DataFrame | None, top_n: int) -> pd.DataFrame:
    if previous is None or previous.empty:
        return pd.DataFrame(columns=["region", "symbol", "name", "prev_rank", "now"])
    prev_top = previous[previous["rank"] <= top_n]
    now = ranked.set_index("symbol")["rank"] if not ranked.empty else pd.Series(dtype=float)
    out = prev_top[~prev_top["symbol"].isin(now[now <= top_n].index)].copy()
    out["now"] = out["symbol"].map(lambda s: f"rank {int(now[s])}" if s in now else "failed gates")
    return out.rename(columns={"rank": "prev_rank"})[["region", "symbol", "name", "prev_rank", "now"]]

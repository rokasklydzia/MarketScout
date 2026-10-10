"""Track A: long-only strategy backtesting with walk-forward selection,
transaction costs and multiple-testing control (deflated Sharpe ratio).

Timing: a signal computed at day t's close is executed at the close of
t + exec_lag (default 1) and earns returns from the following day.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from statistics import NormalDist

from .features import Panel, TD_YEAR

log = logging.getLogger(__name__)
EULER_GAMMA = 0.5772156649015329
_N = NormalDist()  # standard normal; avoids a scipy dependency


# --------------------------------------------------------------------------
# Strategy definitions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategySpec:
    family: str
    params: dict
    rebalance: str  # "M" monthly or "W" weekly

    @property
    def label(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.family}[{p}]"


def _grid(family: str, rebalance: str, **grid) -> list[StrategySpec]:
    keys = list(grid)
    return [StrategySpec(family, dict(zip(keys, vals)), rebalance)
            for vals in itertools.product(*grid.values())]


def default_strategies() -> list[StrategySpec]:
    """Registered before testing; changing this list adds trials to the registry."""
    return (
        _grid("momentum_xs", "M", lookback=[252, 126], top_q=[0.10, 0.20])
        + _grid("trend_ts", "W", sma=[100, 200])
        + _grid("high_52w", "M", within=[0.05, 0.10])
        + _grid("reversal_st", "W", lookback=[5, 10])
        + _grid("low_vol", "M", window=[60, 252])
        + _grid("sector_rotation", "M", top_sectors=[2, 3])
    )


def _top(score: pd.DataFrame, elig: pd.DataFrame, q: float, highest: bool = True,
         min_names: int = 5) -> pd.DataFrame:
    """Equal weights on the top-q fraction of eligible names per row."""
    s = score.where(elig)
    ranks = s.rank(axis=1, ascending=not highest, method="first")
    n = elig.sum(axis=1)
    k = np.maximum(np.ceil(n * q), np.minimum(min_names, n))
    pick = ranks.le(k, axis=0) & s.notna()
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def signal_weights(spec: StrategySpec, panel: Panel, feats: dict[str, pd.DataFrame],
                   elig: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Target weights on every date (only rebalance-date rows are used)."""
    p, f = spec.params, feats
    loc = panel.px_local[elig.columns]
    if spec.family == "momentum_xs":
        score = f["mom_12_1"][elig.columns] if p["lookback"] == 252 else f["mom_6_1"][elig.columns]
        return _top(score, elig, p["top_q"])
    if spec.family == "trend_ts":
        above = (loc > loc.rolling(p["sma"], min_periods=int(p["sma"] * 0.8)).mean()) & elig
        return above.div(elig.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if spec.family == "high_52w":
        near = (f["dist_52w_high"][elig.columns] >= -p["within"]) & (f["sma50_slope_1m"][elig.columns] > 0)
        sel = near & elig
        return sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if spec.family == "reversal_st":
        score = panel.px_eur[elig.columns].pct_change(p["lookback"], fill_method=None)
        return _top(score, elig, 0.10, highest=False)
    if spec.family == "low_vol":
        return _top(f[f"vol_{p['window']}"][elig.columns], elig, 0.20, highest=False)
    if spec.family == "sector_rotation":
        rs = f["rs_6m"][elig.columns].where(elig)
        sec = sectors.reindex(elig.columns)
        med = rs.T.groupby(sec).median().T  # date x sector
        top_secs = med.rank(axis=1, ascending=False, method="first").le(p["top_sectors"])
        member = pd.DataFrame({s: top_secs[sec[s]] if sec[s] in top_secs else False
                               for s in elig.columns}, index=elig.index)
        sel = member & elig
        return sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    raise ValueError(f"Unknown strategy family {spec.family}")


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    s = index.to_series()
    key = s.dt.to_period("M" if freq == "M" else "W-FRI")
    return pd.DatetimeIndex(s.groupby(key).max().values)


@dataclass
class BTResult:
    returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    n_holdings: pd.Series


def simulate(target: pd.DataFrame, rets: pd.DataFrame, rebal: pd.DatetimeIndex,
             cost_bps: pd.DataFrame, exec_lag: int = 1, cost_mult: float = 1.0) -> BTResult:
    """Drifting-weights simulation. Uninvested weight is cash at 0%."""
    idx = rets.index
    R = rets.fillna(0.0).to_numpy()
    T = target.reindex(index=rebal).fillna(0.0).to_numpy()
    C = cost_bps.reindex(index=rebal).fillna(0.0).to_numpy() * cost_mult / 1e4
    exec_at = {}
    pos = idx.get_indexer(rebal)
    for k, i in enumerate(pos):
        j = i + exec_lag
        if 0 <= i and j < len(idx):
            exec_at[j] = k
    w = np.zeros(R.shape[1])
    out_r, out_to, out_c, out_n = (np.zeros(len(idx)) for _ in range(4))
    for i in range(len(idx)):
        pr = float(w @ R[i])
        if w.any():
            w = w * (1 + R[i]) / (1 + pr)
        if i in exec_at:
            k = exec_at[i]
            new = T[k]
            trade = np.abs(new - w)
            out_to[i] = trade.sum() / 2
            out_c[i] = float(trade @ C[k])
            w = new.copy()
        out_r[i] = pr - out_c[i]
        out_n[i] = int((w > 0).sum())
    mk = lambda a: pd.Series(a, index=idx)  # noqa: E731
    return BTResult(mk(out_r), mk(out_to), mk(out_c), mk(out_n))


def cost_matrix(elig: pd.DataFrame, median_tv: pd.DataFrame, region_bps: float,
                commission_bps: float, illiquid_mult: float) -> pd.DataFrame:
    tv = median_tv[elig.columns].where(elig)
    p25 = tv.quantile(0.25, axis=1)
    illiquid = tv.lt(p25, axis=0)
    base = pd.DataFrame(region_bps, index=elig.index, columns=elig.columns)
    return base.where(~illiquid, region_bps * illiquid_mult) + commission_bps


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def max_dd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def perf_metrics(r: pd.Series, bench: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    years = len(r) / TD_YEAR
    cagr = (1 + r).prod() ** (1 / years) - 1
    vol = r.std() * np.sqrt(TD_YEAR)
    dn = r[r < 0].std() * np.sqrt(TD_YEAR)
    mdd = max_dd(r)
    out = {"cagr": cagr, "vol": vol,
           "sharpe": r.mean() / r.std() * np.sqrt(TD_YEAR) if r.std() > 0 else np.nan,
           "sortino": r.mean() * TD_YEAR / dn if dn > 0 else np.nan,
           "max_dd": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
           "hit_rate_daily": float((r > 0).mean()), "n_days": len(r)}
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        out.update(alpha_beta(r, b))
        ex = r - b
        out["excess_ann"] = ex.mean() * TD_YEAR
        m = (1 + r).resample("ME").prod() - (1 + b).resample("ME").prod()
        out["hit_rate_monthly_vs_bench"] = float((m > 0).mean())
        bcagr = (1 + b).prod() ** (1 / years) - 1
        out["bench_cagr"] = bcagr
        out["bench_sharpe"] = b.mean() / b.std() * np.sqrt(TD_YEAR) if b.std() > 0 else np.nan
    return out


def alpha_beta(r: pd.Series, b: pd.Series, maxlags: int = 5) -> dict:
    """OLS r = a + beta*b with Newey-West (Bartlett) HAC standard errors.
    Pure numpy; matches statsmodels OLS(...).fit(cov_type='HAC', maxlags=5)."""
    try:
        y = r.to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(y)), b.to_numpy(dtype=float)])
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        u = (y - X @ beta)[:, None] * X          # score contributions
        S = u.T @ u
        for lag in range(1, maxlags + 1):
            w = 1 - lag / (maxlags + 1)
            G = u[lag:].T @ u[:-lag]
            S += w * (G + G.T)
        cov = XtX_inv @ S @ XtX_inv
        t_alpha = beta[0] / np.sqrt(cov[0, 0])
        return {"alpha_ann": beta[0] * TD_YEAR, "alpha_t_nw": t_alpha, "beta": beta[1]}
    except Exception:  # noqa: BLE001
        return {"alpha_ann": np.nan, "alpha_t_nw": np.nan, "beta": np.nan}


def _skew_kurt(x: np.ndarray) -> tuple[float, float]:
    """Population skewness and (non-excess) kurtosis, as scipy.stats defaults."""
    d = x - x.mean()
    m2 = (d ** 2).mean()
    return float((d ** 3).mean() / m2 ** 1.5), float((d ** 4).mean() / m2 ** 2)


def deflated_sharpe(r: pd.Series, n_trials: int, sr_var: float) -> tuple[float, float]:
    """Bailey & Lopez de Prado (2014). Sharpe ratios in per-period (daily) units.
    Returns (DSR probability, SR0 benchmark)."""
    r = r.dropna()
    T = len(r)
    if T < 30 or r.std() == 0:
        return np.nan, np.nan
    sr = r.mean() / r.std()
    g3, g4 = _skew_kurt(r.to_numpy(dtype=float))
    n = max(int(n_trials), 2)
    sr0 = np.sqrt(max(sr_var, 0)) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n)
                                     + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n * np.e)))
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(_N.cdf((sr - sr0) * np.sqrt(T - 1) / denom)), float(sr0)


def stationary_bootstrap_pvalue(x: pd.Series, n_boot: int = 1000, mean_block: int = 20,
                                seed: int = 7) -> float:
    """P(mean <= 0) under a stationary block bootstrap of x (Politis & Romano)."""
    a = x.dropna().to_numpy()
    n = len(a)
    if n < 50:
        return np.nan
    rng = np.random.default_rng(seed)
    p = 1 / mean_block
    centred = a - a.mean()  # bootstrap under the null of zero mean
    t = np.arange(n)
    means = np.empty(n_boot)
    for b in range(n_boot):
        jumps = rng.random(n) < p
        jumps[0] = True
        blk = np.cumsum(jumps) - 1                      # block id per position
        first = np.flatnonzero(jumps)                   # position where each block starts
        starts = rng.integers(n, size=len(first))       # random source start per block
        idx = (starts[blk] + (t - first[blk])) % n
        means[b] = centred[idx].mean()
    return float((means >= a.mean()).mean())


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------
def walk_forward_folds(index: pd.DatetimeIndex, start: pd.Timestamp, train_years: int,
                       test_months: int, embargo_days: int) -> list[dict]:
    folds = []
    test_start = start + pd.DateOffset(years=train_years)
    while True:
        test_end = test_start + pd.DateOffset(months=test_months)
        test_idx = index[(index >= test_start) & (index < test_end)]
        if len(test_idx) < 20:
            break
        train_end = index[index < test_start]
        train_end = train_end[-embargo_days - 1] if len(train_end) > embargo_days else None
        if train_end is None:
            break
        folds.append({"fold": len(folds) + 1, "train_start": start, "train_end": train_end,
                      "test_start": test_idx[0], "test_end": test_idx[-1]})
        test_start = test_end
    return folds


@dataclass
class TrackAOutput:
    summary: pd.DataFrame
    variants: pd.DataFrame
    folds: pd.DataFrame
    curves: pd.DataFrame
    registry_rows: pd.DataFrame = field(default_factory=pd.DataFrame)


def eligibility(panel: Panel, feats: dict[str, pd.DataFrame], symbols: list[str],
                min_tv: float, excluded: set[str]) -> pd.DataFrame:
    cols = [s for s in symbols if s in panel.px_eur.columns and s not in excluded]
    e = (panel.px_eur[cols].notna()
         & feats["median_tv_20"][cols].ge(min_tv)
         & feats["history_days"][cols].ge(TD_YEAR + 1))
    return e


def run_track_a(panel: Panel, feats: dict[str, pd.DataFrame], universe: pd.DataFrame,
                qa: pd.DataFrame, cfg: dict, min_tv: float,
                strategies: list[StrategySpec] | None = None,
                registry: pd.DataFrame | None = None, run_id: str = "") -> TrackAOutput:
    strategies = strategies or default_strategies()
    excluded = set(qa.loc[qa["exclude_backtest"], "symbol"]) if not qa.empty else set()
    rets_all = panel.rets
    sectors = universe.drop_duplicates("symbol").set_index("symbol")["sector"]
    variant_rows, fold_rows, curve_rows, summary_rows, reg_rows = [], [], [], [], []

    for region, bench_sym in panel.bench.items():
        syms = universe.loc[universe["region"] == region, "symbol"].tolist()
        elig = eligibility(panel, feats, syms, min_tv, excluded)
        if elig.shape[1] < 20 or bench_sym not in rets_all:
            log.warning("Region %s: too few eligible symbols or no benchmark; skipped", region)
            continue
        rets = rets_all[elig.columns]
        bench = rets_all[bench_sym]
        ew = rets.where(elig.shift(1, fill_value=False)).mean(axis=1).fillna(0)
        costs = cost_matrix(elig, feats["median_tv_20"], cfg["cost_bps"][region],
                            cfg["commission_bps"], cfg["illiquid_cost_multiplier"])
        first_valid = elig.any(axis=1).idxmax()
        folds = walk_forward_folds(rets.index, rets.index[0], cfg["initial_train_years"],
                                   cfg["test_block_months"], cfg["embargo_days"])
        if not folds:
            log.warning("Region %s: not enough history for walk-forward", region)
            continue

        # 1) run every registered variant once over the full sample
        runs: dict[str, dict] = {}
        for spec in strategies:
            w = signal_weights(spec, panel, feats, elig, sectors)
            rb = rebalance_dates(rets.index, spec.rebalance)
            base = simulate(w, rets, rb, costs, cfg.get("exec_lag", 1))
            stress = simulate(w, rets, rb, costs, cfg.get("exec_lag", 1),
                              cost_mult=cfg["stress_cost_multiplier"])
            live = base.returns[base.returns.index > first_valid]
            runs[spec.label] = {"spec": spec, "base": base, "stress": stress}
            m = perf_metrics(live, bench)
            sr_daily = live.mean() / live.std() if live.std() > 0 else np.nan
            variant_rows.append({"region": region, "family": spec.family, "variant": spec.label,
                                 **m, "turnover_ann": base.turnover.sum() / (len(live) / TD_YEAR),
                                 "sr_daily": sr_daily})
            reg_rows.append({"trial_id": f"{region}|{spec.label}", "region": region,
                             "family": spec.family, "variant": spec.label, "run_id": run_id,
                             "sr_daily": sr_daily, "n_obs": len(live)})

        # 2) walk-forward: pick best train-Sharpe variant per family, stitch OOS
        families = sorted({s.family for s in strategies})
        for fam in families:
            fam_runs = {k: v for k, v in runs.items() if v["spec"].family == fam}
            oos, oos_stress, chosen = [], [], []
            for fd in folds:
                def train_sr(v):
                    x = v["base"].returns.loc[max(first_valid, fd["train_start"]):fd["train_end"]]
                    return x.mean() / x.std() if len(x) > 20 and x.std() > 0 else -np.inf
                best = max(fam_runs, key=lambda k: train_sr(fam_runs[k]))
                sl = slice(fd["test_start"], fd["test_end"])
                oos.append(fam_runs[best]["base"].returns.loc[sl])
                oos_stress.append(fam_runs[best]["stress"].returns.loc[sl])
                chosen.append(best)
                fm = perf_metrics(fam_runs[best]["base"].returns.loc[sl], bench)
                fold_rows.append({"region": region, "family": fam, **fd, "chosen": best,
                                  "fold_return": (1 + fam_runs[best]["base"].returns.loc[sl]).prod() - 1,
                                  "bench_return": (1 + bench.loc[sl].fillna(0)).prod() - 1,
                                  "fold_sharpe": fm.get("sharpe", np.nan)})
            oos_r = pd.concat(oos)
            oos_s = pd.concat(oos_stress)
            m = perf_metrics(oos_r, bench)
            ms = perf_metrics(oos_s, bench)
            curve_rows.append(pd.DataFrame({"date": oos_r.index, "region": region, "family": fam,
                                            "strategy_ret": oos_r.values,
                                            "bench_ret": bench.reindex(oos_r.index).fillna(0).values,
                                            "ew_ret": ew.reindex(oos_r.index).values}))
            summary_rows.append({"region": region, "family": fam, **m,
                                 "excess_ann_2x_costs": ms.get("excess_ann", np.nan),
                                 "variants_chosen": ", ".join(sorted(set(chosen))),
                                 "n_folds": len(folds), "oos_returns": oos_r})

    summary = pd.DataFrame(summary_rows)
    variants = pd.DataFrame(variant_rows)
    reg_new = pd.DataFrame(reg_rows)
    if summary.empty:
        return TrackAOutput(summary, variants, pd.DataFrame(fold_rows), pd.DataFrame(), reg_new)

    # 3) multiple-testing control across ALL trials ever registered
    reg_all = pd.concat([registry, reg_new], ignore_index=True) if registry is not None else reg_new
    latest = reg_all.drop_duplicates("trial_id", keep="last")
    n_trials = latest["trial_id"].nunique()
    sr_var = float(latest["sr_daily"].var(ddof=1)) if n_trials > 1 else 0.0
    folds_df = pd.DataFrame(fold_rows)
    dsr, sr0, pboot, fold_win, nb_ok = [], [], [], [], []
    for row in summary.itertuples():
        d, s0 = deflated_sharpe(row.oos_returns, n_trials, sr_var)
        dsr.append(d)
        sr0.append(s0)
        b = panel.rets[panel.bench[row.region]].reindex(row.oos_returns.index).fillna(0)
        pboot.append(stationary_bootstrap_pvalue(row.oos_returns - b))
        f = folds_df[(folds_df.region == row.region) & (folds_df.family == row.family)]
        fold_win.append(float((f.fold_return > f.bench_return).mean()))
        v = variants[(variants.region == row.region) & (variants.family == row.family)]
        nb_ok.append(float((v.excess_ann > 0).mean()))
    summary["dsr"] = dsr
    summary["sr0_daily"] = sr0
    summary["n_trials"] = n_trials
    summary["bootstrap_p_excess"] = pboot
    summary["share_folds_beating_bench"] = fold_win
    summary["share_variants_beating_bench"] = nb_ok
    both = summary.groupby("family")["excess_ann"].transform(lambda x: (x > 0).all() and len(x) > 1)
    summary["beats_bench_all_regions"] = both.astype(bool)
    summary["passes"] = ((summary["excess_ann"] > 0) & (summary["dsr"] > cfg["dsr_threshold"])
                         & (summary["excess_ann_2x_costs"] > 0))
    summary = summary.drop(columns=["oos_returns"])
    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    log.info("Track A: %d family x region results, %d pass (N trials = %d)",
             len(summary), int(summary["passes"].sum()), n_trials)
    return TrackAOutput(summary, variants, folds_df, curves, reg_new)

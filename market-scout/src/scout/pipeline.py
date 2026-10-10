"""Orchestration of the daily and weekly runs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from . import backtest, download, features, fundamentals, qa, report, screen, universe
from .config import Settings
from .storage import Store

log = logging.getLogger(__name__)


@dataclass
class RunContext:
    settings: Settings
    source: download.PriceSource | None = None   # None -> yfinance
    universe_df: pd.DataFrame | None = None       # inject for tests / offline
    fundamentals_df: pd.DataFrame | None = None   # inject for tests / offline
    end: date | None = None


def _symbols(ctx: RunContext, uni: pd.DataFrame) -> pd.DataFrame:
    bench = universe.benchmark_rows(ctx.settings)
    return pd.concat([uni[["region", "symbol", "currency"]], bench], ignore_index=True).drop_duplicates("symbol")


def get_universe(ctx: RunContext, refresh: bool) -> pd.DataFrame:
    store = Store(ctx.settings)
    if ctx.universe_df is not None:
        uni = ctx.universe_df.copy()
        asof = ctx.end or date.today()
        uni["asof"] = pd.Timestamp(asof)
        store.write(uni, store.universe_path(asof))
        return uni
    if refresh:
        return universe.build_universe(ctx.settings)
    return store.latest_universe()


def run_daily(ctx: RunContext, full: bool = False) -> dict:
    """Prices + FX + QA. Safe to rerun (idempotent upserts)."""
    uni = get_universe(ctx, refresh=False) if not full else get_universe(ctx, refresh=True)
    syms = _symbols(ctx, uni)
    download.update_currencies(ctx.settings, syms["symbol"].tolist(), ctx.source)
    res = download.update_prices(ctx.settings, syms, ctx.source, full=full, end=ctx.end)
    download.update_fx(ctx.settings, ctx.source, end=ctx.end)
    clean, q = qa.run_qa(ctx.settings)
    return {"requested": res.requested, "failed": len(res.failed),
            "partial": res.failure_share > ctx.settings["download"]["partial_run_threshold"],
            "clean_rows": len(clean)}


def run_weekly(ctx: RunContext, refresh_universe: bool = True, skip_fundamentals: bool = False,
               with_garch: bool = True) -> dict:
    s = ctx.settings
    store = Store(s)
    uni = get_universe(ctx, refresh=refresh_universe)
    syms = _symbols(ctx, uni)
    download.update_currencies(s, syms["symbol"].tolist(), ctx.source)
    res = download.update_prices(s, syms, ctx.source, end=ctx.end)
    download.update_fx(s, ctx.source, end=ctx.end)
    clean, q = qa.run_qa(s)

    # --- features -----------------------------------------------------------
    panel = features.build_panel(clean, s["universe"]["benchmarks"])
    feats = features.compute_features(panel)
    asof = panel.px_eur.index.max()
    snap = features.snapshot(panel, feats, uni, asof, with_garch=with_garch)
    store.write(snap, store.p("features", f"asof={asof:%Y-%m-%d}.parquet"))

    # --- fundamentals -------------------------------------------------------
    if ctx.fundamentals_df is not None:
        fund = ctx.fundamentals_df
    elif skip_fundamentals:
        fp = s.data_dir / "raw" / "fundamentals" / "fundamentals.parquet"
        fund = pd.read_parquet(fp) if fp.exists() else pd.DataFrame()
    else:
        fund = fundamentals.update_fundamentals(s, uni)
    ff = fundamentals.fundamental_factors(fund, asof)

    # --- Track A ------------------------------------------------------------
    reg_path = store.p("results", "backtests", "trial_registry.parquet")
    registry = pd.read_parquet(reg_path) if reg_path.exists() else None
    run_id = f"{asof:%Y-%m-%d}"
    prev_sum = _latest(store, "results/backtests", "summary_", exclude=run_id)
    if registry is not None:  # a same-day rerun replaces that run's trials
        registry = registry[registry["run_id"] != run_id]
    ta = backtest.run_track_a(panel, feats, uni, q, s["backtest"],
                              s["features"]["min_median_traded_value_eur"],
                              registry=registry, run_id=run_id)
    if not ta.registry_rows.empty:
        reg_all = pd.concat([registry, ta.registry_rows], ignore_index=True) if registry is not None else ta.registry_rows
        store.write(reg_all, reg_path)
    for name, df in (("summary", ta.summary), ("variants", ta.variants), ("folds", ta.folds),
                     ("curves", ta.curves)):
        if not df.empty:
            store.write(df, store.p("results", "backtests", f"{name}_{run_id}.parquet"))

    # --- Track B ------------------------------------------------------------
    prev_screen = _latest(store, "results/screens", "screen_", exclude=run_id)
    ranked = screen.run_track_b(snap, ff, s["screen"], prev_screen, q)
    top_n = s["screen"]["top_n_per_region"]
    short = screen.shortlist(ranked, top_n)
    ex = screen.exits(ranked, prev_screen, top_n)
    if not ranked.empty:
        store.write(ranked.drop(columns=[c for c in ranked if c.startswith("z_")]),
                    store.p("results", "screens", f"screen_{run_id}.parquet"))
    short_path = store.p("results", "screens", "shortlist_latest.csv")
    short.to_csv(short_path, index=False)
    short.to_csv(store.p("results", "screens", f"shortlist_{run_id}.csv"), index=False)

    # --- health + outputs -------------------------------------------------------
    cov = {}
    for reg in ("us", "eu"):
        r = ranked[ranked["region"] == reg] if not ranked.empty else ranked
        cov[f"cov_{reg}"] = f"{r['growth_coverage'].mean():.0%}" if len(r) else "–"
    um_files = sorted((s.data_dir / "universe").glob(f"unmapped_*.csv"))
    health = {"universe": int(clean["symbol"].nunique()), "requested": res.requested,
              "failed": len(res.failed),
              "partial": res.failure_share > s["download"]["partial_run_threshold"],
              "excluded": int(q["exclude_backtest"].sum()) if not q.empty else 0,
              "suspended": int(q["flag_suspended"].sum()) if not q.empty else 0,
              "short": int(q["flag_short_history"].sum()) if not q.empty else 0,
              "flagged": int(q.filter(like="flag_").any(axis=1).sum()) if not q.empty else 0,
              "unmapped": len(pd.read_csv(um_files[-1])) if um_files else 0, **cov}
    dash = report.build_dashboard(s.reports_dir / "dashboard.html", asof, panel, feats, ranked, short,
                                  ta, q, health)
    iso = asof.isocalendar()
    weekly = report.build_weekly_report(s.reports_dir / f"weekly_{iso.year}-W{iso.week:02d}.md", asof,
                                        panel, feats, short, ex, ta, prev_sum, health)
    return {"asof": asof, "dashboard": dash, "weekly_report": weekly, "shortlist": short_path,
            "track_a": ta.summary, "shortlist_df": short, "health": health}


def _latest(store: Store, sub: str, prefix: str, exclude: str | None = None) -> pd.DataFrame | None:
    files = sorted((store.root / sub).glob(f"{prefix}*.parquet"))
    if exclude:
        files = [f for f in files if exclude not in f.name]
    return pd.read_parquet(files[-1]) if files else None

"""Dashboard (self-contained HTML, works offline) and weekly Markdown report."""
from __future__ import annotations

import html
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .features import Panel

log = logging.getLogger(__name__)

PALETTE = ["#2563eb", "#d97706", "#059669", "#db2777", "#7c3aed", "#0891b2", "#65a30d", "#dc2626"]
BENCH_COLOR = "#6b7280"
FAMILY_NAMES = {
    "momentum_xs": "Cross-sectional momentum", "trend_ts": "Time-series trend",
    "high_52w": "52-week-high proximity", "reversal_st": "Short-term reversal",
    "low_vol": "Low volatility", "sector_rotation": "Sector rotation",
}
REGION_NAMES = {"us": "US · Russell 1000", "eu": "Europe · STOXX 600"}


# --------------------------------------------------------------------------
# Regime
# --------------------------------------------------------------------------
def regime_table(panel: Panel, feats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for reg, b in panel.bench.items():
        cols = [c for c in panel.px_eur.columns if panel.region.get(c) == reg and c != b]
        if b not in panel.px_local or not cols:
            continue
        px = panel.px_local[b].dropna()
        sma = px.rolling(200).mean()
        above = feats["px_sma200"][cols].gt(0).where(feats["px_sma200"][cols].notna())
        breadth = above.mean(axis=1)
        disp = panel.rets[cols].std(axis=1).rolling(20).mean() * np.sqrt(252)
        rows.append({"region": reg, "bench": b, "last": px.iat[-1], "sma200": sma.iat[-1],
                     "bench_above_sma200": bool(px.iat[-1] > sma.iat[-1]),
                     "breadth": float(breadth.dropna().iat[-1]) if breadth.notna().any() else np.nan,
                     "breadth_4w_ago": float(breadth.dropna().iloc[-21]) if breadth.notna().sum() > 21 else np.nan,
                     "dispersion": float(disp.dropna().iat[-1]) if disp.notna().any() else np.nan,
                     "ret_1m": px.iat[-1] / px.iloc[-22] - 1 if len(px) > 22 else np.nan,
                     "s_px": px, "s_sma": sma, "s_breadth": breadth})
    return pd.DataFrame(rows)


def regime_sentences(reg: pd.DataFrame) -> list[str]:
    out = []
    for r in reg.itertuples():
        trend = "above" if r.bench_above_sma200 else "below"
        bchg = r.breadth - r.breadth_4w_ago if not np.isnan(r.breadth_4w_ago) else np.nan
        bdir = "" if np.isnan(bchg) else (" (up %.0f pts in 4 weeks)" % (100 * bchg) if bchg >= 0
                                          else " (down %.0f pts in 4 weeks)" % (-100 * bchg))
        out.append(f"{REGION_NAMES.get(r.region, r.region)}: benchmark {r.bench} is {trend} its "
                   f"200-day average, {r.ret_1m:+.1%} over the last month; "
                   f"{r.breadth:.0%} of stocks trade above their own 200-day average{bdir}.")
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=48, r=16, t=36, b=36), template="plotly_white",
                      legend=dict(orientation="h", y=-0.15), font=dict(size=12),
                      hovermode="x unified")
    return fig


def fig_regime(reg: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=len(reg), shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.65, 0.35],
                        subplot_titles=[REGION_NAMES.get(r, r) for r in reg["region"]])
    for i, r in enumerate(reg.itertuples(), 1):
        fig.add_trace(go.Scatter(x=r.s_px.index, y=r.s_px, name=f"{r.bench}", line=dict(color=PALETTE[0], width=1.5),
                                 showlegend=i == 1, legendgroup="px"), row=1, col=i)
        fig.add_trace(go.Scatter(x=r.s_sma.index, y=r.s_sma, name="200-day average",
                                 line=dict(color=BENCH_COLOR, dash="dot", width=1.2),
                                 showlegend=i == 1, legendgroup="sma"), row=1, col=i)
        fig.add_trace(go.Scatter(x=r.s_breadth.index, y=r.s_breadth, name="Breadth: share above own SMA200",
                                 line=dict(color=PALETTE[2], width=1.2), showlegend=i == 1,
                                 legendgroup="br"), row=2, col=i)
        fig.update_yaxes(tickformat=".0%", range=[0, 1], row=2, col=i)
    return _layout(fig, 480)


def fig_strategy_curves(curves: pd.DataFrame, region: str) -> go.Figure:
    c = curves[curves["region"] == region]
    fig = go.Figure()
    for i, (fam, g) in enumerate(c.groupby("family")):
        g = g.sort_values("date")
        fig.add_trace(go.Scatter(x=g["date"], y=(1 + g["strategy_ret"]).cumprod(),
                                 name=FAMILY_NAMES.get(fam, fam), line=dict(color=PALETTE[i % 8], width=1.5)))
    g = c.drop_duplicates("date").sort_values("date")
    if not g.empty:
        fig.add_trace(go.Scatter(x=g["date"], y=(1 + g["bench_ret"]).cumprod(), name="Benchmark",
                                 line=dict(color=BENCH_COLOR, width=2.2)))
        fig.add_trace(go.Scatter(x=g["date"], y=(1 + g["ew_ret"].fillna(0)).cumprod(),
                                 name="Equal-weight universe", line=dict(color=BENCH_COLOR, width=1.4, dash="dash")))
    fig.update_yaxes(title="Growth of 1 EUR (out of sample, after costs)")
    fig.update_layout(title=f"{REGION_NAMES.get(region, region)} — stitched out-of-sample results")
    return _layout(fig, 420)


def fig_shortlist_prices(panel: Panel, short: pd.DataFrame, region: str) -> go.Figure | None:
    names = [s for s in short.loc[short["region"] == region, "symbol"] if s in panel.px_local]
    if not names:
        return None
    fig = go.Figure()
    buttons = []
    for k, s in enumerate(names):
        px = panel.px_local[s].dropna().iloc[-504:]
        vis = k == 0
        fig.add_trace(go.Scatter(x=px.index, y=px, name=s, line=dict(color=PALETTE[0], width=1.5), visible=vis))
        fig.add_trace(go.Scatter(x=px.index, y=px.rolling(50).mean(), name="SMA50",
                                 line=dict(color=PALETTE[1], width=1), visible=vis))
        fig.add_trace(go.Scatter(x=px.index, y=px.rolling(200).mean(), name="SMA200",
                                 line=dict(color=BENCH_COLOR, width=1, dash="dot"), visible=vis))
        mask = [False] * (3 * len(names))
        mask[3 * k:3 * k + 3] = [True] * 3
        buttons.append(dict(label=s, method="update", args=[{"visible": mask}, {"title": s}]))
    fig.update_layout(title=names[0], updatemenus=[dict(buttons=buttons, x=1, xanchor="right", y=1.18,
                                                        showactive=True)])
    fig.update_yaxes(title="Adjusted price (local currency)")
    return _layout(fig, 400)


# --------------------------------------------------------------------------
# HTML helpers
# --------------------------------------------------------------------------
def _fmt(v, kind: str = "") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "pctp":
        return f"{v:+.1%}"
    if kind == "f2":
        return f"{v:.2f}"
    if kind == "int":
        return f"{int(v):+d}" if v else "0"
    return html.escape(str(v))


def table_html(df: pd.DataFrame, cols: list[tuple[str, str, str]], cls: str = "") -> str:
    head = "".join(f"<th>{html.escape(lbl)}</th>" for _, lbl, _ in cols)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c, _, kind in cols:
            v = r.get(c)
            css = ""
            if kind in ("pctp", "int") and isinstance(v, (int, float)) and not np.isnan(v):
                css = ' class="pos"' if v > 0 else (' class="neg"' if v < 0 else "")
            if kind == "bool":
                tds.append(f"<td>{'✓' if bool(v) else ''}</td>")
            elif kind == "pass":
                tds.append(f'<td><span class="tag {"ok" if v else "no"}">{"passes" if v else "rejected"}</span></td>')
            else:
                tds.append(f"<td{css}>{_fmt(v, kind)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table class="sortable {cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


CSS = """
:root{--bg:#f8f8f6;--card:#fff;--ink:#1f2328;--muted:#6b7280;--line:#e5e7eb;--pos:#047857;--neg:#b91c1c;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--card:#1f2228;--ink:#e6e6e6;--muted:#9aa0a6;--line:#30343b;--pos:#34d399;--neg:#f87171;--accent:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px}
h3{font-size:15px;margin:20px 0 8px}.muted{color:var(--muted)}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}.kpi b{display:block;font-size:24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}th{cursor:pointer;color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--card)}
td.pos{color:var(--pos)}td.neg{color:var(--neg)}.tag{padding:2px 8px;border-radius:99px;font-size:12px}.tag.ok{background:#d1fae5;color:#065f46}.tag.no{background:#f3f4f6;color:#6b7280}
nav a{color:var(--accent);margin-right:14px;text-decoration:none}ul{margin:6px 0 0 18px;padding:0}.warn{color:var(--neg)}
"""

SORT_JS = """
document.querySelectorAll('table.sortable th').forEach((th,i)=>{th.addEventListener('click',()=>{
const tb=th.closest('table').tBodies[0];const rows=[...tb.rows];const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
const val=r=>{const t=r.cells[i].innerText.replace(/[%,+✓]/g,'');const n=parseFloat(t);return isNaN(n)?t:n};
rows.sort((a,b)=>{const x=val(a),y=val(b);return (x>y?1:x<y?-1:0)*(asc?1:-1)});rows.forEach(r=>tb.appendChild(r))})});
"""


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_dashboard(path: Path, asof: pd.Timestamp, panel: Panel, feats: dict, ranked: pd.DataFrame,
                    short: pd.DataFrame, ta, qa: pd.DataFrame, health: dict) -> Path:
    reg = regime_table(panel, feats)
    figs_html = []
    first = [True]

    def emb(fig: go.Figure | None) -> str:
        if fig is None:
            return '<p class="muted">No data.</p>'
        out = fig.to_html(full_html=False, include_plotlyjs=first[0], config={"displaylogo": False})
        first[0] = False
        return f'<div class="card">{out}</div>'

    summary = ta.summary if ta is not None else pd.DataFrame()
    n_pass = int(summary["passes"].sum()) if not summary.empty else 0
    kpis = [
        ("Universe", f"{health.get('universe', 0):,}", "symbols with clean prices"),
        ("Passing growth gates", f"{int(ranked['passes_gates'].sum()) if 'passes_gates' in ranked else len(ranked):,}",
         "liquidity, trend, history, drawdown"),
        ("Strategy candidates", f"{n_pass} / {len(summary)}", "pass OOS, DSR and 2× cost tests"),
        ("Data issues", f"{health.get('failed', 0)} failed · {health.get('flagged', 0)} flagged",
         "downloads · QA flags"),
    ]
    k_html = "".join(f'<div class="kpi"><span class="muted">{a}</span><b>{b}</b><span class="muted">{c}</span></div>'
                     for a, b, c in kpis)

    parts = [f"<h1>Market scout</h1><p class='muted'>Data as of {asof:%d %b %Y} · returns in EUR · long-only · "
             "research input, not investment advice</p>",
             "<nav><a href='#regime'>Regime</a><a href='#growth'>Growth screen</a>"
             "<a href='#lab'>Strategy lab</a><a href='#health'>Data health</a></nav>",
             f'<div class="kpis">{k_html}</div>']

    parts.append("<h2 id='regime'>Market regime</h2><ul>" +
                 "".join(f"<li>{html.escape(s)}</li>" for s in regime_sentences(reg)) + "</ul>")
    if not reg.empty:
        parts.append(emb(fig_regime(reg)))

    parts.append("<h2 id='growth'>Growth screen</h2><p class='muted'>Top names per region by composite score "
                 "(40% price strength, 35% growth, 15% quality, −10% risk). Click a column to sort.</p>")
    cols = [("rank", "#", ""), ("symbol", "Symbol", ""), ("name", "Name", ""), ("sector", "Sector", ""),
            ("score", "Score", "f2"), ("pillar_price_strength", "Price", "f2"), ("pillar_growth", "Growth", "f2"),
            ("pillar_quality", "Quality", "f2"), ("pillar_risk_penalty", "Risk", "f2"),
            ("rank_change", "Δ rank", "int"), ("new_entry", "New", "bool"), ("strongest", "Strongest factors", "")]
    for region in [r for r in ("us", "eu") if not short.empty and r in set(short["region"])]:
        parts.append(f"<h3>{REGION_NAMES.get(region, region)}</h3>")
        parts.append(f'<div class="card">{table_html(short[short.region == region], cols)}</div>')
        parts.append(emb(fig_shortlist_prices(panel, short, region)))

    parts.append("<h2 id='lab'>Strategy lab</h2><p class='muted'>Parameters are picked on training data only; "
                 "curves stitch the 6-month test blocks. A strategy passes if it beats the benchmark out of sample, "
                 "its deflated Sharpe ratio is above 0.95, and the edge survives doubled costs. "
                 "Universe is today's index members, so absolute results are survivorship-biased.</p>")
    if not summary.empty:
        s = summary.copy()
        s["strategy"] = s["family"].map(FAMILY_NAMES)
        scols = [("region", "Region", ""), ("strategy", "Strategy", ""), ("passes", "Verdict", "pass"),
                 ("cagr", "CAGR", "pct"), ("bench_cagr", "Bench CAGR", "pct"), ("excess_ann", "Excess/yr", "pctp"),
                 ("excess_ann_2x_costs", "Excess @2× cost", "pctp"), ("sharpe", "Sharpe", "f2"),
                 ("dsr", "DSR", "f2"), ("max_dd", "Max DD", "pct"), ("alpha_t_nw", "α t-stat", "f2"),
                 ("share_folds_beating_bench", "Folds won", "pct"), ("bootstrap_p_excess", "Boot p", "f2")]
        parts.append(f'<div class="card">{table_html(s.sort_values(["region", "dsr"], ascending=[True, False]), scols)}</div>')
        parts.append(f"<p class='muted'>Deflated against N = {int(s['n_trials'].iat[0])} registered trials.</p>")
        for region in s["region"].unique():
            parts.append(emb(fig_strategy_curves(ta.curves, region)))

    parts.append("<h2 id='health'>Data health</h2>")
    hrows = [("Symbols requested", health.get("requested", "–")), ("Download failures", health.get("failed", "–")),
             ("Excluded from backtests (missing data or split errors)", health.get("excluded", "–")),
             ("Flagged as suspended", health.get("suspended", "–")),
             ("Short history (< 1 year)", health.get("short", "–")),
             ("Growth-factor coverage, US", health.get("cov_us", "–")),
             ("Growth-factor coverage, Europe", health.get("cov_eu", "–"))]
    parts.append('<div class="card"><table>' + "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in hrows)
                 + "</table></div>")
    if not qa.empty:
        flagged = qa[qa.filter(like="flag_").any(axis=1)].head(50)
        if not flagged.empty:
            fcols = [("symbol", "Symbol", ""), ("region", "Region", ""), ("ccy", "Ccy", ""),
                     ("missing_share", "Missing", "pct"), ("max_gap_days", "Max gap", ""),
                     ("max_stale_run", "Stale run", ""), ("split_errors", "Split errors", "")]
            parts.append("<h3>Flagged symbols (first 50)</h3>")
            parts.append(f'<div class="card">{table_html(flagged, fcols)}</div>')

    doc = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Market scout {asof:%Y-%m-%d}</title><style>{CSS}</style></head><body><main>"
           + "".join(parts) + f"</main><script>{SORT_JS}</script></body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    log.info("Dashboard written to %s", path)
    return path


def build_weekly_report(path: Path, asof: pd.Timestamp, panel: Panel, feats: dict, short: pd.DataFrame,
                        exits_df: pd.DataFrame, ta, prev_summary: pd.DataFrame | None, health: dict) -> Path:
    reg = regime_table(panel, feats)
    iso = asof.isocalendar()
    L = [f"# Market scout — week {iso.year}-W{iso.week:02d}", "",
         f"Data as of {asof:%d %b %Y}. Returns in EUR, long-only. Research input, not investment advice.", "",
         "## 1. Regime", ""]
    L += [f"- {s}" for s in regime_sentences(reg)] + [""]

    L += ["## 2. Growth shortlist changes", ""]
    if short.empty:
        L.append("No names passed the gates this week.")
    for region in [r for r in ("us", "eu") if not short.empty and r in set(short["region"])]:
        g = short[short.region == region]
        new = g[g["new_entry"]]
        L.append(f"**{REGION_NAMES.get(region, region)}** — top {len(g)}; leaders: "
                 + ", ".join(f"{r.symbol} ({r.score:.2f})" for r in g.head(5).itertuples()))
        L.append("")
        L.append("- New entries: " + (", ".join(f"{r.symbol} (#{r.rank})" for r in new.itertuples()) or "none"))
        ex = exits_df[exits_df.region == region] if not exits_df.empty else exits_df
        L.append("- Exits: " + (", ".join(f"{r.symbol} ({r.now})" for r in ex.itertuples()) or "none"))
        L.append("")

    L += ["## 3. Strategy candidates", ""]
    s = ta.summary if ta is not None else pd.DataFrame()
    if s.empty:
        L.append("No backtest results this week.")
    else:
        passing = s[s["passes"]]
        if passing.empty:
            L.append("No strategy passes all acceptance rules (OOS excess > 0, DSR > 0.95, edge at 2× costs).")
        for r in passing.itertuples():
            L.append(f"- PASS {REGION_NAMES.get(r.region, r.region)} · {FAMILY_NAMES.get(r.family, r.family)}: "
                     f"excess {r.excess_ann:+.1%}/yr, DSR {r.dsr:.2f}, Sharpe {r.sharpe:.2f}")
        if prev_summary is not None and not prev_summary.empty:
            m = s.merge(prev_summary[["region", "family", "passes"]], on=["region", "family"],
                        how="left", suffixes=("", "_prev"))
            changed = m[m["passes"] != m["passes_prev"].fillna(False).astype(bool)]
            for r in changed.itertuples():
                L.append(f"- Changed: {r.region} {FAMILY_NAMES.get(r.family, r.family)} now "
                         f"{'passes' if r.passes else 'fails'}")
        best = s.sort_values("dsr", ascending=False).head(3)
        L.append("")
        L.append("Closest to passing: " + "; ".join(
            f"{r.region} {FAMILY_NAMES.get(r.family, r.family)} (DSR {r.dsr:.2f}, excess {r.excess_ann:+.1%})"
            for r in best.itertuples()))
    L += ["", "## 4. Data issues", ""]
    issues = []
    if health.get("partial"):
        issues.append(f"Price run was PARTIAL: {health.get('failed')} of {health.get('requested')} symbols failed.")
    if health.get("failed"):
        issues.append(f"{health['failed']} symbols failed to download (see data/raw/logs).")
    if health.get("unmapped"):
        issues.append(f"{health['unmapped']} holdings could not be mapped to Yahoo symbols (see data/universe).")
    if health.get("flagged"):
        issues.append(f"{health['flagged']} symbols carry QA flags; {health.get('excluded', 0)} excluded from backtests.")
    L += [f"- {i}" for i in issues] or ["- None."]
    L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    log.info("Weekly report written to %s", path)
    return path

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scout import backtest, fundamentals, qa, screen, universe  # noqa: E402
from scout.config import load_settings  # noqa: E402
from scout.pipeline import RunContext, run_weekly  # noqa: E402
from scout.synthetic import SyntheticSource, ishares_csv_text, synthetic_universe  # noqa: E402


# ---------------- universe -------------------------------------------------
def test_parse_and_map_ishares_csv():
    uni = synthetic_universe(n_us=10, n_eu=12)
    df = universe.parse_ishares_csv(ishares_csv_text(uni, "eu"))
    mapped, unmapped = universe.build_region(df, "eu", {})
    assert len(mapped) == 12 and unmapped.empty            # cash line dropped, all mapped
    assert set(mapped["symbol"]) == set(uni.loc[uni.region == "eu", "symbol"])


@pytest.mark.parametrize("ticker,exchange,country,expected", [
    ("BA.", "London Stock Exchange", "United Kingdom", "BA.L"),
    ("BT.A", "London Stock Exchange", "United Kingdom", "BT-A.L"),
    ("NOVO B", "Nasdaq Omx Nordic", "Denmark", "NOVO-B.CO"),
    ("ERIC B", "Nasdaq Omx Nordic", "Sweden", "ERIC-B.ST"),
    ("NESN", "SIX Swiss Exchange", "Switzerland", "NESN.SW"),
    ("SAP", "Xetra", "Germany", "SAP.DE"),
])
def test_eu_symbol_mapping(ticker, exchange, country, expected):
    assert universe.map_symbol("eu", ticker, exchange, country) == expected


def test_us_override():
    df = pd.DataFrame({"ticker": ["BRKB", "AAPL"], "asset_class": ["Equity"] * 2,
                       "exchange": ["New York Stock Exchange Inc.", "NASDAQ"], "location": ["United States"] * 2})
    mapped, _ = universe.build_region(df, "us", {("us", "BRKB"): "BRK-B"})
    assert list(mapped["symbol"]) == ["BRK-B", "AAPL"]


# ---------------- QA -------------------------------------------------------
def test_currency_minor_units():
    assert qa.resolve_currency("SHEL.L", "GBp", "GBP") == ("GBP", 100.0)
    assert qa.resolve_currency("SHEL.L", None, "GBP") == ("GBP", 100.0)
    assert qa.resolve_currency("SAP.DE", "EUR", "EUR") == ("EUR", 1.0)


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    s = load_settings(overrides={"data_dir": str(out / "data"), "reports_dir": str(out / "reports"),
                                 "download": {"pause_seconds": 0}})
    end = date(2026, 10, 9)
    uni = synthetic_universe()
    src = SyntheticSource(uni, end - timedelta(days=int(365.25 * 5)), end)
    res = run_weekly(RunContext(s, source=src, universe_df=uni, fundamentals_df=src.fundamentals(), end=end),
                     refresh_universe=False, with_garch=False)
    q = pd.read_parquet(out / "data" / "clean" / "qa_report.parquet").set_index("symbol")
    clean = pd.read_parquet(out / "data" / "clean" / "prices_daily.parquet")
    return {"res": res, "qa": q, "clean": clean, "out": out}


def test_qa_catches_injected_defects(demo):
    q = demo["qa"]
    assert q.loc["US002", "flag_split_error"] and q.loc["US002", "exclude_backtest"]
    assert not q.loc["US001", "flag_split_error"]           # genuine split is not an error
    assert q.loc["US003", "flag_suspended"]
    assert q.loc["EU000.L", "flag_short_history"]
    assert "US119" not in q.index                          # failed download
    assert demo["res"]["health"]["failed"] == 1


def test_pence_converted(demo):
    c = demo["clean"]
    gbx = c[c.symbol.str.endswith(".L")]
    other = c[c.symbol.str.endswith(".DE")]
    assert gbx["close_eur"].median() < 5 * other["close_eur"].median()  # not 100x too big


def test_outputs_exist(demo):
    r = demo["res"]
    assert Path(r["dashboard"]).stat().st_size > 100_000
    assert Path(r["weekly_report"]).read_text().startswith("# Market scout")
    short = r["shortlist_df"]
    assert (short.groupby("region").size() == 25).all()
    assert "US002" not in set(short["symbol"])              # data-quality gate


# ---------------- backtest engine -----------------------------------------
def test_no_lookahead_and_costs():
    idx = pd.bdate_range("2024-01-01", periods=6)
    rets = pd.DataFrame({"A": [0, 0, 0.10, 0.10, 0, 0]}, index=idx, dtype=float)
    target = pd.DataFrame({"A": [1.0] * 6}, index=idx)
    costs = pd.DataFrame({"A": [10.0] * 6}, index=idx)
    res = backtest.simulate(target, rets, pd.DatetimeIndex([idx[1]]), costs, exec_lag=1)
    # signal on day 1 -> executed at day 2 close -> first return earned is day 3
    assert res.returns.iloc[2] == pytest.approx(-0.001)     # only the 10 bps cost on day 2
    assert res.returns.iloc[3] == pytest.approx(0.10)
    assert res.turnover.iloc[2] == pytest.approx(0.5)


def test_deflated_sharpe_penalises_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 750))
    d_few, _ = backtest.deflated_sharpe(r, n_trials=2, sr_var=0.0004)
    d_many, _ = backtest.deflated_sharpe(r, n_trials=500, sr_var=0.0004)
    assert d_few > d_many


def test_walk_forward_folds_have_embargo():
    idx = pd.bdate_range("2021-10-11", "2026-10-09")
    folds = backtest.walk_forward_folds(idx, idx[0], 2, 6, 21)
    assert len(folds) == 6
    for f in folds:
        gap = ((idx > f["train_end"]) & (idx < f["test_start"])).sum()
        assert gap == 21


# ---------------- fundamentals --------------------------------------------
def test_sec_q4_derived_from_annual():
    def fact(frame, end, val, filed):
        return {"frame": frame, "end": end, "val": val, "filed": filed}
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        fact("CY2025Q1", "2025-03-31", 100, "2025-05-01"), fact("CY2025Q2", "2025-06-30", 110, "2025-08-01"),
        fact("CY2025Q3", "2025-09-30", 120, "2025-11-01"), fact("CY2025", "2025-12-31", 460, "2026-02-15"),
    ]}}}}}
    q = fundamentals.sec_quarterly(facts, "X")
    q4 = q[(q.metric == "revenue") & (q.period_end == "2025-12-31")]
    assert q4["value"].iat[0] == pytest.approx(130)
    assert q4["available"].iat[0] == pd.Timestamp("2026-02-16")  # point-in-time


def test_factors_ignore_unreleased_quarters():
    pe = pd.date_range("2024-03-31", periods=9, freq="QE")
    f = pd.DataFrame({"symbol": "X", "metric": "revenue", "period_end": pe,
                      "available": pe + pd.Timedelta(days=40), "value": np.linspace(100, 180, 9)})
    asof = pe[-1] + pd.Timedelta(days=10)  # last quarter not yet published
    out = fundamentals.fundamental_factors(f, asof)
    assert out["fund_quarters"].iat[0] == 8


# ---------------- screen ----------------------------------------------------
def test_missing_growth_is_neutral_not_inflated():
    n = 12
    snap = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)], "region": "us", "sector": "Tech",
        "median_tv_20": 1e7, "px_sma200": 0.1, "sma50_sma200": 0.05, "history_days": 600,
        "dist_52w_high": -0.05, "mom_12_1": np.linspace(0, 1, n), "rs_sector_6m": np.linspace(0, 1, n),
        "vol_60": 0.3, "max_dd_252": -0.2,
        "rev_growth_yoy": [np.nan] + list(np.linspace(0, 1, n - 1)),
    })
    ranked = screen.run_track_b(snap, pd.DataFrame(), {"gates": {"min_median_traded_value_eur": 5e6,
                                "max_drawdown_from_high": 0.35, "min_history_days": 252},
                                "weights": {"price_strength": .4, "growth": .35, "quality": .15,
                                            "risk_penalty": .1}, "top_n_per_region": 5})
    assert ranked.loc[ranked.symbol == "S0", "growth_coverage"].iat[0] == False  # noqa: E712
    assert ranked["score"].idxmax() == ranked.index[ranked.symbol == f"S{n-1}"][0]

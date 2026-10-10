"""Quarterly fundamentals and growth/quality factors.

US : SEC EDGAR company-facts API (free; carries filing dates, so point-in-time).
EU : yfinance quarterly statements (shorter history, no filing date -> we assume
     availability 60 days after period end).

Everything is normalised to one long table:
    symbol, metric, period_end, available, value
with metrics: revenue, eps, gross_profit, operating_income, fcf
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import requests

from .config import Settings
from .storage import Store

log = logging.getLogger(__name__)

METRICS = ["revenue", "eps", "gross_profit", "operating_income", "fcf"]

SEC_CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}
EU_LAG_DAYS = 60


# --------------------------------------------------------------------------
# SEC EDGAR
# --------------------------------------------------------------------------
class SecClient:
    def __init__(self, user_agent: str, pause: float = 0.15):
        self.h = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.pause = pause

    def cik_map(self) -> dict[str, int]:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=self.h, timeout=30)
        r.raise_for_status()
        return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in r.json().values()}

    def company_facts(self, cik: int) -> dict:
        time.sleep(self.pause)
        r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                         headers=self.h, timeout=30)
        r.raise_for_status()
        return r.json()


def _sec_concept_frame(facts: dict, concepts: list[str]) -> pd.DataFrame:
    """Quarterly values for the first concept that has data, Q4 derived from FY."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for c in concepts:
        if c not in gaap:
            continue
        units = gaap[c]["units"]
        unit = "USD" if "USD" in units else ("USD/shares" if "USD/shares" in units else next(iter(units)))
        df = pd.DataFrame(units[unit])
        if df.empty or "frame" not in df:
            continue
        df = df.dropna(subset=["frame"])
        q = df[df["frame"].str.fullmatch(r"CY\d{4}Q\d")].copy()
        fy = df[df["frame"].str.fullmatch(r"CY\d{4}")].copy()
        if q.empty:
            continue
        q["year"] = q["frame"].str.slice(2, 6).astype(int)
        q["qtr"] = q["frame"].str.slice(7, 8).astype(int)
        # Derive missing Q4 = FY - (Q1+Q2+Q3) for flow concepts
        rows = []
        if not fy.empty:
            fy["year"] = fy["frame"].str.slice(2, 6).astype(int)
            for _, f in fy.iterrows():
                have = q[q["year"] == f["year"]]
                if 4 in set(have["qtr"]) or set(have["qtr"]) != {1, 2, 3}:
                    continue
                rows.append({"year": f["year"], "qtr": 4, "end": f["end"], "filed": f["filed"],
                             "val": f["val"] - have["val"].sum()})
        q = pd.concat([q[["year", "qtr", "end", "filed", "val"]], pd.DataFrame(rows)],
                      ignore_index=True)
        q = q.sort_values("filed").drop_duplicates(["year", "qtr"], keep="first")
        return pd.DataFrame({"period_end": pd.to_datetime(q["end"]),
                             "available": pd.to_datetime(q["filed"]) + pd.Timedelta(days=1),
                             "value": q["val"].astype(float)})
    return pd.DataFrame(columns=["period_end", "available", "value"])


def sec_quarterly(facts: dict, symbol: str) -> pd.DataFrame:
    out = []
    for m in ["revenue", "eps", "gross_profit", "operating_income"]:
        f = _sec_concept_frame(facts, SEC_CONCEPTS[m])
        if not f.empty:
            out.append(f.assign(metric=m))
    ocf = _sec_concept_frame(facts, SEC_CONCEPTS["ocf"])
    capex = _sec_concept_frame(facts, SEC_CONCEPTS["capex"])
    if not ocf.empty:
        fcf = ocf.merge(capex[["period_end", "value"]], on="period_end", how="left",
                        suffixes=("", "_capex"))
        fcf["value"] = fcf["value"] - fcf["value_capex"].fillna(0)
        out.append(fcf.drop(columns="value_capex").assign(metric="fcf"))
    if not out:
        return pd.DataFrame(columns=["symbol", "metric", "period_end", "available", "value"])
    return pd.concat(out, ignore_index=True).assign(symbol=symbol)


# --------------------------------------------------------------------------
# yfinance (Europe)
# --------------------------------------------------------------------------
YF_ROWS = {"revenue": ["Total Revenue", "Operating Revenue"], "eps": ["Diluted EPS", "Basic EPS"],
           "gross_profit": ["Gross Profit"], "operating_income": ["Operating Income"]}


def yf_quarterly(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    t = yf.Ticker(symbol)
    out = []
    inc = t.quarterly_income_stmt
    cf = t.quarterly_cashflow
    for m, names in YF_ROWS.items():
        row = next((n for n in names if inc is not None and n in inc.index), None)
        if row:
            s = inc.loc[row].dropna()
            out.append(pd.DataFrame({"metric": m, "period_end": pd.to_datetime(s.index),
                                     "value": s.values.astype(float)}))
    if cf is not None and "Free Cash Flow" in cf.index:
        s = cf.loc["Free Cash Flow"].dropna()
        out.append(pd.DataFrame({"metric": "fcf", "period_end": pd.to_datetime(s.index),
                                 "value": s.values.astype(float)}))
    if not out:
        return pd.DataFrame(columns=["symbol", "metric", "period_end", "available", "value"])
    df = pd.concat(out, ignore_index=True)
    df["available"] = df["period_end"] + pd.Timedelta(days=EU_LAG_DAYS)
    return df.assign(symbol=symbol)


# --------------------------------------------------------------------------
# Refresh
# --------------------------------------------------------------------------
def update_fundamentals(settings: Settings, universe: pd.DataFrame) -> pd.DataFrame:
    store = Store(settings)
    path = store.p("raw", "fundamentals", "fundamentals.parquet")
    meta_path = store.p("raw", "fundamentals", "fetched.parquet")
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    fetched = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame(
        columns=["symbol", "fetched"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=settings["fundamentals"]["refresh_days"])
    fresh = set(fetched.loc[pd.to_datetime(fetched["fetched"]) > cutoff, "symbol"])
    todo = universe[~universe["symbol"].isin(fresh)]
    log.info("Fundamentals: %d symbols to refresh", len(todo))

    sec = SecClient(settings["fundamentals"]["sec_user_agent"])
    try:
        ciks = sec.cik_map()
    except Exception as exc:  # noqa: BLE001
        log.warning("SEC ticker map unavailable (%s); skipping US fundamentals", exc)
        ciks = {}
    new, done = [], []
    for i, r in enumerate(todo.itertuples(), 1):
        try:
            if r.region == "us":
                cik = ciks.get(r.symbol.upper())
                if cik is None:
                    continue
                new.append(sec_quarterly(sec.company_facts(cik), r.symbol))
            else:
                new.append(yf_quarterly(r.symbol))
                time.sleep(0.3)
            done.append(r.symbol)
        except Exception as exc:  # noqa: BLE001
            log.debug("Fundamentals failed for %s: %s", r.symbol, exc)
        if i % 100 == 0:
            log.info("  %d/%d", i, len(todo))
    if new:
        new_df = pd.concat(new, ignore_index=True)
        merged = pd.concat([old[~old["symbol"].isin(done)] if not old.empty else old, new_df],
                           ignore_index=True)
        store.write(merged, path)
        stamp = pd.DataFrame({"symbol": done, "fetched": pd.Timestamp.now()})
        fetched = pd.concat([fetched[~fetched["symbol"].isin(done)], stamp], ignore_index=True)
        store.write(fetched, meta_path)
        return merged
    return old


# --------------------------------------------------------------------------
# Factors (point-in-time)
# --------------------------------------------------------------------------
def _ttm(s: pd.Series) -> pd.Series:
    return s.rolling(4, min_periods=4).sum()


def fundamental_factors(fund: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Growth and quality factors per symbol using only data available at `asof`."""
    cols = ["symbol", "rev_growth_yoy", "eps_growth_yoy", "rev_growth_accel",
            "gross_margin_trend", "operating_margin", "fcf_margin", "fund_quarters"]
    if fund is None or fund.empty:
        return pd.DataFrame(columns=cols)
    f = fund[pd.to_datetime(fund["available"]) <= asof]
    rows = []
    for sym, g in f.groupby("symbol"):
        w = (g.pivot_table(index="period_end", columns="metric", values="value", aggfunc="last")
             .sort_index())
        w = w[w.index > asof - pd.DateOffset(years=3)]
        r = {"symbol": sym, "fund_quarters": len(w)}
        if "revenue" in w and len(w) >= 5:
            rev = w["revenue"]
            q_yoy = rev / rev.shift(4) - 1
            ttm = _ttm(rev)
            r["rev_growth_yoy"] = (ttm / ttm.shift(4) - 1).iloc[-1] if len(w) >= 8 else q_yoy.iloc[-1]
            r["rev_growth_accel"] = q_yoy.iloc[-1] - q_yoy.iloc[-2] if len(w) >= 6 else np.nan
            for m, key in (("gross_profit", "gm"), ("operating_income", "om"), ("fcf", "fcfm")):
                if m in w:
                    margin = (_ttm(w[m]) / ttm) if len(w) >= 8 else (w[m] / rev)
                    r[key] = margin
            if "gm" in r:
                gm = r.pop("gm")
                r["gross_margin_trend"] = gm.iloc[-1] - gm.shift(4).iloc[-1] if len(gm) > 4 else np.nan
            if "om" in r:
                r["operating_margin"] = r.pop("om").iloc[-1]
            if "fcfm" in r:
                r["fcf_margin"] = r.pop("fcfm").iloc[-1]
        if "eps" in w and len(w) >= 5:
            eps = w["eps"]
            base = _ttm(eps).shift(4) if len(w) >= 8 else eps.shift(4)
            cur = _ttm(eps) if len(w) >= 8 else eps
            b = base.iloc[-1]
            r["eps_growth_yoy"] = (cur.iloc[-1] - b) / abs(b) if b and not np.isnan(b) else np.nan
        rows.append(r)
    out = pd.DataFrame(rows).reindex(columns=cols)
    num = out.columns.drop("symbol")
    out[num] = out[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out

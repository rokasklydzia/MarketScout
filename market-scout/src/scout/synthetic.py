"""Synthetic market for tests and offline demo runs.

Generates a small US + EU universe whose prices follow a factor model with
persistent stock-level drift (so momentum has something to find), plus
deliberate data defects that the QA layer must catch: a missing symbol, a real
split, a fake split error, a suspended (stale) stock and London prices in pence.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

SECTORS = ["Information Technology", "Financials", "Health Care", "Industrials",
           "Consumer Discretionary", "Energy", "Materials", "Utilities"]
EU_LISTINGS = [(".L", "United Kingdom", "GBP", "London Stock Exchange"),
               (".DE", "Germany", "EUR", "Xetra"), (".PA", "France", "EUR", "Nyse Euronext - Euronext Paris"),
               (".SW", "Switzerland", "CHF", "SIX Swiss Exchange"), (".ST", "Sweden", "SEK", "Nasdaq Omx Nordic"),
               (".AS", "Netherlands", "EUR", "Euronext Amsterdam")]
FX_LEVELS = {"USD": 1.10, "GBP": 0.86, "CHF": 0.95, "SEK": 11.3, "DKK": 7.46, "NOK": 11.5, "PLN": 4.3}


def synthetic_universe(n_us: int = 120, n_eu: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_us):
        rows.append({"region": "us", "source_ticker": f"US{i:03d}", "symbol": f"US{i:03d}",
                     "name": f"US Company {i}", "sector": SECTORS[i % len(SECTORS)], "country": "United States",
                     "exchange": "NASDAQ", "currency": "USD", "index_weight": rng.uniform(0.01, 1)})
    for i in range(n_eu):
        suf, ctry, ccy, ex = EU_LISTINGS[i % len(EU_LISTINGS)]
        rows.append({"region": "eu", "source_ticker": f"EU{i:03d}", "symbol": f"EU{i:03d}{suf}",
                     "name": f"EU Company {i}", "sector": SECTORS[(i * 3) % len(SECTORS)], "country": ctry,
                     "exchange": ex, "currency": ccy, "index_weight": rng.uniform(0.01, 1)})
    return pd.DataFrame(rows)


class SyntheticSource:
    """Implements download.PriceSource over a generated market."""

    def __init__(self, universe: pd.DataFrame, start: date, end: date, seed: int = 42,
                 missing: tuple[str, ...] = ("US119",)):
        self.rng = np.random.default_rng(seed)
        self.universe = universe
        self.missing = set(missing)
        self.dates = pd.bdate_range(start, end)
        self._build()

    # -- generation --------------------------------------------------------
    def _build(self) -> None:
        rng, n = self.rng, len(self.dates)
        uni = self.universe
        regions = {"us": rng.normal(0.0004, 0.010, n), "eu": rng.normal(0.0003, 0.009, n)}
        sec_f = {s: rng.normal(0, 0.006, n) for s in SECTORS}
        frames, self.drift = [], {}
        for r in uni.itertuples():
            beta = rng.uniform(0.7, 1.3)
            mu = rng.normal(0, 0.0006)              # persistent alpha -> momentum signal
            vol = rng.uniform(0.010, 0.025)
            ret = beta * regions[r.region] + sec_f[r.sector] + mu + rng.normal(0, vol, n)
            self.drift[r.symbol] = mu
            px = 50 * np.exp(np.cumsum(ret))
            if r.symbol.endswith(".L"):
                px = px * 100                       # quoted in pence like Yahoo
            frames.append(self._ohlcv(r.symbol, px))
        # benchmarks: equal-weight index of each region, in local currency for US / EUR for EU
        for reg, b in (("us", "IWB"), ("eu", "EXSA.DE")):
            px = 100 * np.exp(np.cumsum(regions[reg] + 0.00005))
            frames.append(self._ohlcv(b, px, volume=5e6))
        df = pd.concat(frames, ignore_index=True)
        df = self._inject_defects(df)
        self.prices = df
        # FX random walks (units per EUR)
        fx = []
        for ccy, lvl in FX_LEVELS.items():
            path = lvl * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
            fx.append(self._ohlcv(f"EUR{ccy}=X", path, volume=0))
        self.fx = pd.concat(fx, ignore_index=True)

    def _ohlcv(self, sym: str, px: np.ndarray, volume: float | None = None) -> pd.DataFrame:
        rng, n = self.rng, len(px)
        wig = np.abs(rng.normal(0, 0.006, n))
        vol = volume if volume is not None else rng.lognormal(np.log(4e5), 0.6)
        return pd.DataFrame({
            "date": self.dates, "symbol": sym, "open": np.r_[px[0], px[:-1]],
            "high": px * (1 + wig), "low": px * (1 - wig), "close": px, "adj_close": px,
            "volume": np.maximum(rng.normal(vol, vol * 0.3, n), 0).round() if vol else 0.0,
            "dividends": 0.0, "splits": 0.0})

    def _inject_defects(self, df: pd.DataFrame) -> pd.DataFrame:
        mid = self.dates[len(self.dates) // 2]
        # 1) real 2:1 split on US001: unadjusted close halves, adj_close continuous
        m = (df.symbol == "US001") & (df.date >= mid)
        df.loc[m, ["open", "high", "low", "close"]] /= 2
        df.loc[(df.symbol == "US001") & (df.date == mid), "splits"] = 2.0
        # 2) fake split error on US002: adj_close jumps 70% with no split
        m = (df.symbol == "US002") & (df.date >= mid)
        df.loc[m, ["open", "high", "low", "close", "adj_close"]] *= 1.7
        # 3) suspended stock US003: flat price, zero volume for 10 days
        sus = self.dates[100:110]
        m = (df.symbol == "US003") & df.date.isin(sus)
        price = df.loc[(df.symbol == "US003") & (df.date == sus[0]), "close"].iat[0]
        df.loc[m, ["open", "high", "low", "close", "adj_close"]] = price
        df.loc[m, "volume"] = 0
        # 4) short history: EU000.L listed only in the last 150 days
        df = df[~((df.symbol == "EU000.L") & (df.date < self.dates[-150]))]
        return df

    # -- PriceSource protocol ---------------------------------------------
    def history(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        pool = self.fx if all(s.endswith("=X") for s in symbols) else self.prices
        m = (pool.symbol.isin(set(symbols) - self.missing) & (pool.date >= pd.Timestamp(start))
             & (pool.date <= pd.Timestamp(end)))
        return pool[m].reset_index(drop=True)

    def currencies(self, symbols: list[str]) -> dict[str, str]:
        ccy = dict(zip(self.universe.symbol, self.universe.currency))
        ccy.update({"IWB": "USD", "EXSA.DE": "EUR"})
        return {s: ("GBp" if s.endswith(".L") else ccy.get(s, "USD")) for s in symbols}

    # -- fundamentals --------------------------------------------------------
    def fundamentals(self) -> pd.DataFrame:
        rng = self.rng
        q_ends = pd.date_range(end=self.dates[-1], periods=12, freq="QE")
        rows = []
        for r in self.universe.itertuples():
            if r.region == "eu" and rng.random() < 0.25:
                continue  # incomplete European coverage
            g = 0.02 + 40 * self.drift[r.symbol] + rng.normal(0, 0.01)   # quarterly growth
            rev = 1e9 * np.exp(np.cumsum(np.full(12, g) + rng.normal(0, 0.02, 12)))
            gm = np.clip(rng.uniform(0.2, 0.6) + np.linspace(0, rng.normal(0, 0.03), 12), 0.05, 0.9)
            om = gm - rng.uniform(0.05, 0.2)
            eps = rev * om / 1e8 * (1 + rng.normal(0, 0.05, 12))
            fcf = rev * (om - 0.03)
            for k, pe in enumerate(q_ends):
                avail = pe + pd.Timedelta(days=40)
                for metric, val in (("revenue", rev[k]), ("gross_profit", rev[k] * gm[k]),
                                    ("operating_income", rev[k] * om[k]), ("eps", eps[k]), ("fcf", fcf[k])):
                    rows.append({"symbol": r.symbol, "metric": metric, "period_end": pe,
                                 "available": avail, "value": float(val)})
        return pd.DataFrame(rows)


def ishares_csv_text(universe: pd.DataFrame, region: str) -> str:
    """Render universe rows as an iShares-style holdings CSV (for parser tests)."""
    u = universe[universe.region == region]
    lines = ['"iShares Synthetic ETF"', '"Fund Holdings as of","Oct 08, 2026"', "",
             '"Ticker","Name","Sector","Asset Class","Market Value","Weight (%)","Notional Value",'
             '"Shares","Price","Location","Exchange","Currency","FX Rate","Market Currency"']
    for r in u.itertuples():
        lines.append(f'"{r.source_ticker}","{r.name}","{r.sector}","Equity","1,000","{r.index_weight:.2f}",'
                     f'"1,000","10","100","{r.country}","{r.exchange}","{r.currency}","1","{r.currency}"')
    lines.append('"XTSLA","BLK CSH FND TREASURY SL AGENCY","Cash and/or Derivatives","Money Market","1","0.10",'
                 '"1","1","1","United States","-","USD","1","USD"')
    lines += ["", '"The content contained herein is owned or licensed by BlackRock"']
    return "\n".join(lines)

"""Parquet storage layout.

data/
  universe/asof=YYYY-MM-DD.parquet
  raw/prices/region={us,eu}/year=YYYY/part.parquet
  raw/fx/fx_daily.parquet
  raw/fundamentals/fundamentals.parquet
  clean/prices_daily.parquet
  clean/qa_report.parquet
  features/asof=YYYY-MM-DD.parquet
  results/backtests/  results/screens/  results/reports/
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import Settings

PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume",
                 "dividends", "splits"]


class Store:
    def __init__(self, settings: Settings):
        self.s = settings
        self.root = settings.data_dir

    # ---- generic helpers -------------------------------------------------
    def p(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def write(df: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)  # atomic on the same filesystem
        return path

    @staticmethod
    def read(path: Path, **kw) -> pd.DataFrame:
        return pd.read_parquet(path, **kw)

    # ---- universe --------------------------------------------------------
    def universe_path(self, asof: date) -> Path:
        return self.p("universe", f"asof={asof:%Y-%m-%d}.parquet")

    def latest_universe(self) -> pd.DataFrame:
        files = sorted((self.root / "universe").glob("asof=*.parquet"))
        if not files:
            raise FileNotFoundError("No universe snapshot found. Run the universe step first.")
        return pd.read_parquet(files[-1])

    # ---- raw prices ------------------------------------------------------
    def raw_prices_dir(self) -> Path:
        d = self.root / "raw" / "prices"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_raw_prices(self, df: pd.DataFrame) -> None:
        """Upsert raw prices, partitioned by region and year (last write wins)."""
        if df.empty:
            return
        df = df.copy()
        df["year"] = pd.to_datetime(df["date"]).dt.year
        for (region, year), part in df.groupby(["region", "year"]):
            path = self.raw_prices_dir() / f"region={region}" / f"year={year}" / "part.parquet"
            part = part.drop(columns=["year"])
            if path.exists():
                old = pd.read_parquet(path)
                part = (pd.concat([old, part], ignore_index=True)
                        .drop_duplicates(["symbol", "date"], keep="last"))
            self.write(part.sort_values(["symbol", "date"]), path)

    def read_raw_prices(self) -> pd.DataFrame:
        files = sorted(self.raw_prices_dir().glob("region=*/year=*/part.parquet"))
        if not files:
            return pd.DataFrame(columns=PRICE_COLUMNS + ["region", "currency"])
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def last_price_date(self) -> pd.Timestamp | None:
        df = self.read_raw_prices()
        if df.empty:
            return None
        return pd.to_datetime(df["date"]).max()

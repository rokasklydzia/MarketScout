# market-scout

Weekly market analysis for **Russell 1000 + STOXX Europe 600**. It has two tracks:

- **Track A, strategy backtesting.** Six long-only strategy families are tested with walk-forward parameter selection, transaction costs, a 2× cost stress test and the deflated Sharpe ratio.
- **Track B, growth screening.** Stocks must pass hard gates, then get a composite score: 40% price strength, 35% growth, 15% quality, minus 10% risk. The top 25 per region go to a shortlist for IBKR watchlists.

All returns are measured in EUR. Storage is plain Parquet files, with no database.

## Setup (once)

```bash
cd market-scout
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pip install arch          # optional: GARCH(1,1) volatility (needs scipy; skipped automatically if blocked)
```

Edit `config/settings.yaml` and set `fundamentals.sec_user_agent` to include your email. SEC requires a contact address.

Check that everything works offline. This runs a synthetic market end to end in about 20 seconds:

```bash
python scripts/demo_synthetic.py      # -> demo_output/reports/dashboard.html
python -m pytest -q
```

## First real run

```bash
python scripts/backfill.py            # universe + 5y prices + FX + QA (about 1-2 h, rate-limit dependent)
python scripts/run_weekly.py          # fundamentals, features, both tracks, dashboard, report
```

The first weekly run also pulls fundamentals for all names: SEC for the US and Yahoo for Europe. That adds about 15–30 minutes. Later runs only refresh fundamentals older than 7 days.

**If a holdings download fails:** the iShares pages sometimes block scripts. Open the fund page in a browser and download the holdings CSV. Save it as `data/manual/IWB_holdings.csv` or `data/manual/EXSA_holdings.csv`, and the pipeline will use it.

**Unmapped symbols:** check `data/universe/unmapped_*.csv`, and the QA flags shown in the dashboard. Fix any wrong Yahoo symbols in `config/symbol_overrides.csv`.

## Regular jobs (to schedule later)

| Job | Command | Suggested time (Vilnius) |
|---|---|---|
| Daily prices + QA | `python scripts/run_daily.py` | Tue–Sat 07:00 |
| Weekly analysis | `python scripts/run_weekly.py` | Saturday 09:00 |
| IBKR watchlists | Claude session reads `data/results/screens/shortlist_latest.csv` | Saturday 11:00 |

`run_daily.py` exits with code 2 when more than 2% of symbols fail, so a scheduler can flag it. Every step is idempotent. A missed day is caught up because each run re-pulls the last 10 trading days.

Fast rerun without refreshing the universe, fundamentals or GARCH:
`python scripts/run_weekly.py --no-universe-refresh --skip-fundamentals --no-garch`

## Outputs

| File | What |
|---|---|
| `reports/dashboard.html` | Regime, growth screen (sortable tables + price charts), strategy lab, data health. Works offline. |
| `reports/weekly_YYYY-Www.md` | One-page weekly report. Open it in any Markdown viewer and print to PDF if needed. |
| `data/results/screens/shortlist_latest.csv` | Top 25 per region, handed to IBKR. |
| `data/results/backtests/summary_*.parquet` | Per strategy and region: OOS metrics, DSR, stress test, verdict. |
| `data/results/backtests/trial_registry.parquet` | Every variant ever tested. This is the N used in the DSR. |
| `data/clean/qa_report.parquet` | Per-symbol data-quality flags. |

## Layout

```
config/settings.yaml          all parameters (paths, costs, gates, weights)
src/scout/universe.py         iShares holdings -> Yahoo symbols, dated snapshots
src/scout/download.py         yfinance prices/FX behind a swappable PriceSource
src/scout/qa.py               QA checks, GBp->GBP, EUR conversion
src/scout/fundamentals.py     SEC EDGAR (US) + yfinance (EU), point-in-time factors
src/scout/features.py         momentum, trend, RS, risk, liquidity, GARCH
src/scout/backtest.py         engine, strategies, walk-forward, DSR, bootstrap
src/scout/screen.py           gates + composite score, shortlist, exits
src/scout/report.py           dashboard + weekly report
src/scout/pipeline.py         daily / weekly orchestration
src/scout/synthetic.py        synthetic market for tests and demo
```

**Windows application control.** Smart App Control or a company policy can block compiled
Python extensions ("DLL load failed ... Application Control policy has blocked this file").
The package itself only needs numpy, pandas, pyarrow, requests, yfinance and plotly. If one of
those is blocked, run the pipeline in WSL or on Azure instead.

## Known limitations

- **Survivorship bias.** The universe is today's index members, so absolute backtest returns are overstated. Judge strategies relative to the benchmark. The weekly universe snapshots fix this from now on.
- **yfinance is unofficial.** Pin the version that works. To use a paid source, implement `PriceSource` in `download.py`.
- **European fundamentals are thinner.** Yahoo returns about 5 quarters with no filing dates, so availability is assumed 60 days after quarter end. A missing growth pillar counts as neutral (0).
- **The 5-year window covers about one cycle.** Folds start after a 1-year feature warm-up. The first training window is therefore about 1 year of usable returns, followed by 6 test blocks of 6 months each.

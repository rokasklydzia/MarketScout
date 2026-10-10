"""Weekly job: universe refresh, prices, fundamentals, features, both tracks,
dashboard, weekly report and the IBKR shortlist file.

    python scripts/run_weekly.py
    python scripts/run_weekly.py --no-universe-refresh --skip-fundamentals --no-garch   # fast rerun
"""
import argparse

import _bootstrap  # noqa: F401
from scout.config import load_settings, setup_logging
from scout.pipeline import RunContext, run_weekly


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-universe-refresh", action="store_true", help="reuse the latest universe snapshot")
    ap.add_argument("--skip-fundamentals", action="store_true", help="reuse cached fundamentals")
    ap.add_argument("--no-garch", action="store_true", help="skip GARCH(1,1) fits (faster)")
    args = ap.parse_args()
    setup_logging()
    out = run_weekly(RunContext(load_settings(args.config)),
                     refresh_universe=not args.no_universe_refresh,
                     skip_fundamentals=args.skip_fundamentals, with_garch=not args.no_garch)
    print(f"Weekly run as of {out['asof']:%Y-%m-%d}")
    print(f"  dashboard : {out['dashboard']}")
    print(f"  report    : {out['weekly_report']}")
    print(f"  shortlist : {out['shortlist']}  ({len(out['shortlist_df'])} names)")


if __name__ == "__main__":
    main()

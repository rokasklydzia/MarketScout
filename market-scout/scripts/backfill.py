"""One-off full backfill: universe + 5 years of prices + FX + QA.

    python scripts/backfill.py
"""
import argparse

import _bootstrap  # noqa: F401
from scout.config import load_settings, setup_logging
from scout.pipeline import RunContext, run_daily


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to settings.yaml")
    args = ap.parse_args()
    setup_logging()
    out = run_daily(RunContext(load_settings(args.config)), full=True)
    print(f"Backfill done: {out['clean_rows']:,} clean rows; "
          f"{out['failed']}/{out['requested']} symbols failed"
          + (" (PARTIAL run - rerun later)" if out["partial"] else ""))


if __name__ == "__main__":
    main()

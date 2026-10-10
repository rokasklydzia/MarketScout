"""Daily job: incremental prices (last 10 trading days re-pulled), FX, QA.

    python scripts/run_daily.py
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from scout.config import load_settings, setup_logging
from scout.pipeline import RunContext, run_daily


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    out = run_daily(RunContext(load_settings(args.config)))
    print(f"Daily update: {out['clean_rows']:,} clean rows; {out['failed']}/{out['requested']} failed")
    return 2 if out["partial"] else 0


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end run on a synthetic market (no internet needed).

Writes to ./demo_output so your real data/ folder is untouched.

    python scripts/demo_synthetic.py
"""
import argparse
import shutil
from datetime import date, timedelta
from pathlib import Path

import _bootstrap  # noqa: F401
from scout.config import load_settings, setup_logging
from scout.pipeline import RunContext, run_weekly
from scout.synthetic import SyntheticSource, synthetic_universe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--keep", action="store_true", help="do not wipe the output folder first")
    args = ap.parse_args()
    setup_logging()
    out_dir = Path(args.out).resolve()
    if out_dir.exists() and not args.keep:
        shutil.rmtree(out_dir)
    s = load_settings(overrides={"data_dir": str(out_dir / "data"), "reports_dir": str(out_dir / "reports"),
                                 "download": {"pause_seconds": 0}})
    end = date(2026, 10, 9)
    uni = synthetic_universe()
    src = SyntheticSource(uni, end - timedelta(days=int(365.25 * 5)), end)
    out = run_weekly(RunContext(s, source=src, universe_df=uni, fundamentals_df=src.fundamentals(), end=end),
                     refresh_universe=False)
    print(f"Demo done. Dashboard: {out['dashboard']}\nReport: {out['weekly_report']}")
    print(out["track_a"][["region", "family", "excess_ann", "dsr", "passes"]].to_string(index=False))


if __name__ == "__main__":
    main()

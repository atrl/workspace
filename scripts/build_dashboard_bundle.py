#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], description: str) -> None:
    print(f"[dashboard-bundle] {description}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local SOXL dashboard artifacts.")
    parser.add_argument(
        "--skip-v2-check",
        action="store_true",
        help="Build v2 data but skip the v2 contract checker.",
    )
    parser.add_argument(
        "--skip-retention-gate",
        action="store_true",
        help="Skip refreshing the model retention gate sidecars before dashboard v2 data build.",
    )
    args = parser.parse_args()

    python = sys.executable
    _run([python, "scripts/build_backtest_dashboard.py"], "dashboard.html build")
    if not args.skip_retention_gate:
        _run([python, "scripts/review_model_retention_gates.py"], "model retention gate review")
    _run([python, "scripts/build_dashboard_v2_data.py", "--detect-launchd"], "dashboard v2 data build")
    if not args.skip_v2_check:
        _run([python, "scripts/check_dashboard_v2_data.py"], "dashboard v2 data check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

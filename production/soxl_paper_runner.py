#!/usr/bin/env python3
from __future__ import annotations

"""Replay a saved SOXL actor through the runtime execution contract.

This script exists to validate the post-training path before touching any real
broker. It should remain the default proving ground for runtime changes.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.runtime import replay_profile
from src.soxl_quant.settings import RUNTIME_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOXL paper-trading replay runner")
    parser.add_argument("--profile", default="balanced", choices=["balanced", "aggressive", "smoke"])
    parser.add_argument("--start", required=True, help="start date, e.g. 2024-01-01")
    parser.add_argument("--end", required=True, help="end date, e.g. 2024-12-31")
    parser.add_argument("--tag", default="", help="optional output tag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = f"-{args.tag}" if args.tag else ""
    run_dir = RUNTIME_DIR / f"{args.profile}-{args.start}-{args.end}{suffix}"
    result = replay_profile(args.profile, start=args.start, end=args.end, output_dir=run_dir)
    record = result["record"]
    print(
        f"[paper] profile={record.profile} commands={record.commands} "
        f"approved={record.approved_signals} blocked={record.blocked_signals} "
        f"ending_value={record.ending_account_value:.2f}"
    )
    print(f"[paper] output={Path(run_dir)}")


if __name__ == "__main__":
    main()

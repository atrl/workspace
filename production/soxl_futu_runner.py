#!/usr/bin/env python3
from __future__ import annotations

"""Replay a saved SOXL actor through Futu simulated trading.

This runner bridges the runtime execution contract to Futu OpenAPI's
paper/simulate environment. It behaves like `soxl_paper_runner.py` but
routes orders through Futu instead of the local PaperBroker.

Prerequisites:
  1. OpenD gateway installed and running (default 127.0.0.1:11111)
  2. Futu account authenticated via OpenD QR code
  3. futu-api installed in the active Python environment
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.futu import FutuBroker
from src.soxl_quant.runtime import replay_profile
from src.soxl_quant.settings import RUNTIME_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOXL Futu simulated-trading replay runner")
    parser.add_argument("--profile", default="balanced", choices=["balanced", "aggressive", "smoke", "balanced_budget"])
    parser.add_argument("--start", required=True, help="start date, e.g. 2024-01-01")
    parser.add_argument("--end", required=True, help="end date, e.g. 2024-12-31")
    parser.add_argument("--tag", default="", help="optional output tag")
    parser.add_argument("--host", default="127.0.0.1", help="OpenD host")
    parser.add_argument("--port", type=int, default=11111, help="OpenD port")
    parser.add_argument("--market-order", action="store_true", help="use market order instead of limit")
    parser.add_argument("--unlock-password", default="", help="trade unlock password for Futu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = f"-{args.tag}" if args.tag else ""
    run_dir = RUNTIME_DIR / f"futu-{args.profile}-{args.start}-{args.end}{suffix}"

    broker = FutuBroker(
        host=args.host,
        port=args.port,
        symbol="US.SOXL",
        lot_size=1,
        use_market_order=args.market_order,
        unlock_password=args.unlock_password,
    )
    broker.connect()
    try:
        result = replay_profile(
            args.profile,
            start=args.start,
            end=args.end,
            output_dir=run_dir,
            broker=broker,
        )
        record = result["record"]
        print(
            f"[futu] profile={record.profile} commands={record.commands} "
            f"approved={record.approved_signals} blocked={record.blocked_signals} "
            f"ending_value={record.ending_account_value:.2f}"
        )
        print(f"[futu] output={Path(run_dir)}")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()

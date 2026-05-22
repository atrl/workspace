#!/usr/bin/env python3
from __future__ import annotations

"""Semi-automatic SOXL execution session for Futu simulated trading.

The first Futu paper phase is intentionally human-in-the-loop:
  1. generate the target position from the saved actor
  2. run execution-side risk gates
  3. print an approval payload for inspection
  4. only submit the order when --submit is present
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.futu import FutuBroker
from src.soxl_quant.runtime import plan_manual_session
from src.soxl_quant.settings import RUNTIME_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOXL Futu semi-automatic session")
    parser.add_argument("--profile", default="balanced", choices=["balanced", "aggressive", "smoke", "balanced_budget"])
    parser.add_argument("--as-of", required=True, help="signal date, e.g. 2026-04-15")
    parser.add_argument("--host", default="127.0.0.1", help="OpenD host")
    parser.add_argument("--port", type=int, default=11111, help="OpenD port")
    parser.add_argument("--market-order", action="store_true", help="use market order instead of limit")
    parser.add_argument("--unlock-password", default="", help="trade unlock password for Futu")
    parser.add_argument("--submit", action="store_true", help="actually submit the approved order")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = RUNTIME_DIR / f"futu-live-{args.as_of}"
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
        payload = plan_manual_session(args.profile, args.as_of, broker=broker, output_dir=session_dir)
        signal = payload["signal"]
        decision = payload["risk_gate"]
        account = payload["account"]
        reconcile = payload["reconcile"]
        print(json.dumps({k: asdict(v) for k, v in payload.items()}, ensure_ascii=False, indent=2))

        if not args.submit:
            print("[futu] dry-run only, no order submitted")
            return
        if not decision.approved:
            print("[futu] blocked by risk gate; order not submitted")
            return

        commands, statuses, result = broker.place_target_position(
            date=signal.date,
            symbol=signal.symbol,
            target_position=decision.capped_target_position,
            price=signal.price,
            reason=signal.note,
        )
        (session_dir / "proposed_orders.json").write_text(
            json.dumps([asdict(item) for item in commands], ensure_ascii=False, indent=2),
        )
        (session_dir / "broker_statuses.json").write_text(
            json.dumps([asdict(item) for item in statuses], ensure_ascii=False, indent=2),
        )
        (session_dir / "execution_result.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
        )
        print(f"[futu] order submitted: {asdict(result)}")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()

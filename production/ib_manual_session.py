#!/usr/bin/env python3
from __future__ import annotations

"""Semi-automatic SOXL execution session for Interactive Brokers.

The first live phase is intentionally human-in-the-loop:

1. generate the target position from the saved actor
2. run execution-side risk gates
3. print an approval payload for inspection
4. only submit the order when `--submit` is present
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.ibkr import IBConnectionConfig, IBBroker
from src.soxl_quant.runtime import plan_manual_session
from src.soxl_quant.settings import RUNTIME_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOXL IB semi-automatic session")
    parser.add_argument("--profile", default="balanced", choices=["balanced", "aggressive", "smoke"])
    parser.add_argument("--as-of", required=True, help="signal date, e.g. 2026-04-11")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497, help="IB Gateway/TWS port; 7497 is common paper default")
    parser.add_argument("--client-id", type=int, default=37)
    parser.add_argument("--account", default="", help="optional IB account id")
    parser.add_argument("--connect-timeout", type=float, default=10.0, help="IBKR SDK connect timeout in seconds")
    parser.add_argument("--submit", action="store_true", help="actually submit the approved order")
    parser.add_argument("--market-order", action="store_true", help="use market order instead of limit-at-last-price")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = RUNTIME_DIR / f"live-{args.as_of}"
    broker = IBBroker(
        config=IBConnectionConfig(
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            account=args.account,
            paper=args.port in {7497, 4002},
            connect_timeout_sec=args.connect_timeout,
        ),
        use_market_order=args.market_order,
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
            print("[ib] dry-run only, no order submitted")
            return
        if not decision.approved:
            print("[ib] blocked by risk gate; order not submitted")
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
            encoding="utf-8",
        )
        (session_dir / "broker_responses.json").write_text(
            json.dumps(
                {
                    "statuses": [asdict(item) for item in statuses],
                    "result": asdict(result),
                    "account": asdict(account),
                    "reconcile": asdict(reconcile),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[ib] submitted status={result.status} order_id={result.broker_order_id}")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Fast preflight for local IBKR TWS / IB Gateway socket readiness."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.ibkr import IBConnectionConfig, IBBroker, check_ibkr_port


HELP = """
IBKR TWS / IB Gateway must be installed, logged in, and API socket access must
be enabled before using the IBKR broker adapter.

Common default ports:
  TWS live:        7496
  TWS paper:       7497
  IB Gateway live: 4001
  IB Gateway paper:4002

In TWS, enable socket clients under:
  Global Configuration -> API -> Settings -> Enable ActiveX and Socket Clients
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check IBKR TWS/Gateway readiness")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=37)
    parser.add_argument("--account", default="")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--port-only", action="store_true", help="Only test TCP reachability, not SDK login readiness")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not check_ibkr_port(args.host, args.port, timeout=args.timeout):
        print(f"FAIL: IBKR TWS/Gateway is NOT reachable at {args.host}:{args.port}")
        print(HELP.strip())
        return 1

    if args.port_only:
        print(f"OK: IBKR TWS/Gateway socket is reachable at {args.host}:{args.port}")
        return 0

    broker = IBBroker(
        config=IBConnectionConfig(
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            account=args.account,
            paper=args.port in {7497, 4002},
            connect_timeout_sec=args.timeout,
        )
    )
    try:
        broker.connect()
        ok, msg = broker.healthcheck()
    except Exception as exc:
        print(f"FAIL: IBKR SDK connection failed at {args.host}:{args.port}: {exc}")
        print(HELP.strip())
        return 1
    finally:
        try:
            broker.disconnect()
        except Exception:
            pass

    if ok:
        print(f"OK: {msg}")
        return 0
    print(f"FAIL: IBKR SDK healthcheck failed: {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

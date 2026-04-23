#!/usr/bin/env python3
"""SOXL daily paper-trading runner.

Closed loop: data update -> model selection -> signal -> risk gate -> order -> notify -> persist.

Examples:
    PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py
    PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker futu --unlock-password XXX
    PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --dry-run
    PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --healthcheck
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production.futu_broker import FutuBroker
from production.notify import get_notifier
from scripts.update_market_data import update_master_csv, refresh_qlib
from src.soxl_quant.data import init_qlib
from src.soxl_quant.runtime import (
    BrokerAdapter,
    PaperBroker,
    auto_select_profile,
    plan_manual_session,
    promotion_gate_status,
)
from src.soxl_quant.settings import ACCOUNT, QLIB_DATA_PATH, RUNTIME_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOXL daily paper-trading runner")
    parser.add_argument("--profile-family", default="balanced", help="Model family for auto-selection")
    parser.add_argument("--profile", default="", help="Override: hard-coded profile name (skip auto-select)")
    parser.add_argument("--broker", default="paper", choices=["paper", "futu"], help="Broker adapter")
    parser.add_argument("--futu-host", default="127.0.0.1", help="OpenD host")
    parser.add_argument("--futu-port", type=int, default=11111, help="OpenD port")
    parser.add_argument("--unlock-password", default="", help="Futu trade unlock password")
    parser.add_argument("--market-order", action="store_true", help="Use market order (Futu only)")
    parser.add_argument("--notify", action="store_true", default=True, help="Send notifications")
    parser.add_argument("--no-notify", action="store_true", help="Disable notifications")
    parser.add_argument("--dry-run", action="store_true", help="Signal only, do not place orders")
    parser.add_argument("--healthcheck", action="store_true", help="Run health checks and exit")
    parser.add_argument("--skip-data-update", action="store_true", help="Skip market data update")
    parser.add_argument("--initial-cash", type=float, default=ACCOUNT, help="Initial cash for paper broker")
    return parser.parse_args()


def _init_broker(args: argparse.Namespace) -> BrokerAdapter:
    if args.broker == "futu":
        broker = FutuBroker(
            host=args.futu_host,
            port=args.futu_port,
            symbol="US.SOXL",
            lot_size=1,
            use_market_order=args.market_order,
            unlock_password=args.unlock_password,
        )
        broker.connect()
        return broker
    return PaperBroker(initial_cash=args.initial_cash)


def _last_trading_day() -> str:
    today = datetime.now().date()
    if today.weekday() == 0:
        delta = 3
    elif today.weekday() == 6:
        delta = 2
    else:
        delta = 1
    last = today - timedelta(days=delta)
    return last.strftime("%Y-%m-%d")


def _is_data_fresh(last_date_str: str) -> bool:
    last = pd.Timestamp(last_date_str).date()
    expected = pd.Timestamp(_last_trading_day()).date()
    return last >= expected


def _format_signal_message(
    signal,
    decision,
    account,
    reconcile,
    profile: str,
    model_path: Path,
    executed: bool,
) -> Tuple[str, str]:
    title = f"SOXL {'EXEC' if executed else 'BLOCKED'} | {signal.date}"
    body_lines = [
        f"Profile: {profile}",
        f"Model: {model_path.name}",
        f"Price: ${signal.price:.2f}",
        f"Current Pos: {signal.current_position:.1%}",
        f"Target Pos: {signal.target_position:.1%}",
        f"Action: {signal.action}",
        f"Risk Gate: {'APPROVED' if decision.approved else 'BLOCKED'}",
    ]
    if not decision.approved:
        body_lines.append(f"Reason: {'; '.join(decision.blocking_reasons)}")
    body_lines.extend([
        f"Account Value: ${account.account_value:,.2f}",
        f"Reconcile: {reconcile.note}",
    ])
    return title, "\n".join(body_lines)


def _determine_level(decision, reconcile) -> str:
    if not decision.approved:
        return "warning"
    gap = abs(reconcile.position_gap)
    if gap > 0.10:
        return "error"
    return "info"


def healthcheck(args: argparse.Namespace) -> Dict[str, object]:
    results: Dict[str, object] = {}

    from src.soxl_quant.settings import CSV_PATH
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    last_date = str(df["date"].max().date())
    data_ok = _is_data_fresh(last_date)
    results["data"] = {"ok": data_ok, "last_date": last_date}

    try:
        if args.profile:
            from src.soxl_quant.runtime import load_actor_bundle
            load_actor_bundle(args.profile)
            model_ok = True
            model_name = args.profile
        else:
            profile_name, model_path = auto_select_profile(args.profile_family)
            model_ok = True
            model_name = f"{profile_name} ({model_path.name})"
    except Exception as exc:
        model_ok = False
        model_name = str(exc)
    results["model"] = {"ok": model_ok, "name": model_name}

    broker: Optional[BrokerAdapter] = None
    try:
        broker = _init_broker(args)
        broker_ok, broker_msg = broker.healthcheck()
    except Exception as exc:
        broker_ok = False
        broker_msg = str(exc)
    finally:
        if broker is not None and hasattr(broker, "disconnect"):
            try:
                broker.disconnect()
            except Exception:
                pass
    results["broker"] = {"ok": broker_ok, "msg": broker_msg}

    notifier = get_notifier(prefer_bark=True)
    results["notify"] = {"channel": notifier.__class__.__name__}

    all_ok = data_ok and model_ok and broker_ok
    results["all_ok"] = all_ok
    return results


def main() -> None:
    args = parse_args()

    if args.healthcheck:
        results = healthcheck(args)
        print(json.dumps(results, indent=2, default=str))
        sys.exit(0 if results["all_ok"] else 1)

    use_notify = args.notify and not args.no_notify
    notifier = get_notifier(prefer_bark=True) if use_notify else None

    # 1. Data update
    if not args.skip_data_update:
        try:
            update_result = update_master_csv(dry_run=args.dry_run)
            refresh_qlib(dry_run=args.dry_run)
            if use_notify and update_result["new_rows"] > 0:
                notifier.send(
                    title="SOXL Data Updated",
                    body=f"New rows: {update_result['new_rows']}, last date: {update_result['last_date']}",
                    level="info",
                )
        except Exception as exc:
            msg = f"Data update failed: {exc}"
            print(f"[paper] ERROR: {msg}", file=sys.stderr)
            if use_notify:
                notifier.send(title="SOXL Data Update Failed", body=msg, level="error")
            sys.exit(1)
    else:
        print("[paper] skipped data update")

    # Re-init qlib so runtime reads the latest binary
    init_qlib(QLIB_DATA_PATH)

    # 2. Model selection
    if args.profile:
        profile_name = args.profile
        from src.soxl_quant.runtime import latest_model_path
        model_path = latest_model_path(profile_name)
        print(f"[paper] using hard-coded profile={profile_name} model={model_path.name}")
    else:
        profile_name, model_path = auto_select_profile(args.profile_family)
        print(f"[paper] auto-selected profile={profile_name} model={model_path.name}")

    # 3. Determine as_of date (latest available bar)
    from src.soxl_quant.settings import CSV_PATH
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    as_of = str(df["date"].max().date())
    print(f"[paper] as_of={as_of}")

    # 4. Broker
    broker = _init_broker(args)
    try:
        # 5. Signal generation
        result = plan_manual_session(
            profile_name=profile_name,
            as_of=as_of,
            broker=broker,
            model_path=model_path,
            output_dir=None,
        )
        signal = result["signal"]
        decision = result["risk_gate"]
        account = result["account"]
        reconcile = result["reconcile"]

        print(f"[paper] signal action={signal.action} target={signal.target_position:.2%} price=${signal.price:.2f}")
        print(f"[paper] risk_gate approved={decision.approved}")

        # 6. Execution
        executed = False
        if decision.approved and not args.dry_run:
            commands, statuses, exec_result = broker.place_target_position(
                date=signal.date,
                symbol=signal.symbol,
                target_position=decision.capped_target_position,
                price=signal.price,
                reason=signal.note,
            )
            executed = exec_result.submitted
            print(f"[paper] order submitted={exec_result.submitted} status={exec_result.status}")
        elif args.dry_run:
            print("[paper] dry-run: skipped order placement")
        else:
            print(f"[paper] blocked: {'; '.join(decision.blocking_reasons)}")

        # 7. Notification
        if use_notify:
            title, body = _format_signal_message(
                signal, decision, account, reconcile,
                profile_name, model_path, executed,
            )
            level = _determine_level(decision, reconcile)
            notifier.send(title=title, body=body, level=level)

        # 8. Persist
        run_dir = RUNTIME_DIR / "paper_trading" / signal.date
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, value in result.items():
            (run_dir / f"{name}.json").write_text(
                json.dumps(asdict(value), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        # Also write a summary line
        summary = {
            "date": signal.date,
            "profile": profile_name,
            "model": str(model_path),
            "broker": args.broker,
            "dry_run": args.dry_run,
            "executed": executed,
            "signal": asdict(signal),
            "risk_gate": asdict(decision),
            "account": asdict(account),
            "reconcile": asdict(reconcile),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"[paper] output={run_dir}")

    finally:
        if hasattr(broker, "disconnect"):
            broker.disconnect()


if __name__ == "__main__":
    main()

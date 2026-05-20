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
from dataclasses import asdict, replace
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
    next_runtime_execution_overlay_state,
    plan_manual_session,
    promotion_gate_status,
    reconcile_positions,
    save_runtime_execution_overlay_state,
)
from src.soxl_quant.settings import ACCOUNT, QLIB_DATA_PATH, RUNTIME_DIR, load_profile


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
    parser.add_argument("--discord", action="store_true", help="Also send notifications to Discord")
    parser.add_argument("--dry-run", action="store_true", help="Signal only, do not place orders")
    parser.add_argument("--healthcheck", action="store_true", help="Run health checks and exit")
    parser.add_argument("--skip-data-update", action="store_true", help="Skip market data update")
    parser.add_argument("--initial-cash", type=float, default=ACCOUNT, help="Initial cash for paper broker")
    parser.add_argument(
        "--paper-step-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In paper mode, clip large target changes to the risk-gate step limit instead of blocking",
    )
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
    return _load_paper_broker_state(initial_cash=args.initial_cash)


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
    return bool(_data_freshness_status(last_date_str)["ok"])


def _data_freshness_status(last_date_str: str, expected_date_str: Optional[str] = None) -> Dict[str, object]:
    last = pd.Timestamp(last_date_str).date()
    expected_str = expected_date_str or _last_trading_day()
    expected = pd.Timestamp(expected_str).date()
    stale_days = max((expected - last).days, 0)
    return {
        "ok": last >= expected,
        "last_date": last.isoformat(),
        "expected_last_trading_day": expected.isoformat(),
        "stale_days": int(stale_days),
    }


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


PAPER_TRADING_CSV = RUNTIME_DIR / "paper_trading" / "paper_trading_soxl_daily.csv"
PAPER_BROKER_STATE = RUNTIME_DIR / "paper_trading" / "paper_broker_state.json"
DAILY_DRAWDOWN_LIMIT = -0.20  # 20% single-day drawdown circuit breaker


def _load_paper_broker_state(
    state_path: Path = PAPER_BROKER_STATE,
    initial_cash: float = ACCOUNT,
) -> PaperBroker:
    broker = PaperBroker(initial_cash=initial_cash)
    if not state_path.exists():
        return broker
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    broker.cash = float(payload.get("cash", initial_cash))
    broker.shares = int(payload.get("shares", 0))
    return broker


def _save_paper_broker_state(
    broker: BrokerAdapter,
    date: str,
    price: float,
    state_path: Path = PAPER_BROKER_STATE,
) -> None:
    if not isinstance(broker, PaperBroker):
        return
    account = broker.get_account_snapshot(date, price)
    payload = {
        "date": account.date,
        "cash": account.cash,
        "shares": account.shares,
        "market_value": account.market_value,
        "account_value": account.account_value,
        "position": account.position,
        "price": float(price),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _check_daily_drawdown(account_value: float, drawdown_limit: float = DAILY_DRAWDOWN_LIMIT) -> Tuple[bool, float]:
    """Check if today's drawdown exceeds the circuit breaker threshold.

    Returns (ok, drawdown_pct) where ok=True means trading is allowed.
    """
    if not PAPER_TRADING_CSV.exists():
        return True, 0.0
    df = pd.read_csv(PAPER_TRADING_CSV)
    if df.empty or len(df) < 2:
        return True, 0.0
    peak = df["strategy_nav"].max()
    if peak <= 0:
        return True, 0.0
    drawdown = (account_value - peak) / peak
    return drawdown >= drawdown_limit, drawdown


def _append_daily_csv(
    date: str,
    price: float,
    strategy_nav: float,
    position: float,
    action: str,
    csv_path: Path = PAPER_TRADING_CSV,
) -> None:
    """Append a daily row to the cumulative paper-trading CSV (dashboard-compatible)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    prev_bh_nav = None
    prev_price = None
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty:
            prev_bh_nav = float(df["buyhold_nav"].iloc[-1])
            prev_price = float(df["price"].iloc[-1])

    if prev_bh_nav is not None and prev_price is not None and prev_price > 0:
        buyhold_nav = prev_bh_nav * (price / prev_price)
    else:
        buyhold_nav = ACCOUNT

    excess = strategy_nav - buyhold_nav
    excess_pct = (excess / buyhold_nav * 100) if buyhold_nav != 0 else 0.0

    row = {
        "datetime": date,
        "strategy_nav": round(strategy_nav, 6),
        "price": round(price, 6),
        "buyhold_nav": round(buyhold_nav, 6),
        "position": round(position, 6),
        "excess": round(excess, 6),
        "excess_pct": round(excess_pct, 6),
        "action": action,
    }

    import csv as csv_module
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[paper] appended to {csv_path}")


def _append_execution_daily_csv(
    signal,
    broker: BrokerAdapter,
    commands,
    csv_path: Path = PAPER_TRADING_CSV,
):
    """Append the dashboard row from the broker state after execution."""
    account = broker.get_account_snapshot(signal.date, signal.price)
    if commands and len(commands) > 0:
        action = commands[0].side
    else:
        action = "HOLD"
    _append_daily_csv(
        date=signal.date,
        price=signal.price,
        strategy_nav=account.account_value,
        position=account.position,
        action=action,
        csv_path=csv_path,
    )
    return account


def _post_execution_audit(signal, decision, broker: BrokerAdapter):
    account = broker.get_account_snapshot(signal.date, signal.price)
    expected_position = decision.capped_target_position if decision.approved else signal.current_position
    reconcile = reconcile_positions(signal.date, expected_position, account.position)
    return account, reconcile


def _block_risk_decision(decision, reason: str):
    return replace(
        decision,
        approved=False,
        blocking_reasons=(*decision.blocking_reasons, reason),
    )


def healthcheck(args: argparse.Namespace) -> Dict[str, object]:
    results: Dict[str, object] = {}

    from src.soxl_quant.settings import CSV_PATH
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    last_date = str(df["date"].max().date())
    data_status = _data_freshness_status(last_date)
    data_ok = bool(data_status["ok"])
    results["data"] = data_status

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

    notifier = get_notifier(prefer_bark=True, include_discord=args.discord)
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
    notifier = get_notifier(prefer_bark=True, include_discord=args.discord) if use_notify else None

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
            clip_position_change=args.broker == "paper" and args.paper_step_clip,
        )
        signal = result["signal"]
        decision = result["risk_gate"]
        account = result["account"]
        reconcile = result["reconcile"]

        print(f"[paper] signal action={signal.action} target={signal.target_position:.2%} price=${signal.price:.2f}")
        print(f"[paper] risk_gate approved={decision.approved}")

        # 6. Daily drawdown circuit breaker
        dd_ok, dd_pct = _check_daily_drawdown(account.account_value)
        if not dd_ok:
            print(f"[paper] CIRCUIT BREAKER: daily drawdown {dd_pct:.2%} exceeds limit {DAILY_DRAWDOWN_LIMIT:.2%}")
            decision = _block_risk_decision(
                decision,
                f"daily_drawdown_circuit_breaker: {dd_pct:.2%} < {DAILY_DRAWDOWN_LIMIT:.2%}",
            )
            result["risk_gate"] = decision
            if use_notify:
                notifier.send(
                    title="SOXL CIRCUIT BREAKER TRIGGERED",
                    body=f"Daily drawdown {dd_pct:.2%} exceeds limit {DAILY_DRAWDOWN_LIMIT:.2%}. All trading BLOCKED.",
                    level="error",
                )

        # 7. Execution
        executed = False
        commands = []
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

        if not args.dry_run:
            account, reconcile = _post_execution_audit(signal, decision, broker)
            result["account"] = account
            result["reconcile"] = reconcile
            profile_cfg = load_profile(profile_name)
            overlay_state = next_runtime_execution_overlay_state(
                profile_cfg,
                previous_state=result["execution_overlay_state"],
                signal=signal,
                account_value=account.account_value,
                order_submitted=executed,
            )
            save_runtime_execution_overlay_state(profile_cfg, overlay_state, date=signal.date)
            result["execution_overlay_state"] = overlay_state

        # 7. Notification
        if use_notify:
            title, body = _format_signal_message(
                signal, decision, account, reconcile,
                profile_name, model_path, executed,
            )
            level = _determine_level(decision, reconcile)
            notifier.send(title=title, body=body, level=level)

        # 8. Persist JSON
        run_dir = RUNTIME_DIR / "paper_trading" / signal.date
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, value in result.items():
            (run_dir / f"{name}.json").write_text(
                json.dumps(asdict(value), ensure_ascii=False, indent=2), encoding="utf-8"
            )
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

        # 9. Append to cumulative CSV for dashboard
        if not args.dry_run:
            _append_execution_daily_csv(signal, broker, commands)
            _save_paper_broker_state(broker, signal.date, signal.price)

    finally:
        if hasattr(broker, "disconnect"):
            broker.disconnect()


if __name__ == "__main__":
    main()

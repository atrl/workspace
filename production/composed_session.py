#!/usr/bin/env python3
"""Semi-automatic composed budget-usage session for SOXL.

Runs TWO actors (slow + fast), composes their signals via budget_usage logic,
and produces a single target position for execution.

Usage:
    PYTHONPATH=. qlib_venv/bin/python production/composed_session.py \
        --slow-profile balanced \
        --fast-profile balanced_rule_residual_imitation_v4 \
        --as-of 2026-04-27 \
        --dry-run

    # With Futu broker (live trading)
    PYTHONPATH=. qlib_venv/bin/python production/composed_session.py \
        --slow-profile balanced \
        --fast-profile balanced_rule_residual_imitation_v4 \
        --as-of 2026-04-27 \
        --broker futu \
        --unlock-password XXX \
        --submit
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production.futu_broker import FutuBroker
from production.notify import get_notifier
from src.soxl_quant.policy.two_speed_budget import (
    TwoSpeedBudgetConfig,
    compose_budget_usage_position,
    infer_mode_from_slow_position,
    maybe_rebalance_position,
)
from src.soxl_quant.runtime import (
    BrokerAdapter,
    PaperBroker,
    plan_manual_session,
    promotion_gate_status,
    risk_gate,
)
from src.soxl_quant.settings import ACCOUNT, RUNTIME_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composed budget-usage session")
    parser.add_argument("--slow-profile", required=True, help="Slow actor profile name")
    parser.add_argument("--fast-profile", required=True, help="Fast actor profile name")
    parser.add_argument("--as-of", required=True, help="Signal date YYYY-MM-DD")
    parser.add_argument("--broker", default="paper", choices=["paper", "futu"])
    parser.add_argument("--futu-host", default="127.0.0.1")
    parser.add_argument("--futu-port", type=int, default=11111)
    parser.add_argument("--unlock-password", default="")
    parser.add_argument("--market-order", action="store_true")
    parser.add_argument("--submit", action="store_true", help="Actually submit orders")
    parser.add_argument("--dry-run", action="store_true", help="Signal only, no broker interaction")
    parser.add_argument("--notify", action="store_true", default=True)
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--initial-cash", type=float, default=ACCOUNT)
    return parser.parse_args()


def _init_broker(args: argparse.Namespace) -> BrokerAdapter:
    if args.dry_run:
        return PaperBroker(initial_cash=args.initial_cash)
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


def main() -> None:
    args = parse_args()
    session_dir = RUNTIME_DIR / f"composed-live-{args.as_of}"
    session_dir.mkdir(parents=True, exist_ok=True)

    broker = _init_broker(args)

    # Run both profiles independently through the existing pipeline
    slow_payload = plan_manual_session(args.slow_profile, args.as_of, broker=broker)
    fast_payload = plan_manual_session(args.fast_profile, args.as_of, broker=broker)

    slow_signal = slow_payload["signal"]
    fast_signal = fast_payload["signal"]

    slow_pos = slow_signal.target_position
    fast_pos = fast_signal.target_position

    # Compose via budget_usage
    mode = infer_mode_from_slow_position(slow_pos)
    cfg = TwoSpeedBudgetConfig()
    raw_target = compose_budget_usage_position(fast_pos, mode, cfg)

    # Rebalance (respect trade threshold and lock)
    current_pos = slow_signal.current_position
    # In production, track days_since_trade from a persistent ledger
    # For now, assume we can trade if threshold is met
    days_since_trade = cfg.trade_lock_days  # conservative: allow immediate
    target = maybe_rebalance_position(current_pos, raw_target, days_since_trade, cfg)

    # Build synthetic signal for risk gate using slow profile config
    from src.soxl_quant.runtime import SignalIntent

    signal = SignalIntent(
        date=slow_signal.date,
        symbol="SOXL",
        profile=args.slow_profile,
        model_path=slow_signal.model_path,
        seed=None,
        action=0 if abs(target - current_pos) < 0.01 else (1 if target > current_pos else 2),
        target_position=target,
        current_position=current_pos,
        price=slow_signal.price,
        regime_risk=0.0,
        note=f"composed(slow={args.slow_profile}:{slow_pos:.2%}, fast={args.fast_profile}:{fast_pos:.2%}, mode={mode}, raw={raw_target:.2%})",
    )

    from pathlib import Path
    decision = risk_gate(broker, signal, model_ready=promotion_gate_status(Path(slow_signal.model_path)))

    payload = {
        "date": slow_signal.date,
        "slow_profile": args.slow_profile,
        "fast_profile": args.fast_profile,
        "slow_position": slow_pos,
        "fast_position": fast_pos,
        "mode": mode,
        "raw_target": raw_target,
        "target_position": target,
        "current_position": current_pos,
        "price": slow_signal.price,
        "approved": decision.approved,
        "blocking_reasons": decision.blocking_reasons,
        "capped_target": decision.capped_target_position if decision.approved else current_pos,
        "note": signal.note,
    }

    (session_dir / "composed_signal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 60)
    print("COMPOSED BUDGET-USAGE SIGNAL")
    print("=" * 60)
    print(f"Date:       {slow_signal.date}")
    print(f"Price:      ${slow_signal.price:.2f}")
    print(f"Slow:       {args.slow_profile} = {slow_pos:.2%}")
    print(f"Fast:       {args.fast_profile} = {fast_pos:.2%}")
    print(f"Mode:       {mode}")
    print(f"Band:       {cfg.mode_bands.get(mode)}")
    print(f"Raw target: {raw_target:.2%}")
    print(f"Current:    {current_pos:.2%}")
    print(f"Final:      {target:.2%}")
    print(f"Risk gate:  {'APPROVED' if decision.approved else 'BLOCKED'}")
    if not decision.approved:
        print(f"Reasons:    {decision.blocking_reasons}")
    print("=" * 60)

    if args.dry_run:
        print("[composed] dry-run only, no order submitted")
        return

    if not decision.approved:
        print(f"[composed] BLOCKED by risk gate: {decision.blocking_reasons}")
        return

    if not args.submit:
        print("[composed] signal generated but --submit not set; no order placed")
        return

    # Place order
    commands, statuses, result = broker.place_target_position(
        date=slow_signal.date,
        symbol="SOXL",
        target_position=decision.capped_target_position,
        price=slow_signal.price,
        reason=signal.note,
    )
    (session_dir / "proposed_orders.json").write_text(
        json.dumps([asdict(item) for item in commands], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (session_dir / "broker_statuses.json").write_text(
        json.dumps([asdict(item) for item in statuses], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (session_dir / "execution_result.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    notifier = get_notifier(prefer_bark=True) if args.notify and not args.no_notify else None
    if notifier:
        title = f"SOXL COMPOSED {'EXEC' if result.success else 'FAIL'} | {slow_signal.date}"
        body = (
            f"Slow: {args.slow_profile}={slow_pos:.1%}\n"
            f"Fast: {args.fast_profile}={fast_pos:.1%}\n"
            f"Mode: {mode}\n"
            f"Target: {target:.1%} → Capped: {decision.capped_target_position:.1%}\n"
            f"Price: ${slow_signal.price:.2f}"
        )
        notifier.send(title, body)

    print(f"[composed] order {'SUCCESS' if result.success else 'FAIL'}: {result.note}")


if __name__ == "__main__":
    main()

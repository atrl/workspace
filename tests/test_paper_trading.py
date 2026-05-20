from __future__ import annotations

from pathlib import Path

import pandas as pd

from production.soxl_paper_trading import (
    _append_execution_daily_csv,
    _block_risk_decision,
    _load_paper_broker_state,
    _post_execution_audit,
    _save_paper_broker_state,
)
from src.soxl_quant.runtime import PaperBroker, SignalIntent, risk_gate


def test_append_execution_daily_csv_uses_post_order_paper_snapshot(tmp_path: Path):
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    signal = SignalIntent(
        date="2026-05-14",
        symbol="SOXL",
        profile="balanced",
        model_path="runtime/model/card/balanced/soxl_balanced_actor.pt",
        seed=42,
        action=7,
        target_position=0.5,
        current_position=0.0,
        price=50.0,
        regime_risk=0.5,
        note="test",
    )
    commands, _statuses, _result = broker.place_target_position(
        date=signal.date,
        symbol=signal.symbol,
        target_position=signal.target_position,
        price=signal.price,
        reason=signal.note,
    )
    out = tmp_path / "paper_trading_soxl_daily.csv"

    _append_execution_daily_csv(signal, broker, commands, csv_path=out)

    row = pd.read_csv(out).iloc[-1]
    assert row["action"] == "BUY"
    assert row["position"] == 0.5
    assert row["strategy_nav"] == 100_000.0


def test_paper_broker_state_roundtrip(tmp_path: Path):
    state_path = tmp_path / "paper_broker_state.json"
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    broker.place_target_position(
        date="2026-05-14",
        symbol="SOXL",
        target_position=0.5,
        price=50.0,
        reason="test",
    )

    _save_paper_broker_state(broker, date="2026-05-14", price=50.0, state_path=state_path)
    restored = _load_paper_broker_state(state_path=state_path, initial_cash=100_000.0)

    assert restored.cash == broker.cash
    assert restored.shares == broker.shares
    assert restored.get_position("SOXL", 50.0) == broker.get_position("SOXL", 50.0)


def test_post_execution_audit_uses_realized_paper_position():
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    signal = SignalIntent(
        date="2026-05-14",
        symbol="SOXL",
        profile="balanced_rule_residual_v19",
        model_path="runtime/model/card/balanced_rule_residual_v19/soxl_balanced_rule_residual_v19_actor.pt",
        seed=42,
        action=0,
        target_position=0.65,
        current_position=0.0,
        price=50.0,
        regime_risk=0.5,
        note="test",
    )
    decision = risk_gate(broker, signal, max_position_change=0.35, clip_position_change=True)
    broker.place_target_position(
        date=signal.date,
        symbol=signal.symbol,
        target_position=decision.capped_target_position,
        price=signal.price,
        reason=signal.note,
    )

    account, reconcile = _post_execution_audit(signal, decision, broker)

    assert account.position == 0.35
    assert reconcile.expected_position == 0.35
    assert reconcile.action_required == "none"


def test_block_risk_decision_preserves_frozen_decision_contract():
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    signal = SignalIntent(
        date="2026-05-14",
        symbol="SOXL",
        profile="balanced_rule_residual_v19",
        model_path="runtime/model/card/balanced_rule_residual_v19/soxl_balanced_rule_residual_v19_actor.pt",
        seed=42,
        action=0,
        target_position=0.2,
        current_position=0.0,
        price=50.0,
        regime_risk=0.5,
        note="test",
    )
    decision = risk_gate(broker, signal)

    blocked = _block_risk_decision(decision, "daily_drawdown_circuit_breaker")

    assert decision.approved is True
    assert blocked.approved is False
    assert blocked.blocking_reasons[-1] == "daily_drawdown_circuit_breaker"

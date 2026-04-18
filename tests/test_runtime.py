from __future__ import annotations

from src.soxl_quant.rl import TrancheLedger
from src.soxl_quant.runtime import SignalIntent, PaperBroker, reconcile_positions, risk_gate


def test_position_ledger_tracks_entry_and_age():
    ledger = TrancheLedger()
    ledger.update(0.0, 0.4, 100.0)
    ledger.update(0.4, 0.4, 105.0)
    age, avg_pnl, add_pnl, high_gap, _last_tranche_pnl, _n_tranches = ledger.snapshot(102.0)
    assert age > 0.0
    assert avg_pnl > 0.0
    assert add_pnl > 0.0
    assert high_gap < 0.0


def test_paper_broker_creates_lot_sized_orders():
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    commands = broker.rebalance("2024-01-02", "SOXL", target_position=0.5, price=50.0, reason="test")
    assert len(commands) == 1
    assert commands[0].side == "BUY"
    assert commands[0].quantity % 100 == 0
    assert broker.shares == commands[0].quantity


def test_risk_gate_blocks_large_one_shot_position_jump():
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    signal = SignalIntent(
        date="2024-01-02",
        symbol="SOXL",
        profile="balanced",
        model_path="artifacts/models/example.pt",
        seed=20260411,
        action=10,
        target_position=0.8,
        current_position=0.0,
        price=50.0,
        regime_risk=0.3,
        note="regime=0.30",
    )
    decision = risk_gate(broker, signal, max_position_change=0.35)
    assert decision.approved is False
    assert any("exceeds limit" in reason for reason in decision.blocking_reasons)


def test_reconcile_positions_flags_meaningful_gap():
    record = reconcile_positions("2024-01-02", expected_position=0.40, actual_position=0.22, tolerance=0.03)
    assert record.action_required == "manual_review"
    assert record.position_gap < 0.0


def test_risk_gate_blocks_non_promoted_model():
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    signal = SignalIntent(
        date="2024-01-02",
        symbol="SOXL",
        profile="balanced",
        model_path="artifacts/models/example.pt",
        seed=20260411,
        action=5,
        target_position=0.2,
        current_position=0.1,
        price=50.0,
        regime_risk=0.3,
        note="regime=0.30",
    )
    decision = risk_gate(broker, signal, model_ready=(False, "model failed promotion gate"))
    assert decision.approved is False
    assert any("promotion gate" in reason for reason in decision.blocking_reasons)

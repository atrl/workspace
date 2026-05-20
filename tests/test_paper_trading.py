from __future__ import annotations

from pathlib import Path

import pandas as pd

from production.soxl_paper_trading import _append_execution_daily_csv
from src.soxl_quant.runtime import PaperBroker, SignalIntent


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

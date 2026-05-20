from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from production import soxl_paper_trading
from production.soxl_paper_trading import (
    _append_execution_daily_csv,
    _block_risk_decision,
    _data_freshness_status,
    _last_trading_day,
    _load_paper_broker_state,
    _run_data_update,
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


def test_data_freshness_status_reports_expected_date_and_lag():
    status = _data_freshness_status("2026-05-15", expected_date_str="2026-05-20")

    assert status == {
        "ok": False,
        "last_date": "2026-05-15",
        "expected_last_trading_day": "2026-05-20",
        "stale_days": 5,
    }


def test_last_trading_day_waits_for_new_york_post_close_cutoff():
    shanghai = ZoneInfo("Asia/Shanghai")

    before_us_close = datetime(2026, 5, 21, 2, 23, tzinfo=shanghai)
    after_post_close_window = datetime(2026, 5, 21, 5, 35, tzinfo=shanghai)

    assert _last_trading_day(now=before_us_close) == "2026-05-19"
    assert _last_trading_day(now=after_post_close_window) == "2026-05-20"


def test_last_trading_day_uses_previous_friday_before_monday_us_close():
    shanghai = ZoneInfo("Asia/Shanghai")
    monday_preopen = datetime(2026, 5, 18, 21, 0, tzinfo=shanghai)

    assert _last_trading_day(now=monday_preopen) == "2026-05-15"


def test_data_update_auto_provider_falls_back_to_yahoo_api(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_update_master_csv(*, dry_run: bool, provider: str):
        calls.append(provider)
        if provider == "yfinance":
            raise RuntimeError("No overlapping dates for SOXL/SOXX/VIX")
        return {
            "before_rows": 1,
            "after_rows": 2,
            "new_rows": 1,
            "last_date": "2026-05-19",
            "dry_run": dry_run,
        }

    monkeypatch.setattr(soxl_paper_trading, "update_master_csv", fake_update_master_csv)
    monkeypatch.setattr(soxl_paper_trading, "refresh_qlib", lambda dry_run: None)

    result = _run_data_update(dry_run=False, provider="auto")

    assert calls == ["yfinance", "yahoo_api"]
    assert result["last_date"] == "2026-05-19"


def test_main_blocks_non_dry_run_when_data_remains_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    csv_path = tmp_path / "soxl_features.csv"
    csv_path.write_text(
        "date,open,high,low,close,volume\n"
        "2026-05-15,1,1,1,1,100\n",
        encoding="utf-8",
    )
    initialized = False

    def fail_if_initialized(_qlib_path):
        nonlocal initialized
        initialized = True
        raise AssertionError("stale non-dry-run data should block before qlib init")

    monkeypatch.setattr(
        soxl_paper_trading,
        "update_master_csv",
        lambda dry_run, provider: {
            "before_rows": 1,
            "after_rows": 1,
            "new_rows": 0,
            "last_date": "2026-05-15",
            "dry_run": dry_run,
        },
    )
    monkeypatch.setattr(soxl_paper_trading, "refresh_qlib", lambda dry_run: None)
    monkeypatch.setattr(soxl_paper_trading, "init_qlib", fail_if_initialized)
    monkeypatch.setattr(soxl_paper_trading, "_last_trading_day", lambda: "2026-05-20")

    from src.soxl_quant import settings

    monkeypatch.setattr(settings, "CSV_PATH", csv_path)
    monkeypatch.setattr(sys, "argv", ["soxl_paper_trading.py", "--no-notify"])

    with pytest.raises(SystemExit) as exc_info:
        soxl_paper_trading.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert initialized is False
    assert "Data update completed but data is stale" in captured.err
    assert "last_date=2026-05-15 expected=2026-05-20 stale_days=5" in captured.err

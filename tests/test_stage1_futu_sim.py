from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest


ROOT = Path(__file__).resolve().parents[1]
NEW_YORK = ZoneInfo("America/New_York")


def test_inference_loader_keeps_latest_feature_rows_without_future_labels():
    from src.workflow.direct_continuous_soxx_allocation import (
        DirectContinuousAllocationSpec,
        _load_frame,
    )

    default, _, _ = _load_frame(DirectContinuousAllocationSpec(root=ROOT))
    inference, audit, features = _load_frame(
        DirectContinuousAllocationSpec(root=ROOT),
        include_incomplete_latest=True,
    )

    assert inference.index.max() > default.index.max()
    assert inference.iloc[-1][features].notna().all()
    assert inference.iloc[-1]["soxl_forward_open_return"] != inference.iloc[-1][
        "soxl_forward_open_return"
    ]
    assert audit["include_incomplete_latest"] is True


def test_stage1_runtime_target_has_exact_backtest_parity_and_legal_weights():
    from src.workflow.stage1_futu_sim import SOURCE_RECORDER_ID, build_runtime_target

    target = build_runtime_target(root=ROOT)

    assert target["status"] in {"ready", "pending_next_open_mark"}
    assert target["runtime_replay_parity_max_abs_error"] <= 1e-10
    assert sum(target["weights"].values()) == pytest.approx(1.0)
    assert all(value >= 0.0 for value in target["weights"].values())
    assert target["source_audit"]["stage1_recorder_id"] == SOURCE_RECORDER_ID


def test_stage1_matched_comparison_beats_legacy_reference_and_recomputes_risk():
    from src.workflow.intraday_recovery_direct_allocation import _passes_risk
    from src.workflow.stage1_futu_sim import build_matched_comparison

    comparison, stage1, legacy = build_matched_comparison(root=ROOT)

    assert not stage1.empty
    assert not legacy.empty
    assert isinstance(_passes_risk(comparison["stage1_10bps"]), bool)
    assert comparison["stage1_minus_soxlbacktest_10bps_return_pct_points"] > 0.0
    assert comparison["window"]["end"] >= "2026-07-24"


def test_futu_simulated_broker_dry_run_has_no_real_trading_path(tmp_path, monkeypatch):
    from src.api.futu_sim import FutuSimulatedBroker

    broker = FutuSimulatedBroker()
    broker._futu = SimpleNamespace(
        RET_OK=0,
        TrdEnv=SimpleNamespace(SIMULATE="SIMULATE"),
        TrdSide=SimpleNamespace(BUY="BUY", SELL="SELL"),
        OrderType=SimpleNamespace(MARKET="MARKET"),
    )
    broker._trade_ctx = object()
    broker._account_id = 123
    monkeypatch.setattr(
        broker,
        "account_snapshot",
        lambda: {
            "environment": "SIMULATE",
            "real_trading_supported": False,
            "positions": {"SOXL": 0, "SOXX": 0},
        },
    )

    result = broker.rebalance(
        weights={"SOXL": 0.2, "SOXX": 0.5, "cash": 0.3},
        prices={"SOXL": 50.0, "SOXX": 250.0},
        notional=10_000.0,
        signal_date="2026-07-24",
        slot="open",
        state_path=tmp_path / "state.json",
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["environment"] == "SIMULATE"
    assert result["real_broker_order"] is False
    assert result["target_quantities"] == {"SOXL": 40, "SOXX": 20}
    assert not (tmp_path / "state.json").exists()


def test_futu_simulated_broker_blocks_when_futu_market_is_closed(tmp_path, monkeypatch):
    from src.api.futu_sim import FutuSimulatedBroker

    broker = FutuSimulatedBroker()
    broker._futu = SimpleNamespace(
        RET_OK=0,
        TrdEnv=SimpleNamespace(SIMULATE="SIMULATE"),
        TrdSide=SimpleNamespace(BUY="BUY", SELL="SELL"),
        OrderType=SimpleNamespace(MARKET="MARKET"),
    )
    broker._trade_ctx = object()
    broker._account_id = 123
    monkeypatch.setattr(broker, "us_market_state", lambda: "CLOSED")

    with pytest.raises(RuntimeError, match="regular session is not open"):
        broker.rebalance(
            weights={"SOXL": 0.2, "SOXX": 0.5, "cash": 0.3},
            prices={"SOXL": 50.0, "SOXX": 250.0},
            notional=10_000.0,
            signal_date="2026-07-27",
            slot="open",
            state_path=tmp_path / "state.json",
            dry_run=False,
            now=datetime(2026, 7, 27, 9, 31, tzinfo=NEW_YORK),
        )

    assert not (tmp_path / "state.json").exists()


def test_stage1_schedule_detects_dst_safe_et_slots():
    from src.schedule.stage1_futu import detect_slot

    assert detect_slot(datetime(2026, 7, 27, 9, 31, tzinfo=NEW_YORK)) == "open"
    assert detect_slot(datetime(2026, 7, 27, 10, 31, tzinfo=NEW_YORK)) == "recovery"
    assert detect_slot(datetime(2026, 7, 27, 20, 20, tzinfo=NEW_YORK)) == "postclose"
    assert detect_slot(datetime(2026, 7, 26, 9, 31, tzinfo=NEW_YORK)) == "not_due"


def test_stage1_refresh_fails_fast_when_opend_is_unreachable(tmp_path, monkeypatch):
    from src.schedule import stage1_futu

    monkeypatch.setattr(stage1_futu, "opend_reachable", lambda _host, _port: False)
    monkeypatch.setattr(
        stage1_futu,
        "update_market_data",
        lambda **_kwargs: pytest.fail("market refresh must not start without OpenD"),
    )

    with pytest.raises(RuntimeError, match="OpenD is not reachable"):
        stage1_futu.refresh_stage1_data(root=tmp_path)


def test_stage1_execution_gate_blocks_failed_position_risk(tmp_path):
    from src.schedule.stage1_futu import _production_execution_gate

    card = tmp_path / "production" / "stage1_futu_sim_v1"
    card.mkdir(parents=True)
    (card / "summary.json").write_text(
        json.dumps(
            {
                "decision": "paper_sim_blocked_failed_position_risk_gates",
                "risk_gate_pass": False,
            }
        ),
        encoding="utf-8",
    )

    assert _production_execution_gate(tmp_path) == (
        False,
        "failed_position_risk_gates",
    )


def test_recovery_reuses_persisted_open_mark(tmp_path):
    from src.schedule.stage1_futu import _latest_open_settlement_prices

    state = {
        "version": 1,
        "environment": "SIMULATE",
        "real_trading_supported": False,
        "completed_slots": {
            "2026-07-23:open": {
                "execution_session_date": "2026-07-24",
                "prices": {"SOXL": 149.0, "SOXX": 540.0},
                "recorded_at": "2026-07-24T21:31:00+08:00",
            },
            "2026-07-24:open": {
                "execution_session_date": "2026-07-27",
                "prices": {"SOXL": 137.0, "SOXX": 528.0},
                "recorded_at": "2026-07-27T21:31:00+08:00",
            },
        },
    }
    path = tmp_path / "execution_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    assert _latest_open_settlement_prices(
        path,
        session_date="2026-07-27",
    ) == {"SOXL": 137.0, "SOXX": 528.0}


def test_stage1_production_card_contract():
    card = ROOT / "production" / "stage1_futu_sim_v1"
    notebook = json.loads((card / "stage1_futu_sim_v1.ipynb").read_text(encoding="utf-8"))
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert "SIMULATE" in json.dumps(notebook, ensure_ascii=False)
    assert "soxlbacktest" in json.dumps(notebook, ensure_ascii=False)
    expected_decision = (
        "paper_sim_enabled_waiting_next_us_session"
        if summary["risk_gate_pass"]
        else "paper_sim_blocked_failed_position_risk_gates"
    )
    assert summary["decision"] == expected_decision


def test_stage1_production_card_registers_dashboard_entrypoint():
    card = ROOT / "production" / "stage1_futu_sim_v1"
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    workflow = (card / "workflow.yaml").read_text(encoding="utf-8")

    assert summary["dashboard"]["source"] == "qlib_recorder"
    assert "custom_interval_inference" in summary["dashboard"]["features"]
    assert "production/run_stage1_dashboard.sh" in workflow
    assert "benchmarks: [SOXL, SOXX]" in workflow

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


def test_stage1_runtime_target_settles_pending_label_at_next_open():
    from src.workflow.stage1_futu_sim import build_runtime_target

    target = build_runtime_target(
        root=ROOT,
        current_prices={"SOXL": 126.21, "SOXX": 521.45},
    )

    assert target["status"] == "ready"
    assert target["signal_date"] > target["latest_complete_signal_date"]
    assert len(target["pending_signal_dates"]) == 1
    assert target["pending_settled"] == target["pending_signal_dates"]


def test_stage1_runtime_target_uses_fresh_paper_cycle_state_instead_of_historical_drawdown():
    from src.workflow.stage1_futu_sim import RuntimeState, build_runtime_target

    target = build_runtime_target(
        root=ROOT,
        current_prices={"SOXL": 114.72, "SOXX": 504.89},
        live_controller_state=RuntimeState(),
    )

    assert target["status"] == "ready"
    assert target["controller_state_source"] == "paper_ledger_cycle"
    assert target["portfolio_drawdown"] == pytest.approx(0.0)
    assert target["weights"]["SOXL"] > 0.0
    assert target["weights"]["SOXX"] > 0.3


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
        portfolio_equity=10_000.0,
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
            portfolio_equity=10_000.0,
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


def test_stage1_refresh_uses_daily_fallback_and_marks_intraday_stale_when_opend_is_unreachable(
    tmp_path, monkeypatch
):
    from src.schedule import stage1_futu

    monkeypatch.setattr(stage1_futu, "opend_reachable", lambda _host, _port: False)
    calls = []
    monkeypatch.setattr(
        stage1_futu,
        "update_market_data",
        lambda **kwargs: calls.append(kwargs)
        or {"new_last_date": "2026-07-30", "provider": "yahoo_api"},
    )
    monkeypatch.setattr(stage1_futu, "update_multi_3x_etf_data", lambda **_kwargs: {})
    monkeypatch.setattr(
        stage1_futu,
        "build_soxs_defensive_provider",
        lambda **_kwargs: {"qlib_manifest": {"calendar_end": "2026-07-30"}},
    )
    monkeypatch.setattr(
        stage1_futu,
        "rebuild_normalized_provider",
        lambda **_kwargs: {"calendar_end": "2026-07-30"},
    )

    result = stage1_futu.refresh_stage1_data(root=tmp_path, end="2026-07-30")

    assert calls[0]["provider"] == "auto"
    assert result["status"] == "needs_data"
    assert result["freshness"]["checks"]["market"] is True
    assert result["freshness"]["checks"]["intraday"] is False


def test_stage1_execution_gate_fails_closed_on_failed_position_risk(tmp_path):
    from src.schedule.stage1_futu import _production_execution_gate

    card = tmp_path / "production" / "stage1_futu_sim_v1"
    card.mkdir(parents=True)
    (card / "summary.json").write_text(
        json.dumps(
            {
                "decision": "paper_sim_blocked_failed_position_risk_gates",
                "risk_gate_pass": False,
                "latest_target": {
                    "status": "ready",
                    "signal_date": "2026-07-30",
                    "source_audit": {"market_audit": {"common_feature_end": "2026-07-30"}},
                },
            }
        ),
        encoding="utf-8",
    )

    assert _production_execution_gate(tmp_path) == (
        "blocked",
        "failed_position_risk_gates",
    )


def test_stage1_execution_gate_allows_explicit_restarted_paper_trial_with_live_controller(
    tmp_path,
):
    from src.schedule.stage1_futu import _production_execution_gate

    card = tmp_path / "production" / "stage1_futu_sim_v1"
    card.mkdir(parents=True)
    (card / "summary.json").write_text(
        json.dumps(
            {
                "decision": "paper_sim_blocked_failed_position_risk_gates",
                "risk_gate_pass": False,
                "latest_target": {"status": "blocked"},
            }
        ),
        encoding="utf-8",
    )
    target = {
        "status": "ready",
        "signal_date": "2026-07-31",
        "controller_state_source": "paper_ledger_cycle",
        "source_audit": {"market_audit": {"common_feature_end": "2026-07-31"}},
    }

    assert _production_execution_gate(
        tmp_path,
        target=target,
        required_signal_date="2026-07-31",
        paper_trial_authorized=True,
    ) == ("normal", "user_authorized_restarted_paper_trial")


def test_futu_simulated_broker_close_only_never_increases_positions(tmp_path, monkeypatch):
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
        lambda: {"positions": {"SOXL": 223, "SOXX": 99}},
    )

    result = broker.rebalance(
        weights={"SOXL": 0.4, "SOXX": 0.3, "cash": 0.3},
        prices={"SOXL": 100.0, "SOXX": 500.0},
        portfolio_equity=92_500.0,
        signal_date="2026-07-30",
        slot="open",
        state_path=tmp_path / "state.json",
        dry_run=True,
        execution_mode="close_only",
    )

    assert result["target_quantities"] == {"SOXL": 223, "SOXX": 55}
    assert result["orders"] == [
        {
            "symbol": "SOXX",
            "code": "US.SOXX",
            "side": "SELL",
            "quantity": 44,
            "status": "dry_run",
            "order_id": "",
        }
    ]


def test_futu_execution_slot_is_completed_only_after_fill_confirmation(tmp_path):
    from src.api.futu_sim import reconcile_execution_state

    state_path = tmp_path / "execution_state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "environment": "SIMULATE",
                "real_trading_supported": False,
                "completed_slots": {},
                "pending_slots": {
                    "2026-07-30:open": {
                        "order_ids": ["100", "101"],
                        "status": "submitted_awaiting_fill_confirmation",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    pending = reconcile_execution_state(
        state_path,
        [
            {"order_id": "100", "order_status": "FILLED_ALL"},
            {"order_id": "101", "order_status": "SUBMITTED"},
        ],
    )
    assert "2026-07-30:open" in pending["pending_slots"]
    completed = reconcile_execution_state(
        state_path,
        [
            {"order_id": "100", "order_status": "FILLED_ALL"},
            {"order_id": "101", "order_status": "FILLED_ALL"},
        ],
    )
    assert completed["pending_slots"] == {}
    assert completed["completed_slots"]["2026-07-30:open"]["status"] == "filled_confirmed"


def test_futu_execution_reconciles_legacy_zero_order_pending_as_completed(tmp_path):
    from src.api.futu_sim import reconcile_execution_state

    state_path = tmp_path / "execution_state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "environment": "SIMULATE",
                "real_trading_supported": False,
                "completed_slots": {},
                "pending_slots": {
                    "2026-07-30:recovery": {
                        "order_ids": [],
                        "status": "submitted_awaiting_fill_confirmation",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    state = reconcile_execution_state(state_path, [])

    assert state["pending_slots"] == {}
    assert state["completed_slots"]["2026-07-30:recovery"]["status"] == (
        "no_change_completed"
    )


def test_stage1_cycle_activates_only_after_tracked_positions_are_flat(tmp_path):
    from src.api.stage1_cycle import activate_flat_stage1_cycle

    cycle = {
        "version": 1,
        "cycle_id": "stage1_2026-08-03_v2",
        "status": "flatten_pending",
        "broker_reserve_cash": 900_000.0,
    }
    with pytest.raises(RuntimeError, match="not complete"):
        activate_flat_stage1_cycle(
            cycle,
            snapshot={"cash": 969_000.0, "positions": {"SOXL": 0, "SOXX": 55}},
            session_date="2026-08-03",
        )

    active = activate_flat_stage1_cycle(
        cycle,
        snapshot={"cash": 997_250.0, "positions": {"SOXL": 0, "SOXX": 0}},
        session_date="2026-08-03",
    )
    assert active["status"] == "active"
    assert active["initial_equity"] == pytest.approx(97_250.0)
    assert active["opening_positions"] == {"SOXL": 0, "SOXX": 0}


def test_stage1_ledger_rebuilds_confirmed_fills_and_live_drawdown(tmp_path):
    from src.api.stage1_ledger import reconcile_stage1_ledger

    orders = [
        {
            "order_id": "1",
            "code": "US.SOXL",
            "trd_side": "BUY",
            "order_status": "FILLED_ALL",
            "dealt_qty": 193,
            "dealt_avg_price": 139.515,
            "create_time": "2026-07-27 09:31:24",
            "updated_time": "2026-07-27 09:31:27",
            "remark": "stage1_open_2026-07-24",
        },
        {
            "order_id": "2",
            "code": "US.SOXX",
            "trd_side": "BUY",
            "order_status": "FILLED_ALL",
            "dealt_qty": 94,
            "dealt_avg_price": 531.24,
            "create_time": "2026-07-27 09:31:24",
            "updated_time": "2026-07-27 09:31:27",
            "remark": "stage1_open_2026-07-24",
        },
        {
            "order_id": "3",
            "code": "US.SOXL",
            "trd_side": "BUY",
            "order_status": "FILLED_ALL",
            "dealt_qty": 30,
            "dealt_avg_price": 120.59,
            "create_time": "2026-07-27 10:32:48",
            "updated_time": "2026-07-27 10:32:57",
            "remark": "stage1_recovery_2026-07-24",
        },
        {
            "order_id": "4",
            "code": "US.SOXX",
            "trd_side": "BUY",
            "order_status": "FILLED_ALL",
            "dealt_qty": 5,
            "dealt_avg_price": 506.69,
            "create_time": "2026-07-27 10:32:48",
            "updated_time": "2026-07-27 10:32:57",
            "remark": "stage1_recovery_2026-07-24",
        },
    ]
    state = reconcile_stage1_ledger(
        path=tmp_path / "paper_ledger.json",
        initial_equity=100_000.0,
        broker_reserve_cash=900_000.0,
        inception_date="2026-07-27",
        snapshot={"cash": 916_981.935, "positions": {"SOXL": 223, "SOXX": 99}},
        orders=orders,
        prices={"SOXL": 114.72, "SOXX": 504.53},
        as_of="2026-07-30T16:00:00-04:00",
        settled_marks=[
            {"date": "2026-07-29", "SOXL": 91.99, "SOXX": 465.0},
            {"date": "2026-07-30", "SOXL": 114.72, "SOXX": 504.53},
        ],
    )

    assert state["positions"] == {"SOXL": 223, "SOXX": 99}
    assert state["cash"] == pytest.approx(16_981.935)
    assert state["total_inferred_fees"] == pytest.approx(3.96)
    assert state["equity"] == pytest.approx(92_512.965)
    assert state["drawdown"] == pytest.approx(-0.07487035)
    assert state["settled_marks"][0]["equity"] == pytest.approx(83_530.705)

    previous_mark = dict(state["settled_marks"][0])
    later_orders = orders + [
        {
            "order_id": "5",
            "code": "US.SOXL",
            "trd_side": "SELL",
            "order_status": "FILLED_ALL",
            "dealt_qty": 223,
            "dealt_avg_price": 131.5,
            "create_time": "2026-07-31 09:31:30",
            "updated_time": "2026-07-31 09:31:35",
            "remark": "stage1_open_2026-07-30",
        }
    ]
    updated = reconcile_stage1_ledger(
        path=tmp_path / "paper_ledger.json",
        initial_equity=100_000.0,
        broker_reserve_cash=900_000.0,
        inception_date="2026-07-27",
        snapshot={"cash": 946_302.435, "positions": {"SOXL": 0, "SOXX": 99}},
        orders=later_orders,
        prices={"SOXL": 131.5, "SOXX": 529.0},
        as_of="2026-07-31T09:32:00-04:00",
        settled_marks=[
            {"date": "2026-07-29", "SOXL": 91.99, "SOXX": 465.0},
            {"date": "2026-07-30", "SOXL": 114.72, "SOXX": 504.53},
        ],
    )

    assert updated["settled_marks"][0] == previous_mark


def test_stage1_ledger_blocks_position_mismatch(tmp_path):
    from src.api.stage1_ledger import reconcile_stage1_ledger

    with pytest.raises(RuntimeError, match="position reconciliation failed"):
        reconcile_stage1_ledger(
            path=tmp_path / "paper_ledger.json",
            initial_equity=100_000.0,
            broker_reserve_cash=900_000.0,
            inception_date="2026-07-27",
            snapshot={"cash": 1_000_000.0, "positions": {"SOXL": 1, "SOXX": 0}},
            orders=[],
            prices={"SOXL": 100.0, "SOXX": 500.0},
            as_of="2026-07-30T16:00:00-04:00",
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


def test_stage1_bark_message_reports_submitted_sim_orders():
    from src.schedule.stage1_futu import build_stage1_bark_message

    message = build_stage1_bark_message(
        {
            "status": "submitted",
            "slot": "open",
            "real_broker_order": False,
            "target": {
                "signal_date": "2026-07-30",
                "weights": {"SOXL": 0.2, "SOXX": 0.5, "cash": 0.3},
            },
            "execution": {
                "status": "submitted",
                "environment": "SIMULATE",
                "orders": [
                    {"symbol": "SOXL", "side": "BUY", "quantity": 10, "status": "submitted"},
                    {"symbol": "SOXX", "side": "SELL", "quantity": 2, "status": "submitted"},
                ],
            },
        },
        now=datetime(2026, 7, 30, 21, 31, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert message["title"] == "SOXL 模拟盘 submitted"
    assert "SIMULATE" in message["body"]
    assert "BUY 10 SOXL" in message["body"]
    assert "SELL 2 SOXX" in message["body"]
    assert "目标 SOXL 20.0% / SOXX 50.0% / 现金 30.0%" in message["body"]
    assert "真实下单=false" in message["body"]
    assert message["group"] == "SOXL Stage1 模拟盘"


def test_stage1_bark_message_reports_close_only_live_ledger():
    from src.schedule.stage1_futu import build_stage1_bark_message

    message = build_stage1_bark_message(
        {
            "status": "dry_run",
            "slot": "open",
            "reason": "failed_position_risk_gates",
            "execution_mode": "close_only",
            "target": {
                "status": "ready",
                "signal_date": "2026-07-30",
                "weights": {"SOXL": 0.0, "SOXX": 0.3, "cash": 0.7},
            },
            "ledger": {
                "positions": {"SOXL": 223, "SOXX": 99},
                "equity": 92_512.965,
                "drawdown": -0.07487035,
            },
            "real_broker_order": False,
        }
    )

    assert "执行模式 close_only" in message["body"]
    assert "持仓 SOXL 223 / SOXX 99" in message["body"]
    assert "策略净值 $92,512.97 / 真实回撤 -7.49%" in message["body"]
    assert "推理状态 ready" in message["body"]


def test_stage1_daily_heartbeat_keeps_latest_actionable_run(tmp_path):
    from src.schedule.stage1_futu import run_stage1_futu_task

    runtime_dir = tmp_path / "runtime" / "soxl" / "stage1_futu_sim"
    runtime_dir.mkdir(parents=True)
    actionable = {
        "status": "blocked",
        "reason": "failed_position_risk_gates",
        "slot": "recovery",
        "real_broker_order": False,
        "recorded_at": "2026-07-30T22:31:00+08:00",
    }
    (runtime_dir / "latest_actionable_run.json").write_text(
        json.dumps(actionable), encoding="utf-8"
    )

    result = run_stage1_futu_task(
        root=tmp_path,
        slot="heartbeat",
        now=datetime(2026, 7, 31, 10, 5, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert result["status"] == "heartbeat"
    assert result["slot"] == "daily_heartbeat"
    assert result["latest_actionable_run"]["status"] == "blocked"
    assert result["latest_actionable_run"]["reason"] == "failed_position_risk_gates"
    assert result["real_broker_order"] is False


def test_stage1_bark_message_reports_daily_heartbeat_and_failure():
    from src.schedule.stage1_futu import build_stage1_bark_message

    heartbeat = build_stage1_bark_message(
        {
            "status": "heartbeat",
            "slot": "daily_heartbeat",
            "latest_actionable_run": {
                "status": "blocked",
                "slot": "recovery",
                "reason": "failed_position_risk_gates",
                "recorded_at": "2026-07-30T22:31:00+08:00",
            },
            "real_broker_order": False,
        },
        now=datetime(2026, 7, 31, 10, 5, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    failure = build_stage1_bark_message(
        {
            "status": "failed",
            "slot": "postclose",
            "reason": "blocked: Futu OpenD is not reachable",
            "real_broker_order": False,
        },
        now=datetime(2026, 7, 31, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert heartbeat["title"] == "SOXL 模拟盘每日心跳"
    assert "最近任务 recovery / blocked" in heartbeat["body"]
    assert "failed_position_risk_gates" in heartbeat["body"]
    assert failure["title"] == "SOXL 模拟盘 failed"
    assert "Futu OpenD is not reachable" in failure["body"]


def test_stage1_main_records_and_notifies_uncaught_failure(tmp_path, monkeypatch):
    from src.schedule import stage1_futu

    delivered = []
    monkeypatch.setattr(stage1_futu, "_resolve_root", lambda _root: tmp_path)
    monkeypatch.setattr(
        stage1_futu,
        "run_stage1_futu_task",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("provider failed")),
    )
    monkeypatch.setattr(
        stage1_futu,
        "deliver_stage1_bark",
        lambda result, **_kwargs: delivered.append(result) or {"status": "sent"},
    )

    exit_code = stage1_futu.main(["--slot", "postclose", "--notify"])
    written = json.loads(
        (tmp_path / "runtime" / "soxl" / "stage1_futu_sim" / "latest_actionable_run.json").read_text(
            encoding="utf-8"
        )
    )

    assert exit_code == 1
    assert written["status"] == "failed"
    assert written["reason"] == "provider failed"
    assert delivered[0]["status"] == "failed"


def test_stage1_launchagents_enable_event_push_and_daily_heartbeat():
    event_plist = (ROOT / "production" / "com.soxl.stage1.futu.sim.plist").read_text(
        encoding="utf-8"
    )
    daily_plist = (
        ROOT / "production" / "com.soxl.stage1.futu.sim.heartbeat.plist"
    ).read_text(encoding="utf-8")
    run_script = (ROOT / "production" / "run_stage1_futu_sim.sh").read_text(
        encoding="utf-8"
    )
    summary = json.loads(
        (ROOT / "production" / "stage1_futu_sim_v1" / "summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert "<string>--notify</string>" in event_plist
    assert "<string>--broker-reserve-cash</string>" in event_plist
    assert "<string>900000</string>" in event_plist
    assert "com.soxl.stage1.futu.sim.heartbeat" in daily_plist
    assert "<string>heartbeat</string>" in daily_plist
    assert "<key>Hour</key><integer>10</integer>" in daily_plist
    assert "<key>Minute</key><integer>5</integer>" in daily_plist
    assert "<key>Weekday</key>" not in daily_plist
    assert ".config/bark/remote.env" in run_script
    assert ".config/bark/android-current.env" in run_script
    assert summary["execution"]["notifications"]["daily_heartbeat"] == (
        "10:05 Asia/Shanghai every calendar day"
    )

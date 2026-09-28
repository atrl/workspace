from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.api.futu_sim import FutuSimulatedBroker, reconcile_execution_state
from src.api.stage1_ledger import reconcile_stage1_ledger


@pytest.fixture
def execution(tmp_path, monkeypatch):
    broker = FutuSimulatedBroker()
    broker._futu = SimpleNamespace(
        RET_OK=0,
        TrdEnv=SimpleNamespace(SIMULATE="SIMULATE"),
        TrdSide=SimpleNamespace(BUY="BUY", SELL="SELL"),
        OrderType=SimpleNamespace(MARKET="MARKET"),
    )
    broker._account_id = 123
    monkeypatch.setattr(broker, "us_market_state", lambda: "MORNING")
    monkeypatch.setattr(broker, "account_snapshot", lambda: {"positions": {"SOXL": 200, "SOXX": 0}})
    kwargs = {
        "weights": {"SOXL": 0.1, "SOXX": 0.4, "cash": 0.5},
        "prices": {"SOXL": 100.0, "SOXX": 200.0},
        "portfolio_equity": 100_000.0,
        "signal_date": "2026-09-25",
        "slot": "open",
        "state_path": tmp_path / "state.json",
        "dry_run": False,
        "now": datetime(2026, 9, 28, 9, 31, tzinfo=ZoneInfo("America/New_York")),
        "remark_prefix": "stage1_test",
    }
    return broker, kwargs


def _history(order_id, code, side, quantity, status, filled):
    return {
        "order_id": order_id,
        "code": code,
        "trd_side": side,
        "qty": quantity,
        "order_status": status,
        "dealt_qty": filled,
        "dealt_avg_price": 100.0 if code == "US.SOXL" else 200.0,
        "remark": "stage1_test_open_2026-09-25",
        "create_time": "2026-09-28 09:31:00",
    }


def test_plan_and_each_accepted_leg_are_durable_before_next_submission(execution):
    broker, kwargs = execution
    seen = []

    def place_order(**request):
        state = json.loads(kwargs["state_path"].read_text())
        pending = state["pending_slots"]["2026-09-25:open"]
        seen.append(request)
        assert len(pending["orders"]) == 2
        assert pending["orders"][len(seen) - 1]["status"] == "submitting"
        if len(seen) == 2:
            assert pending["order_ids"] == ["first"]
            return -1, "transport failed after broker acceptance"
        return 0, pd.DataFrame([{"order_id": "first"}])

    broker._trade_ctx = SimpleNamespace(place_order=place_order)
    with pytest.raises(RuntimeError, match="submission failed"):
        broker.rebalance(**kwargs)
    state = json.loads(kwargs["state_path"].read_text())
    pending = state["pending_slots"]["2026-09-25:open"]
    assert pending["order_ids"] == ["first"]
    assert pending["orders"][1]["status"] == "submission_unknown"
    assert broker.rebalance(**kwargs)["status"] == "duplicate_or_pending_skipped"
    with pytest.raises(RuntimeError, match="unresolved Futu execution"):
        broker.rebalance(**(kwargs | {"slot": "recovery"}))
    assert len(seen) == 2

    rows = [
        _history("first", "US.SOXL", "SELL", 100, "FILLED_ALL", 100),
        _history("second", "US.SOXX", "BUY", 200, "FAILED", 0),
    ]
    reconciled = reconcile_execution_state(kwargs["state_path"], rows)
    assert reconciled["pending_slots"] == {}
    assert reconciled["completed_slots"] == {}
    failed = reconciled["failed_slots"]["2026-09-25:open"]
    assert failed["status"] == "partial_terminal"
    assert failed["order_ids"] == ["first", "second"]
    assert broker.rebalance(**kwargs)["status"] == "duplicate_or_pending_skipped"
    assert len(seen) == 2


@pytest.mark.parametrize("failure", ["exception", "missing_id"])
def test_unknown_submission_never_becomes_zero_order_completed(execution, failure):
    broker, kwargs = execution
    calls = []

    def place_order(**request):
        calls.append(request)
        if failure == "exception":
            raise TimeoutError("acceptance unknown")
        return 0, pd.DataFrame([{"order_id": ""}])

    broker._trade_ctx = SimpleNamespace(place_order=place_order)
    with pytest.raises((RuntimeError, TimeoutError)):
        broker.rebalance(**kwargs)
    state = reconcile_execution_state(kwargs["state_path"], [])
    assert state["completed_slots"] == {}
    assert state["pending_slots"]["2026-09-25:open"]["orders"][1]["status"] == "not_submitted"
    assert broker.rebalance(**kwargs)["status"] == "duplicate_or_pending_skipped"
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("status", "filled", "bucket", "outcome"),
    [
        ("FILLED_ALL", 100, "completed_slots", "filled_confirmed"),
        ("CANCELLED_PART", 40, "failed_slots", "partial_terminal"),
        ("CANCELLED_ALL", 0, "failed_slots", "failed_terminal"),
        ("FAILED", 0, "failed_slots", "failed_terminal"),
        ("DELETED", 0, "failed_slots", "failed_terminal"),
        ("FILLED_PART", 40, "pending_slots", "submitted_awaiting_fill_confirmation"),
        ("UNKNOWN", 0, "pending_slots", "submitted_awaiting_fill_confirmation"),
        ("DISABLED", 0, "pending_slots", "submitted_awaiting_fill_confirmation"),
    ],
)
def test_terminal_and_partial_fill_reconciliation(execution, status, filled, bucket, outcome):
    broker, kwargs = execution
    kwargs["weights"] = {"SOXL": 0.1, "SOXX": 0.0, "cash": 0.9}
    broker._trade_ctx = SimpleNamespace(place_order=lambda **_: (0, pd.DataFrame([{"order_id": "one"}])))
    broker.rebalance(**kwargs)
    state = reconcile_execution_state(
        kwargs["state_path"], [_history("one", "US.SOXL", "SELL", 100, status, filled)]
    )
    assert state[bucket]["2026-09-25:open"]["status"] == outcome
    if bucket != "completed_slots":
        assert state["completed_slots"] == {}


def test_ambiguous_broker_match_remains_pending(execution):
    broker, kwargs = execution
    broker._trade_ctx = SimpleNamespace(place_order=lambda **_: (-1, "timeout"))
    with pytest.raises(RuntimeError):
        broker.rebalance(**kwargs)
    rows = [_history(order_id, "US.SOXL", "SELL", 100, "FILLED_ALL", 100) for order_id in ("a", "b")]
    state = reconcile_execution_state(kwargs["state_path"], rows)
    assert state["pending_slots"]["2026-09-25:open"]["orders"][0]["order_id"] == ""
    assert state["completed_slots"] == {}


def test_legacy_blank_order_id_is_not_a_zero_order_slot(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "environment": "SIMULATE", "real_trading_supported": False,
        "pending_slots": {"slot": {"order_ids": [""], "status": "submitted_awaiting_fill_confirmation"}},
    }))
    state = reconcile_execution_state(path, [])
    assert "slot" in state["pending_slots"]
    assert state["completed_slots"] == {}


def test_terminal_partial_fills_are_included_in_reconciled_ledger(tmp_path):
    orders = [
        _history("partial", "US.SOXL", "BUY", 100, "CANCELLED_PART", 40),
        _history("rejected", "US.SOXX", "BUY", 100, "FAILED", 0),
        _history("cancelled", "US.SOXX", "BUY", 100, "CANCELLED_ALL", 0),
    ]
    state = reconcile_stage1_ledger(
        path=tmp_path / "ledger.json", initial_equity=100_000.0, broker_reserve_cash=900_000.0,
        inception_date="2026-09-28", snapshot={"cash": 996_000.0, "positions": {"SOXL": 40, "SOXX": 0}},
        orders=orders, prices={"SOXL": 101.0, "SOXX": 200.0}, as_of="2026-09-28T10:00:00-04:00",
    )
    assert state["positions"] == {"SOXL": 40, "SOXX": 0}
    assert state["cash"] == 96_000.0
    assert state["equity"] == 100_040.0
    assert len(state["fills"]) == 1
    assert state["fills"][0]["status"] == "CANCELLED_PART"


@pytest.mark.parametrize(("status", "filled"), [("FILLED_PART", 40), ("UNKNOWN", 0), ("FAILED", 40), ("FILLED_ALL", 40)])
def test_unresolved_or_inconsistent_fill_evidence_blocks_ledger(tmp_path, status, filled):
    with pytest.raises(RuntimeError, match="blocked:"):
        reconcile_stage1_ledger(
            path=tmp_path / "ledger.json", initial_equity=100_000.0, broker_reserve_cash=900_000.0,
            inception_date="2026-09-28", snapshot={"cash": 996_000.0, "positions": {"SOXL": 40, "SOXX": 0}},
            orders=[_history("one", "US.SOXL", "BUY", 100, status, filled)],
            prices={"SOXL": 101.0, "SOXX": 200.0}, as_of="2026-09-28T10:00:00-04:00",
        )

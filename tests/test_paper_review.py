from __future__ import annotations

from pathlib import Path

from src.soxl_quant.app.paper_review import public_paper_review_from_summary


def _assert_no_sensitive_public_fields(payload: object) -> None:
    forbidden_keys = {
        "account_value",
        "cash",
        "shares",
        "market_value",
        "current_position",
        "resulting_position",
        "actual_position",
        "position_gap",
        "filled_qty",
        "quantity",
        "broker_order_id",
        "reconcile",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                assert str(key) not in forbidden_keys
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)


def test_public_paper_review_uses_order_entry_fields_not_account_holdings():
    review = public_paper_review_from_summary(
        {
            "date": "2026-05-22",
            "profile": "v19reg_v22_budget_usage_bull_guard_v1",
            "broker": "futu",
            "dry_run": False,
            "executed": True,
            "signal": {
                "symbol": "SOXL",
                "target_position": 0.85,
                "current_position": 0.12,
                "price": 190.56,
            },
            "risk_gate": {
                "approved": True,
                "capped_target_position": 0.35,
            },
            "execution_result": {
                "submitted": True,
                "status": "filled",
                "target_position": 0.35,
                "resulting_position": 0.34,
                "filled_qty": 100,
                "broker_order_id": "secret-order-id",
            },
            "account": {
                "cash": 1234.56,
                "shares": 100,
                "market_value": 19056.0,
                "account_value": 20290.56,
                "position": 0.34,
            },
            "reconcile": {
                "actual_position": 0.34,
                "position_gap": -0.01,
                "action_required": "none",
            },
        },
        summary_path=Path("runtime/soxl/paper_trading/2026-05-22/summary.json"),
    )

    assert review == {
        "date": "2026-05-22",
        "profile": "v19reg_v22_budget_usage_bull_guard_v1",
        "broker": "futu",
        "dry_run": False,
        "entry_status": "filled",
        "entry_side": "BUY",
        "entry_symbol": "SOXL",
        "entry_price": 190.56,
        "entry_target_position": 0.35,
        "entry_signal_target_position": 0.85,
        "risk_gate_approved": True,
        "submitted": True,
        "summary_path": "runtime/soxl/paper_trading/2026-05-22/summary.json",
    }
    _assert_no_sensitive_public_fields(review)


def test_public_paper_review_handles_blocked_signal_without_holding_fields():
    review = public_paper_review_from_summary(
        {
            "date": "2026-05-22",
            "profile": "blocked_profile",
            "broker": "paper",
            "dry_run": True,
            "executed": False,
            "signal": {
                "symbol": "SOXL",
                "target_position": 0.2,
                "current_position": 0.6,
                "price": 190.56,
            },
            "risk_gate": {
                "approved": False,
                "capped_target_position": 0.2,
                "blocking_reasons": ["open orders must be reconciled before another rebalance"],
            },
        }
    )

    assert review["entry_status"] == "blocked"
    assert review["entry_side"] == "BLOCKED"
    assert review["entry_target_position"] == 0.2
    assert review["risk_gate_approved"] is False
    assert review["blocking_reasons"] == ["open orders must be reconciled before another rebalance"]
    _assert_no_sensitive_public_fields(review)

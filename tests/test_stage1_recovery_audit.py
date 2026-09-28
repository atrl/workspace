import copy

import pytest

from src.analysis.stage1_recovery import reconcile_observed_fills


def evidence():
    orders = [
        {"order_id": "1", "code": "US.SOXL", "create_time": "2026-08-03 09:31:00",
         "order_status": "FILLED_ALL", "trd_side": "BUY", "qty": 2,
         "dealt_qty": 2, "dealt_avg_price": 100, "remark": "stage1_v2_open"},
        {"order_id": "2", "code": "US.SOXX", "create_time": "2026-08-03 09:31:00",
         "order_status": "FILLED_ALL", "trd_side": "BUY", "qty": 1,
         "dealt_qty": 1, "dealt_avg_price": 500, "remark": "stage1_v2_open"},
    ]
    positions = [{"code": "US.SOXL", "qty": 2, "market_val": 220},
                 {"code": "US.SOXX", "qty": 1, "market_val": 550}]
    history = {"initial_equity": 1000, "inception_date": "2026-08-03",
               "remark_prefix": "stage1_v2_", "fills": [
                   {"order_id": "1", "symbol": "SOXL", "side": "BUY", "quantity": 2, "price": 100}]}
    return orders, positions, {"cash": 303, "total_assets": 1073}, history


def test_positive_cash_residual_remains_unclassified_and_history_matches():
    fills, result = reconcile_observed_fills(*evidence())
    assert fills.cash_flow.sum() == -700
    assert result["trade_only_cash"] == 300
    assert result["unclassified_net_cash_adjustment"] == 3
    assert result["historical_fill_count_matched"] == 1
    assert "fee" not in {key.lower() for key in result}


@pytest.mark.parametrize("case", ["partial_fill", "missing_fill", "changed_history", "bad_equity"])
def test_incomplete_or_inconsistent_broker_evidence_blocks(case):
    orders, positions, balance, history = copy.deepcopy(evidence())
    if case == "partial_fill":
        orders[0]["order_status"] = "FILLED_PART"
    elif case == "missing_fill":
        orders.pop(0)
    elif case == "changed_history":
        history["fills"][0]["price"] = 101
    else:
        balance["total_assets"] = 1074
    with pytest.raises(ValueError):
        reconcile_observed_fills(orders, positions, balance, history)

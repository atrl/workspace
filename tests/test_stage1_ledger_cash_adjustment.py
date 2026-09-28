import copy
import hashlib
import json

import pytest

from src.api.stage1_ledger import reconcile_stage1_ledger


def confirmed_cycle(tmp_path):
    order = {"order_id": "1", "code": "US.SOXL", "trd_side": "BUY", "order_status": "FILLED_ALL",
             "dealt_qty": 2, "dealt_avg_price": 10, "create_time": "2026-08-03 09:31:00",
             "remark": "stage1_v2_open_2026-07-31"}
    sources = {}
    for name, payload in {"balance": [{"cash": 81.5}], "positions": [{"code": "US.SOXL", "qty": 2}],
                          "orders": [order]}.items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(payload))
        sources[name] = {"source_ref": str(p), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    confirmation = {
        "schema": "stage1_cycle_cash_adjustment_confirmation_v1",
        "classification": "unallocated_net_investment_cash_flow", "cycle_id": "stage1_2026-08-03_v2",
        "initial_equity": 100.0, "broker_reserve_cash": 0.0, "inception_date": "2026-08-03", "remark_prefix": "stage1_v2_",
        "snapshot_observed_at": "2026-09-28T04:15:00-04:00", "anchor_net_cash_adjustment": 1.5, "sources": sources,
        "user_confirmation": {"text": "没有加减资金，也没有再次重置", "external_cash_flows": False, "account_reset": False,
                              "scope_start": "2026-08-03", "scope_end_snapshot": "2026-09-28T04:15:00-04:00"},
    }
    return dict(path=tmp_path / "ledger.json", initial_equity=100, broker_reserve_cash=0, inception_date="2026-08-03",
                cycle_id=confirmation["cycle_id"], remark_prefix="stage1_v2_", snapshot={"cash": 81.5, "positions": {"SOXL": 2, "SOXX": 0}},
                orders=[order], prices={"SOXL": 11, "SOXX": 20}, as_of="2026-09-28T09:00:00-04:00",
                cash_adjustment_confirmation=confirmation)


def test_confirmed_net_cash_residual_is_not_dividend_or_known_fee(tmp_path):
    args = confirmed_cycle(tmp_path)
    args["settled_marks"] = [{"date": "2026-09-25", "SOXL": 11, "SOXX": 20},
                             {"date": "2026-09-28", "SOXL": 11, "SOXX": 20}]
    ledger = reconcile_stage1_ledger(**args)
    assert ledger["cash"] == 81.5
    assert ledger["equity"] == 103.5
    assert ledger["total_inferred_fees"] is None
    assert ledger["total_confirmed_cash_income"] == 0
    assert ledger["net_investment_cash_adjustment"]["anchor_net_cash_adjustment"] == 1.5
    assert ledger["settled_marks"][0]["cash"] == 80  # Do not invent a historical receipt date.
    assert ledger["settled_marks"][1]["cash"] == 81.5  # Known cash anchor is available by this date.
    repeated = reconcile_stage1_ledger(**args)
    assert repeated["net_investment_cash_adjustment"] == ledger["net_investment_cash_adjustment"]


def test_appended_fills_allow_inferred_deductions_but_never_reuse_confirmation_for_new_credit(tmp_path):
    args = confirmed_cycle(tmp_path)
    reconcile_stage1_ledger(**args)
    args["orders"].append({"order_id": "2", "code": "US.SOXL", "trd_side": "BUY", "order_status": "FILLED_ALL",
                            "dealt_qty": 1, "dealt_avg_price": 10, "create_time": "2026-09-28 09:31:00",
                            "remark": "stage1_v2_open_2026-09-25"})
    args.update(snapshot={"cash": 70.5, "positions": {"SOXL": 3, "SOXX": 0}}, as_of="2026-09-28T09:32:00-04:00")
    ledger = reconcile_stage1_ledger(**args)
    assert ledger["total_inferred_fees"] is None
    assert ledger["net_investment_cash_adjustment"]["post_confirmation_inferred_cash_deductions"] == 1
    args["snapshot"]["cash"] = 71
    with pytest.raises(RuntimeError, match="new unexplained cash credit"):
        reconcile_stage1_ledger(**args)
    args["snapshot"]["cash"] = 70
    with pytest.raises(RuntimeError, match="cash changed without new confirmed fills"):
        reconcile_stage1_ledger(**args)


@pytest.mark.parametrize("issue", ["other_cycle", "no_confirmation", "reset", "tampered_source", "changed_fill", "extra_cash", "wrong_amount"])
def test_unscoped_or_changed_evidence_blocks(tmp_path, issue):
    args = confirmed_cycle(tmp_path)
    c = args["cash_adjustment_confirmation"]
    if issue == "other_cycle":
        args["cycle_id"] = "other_cycle"
    elif issue == "no_confirmation":
        c["user_confirmation"]["text"] = ""
    elif issue == "reset":
        c["user_confirmation"]["account_reset"] = True
    elif issue == "tampered_source":
        c["sources"]["balance"]["source_sha256"] = "0" * 64
    elif issue == "changed_fill":
        args["orders"][0]["dealt_avg_price"] = 11
    elif issue == "extra_cash":
        args["snapshot"]["cash"] = 82
    else:
        c["anchor_net_cash_adjustment"] = 2
    with pytest.raises(RuntimeError, match="blocked"):
        reconcile_stage1_ledger(**args)


def test_existing_confirmation_cannot_be_silently_replaced(tmp_path):
    args = confirmed_cycle(tmp_path)
    reconcile_stage1_ledger(**args)
    args["cash_adjustment_confirmation"] = copy.deepcopy(args["cash_adjustment_confirmation"])
    args["cash_adjustment_confirmation"]["user_confirmation"]["text"] = "new text"
    with pytest.raises(RuntimeError, match="immutable cash adjustment confirmation changed"):
        reconcile_stage1_ledger(**args)

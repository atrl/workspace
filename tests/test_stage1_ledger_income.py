import copy
import hashlib
import json

import pytest

from src.api.stage1_ledger import reconcile_stage1_ledger


def source(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return {"source_ref": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inputs(tmp_path):
    event = {"symbol": "SOXL", "ex_div_date": "2026-09-22", "per_share_cash": 1.0,
             "currency": "USD", "source_api": "futu.get_rehab",
             **source(tmp_path, "rehab.json", [{"ex_div_date": "2026-09-22", "per_cash_div": 1.0}])}
    kwargs = dict(path=tmp_path / "ledger.json", initial_equity=100, broker_reserve_cash=0,
                  inception_date="2026-08-03", snapshot={"cash": 80, "positions": {"SOXL": 2, "SOXX": 0}},
                  orders=[{"order_id": "1", "code": "US.SOXL", "trd_side": "BUY", "order_status": "FILLED_ALL",
                           "dealt_qty": 2, "dealt_avg_price": 10, "create_time": "2026-08-03 09:31:00",
                           "remark": "stage1_open_2026-07-31"}],
                  prices={"SOXL": 11, "SOXX": 20}, as_of="2026-09-28T09:00:00-04:00")
    credit = {"kind": "broker_statement", "category": "DIVIDEND", "currency": "USD",
              "transaction_id": "cash-dividend-1", "credited_at": "2026-09-23", "credited_amount": 2.0,
              **source(tmp_path, "statement.json", {"transaction_id": "cash-dividend-1", "credited_at": "2026-09-23",
                                                   "amount": 2.0, "type": "DIVIDEND", "symbol": "SOXL"})}
    return kwargs, event, credit


def test_entitlement_alone_never_relabels_extra_cash_as_dividend(tmp_path):
    kwargs, event, _ = inputs(tmp_path)
    state = reconcile_stage1_ledger(**kwargs, income_events=[event])
    assert state["total_expected_gross_income"] == 2
    assert state["total_confirmed_cash_income"] == 0
    kwargs["snapshot"]["cash"] = 81
    with pytest.raises(RuntimeError, match="unowned cash flows"):
        reconcile_stage1_ledger(**kwargs, income_events=[event])


def test_confirmed_income_is_deduplicated_retained_and_applied_on_credit_date(tmp_path):
    kwargs, event, credit = inputs(tmp_path)
    event["credit_evidence"] = credit
    kwargs["snapshot"]["cash"] = 81.5
    kwargs["settled_marks"] = [{"date": "2026-09-22", "SOXL": 11, "SOXX": 20},
                               {"date": "2026-09-23", "SOXL": 11, "SOXX": 20}]
    state = reconcile_stage1_ledger(**kwargs, income_events=[event, copy.deepcopy(event)])
    assert state["total_confirmed_cash_income"] == 2
    assert state["total_inferred_fees"] == pytest.approx(0.5)
    assert len(state["income_events"]) == 1
    assert state["settled_marks"][0]["cash"] == pytest.approx(79.5)
    assert state["settled_marks"][1]["cash"] == pytest.approx(81.5)
    # Future runs need not rediscover an already accepted immutable event.
    repeated = reconcile_stage1_ledger(**kwargs)
    assert repeated["income_events"] == state["income_events"]
    assert repeated["settled_marks"] == state["settled_marks"]


def test_unexplained_extra_cash_still_blocks_with_confirmed_dividend(tmp_path):
    kwargs, event, credit = inputs(tmp_path)
    event["credit_evidence"] = credit
    kwargs["snapshot"]["cash"] = 83
    with pytest.raises(RuntimeError, match="unowned cash flows"):
        reconcile_stage1_ledger(**kwargs, income_events=[event])


@pytest.mark.parametrize("bad", ["hash", "future", "over_credit", "same_quote", "changed_entitlement"])
def test_income_evidence_must_be_immutable_and_match_owned_entitlement(tmp_path, bad):
    kwargs, event, credit = inputs(tmp_path)
    event["credit_evidence"] = credit
    if bad == "hash":
        event["source_sha256"] = "0" * 64
    elif bad == "future":
        credit["credited_at"] = "2026-09-28T10:00:00-04:00"
    elif bad == "over_credit":
        credit["credited_amount"] = 3
    elif bad == "same_quote":
        credit.update(source_ref=event["source_ref"], source_sha256=event["source_sha256"])
    else:
        event["per_share_cash"] = 2
    with pytest.raises(RuntimeError, match="blocked"):
        reconcile_stage1_ledger(**kwargs, income_events=[event])


def test_income_uses_ex_date_holdings_and_never_today_holdings(tmp_path):
    kwargs, event, credit = inputs(tmp_path)
    kwargs["orders"].append({"order_id": "2", "code": "US.SOXL", "trd_side": "BUY", "order_status": "FILLED_ALL",
                              "dealt_qty": 3, "dealt_avg_price": 10, "create_time": "2026-09-22 09:31:00",
                              "remark": "stage1_open_2026-09-21"})
    kwargs["snapshot"] = {"cash": 52, "positions": {"SOXL": 5, "SOXX": 0}}
    event["credit_evidence"] = credit
    state = reconcile_stage1_ledger(**kwargs, income_events=[event])
    assert state["income_events"][0]["eligible_quantity"] == 2
    assert state["total_confirmed_cash_income"] == 2


def test_late_credit_does_not_silently_rewrite_frozen_marks(tmp_path):
    kwargs, event, credit = inputs(tmp_path)
    kwargs["settled_marks"] = [{"date": "2026-09-24", "SOXL": 11, "SOXX": 20}]
    reconcile_stage1_ledger(**kwargs, income_events=[event])
    event["credit_evidence"] = credit
    kwargs["snapshot"]["cash"] = 82
    with pytest.raises(RuntimeError, match="explicit settled-mark repair"):
        reconcile_stage1_ledger(**kwargs, income_events=[event])

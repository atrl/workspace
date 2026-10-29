from copy import deepcopy

import pytest

from src.core.stage1_cap_free_admission import (
    CLOSED_COMPARATORS, validate_cap_free_receiver_plan,
    validate_cap_free_result_claim,
)


def _plan():
    return {
        "schema_version": "stage1_cap_free_receiver_v1",
        "parent": "dated_f696_b472_own_state_2018_2022",
        "clock": {"kind": "completed_minute_t_plus_1_decision_t_plus_2_fill",
                  "availability_status": "historical_t_plus_1_assumption"},
        "target": {"kind": "numeric_fee_net_own_state_relative_soxx_value",
                   "includes_tail_path": True, "maturity_and_overlap_frozen": True},
        "policy": {"output": "continuous_soxl_soxx_cash_simplex",
                   "uses_frozen_b472_action_model": False,
                   "train_beta_cap": "none", "inference_beta_cap": "none"},
        "budget": {"maximum_model_fits": 1},
        "controls": ["frozen_stage1", "soxx", "matched_state", "same_risk_simple"],
        "closed_route_differences": {key: "New receiver requires actual minute state and"
                                          " matched Stage1 action value" for key in CLOSED_COMPARATORS},
        "distinct_decision_time_mechanism": "Frozen source-known conditional minute state",
        "historical_2023_2026_is_independent": False,
    }


@pytest.mark.parametrize("field,value", [
    ("target", {"kind": "cash_origin_argmax_winner", "includes_tail_path": False,
                "maturity_and_overlap_frozen": True}),
    ("policy", {"output": "continuous_soxl_soxx_cash_simplex",
                "uses_frozen_b472_action_model": True,
                "train_beta_cap": "legacy", "inference_beta_cap": "none"}),
    ("budget", {"maximum_model_fits": 0}),
    ("closed_route_differences", {}),
])
def test_rejects_previous_stage1_category_errors(field, value):
    plan = _plan()
    plan[field] = value
    with pytest.raises(ValueError, match="blocked:"):
        validate_cap_free_receiver_plan(plan)


def test_receipt_cannot_claim_training_from_cap3_or_account_from_fit():
    plan = _plan()
    validate_cap_free_receiver_plan(plan)
    zero_fit = {"claim": "trained_receiver", "actual_model_fits": 0,
                "model_recorder_id": None, "training_domain_parity_passed": False}
    with pytest.raises(ValueError, match="fitted, same-domain"):
        validate_cap_free_result_claim(plan, zero_fit)
    trained = {"claim": "trained_receiver", "actual_model_fits": 1,
               "model_recorder_id": "verified-recorder", "training_domain_parity_passed": True}
    validate_cap_free_result_claim(plan, trained)
    account = deepcopy(trained)
    account["claim"] = "funded_0bp_pass"
    with pytest.raises(ValueError, match="complete SOXX gates"):
        validate_cap_free_result_claim(plan, account)

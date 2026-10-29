"""Preflight and claim checks for a new Stage1 cap-free receiver.

This catches the specific category mistakes exposed by the frozen Stage1 cards.
It does not certify that a proposed feature is novel or economically useful.
"""
from __future__ import annotations

from collections.abc import Mapping


CLOSED_COMPARATORS = frozenset({
    "stable_excess_integration_20261002",
    "benchmark_risk_excess_20261002",
    "constrained_net_excess_20261003",
    "benchmark_anchor_utility_20261002",
    "independent_tree_allocation_20261027",
    "stage1_f696_no_beta_cap_ablation_20261029",
})


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"blocked: {reason}")


def validate_cap_free_receiver_plan(plan: Mapping) -> None:
    """Reject a repeat ablation, proxy label, clock mismatch or missing comparison."""
    _require(plan.get("schema_version") == "stage1_cap_free_receiver_v1",
             "cap-free receiver contract version is absent")
    _require(plan.get("parent") == "dated_f696_b472_own_state_2018_2022",
             "training parent is not the dated matched Stage1 account")
    clock = plan.get("clock", {})
    _require(clock.get("kind") == "completed_minute_t_plus_1_decision_t_plus_2_fill"
             and clock.get("availability_status") in
             {"historical_t_plus_1_assumption", "verified_receipt"},
             "minute source has no explicit causal availability contract")
    target = plan.get("target", {})
    _require(target.get("kind") == "numeric_fee_net_own_state_relative_soxx_value"
             and target.get("includes_tail_path") is True
             and target.get("maturity_and_overlap_frozen") is True,
             "training label is a proxy rather than mature own-state net value")
    policy = plan.get("policy", {})
    _require(policy.get("output") == "continuous_soxl_soxx_cash_simplex"
             and policy.get("uses_frozen_b472_action_model") is False,
             "candidate is an old action model or hard-argmax winner")
    _require(policy.get("train_beta_cap") == policy.get("inference_beta_cap") == "none",
             "training and inference beta-cap domains differ")
    _require(plan.get("budget", {}).get("maximum_model_fits", 0) >= 1,
             "a zero-fit ablation cannot be called a retrained receiver")
    controls = set(plan.get("controls", ()))
    _require({"frozen_stage1", "soxx", "matched_state", "same_risk_simple"} <= controls,
             "parent, benchmark, state or simple-risk control is missing")
    reviewed = plan.get("closed_route_differences", {})
    _require(isinstance(reviewed, Mapping) and CLOSED_COMPARATORS <= reviewed.keys()
             and all(isinstance(reviewed[key], str) and reviewed[key].strip()
                     for key in CLOSED_COMPARATORS),
             "closed cap-free and one-step routes were not reviewed")
    _require(bool(plan.get("distinct_decision_time_mechanism")),
             "no distinct decision-time information mechanism was frozen")
    _require(plan.get("historical_2023_2026_is_independent") is False,
             "reused market history cannot be declared independent")


def validate_cap_free_result_claim(plan: Mapping, receipt: Mapping) -> None:
    """Require actual fits and gated account evidence before stronger claims."""
    validate_cap_free_receiver_plan(plan)
    claim = receipt.get("claim")
    _require(claim in {"source_only", "trained_receiver", "funded_0bp_pass",
                       "historical_research_passed"}, "unknown evidence claim")
    if claim == "source_only":
        return
    _require(receipt.get("actual_model_fits", 0) >= 1
             and bool(receipt.get("model_recorder_id"))
             and receipt.get("training_domain_parity_passed") is True,
             "trained claim has no fitted, same-domain model evidence")
    if claim == "trained_receiver":
        return
    _require(receipt.get("information_gate_passed") is True
             and receipt.get("funded_0bp_qlib_account") is True
             and receipt.get("all_soxx_0bp_gates_passed") is True,
             "0bp account claim lacks information or complete SOXX gates")
    if claim == "funded_0bp_pass":
        return
    _require(receipt.get("all_soxx_2bp_gates_passed") is True
             and receipt.get("all_soxx_5bp_gates_passed") is True,
             "historical pass lacks the same frozen model at 2/5bp")

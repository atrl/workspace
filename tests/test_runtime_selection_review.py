from __future__ import annotations

from scripts.review_runtime_selection import (
    build_runtime_selection_review,
    exit_code_for_review,
    runtime_selection_context,
)


def _model_summary(*, eligible: bool, fixed_split: bool, walk_forward_evaluated: bool) -> dict:
    return {
        "profile": "balanced_rule_residual_v22",
        "best_score": 91.94,
        "promotion": {
            "eligible": eligible,
            "fixed_split": {"eligible": fixed_split, "checks": {}},
            "walk_forward": {"eligible": eligible and walk_forward_evaluated, "checks": {}},
            "walk_forward_evaluated": walk_forward_evaluated,
        },
    }


def _overlay_review() -> dict:
    return {
        "variant": "selective_release_recovery_post_release_cap_v1",
        "decision": "promotion_review_candidate",
        "checks": {
            "runtime_profile_bound": {
                "passed": True,
                "detail": "profile binds the overlay to runtime execution config",
            }
        },
    }


def test_runtime_selection_review_blocks_when_model_or_profile_gate_is_missing():
    review = build_runtime_selection_review(
        profile_name="balanced_rule_residual_v22",
        model_summary=_model_summary(
            eligible=False,
            fixed_split=True,
            walk_forward_evaluated=False,
        ),
        overlay_review=_overlay_review(),
        profile_overlay_auto_select_allowed=False,
        current_auto_select_profile="balanced_rule_residual_v19",
    )

    assert review["decision"] == "blocked"
    assert review["runtime_auto_select_allowed"] is False
    assert review["checks"]["model_fixed_split_eligible"]["passed"] is True
    assert review["checks"]["model_walk_forward_evaluated"]["passed"] is False
    assert review["checks"]["profile_overlay_auto_select_allowed"]["passed"] is False
    assert "model walk-forward promotion gate not evaluated" in review["blocking_reasons"]
    assert "profile overlay auto_select_allowed=false" in review["blocking_reasons"]


def test_runtime_selection_review_can_approve_only_when_all_gates_pass():
    review = build_runtime_selection_review(
        profile_name="balanced_rule_residual_v22",
        model_summary=_model_summary(
            eligible=True,
            fixed_split=True,
            walk_forward_evaluated=True,
        ),
        overlay_review=_overlay_review(),
        profile_overlay_auto_select_allowed=True,
        current_auto_select_profile="balanced_rule_residual_v22",
    )

    assert review["decision"] == "approved"
    assert review["runtime_auto_select_allowed"] is True
    assert review["blocking_reasons"] == []


def test_runtime_selection_review_script_succeeds_when_review_blocks_selection():
    assert exit_code_for_review({"decision": "blocked"}) == 0


def test_runtime_selection_context_uses_explicit_family_for_auto_select():
    context = runtime_selection_context("balanced_rule_residual_v22", family="balanced")

    assert context["profile_overlay_auto_select_allowed"] is False
    assert context["current_auto_select_profile"] == "balanced_rule_residual_v19"

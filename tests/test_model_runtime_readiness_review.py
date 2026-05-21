from __future__ import annotations

from pathlib import Path

from scripts.review_model_runtime_readiness import (
    build_model_runtime_readiness_review,
    exit_code_for_review,
    relative_output_label,
)


def _composer_review(*, approved: bool = True) -> dict:
    return {
        "profile": "v19reg_v22_budget_usage_bull_guard_v1",
        "decision": "approved" if approved else "blocked",
        "runtime_model_gate_allowed": approved,
        "components": {
            "slow_profile": "balanced_rule_residual_v19_reg",
            "fast_profile": "balanced_rule_residual_v22",
            "slow_model": "soxl_balanced_rule_residual_v19_reg_actor.pt",
            "fast_model": "soxl_balanced_rule_residual_v22_actor.pt",
        },
        "checks": {
            "required_windows_present": {"passed": True},
            "bull_guard_enabled": {"passed": True},
            "all_windows_return_beats_fast": {"passed": True},
            "all_windows_maxdd_not_worse_than_fast": {"passed": True},
            "wf_2022_defense_preserved": {"passed": True},
        },
        "windows": {
            "wf_2020": {"delta_return": 36.85, "delta_max_dd": -8.21, "delta_avg_pos": -0.04},
            "wf_2022": {"delta_return": 23.42, "delta_max_dd": -19.30, "delta_avg_pos": -0.04},
            "wf_2023_2024": {"delta_return": 25.64, "delta_max_dd": -3.03, "delta_avg_pos": -0.02},
        },
        "blocking_reasons": [] if approved else ["unit-test blocked"],
    }


def _goal_gates() -> dict:
    return {
        "overall": {
            "status": "pending",
            "issues": [
                "external_vercel_setup",
                "scheduled_postclose_target",
                "scheduled_paper_proof",
            ],
        },
        "gates": {
            "model": {
                "status": "ok",
                "detail": "v19reg_v22_budget_usage_bull_guard_v1 (slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)",
            },
            "ordinary_paper_proof": {
                "status": "pending",
                "detail": "selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped",
            },
        },
    }


def _blocked_summary(profile: str, failed_check: str) -> dict:
    return {
        "profile": profile,
        "promotion": {
            "eligible": False,
            "fixed_split": {
                "eligible": False,
                "checks": {
                    failed_check: {
                        "passed": False,
                        "actual": 55.25,
                        "limit": 55.0,
                    }
                },
            },
            "walk_forward_evaluated": False,
        },
    }


def test_model_runtime_readiness_waits_for_scheduled_paper_proof_and_blocks_duplicates():
    review = build_model_runtime_readiness_review(
        current_profile="v19reg_v22_budget_usage_bull_guard_v1",
        composer_review=_composer_review(),
        production_goal_gates=_goal_gates(),
        v22_continuation_summary=_blocked_summary(
            "balanced_rule_residual_v22_continued_retention_v1",
            "test_2022_max_dd",
        ),
        direct_soxx_summary=_blocked_summary(
            "balanced_rule_residual_v22_soxx",
            "test_2022_avg_pos",
        ),
        soxx_walkforward_review={
            "decision": {
                "decision": "blocked",
                "checks": {"wf_2023_2024_recovery_improved": False},
            }
        },
    )

    assert review["decision"] == "ready_waiting_scheduled_paper_proof"
    assert review["runtime_model_ready"] is True
    assert review["should_open_training_branch"] is False
    assert review["production_blockers"] == [
        "external_vercel_setup",
        "scheduled_postclose_target",
        "scheduled_paper_proof",
    ]
    assert review["checks"]["composer_runtime_promotion_approved"]["passed"] is True
    assert review["checks"]["production_model_gate_matches_current_profile"]["passed"] is True
    assert review["checks"]["ordinary_paper_proof_is_fresh_non_duplicate"]["passed"] is False
    blocked = {row["profile"]: row["reason"] for row in review["blocked_experiments"]}
    assert "balanced_rule_residual_v22_continued_retention_v1" in blocked
    assert "balanced_rule_residual_v22_soxx" in blocked
    assert "soxx_relative_weakness_walkforward_v1" in blocked


def test_model_runtime_readiness_blocks_when_composer_review_is_not_approved():
    review = build_model_runtime_readiness_review(
        current_profile="v19reg_v22_budget_usage_bull_guard_v1",
        composer_review=_composer_review(approved=False),
        production_goal_gates=_goal_gates(),
        v22_continuation_summary=_blocked_summary(
            "balanced_rule_residual_v22_continued_retention_v1",
            "test_2022_max_dd",
        ),
        direct_soxx_summary=_blocked_summary(
            "balanced_rule_residual_v22_soxx",
            "test_2022_avg_pos",
        ),
        soxx_walkforward_review={"decision": {"decision": "blocked"}},
    )

    assert review["decision"] == "blocked"
    assert review["runtime_model_ready"] is False
    assert exit_code_for_review(review) == 1


def test_relative_output_label_accepts_relative_output_dir():
    assert relative_output_label(Path("runtime/model/card/model_runtime_readiness_v1")) == Path(
        "runtime/model/card/model_runtime_readiness_v1"
    )

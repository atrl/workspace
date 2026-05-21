from __future__ import annotations

from pathlib import Path

from scripts.review_budget_usage_runtime_promotion import (
    build_runtime_promotion_review,
    relative_output_label,
)


def _window_payload(
    *,
    window: str,
    candidate_return: float,
    fast_return: float,
    candidate_max_dd: float,
    fast_max_dd: float,
    candidate_avg_pos: float,
) -> dict:
    candidate = f"v19reg_v22_budget_usage_bull_guard_v1_{window}"
    fast = f"balanced_rule_residual_v22_{window}"
    slow = f"balanced_rule_residual_v19_reg_{window}"
    return {
        "profile": candidate,
        "slow_label": slow,
        "fast_label": fast,
        "composed_label": candidate,
        "config": {
            "bull_participation_guard": {
                "enabled": True,
                "action": "floor_raw_target_to_fast_position",
            }
        },
        "variants": {
            slow: {"test_proxy": {"return": fast_return, "max_dd": fast_max_dd, "avg_pos": candidate_avg_pos}},
            fast: {"test_proxy": {"return": fast_return, "max_dd": fast_max_dd, "avg_pos": candidate_avg_pos}},
            candidate: {
                "test_proxy": {
                    "return": candidate_return,
                    "sharpe": 1.0,
                    "max_dd": candidate_max_dd,
                    "avg_pos": candidate_avg_pos,
                    "trades": 20,
                }
            },
        },
    }


def test_budget_usage_runtime_promotion_review_approves_valid_walkforward_candidate():
    review = build_runtime_promotion_review(
        profile="v19reg_v22_budget_usage_bull_guard_v1",
        slow_profile="balanced_rule_residual_v19_reg",
        fast_profile="balanced_rule_residual_v22",
        slow_model="soxl_balanced_rule_residual_v19_reg_actor.pt",
        fast_model="soxl_balanced_rule_residual_v22_actor.pt",
        window_payloads={
            "wf_2020": _window_payload(
                window="wf_2020",
                candidate_return=98.27,
                fast_return=61.42,
                candidate_max_dd=26.14,
                fast_max_dd=34.35,
                candidate_avg_pos=0.399,
            ),
            "wf_2022": _window_payload(
                window="wf_2022",
                candidate_return=-20.68,
                fast_return=-44.11,
                candidate_max_dd=26.65,
                fast_max_dd=45.95,
                candidate_avg_pos=0.184,
            ),
            "wf_2023_2024": _window_payload(
                window="wf_2023_2024",
                candidate_return=58.19,
                fast_return=32.56,
                candidate_max_dd=30.18,
                fast_max_dd=33.21,
                candidate_avg_pos=0.337,
            ),
        },
    )

    assert review["decision"] == "approved"
    assert review["runtime_model_gate_allowed"] is True
    assert review["blocking_reasons"] == []
    assert review["checks"]["all_windows_return_beats_fast"]["passed"] is True
    assert review["checks"]["all_windows_maxdd_not_worse_than_fast"]["passed"] is True
    assert review["checks"]["wf_2022_defense_preserved"]["passed"] is True


def test_budget_usage_runtime_promotion_review_blocks_when_recovery_window_loses_to_fast():
    review = build_runtime_promotion_review(
        profile="v19reg_v22_budget_usage_bull_guard_v1",
        slow_profile="balanced_rule_residual_v19_reg",
        fast_profile="balanced_rule_residual_v22",
        slow_model="soxl_balanced_rule_residual_v19_reg_actor.pt",
        fast_model="soxl_balanced_rule_residual_v22_actor.pt",
        window_payloads={
            "wf_2020": _window_payload(
                window="wf_2020",
                candidate_return=98.27,
                fast_return=61.42,
                candidate_max_dd=26.14,
                fast_max_dd=34.35,
                candidate_avg_pos=0.399,
            ),
            "wf_2022": _window_payload(
                window="wf_2022",
                candidate_return=-20.68,
                fast_return=-44.11,
                candidate_max_dd=26.65,
                fast_max_dd=45.95,
                candidate_avg_pos=0.184,
            ),
            "wf_2023_2024": _window_payload(
                window="wf_2023_2024",
                candidate_return=25.00,
                fast_return=32.56,
                candidate_max_dd=30.18,
                fast_max_dd=33.21,
                candidate_avg_pos=0.337,
            ),
        },
    )

    assert review["decision"] == "blocked"
    assert review["runtime_model_gate_allowed"] is False
    assert "candidate does not beat fast path return in every required window" in review["blocking_reasons"]


def test_relative_output_label_accepts_relative_output_dir():
    assert relative_output_label(Path("runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1")) == Path(
        "runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1"
    )

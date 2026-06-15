from __future__ import annotations

import json

import pandas as pd

from src.soxl_quant.app.strict_candidate_retention_filter import (
    derive_strict_candidate_retention_filter,
    render_strict_candidate_retention_filter,
)


def _strict_gate_payload(*, passing_profile_count: int = 0) -> dict[str, object]:
    return {
        "summary": {
            "passing_profile_count": passing_profile_count,
            "passing_profiles": ["full_gate_model"] if passing_profile_count else [],
            "profiles_scanned": 3,
            "reference_profile": "soxl_backtest_policy_reimplementation_v1",
            "reference_return_pct": 145.736343,
            "reference_max_drawdown_pct": 13.554089,
            "current_best_return_pct": 224.361539,
            "required_best_return_pct": 213.143462,
            "max_walkforward_drawdown_pct": 20.0,
            "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
        },
        "rows": [
            {
                "profile": "soxl_backtest_policy_reimplementation_v1",
                "status": "reference",
                "gate_pass": True,
                "retention_decision": "keep_reference_baseline",
                "failures": [],
                "ytd_return_pct": 145.736343,
                "ytd_max_drawdown_pct": 13.554089,
                "worst_wf_max_drawdown_pct": None,
                "missing_wf_window_count": 0,
                "source_path": "runtime/production/card/soxl_backtest_policy_reimplementation_v1/legacy_comparison_metrics.csv",
            },
            {
                "profile": "mode_conditioned_rebound_reward_train_v2",
                "status": "ablation",
                "gate_pass": False,
                "retention_decision": "retain_as_evidence_only",
                "failures": ["wf_max_drawdown_ge_20:wf_2020", "ytd_return_below_best_alignment"],
                "ytd_return_pct": 177.573569,
                "ytd_max_drawdown_pct": 27.278641,
                "worst_wf_max_drawdown_pct": 21.83172,
                "missing_wf_window_count": 0,
                "source_path": "runtime/model/card/mode_conditioned_rebound_reward_train_v2/mode_conditioned_rebound_reward_train_v2_soxl_daily.csv",
            },
        ],
    }


def _soxlbacktest_summary() -> dict[str, object]:
    return {
        "decision": "soxlbacktest_reference_qlib_aligned_not_promotable",
        "canonical_reference_baseline": "soxl_backtest_policy_reimplementation_v1",
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "selected_buy_sell_reference": "legacy_default",
        "selected_risk_floor": "wf_pass_ytd18_15_15_02",
        "legacy_default": {
            "worst_wf_max_drawdown_pct": 84.918053,
            "ytd_return_pct": 132.077977,
            "failures": "worst_wf_max_drawdown_ge_20;ytd_return_below_current_best_alignment",
        },
        "wf_pass_ytd18_15_15_02": {
            "worst_wf_max_drawdown_pct": 18.514585,
            "ytd_return_pct": 18.30891,
            "failures": "ytd_return_below_soxlbacktest_reference;ytd_return_below_current_best_alignment",
        },
    }


def _checkpoint_selection() -> dict[str, object]:
    return {
        "experiment": "cap10_three_fold_checkpoint_drift_selector_v1",
        "profile": "sparse_release_bc_ppo_screen_v1",
        "decision": "three_fold_checkpoint_selector_partial_pass_no_promotion",
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "training_allowed_from_this_card": False,
        "failed_folds": ["wf_2020"],
        "reference_baseline": "soxlbacktest_qlib_buysell_reference",
        "target_max_dd_pct": 20.0,
        "target_relative_return_pp": 30.0,
    }


def _checkpoint_selection_by_fold() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": "wf_2020",
                "selected_candidate": "bc_initialization",
                "selected_kind": "bc_initialization",
                "selected_target_gate": False,
                "selected_drift_gate": True,
                "selected_candidate_gate": False,
                "selected_relative_return": -285.449375,
                "selected_max_dd": 19.417943,
            },
            {
                "fold": "wf_2022",
                "selected_candidate": "bc_initialization",
                "selected_kind": "bc_initialization",
                "selected_target_gate": True,
                "selected_drift_gate": True,
                "selected_candidate_gate": True,
                "selected_relative_return": 47.131485,
                "selected_max_dd": 18.657201,
            },
        ]
    )


def _checkpoint_candidate_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": "wf_2020",
                "candidate": "bc_initialization",
                "candidate_kind": "bc_initialization",
                "test_relative_return": -285.449375,
                "max_dd_delta_vs_bc": 0.0,
            },
            {
                "fold": "wf_2020",
                "candidate": "ppo_retained_iter_4",
                "candidate_kind": "ppo_retained",
                "test_relative_return": -279.467881,
                "max_dd_delta_vs_bc": 5.91584,
            },
        ]
    )


def _rows_by_id(result: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["candidate_id"]): row for row in result["rows"]}


def test_filter_blocks_all_actor_retention_when_strict_gate_has_no_passing_profiles():
    result = derive_strict_candidate_retention_filter(
        strict_gate_payload=_strict_gate_payload(),
        soxlbacktest_summary=_soxlbacktest_summary(),
        checkpoint_selection=_checkpoint_selection(),
        checkpoint_selection_by_fold=_checkpoint_selection_by_fold(),
        checkpoint_candidate_metrics=_checkpoint_candidate_metrics(),
    )

    assert result["summary"]["decision"] == "strict_candidate_retention_filter_no_promotable_candidates"
    assert result["summary"]["promotable_profile_count"] == 0
    assert result["summary"]["actor_retention_allowed_count"] == 0
    assert result["summary"]["ppo_allowed_next"] is False
    row = _rows_by_id(result)["mode_conditioned_rebound_reward_train_v2"]
    assert row["candidate_type"] == "profile"
    assert row["promotion_allowed"] is False
    assert row["actor_retention_allowed"] is False
    assert row["diagnostic_visible_allowed"] is True
    assert row["primary_blocker"] == "wf_max_drawdown_ge_20:wf_2020"
    assert row["required_next_mechanism"] == "strict_walkforward_training_objective"
    json.dumps(result)


def test_filter_keeps_soxlbacktest_as_reference_only_not_runnable_actor():
    result = derive_strict_candidate_retention_filter(
        strict_gate_payload=_strict_gate_payload(),
        soxlbacktest_summary=_soxlbacktest_summary(),
        checkpoint_selection=_checkpoint_selection(),
        checkpoint_selection_by_fold=_checkpoint_selection_by_fold(),
        checkpoint_candidate_metrics=_checkpoint_candidate_metrics(),
    )
    rows = _rows_by_id(result)

    reference = rows["soxl_backtest_policy_reimplementation_v1"]
    assert reference["candidate_type"] == "reference_baseline"
    assert reference["diagnostic_visible_allowed"] is True
    assert reference["runnable_actor"] is False
    assert reference["promotion_allowed"] is False
    assert reference["actor_retention_allowed"] is False
    assert reference["primary_blocker"] == "reference_baseline_comparison_only"
    assert reference["required_next_mechanism"] == "score_trained_candidates_against_reference"

    qlib_screen = rows["soxlbacktest_qlib_walkforward_baseline_v1"]
    assert qlib_screen["candidate_type"] == "qlib_reference_screen"
    assert qlib_screen["promotion_allowed"] is False
    assert qlib_screen["actor_retention_allowed"] is False
    assert qlib_screen["runnable_actor"] is False
    assert qlib_screen["primary_blocker"] == "soxlbacktest_reference_qlib_aligned_not_promotable"


def test_filter_marks_three_fold_checkpoint_selector_as_diagnostic_only_and_stops_ppo_drift():
    result = derive_strict_candidate_retention_filter(
        strict_gate_payload=_strict_gate_payload(),
        soxlbacktest_summary=_soxlbacktest_summary(),
        checkpoint_selection=_checkpoint_selection(),
        checkpoint_selection_by_fold=_checkpoint_selection_by_fold(),
        checkpoint_candidate_metrics=_checkpoint_candidate_metrics(),
    )

    selector = _rows_by_id(result)["cap10_three_fold_checkpoint_drift_selector_v1"]
    assert selector["candidate_type"] == "checkpoint_selector"
    assert selector["promotion_allowed"] is False
    assert selector["actor_retention_allowed"] is False
    assert selector["training_allowed_next"] is False
    assert selector["ppo_allowed_next"] is False
    assert selector["primary_failed_fold"] == "wf_2020"
    assert selector["primary_blocker"] == "trend_underparticipation"
    assert selector["required_next_mechanism"] == "train_bull_trend_participation_with_drawdown_drift_stop_no_promotion"


def test_filter_allows_promotion_review_only_for_strict_gate_profile_pass():
    payload = _strict_gate_payload(passing_profile_count=1)
    payload["rows"].append(
        {
            "profile": "full_gate_model",
            "status": "candidate",
            "gate_pass": True,
            "retention_decision": "retain_candidate",
            "failures": [],
            "ytd_return_pct": 225.0,
            "ytd_max_drawdown_pct": 12.0,
            "worst_wf_max_drawdown_pct": 14.0,
            "missing_wf_window_count": 0,
            "source_path": "runtime/model/card/full_gate_model/full_gate_model_soxl_daily.csv",
        }
    )

    result = derive_strict_candidate_retention_filter(
        strict_gate_payload=payload,
        soxlbacktest_summary=_soxlbacktest_summary(),
        checkpoint_selection={},
        checkpoint_selection_by_fold=pd.DataFrame(
            [
                {
                    "fold": "wf_2020",
                    "selected_candidate": "full_gate_model",
                    "selected_target_gate": True,
                    "selected_drift_gate": True,
                    "selected_candidate_gate": True,
                    "selected_relative_return": 35.0,
                    "selected_max_dd": 14.0,
                },
                {
                    "fold": "wf_2022",
                    "selected_candidate": "full_gate_model",
                    "selected_target_gate": True,
                    "selected_drift_gate": True,
                    "selected_candidate_gate": True,
                    "selected_relative_return": 40.0,
                    "selected_max_dd": 12.0,
                },
                {
                    "fold": "wf_2023_2024",
                    "selected_candidate": "full_gate_model",
                    "selected_target_gate": True,
                    "selected_drift_gate": True,
                    "selected_candidate_gate": True,
                    "selected_relative_return": 45.0,
                    "selected_max_dd": 10.0,
                },
            ]
        ),
        checkpoint_candidate_metrics=pd.DataFrame(),
    )

    profile = _rows_by_id(result)["full_gate_model"]
    assert result["summary"]["decision"] == "strict_candidate_retention_filter_promotion_review_required"
    assert profile["promotion_allowed"] is True
    assert profile["actor_retention_allowed"] is True
    assert profile["primary_blocker"] == ""
    assert "full_gate_model" in render_strict_candidate_retention_filter(result)

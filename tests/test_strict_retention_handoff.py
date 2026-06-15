from __future__ import annotations

import json

import pandas as pd

from src.soxl_quant.app.strict_retention_handoff import (
    derive_strict_retention_handoff,
    render_strict_retention_handoff,
)


def _strict_gate_summary(*, passing_profile_count: int = 0) -> dict[str, object]:
    return {
        "passing_profile_count": passing_profile_count,
        "passing_profiles": ["full_gate_candidate"] if passing_profile_count else [],
        "profiles_scanned": 12,
        "reference_profile": "soxl_backtest_policy_reimplementation_v1",
        "reference_return_pct": 145.736343,
        "reference_max_drawdown_pct": 13.554089,
        "current_best_return_pct": 224.361539,
        "best_return_ratio": 0.95,
        "required_best_return_pct": 213.143462,
        "max_walkforward_drawdown_pct": 20.0,
        "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
    }


def _selection_by_fold(*, wf2020_max_dd: float = 19.417943) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": "wf_2020",
                "status": "evaluated",
                "selected_candidate": "bc_initialization",
                "selected_kind": "bc_initialization",
                "selected_target_gate": False,
                "selected_drift_gate": True,
                "selected_candidate_gate": False,
                "selected_loss": 10.514979,
                "selected_relative_return": -285.449375,
                "selected_max_dd": wf2020_max_dd,
                "bc_relative_return": -285.449375,
                "bc_max_dd": wf2020_max_dd,
                "candidate_count": 4,
            },
            {
                "fold": "wf_2022",
                "status": "evaluated",
                "selected_candidate": "bc_initialization",
                "selected_kind": "bc_initialization",
                "selected_target_gate": True,
                "selected_drift_gate": True,
                "selected_candidate_gate": True,
                "selected_loss": 0.0,
                "selected_relative_return": 47.131485,
                "selected_max_dd": 18.657201,
                "bc_relative_return": 47.131485,
                "bc_max_dd": 18.657201,
                "candidate_count": 4,
            },
            {
                "fold": "wf_2023_2024",
                "status": "evaluated",
                "selected_candidate": "bc_initialization",
                "selected_kind": "bc_initialization",
                "selected_target_gate": True,
                "selected_drift_gate": True,
                "selected_candidate_gate": True,
                "selected_loss": 0.0,
                "selected_relative_return": 38.712278,
                "selected_max_dd": 10.403189,
                "bc_relative_return": 38.712278,
                "bc_max_dd": 10.403189,
                "candidate_count": 4,
            },
        ]
    )


def _candidate_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": "wf_2020",
                "candidate": "bc_initialization",
                "candidate_kind": "bc_initialization",
                "iter": 0,
                "test_relative_return": -285.449375,
                "test_max_dd": 19.417943,
                "relative_delta_vs_bc": 0.0,
                "max_dd_delta_vs_bc": 0.0,
                "target_gate": False,
                "drift_gate": True,
                "candidate_gate": False,
            },
            {
                "fold": "wf_2020",
                "candidate": "ppo_retained_iter_4",
                "candidate_kind": "ppo_retained",
                "iter": 4,
                "test_relative_return": -279.467881,
                "test_max_dd": 25.333784,
                "relative_delta_vs_bc": 5.981494,
                "max_dd_delta_vs_bc": 5.91584,
                "target_gate": False,
                "drift_gate": False,
                "candidate_gate": False,
            },
        ]
    )


def test_handoff_blocks_ppo_when_wf2020_is_underparticipation_with_drawdown_drift():
    handoff = derive_strict_retention_handoff(
        strict_gate_summary=_strict_gate_summary(),
        checkpoint_selection_by_fold=_selection_by_fold(),
        checkpoint_candidate_metrics=_candidate_metrics(),
    )

    assert handoff["decision"] == "train_bull_trend_participation_with_drawdown_drift_stop_no_promotion"
    assert handoff["promotion_allowed"] is False
    assert handoff["actor_retention_allowed"] is False
    assert handoff["ppo_allowed_next"] is False
    assert handoff["diagnostic_training_allowed_next"] is True
    assert handoff["primary_failed_fold"] == "wf_2020"
    assert handoff["reference_baseline"] == "soxl_backtest_policy_reimplementation_v1"
    assert handoff["reference_return_pct"] == 145.736343
    assert handoff["required_best_return_pct"] == 213.143462
    assert handoff["fold_blockers"][0]["blocker"] == "trend_underparticipation"
    assert handoff["fold_blockers"][0]["selected_max_dd_pct"] == 19.417943
    assert handoff["fold_blockers"][0]["selected_relative_return_pp"] == -285.449375
    assert handoff["fold_blockers"][0]["ppo_max_dd_drift_pct"] == 5.91584
    assert "wf2020_trend_participation_floor" in handoff["objective_terms"]
    assert "drawdown_drift_penalty" in handoff["objective_terms"]
    assert "soxlbacktest_reference_alignment" in handoff["objective_terms"]
    assert "do_not_promote_worst_wf_maxdd_ge_20" in handoff["stop_conditions"]
    assert "do_not_continue_same_ppo_when_maxdd_drift_exceeds_2pp" in handoff["stop_conditions"]
    json.dumps(handoff)

    rendered = render_strict_retention_handoff(handoff)
    assert "wf_2020" in rendered
    assert "no promotion" in rendered


def test_handoff_returns_promotion_review_only_when_strict_gate_and_all_folds_pass():
    passing_selection = _selection_by_fold()
    passing_selection.loc[:, "selected_target_gate"] = True
    passing_selection.loc[:, "selected_candidate_gate"] = True
    passing_selection.loc[passing_selection["fold"] == "wf_2020", "selected_relative_return"] = 35.0

    handoff = derive_strict_retention_handoff(
        strict_gate_summary=_strict_gate_summary(passing_profile_count=1),
        checkpoint_selection_by_fold=passing_selection,
        checkpoint_candidate_metrics=pd.DataFrame(),
    )

    assert handoff["decision"] == "candidate_promotion_review"
    assert handoff["promotion_allowed"] is True
    assert handoff["actor_retention_allowed"] is True
    assert handoff["fold_blockers"] == []


def test_handoff_classifies_failed_fold_with_drawdown_breach_as_drawdown_control():
    handoff = derive_strict_retention_handoff(
        strict_gate_summary=_strict_gate_summary(),
        checkpoint_selection_by_fold=_selection_by_fold(wf2020_max_dd=22.5),
        checkpoint_candidate_metrics=pd.DataFrame(),
    )

    assert handoff["decision"] == "train_drawdown_control_objective_no_promotion"
    assert handoff["primary_failed_fold"] == "wf_2020"
    assert handoff["fold_blockers"][0]["blocker"] == "drawdown_control"
    assert "wf_drawdown_control_state_cap" in handoff["objective_terms"]

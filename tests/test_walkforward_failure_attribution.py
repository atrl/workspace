from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.walkforward_failure_attribution import (
    classify_walkforward_checkpoint_failures,
    summarize_walkforward_failure_attribution,
    write_walkforward_failure_attribution,
)


def test_classify_walkforward_checkpoint_failures_marks_bull_underparticipation_when_drawdown_passes():
    candidates = pd.DataFrame(
        [
            {
                "fold": "wf_2020",
                "candidate": "bc_initialization",
                "test_return": 40.58,
                "test_buy_and_hold_return": 333.33,
                "test_relative_return": -292.75,
                "test_max_dd": 14.98,
                "target_gate": False,
                "candidate_gate": False,
            },
            {
                "fold": "wf_2020",
                "candidate": "ppo_retained_iter_4",
                "test_return": 61.32,
                "test_buy_and_hold_return": 333.33,
                "test_relative_return": -272.01,
                "test_max_dd": 20.92,
                "target_gate": False,
                "candidate_gate": False,
            },
            {
                "fold": "wf_2022",
                "candidate": "bc_initialization",
                "test_return": -12.70,
                "test_buy_and_hold_return": -78.57,
                "test_relative_return": 65.86,
                "test_max_dd": 18.80,
                "target_gate": True,
                "candidate_gate": True,
            },
        ]
    )
    selected = pd.DataFrame(
        [
            {
                "fold": "wf_2020",
                "selected_candidate": "bc_initialization",
                "selected_relative_return": -292.75,
                "selected_max_dd": 14.98,
                "selected_candidate_gate": False,
            },
            {
                "fold": "wf_2022",
                "selected_candidate": "bc_initialization",
                "selected_relative_return": 65.86,
                "selected_max_dd": 18.80,
                "selected_candidate_gate": True,
            },
        ]
    )

    report = classify_walkforward_checkpoint_failures(candidates, selected)
    by_fold = report.set_index("fold")

    assert by_fold.loc["wf_2020", "failure_type"] == "bull_underparticipation"
    assert by_fold.loc["wf_2020", "selected_drawdown_gap_pct"] == 0.0
    assert by_fold.loc["wf_2020", "selected_relative_gap_pp"] == 322.75
    assert by_fold.loc["wf_2020", "best_relative_candidate"] == "ppo_retained_iter_4"
    assert by_fold.loc["wf_2020", "best_relative_max_dd_gap_pct"] == 0.92
    assert by_fold.loc["wf_2020", "checkpoint_selection_exhausted"] is True
    assert by_fold.loc["wf_2020", "next_training_target"] == "bull_rebound_participation_lift_with_drawdown_budget"
    assert by_fold.loc["wf_2022", "failure_type"] == "pass"


def test_summarize_walkforward_failure_attribution_routes_to_target_rebuild():
    attribution = pd.DataFrame(
        [
            {
                "fold": "wf_2020",
                "failure_type": "bull_underparticipation",
                "selected_candidate_gate": False,
                "checkpoint_selection_exhausted": True,
                "selected_relative_gap_pp": 322.75,
                "selected_drawdown_gap_pct": 0.0,
            },
            {
                "fold": "wf_2022",
                "failure_type": "pass",
                "selected_candidate_gate": True,
                "checkpoint_selection_exhausted": False,
                "selected_relative_gap_pp": 0.0,
                "selected_drawdown_gap_pct": 0.0,
            },
        ]
    )

    summary = summarize_walkforward_failure_attribution(attribution)

    assert summary["promotion_allowed"] is False
    assert summary["failed_folds"] == ["wf_2020"]
    assert summary["primary_failure_type"] == "bull_underparticipation"
    assert summary["next_mechanism"] == "rebuild_wf2020_bull_participation_target_before_full_retention_training"


def test_write_walkforward_failure_attribution_outputs_csv_and_summary(tmp_path):
    candidates_path = tmp_path / "candidates.csv"
    selected_path = tmp_path / "selected.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        [
            {
                "fold": "wf_2020",
                "candidate": "bc_initialization",
                "test_return": 40.0,
                "test_buy_and_hold_return": 300.0,
                "test_relative_return": -260.0,
                "test_max_dd": 12.0,
            }
        ]
    ).to_csv(candidates_path, index=False)
    pd.DataFrame(
        [
            {
                "fold": "wf_2020",
                "selected_candidate": "bc_initialization",
                "selected_relative_return": -260.0,
                "selected_max_dd": 12.0,
                "selected_candidate_gate": False,
            }
        ]
    ).to_csv(selected_path, index=False)

    result = write_walkforward_failure_attribution(
        candidates_path=candidates_path,
        selected_by_fold_path=selected_path,
        output_dir=output_dir,
    )

    assert result["attribution_path"].exists()
    assert result["summary_path"].exists()
    assert pd.read_csv(result["attribution_path"]).loc[0, "failure_type"] == "bull_underparticipation"
    assert result["summary"]["failed_folds"] == ["wf_2020"]


def test_missing_candidate_metrics_still_summarizes():
    report = classify_walkforward_checkpoint_failures(
        pd.DataFrame(
            [
                {
                    "fold": "wf_2022",
                    "candidate": "bc_initialization",
                    "test_return": -12.7,
                    "test_buy_and_hold_return": -78.5,
                    "test_relative_return": 65.8,
                    "test_max_dd": 18.8,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "fold": "wf_2020",
                    "selected_candidate": "bc_initialization",
                    "selected_relative_return": -260.0,
                    "selected_max_dd": 12.0,
                    "selected_candidate_gate": False,
                }
            ]
        ),
    )

    summary = summarize_walkforward_failure_attribution(report)

    assert report.loc[0, "failure_type"] == "missing_candidate_metrics"
    assert summary["failed_folds"] == ["wf_2020"]

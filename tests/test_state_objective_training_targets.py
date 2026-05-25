from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.state_objective_training_targets import (
    StateObjectiveBCReadinessConfig,
    StateObjectiveReboundClusterConfig,
    StateObjectiveTrainingTargetConfig,
    build_fold_local_state_objective_training_targets,
    build_state_objective_training_targets,
    discover_rebound_clusters_from_daily_path,
    filter_state_objective_targets_for_train_window,
    score_state_objective_bc_readiness,
    score_state_objective_checkpoint_candidates,
    score_state_objective_target_alignment,
    summarize_state_objective_bc_readiness,
    summarize_state_objective_checkpoint_selection,
    summarize_state_objective_target_alignment,
    summarize_state_objective_training_targets,
)


def test_training_targets_mark_rebound_cluster_and_floor_teacher_position():
    daily_paths = {
        "strict_oos": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-02-01", "2026-02-03", "2026-02-04"]),
                "strategy_nav": [100.0, 101.0, 102.0],
                "buyhold_nav": [100.0, 120.0, 130.0],
                "router_position": [0.12, 0.18, 0.42],
                "rebound_state": [False, True, True],
                "risk_state": [False, False, False],
            }
        )
    }
    window_metrics = pd.DataFrame(
        {
            "candidate": ["teacher"],
            "window": ["strict_oos"],
            "relative_return": [-45.0],
            "max_dd": [12.0],
        }
    )
    cluster_metrics = pd.DataFrame(
        {
            "candidate": ["teacher"],
            "cluster": ["cluster_2026_high_vol_rebound"],
            "start": ["2026-02-03"],
            "end": ["2026-02-04"],
            "relative_return": [-140.0],
            "max_dd": [8.0],
            "avg_pos": [0.20],
        }
    )

    targets = build_state_objective_training_targets(
        daily_paths,
        window_metrics=window_metrics,
        cluster_metrics=cluster_metrics,
        candidate="teacher",
    )

    cluster_rows = targets[targets["cluster"] == "cluster_2026_high_vol_rebound"]
    outside_rows = targets[targets["cluster"] == ""]
    assert cluster_rows["is_rebound_cluster"].all()
    assert cluster_rows["rebound_participation_weight"].min() > 0
    assert cluster_rows["rebound_target_position"].iloc[0] == 0.35
    assert cluster_rows["rebound_target_position"].iloc[1] == 0.42
    assert outside_rows["rebound_participation_weight"].max() == 0.0
    assert outside_rows["rebound_target_position"].iloc[0] == 0.12


def test_training_targets_mark_downtrend_drawdown_cap_from_window_gap():
    daily_paths = {
        "wf_2020": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-03-02", "2020-03-03"]),
                "strategy_nav": [100.0, 90.0],
                "buyhold_nav": [100.0, 80.0],
                "position": [0.62, 0.55],
                "rebound_state": [False, False],
                "risk_state": [True, False],
            }
        )
    }
    window_metrics = pd.DataFrame(
        {
            "candidate": ["teacher"],
            "window": ["wf_2020"],
            "relative_return": [40.0],
            "max_dd": [27.0],
        }
    )
    cluster_metrics = pd.DataFrame(
        columns=["candidate", "cluster", "start", "end", "relative_return", "max_dd", "avg_pos"]
    )

    targets = build_state_objective_training_targets(
        daily_paths,
        window_metrics=window_metrics,
        cluster_metrics=cluster_metrics,
        candidate="teacher",
        config=StateObjectiveTrainingTargetConfig(downtrend_position_cap=0.25),
    )

    risk_row = targets.loc[targets["is_downtrend_state"]].iloc[0]
    normal_row = targets.loc[~targets["is_downtrend_state"]].iloc[0]
    assert risk_row["downtrend_drawdown_weight"] > 0
    assert risk_row["risk_target_position"] == 0.25
    assert normal_row["downtrend_drawdown_weight"] == 0.0
    assert normal_row["risk_target_position"] == 0.55


def test_training_target_summary_reports_top_weighted_windows_and_terms():
    daily_paths = {
        "wf_2020": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                "strategy_nav": [100.0, 101.0],
                "buyhold_nav": [100.0, 120.0],
                "position": [0.10, 0.15],
                "rebound_state": [False, True],
                "risk_state": [False, False],
            }
        )
    }
    window_metrics = pd.DataFrame(
        {
            "candidate": ["teacher"],
            "window": ["wf_2020"],
            "relative_return": [-20.0],
            "max_dd": [18.0],
        }
    )
    cluster_metrics = pd.DataFrame(
        {
            "candidate": ["teacher"],
            "cluster": ["cluster_a"],
            "start": ["2020-01-03"],
            "end": ["2020-01-03"],
            "relative_return": [-70.0],
            "max_dd": [5.0],
            "avg_pos": [0.15],
        }
    )
    targets = build_state_objective_training_targets(
        daily_paths,
        window_metrics=window_metrics,
        cluster_metrics=cluster_metrics,
        candidate="teacher",
    )

    summary = summarize_state_objective_training_targets(targets)

    assert summary["rows"] == 2
    assert summary["top_weighted_window"] == "wf_2020"
    assert summary["rebound_rows"] == 1
    assert summary["downtrend_rows"] == 0
    assert "high_vol_rebound_participation_floor" in summary["required_terms"]
    assert "walkforward_drawdown_gap_penalty" in summary["required_terms"]


def test_target_alignment_prefers_rebound_fit_and_risk_cap_compliance():
    targets = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-02-03", "2026-02-04"]),
            "window": ["strict_oos", "strict_oos"],
            "rebound_target_position": [0.35, 0.30],
            "risk_target_position": [0.35, 0.20],
            "teacher_position": [0.12, 0.30],
            "objective_weight": [8.0, 12.0],
            "is_rebound_cluster": [True, False],
            "is_downtrend_state": [False, True],
        }
    )
    candidate_paths = {
        "good": {
            "strict_oos": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-02-03", "2026-02-04"]),
                    "strategy_nav": [100.0, 106.0],
                    "buyhold_nav": [100.0, 104.0],
                    "position": [0.35, 0.15],
                }
            )
        },
        "bad": {
            "strict_oos": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-02-03", "2026-02-04"]),
                    "strategy_nav": [100.0, 106.0],
                    "buyhold_nav": [100.0, 104.0],
                    "position": [0.10, 0.60],
                }
            )
        },
    }

    scored = score_state_objective_target_alignment(candidate_paths, targets)

    assert scored.iloc[0]["candidate"] == "good"
    by_candidate = scored.set_index("candidate")
    assert by_candidate.loc["bad", "rebound_position_mae"] > by_candidate.loc["good", "rebound_position_mae"]
    assert by_candidate.loc["bad", "risk_cap_violation"] > by_candidate.loc["good", "risk_cap_violation"]


def test_target_alignment_penalizes_missing_target_coverage():
    targets = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "window": ["wf_2020", "wf_2020"],
            "rebound_target_position": [0.20, 0.20],
            "risk_target_position": [0.20, 0.20],
            "teacher_position": [0.20, 0.20],
            "objective_weight": [1.0, 1.0],
            "is_rebound_cluster": [False, False],
            "is_downtrend_state": [False, False],
        }
    )
    candidate_paths = {
        "covered": {
            "wf_2020": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-01-02", "2026-01-03"]),
                    "strategy_nav": [100.0, 103.0],
                    "buyhold_nav": [100.0, 100.0],
                    "position": [0.20, 0.20],
                }
            )
        },
        "missing": {
            "wf_2020": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-01-02"]),
                    "strategy_nav": [100.0],
                    "buyhold_nav": [100.0],
                    "position": [0.20],
                }
            )
        },
    }

    scored = score_state_objective_target_alignment(candidate_paths, targets)

    assert scored.set_index("candidate").loc["missing", "missing_target_ratio"] == 0.5
    assert scored.iloc[0]["candidate"] == "covered"


def test_target_alignment_summary_blocks_promotion_without_full_gate():
    scored = pd.DataFrame(
        {
            "candidate": ["near"],
            "alignment_loss": [12.0],
            "window_gate": [False],
            "coverage_gate": [True],
            "missing_target_ratio": [0.0],
            "rebound_position_mae": [0.1],
            "risk_cap_violation": [0.0],
        }
    )

    summary = summarize_state_objective_target_alignment(scored)

    assert summary["decision"] == "target_alignment_no_promotion"
    assert summary["selected_candidate"] == "near"
    assert summary["next_mechanism"] == "train_or_checkpoint_score_state_objective_targets"


def test_checkpoint_selection_prefers_state_objective_loss_over_legacy_score():
    alignment = pd.DataFrame(
        {
            "candidate": ["legacy_high", "state_fit"],
            "alignment_loss": [2.0, 0.5],
            "window_gate_loss": [300.0, 30.0],
            "missing_target_ratio": [0.0, 0.0],
            "window_gate": [False, False],
            "coverage_gate": [True, True],
            "rebound_position_mae": [0.2, 0.05],
            "risk_cap_violation": [0.3, 0.01],
            "score": [120.0, 20.0],
        }
    )

    selected = score_state_objective_checkpoint_candidates(alignment)

    assert selected.iloc[0]["candidate"] == "state_fit"
    assert selected.iloc[0]["checkpoint_objective_loss"] < selected.iloc[1]["checkpoint_objective_loss"]


def test_checkpoint_selection_summary_requires_window_and_coverage_gates_for_promotion():
    scored = pd.DataFrame(
        {
            "candidate": ["near"],
            "checkpoint_objective_loss": [0.7],
            "window_gate": [False],
            "coverage_gate": [True],
            "alignment_loss": [0.2],
            "window_gate_loss": [50.0],
            "missing_target_ratio": [0.0],
        }
    )

    summary = summarize_state_objective_checkpoint_selection(scored)

    assert summary["decision"] == "checkpoint_objective_no_promotion"
    assert summary["selected_candidate"] == "near"
    assert summary["next_mechanism"] == "train_state_objective_checkpoint_or_bc"


def test_checkpoint_selection_summary_allows_review_when_all_gates_pass():
    scored = pd.DataFrame(
        {
            "candidate": ["pass"],
            "checkpoint_objective_loss": [0.1],
            "window_gate": [True],
            "coverage_gate": [True],
            "alignment_loss": [0.1],
            "window_gate_loss": [0.0],
            "missing_target_ratio": [0.0],
        }
    )

    summary = summarize_state_objective_checkpoint_selection(scored)

    assert summary["decision"] == "checkpoint_objective_promotion_review"
    assert summary["next_mechanism"] == "rerun_walkforward_promotion_review"


def test_bc_readiness_blocks_global_oos_targets_even_when_distribution_is_healthy():
    label_distribution = pd.DataFrame(
        {
            "profile": ["risk_budget_usage_v1", "balanced_rule_residual_v22"],
            "samples": [755, 755],
            "unique_actions": [5, 5],
            "dominant_action_share": [0.42, 0.97],
            "action_entropy": [1.30, 0.16],
        }
    )

    scored = score_state_objective_bc_readiness(
        label_distribution,
        target_scope="global_oos_walkforward",
    )
    summary = summarize_state_objective_bc_readiness(scored)

    top = scored.iloc[0]
    assert top["profile"] == "risk_budget_usage_v1"
    assert bool(top["label_distribution_gate"]) is True
    assert bool(top["fold_local_scope_gate"]) is False
    assert bool(top["bc_readiness_gate"]) is False
    assert summary["decision"] == "state_objective_bc_scope_blocked_no_promotion"
    assert summary["next_mechanism"] == "generate_fold_local_state_objective_targets"


def test_bc_readiness_approves_fold_local_budget_usage_distribution():
    label_distribution = pd.DataFrame(
        {
            "profile": ["balanced_rule_residual_v22", "risk_budget_usage_v1"],
            "samples": [755, 755],
            "unique_actions": [5, 5],
            "dominant_action_share": [0.97, 0.42],
            "action_entropy": [0.16, 1.30],
        }
    )

    scored = score_state_objective_bc_readiness(
        label_distribution,
        target_scope="fold_local",
        config=StateObjectiveBCReadinessConfig(max_dominant_action_share=0.90),
    )
    summary = summarize_state_objective_bc_readiness(scored)

    assert scored.iloc[0]["profile"] == "risk_budget_usage_v1"
    assert bool(scored.iloc[0]["bc_readiness_gate"]) is True
    assert bool(scored.set_index("profile").loc["balanced_rule_residual_v22", "label_distribution_gate"]) is False
    assert summary["decision"] == "state_objective_bc_ready_for_training"
    assert summary["next_mechanism"] == "run_fold_local_bc_warm_start_then_walkforward_ppo"


def test_filter_state_objective_targets_for_train_window_rejects_eval_overlap():
    targets = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "teacher_position": [0.1, 0.2, 0.3],
        }
    )

    filtered = filter_state_objective_targets_for_train_window(
        targets,
        train_start="2020-01-02",
        train_end="2020-01-03",
        eval_start="2020-01-06",
        eval_end="2020-01-10",
    )

    assert filtered["datetime"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-02", "2020-01-03"]

    try:
        filter_state_objective_targets_for_train_window(
            targets,
            train_start="2020-01-02",
            train_end="2020-01-06",
            eval_start="2020-01-03",
            eval_end="2020-01-10",
        )
    except ValueError as exc:
        assert "overlaps eval window" in str(exc)
    else:
        raise AssertionError("overlapping train/eval windows did not raise")


def test_fold_local_target_builder_filters_eval_rows_and_future_clusters():
    daily_paths = {
        "teacher_path": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
                "strategy_nav": [100.0, 90.0, 95.0],
                "buyhold_nav": [100.0, 120.0, 130.0],
                "router_position": [0.20, 0.15, 0.80],
                "rebound_state": [False, True, True],
                "risk_state": [False, False, True],
            }
        )
    }
    cluster_metrics = pd.DataFrame(
        {
            "candidate": ["teacher", "teacher"],
            "cluster": ["train_rebound", "future_eval_rebound"],
            "start": ["2020-01-03", "2020-01-06"],
            "end": ["2020-01-03", "2020-01-06"],
            "relative_return": [-80.0, -120.0],
            "max_dd": [8.0, 9.0],
            "avg_pos": [0.10, 0.05],
        }
    )

    targets = build_fold_local_state_objective_training_targets(
        daily_paths,
        folds=[
            {
                "name": "wf_2022_train",
                "source_window": "teacher_path",
                "train_start": "2020-01-02",
                "train_end": "2020-01-03",
                "eval_start": "2020-01-06",
                "eval_end": "2020-01-10",
            }
        ],
        cluster_metrics=cluster_metrics,
        candidate="teacher",
    )

    assert targets["datetime"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-02", "2020-01-03"]
    assert targets["target_scope"].unique().tolist() == ["fold_local"]
    assert targets["fold"].unique().tolist() == ["wf_2022_train"]
    assert targets["cluster"].tolist() == ["", "train_rebound"]
    assert "future_eval_rebound" not in targets["cluster"].tolist()


def test_fold_local_target_builder_filters_clusters_by_fold_when_available():
    daily_paths = {
        "teacher_path": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                "strategy_nav": [100.0, 90.0],
                "buyhold_nav": [100.0, 120.0],
                "router_position": [0.20, 0.15],
                "rebound_state": [False, True],
                "risk_state": [False, False],
            }
        )
    }
    cluster_metrics = pd.DataFrame(
        {
            "candidate": ["teacher", "teacher"],
            "fold": ["other_fold", "wf_2022_train"],
            "cluster": ["other_fold_rebound", "train_rebound"],
            "start": ["2020-01-03", "2020-01-03"],
            "end": ["2020-01-03", "2020-01-03"],
            "relative_return": [-200.0, -80.0],
            "max_dd": [8.0, 8.0],
            "avg_pos": [0.01, 0.10],
        }
    )

    targets = build_fold_local_state_objective_training_targets(
        daily_paths,
        folds=[
            {
                "name": "wf_2022_train",
                "source_window": "teacher_path",
                "train_start": "2020-01-02",
                "train_end": "2020-01-03",
                "eval_start": "2020-01-06",
                "eval_end": "2020-01-10",
            }
        ],
        cluster_metrics=cluster_metrics,
        candidate="teacher",
    )

    assert targets["cluster"].tolist() == ["", "train_rebound"]


def test_fold_local_target_builder_rejects_overlapping_train_eval_window():
    daily_paths = {
        "teacher_path": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                "strategy_nav": [100.0, 101.0],
                "buyhold_nav": [100.0, 99.0],
                "position": [0.20, 0.30],
            }
        )
    }

    try:
        build_fold_local_state_objective_training_targets(
            daily_paths,
            folds=[
                {
                    "name": "bad",
                    "source_window": "teacher_path",
                    "train_start": "2020-01-02",
                    "train_end": "2020-01-03",
                    "eval_start": "2020-01-03",
                    "eval_end": "2020-01-10",
                }
            ],
            cluster_metrics=pd.DataFrame(),
            candidate="teacher",
        )
    except ValueError as exc:
        assert "overlaps eval window" in str(exc)
    else:
        raise AssertionError("overlapping fold did not raise")


def test_discover_rebound_clusters_groups_contiguous_train_rebound_rows():
    daily = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-08",
                    "2020-01-09",
                ]
            ),
            "strategy_nav": [100.0, 98.0, 104.0, 102.0, 103.0, 106.0],
            "buyhold_nav": [100.0, 110.0, 120.0, 118.0, 122.0, 124.0],
            "router_position": [0.20, 0.18, 0.22, 0.24, 0.26, 0.30],
            "rebound_state": [False, True, True, False, True, True],
        }
    )

    clusters = discover_rebound_clusters_from_daily_path(
        daily,
        candidate="teacher",
        fold="wf_2022_train",
        train_start="2020-01-02",
        train_end="2020-01-09",
        config=StateObjectiveReboundClusterConfig(min_rows=2),
    )

    assert clusters["cluster"].tolist() == [
        "wf_2022_train_rebound_1",
        "wf_2022_train_rebound_2",
    ]
    assert clusters["candidate"].tolist() == ["teacher", "teacher"]
    assert clusters["fold"].tolist() == ["wf_2022_train", "wf_2022_train"]
    assert clusters["rows"].tolist() == [2, 2]
    assert clusters["start"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-03", "2020-01-08"]
    assert clusters["end"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-06", "2020-01-09"]
    assert set(["return", "buy_and_hold_return", "relative_return", "max_dd", "avg_pos"]).issubset(
        clusters.columns
    )


def test_discover_rebound_clusters_filters_short_runs_and_eval_dates():
    daily = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-08",
                    "2020-01-09",
                ]
            ),
            "strategy_nav": [100.0, 101.0, 102.0, 97.0, 99.0, 101.0],
            "buyhold_nav": [100.0, 103.0, 104.0, 106.0, 108.0, 109.0],
            "position": [0.20, 0.22, 0.24, 0.10, 0.15, 0.18],
            "rebound_state": [True, True, True, False, True, True],
        }
    )

    clusters = discover_rebound_clusters_from_daily_path(
        daily,
        candidate="teacher",
        fold="wf_2022_train",
        train_start="2020-01-02",
        train_end="2020-01-06",
        config=StateObjectiveReboundClusterConfig(min_rows=3),
    )

    assert clusters["cluster"].tolist() == ["wf_2022_train_rebound_1"]
    assert clusters["start"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-02"]
    assert clusters["end"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-06"]
    assert clusters["rows"].tolist() == [3]

from __future__ import annotations

import pandas as pd

from src.soxl_quant.app import state_objective_training_targets
from src.soxl_quant.app.state_objective_training_targets import (
    BullReentryParticipationTargetConfig,
    StateObjectiveDrawdownControlTargetConfig,
    StateObjectiveBCReadinessConfig,
    StateObjectiveReboundTargetEnrichmentConfig,
    StateObjectiveReboundClusterConfig,
    StateObjectiveTrainingTargetConfig,
    TrendParticipationTargetConfig,
    apply_state_objective_sparse_release_gate,
    apply_bull_reentry_participation_target_gate,
    apply_trend_participation_target_gate,
    audit_state_objective_target_lift,
    audit_state_objective_target_scope,
    build_fold_local_joint_target_contract,
    build_drawdown_control_training_targets,
    build_fold_local_state_objective_training_targets,
    build_state_objective_training_targets,
    discover_rebound_clusters_from_daily_path,
    enrich_state_objective_rebound_targets,
    filter_state_objective_targets_for_train_window,
    score_state_objective_bc_readiness,
    score_state_objective_checkpoint_candidates,
    score_state_objective_target_alignment,
    summarize_state_objective_bc_readiness,
    summarize_state_objective_checkpoint_selection,
    summarize_state_objective_target_alignment,
    summarize_state_objective_target_lift_audit,
    summarize_state_objective_target_scope_audit,
    summarize_state_objective_training_targets,
)


def test_drawdown_control_targets_cap_high_exposure_inside_failed_interval():
    daily = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2022-03-30", "2022-04-01", "2022-04-04", "2022-04-05"]),
            "nav": [1.00, 0.94, 0.82, 0.84],
            "position": [0.55, 0.50, 0.46, 0.20],
            "drawdown_63": [-0.05, -0.12, -0.30, -0.28],
        }
    )
    intervals = pd.DataFrame(
        {
            "rank": [1],
            "start": ["2022-04-01"],
            "end": ["2022-04-04"],
            "max_dd": [18.0],
        }
    )

    targets = build_drawdown_control_training_targets(
        daily,
        intervals=intervals,
        candidate="ppo_selected",
        window="wf_2022",
        config=StateObjectiveDrawdownControlTargetConfig(
            drawdown_state_threshold=0.10,
            downtrend_position_cap=0.25,
            min_position_delta=0.05,
        ),
    )

    by_date = targets.set_index("datetime")
    calm = by_date.loc[pd.Timestamp("2022-03-30")]
    risk = by_date.loc[pd.Timestamp("2022-04-04")]
    low_exposure = by_date.loc[pd.Timestamp("2022-04-05")]
    assert bool(calm["drawdown_control_state"]) is False
    assert calm["risk_target_position"] == 0.55
    assert bool(risk["drawdown_control_state"]) is True
    assert bool(risk["is_downtrend_state"]) is True
    assert risk["risk_target_position"] == 0.25
    assert risk["risk_cap_delta"] == 0.21
    assert risk["source_interval_rank"] == 1
    assert risk["downtrend_drawdown_weight"] > 0
    assert risk["objective_weight"] > calm["objective_weight"]
    assert bool(low_exposure["drawdown_control_state"]) is True
    assert bool(low_exposure["is_downtrend_state"]) is False
    assert low_exposure["risk_target_position"] == 0.20


def test_drawdown_control_targets_accept_existing_percent_drawdown_column():
    daily = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2022-01-03", "2022-01-04"]),
            "drawdown": [0.0, 16.0],
            "target_position": [0.40, 0.60],
        }
    )

    targets = build_drawdown_control_training_targets(
        daily,
        candidate="candidate_a",
        window="wf_2022",
        config=StateObjectiveDrawdownControlTargetConfig(
            drawdown_state_threshold=0.10,
            downtrend_position_cap=0.30,
        ),
    )

    risky = targets.set_index("datetime").loc[pd.Timestamp("2022-01-04")]
    assert risky["path_drawdown"] == 0.16
    assert risky["teacher_position"] == 0.60
    assert risky["risk_target_position"] == 0.30
    assert risky["window_drawdown_gap"] == 1.0


def test_drawdown_control_targets_cap_objective_weight_to_limit_overfit():
    daily = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2022-01-03", "2022-01-04"]),
            "drawdown": [0.0, 0.45],
            "position": [0.50, 0.70],
        }
    )

    targets = build_drawdown_control_training_targets(
        daily,
        candidate="candidate_a",
        window="wf_2022",
        config=StateObjectiveDrawdownControlTargetConfig(
            drawdown_gap_weight=100.0,
            max_objective_weight=12.0,
        ),
    )

    assert targets["objective_weight"].max() == 12.0


def test_short_volshock_de_risk_target_gate_uses_fold_local_thresholds_and_caps_high_exposure():
    targets = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2022-01-03"]),
            "fold": ["wf_2022_train", "wf_2022_train", "wf_2023_2024_train"],
            "window": ["wf_2022_train", "wf_2022_train", "wf_2023_2024_train"],
            "teacher_position": [0.42, 0.04, 0.50],
            "risk_target_position": [0.42, 0.04, 0.50],
            "objective_weight": [1.0, 2.0, 1.0],
            "is_downtrend_state": [False, False, False],
        }
    )
    feature_frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2022-01-03"]),
            "realized_vol_21": [0.90, 0.90, 1.10],
            "vol_of_vol_21": [0.30, 0.30, 0.30],
            "downside_semivol_21": [0.20, 0.20, 0.80],
        }
    )
    thresholds = pd.DataFrame(
        {
            "fold": ["wf_2022", "wf_2023_2024"],
            "realized_vol_21_q80": [0.80, 1.00],
            "vol_of_vol_21_q80": [0.25, 0.25],
            "downside_semivol_21_q80": [0.60, 0.70],
        }
    )

    revised = state_objective_training_targets.apply_short_volshock_de_risk_target_gate(
        targets,
        feature_frame,
        thresholds,
        config=state_objective_training_targets.ShortVolshockDeRiskTargetConfig(
            target_cap=0.05,
            min_feature_hits=2,
            min_position_delta=0.02,
        ),
    )

    by_date = revised.set_index("datetime")
    capped = by_date.loc[pd.Timestamp("2020-01-02")]
    already_low = by_date.loc[pd.Timestamp("2020-01-03")]
    other_fold = by_date.loc[pd.Timestamp("2022-01-03")]
    assert bool(capped["short_volshock_state"]) is True
    assert capped["short_volshock_feature_hits"] == 2
    assert bool(capped["short_volshock_target_state"]) is True
    assert capped["risk_target_position_before_short_volshock"] == 0.42
    assert capped["risk_target_position"] == 0.05
    assert capped["short_volshock_risk_cap_delta"] == 0.37
    assert bool(capped["is_downtrend_state"]) is True
    assert capped["objective_weight"] > 1.0
    assert bool(already_low["short_volshock_state"]) is True
    assert bool(already_low["short_volshock_target_state"]) is False
    assert already_low["risk_target_position"] == 0.04
    assert bool(other_fold["short_volshock_state"]) is True
    assert other_fold["short_volshock_feature_hits"] == 3
    assert other_fold["risk_target_position"] == 0.05


def test_short_volshock_target_summary_routes_ready_targets_to_bc_or_checkpoint():
    targets = pd.DataFrame(
        {
            "short_volshock_state": [True, True, False],
            "short_volshock_target_state": [True, False, False],
            "short_volshock_risk_cap_delta": [0.30, 0.0, 0.0],
            "fold": ["wf_2022_train", "wf_2022_train", "wf_2023_2024_train"],
        }
    )

    summary = state_objective_training_targets.summarize_short_volshock_de_risk_targets(targets)

    assert summary["decision"] == "short_volshock_de_risk_targets_ready"
    assert summary["rows"] == 3
    assert summary["short_volshock_state_rows"] == 2
    assert summary["target_state_rows"] == 1
    assert summary["next_mechanism"] == "score_checkpoints_or_run_bc_with_short_volshock_targets"


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


def test_rebound_target_enrichment_expands_neighbors_inside_same_fold_only():
    targets = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-05",
                ]
            ),
            "fold": ["wf_a", "wf_a", "wf_a", "wf_a", "wf_a", "wf_b"],
            "window": ["wf_a", "wf_a", "wf_a", "wf_a", "wf_a", "wf_b"],
            "cluster": ["", "", "wf_a_rebound_1", "", "", ""],
            "teacher_position": [0.10, 0.18, 0.42, 0.22, 0.15, 0.70],
            "rebound_target_position": [0.10, 0.18, 0.42, 0.22, 0.15, 0.70],
            "risk_target_position": [0.10, 0.18, 0.42, 0.22, 0.15, 0.70],
            "objective_weight": [1.0, 1.0, 10.0, 1.0, 1.0, 1.0],
            "rebound_participation_weight": [0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
            "cluster_participation_gap": [0.0, 0.0, 0.17, 0.0, 0.0, 0.0],
            "is_rebound_cluster": [False, False, True, False, False, False],
            "is_downtrend_state": [False, False, False, False, False, False],
        }
    )

    enriched = enrich_state_objective_rebound_targets(
        targets,
        StateObjectiveReboundTargetEnrichmentConfig(radius=1, min_target_position=0.35),
    )

    wf_a = enriched[enriched["fold"] == "wf_a"].set_index("datetime").sort_index()
    assert wf_a.loc[pd.Timestamp("2026-01-05"), "is_rebound_cluster"]
    assert wf_a.loc[pd.Timestamp("2026-01-07"), "is_rebound_cluster"]
    assert not wf_a.loc[pd.Timestamp("2026-01-02"), "is_rebound_cluster"]
    assert enriched.loc[enriched["fold"] == "wf_b", "is_rebound_cluster"].tolist() == [False]
    assert wf_a.loc[pd.Timestamp("2026-01-05"), "rebound_target_position"] == 0.35
    assert wf_a.loc[pd.Timestamp("2026-01-07"), "rebound_enrichment_distance"] == 1
    assert wf_a.loc[pd.Timestamp("2026-01-07"), "rebound_enrichment_source"] == "neighbor_rebound"
    assert wf_a.loc[pd.Timestamp("2026-01-05"), "objective_weight"] > 1.0


def test_rebound_target_enrichment_does_not_mark_downtrend_neighbors_by_default():
    targets = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            "fold": ["wf_a", "wf_a", "wf_a"],
            "window": ["wf_a", "wf_a", "wf_a"],
            "cluster": ["", "wf_a_rebound_1", ""],
            "teacher_position": [0.15, 0.45, 0.30],
            "rebound_target_position": [0.15, 0.45, 0.30],
            "risk_target_position": [0.15, 0.45, 0.25],
            "objective_weight": [1.0, 10.0, 1.0],
            "rebound_participation_weight": [0.0, 9.0, 0.0],
            "is_rebound_cluster": [False, True, False],
            "is_downtrend_state": [False, False, True],
        }
    )

    enriched = enrich_state_objective_rebound_targets(
        targets,
        StateObjectiveReboundTargetEnrichmentConfig(radius=1),
    )

    downtrend_neighbor = enriched.loc[enriched["datetime"] == pd.Timestamp("2026-01-07")].iloc[0]
    assert not downtrend_neighbor["is_rebound_cluster"]
    assert downtrend_neighbor["rebound_enrichment_source"] == ""


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


def test_target_scope_audit_allows_fold_local_train_rows_without_eval_overlap():
    targets = pd.DataFrame(
        {
            "datetime": ["2020-01-02", "2020-01-03"],
            "window": ["wf_2022_train", "wf_2022_train"],
            "target_scope": ["fold_local", "fold_local"],
        }
    )
    folds = [
        {
            "name": "wf_2022_train",
            "train_start": "2020-01-01",
            "train_end": "2020-12-31",
            "eval_start": "2022-01-01",
            "eval_end": "2022-12-31",
        }
    ]

    audit = audit_state_objective_target_scope(targets, folds=folds)
    summary = summarize_state_objective_target_scope_audit(audit)

    row = audit.iloc[0]
    assert bool(row["leakage_gate"]) is True
    assert row["eval_overlap_rows"] == 0
    assert summary["decision"] == "state_objective_targets_training_ready"


def test_target_scope_audit_blocks_global_targets_that_touch_eval_dates():
    targets = pd.DataFrame(
        {
            "datetime": ["2020-01-02", "2022-03-01"],
            "window": ["strict_oos", "strict_oos"],
        }
    )
    folds = [
        {
            "name": "wf_2022_train",
            "train_start": "2020-01-01",
            "train_end": "2020-12-31",
            "eval_start": "2022-01-01",
            "eval_end": "2022-12-31",
        }
    ]

    audit = audit_state_objective_target_scope(targets, folds=folds)
    summary = summarize_state_objective_target_scope_audit(audit)

    row = audit.iloc[0]
    assert bool(row["leakage_gate"]) is False
    assert row["eval_overlap_rows"] == 1
    assert row["scope_issue"] == "global_or_unspecified_target_scope"
    assert summary["decision"] == "state_objective_targets_leakage_guard_no_training"
    assert summary["next_mechanism"] == "regenerate_fold_local_state_objective_targets"


def test_target_scope_audit_matches_fold_local_targets_to_their_own_fold_only():
    targets = pd.DataFrame(
        {
            "datetime": ["2020-01-02", "2020-01-03"],
            "window": ["wf_2022_train", "wf_2022_train"],
            "fold": ["wf_2022_train", "wf_2022_train"],
            "target_scope": ["fold_local", "fold_local"],
        }
    )
    folds = [
        {
            "name": "wf_2022_train",
            "train_start": "2020-01-01",
            "train_end": "2020-12-31",
            "eval_start": "2022-01-01",
            "eval_end": "2022-12-31",
        },
        {
            "name": "wf_2023_2024_train",
            "train_start": "2020-01-01",
            "train_end": "2022-12-31",
            "eval_start": "2023-01-01",
            "eval_end": "2024-12-31",
        },
    ]

    audit = audit_state_objective_target_scope(targets, folds=folds)

    assert audit["fold"].tolist() == ["wf_2022_train"]
    assert bool(audit.iloc[0]["leakage_gate"]) is True


def test_target_lift_audit_blocks_rebound_rows_erased_by_downtrend_cap():
    targets = pd.DataFrame(
        {
            "datetime": ["2020-03-18", "2020-03-19", "2020-07-01"],
            "fold": ["wf_2022_train", "wf_2022_train", "wf_2022_train"],
            "teacher_position": [0.15, 0.60, 0.40],
            "rebound_target_position": [0.35, 0.65, 0.40],
            "risk_target_position": [0.15, 0.25, 0.40],
            "objective_weight": [10.0, 12.0, 1.0],
            "is_rebound_cluster": [True, True, False],
            "is_downtrend_state": [True, True, False],
        }
    )

    audit = audit_state_objective_target_lift(targets, group_cols=("fold",))
    summary = summarize_state_objective_target_lift_audit(audit)

    row = audit.iloc[0]
    assert row["rebound_rows"] == 2
    assert row["current_positive_rebound_lift_rows"] == 0
    assert row["candidate_release_positive_lift_rows"] == 2
    assert bool(row["lift_semantics_gate"]) is False
    assert summary["decision"] == "state_objective_target_lift_blocked_no_training"
    assert summary["training_allowed"] is False
    assert summary["next_mechanism"] == "revise_state_objective_target_semantics_before_weighted_bc"


def test_target_lift_audit_allows_current_targets_with_effective_rebound_lift():
    targets = pd.DataFrame(
        {
            "datetime": ["2020-03-18", "2020-03-19"],
            "fold": ["wf_2022_train", "wf_2022_train"],
            "teacher_position": [0.15, 0.60],
            "rebound_target_position": [0.35, 0.60],
            "risk_target_position": [0.15, 0.60],
            "objective_weight": [10.0, 12.0],
            "is_rebound_cluster": [True, True],
            "is_downtrend_state": [False, False],
        }
    )

    audit = audit_state_objective_target_lift(
        targets,
        target_position_priority="rebound",
        group_cols=("fold",),
    )
    summary = summarize_state_objective_target_lift_audit(audit)

    assert audit.iloc[0]["current_positive_rebound_lift_rows"] == 1
    assert bool(audit.iloc[0]["lift_semantics_gate"]) is True
    assert summary["decision"] == "state_objective_target_lift_training_ready"
    assert summary["training_allowed"] is True


def test_sparse_release_gate_revises_risk_target_while_preserving_baseline_lift():
    targets = pd.DataFrame(
        {
            "datetime": ["2020-03-18", "2020-03-19", "2020-07-01"],
            "fold": ["wf_2022_train", "wf_2022_train", "wf_2022_train"],
            "teacher_position": [0.15, 0.60, 0.40],
            "rebound_target_position": [0.35, 0.65, 0.40],
            "risk_target_position": [0.15, 0.25, 0.40],
            "objective_weight": [10.0, 12.0, 1.0],
            "is_rebound_cluster": [True, True, False],
            "is_downtrend_state": [True, True, False],
        }
    )
    release_scores = pd.DataFrame(
        {
            "datetime": ["2020-03-18", "2020-03-19", "2020-07-01"],
            "feature_set": ["soxx_spread_oscillation"] * 3,
            "threshold": [0.75, 0.75, 0.75],
            "predicted_release": [True, False, True],
        }
    )

    revised = apply_state_objective_sparse_release_gate(
        targets,
        release_scores,
        feature_set="soxx_spread_oscillation",
        threshold=0.75,
    )
    audit = audit_state_objective_target_lift(
        revised,
        group_cols=("fold",),
        baseline_risk_position_col="risk_target_position_original",
    )
    summary = summarize_state_objective_target_lift_audit(audit)

    by_date = revised.set_index("datetime")
    assert bool(by_date.loc["2020-03-18", "release_gate_state"]) is True
    assert by_date.loc["2020-03-18", "risk_target_position_original"] == 0.15
    assert by_date.loc["2020-03-18", "risk_target_position"] == 0.35
    assert bool(by_date.loc["2020-03-19", "release_gate_state"]) is False
    assert by_date.loc["2020-03-19", "risk_target_position"] == 0.25
    assert bool(by_date.loc["2020-07-01", "release_gate_state"]) is False
    assert audit.iloc[0]["current_positive_rebound_lift_rows"] == 1
    assert audit.iloc[0]["candidate_release_positive_lift_rows"] == 2
    assert summary["decision"] == "state_objective_target_lift_training_ready"


def test_joint_target_contract_preserves_sparse_release_lift_under_downtrend_cap():
    release_targets = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-06-01"]),
            "fold": ["wf_2022_train"],
            "target_scope": ["fold_local"],
            "teacher_position": [0.20],
            "rebound_target_position": [0.45],
            "risk_target_position_original": [0.25],
            "risk_target_position": [0.45],
            "is_rebound_cluster": [True],
            "is_downtrend_state": [True],
            "release_gate_state": [True],
            "objective_weight": [3.0],
        }
    )
    drawdown_targets = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-06-01"]),
            "fold": ["wf_2022_train"],
            "target_scope": ["fold_local"],
            "teacher_position": [0.50],
            "rebound_target_position": [0.50],
            "risk_target_position": [0.25],
            "is_rebound_cluster": [False],
            "is_downtrend_state": [True],
            "drawdown_control_state": [True],
            "objective_weight": [5.0],
        }
    )
    folds = [
        {
            "name": "wf_2022_train",
            "train_start": "2020-01-01",
            "train_end": "2020-12-31",
            "eval_start": "2022-01-01",
            "eval_end": "2022-12-31",
        }
    ]

    joint = build_fold_local_joint_target_contract(
        release_targets,
        drawdown_targets,
        folds=folds,
    )

    assert len(joint) == 1
    row = joint.iloc[0]
    assert row["fold"] == "wf_2022_train"
    assert row["target_scope"] == "fold_local"
    assert bool(row["is_rebound_cluster"])
    assert bool(row["is_downtrend_state"])
    assert bool(row["release_gate_state"])
    assert bool(row["drawdown_control_state"])
    assert row["risk_target_position_original"] == 0.25
    assert row["risk_target_position"] == 0.45
    assert row["rebound_target_position"] == 0.50
    assert row["source_component_count"] == 2

    lift_summary = summarize_state_objective_target_lift_audit(
        audit_state_objective_target_lift(
            joint,
            group_cols=("fold",),
            baseline_risk_position_col="risk_target_position_original",
        )
    )
    assert lift_summary["decision"] == "state_objective_target_lift_training_ready"
    assert lift_summary["current_positive_rebound_lift_rows"] == 1


def test_joint_target_contract_projects_auxiliary_train_rows_without_eval_overlap():
    primary = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02"]),
            "fold": ["wf_2022_train"],
            "target_scope": ["fold_local"],
            "teacher_position": [0.20],
            "rebound_target_position": [0.35],
            "risk_target_position": [0.35],
            "risk_target_position_original": [0.20],
            "is_rebound_cluster": [True],
            "is_downtrend_state": [True],
            "release_gate_state": [True],
        }
    )
    auxiliary = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2019-12-31", "2020-06-01", "2024-02-01"]),
            "fold": ["wf_2022_train", "wf_2022_train", "wf_2022_train"],
            "target_scope": ["fold_local", "fold_local", "fold_local"],
            "teacher_position": [0.45, 0.55, 0.70],
            "rebound_target_position": [0.45, 0.55, 0.70],
            "risk_target_position": [0.25, 0.25, 0.25],
            "is_rebound_cluster": [False, False, False],
            "is_downtrend_state": [True, True, True],
            "drawdown_control_state": [True, True, True],
        }
    )
    folds = [
        {
            "name": "wf_2022_train",
            "train_start": "2020-01-01",
            "train_end": "2020-12-31",
            "eval_start": "2022-01-01",
            "eval_end": "2022-12-31",
        },
        {
            "name": "wf_2023_2024_train",
            "train_start": "2020-01-01",
            "train_end": "2022-12-31",
            "eval_start": "2024-01-01",
            "eval_end": "2024-12-31",
        },
    ]

    joint = build_fold_local_joint_target_contract(primary, auxiliary, folds=folds)

    by_fold = joint.groupby("fold")["datetime"].apply(lambda s: set(pd.to_datetime(s).dt.strftime("%Y-%m-%d")))
    assert "2019-12-31" not in by_fold["wf_2022_train"]
    assert "2020-06-01" in by_fold["wf_2023_2024_train"]
    assert "2024-02-01" not in by_fold["wf_2023_2024_train"]

    scope_summary = summarize_state_objective_target_scope_audit(
        audit_state_objective_target_scope(joint, folds=folds)
    )
    assert scope_summary["decision"] == "state_objective_targets_training_ready"
    assert scope_summary["eval_overlap_rows"] == 0


def test_trend_participation_target_gate_lifts_guarded_train_rows_without_downtrend_conflict():
    dates = pd.date_range("2020-01-02", periods=4, freq="B")
    targets = pd.DataFrame(
        {
            "datetime": dates,
            "fold": ["wf_2023_2024_train"] * 4,
            "window": ["wf_2023_2024_train"] * 4,
            "target_scope": ["fold_local"] * 4,
            "teacher_position": [0.18, 0.22, 0.18, 0.18],
            "rebound_target_position": [0.18, 0.22, 0.18, 0.18],
            "risk_target_position": [0.18, 0.22, 0.18, 0.18],
            "objective_weight": [1.0, 1.0, 1.0, 1.0],
            "is_rebound_cluster": [False, False, False, False],
            "is_downtrend_state": [False, False, True, False],
        }
    )
    features = pd.DataFrame(
        {
            "datetime": dates,
            "anchor_gap_180": [0.40, 0.72, 0.80, 0.78],
            "mom_63": [0.20, 0.55, 0.60, 0.58],
            "trend_slope_63": [0.04, 0.15, 0.16, 0.14],
            "trend_gap_63": [0.03, 0.20, 0.22, 0.21],
            "vix_percentile_252": [0.30, 0.28, 0.25, 0.72],
            "realized_vol_63": [0.50, 0.55, 0.52, 0.50],
            "drawdown_63": [-0.05, -0.08, -0.06, -0.05],
        }
    )

    result = apply_trend_participation_target_gate(
        targets,
        features,
        TrendParticipationTargetConfig(
            enabled=True,
            min_target_position=0.55,
            max_target_position=0.65,
            max_teacher_position=0.35,
            min_anchor_gap_180=0.60,
            min_mom_63=0.50,
            min_trend_slope_63=0.10,
            min_trend_gap_63=0.10,
            max_vix_percentile_252=0.40,
            max_realized_vol_63=0.70,
            min_drawdown_63=-0.20,
            objective_weight_bonus=4.0,
            exclude_downtrend=True,
        ),
    )

    assert result["trend_participation_target_state"].tolist() == [False, True, False, False]
    assert result["is_rebound_cluster"].tolist() == [False, True, False, False]
    assert result["rebound_target_position"].round(4).tolist() == [0.18, 0.55, 0.18, 0.18]
    assert result["risk_target_position_original"].round(4).tolist() == [0.18, 0.22, 0.18, 0.18]
    assert result["risk_target_position"].round(4).tolist() == [0.18, 0.22, 0.18, 0.18]
    assert result["rebound_participation_weight"].round(4).tolist() == [0.0, 4.0, 0.0, 0.0]

    lift = audit_state_objective_target_lift(
        result,
        group_cols=("fold",),
        baseline_risk_position_col="risk_target_position_original",
    )
    assert lift.iloc[0]["current_positive_rebound_lift_rows"] == 1


def test_bull_reentry_participation_gate_lifts_fold_local_buy_rows_without_eval_or_risk_leakage():
    targets = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2020-03-24",
                    "2020-03-25",
                    "2020-04-01",
                    "2022-01-03",
                    "2020-04-03",
                    "2020-04-06",
                ]
            ),
            "fold": ["wf_2020_train"] * 6,
            "window": ["wf_2020_train"] * 6,
            "target_scope": ["fold_local", "fold_local", "fold_local", "fold_local", "fold_local", "global"],
            "train_start": ["2020-01-01"] * 6,
            "train_end": ["2020-12-31"] * 6,
            "eval_start": ["2022-01-01"] * 6,
            "eval_end": ["2022-12-31"] * 6,
            "action": ["BUY", "HOLD", "HOLD", "BUY", "BUY", "BUY"],
            "teacher_position": [0.20, 0.25, 0.35, 0.25, 0.50, 0.20],
            "rebound_target_position": [0.20, 0.25, 0.35, 0.25, 0.50, 0.20],
            "risk_target_position_original": [0.20, 0.25, 0.35, 0.25, 0.50, 0.20],
            "risk_target_position": [0.20, 0.25, 0.35, 0.25, 0.50, 0.20],
            "objective_weight": [1.0, 1.5, 2.0, 1.0, 5.0, 1.0],
            "rebound_participation_weight": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "is_rebound_cluster": [False, False, False, False, True, False],
            "is_downtrend_state": [False, False, False, False, True, False],
            "drawdown_control_state": [False, False, False, False, True, False],
            "release_gate_state": [False, False, True, False, True, False],
        }
    )

    result = apply_bull_reentry_participation_target_gate(
        targets,
        BullReentryParticipationTargetConfig(
            enabled=True,
            min_target_position=0.70,
            max_target_position=0.85,
            objective_weight_bonus=5.0,
            forward_hold_days=2,
            include_release_gate=True,
            exclude_downtrend=True,
            exclude_drawdown_control=True,
        ),
    )

    by_date = result.set_index("datetime")
    lifted_buy = by_date.loc[pd.Timestamp("2020-03-24")]
    lifted_continuation = by_date.loc[pd.Timestamp("2020-03-25")]
    lifted_release = by_date.loc[pd.Timestamp("2020-04-01")]
    eval_row = by_date.loc[pd.Timestamp("2022-01-03")]
    risk_row = by_date.loc[pd.Timestamp("2020-04-03")]
    global_row = by_date.loc[pd.Timestamp("2020-04-06")]

    assert bool(lifted_buy["bull_reentry_target_state"]) is True
    assert lifted_buy["risk_target_position_original"] == 0.20
    assert lifted_buy["risk_target_position"] == 0.70
    assert lifted_buy["rebound_target_position"] == 0.70
    assert round(float(lifted_buy["bull_reentry_target_lift"]), 4) == 0.50
    assert lifted_buy["objective_weight"] == 6.0
    assert lifted_buy["rebound_participation_weight"] == 5.0
    assert lifted_buy["bull_reentry_target_reason"] == "fold_local_bull_reentry_participation"

    assert bool(lifted_continuation["bull_reentry_target_state"]) is True
    assert lifted_continuation["risk_target_position"] == 0.70
    assert lifted_continuation["objective_weight"] == 6.0
    assert bool(lifted_release["bull_reentry_target_state"]) is True
    assert lifted_release["risk_target_position"] == 0.70
    assert bool(eval_row["bull_reentry_target_state"]) is False
    assert eval_row["risk_target_position"] == 0.25
    assert bool(risk_row["bull_reentry_target_state"]) is False
    assert risk_row["risk_target_position"] == 0.50
    assert bool(global_row["bull_reentry_target_state"]) is False
    assert global_row["risk_target_position"] == 0.20


def test_trend_participation_target_gate_requires_configured_features():
    targets = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02"]),
            "teacher_position": [0.18],
            "rebound_target_position": [0.18],
            "risk_target_position": [0.18],
            "is_rebound_cluster": [False],
            "is_downtrend_state": [False],
        }
    )
    features = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02"]),
            "mom_63": [0.55],
            "trend_slope_63": [0.15],
            "trend_gap_63": [0.20],
            "vix_percentile_252": [0.28],
            "realized_vol_63": [0.55],
            "drawdown_63": [-0.08],
        }
    )

    try:
        apply_trend_participation_target_gate(
            targets,
            features,
            TrendParticipationTargetConfig(enabled=True, min_anchor_gap_180=0.60),
        )
    except ValueError as exc:
        assert "anchor_gap_180" in str(exc)
    else:
        raise AssertionError("missing anchor_gap_180 did not raise ValueError")

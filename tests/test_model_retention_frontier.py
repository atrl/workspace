from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_retention_frontier_prioritizes_training_ready_soxlbacktest_contract():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_prioritizes_training_ready_soxlbacktest_contract(tmp_path)


def _run_retention_frontier_prioritizes_training_ready_soxlbacktest_contract(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "target_ready" / "summary.json",
        {
            "experiment": "target_ready",
            "decision": "wf2020_soxlbacktest_target_contract_ready_for_bc_screen_no_promotion",
            "promotion_allowed": False,
            "reference_baseline": "soxlbacktest_qlib_buysell_reference",
            "target_contract_gate": True,
            "bc_readiness_gate": True,
            "bc_screen_allowed_next": True,
            "wf_2020_coverage": True,
            "next_mechanism": "run_three_fold_bc_slice_screen_before_any_ppo",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "checkpoint_partial" / "summary.json",
        {
            "experiment": "checkpoint_partial",
            "decision": "three_fold_checkpoint_metric_replay_partial_pass_no_promotion",
            "promotion_allowed": False,
            "selected_by_fold": {
                "wf_2020": {
                    "selected_max_dd": 14.98,
                    "selected_relative_return": -292.75,
                    "selected_candidate_gate": False,
                },
                "wf_2022": {
                    "selected_max_dd": 18.8,
                    "selected_relative_return": 65.86,
                    "selected_candidate_gate": True,
                },
                "wf_2023_2024": {
                    "selected_max_dd": 8.96,
                    "selected_relative_return": 35.57,
                    "selected_candidate_gate": True,
                },
            },
            "target_relative_return_pp": 30.0,
            "next_mechanism": "inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "official_replay" / "summary.json",
        {
            "selection": {
                "experiment": "official_replay",
                "decision": "budget_usage_bc_official_replay_no_promotion",
                "promotion_allowed": False,
                "ytd_return_pct": 27.18,
                "wf_max_dd_pct": {
                    "wf_2020": 21.35,
                    "wf_2022": 14.8,
                    "wf_2023_2024": 15.51,
                },
                "wf_return_pct": {
                    "wf_2020": 16.24,
                    "wf_2022": -12.14,
                    "wf_2023_2024": 24.28,
                },
                "next_mechanism": "run_fold_local_ppo_only_if_ytd_and_wf_gap_is_not_structural_else_revise_targets",
            }
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert result["summary"]["recommended_next_source"] == "target_ready"
    assert result["summary"]["recommended_next_action"] == "run_three_fold_bc_slice_screen_before_any_ppo"
    assert by_experiment["target_ready"]["primary_blocker"] == "training_contract_ready"
    assert by_experiment["target_ready"]["frontier_rank"] == 1
    assert by_experiment["checkpoint_partial"]["failed_wf_windows"] == "wf_2020"
    assert by_experiment["checkpoint_partial"]["relative_return_gap_to_target_pp"] == 322.75
    assert by_experiment["official_replay"]["primary_blocker"] == "joint_drawdown_return_gap"
    assert by_experiment["official_replay"]["ytd_gap_to_best_pct"] == 185.96


def test_write_retention_frontier_outputs_writes_csv_summary_and_conclusion():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_write_retention_frontier_outputs_writes_csv_summary_and_conclusion(tmp_path)


def _run_write_retention_frontier_outputs_writes_csv_summary_and_conclusion(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import write_retention_frontier_outputs

    output = tmp_path / "frontier"
    write_retention_frontier_outputs(
        {
            "summary": {
                "recommended_next_source": "target_ready",
                "recommended_next_action": "run_three_fold_bc_slice_screen_before_any_ppo.",
                "rows_scanned": 1,
                "completed_handoff_count": 2,
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
            },
            "rows": [
                {
                    "frontier_rank": 1,
                    "experiment": "target_ready",
                    "decision": "wf2020_soxlbacktest_target_contract_ready_for_bc_screen_no_promotion",
                    "primary_blocker": "training_contract_ready",
                    "next_action": "run_three_fold_bc_slice_screen_before_any_ppo",
                }
            ],
        },
        output_dir=output,
    )

    frontier = pd.read_csv(output / "retention_frontier.csv")
    assert frontier.iloc[0]["experiment"] == "target_ready"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["recommended_next_source"] == "target_ready"
    conclusion = (output / "conclusion.md").read_text(encoding="utf-8")
    assert "target_ready" in conclusion
    assert "Completed handoffs: `2`" in conclusion
    assert "run_three_fold_bc_slice_screen_before_any_ppo." in conclusion
    assert "run_three_fold_bc_slice_screen_before_any_ppo.`." not in conclusion


def test_retention_frontier_does_not_scan_its_own_output_card():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_does_not_scan_its_own_output_card(tmp_path)


def _run_retention_frontier_does_not_scan_its_own_output_card(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_frontier_v1" / "summary.json",
        {
            "summary": {
                "recommended_next_source": "stale_self",
                "recommended_next_action": "do_not_recommend_self",
            },
            "rows": [],
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "target_ready" / "summary.json",
        {
            "experiment": "target_ready",
            "decision": "wf2020_soxlbacktest_target_contract_ready_for_bc_screen_no_promotion",
            "promotion_allowed": False,
            "target_contract_gate": True,
            "bc_readiness_gate": True,
            "bc_screen_allowed_next": True,
            "next_mechanism": "run_three_fold_bc_slice_screen_before_any_ppo",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)

    assert result["summary"]["rows_scanned"] == 1
    assert [row["experiment"] for row in result["rows"]] == ["target_ready"]


def test_retention_frontier_preserves_explicit_diagnostic_blocker_detail():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_preserves_explicit_diagnostic_blocker_detail(tmp_path)


def _run_retention_frontier_preserves_explicit_diagnostic_blocker_detail(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "bc_replay" / "summary.json",
        {
            "experiment": "bc_replay",
            "decision": "partial_missing_wf_targets_and_slice_failed_no_training",
            "promotion_allowed": False,
            "primary_blocker": "missing_nonleaking_wf2020_targets_and_wf2023_slice_fit_failed",
            "next_mechanism": "repair_nonleaking_wf2020_target_contract_and_retune_wf2023_slice",
            "metrics": [
                {
                    "window": "wf_2022",
                    "max_dd_pct": 18.9,
                    "relative_return_pct": 62.8,
                },
                {
                    "window": "wf_2023_2024",
                    "max_dd_pct": 9.7,
                    "relative_return_pct": 39.0,
                },
            ],
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    row = result["rows"][0]

    assert row["experiment"] == "bc_replay"
    assert row["missing_wf_windows"] == "wf_2020"
    assert row["primary_blocker"] == "missing_nonleaking_wf2020_targets_and_wf2023_slice_fit_failed"
    assert result["summary"]["recommended_primary_blocker"] == (
        "missing_nonleaking_wf2020_targets_and_wf2023_slice_fit_failed"
    )


def test_retention_frontier_advances_past_completed_training_contract_handoff():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_advances_past_completed_training_contract_handoff(tmp_path)


def _run_retention_frontier_advances_past_completed_training_contract_handoff(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "target_ready" / "summary.json",
        {
            "experiment": "target_ready",
            "decision": "wf2020_soxlbacktest_target_contract_ready_for_bc_screen_no_promotion",
            "promotion_allowed": False,
            "target_contract_gate": True,
            "bc_readiness_gate": True,
            "bc_screen_allowed_next": True,
            "next_mechanism": "run_three_fold_bc_slice_screen_before_any_ppo",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "bc_done" / "summary.json",
        {
            "experiment": "bc_done",
            "decision": "three_fold_bc_slice_ready_for_checkpoint_screen_no_promotion",
            "promotion_allowed": False,
            "source_decisions": {
                "target_ready": "wf2020_soxlbacktest_target_contract_ready_for_bc_screen_no_promotion"
            },
            "next_mechanism": "run_walkforward_checkpoint_or_ppo_screen_with_bc_warmstart_against_soxlbacktest_reference_and_drawdown_drift_stop_rules",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "checkpoint_partial" / "summary.json",
        {
            "experiment": "checkpoint_partial",
            "decision": "three_fold_checkpoint_metric_replay_partial_pass_no_promotion",
            "promotion_allowed": False,
            "direct_comparison_baseline": "bc_done",
            "selected_by_fold": {
                "wf_2020": {
                    "selected_max_dd": 14.98,
                    "selected_relative_return": -292.75,
                    "selected_candidate_gate": False,
                },
                "wf_2022": {
                    "selected_max_dd": 18.8,
                    "selected_relative_return": 65.86,
                    "selected_candidate_gate": True,
                },
                "wf_2023_2024": {
                    "selected_max_dd": 8.96,
                    "selected_relative_return": 35.57,
                    "selected_candidate_gate": True,
                },
            },
            "target_relative_return_pp": 30.0,
            "next_mechanism": "inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert result["summary"]["recommended_next_source"] == "checkpoint_partial"
    assert result["summary"]["recommended_next_action"] == "inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training"
    assert by_experiment["target_ready"]["handoff_completed_by"] == "bc_done"
    assert by_experiment["target_ready"]["primary_blocker"] == "completed_handoff"
    assert by_experiment["bc_done"]["handoff_completed_by"] == "checkpoint_partial"
    assert by_experiment["bc_done"]["primary_blocker"] == "completed_handoff"
    assert by_experiment["checkpoint_partial"]["frontier_rank"] == 1
    assert by_experiment["checkpoint_partial"]["primary_blocker"] == "walkforward_relative_return_gap"


def test_retention_frontier_uses_sidecar_manifest_paths_as_downstream_handoffs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_sidecar_manifest_paths_as_downstream_handoffs(tmp_path)


def test_retention_frontier_uses_source_card_sidecar_columns_as_downstream_handoffs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_source_card_sidecar_columns_as_downstream_handoffs(tmp_path)


def test_retention_frontier_uses_source_paths_as_downstream_handoffs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_source_paths_as_downstream_handoffs(tmp_path)


def test_retention_frontier_uses_source_guard_experiment_as_downstream_handoff():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_source_guard_experiment_as_downstream_handoff(tmp_path)


def test_retention_frontier_uses_nested_source_experiment_as_downstream_handoff():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_nested_source_experiment_as_downstream_handoff(tmp_path)


def test_retention_frontier_uses_source_diagnostic_lineage_as_downstream_handoff():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_source_diagnostic_lineage_as_downstream_handoff(tmp_path)


def test_retention_frontier_uses_handoff_decision_as_downstream_completion():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_handoff_decision_as_downstream_completion(tmp_path)


def test_retention_frontier_does_not_treat_audit_frontier_top_candidates_as_completion():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_does_not_treat_audit_frontier_top_candidates_as_completion(tmp_path)


def _run_retention_frontier_uses_sidecar_manifest_paths_as_downstream_handoffs(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "child_screen" / "summary.json",
        {
            "experiment": "child_screen",
            "decision": "official_child_screen_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "aggregate_child_mechanism",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "aggregate_screen" / "summary.json",
        {
            "experiment": "aggregate_screen",
            "decision": "aggregate_ready_for_next_target_revision",
            "promotion_allowed": False,
            "selected_by_fold": {
                "wf_2020": {"selected_max_dd": 14.0, "selected_relative_return": -20.0},
                "wf_2022": {"selected_max_dd": 18.0, "selected_relative_return": 40.0},
                "wf_2023_2024": {"selected_max_dd": 10.0, "selected_relative_return": 35.0},
            },
            "target_relative_return_pp": 30.0,
            "next_mechanism": "revise_target_from_aggregate",
        },
    )
    _write_csv(
        tmp_path / "runtime" / "model_selection" / "card" / "aggregate_screen" / "source_manifest.csv",
        [
            {
                "source_label": "child",
                "path": "runtime/model_selection/card/child_screen/official_replay_metrics.csv",
                "rows": 2,
            }
        ],
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert result["summary"]["recommended_next_source"] == "aggregate_screen"
    assert by_experiment["child_screen"]["primary_blocker"] == "completed_handoff"
    assert by_experiment["child_screen"]["handoff_completed_by"] == "aggregate_screen"
    assert by_experiment["aggregate_screen"]["frontier_rank"] == 1


def _run_retention_frontier_uses_source_diagnostic_lineage_as_downstream_handoff(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "failed_fold_diagnostic" / "summary.json",
        {
            "experiment": "failed_fold_diagnostic",
            "decision": "requires_target_revision_no_promotion",
            "promotion_allowed": False,
            "metrics": [
                {"window": "wf_2020", "max_dd_pct": 20.9, "relative_return_pct": 5.0},
                {"window": "wf_2022", "max_dd_pct": 15.6, "relative_return_pct": 71.0},
                {"window": "wf_2023_2024", "max_dd_pct": 11.7, "relative_return_pct": 12.0},
            ],
            "next_mechanism": "build_revision_then_rerun_checkpoint",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "target_revision" / "summary.json",
        {
            "experiment": "target_revision",
            "decision": "target_revision_ready_no_promotion",
            "promotion_allowed": False,
            "source_failed_fold_diagnostic": "failed_fold_diagnostic",
            "target_contract_gate": True,
            "bc_readiness_gate": True,
            "bc_screen_allowed_next": True,
            "next_mechanism": "rerun_checkpoint_metric_replay",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["failed_fold_diagnostic"]["handoff_completed_by"] == "target_revision"
    assert by_experiment["failed_fold_diagnostic"]["primary_blocker"] == "completed_handoff"
    assert result["summary"]["recommended_next_source"] == "target_revision"


def _run_retention_frontier_uses_handoff_decision_as_downstream_completion(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "official_handoff" / "summary.json",
        {
            "experiment": "official_handoff",
            "decision": "official_candidate_pool_no_full_gate_training_objective_handoff",
            "promotion_allowed": False,
            "next_mechanism": "train_rebound_participation_objective",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "bull_reentry_objective" / "summary.json",
        {
            "experiment": "bull_reentry_objective",
            "decision": "bull_reentry_participation_objective_partial_improvement_no_promotion",
            "promotion_allowed": False,
            "handoff": {
                "decision": "train_rebound_participation_objective",
                "promotion": "blocked",
            },
            "metrics": [
                {"window": "wf_2020", "max_dd_pct": 19.3, "relative_return_pct": -0.4},
                {"window": "wf_2022", "max_dd_pct": 13.2, "relative_return_pct": 52.0},
                {"window": "wf_2023_2024", "max_dd_pct": 12.5, "relative_return_pct": 40.0},
            ],
            "next_mechanism": "convert_best_participation_pattern_to_checkpoint_or_reward_objective",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["official_handoff"]["handoff_completed_by"] == "bull_reentry_objective"
    assert by_experiment["official_handoff"]["primary_blocker"] == "completed_handoff"
    assert result["summary"]["recommended_next_source"] == "bull_reentry_objective"


def _run_retention_frontier_does_not_treat_audit_frontier_top_candidates_as_completion(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "state_head_revision" / "summary.json",
        {
            "experiment": "state_head_revision",
            "decision": "state_head_revision_handoff_no_training_no_actor_retention",
            "promotion_allowed": False,
            "metrics": [
                {"window": "wf_2020", "max_dd_pct": 26.3, "relative_return_pct": 27.8},
                {"window": "wf_2022", "max_dd_pct": 15.4, "relative_return_pct": 72.3},
                {"window": "wf_2023_2024", "max_dd_pct": 30.5, "relative_return_pct": -16.0},
            ],
            "target_relative_return_pp": 30.0,
            "next_mechanism": "build_risk_gated_state_head_target_then_rerun_official_replay",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "current_audit" / "summary.json",
        {
            "experiment": "current_audit",
            "decision": "audit_current_frontier_no_training",
            "promotion_allowed": False,
        },
    )
    _write_csv(
        tmp_path / "runtime" / "model_selection" / "card" / "current_audit" / "frontier_top_candidates.csv",
        [
            {
                "frontier_rank": 1,
                "experiment": "state_head_revision",
                "source_path": "runtime/model_selection/card/state_head_revision/summary.json",
            }
        ],
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["state_head_revision"]["handoff_completed_by"] == ""
    assert by_experiment["state_head_revision"]["primary_blocker"] == "walkforward_relative_return_gap"
    assert result["summary"]["recommended_next_source"] == "state_head_revision"


def _run_retention_frontier_uses_source_card_sidecar_columns_as_downstream_handoffs(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "metric_handoff" / "summary.json",
        {
            "experiment": "metric_handoff",
            "decision": "metric_replay_handoff_completed_no_promotion",
            "promotion_allowed": False,
            "selected_by_fold": {
                "wf_2020": {"selected_max_dd": 14.0, "selected_relative_return": -10.0},
                "wf_2022": {"selected_max_dd": 18.0, "selected_relative_return": 40.0},
                "wf_2023_2024": {"selected_max_dd": 10.0, "selected_relative_return": 35.0},
            },
            "target_relative_return_pp": 30.0,
            "next_mechanism": "inspect_metric_gap",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "fold_attribution" / "summary.json",
        {
            "experiment": "fold_attribution",
            "decision": "fold_frontier_gap_diagnosed",
            "promotion_allowed": False,
            "next_mechanism": "export_checkpoint_daily_paths",
        },
    )
    _write_csv(
        tmp_path / "runtime" / "model_selection" / "card" / "fold_attribution" / "normalized_candidate_frontier.csv",
        [
            {
                "source_card": "metric_handoff",
                "source_file": "runtime/model_selection/card/metric_handoff/checkpoint_candidate_metrics.csv",
                "candidate": "ppo_retained_iter_4",
            }
        ],
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["metric_handoff"]["primary_blocker"] == "completed_handoff"
    assert by_experiment["metric_handoff"]["handoff_completed_by"] == "fold_attribution"


def _run_retention_frontier_uses_source_paths_as_downstream_handoffs(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "metric_handoff" / "summary.json",
        {
            "experiment": "metric_handoff",
            "decision": "metric_replay_handoff_completed_no_promotion",
            "promotion_allowed": False,
            "selected_by_fold": {
                "wf_2020": {"selected_max_dd": 14.0, "selected_relative_return": -292.0},
                "wf_2022": {"selected_max_dd": 18.0, "selected_relative_return": 65.0},
                "wf_2023_2024": {"selected_max_dd": 9.0, "selected_relative_return": 35.0},
            },
            "target_relative_return_pp": 30.0,
            "next_mechanism": "inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "target_readiness" / "summary.json",
        {
            "decision": "target_revision_before_retained_actor_training",
            "promotion_allowed": False,
            "source_paths": [
                "runtime/model_selection/card/metric_handoff/summary.json",
                "dashboard_data/model_replay_freshness.json",
            ],
            "recommended_next_source": "wf2020_failed_fold_audit",
            "recommended_next_action": "build_fold_local_target",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "wf2020_failed_fold_audit" / "summary.json",
        {
            "experiment": "wf2020_failed_fold_audit",
            "decision": "wf2020_failed_fold_requires_target_revision_no_promotion",
            "promotion_allowed": False,
            "next_training_allowed_after_target_revision": True,
            "next_mechanism": "build_fold_local_target",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["metric_handoff"]["primary_blocker"] == "completed_handoff"
    assert by_experiment["metric_handoff"]["handoff_completed_by"] == "target_readiness"
    assert result["summary"]["recommended_next_source"] == "wf2020_failed_fold_audit"


def _run_retention_frontier_uses_source_guard_experiment_as_downstream_handoff(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "source_guard" / "summary.json",
        {
            "experiment": "source_guard",
            "decision": "drawdown_budget_guard_screen_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "move_drawdown_control_into_checkpoint_objective_or_state_conditioned_training",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "target_handoff" / "summary.json",
        {
            "selection": {
                "experiment": "target_handoff",
                "decision": "drawdown_control_target_handoff_no_promotion",
                "promotion_allowed": False,
                "source_guard_experiment": "source_guard",
                "next_mechanism": "train_or_checkpoint_score_state_objective_drawdown_targets",
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "checkpoint_score" / "summary.json",
        {
            "experiment": "checkpoint_score",
            "decision": "checkpoint_score_ready_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "run_fold_local_drawdown_control_bc_or_checkpoint_objective_training",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["source_guard"]["primary_blocker"] == "completed_handoff"
    assert by_experiment["source_guard"]["handoff_completed_by"] == "target_handoff"
    assert result["summary"]["recommended_next_source"] == "checkpoint_score"


def _run_retention_frontier_uses_nested_source_experiment_as_downstream_handoff(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "checkpoint_score" / "summary.json",
        {
            "experiment": "checkpoint_score",
            "decision": "checkpoint_objective_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "run_fold_local_drawdown_control_bc_or_checkpoint_objective_training",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "foldlocal_bc" / "summary.json",
        {
            "experiment": "foldlocal_bc",
            "decision": "foldlocal_bc_warmstart_ready_no_promotion",
            "promotion_allowed": False,
            "coverage": {
                "source_checkpoint_score_experiment": "checkpoint_score",
            },
            "next_mechanism": "run_foldlocal_bc_warmstart_walkforward_ppo_screen",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["checkpoint_score"]["primary_blocker"] == "completed_handoff"
    assert by_experiment["checkpoint_score"]["handoff_completed_by"] == "foldlocal_bc"


def test_retention_frontier_uses_selection_sidecar_lineage_as_downstream_handoff():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_selection_sidecar_lineage_as_downstream_handoff(tmp_path)


def test_retention_frontier_uses_top_level_selection_lineage_when_summary_exists():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_top_level_selection_lineage_when_summary_exists(tmp_path)


def test_retention_frontier_uses_reward_cards_as_downstream_handoffs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_reward_cards_as_downstream_handoffs(tmp_path)


def test_retention_frontier_uses_factor_cards_as_downstream_handoffs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_factor_cards_as_downstream_handoffs(tmp_path)


def _run_retention_frontier_uses_factor_cards_as_downstream_handoffs(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path
        / "runtime"
        / "model_selection"
        / "card"
        / "state_head_stable_rebound_label_screen_v1"
        / "summary.json",
        {
            "experiment": "state_head_stable_rebound_label_screen_v1",
            "decision": "stable_rebound_label_screen_no_candidate_no_training",
            "promotion_allowed": False,
            "training_allowed": False,
            "next_mechanism": "mine_new_rebound_state_factor_before_state_head_training",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "next_training_screen" / "summary.json",
        {
            "experiment": "next_training_screen",
            "decision": "state_head_training_screen_ready_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "train_state_head_if_factor_passes",
        },
    )
    _write_json(
        tmp_path / "runtime" / "factor" / "card" / "rebound_state_factor_mining_v1" / "summary.json",
        {
            "experiment": "rebound_state_factor_mining_v1",
            "decision": "rebound_state_factor_screen_completed_no_direct_training",
            "promotion_allowed": False,
            "source_experiment": "state_head_stable_rebound_label_screen_v1",
            "next_mechanism": "handoff_factor_result_to_state_head_screen",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert result["summary"]["rows_scanned"] == 2
    assert result["summary"]["completed_handoff_count"] == 1
    assert result["summary"]["recommended_next_source"] == "next_training_screen"
    assert by_experiment["state_head_stable_rebound_label_screen_v1"]["primary_blocker"] == "completed_handoff"
    assert (
        by_experiment["state_head_stable_rebound_label_screen_v1"]["handoff_completed_by"]
        == "rebound_state_factor_mining_v1"
    )


def _run_retention_frontier_uses_reward_cards_as_downstream_handoffs(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path
        / "runtime"
        / "model_selection"
        / "card"
        / "short_volshock_drawdown_drift_feature_audit_v1"
        / "summary.json",
        {
            "experiment": "short_volshock_drawdown_drift_feature_audit_v1",
            "decision": "short_volshock_drawdown_drift_joint_objective_needed_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "screen drawdown-drift reward/checkpoint penalty with trend-preservation term before PPO",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "next_training_screen" / "summary.json",
        {
            "experiment": "next_training_screen",
            "decision": "short_foldlocal_bc_or_ppo_training_screen_ready_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "run full fixed-split and walk-forward replay review",
        },
    )
    _write_json(
        tmp_path
        / "runtime"
        / "reward"
        / "card"
        / "short_volshock_drawdown_drift_reward_recipe_screen_v1"
        / "summary.json",
        {
            "experiment": "short_volshock_drawdown_drift_reward_recipe_screen_v1",
            "decision": "drawdown_drift_reward_recipe_ready_for_short_training_screen",
            "promotion_allowed": False,
            "training_allowed_next": True,
            "next_mechanism": "short_foldlocal_bc_or_ppo_training_screen",
        },
    )
    _write_json(
        tmp_path
        / "runtime"
        / "reward"
        / "card"
        / "short_volshock_drawdown_drift_reward_recipe_screen_v1"
        / "selection.json",
        {
            "experiment": "short_volshock_drawdown_drift_reward_recipe_screen_v1",
            "direct_comparison_baseline": [
                "short_volshock_drawdown_drift_feature_audit_v1",
            ],
            "source_decisions": {
                "drawdown_drift_audit_decision": (
                    "short_volshock_drawdown_drift_joint_objective_needed_no_promotion"
                )
            },
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert result["summary"]["rows_scanned"] == 2
    assert result["summary"]["completed_handoff_count"] == 1
    assert result["summary"]["recommended_next_source"] == "next_training_screen"
    assert (
        by_experiment["short_volshock_drawdown_drift_feature_audit_v1"]["primary_blocker"]
        == "completed_handoff"
    )
    assert (
        by_experiment["short_volshock_drawdown_drift_feature_audit_v1"]["handoff_completed_by"]
        == "short_volshock_drawdown_drift_reward_recipe_screen_v1"
    )


def _run_retention_frontier_uses_selection_sidecar_lineage_as_downstream_handoff(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "sequence_source" / "summary.json",
        {
            "experiment": "sequence_source",
            "decision": "sequence_risk_state_candidate_ready_for_target_screen_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "add_sequence_risk_state_cap_to_joint_contract_then_screen_bc_and_checkpoint_drift",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "joint_bc_screen" / "summary.json",
        {
            "experiment": "joint_bc_screen",
            "decision": "sequence_trend_joint_bc_mixed_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "inspect_failed_fold_action_distribution_then_try_drawdown_drift_reward_or_checkpoint_selector",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "joint_bc_screen" / "selection.json",
        {
            "experiment": "joint_bc_screen",
            "source_selection": {
                "sequence_target": {
                    "experiment": "cap_target_intermediate",
                    "direct_comparison_baseline": "sequence_source",
                }
            },
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["sequence_source"]["primary_blocker"] == "completed_handoff"
    assert by_experiment["sequence_source"]["handoff_completed_by"] == "joint_bc_screen"
    assert result["summary"]["recommended_next_source"] == "joint_bc_screen"


def _run_retention_frontier_uses_top_level_selection_lineage_when_summary_exists(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "alignment_screen" / "summary.json",
        {
            "experiment": "alignment_screen",
            "decision": "target_alignment_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "train_or_checkpoint_score_state_objective_targets",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "checkpoint_selection" / "summary.json",
        {
            "experiment": "checkpoint_selection",
            "decision": "checkpoint_objective_no_promotion",
            "promotion_allowed": False,
            "next_mechanism": "train_state_objective_checkpoint_or_bc",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "checkpoint_selection" / "selection.json",
        {
            "experiment": "checkpoint_selection",
            "decision": "checkpoint_objective_no_promotion",
            "direct_comparison_baseline": "alignment_screen",
            "summary": {
                "decision": "checkpoint_objective_no_promotion",
                "next_mechanism": "train_state_objective_checkpoint_or_bc",
            },
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["alignment_screen"]["primary_blocker"] == "completed_handoff"
    assert by_experiment["alignment_screen"]["handoff_completed_by"] == "checkpoint_selection"


def test_retention_frontier_recommends_actionable_row_before_actionless_diagnostic():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_recommends_actionable_row_before_actionless_diagnostic(tmp_path)


def _run_retention_frontier_recommends_actionable_row_before_actionless_diagnostic(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "aaa_actionless_diagnostic" / "summary.json",
        {
            "experiment": "aaa_actionless_diagnostic",
            "decision": "diagnostic_no_objective_pass_no_promotion",
            "promotion_allowed": False,
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "zzz_actionable_screen" / "summary.json",
        {
            "experiment": "zzz_actionable_screen",
            "decision": "screen_ready_for_next_target_revision",
            "promotion_allowed": False,
            "next_mechanism": "inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)

    assert result["summary"]["recommended_next_source"] == "zzz_actionable_screen"
    assert result["summary"]["recommended_next_action"] == (
        "inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training"
    )


def test_retention_frontier_uses_fold_specific_relative_return_targets():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_uses_fold_specific_relative_return_targets(tmp_path)


def _run_retention_frontier_uses_fold_specific_relative_return_targets(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "wf2020_calibrated" / "summary.json",
        {
            "experiment": "wf2020_calibrated",
            "decision": "wf2020_reference_floor_calibrated_no_promotion",
            "promotion_allowed": False,
            "selected_by_fold": {
                "wf_2020": {"selected_max_dd": 15.49, "selected_relative_return": -277.74},
                "wf_2022": {"selected_max_dd": 18.8, "selected_relative_return": 65.86},
                "wf_2023_2024": {"selected_max_dd": 8.96, "selected_relative_return": 35.57},
            },
            "target_relative_return_pp": 30.0,
            "target_relative_return_by_fold_pp": {
                "wf_2020": -263.73,
                "wf_2022": 30.0,
                "wf_2023_2024": 30.0,
            },
            "next_mechanism": "build_wf2020_reference_floor_target_screen",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    row = {item["experiment"]: item for item in result["rows"]}["wf2020_calibrated"]

    assert row["failed_wf_windows"] == "wf_2020"
    assert row["relative_return_gap_to_target_pp"] == 14.01
    assert row["primary_blocker"] == "walkforward_relative_return_gap"


def test_retention_frontier_prioritizes_target_revision_ready_failed_fold_audit():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_prioritizes_target_revision_ready_failed_fold_audit(tmp_path)


def test_retention_frontier_prioritizes_full_evidence_bc_failure_over_missing_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_frontier_prioritizes_full_evidence_bc_failure_over_missing_evidence(tmp_path)


def _run_retention_frontier_prioritizes_full_evidence_bc_failure_over_missing_evidence(
    tmp_path: Path,
) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "old_missing_evidence" / "summary.json",
        {
            "experiment": "old_missing_evidence",
            "decision": "old_diagnostic_no_replay_yet",
            "promotion_allowed": False,
            "next_mechanism": "build_missing_walkforward_replay",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "bc_screen_failed" / "summary.json",
        {
            "experiment": "bc_screen_failed",
            "decision": "bc_slice_failed_no_promotion",
            "promotion_allowed": False,
            "primary_blocker": "bc_candidate_or_slice_gate_failed",
            "selected_by_fold": {
                "wf_2020": {"selected_max_dd": 35.85, "selected_relative_return": -40.58},
                "wf_2022": {"selected_max_dd": 17.62, "selected_relative_return": 77.13},
                "wf_2023_2024": {"selected_max_dd": 25.53, "selected_relative_return": 20.61},
            },
            "target_relative_return_pp": 30.0,
            "next_mechanism": "repair_sparse_slice_fit_or_state_head_feature_before_ppo",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["bc_screen_failed"]["primary_blocker"] == "bc_candidate_or_slice_gate_failed"
    assert by_experiment["old_missing_evidence"]["primary_blocker"] == "missing_walkforward_evidence"
    assert result["summary"]["recommended_next_source"] == "bc_screen_failed"


def _run_retention_frontier_prioritizes_target_revision_ready_failed_fold_audit(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_frontier import scan_model_retention_frontier

    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json",
        {
            "summary": {
                "required_best_return_pct": 213.14,
                "reference_return_pct": 132.08,
                "max_walkforward_drawdown_pct": 20.0,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            }
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "metric_handoff" / "summary.json",
        {
            "experiment": "metric_handoff",
            "decision": "metric_replay_handoff_completed_no_promotion",
            "promotion_allowed": False,
            "reference_baseline": "soxlbacktest_qlib_buysell_reference",
            "selected_by_fold": {
                "wf_2020": {"selected_max_dd": 14.98, "selected_relative_return": -292.75},
                "wf_2022": {"selected_max_dd": 18.8, "selected_relative_return": 65.86},
                "wf_2023_2024": {"selected_max_dd": 8.96, "selected_relative_return": 35.57},
            },
            "target_relative_return_pp": 30.0,
            "next_mechanism": "inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training",
        },
    )
    _write_json(
        tmp_path / "runtime" / "model_selection" / "card" / "failed_fold_objective" / "summary.json",
        {
            "experiment": "failed_fold_objective",
            "decision": "wf2020_failed_fold_requires_bull_reentry_target_revision_no_promotion",
            "promotion_allowed": False,
            "reference_baseline": "soxlbacktest_qlib_buysell_reference",
            "direct_comparison_baseline": "metric_handoff",
            "failed_folds": ["wf_2020"],
            "primary_failure": "wf_2020_relative_underparticipation_after_metric_replay",
            "next_training_allowed_after_target_revision": True,
            "next_mechanism": "build_fold_local_bull_reentry_participation_target_then_rerun_three_fold_bc_checkpoint_metric_replay",
        },
    )

    result = scan_model_retention_frontier(root=tmp_path)
    by_experiment = {row["experiment"]: row for row in result["rows"]}

    assert by_experiment["metric_handoff"]["primary_blocker"] == "completed_handoff"
    assert by_experiment["failed_fold_objective"]["primary_blocker"] == "target_revision_ready"
    assert result["summary"]["recommended_next_source"] == "failed_fold_objective"
    assert result["summary"]["recommended_primary_blocker"] == "target_revision_ready"
    assert result["summary"]["recommended_next_action"] == (
        "build_fold_local_bull_reentry_participation_target_then_rerun_three_fold_bc_checkpoint_metric_replay"
    )

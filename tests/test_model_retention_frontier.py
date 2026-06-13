from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
                "recommended_next_action": "run_three_fold_bc_slice_screen_before_any_ppo",
                "rows_scanned": 1,
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
    assert "run_three_fold_bc_slice_screen_before_any_ppo" in conclusion


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

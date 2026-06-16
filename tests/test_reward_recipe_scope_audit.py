from __future__ import annotations

import pandas as pd

import src.soxl_quant.app.reward_checkpoint_screen as reward_checkpoint_screen


def test_reward_checkpoint_recipe_scope_audit_blocks_training_without_wf2020_and_foldlocal_targets():
    contract = {
        "decision": "reward_checkpoint_recipe_contract_ready",
        "selected_candidate": "teacher_short_volshock_cap05_hit2",
        "final_required_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
        "missing_final_required_windows": ["wf_2020"],
        "leakage_guard": {
            "copy_selected_daily_positions_into_training_labels": False,
            "eval_window_positions_are_diagnostic_only": True,
            "next_target_generation_must_be_fold_local": True,
            "eval_overlap_rows_required": 0,
        },
        "target_contract": {
            "reference_profile": "soxl_backtest_policy_reimplementation_v1",
            "required_best_return_pct": 213.143462,
            "final_retention_max_dd_pct": 20.0,
        },
    }
    teacher_window_coverage = pd.DataFrame(
        [
            {
                "candidate": "short_volshock_cap05_hit2",
                "present_windows": "wf_2022;wf_2023_2024",
                "missing_final_required_windows": "wf_2020",
                "final_window_coverage_gate": False,
            }
        ]
    )
    selected_fold_metrics = pd.DataFrame(
        [
            {
                "fold": "wf_2022",
                "daily_path": "runtime/.../wf_2022.csv",
                "teacher_usage": "official_replay_reference_only_not_training_label",
                "teacher_status": "pass",
                "full_gate": True,
                "official_warm_relative_return": 83.56,
                "official_warm_max_dd": 8.75,
            },
            {
                "fold": "wf_2023_2024",
                "daily_path": "runtime/.../wf_2023_2024.csv",
                "teacher_usage": "official_replay_reference_only_not_training_label",
                "teacher_status": "near_miss",
                "full_gate": False,
                "official_warm_relative_return": 29.82,
                "official_warm_max_dd": 15.11,
            },
        ]
    )

    assert hasattr(reward_checkpoint_screen, "audit_reward_checkpoint_recipe_scope")
    result = reward_checkpoint_screen.audit_reward_checkpoint_recipe_scope(
        contract,
        teacher_window_coverage=teacher_window_coverage,
        selected_fold_metrics=selected_fold_metrics,
        fold_local_target_manifest=None,
    )

    assert result["summary"]["decision"] == "recipe_scope_audit_blocks_training"
    assert result["summary"]["training_allowed"] is False
    assert result["summary"]["coverage_gate"] is False
    assert result["summary"]["scope_gate"] is False
    assert result["summary"]["missing_final_required_windows"] == ["wf_2020"]
    assert result["required_gate_rows"].loc[
        result["required_gate_rows"]["gate"].eq("consume_reward_checkpoint_recipe_in_code"),
        "status",
    ].item() == "passed"
    assert result["required_gate_rows"].loc[
        result["required_gate_rows"]["gate"].eq("add_wf_2020_official_replay_coverage"),
        "status",
    ].item() == "blocked"
    assert result["required_gate_rows"].loc[
        result["required_gate_rows"]["gate"].eq("fold_local_target_scope_audit"),
        "status",
    ].item() == "blocked"
    assert set(result["coverage_rows"]["window"]) == {"wf_2020", "wf_2022", "wf_2023_2024"}
    assert result["coverage_rows"].set_index("window").loc["wf_2020", "status"] == "missing"
    assert result["scope_rows"]["usage"].eq("reference_only").all()
    assert result["scope_rows"]["target_ready"].eq(False).all()


def test_reward_checkpoint_recipe_scope_audit_allows_training_only_after_coverage_and_foldlocal_manifest():
    contract = {
        "decision": "reward_checkpoint_recipe_contract_ready",
        "selected_candidate": "teacher_candidate",
        "final_required_windows": ["wf_2020", "wf_2022"],
        "leakage_guard": {
            "copy_selected_daily_positions_into_training_labels": False,
            "eval_window_positions_are_diagnostic_only": True,
            "next_target_generation_must_be_fold_local": True,
            "eval_overlap_rows_required": 0,
        },
    }
    selected_fold_metrics = pd.DataFrame(
        [
            {
                "fold": "wf_2020",
                "daily_path": "targets/wf_2020.csv",
                "teacher_usage": "fold_local_target_candidate",
                "teacher_status": "pass",
                "full_gate": True,
            },
            {
                "fold": "wf_2022",
                "daily_path": "targets/wf_2022.csv",
                "teacher_usage": "fold_local_target_candidate",
                "teacher_status": "pass",
                "full_gate": True,
            },
        ]
    )
    fold_local_manifest = pd.DataFrame(
        [
            {"fold": "wf_2020", "eval_overlap_rows": 0, "target_path": "targets/wf_2020.csv"},
            {"fold": "wf_2022", "eval_overlap_rows": 0, "target_path": "targets/wf_2022.csv"},
        ]
    )

    result = reward_checkpoint_screen.audit_reward_checkpoint_recipe_scope(
        contract,
        teacher_window_coverage=pd.DataFrame(),
        selected_fold_metrics=selected_fold_metrics,
        fold_local_target_manifest=fold_local_manifest,
    )

    assert result["summary"]["decision"] == "recipe_scope_audit_training_ready"
    assert result["summary"]["training_allowed"] is True
    assert result["summary"]["coverage_gate"] is True
    assert result["summary"]["scope_gate"] is True
    assert result["required_gate_rows"]["status"].tolist() == ["passed", "passed", "passed"]


def test_reward_checkpoint_recipe_scope_audit_uses_foldlocal_train_manifest_for_missing_wf2020_targets():
    contract = {
        "decision": "reward_checkpoint_recipe_contract_ready",
        "selected_candidate": "teacher_short_volshock_cap05_hit2",
        "final_required_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
        "missing_final_required_windows": ["wf_2020"],
        "leakage_guard": {
            "copy_selected_daily_positions_into_training_labels": False,
            "eval_window_positions_are_diagnostic_only": True,
            "next_target_generation_must_be_fold_local": True,
            "eval_overlap_rows_required": 0,
        },
    }
    selected_fold_metrics = pd.DataFrame(
        [
            {
                "fold": "wf_2022",
                "daily_path": "official/wf_2022.csv",
                "teacher_usage": "official_replay_reference_only_not_training_label",
                "teacher_status": "pass",
                "full_gate": True,
            },
            {
                "fold": "wf_2023_2024",
                "daily_path": "official/wf_2023_2024.csv",
                "teacher_usage": "official_replay_reference_only_not_training_label",
                "teacher_status": "near_miss",
                "full_gate": False,
            },
        ]
    )
    fold_local_manifest = pd.DataFrame(
        [
            {
                "fold": "wf_2020_train",
                "window": "wf_2020_train",
                "target_path": "targets/wf_2020.csv",
                "eval_overlap_rows": 0,
            },
            {
                "fold": "wf_2022_train",
                "window": "wf_2022_train",
                "target_path": "targets/wf_2022.csv",
                "eval_overlap_rows": 0,
            },
            {
                "fold": "wf_2023_2024_train",
                "window": "wf_2023_2024_train",
                "target_path": "targets/wf_2023_2024.csv",
                "eval_overlap_rows": 0,
            },
        ]
    )

    result = reward_checkpoint_screen.audit_reward_checkpoint_recipe_scope(
        contract,
        teacher_window_coverage=pd.DataFrame(),
        selected_fold_metrics=selected_fold_metrics,
        fold_local_target_manifest=fold_local_manifest,
    )

    assert result["summary"]["decision"] == "recipe_scope_audit_training_ready"
    assert result["summary"]["training_allowed"] is True
    assert result["summary"]["coverage_gate"] is True
    assert result["summary"]["target_coverage_gate"] is True
    assert result["summary"]["official_replay_coverage_gate"] is False
    assert result["summary"]["promotion_allowed"] is False
    assert result["summary"]["missing_final_required_windows"] == []
    assert set(result["coverage_rows"].loc[result["coverage_rows"]["present"], "window"]) == {
        "wf_2020",
        "wf_2022",
        "wf_2023_2024",
    }
    assert set(result["scope_rows"].loc[result["scope_rows"]["target_ready"], "fold"]) == {
        "wf_2020",
        "wf_2022",
        "wf_2023_2024",
    }
    assert result["required_gate_rows"].loc[
        result["required_gate_rows"]["gate"].eq("add_wf_2020_official_replay_coverage"),
        "status",
    ].item() == "blocked"
    assert result["required_gate_rows"].loc[
        result["required_gate_rows"]["gate"].eq("rerun_strict_final_walkforward_retention_gate"),
        "status",
    ].item() == "blocked"

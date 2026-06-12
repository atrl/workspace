from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import audit_underperforming_backtests as audit


def test_underperforming_cleanup_decisions_protect_runtime_component_and_prune_ablation(tmp_path):
    root = tmp_path
    baseline_path = (
        root
        / "runtime/production/card/soxl_backtest_policy_reimplementation_v1/legacy_comparison_metrics.csv"
    )
    comparison_path = (
        root / "runtime/production/card/soxl_backtest_jan_june_windows_v1/all_model_window_metrics.csv"
    )
    baseline_path.parent.mkdir(parents=True)
    comparison_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "window": "ytd_2026",
                "new_return_pct": 132.0,
                "legacy_return_pct": 132.0,
            }
        ]
    ).to_csv(baseline_path, index=False)
    pd.DataFrame(
        [
            {
                "window": "ytd_2026_common",
                "profile": "smoke",
                "status": "canonical",
                "model_kind": "dashboard_replay",
                "model_return_pct": 224.0,
                "source_path": "runtime/model/card/smoke/smoke_soxl_daily.csv",
            },
            {
                "window": "ytd_2026_common",
                "profile": "balanced_rule_residual_v22",
                "status": "canonical",
                "model_kind": "dashboard_replay",
                "model_return_pct": 116.0,
                "source_path": (
                    "runtime/model/card/balanced_rule_residual_v22/"
                    "balanced_rule_residual_v22_soxl_daily.csv"
                ),
            },
            {
                "window": "ytd_2026_common",
                "profile": "balanced_rule_residual_v22_continued_retention_v1",
                "status": "ablation",
                "model_kind": "dashboard_replay",
                "model_return_pct": 116.0,
                "source_path": (
                    "runtime/model/card/balanced_rule_residual_v22_continued_retention_v1/"
                    "balanced_rule_residual_v22_continued_retention_v1_soxl_daily.csv"
                ),
            },
            {
                "window": "ytd_2026_common",
                "profile": "v19reg_v22_budget_usage_bull_guard_v1",
                "status": "production_composer",
                "model_kind": "runtime_composer",
                "model_return_pct": 3.0,
                "source_path": (
                    "runtime/production/card/current_model_ytd_june_comparison_v1/"
                    "current_model_daily.csv"
                ),
            },
        ]
    ).to_csv(comparison_path, index=False)

    decisions = audit.build_cleanup_decisions(
        root=root,
        baseline_metrics_path=baseline_path,
        comparison_metrics_path=comparison_path,
    )
    by_profile = {item["profile"]: item for item in decisions}

    assert by_profile["soxl_backtest_policy_reimplementation_v1"]["decision"] == "keep_primary"
    assert by_profile["smoke"]["decision"] == "keep_executable"
    assert by_profile["balanced_rule_residual_v22"]["decision"] == "protect_runtime_component"
    assert by_profile["v19reg_v22_budget_usage_bull_guard_v1"]["decision"] == "protect_production_runtime"
    assert by_profile["balanced_rule_residual_v22_continued_retention_v1"]["decision"] == "archive_idea_only"
    assert by_profile["balanced_rule_residual_v22_continued_retention_v1"]["cleanup_action"] == (
        "delete_profile_and_actor_checkpoints"
    )


def test_candidate_delete_paths_are_limited_to_profile_and_actor_checkpoints(tmp_path):
    profile_dir = tmp_path / "runtime/model/card/balanced_rule_residual_v22_continued_retention_v1"
    profile_dir.mkdir(parents=True)
    for name in [
        "profile.yaml",
        "notes.md",
        "soxl_balanced_rule_residual_v22_continued_retention_v1_actor.pt",
        "soxl_balanced_rule_residual_v22_continued_retention_v1_checkpoint_iter10_actor.pt",
        "soxl_balanced_rule_residual_v22_continued_retention_v1_summary.json",
    ]:
        (profile_dir / name).write_text("x", encoding="utf-8")

    paths = audit.candidate_delete_paths(
        tmp_path, "balanced_rule_residual_v22_continued_retention_v1"
    )

    assert paths == [
        Path("runtime/model/card/balanced_rule_residual_v22_continued_retention_v1/profile.yaml"),
        Path(
            "runtime/model/card/balanced_rule_residual_v22_continued_retention_v1/"
            "soxl_balanced_rule_residual_v22_continued_retention_v1_actor.pt"
        ),
        Path(
            "runtime/model/card/balanced_rule_residual_v22_continued_retention_v1/"
            "soxl_balanced_rule_residual_v22_continued_retention_v1_checkpoint_iter10_actor.pt"
        ),
    ]

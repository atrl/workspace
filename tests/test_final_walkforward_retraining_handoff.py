from __future__ import annotations

import csv
import json
from pathlib import Path


CARD = Path("runtime/model_selection/card/final_walkforward_retraining_handoff_v1")


def _load_json(name: str) -> dict:
    return json.loads((CARD / name).read_text(encoding="utf-8"))


def test_final_walkforward_retraining_handoff_keeps_no_promotion_boundary():
    summary = _load_json("summary.json")
    selection = _load_json("selection.json")
    recipe = _load_json("training_objective_recipe.json")

    assert summary["promotion_allowed"] is False
    assert summary["actor_retention_allowed"] is False
    assert summary["strict_candidate_retention"]["actor_retention_allowed_count"] == 0
    assert summary["strict_candidate_retention"]["promotable_profile_count"] == 0
    assert selection["selected_reference_baseline"] == "soxl_backtest_policy_reimplementation_v1"
    assert selection["training_allowed_next"] is True
    assert recipe["reference_role"] == "qlib_aligned_buy_sell_reference_only"
    assert recipe["long_ppo_allowed"] is False
    assert recipe["short_training_screen_allowed"] is True
    assert recipe["required_gates_for_future_retention"]["max_walkforward_drawdown_pct"] == 20.0
    assert recipe["required_gates_for_future_retention"]["required_best_return_pct"] == 213.143462
    assert "retain SOXLBacktest reference as actor" in recipe["blocked_paths"]


def test_final_walkforward_retraining_handoff_sources_and_fold_gates_are_explicit():
    with (CARD / "source_manifest.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    assert source_rows
    assert all(row["exists"] == "True" for row in source_rows)

    with (CARD / "fold_gate_matrix.csv").open(newline="", encoding="utf-8") as handle:
        fold_rows = list(csv.DictReader(handle))
    assert any(
        row["source_label"] == "soxlbacktest_qlib_walkforward_baseline_v1"
        and row["selected_candidate"] == "wf_pass_ytd18_15_15_02"
        and row["candidate_gate"] == "False"
        and "ytd_return_below_current_best_alignment" in row["blocker"]
        for row in fold_rows
    )
    assert any(
        row["source_label"] == "state_conditioned_multi_objective_bc_screen_v1"
        and row["candidate_gate"] == "False"
        and "drawdown gate failed" in row["blocker"]
        for row in fold_rows
    )

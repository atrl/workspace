from __future__ import annotations

import csv
import json
from pathlib import Path


CARD = Path("runtime/model_selection/card/short_foldlocal_drawdown_drift_retention_audit_v1")


def _load_json(name: str) -> dict:
    return json.loads((CARD / name).read_text(encoding="utf-8"))


def test_short_foldlocal_retention_audit_blocks_actor_retention():
    summary = _load_json("summary.json")
    selection = _load_json("selection.json")

    assert summary["decision"] == "short_foldlocal_drawdown_drift_no_retained_actor"
    assert summary["promotion_allowed"] is False
    assert summary["actor_retention_allowed"] is False
    assert summary["retention_gate_pass_count"] == 0
    assert summary["strict_candidate_retention"]["passing_profile_count"] == 0
    assert summary["strict_candidate_retention"]["actor_retention_allowed_count"] == 0
    assert summary["required_gates"]["max_walkforward_drawdown_pct"] == 20.0
    assert summary["required_gates"]["required_best_return_pct"] == 213.143462
    assert selection["selected_reference_baseline"] == "soxl_backtest_policy_reimplementation_v1"
    assert selection["soxlbacktest_reference_role"] == "qlib_reference_baseline_not_actor"
    assert selection["next_mechanism"] == (
        "fold_local_drawdown_drift_reward_or_checkpoint_screen_with_full_wf2020_wf2022_wf2023_2024_gate"
    )


def test_short_foldlocal_retention_audit_sources_and_blockers_are_explicit():
    with (CARD / "source_manifest.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    assert source_rows
    assert all(row["exists"] == "True" for row in source_rows)

    with (CARD / "candidate_gate_matrix.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_id = {row["candidate_id"]: row for row in rows}
    assert by_id["strict_profile_pool"]["retention_gate_pass"] == "False"
    assert by_id["strict_profile_pool"]["blocker"] == "strict_retention_gate_has_zero_passing_profiles"

    qlib_reference = by_id["soxlbacktest::wf_pass_ytd18_15_15_02"]
    assert qlib_reference["candidate_type"] == "qlib_reference_screen"
    assert qlib_reference["walkforward_drawdown_pass"] == "True"
    assert qlib_reference["retention_gate_pass"] == "False"
    assert "return_below_current_best_alignment" in qlib_reference["blocker"]

    bc_screen = by_id["short_reward::bc_initialization"]
    assert bc_screen["candidate_type"] == "short_training_screen"
    assert bc_screen["screen_gate_pass"] == "True"
    assert bc_screen["retention_gate_pass"] == "False"
    assert "missing_wf_2020" in bc_screen["blocker"]

    ppo_screen = by_id["short_reward::reward_ppo_selected"]
    assert ppo_screen["screen_gate_pass"] == "False"
    assert ppo_screen["max_dd_pct"] == "22.780772"
    assert "ppo_drawdown_drift" in ppo_screen["blocker"]

    official = by_id["official_replay::short_volshock_cap05_hit2_no_guard"]
    assert official["screen_gate_pass"] == "False"
    assert official["min_relative_return_pp"] == "29.817263"
    assert official["max_dd_pct"] == "15.107381"
    assert "near_miss_no_promotion" in official["blocker"]

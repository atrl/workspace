from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import src.workflow as workflow_exports
from src.workflow.macro_state_incremental_attribution import (
    EXPERIMENT,
    FROZEN_POSITION_SPEC,
    MacroStateIncrementalAttributionSpec,
    _paired_block_bootstrap,
)


def test_paired_block_bootstrap_is_deterministic_and_directional():
    dates = pd.date_range("2026-01-02", periods=40, freq="B")
    reference = pd.DataFrame({"datetime": dates, "strategy_return": [0.0] * len(dates)})
    candidate = pd.DataFrame({"datetime": dates, "strategy_return": [0.001] * len(dates)})
    first = _paired_block_bootstrap(candidate, reference, samples=200, block_days=5, seed=7)
    second = _paired_block_bootstrap(candidate, reference, samples=200, block_days=5, seed=7)
    assert first == second
    assert first["paired_oos_days"] == 40
    assert first["paired_return_delta_pct"] > 0.0
    assert first["bootstrap_ci_low_pct"] > 0.0
    assert first["bootstrap_probability_positive"] == 1.0


def test_frozen_position_contract_matches_baseline_candidate():
    assert FROZEN_POSITION_SPEC["threshold"] == 0.45
    assert FROZEN_POSITION_SPEC["scale"] == 0.65
    assert FROZEN_POSITION_SPEC["max_position"] == 0.65
    assert FROZEN_POSITION_SPEC["rebalance_threshold"] == 0.10
    assert FROZEN_POSITION_SPEC["min_hold_days"] == 3
    assert FROZEN_POSITION_SPEC["max_daily_change"] == 0.35


def test_incremental_workflow_export_contract():
    assert workflow_exports.MacroStateIncrementalAttributionSpec is MacroStateIncrementalAttributionSpec
    assert callable(workflow_exports.run_macro_state_incremental_attribution_card)


def test_generated_incremental_card_is_slim_frozen_and_recorder_backed():
    root = Path(__file__).resolve().parents[1]
    card = root / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["base_feature_count"] == 63
    assert summary["macro_feature_count"] == 18
    assert summary["frozen_model"]["candidate"] == (
        "lgb_xgb_consensus_probability_p0.45_s0.65_h0.1_hold3_mdc0.35"
    )
    assert summary["strict_gate"]["selection_uses_outer_test"] is False
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    add_one = pd.read_csv(card / "add_one_attribution.csv")
    drop_one = pd.read_csv(card / "drop_one_attribution.csv")
    diagnostics = pd.read_csv(card / "feature_diagnostics.csv")
    assert len(add_one) == len(drop_one) == len(diagnostics) == 18
    assert add_one["frozen_candidate"].nunique() == 1
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    assert not (card / "attribution_positions.csv").exists()
    assert not (card / "attribution_walk_forward_positions.csv").exists()
    recorder_dir = root / summary["recorder_local_dir"]
    recorder_artifacts = recorder_dir / "artifacts" / "macro_state_incremental_attribution"
    assert (recorder_artifacts / "attribution_positions.csv").exists()
    assert (recorder_artifacts / "attribution_walk_forward_positions.csv").exists()

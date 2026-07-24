from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CARD = (
    ROOT
    / "research"
    / "pit_constituent_auxiliary_training_allocation_v1"
)


def test_pit_auxiliary_card_is_slim_recorder_first_and_relative_wf_passes():
    summary = json.loads((CARD / "summary.json").read_text())
    selection = pd.read_csv(CARD / "model_selection.csv")
    audit = pd.read_csv(CARD / "objective_gate_audit.csv")

    assert summary["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )
    assert summary["development"]["selection_uses_outer_test"] is False
    assert summary["relative_objective_pass"] is True
    assert summary["objective_pass"] is False
    selected = selection.loc[selection["selected"]].iloc[0]
    assert selected["candidate"] == "pit_only_aux_005"
    assert selected["ratio_10bps"] >= 1.20
    assert selected["ratio_25bps"] > 1.05
    assert selected["positive_relative_years"] >= 5
    assert selected["incremental_ratio_10bps_vs_primary"] > 0.0
    outer_gate = audit.loc[
        audit["gate"].eq("outer_not_used_for_selection"), "passed"
    ].iloc[0]
    assert bool(outer_gate)
    for forbidden in [
        "predictions.csv",
        "positions.csv",
        "strategy_returns.csv",
        "training_diagnostics.csv",
    ]:
        assert not (CARD / forbidden).exists()


def test_pit_auxiliary_card_records_direct_model_and_training_only_boundary():
    summary = json.loads((CARD / "summary.json").read_text())
    notebook = (
        CARD / "pit_constituent_auxiliary_training_allocation_v1.ipynb"
    ).read_text()

    assert summary["architecture"]["base_position"] is None
    assert summary["architecture"]["prior_position"] is None
    assert "no constituent or 2x input at inference" in summary[
        "architecture"
    ]["inference_inputs"]
    assert (
        "run_pit_constituent_auxiliary_training_allocation_card"
        in notebook
    )
    assert (CARD / "conclusion.md").read_text().startswith("# 结论")

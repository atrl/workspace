from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.hierarchical_soxx_excess_prior_overlay import (
    EXPERIMENT,
    PriorOverlayProfile,
    _weights,
)


ROOT = Path(__file__).resolve().parents[1]


def test_explicit_prior_is_additive_and_cash_needs_both_gates() -> None:
    index = pd.date_range("2024-01-02", periods=4)
    frame = pd.DataFrame(
        {
            "base_constrained_ref": [0.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )
    predictions = pd.DataFrame(
        {
            "soxl_relative_alpha_5d_prediction": [0.01] * 4,
            "crash_hazard_5d_prediction": [0.20, 0.20, 0.80, 0.80],
            "cash_action_5d_prediction": [0.20, 0.20, 0.20, 0.80],
        },
        index=index,
    )
    profile = PriorOverlayProfile(
        name="test",
        model_soxl_share=0.20,
        alpha_threshold=0.001,
        hazard_threshold=0.70,
        cash_probability_threshold=0.50,
        hazard_soxl_cap_fraction=1.0,
        defense_cash=0.55,
        prior_weight=0.50,
        prior_soxl_cap=0.50,
    )

    weights, actions = _weights(frame, predictions, profile)

    assert actions["model_soxl_share"].tolist() == [0.20] * 4
    assert actions["prior_soxl_share"].tolist() == [0.0, 0.25, 0.25, 0.25]
    assert actions["cash_entry"].tolist() == [False, False, False, True]
    assert weights[:, 3].tolist() == [0.10, 0.10, 0.10, 0.55]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert actions.loc[index[1], "combined_soxl_share"] == 0.45
    assert actions.loc[index[2], "combined_soxl_share"] == 0.20


def test_generated_prior_overlay_card_passes_without_outer_selection() -> None:
    card = ROOT / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    selected = summary["selected_model"]
    outer10 = next(
        row
        for row in summary["outer_selected_diagnostic"]
        if row["cost_bps"] == 10
    )

    assert summary["objective_pass"] is True
    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_gate_pass"] is True
    assert selected["all_gate"] is True
    assert selected["ratio_10bps"] >= 1.20
    assert selected["ratio_25bps"] > 1.0
    assert selected["max_drawdown_pct"] > selected["soxx_max_drawdown_pct"]
    assert (
        selected["longest_underwater_days"]
        < selected["soxx_longest_underwater_days"]
    )
    assert outer10["terminal_wealth_ratio_vs_soxx"] >= 1.20
    assert outer10["max_drawdown_pct"] > outer10["soxx_max_drawdown_pct"]
    assert (
        outer10["max_drawdown_recovery_days"]
        <= outer10["soxx_max_drawdown_recovery_days"]
    )
    assert outer10["downside_avg_cash_weight_pct"] > 10.0
    assert (
        summary["architecture"]["explicit_prior"]
        == "base_constrained_ref is an additive allocation prior with its own cap and attribution; never a model feature"
    )
    required_gates = [
        row for row in summary["objective_gate_audit"] if row["required"]
    ]
    assert required_gates
    assert all(row["passed"] for row in required_gates)


def test_prior_overlay_card_keeps_large_outputs_in_recorder() -> None:
    card = ROOT / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))

    assert (
        summary["artifact_policy"]["policy"]
        == "slim_card_recorder_first"
    )
    assert set(summary["sidecars"]).issuperset(
        {
            "prior_incremental_attribution",
            "objective_gate_audit",
            "action_attribution",
            "regime_metrics",
            "risk_analysis",
        }
    )
    for filename in ("positions.csv", "strategy_returns.csv"):
        assert not (card / filename).exists()

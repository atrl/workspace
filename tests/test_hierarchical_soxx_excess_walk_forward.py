from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.hierarchical_soxx_excess_walk_forward import (
    EXPERIMENT,
    HierarchicalSoxxExcessSpec,
    PolicyProfile,
    _held_action,
    _policy_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cash_increase_requires_hazard_and_cash_gate() -> None:
    predictions = pd.DataFrame(
        {
            "soxl_relative_alpha_5d_prediction": [0.01, 0.01, 0.01, 0.01],
            "crash_hazard_5d_prediction": [0.20, 0.80, 0.20, 0.80],
            "cash_action_5d_prediction": [0.20, 0.20, 0.80, 0.80],
        },
        index=pd.date_range("2020-01-01", periods=4),
    )
    profile = PolicyProfile(
        name="test",
        alpha_threshold=0.001,
        maximum_soxl_share=0.40,
        hazard_threshold=0.50,
        cash_probability_threshold=0.50,
        hazard_soxl_cap_fraction=0.0,
        defense_cash=0.55,
    )
    weights, actions = _policy_weights(
        predictions,
        profile,
        HierarchicalSoxxExcessSpec(root=ROOT),
    )

    assert actions["cash_entry"].tolist() == [False, False, False, True]
    assert weights[:, 3].tolist() == [0.10, 0.10, 0.10, 0.55]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights[1, 0] == 0.0


def test_cash_hold_extends_only_a_valid_entry() -> None:
    raw = np.asarray([False, True, False, False, True, False])
    assert _held_action(raw, 3).tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_generated_card_is_recorder_first_and_outer_is_not_selected() -> None:
    card = ROOT / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))

    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_test_use"] == "diagnostic_only_previously_observed"
    assert (
        summary["architecture"]["base_constrained_ref"]
        == "explicit benchmark only; excluded from all model feature bundles"
    )
    assert (
        summary["artifact_policy"]["policy"]
        == "slim_card_recorder_first"
    )
    assert summary["artifact_policy"]["workflow_config"] == "workflow.yaml"
    assert set(summary["sidecars"]).issuperset(
        {
            "head_oos_audit",
            "outer_head_transfer_diagnostic",
            "action_attribution",
            "policy_ablation",
            "risk_analysis",
        }
    )
    for recorder_only in (
        "predictions.csv",
        "positions.csv",
        "strategy_returns.csv",
    ):
        assert not (card / recorder_only).exists()

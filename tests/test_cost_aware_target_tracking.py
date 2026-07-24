from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow.cost_aware_target_tracking import (
    EXPERIMENT,
    apply_split_fixed_decision_cost_stress,
    asymmetric_track_open_weights,
    select_cost_aware_development_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_asymmetric_tracking_slows_risk_up_but_not_risk_down():
    previous = np.array([0.10, 0.20, 0.70])
    desired_up = np.array([0.20, 0.25, 0.55])
    tracked, action = asymmetric_track_open_weights(
        previous,
        desired_up,
        risk_up_fraction=0.5,
    )

    np.testing.assert_allclose(tracked, [0.15, 0.225, 0.625])
    assert action == "partial_risk_increase_or_rotation"
    assert tracked.sum() == pytest.approx(1.0)

    desired_down = np.array([0.05, 0.15, 0.80])
    tracked_down, down_action = asymmetric_track_open_weights(
        previous,
        desired_down,
        risk_up_fraction=0.5,
    )
    np.testing.assert_allclose(tracked_down, desired_down)
    assert down_action == "full_risk_reduction"


def test_split_cost_stress_prices_open_and_intraday_legs_separately():
    gross = np.array([0.02, -0.01])
    open_turnover = np.array([0.50, 0.25])
    intraday_turnover = np.array([0.10, 0.00])

    stressed = apply_split_fixed_decision_cost_stress(
        gross,
        open_turnover,
        intraday_turnover,
        open_cost_bps=25.0,
        intraday_cost_bps=10.0,
    )

    np.testing.assert_allclose(
        stressed,
        gross - open_turnover * 0.0025 - intraday_turnover * 0.001,
    )
    with pytest.raises(ValueError):
        apply_split_fixed_decision_cost_stress(
            gross,
            open_turnover[:1],
            intraday_turnover,
            open_cost_bps=25.0,
            intraday_cost_bps=10.0,
        )


def test_development_selection_uses_expected_cost_and_excludes_outer():
    candidates = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "segment": "development_walk_forward",
                "selection_uses_outer_test": False,
                "development_gate_pass": True,
                "expected_cost_terminal_wealth": 3.0,
                "expected_cost_return_pct": 200.0,
                "total_turnover": 10.0,
                "risk_up_fraction": 1.0,
            },
            {
                "variant": "partial",
                "segment": "development_walk_forward",
                "selection_uses_outer_test": False,
                "development_gate_pass": True,
                "expected_cost_terminal_wealth": 3.1,
                "expected_cost_return_pct": 210.0,
                "total_turnover": 8.0,
                "risk_up_fraction": 0.5,
            },
            {
                "variant": "outer_leak",
                "segment": "outer_diagnostic",
                "selection_uses_outer_test": True,
                "development_gate_pass": True,
                "expected_cost_terminal_wealth": 99.0,
                "expected_cost_return_pct": 9_800.0,
                "total_turnover": 1.0,
                "risk_up_fraction": 1.0 / 3.0,
            },
        ]
    )

    selected = select_cost_aware_development_candidate(candidates)

    assert selected["variant"] == "partial"


def test_cost_aware_card_is_slim_and_reproduces_current_baseline():
    card = ROOT / "research" / EXPERIMENT
    if not (card / "summary.json").exists():
        pytest.skip("cost-aware research card has not been executed")
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    notebook = (card / f"{EXPERIMENT}.ipynb").read_text(encoding="utf-8")

    assert summary["selection_uses_outer_test"] is False
    assert summary["baseline"]["reproduction_pass"] is True
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["recorder_id"] in notebook
    artifact_dir = (
        Path(summary["recorder_local_dir"]) / "artifacts" / EXPERIMENT
    )
    for name in (
        "positions.csv",
        "strategy_returns.csv",
        "stage1_training_frame.csv",
        "model_selection_full.csv",
        "outer_comparison_full.csv",
    ):
        assert (artifact_dir / name).exists(), name
        assert not (card / name).exists(), name

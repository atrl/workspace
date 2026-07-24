from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.workflow.intraday_on_policy_staged_validation import (
    EXPERIMENT,
    IntradayOnPolicyStagedValidationSpec,
    annotate_parameter_neighborhood,
    select_robust_development_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def _candidate(
    *,
    band: float,
    threshold: float,
    wealth: float,
    outer_return: float,
) -> dict:
    return {
        "open_no_trade_band": band,
        "second_cut_bars": 8,
        "stage2_logistic_c": 0.3,
        "stage2_probability_threshold": threshold,
        "stage2_action_beta": 0.8,
        "terminal_wealth": wealth,
        "base_development_gate": True,
        "stage2_action_count": 25,
        "selection_uses_outer_test": False,
        "outer_return_for_test_only": outer_return,
    }


def test_robust_selection_uses_neighbors_and_not_outer_diagnostic():
    candidates = pd.DataFrame(
        [
            _candidate(
                band=band,
                threshold=threshold,
                wealth=wealth,
                outer_return=outer,
            )
            for band, values in (
                (0.0, ((0.55, 5.0, -99.0), (0.60, 5.1, -99.0), (0.65, 5.0, -99.0))),
                (0.005, ((0.55, 5.01, 999.0), (0.60, 5.11, 999.0), (0.65, 5.01, 999.0))),
            )
            for threshold, wealth, outer in values
        ]
    )
    annotated = annotate_parameter_neighborhood(candidates)
    selected = select_robust_development_candidate(
        annotated,
        spec=IntradayOnPolicyStagedValidationSpec(root=ROOT),
    )

    assert selected["open_no_trade_band"] == 0.0
    assert selected["neighbor_count"] == 3
    assert selected["stage2_probability_threshold"] == 0.60


def test_on_policy_card_is_recorder_first_and_not_promoted():
    card = ROOT / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    notebook = (card / f"{EXPERIMENT}.ipynb").read_text(encoding="utf-8")

    assert summary["decision"] == (
        "not_promoted_on_policy_return_or_stability_gate_failed"
    )
    assert summary["selection_uses_outer_test"] is False
    assert summary["selection_gates"]["robust_gate_pass"] is True
    assert (
        summary["selection_gates"]["return_preservation_gate_pass"]
        is False
    )
    assert summary["training_sample_audit"][
        "development_on_policy_stage2_samples"
    ] >= 20
    assert summary["selected_model"]["open_no_trade_band"] == 0.0
    assert summary["outer_diagnostic_metrics_10bps"][
        "max_drawdown_pct"
    ] >= -30.0
    assert summary["fixed_decision_stress_metrics_25bps"][
        "worst_rolling_5d_pct"
    ] >= -12.0
    assert summary["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )
    assert summary["recorder_id"] in notebook
    artifact_dir = (
        Path(summary["recorder_local_dir"])
        / "artifacts"
        / EXPERIMENT
    )
    for name in (
        "positions.csv",
        "strategy_returns.csv",
        "stage1_training_frame.csv",
        "on_policy_stage2_training_frame.csv",
        "model_selection_full.csv",
    ):
        assert (artifact_dir / name).exists(), name
        assert not (card / name).exists(), name

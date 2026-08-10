from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "stage1_transition_aware_action_value_v1"


def test_transition_aware_residual_router_card_contract():
    required = {
        "stage1_transition_aware_action_value_v1.ipynb",
        "workflow.yaml",
        "summary.json",
        "conclusion.md",
        "selected_model.json",
        "model_selection.csv",
        "backtest_benchmark_comparison.csv",
        "annual_metrics.csv",
        "risk_analysis.csv",
        "training_summary.csv",
        "analysis_summary.json",
        "failure_modes.md",
    }
    assert required <= {path.name for path in CARD.iterdir()}
    assert not {
        "predictions.csv",
        "portfolio_report.csv",
        "positions.csv",
        "all_model_predictions.csv",
    } & {path.name for path in CARD.iterdir()}


def test_residual_router_beats_frozen_stage1_without_changing_fallback():
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    baseline = summary["development_baseline_metrics_10bps"]
    candidate = summary["development_selected_metrics_10bps"]
    stress = summary["stress_25bps_comparison"]
    gates = summary["promotion_gates"]

    assert summary["source_stage1"]["frozen"] is True
    assert summary["source_stage1"]["stage2_enabled"] is False
    assert summary["selection_uses_outer_test"] is False
    assert summary["decision"] == (
        "candidate_for_clean_forward_validation_not_promoted"
    )
    assert candidate["return_pct"] > baseline["return_pct"]
    assert candidate["max_drawdown_pct"] >= baseline["max_drawdown_pct"]
    assert stress["development_delta_pct_points"] > 0.0
    assert gates["development_return_gate_pass"] is True
    assert gates["development_risk_gate_pass"] is True
    assert gates["stress_cost_gate_pass"] is True
    assert gates["baseline_fallback_identity_gate_pass"] is True
    assert gates["outer_used_for_selection"] is False
    assert summary["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )


def test_residual_router_keeps_baseline_and_outer_diagnostic_separate():
    comparison = pd.read_csv(CARD / "model_selection.csv")
    assert {
        "stage1_baseline",
        "transition_aware_residual_router",
    } == set(comparison["variant"])
    assert not comparison["selection_uses_outer_test"].astype(bool).any()
    assert set(comparison["segment"]) == {
        "development_walk_forward",
        "outer_diagnostic",
    }


def test_residual_router_notebook_is_qlib_native():
    notebook = json.loads(
        (
            CARD
            / "stage1_transition_aware_action_value_v1.ipynb"
        ).read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "import qlib" in text
    assert "qlib.init" in text
    assert "qlib.data import D" in text
    assert "qlib.workflow import R" in text
    assert "run_stage1_transition_aware_action_value_card" in text
    assert "conclusion.md" in text

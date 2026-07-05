from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import src.workflow as workflow_exports
from src.workflow.probability_budget_open2open import (
    EXPERIMENT,
    ProbabilityBudgetOpen2OpenCardSpec,
    _position_spec_for_probability_budget,
    _walk_forward_return_gate_summary,
)
from src.workflow.risk_hardened_open2open import _build_frame_bundle, _feature_groups


ROOT = Path(__file__).resolve().parents[1]


def test_probability_budget_position_spec_replaces_base_boost_and_risk_gate():
    assert workflow_exports.ProbabilityBudgetOpen2OpenCardSpec is ProbabilityBudgetOpen2OpenCardSpec
    assert workflow_exports.run_probability_budget_open2open_card.__name__ == "run_probability_budget_open2open_card"

    spec = ProbabilityBudgetOpen2OpenCardSpec(root=ROOT)
    bundle = _build_frame_bundle(spec)
    feature_cols = _feature_groups(bundle)["base_cogalpha_soxx"]

    position_spec = _position_spec_for_probability_budget(feature_cols, spec=spec, test=bundle.test)

    assert {item["model"] for item in position_spec.model_specs} >= {
        "lgbm_classifier",
        "lgbm_regressor",
        "xgb_classifier",
        "xgb_regressor",
        "lgb_xgb_consensus",
    }
    assert all(item["position_mode"] == "probability" for item in position_spec.position_specs or [])
    assert all("base_position_col" not in item for item in position_spec.position_specs or [])
    assert all("risk_gate_col" not in item for item in position_spec.position_specs or [])
    assert all("objective_baseline_position_col" not in item for item in position_spec.position_specs or [])
    assert {item["min_position"] for item in position_spec.position_specs or []} == {0.015}


def test_walk_forward_return_gate_requires_positive_return_and_risk_pass():
    metrics = pd.DataFrame(
        [
            {"risk_gate_pass": True, "test_return_pct": 1.0},
            {"risk_gate_pass": True, "test_return_pct": 0.2},
        ]
    )
    assert _walk_forward_return_gate_summary(metrics)["all_folds_pass_return_gate"] is True

    metrics.loc[1, "test_return_pct"] = -0.1
    assert _walk_forward_return_gate_summary(metrics)["all_folds_pass_return_gate"] is False


def test_probability_budget_open2open_card_walk_forward_passes_without_old_architecture():
    card = ROOT / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    model_selection = pd.read_csv(card / "model_selection.csv")
    risk_analysis = pd.read_csv(card / "risk_analysis.csv")
    walk_forward = pd.read_csv(card / summary["sidecars"]["backtest_walk_forward_metrics"])
    selected_params = summary["position_model"]["selected_candidate"]["test"]["params"]

    assert summary["experiment"] == EXPERIMENT
    assert summary["benchmark_basis"] == "open_to_open"
    assert summary["model_selection"]["selected_model"] == "lgb_xgb_consensus"
    assert summary["position_model"]["selected_candidate"]["test"]["position_mode"] == "probability"
    assert "base_plus_probability" not in selected_params
    assert "base_position_col" not in selected_params
    assert "risk_gate_col" not in selected_params
    assert "objective_baseline_position_col" not in selected_params
    assert summary["probability_budget_goal"]["architecture_replacement_pass"] is True
    assert summary["position_model"]["walk_forward"]["all_folds_pass_risk_gates"] is True
    assert summary["position_model"]["walk_forward"]["all_folds_pass_return_gate"] is True
    assert summary["position_model"]["walk_forward"]["positive_test_fold_ratio"] == 1.0
    assert summary["position_model"]["walk_forward"]["min_test_return_pct"] > 0.0
    assert summary["promotion_decision"] == "candidate_for_walk_forward_validated_probability_budget_under_base_constrained_ref"
    assert summary["position_model"]["comparison"]["beats_base_constrained_ref"] is False
    assert walk_forward["risk_gate_pass"].all()
    assert (walk_forward["test_return_pct"] > 0.0).all()
    direct_base = risk_analysis.loc[risk_analysis["check"] == "direct_base_comparison"].iloc[0]
    assert direct_base["status"] == "fail"
    assert float(direct_base["value"]) < 0.0
    selected = model_selection.loc[model_selection["selected"].map(bool)]
    assert set(selected["position_mode"]) == {"probability"}

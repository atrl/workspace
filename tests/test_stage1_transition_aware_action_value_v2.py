from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "stage1_transition_aware_action_value_v2"


def _summary() -> dict:
    return json.loads((CARD / "summary.json").read_text(encoding="utf-8"))


def test_v2_card_is_slim_and_recorder_first():
    required = {
        "stage1_transition_aware_action_value_v2.ipynb",
        "workflow.yaml",
        "summary.json",
        "conclusion.md",
        "selected_model.json",
        "model_selection.csv",
        "backtest_benchmark_comparison.csv",
        "annual_metrics.csv",
        "risk_analysis.csv",
        "training_summary.csv",
        "calibration_diagnostics.csv",
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
    assert _summary()["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )


def test_v2_repairs_train_serve_turnover_contract():
    summary = _summary()
    audit = summary["dataset_audit"]
    features = audit["features"]

    assert "candidate_current_turnover" in features
    assert "candidate_next_reset_turnover" not in features
    assert "candidate_turnover" not in features
    assert audit["feature_uses_future_reset_cost"] is False
    assert audit["label_uses_future_reset_cost"] is True
    assert summary["promotion_gates"][
        "train_serve_feature_contract_pass"
    ] is True


def test_v2_abstains_when_lower_bound_has_no_supported_override():
    summary = _summary()
    comparison = pd.read_csv(CARD / "model_selection.csv")
    development = comparison[
        comparison["segment"].eq("development_walk_forward")
    ].set_index("variant")

    assert summary["decision"] == (
        "not_promoted_no_positive_lower_bound_overrides"
    )
    assert summary["promotion"] == "blocked"
    assert summary["incremental_action_audit"][
        "development_override_count"
    ] == 0
    assert summary["incremental_action_audit"][
        "duplicate_decision_key_count"
    ] == 0
    assert development.loc[
        "net_advantage_lcb_router", "terminal_wealth"
    ] == development.loc["stage1_baseline", "terminal_wealth"]
    assert summary["promotion_gates"][
        "baseline_fallback_identity_gate_pass"
    ] is True
    assert summary["promotion_gates"][
        "unique_daily_decision_key_gate_pass"
    ] is True


def test_v2_uses_time_ordered_qlib_dataset_and_recorder():
    summary = _summary()
    diagnostics = pd.read_csv(CARD / "calibration_diagnostics.csv")
    notebook = json.loads(
        (
            CARD / "stage1_transition_aware_action_value_v2.ipynb"
        ).read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_status"] == (
        "reused_diagnostic_not_clean_unseen_holdout"
    )
    assert diagnostics["training_end"].notna().all()
    assert "import qlib" in text
    assert "qlib.init" in text
    assert "qlib.data import D" in text
    assert "qlib.workflow import R" in text
    assert "run_stage1_transition_aware_action_value_v2_card" in text

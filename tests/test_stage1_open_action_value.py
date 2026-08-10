from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.workflow.stage1_open_action_value import ACTION_FEATURES


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "stage1_open_action_value_v1"


def test_open_action_features_are_known_before_open_execution():
    assert not any(name.startswith("early_") for name in ACTION_FEATURES)
    assert not any(name.startswith("log_volume_") for name in ACTION_FEATURES)
    assert {
        "portfolio_drawdown",
        "target_soxl_weight",
        "pretrade_soxl_weight",
        "effective_beta_delta",
        "open_turnover",
    } <= set(ACTION_FEATURES)


def test_generated_card_beats_confidence_veto_and_reduces_frequency():
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    baseline = summary["development_baseline"]
    candidate = summary["development_candidate"]

    assert summary["decision"] == (
        "validated_research_candidate_for_clean_forward_shadow"
    )
    assert summary["development_gate_pass"] is True
    assert summary["selection_uses_outer_test"] is False
    assert all(summary["development_gates"].values())
    assert candidate["return_pct"] > baseline["return_pct"]
    assert candidate["max_drawdown_pct"] >= baseline["max_drawdown_pct"]
    assert candidate["worst_rolling_5d_pct"] >= baseline["worst_rolling_5d_pct"]
    assert candidate["open_trade_days"] < baseline["open_trade_days"]
    assert candidate["total_turnover"] <= baseline["total_turnover"]
    assert summary["frequency_change"]["open_trade_day_reduction_pct"] >= 8.0
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"


def test_annual_walk_forward_floor_and_outer_separation():
    annual = pd.read_csv(CARD / "annual_metrics.csv")
    candidate = annual[
        annual["variant"].eq("open_action_value_candidate")
        & annual["segment"].eq("development_walk_forward")
        & annual["year"].ge(2017)
    ]
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))

    assert candidate["wealth_ratio_vs_baseline"].min() >= 0.985
    assert summary["outer_role"] == "reused_diagnostic_only_not_model_selection"
    assert summary["production_change"] == "blocked_pending_clean_forward_shadow"
    assert summary["development_gates"]["baseline_fallback_identity"] is True


def test_card_is_slim_and_notebook_is_qlib_native():
    forbidden = {
        "predictions.csv",
        "signals.csv",
        "portfolio_report.csv",
        "qlib_portfolio_report.csv",
        "qlib_positions.csv",
        "all_model_predictions.csv",
    }
    assert not forbidden.intersection(path.name for path in CARD.iterdir())
    notebook = json.loads(
        (CARD / "stage1_open_action_value_v1.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "qlib.init" in source
    assert "D.calendar" in source
    assert "DatasetH" in source
    assert "R.start" in source
    assert "run_stage1_open_action_value_card" in source
    assert "中文结论" in source

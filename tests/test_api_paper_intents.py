from __future__ import annotations

import json
import math

import pytest

from src.api import (
    PaperBrokerAdapter,
    build_target_position_intent_from_dashboard,
    execute_paper_intent,
    write_intent_ledger,
)


def _dashboard_payload() -> dict:
    return {
        "schema_version": "dashboard.model_matrix.v1",
        "artifact_id": "factor_feature_set_matrix_v1",
        "top_models": [
            {
                "rank": 1,
                "model": "random_forest_clf__all_features",
                "base_model": "random_forest_clf",
                "feature_set": "all_features",
                "avg_position": 0.74,
                "stability_score": 54.2,
                "max_drawdown_pct": 21.6,
            }
        ],
        "summary": {"status": "feature_set_matrix_completed"},
    }


def test_build_target_position_intent_from_dashboard_clamps_and_records_model_source():
    payload = _dashboard_payload()

    intent = build_target_position_intent_from_dashboard(
        payload,
        instrument="SOXL",
        max_position=0.5,
        min_position=0.1,
    )

    assert intent.schema_version == "api.execution_intent.v1"
    assert intent.mode == "paper"
    assert intent.instrument == "SOXL"
    assert intent.target_position == 0.5
    assert intent.source_model == "random_forest_clf__all_features"
    assert intent.source_artifact == "factor_feature_set_matrix_v1"
    assert intent.metadata["raw_avg_position"] == 0.74
    assert "真实 broker" in intent.guardrail


def test_build_target_position_intent_records_dashboard_review_context_and_blocks_unsafe_top_model():
    payload = _dashboard_payload()
    payload["analysis_overview"] = {
        "schema_version": "analysis.overview.v1",
        "top_weakness": {
            "model": "random_forest_clf__all_features",
            "segment": "stress_window::stress",
            "decision_hint": "block_or_rework",
        },
        "largest_relative_return_spread_segments": [
            {
                "segment": "stress_window::stress",
                "best_relative_return_model": "ridge_mse__all_features",
                "worst_relative_return_model": "random_forest_clf__all_features",
                "relative_return_pct_spread": 8.5,
            }
        ],
    }
    payload["objective"] = {
        "promotion_scores": [
            {
                "rank": 1,
                "model": "random_forest_clf__all_features",
                "promotion_score": -4.0,
                "promotion_decision": "blocked",
                "promotion_blockers": "max_drawdown_pct",
            }
        ]
    }
    payload["backtest"] = {
        "config": {
            "schema_version": "backtest.config.v1",
            "cost_model": {"open_cost_bps": 1.0, "close_cost_bps": 1.0, "slippage_bps": 1.5},
        },
        "walk_forward_splits": [{"fold": 1}, {"fold": 2}],
    }

    intent = build_target_position_intent_from_dashboard(
        payload,
        instrument="SOXL",
        min_position=0.1,
        max_position=0.75,
    )

    assert intent.target_position == 0.1
    assert intent.metadata["execution_gate"] == "blocked_to_min_position"
    assert intent.metadata["promotion_decision"] == "blocked"
    assert intent.metadata["promotion_blockers"] == "max_drawdown_pct"
    assert intent.metadata["analysis_overview_schema_version"] == "analysis.overview.v1"
    assert intent.metadata["analysis_top_weakness_model"] == "random_forest_clf__all_features"
    assert intent.metadata["analysis_top_weakness_decision_hint"] == "block_or_rework"
    assert intent.metadata["analysis_top_gap_segment"] == "stress_window::stress"
    assert intent.metadata["analysis_top_gap_spread"] == 8.5
    assert intent.metadata["backtest_schema_version"] == "backtest.config.v1"
    assert intent.metadata["backtest_walk_forward_fold_count"] == 2
    assert intent.metadata["backtest_slippage_bps"] == 1.5


def test_build_target_position_intent_blocks_top_model_without_complete_run_record():
    payload = _dashboard_payload()
    payload["model_run_coverage"] = {
        "schema_version": "model.run_ledger_summary.v1",
        "expected_model_count": 2,
        "trained_model_count": 1,
        "backtested_model_count": 1,
        "all_models_completed": False,
        "missing_models": ["random_forest_clf__all_features"],
    }
    payload["model_run_ledger"] = [
        {
            "schema_version": "model.run_ledger.v1",
            "model": "random_forest_clf__all_features",
            "base_model": "random_forest_clf",
            "trained": True,
            "predicted": True,
            "backtested": False,
            "objective_evaluated": True,
            "promotion_evaluated": True,
            "analysis_recorded": False,
            "manifested": False,
        }
    ]

    intent = build_target_position_intent_from_dashboard(
        payload,
        instrument="SOXL",
        min_position=0.1,
        max_position=0.75,
    )

    assert intent.target_position == 0.1
    assert intent.metadata["execution_gate"] == "blocked_to_min_position"
    assert intent.metadata["run_status"] == "incomplete"
    assert intent.metadata["run_trained"] is True
    assert intent.metadata["run_backtested"] is False
    assert intent.metadata["run_analysis_recorded"] is False
    assert intent.metadata["model_run_all_completed"] is False
    assert intent.metadata["model_run_missing_models"] == "random_forest_clf__all_features"


def test_build_target_position_intent_normalizes_empty_promotion_blockers_from_dashboard_csv():
    payload = _dashboard_payload()
    payload["objective"] = {
        "promotion_scores": [
            {
                "rank": 1,
                "model": "random_forest_clf__all_features",
                "promotion_score": 12.0,
                "promotion_decision": "promote_candidate",
                "promotion_blockers": math.nan,
            }
        ]
    }

    intent = build_target_position_intent_from_dashboard(payload, instrument="SOXL")

    assert intent.metadata["promotion_blockers"] == ""


def test_execute_paper_intent_places_target_position_on_paper_adapter_only():
    intent = build_target_position_intent_from_dashboard(_dashboard_payload(), instrument="SOXL")
    broker = PaperBrokerAdapter(cash=10000.0, shares=0)

    result = execute_paper_intent(intent, broker=broker, price=100.0)

    assert result.schema_version == "api.paper_execution_result.v1"
    assert result.status == "paper_executed"
    assert result.intent.target_position == 0.74
    assert result.snapshot.shares == 74
    assert abs(result.snapshot.position - 0.74) < 1e-12
    assert result.broker_name == "paper"


def test_execute_paper_intent_rejects_live_mode():
    intent = build_target_position_intent_from_dashboard(_dashboard_payload(), instrument="SOXL")
    live_like = intent.with_mode("live")

    with pytest.raises(ValueError, match="paper/sim"):
        execute_paper_intent(live_like, broker=PaperBrokerAdapter(), price=100.0)


def test_write_intent_ledger_records_intent_and_execution_result(tmp_path):
    intent = build_target_position_intent_from_dashboard(_dashboard_payload(), instrument="SOXL")
    result = execute_paper_intent(intent, broker=PaperBrokerAdapter(cash=10000.0), price=100.0)

    path = write_intent_ledger(tmp_path / "intent_ledger.json", intent=intent, result=result)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "api.intent_ledger.v1"
    assert payload["intent"]["source_model"] == "random_forest_clf__all_features"
    assert payload["execution_result"]["status"] == "paper_executed"
    assert payload["execution_result"]["snapshot"]["shares"] == 74

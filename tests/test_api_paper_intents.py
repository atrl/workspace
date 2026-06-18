from __future__ import annotations

import json

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

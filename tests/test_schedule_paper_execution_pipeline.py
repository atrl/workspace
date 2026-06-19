from __future__ import annotations

import json

import pandas as pd

import src.data as data_api
from src.schedule import PaperExecutionPipelineConfig, classify_market_calendar_day, run_paper_execution_once


def _publish_unit_registry(tmp_path):
    raw = pd.DataFrame(
        [
            {"date": "2026-01-02", "instrument": "SOXL", "close": 10.0, "volume": 100},
            {"date": "2026-01-05", "instrument": "SOXL", "close": 12.0, "volume": 120},
        ]
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05"]), name="date")
    snapshot = data_api.build_canonical_snapshot(
        raw,
        name="unit_soxl_daily",
        universe=("SOXL",),
        calendar=calendar,
        calendar_path="data/qlib/soxl/calendars/day.txt",
        source_paths=("provider:unit",),
        start="2026-01-01",
        end="2026-01-05",
        as_of="2026-01-05",
    )
    data_api.publish_canonical_snapshot(
        snapshot,
        output_dir=tmp_path / "data" / "snapshots" / "unit_soxl_daily" / "2026-01-05",
        dataset_key="daily.unit_soxl",
        registry_path=tmp_path / "data" / "data_registry.json",
        published_at="2026-01-05T21:00:00Z",
    )
    return tmp_path / "data" / "data_registry.json"


def _write_dashboard_export(tmp_path):
    path = tmp_path / "dashboard" / "dashboard_export.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "dashboard.model_matrix.v1",
                "artifact_id": "unit_dashboard",
                "artifact_type": "dashboard_export",
                "manifest_source": {
                    "kind": "analysis_manifest",
                    "path": "matrix_card/analysis_manifest.json",
                    "manifest_hash": "a" * 64,
                    "producer": "src.analysis.exports",
                },
                "top_models": [
                    {
                        "rank": 1,
                        "model": "ridge_mse__all_features",
                        "base_model": "ridge_mse",
                        "feature_set": "all_features",
                        "avg_position": 0.5,
                        "stability_score": 4.0,
                        "max_drawdown_pct": 8.0,
                    }
                ],
                "objective": {
                    "promotion_scores": [
                        {
                            "rank": 1,
                            "model": "ridge_mse__all_features",
                            "promotion_score": 12.0,
                            "promotion_decision": "promote_candidate",
                            "promotion_blockers": "",
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_paper_execution_pipeline_reads_dashboard_and_registry_then_writes_execution_evidence(tmp_path):
    registry_path = _publish_unit_registry(tmp_path)
    dashboard_path = _write_dashboard_export(tmp_path)
    events: list[str] = []
    config = PaperExecutionPipelineConfig(
        name="paper_execution",
        dashboard_export_path=dashboard_path,
        data_registry_path=registry_path,
        dataset_key="daily.unit_soxl",
        instrument="SOXL",
        intent_ledger_path=tmp_path / "runtime" / "intent_ledger.json",
        paper_execution_result_path=tmp_path / "runtime" / "paper_execution_result.json",
        initial_cash=1000.0,
        notify_details=True,
    )

    result = run_paper_execution_once(config, lock_dir=tmp_path / "locks", notify=events.append)

    ledger = json.loads(config.intent_ledger_path.read_text(encoding="utf-8"))
    execution = json.loads(config.paper_execution_result_path.read_text(encoding="utf-8"))
    assert result.status == "success"
    assert result.payload["schema_version"] == "schedule.paper_execution.v1"
    assert result.payload["status"] == "paper_executed"
    assert result.payload["live_broker_enabled"] is False
    assert result.payload["mode"] == "paper"
    assert result.payload["instrument"] == "SOXL"
    assert result.payload["dataset_key"] == "daily.unit_soxl"
    assert result.payload["quote"]["date"] == "2026-01-05"
    assert result.payload["quote"]["price"] == 12.0
    assert result.payload["source_model"] == "ridge_mse__all_features"
    assert result.payload["execution_gate"] == "paper_review"
    assert result.payload["dashboard_manifest_status"] == "complete"
    assert result.payload["dashboard_manifest_kind"] == "analysis_manifest"
    assert result.payload["dashboard_manifest_hash"] == "a" * 64
    assert result.payload["target_position"] == 0.5
    assert result.payload["snapshot"]["shares"] == 41
    assert abs(result.payload["snapshot"]["position"] - 0.492) < 1e-3
    assert ledger["schema_version"] == "api.intent_ledger.v1"
    assert ledger["intent"]["source_artifact"] == "unit_dashboard"
    assert ledger["execution_result"]["snapshot"]["price"] == 12.0
    assert execution["schema_version"] == "api.paper_execution_result.v1"
    assert execution["status"] == "paper_executed"
    assert events == [
        "paper_execution:success | mode=paper | gate=paper_review | model=ridge_mse__all_features "
        "| target=0.5 | dashboard_manifest=analysis_manifest/complete | quote=SOXL@2026-01-05/12 "
        f"| shares=41 | ledger={config.intent_ledger_path}"
    ]


def test_paper_execution_pipeline_notification_includes_model_difference_gate_context(tmp_path):
    registry_path = _publish_unit_registry(tmp_path)
    dashboard_path = tmp_path / "dashboard" / "dashboard_export.json"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(
        json.dumps(
            {
                "schema_version": "dashboard.model_matrix.v1",
                "artifact_id": "difference_dashboard",
                "artifact_type": "dashboard_export",
                "manifest_source": {
                    "kind": "analysis_manifest",
                    "path": "matrix_card/analysis_manifest.json",
                    "manifest_hash": "b" * 64,
                    "producer": "src.analysis.exports",
                },
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
                "model_difference_explanations": [
                    {
                        "schema_version": "analysis.model_difference_explanation.v1",
                        "priority_rank": 1,
                        "segment": "stress_window::stress",
                        "baseline_model": "ridge_mse__all_features",
                        "winner_model": "ridge_mse__all_features",
                        "loser_model": "random_forest_clf__all_features",
                        "relative_return_pct_spread": 6.5,
                        "explanation_signal": "baseline_defends",
                        "decision_hint": "block_or_rework_loser",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    events: list[str] = []
    config = PaperExecutionPipelineConfig(
        name="paper_execution",
        dashboard_export_path=dashboard_path,
        data_registry_path=registry_path,
        dataset_key="daily.unit_soxl",
        instrument="SOXL",
        intent_ledger_path=tmp_path / "runtime" / "intent_ledger.json",
        paper_execution_result_path=tmp_path / "runtime" / "paper_execution_result.json",
        initial_cash=1000.0,
        min_position=0.1,
        max_position=0.75,
        notify_details=True,
    )

    result = run_paper_execution_once(config, lock_dir=tmp_path / "locks", notify=events.append)

    ledger = json.loads(config.intent_ledger_path.read_text(encoding="utf-8"))
    assert result.status == "success"
    assert result.payload["execution_gate"] == "blocked_to_min_position"
    assert result.payload["target_position"] == 0.1
    assert result.payload["model_difference_schema_version"] == "analysis.model_difference_explanation.v1"
    assert result.payload["model_difference_top_signal"] == "baseline_defends"
    assert result.payload["model_difference_top_decision_hint"] == "block_or_rework_loser"
    assert result.payload["model_difference_top_segment"] == "stress_window::stress"
    assert result.payload["model_difference_top_loser_model"] == "random_forest_clf__all_features"
    assert result.payload["model_difference_top_winner_model"] == "ridge_mse__all_features"
    assert result.payload["model_difference_top_spread"] == 6.5
    assert result.payload["model_difference_marks_source_loser"] is True
    assert ledger["intent"]["metadata"]["model_difference_marks_source_loser"] is True
    assert "model_difference=baseline_defends/block_or_rework_loser" in events[0]
    assert "stress_window::stress random_forest_clf__all_features->ridge_mse__all_features spread=6.5" in events[0]


def test_paper_execution_pipeline_skips_closed_market_day_before_reading_inputs(tmp_path):
    events: list[str] = []
    decision = classify_market_calendar_day(
        "2026-01-01",
        trading_calendar=("2026-01-02",),
        holiday_reasons={"2026-01-01": "New Year's Day"},
    )
    config = PaperExecutionPipelineConfig(
        name="paper_execution",
        dashboard_export_path=tmp_path / "missing_dashboard_export.json",
        data_registry_path=tmp_path / "missing_data_registry.json",
        dataset_key="daily.unit_soxl",
        instrument="SOXL",
        intent_ledger_path=tmp_path / "runtime" / "intent_ledger.json",
        paper_execution_result_path=tmp_path / "runtime" / "paper_execution_result.json",
        notify_details=True,
        market_calendar_decision=decision,
    )

    result = run_paper_execution_once(config, lock_dir=tmp_path / "locks", notify=events.append)

    assert result.status == "skipped"
    assert result.detail == "New Year's Day"
    assert result.payload["market_calendar_decision"]["market_status"] == "closed"
    assert events == []
    assert not config.intent_ledger_path.exists()
    assert not config.paper_execution_result_path.exists()
    assert not (tmp_path / "locks" / "paper_execution.lock").exists()


def test_paper_execution_pipeline_rejects_live_mode_before_writing_evidence(tmp_path):
    registry_path = _publish_unit_registry(tmp_path)
    dashboard_path = _write_dashboard_export(tmp_path)
    config = PaperExecutionPipelineConfig(
        name="paper_execution",
        dashboard_export_path=dashboard_path,
        data_registry_path=registry_path,
        dataset_key="daily.unit_soxl",
        instrument="SOXL",
        intent_ledger_path=tmp_path / "runtime" / "intent_ledger.json",
        paper_execution_result_path=tmp_path / "runtime" / "paper_execution_result.json",
        mode="live",
    )

    result = run_paper_execution_once(config, lock_dir=tmp_path / "locks")

    assert result.status == "failed"
    assert "paper/sim" in result.detail
    assert not config.intent_ledger_path.exists()
    assert not config.paper_execution_result_path.exists()

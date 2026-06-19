from __future__ import annotations

import json
from pathlib import Path

from src.core import DatasetSnapshot, publish_production_manifest
from src.schedule import ProductionHealthCheckConfig, run_production_health_check_once


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _publish_strategy(tmp_path: Path, *, include_ledger: bool = True) -> tuple[Path, Path]:
    card = tmp_path / "production" / "paper_sim_replay_v1"
    _write_json(card / "production_summary.json", {"status": "paper_replay_completed"})
    _write_json(card / "dashboard_export.json", {"schema_version": "dashboard.model_matrix.v1"})
    _write_json(card / "paper_execution_result.json", {"status": "paper_executed"})
    _write_json(card / "data" / "data_registry.json", {"schema_version": "data.registry.v1"})
    _write_json(card / "matrix_card" / "analysis_manifest.json", {"schema_version": "artifact.v1"})
    if include_ledger:
        _write_json(card / "intent_ledger.json", {"schema_version": "api.intent_ledger.v1"})
    (card / "paper_sim_replay_v1.ipynb").write_text("{}", encoding="utf-8")
    dataset = DatasetSnapshot(
        name="paper_replay_soxl_daily",
        universe=("SOXL",),
        start="2026-01-02",
        end="2026-06-09",
        calendar_path="data/qlib/soxl/calendars/day.txt",
        data_paths=("data/data_registry.json",),
    )
    publish_production_manifest(
        strategy_id="paper_sim_replay_v1",
        output_dir=card,
        registry_path=tmp_path / "production" / "production_registry.json",
        notebook_path="paper_sim_replay_v1.ipynb",
        dataset=dataset,
        paths={
            "summary": "production_summary.json",
            "dashboard_export": "dashboard_export.json",
            "intent_ledger": "intent_ledger.json",
            "paper_execution_result": "paper_execution_result.json",
            "data_registry": "data/data_registry.json",
            "analysis_manifest": "matrix_card/analysis_manifest.json",
        },
        metadata={
            "status": "paper_replay_completed",
            "mode": "paper",
            "live_broker_enabled": False,
            "paper_execution_pipeline_schema_version": "schedule.paper_execution.v1",
            "paper_execution_job_status": "success",
        },
        published_at="2026-06-09T21:30:00Z",
    )
    return card, tmp_path / "production" / "production_registry.json"


def test_production_health_check_reads_registry_manifest_and_writes_dashboard_ready_export(tmp_path):
    card, registry_path = _publish_strategy(tmp_path)
    output_path = tmp_path / "dashboard" / "production_health.json"
    events: list[str] = []
    config = ProductionHealthCheckConfig(
        name="production_health_check",
        registry_path=registry_path,
        strategy_id="paper_sim_replay_v1",
        output_path=output_path,
        notify_details=True,
    )

    result = run_production_health_check_once(config, lock_dir=tmp_path / "locks", notify=events.append)
    export = json.loads(output_path.read_text(encoding="utf-8"))

    assert result.status == "success"
    assert result.payload == export
    assert result.payload["schema_version"] == "schedule.production_health.v1"
    assert result.payload["strategy_id"] == "paper_sim_replay_v1"
    assert result.payload["health_status"] == "healthy"
    assert result.payload["is_healthy"] is True
    assert result.payload["mode"] == "paper"
    assert result.payload["live_broker_enabled"] is False
    assert result.payload["paper_execution_pipeline_schema_version"] == "schedule.paper_execution.v1"
    assert result.payload["paper_execution_job_status"] == "success"
    assert result.payload["manifest_path"] == str(card / "production_manifest.json")
    assert result.payload["manifest_hash"] == json.loads((card / "production_manifest.json").read_text())["manifest_hash"]
    assert result.payload["checked_path_count"] == 7
    assert result.payload["missing_required_paths"] == []
    assert result.payload["dashboard_export_path"].endswith("dashboard_export.json")
    assert events == [
        "production_health_check:success | strategy=paper_sim_replay_v1 | health=healthy "
        f"| missing=0 | paper_execution=schedule.paper_execution.v1/success | manifest={card / 'production_manifest.json'}"
    ]


def test_production_health_check_reports_missing_required_manifest_paths_without_failing_job(tmp_path):
    _, registry_path = _publish_strategy(tmp_path, include_ledger=False)
    config = ProductionHealthCheckConfig(
        name="production_health_check",
        registry_path=registry_path,
        strategy_id="paper_sim_replay_v1",
        required_paths=("summary", "intent_ledger", "paper_execution_result"),
        notify_details=True,
    )
    events: list[str] = []

    result = run_production_health_check_once(config, lock_dir=tmp_path / "locks", notify=events.append)

    assert result.status == "success"
    assert result.payload["health_status"] == "missing_required_paths"
    assert result.payload["is_healthy"] is False
    assert result.payload["missing_required_paths"] == ["intent_ledger"]
    assert "health=missing_required_paths" in events[0]
    assert "missing=1" in events[0]


def test_production_health_check_detects_manifest_content_hash_mismatch(tmp_path):
    card, registry_path = _publish_strategy(tmp_path)
    manifest_path = card / "production_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["status"] = "paper_replay_changed_after_publish"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    config = ProductionHealthCheckConfig(
        name="production_health_check",
        registry_path=registry_path,
        strategy_id="paper_sim_replay_v1",
        required_paths=("summary", "intent_ledger", "paper_execution_result"),
        notify_details=True,
    )
    events: list[str] = []

    result = run_production_health_check_once(config, lock_dir=tmp_path / "locks", notify=events.append)

    assert result.status == "success"
    assert result.payload["health_status"] == "manifest_hash_mismatch"
    assert result.payload["is_healthy"] is False
    assert result.payload["manifest_hash_matches_registry"] is True
    assert result.payload["manifest_hash_matches_content"] is False
    assert result.payload["computed_manifest_hash"] != result.payload["manifest_hash"]
    assert "health=manifest_hash_mismatch" in events[0]

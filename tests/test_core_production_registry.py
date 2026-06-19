from __future__ import annotations

import json

from src.core import (
    DatasetSnapshot,
    build_production_manifest,
    production_manifest_content_hash,
    publish_production_manifest,
)


def _dataset() -> DatasetSnapshot:
    return DatasetSnapshot(
        name="paper_replay_soxl_daily",
        universe=("SOXL",),
        start="2026-01-02",
        end="2026-06-09",
        calendar_path="data/qlib/soxl/calendars/day.txt",
        data_paths=("production/paper_sim_replay_v1/data/data_registry.json",),
    )


def test_build_production_manifest_records_notebook_sidecars_and_hash():
    manifest = build_production_manifest(
        strategy_id="paper_sim_replay_v1",
        notebook_path="paper_sim_replay_v1.ipynb",
        dataset=_dataset(),
        paths={
            "summary": "production_summary.json",
            "dashboard_export": "dashboard_export.json",
            "intent_ledger": "intent_ledger.json",
            "market_quote": "market_quote.json",
            "paper_execution_result": "paper_execution_result.json",
            "data_registry": "data/data_registry.json",
            "analysis_manifest": "matrix_card/analysis_manifest.json",
        },
        metadata={
            "mode": "paper",
            "live_broker_enabled": False,
            "market_data_dataset_key": "daily.soxl_paper_replay",
            "status": "paper_replay_completed",
        },
    )

    payload = manifest.to_dict()

    assert payload["schema_version"] == "production.manifest.v1"
    assert payload["artifact_type"] == "production_strategy"
    assert payload["producer"] == "src.core.production"
    assert payload["artifact_id"] == "paper_sim_replay_v1"
    assert payload["paths"]["notebook"] == "paper_sim_replay_v1.ipynb"
    assert payload["paths"]["summary"] == "production_summary.json"
    assert payload["metadata"]["mode"] == "paper"
    assert payload["metadata"]["live_broker_enabled"] is False
    assert len(manifest.manifest_hash()) == 64


def test_publish_production_manifest_writes_manifest_and_registry_history(tmp_path):
    dataset = _dataset()
    first = publish_production_manifest(
        strategy_id="paper_sim_replay_v1",
        output_dir=tmp_path / "production" / "paper_sim_replay_v1",
        registry_path=tmp_path / "production" / "production_registry.json",
        notebook_path="paper_sim_replay_v1.ipynb",
        dataset=dataset,
        paths={"summary": "production_summary.json", "intent_ledger": "intent_ledger.json"},
        metadata={"mode": "paper", "status": "paper_replay_completed", "run_id": "run_1"},
        published_at="2026-06-09T21:30:00Z",
    )
    second = publish_production_manifest(
        strategy_id="paper_sim_replay_v1",
        output_dir=tmp_path / "production" / "paper_sim_replay_v1",
        registry_path=tmp_path / "production" / "production_registry.json",
        notebook_path="paper_sim_replay_v1.ipynb",
        dataset=dataset,
        paths={"summary": "production_summary.json", "intent_ledger": "intent_ledger.json"},
        metadata={"mode": "paper", "status": "paper_replay_completed", "run_id": "run_2"},
        published_at="2026-06-10T21:30:00Z",
    )

    manifest_path = tmp_path / "production" / "paper_sim_replay_v1" / "production_manifest.json"
    registry_path = tmp_path / "production" / "production_registry.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    current = registry["strategies"]["paper_sim_replay_v1"]["current"]
    history = registry["strategies"]["paper_sim_replay_v1"]["history"]

    assert first["schema_version"] == "production.manifest.v1"
    assert second["schema_version"] == "production.manifest.v1"
    assert manifest["manifest_hash"] == second["manifest_hash"]
    assert registry["schema_version"] == "production.registry.v1"
    assert current["strategy_id"] == "paper_sim_replay_v1"
    assert current["manifest_path"].endswith("production/paper_sim_replay_v1/production_manifest.json")
    assert current["manifest_hash"] == second["manifest_hash"]
    assert current["status"] == "paper_replay_completed"
    assert current["mode"] == "paper"
    assert current["published_at"] == "2026-06-10T21:30:00Z"
    assert [item["manifest_hash"] for item in history] == [first["manifest_hash"], second["manifest_hash"]]
    assert len(registry["registry_hash"]) == 64


def test_production_manifest_content_hash_recomputes_saved_payload_hash(tmp_path):
    payload = publish_production_manifest(
        strategy_id="paper_sim_replay_v1",
        output_dir=tmp_path / "production" / "paper_sim_replay_v1",
        registry_path=None,
        notebook_path="paper_sim_replay_v1.ipynb",
        dataset=_dataset(),
        paths={"summary": "production_summary.json"},
        metadata={"mode": "paper", "status": "paper_replay_completed"},
        published_at="2026-06-09T21:30:00Z",
    )

    tampered = dict(payload)
    tampered["metadata"] = {**tampered["metadata"], "status": "paper_replay_changed"}

    assert production_manifest_content_hash(payload) == payload["manifest_hash"]
    assert production_manifest_content_hash(tampered) != payload["manifest_hash"]

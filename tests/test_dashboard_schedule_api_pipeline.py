from __future__ import annotations

import json

from src.api import PaperBrokerAdapter
from src.dashboard import load_model_matrix_dashboard
from src.schedule import ScheduleJob, run_job_once


def test_dashboard_loads_quantbench_matrix_sidecars_without_raw_model_internals():
    payload = load_model_matrix_dashboard("research/quantbench_model_matrix_smoke_v1")

    assert payload["summary"]["model_count"] == 32
    assert payload["top_models"][0]["rank"] == 1
    assert "model_comparison" in payload["artifacts"]
    assert "predictions" not in payload["artifacts"]


def test_schedule_run_job_once_uses_lock_and_reports_notification(tmp_path):
    events: list[str] = []
    calls = {"count": 0}

    def task():
        calls["count"] += 1
        return {"rows": 32}

    job = ScheduleJob(name="dashboard_export", task="refresh_dashboard", cron="manual")
    result = run_job_once(job, task=task, lock_dir=tmp_path, notify=events.append)

    assert result.status == "success"
    assert result.payload == {"rows": 32}
    assert calls["count"] == 1
    assert events == ["dashboard_export:success"]
    assert not (tmp_path / "dashboard_export.lock").exists()


def test_schedule_run_job_once_skips_when_lock_exists(tmp_path):
    (tmp_path / "dashboard_export.lock").write_text(json.dumps({"owner": "test"}), encoding="utf-8")
    job = ScheduleJob(name="dashboard_export", task="refresh_dashboard", cron="manual")
    result = run_job_once(job, task=lambda: {"should_not_run": True}, lock_dir=tmp_path)

    assert result.status == "locked"
    assert result.payload == {}


def test_paper_broker_places_target_position_and_reports_exposure():
    broker = PaperBrokerAdapter(cash=10000.0, shares=0)
    ok, detail = broker.healthcheck()

    snapshot = broker.place_target_position(target_position=0.35, price=100.0)

    assert ok is True
    assert "ready" in detail
    assert snapshot.shares == 35
    assert snapshot.cash == 6500.0
    assert abs(snapshot.position - 0.35) < 1e-12

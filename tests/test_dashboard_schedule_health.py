from __future__ import annotations

import json

from src.dashboard import build_schedule_health, write_schedule_health
from src.schedule import JobResult


def test_build_schedule_health_summarizes_failed_retry_metadata():
    result = JobResult(
        name="model_dashboard_refresh",
        status="failed",
        payload={
            "attempts": 2,
            "retry_schema_version": "schedule.retry_attempts.v1",
            "retry_error_count": 2,
            "retry_errors": [
                {"attempt": 1, "error_type": "RuntimeError", "message": "transient failure"},
                {"attempt": 2, "error_type": "RuntimeError", "message": "permanent failure"},
            ],
        },
        detail="permanent failure",
    )

    health = build_schedule_health([result])

    assert health["schema_version"] == "dashboard.schedule_health.v1"
    assert health["job_count"] == 1
    assert health["failed_job_count"] == 1
    assert health["total_attempts"] == 2
    assert health["total_retry_error_count"] == 2
    assert health["top_failure_job"] == "model_dashboard_refresh"
    assert health["top_failure_last_error"] == "RuntimeError:permanent failure"
    assert health["jobs"] == [
        {
            "name": "model_dashboard_refresh",
            "status": "failed",
            "attempts": 2,
            "retry_schema_version": "schedule.retry_attempts.v1",
            "retry_error_count": 2,
            "last_error_attempt": 2,
            "last_error_type": "RuntimeError",
            "last_error_message": "permanent failure",
            "detail": "permanent failure",
        }
    ]


def test_write_schedule_health_persists_dashboard_ready_artifact(tmp_path):
    output_path = tmp_path / "schedule_health.json"
    results = [
        JobResult(
            name="data_update",
            status="success",
            payload={
                "attempts": 1,
                "retry_schema_version": "schedule.retry_attempts.v1",
                "retry_error_count": 0,
                "retry_errors": [],
            },
        ),
        JobResult(
            name="model_dashboard_refresh",
            status="failed",
            payload={
                "attempts": 2,
                "retry_schema_version": "schedule.retry_attempts.v1",
                "retry_error_count": 2,
                "retry_errors": [
                    {"attempt": 1, "error_type": "RuntimeError", "message": "temporary dashboard failure"},
                    {"attempt": 2, "error_type": "RuntimeError", "message": "permanent dashboard failure"},
                ],
            },
            detail="permanent dashboard failure",
        ),
    ]

    payload = write_schedule_health(results, output_path=output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == payload
    assert saved["schema_version"] == "dashboard.schedule_health.v1"
    assert saved["job_count"] == 2
    assert saved["failed_job_count"] == 1
    assert saved["top_failure_job"] == "model_dashboard_refresh"
    assert saved["top_failure_last_error"] == "RuntimeError:permanent dashboard failure"

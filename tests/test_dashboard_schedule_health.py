from __future__ import annotations

from src.dashboard import build_schedule_health
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

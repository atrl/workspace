from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


SCHEDULE_HEALTH_SCHEMA_VERSION = "dashboard.schedule_health.v1"


def _field(result: Any, key: str, default: Any = "") -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _payload(result: Any) -> dict[str, Any]:
    value = _field(result, "payload", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _last_retry_error(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors = payload.get("retry_errors")
    rows = errors if isinstance(errors, list) else []
    last = rows[-1] if rows and isinstance(rows[-1], Mapping) else {}
    return dict(last)


def _job_row(result: Any) -> dict[str, Any]:
    payload = _payload(result)
    last_error = _last_retry_error(payload)
    return {
        "name": str(_field(result, "name", "")),
        "status": str(_field(result, "status", "")),
        "attempts": int(payload.get("attempts", 0) or 0),
        "retry_schema_version": str(payload.get("retry_schema_version", "")),
        "retry_error_count": int(payload.get("retry_error_count", 0) or 0),
        "last_error_attempt": int(last_error.get("attempt", 0) or 0),
        "last_error_type": str(last_error.get("error_type", "")),
        "last_error_message": str(last_error.get("message", "")),
        "detail": str(_field(result, "detail", "")),
    }


def build_schedule_health(results: Iterable[Any]) -> dict[str, Any]:
    """Build a dashboard-readable schedule health summary from job results."""

    rows = [_job_row(result) for result in results]
    failed_rows = [row for row in rows if row["status"] == "failed"]
    ordered_failures = sorted(
        failed_rows,
        key=lambda row: (int(row.get("retry_error_count", 0)), int(row.get("attempts", 0))),
        reverse=True,
    )
    top_failure = ordered_failures[0] if ordered_failures else {}
    top_error = ""
    if top_failure and top_failure.get("last_error_type"):
        top_error = f"{top_failure.get('last_error_type', '')}:{top_failure.get('last_error_message', '')}"
    return {
        "schema_version": SCHEDULE_HEALTH_SCHEMA_VERSION,
        "job_count": int(len(rows)),
        "failed_job_count": int(len(failed_rows)),
        "total_attempts": int(sum(int(row.get("attempts", 0) or 0) for row in rows)),
        "total_retry_error_count": int(sum(int(row.get("retry_error_count", 0) or 0) for row in rows)),
        "top_failure_job": str(top_failure.get("name", "")) if top_failure else "",
        "top_failure_last_error": top_error,
        "jobs": rows,
    }

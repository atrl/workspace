from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ScheduleJob:
    name: str
    task: str
    cron: str
    enabled: bool = True


@dataclass(frozen=True)
class JobResult:
    name: str
    status: str
    payload: dict[str, Any]
    detail: str = ""


class ScheduleRetryError(RuntimeError):
    def __init__(self, message: str, *, payload: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.payload = dict(payload)


def run_task_with_retries(
    task: Callable[[], dict[str, Any]],
    *,
    retries: int = 0,
    failure_message: str = "task failed",
) -> dict[str, Any]:
    retry_errors: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(max(int(retries), 0) + 1):
        attempt_number = attempt + 1
        try:
            payload = dict(task())
            payload["attempts"] = attempt_number
            payload["retry_schema_version"] = "schedule.retry_attempts.v1"
            payload["retry_error_count"] = int(len(retry_errors))
            payload["retry_errors"] = retry_errors
            return payload
        except Exception as exc:
            last_error = exc
            retry_errors.append(
                {
                    "attempt": attempt_number,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            if attempt >= max(int(retries), 0):
                break
    failure_payload = {
        "attempts": int(len(retry_errors)),
        "retry_schema_version": "schedule.retry_attempts.v1",
        "retry_error_count": int(len(retry_errors)),
        "retry_errors": retry_errors,
    }
    raise ScheduleRetryError(str(last_error) if last_error else failure_message, payload=failure_payload)


def format_job_result_notification(result: JobResult) -> str:
    if result.status == "skipped":
        decision = result.payload.get("market_calendar_decision")
        if isinstance(decision, Mapping):
            parts = [f"{result.name}:skipped"]
            market_status = str(decision.get("market_status", ""))
            skip_reason = str(decision.get("skip_reason") or result.detail)
            next_trading_date = str(decision.get("next_trading_date", ""))
            if market_status:
                parts.append(f"market_status={market_status}")
            if skip_reason:
                parts.append(f"skip_reason={skip_reason}")
            if next_trading_date:
                parts.append(f"next_trading_date={next_trading_date}")
            return " | ".join(parts)
        if result.detail:
            return f"{result.name}:skipped | skip_reason={result.detail}"
    return f"{result.name}:{result.status}"


def run_job_once(
    job: ScheduleJob,
    *,
    task: Callable[[], dict[str, Any]],
    lock_dir: str | Path,
    notify: Callable[[str], None] | None = None,
    market_calendar_decision: Mapping[str, Any] | None = None,
    notify_skipped: bool = False,
) -> JobResult:
    if not job.enabled:
        result = JobResult(name=job.name, status="skipped", payload={}, detail="disabled")
        if notify is not None and notify_skipped:
            notify(format_job_result_notification(result))
        return result

    if market_calendar_decision is not None and not bool(market_calendar_decision.get("should_run", True)):
        decision = dict(market_calendar_decision)
        detail = str(decision.get("skip_reason") or decision.get("market_status") or "market_calendar")
        result = JobResult(
            name=job.name,
            status="skipped",
            payload={"market_calendar_decision": decision},
            detail=detail,
        )
        if notify is not None and notify_skipped:
            notify(format_job_result_notification(result))
        return result

    lock_root = Path(lock_dir)
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{job.name}.lock"
    if lock_path.exists():
        return JobResult(name=job.name, status="locked", payload={}, detail=str(lock_path))
    lock_path.write_text(json.dumps({"job": job.name, "task": job.task}, ensure_ascii=False), encoding="utf-8")
    try:
        payload = task()
        result = JobResult(name=job.name, status="success", payload=dict(payload))
        if notify is not None:
            notify(f"{job.name}:success")
        return result
    except Exception as exc:
        if notify is not None:
            notify(f"{job.name}:failed")
        payload = dict(exc.payload) if isinstance(exc, ScheduleRetryError) else {}
        return JobResult(name=job.name, status="failed", payload=payload, detail=str(exc))
    finally:
        lock_path.unlink(missing_ok=True)

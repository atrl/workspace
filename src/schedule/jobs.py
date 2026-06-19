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


def run_job_once(
    job: ScheduleJob,
    *,
    task: Callable[[], dict[str, Any]],
    lock_dir: str | Path,
    notify: Callable[[str], None] | None = None,
    market_calendar_decision: Mapping[str, Any] | None = None,
) -> JobResult:
    if not job.enabled:
        return JobResult(name=job.name, status="skipped", payload={}, detail="disabled")

    if market_calendar_decision is not None and not bool(market_calendar_decision.get("should_run", True)):
        decision = dict(market_calendar_decision)
        detail = str(decision.get("skip_reason") or decision.get("market_status") or "market_calendar")
        return JobResult(
            name=job.name,
            status="skipped",
            payload={"market_calendar_decision": decision},
            detail=detail,
        )

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
        return JobResult(name=job.name, status="failed", payload={}, detail=str(exc))
    finally:
        lock_path.unlink(missing_ok=True)

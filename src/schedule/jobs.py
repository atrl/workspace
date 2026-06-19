from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RefreshPlan:
    name: str
    lock_path: Path
    attempts: int
    delay_seconds: float


class JobLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __enter__(self) -> "JobLock":
        if self.path.exists():
            raise RuntimeError(f"job already locked: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(time.time()), encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            self.path.unlink()


def retry(func: Callable[[], T], *, attempts: int, delay_seconds: float) -> T:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return func()
        except Exception as error:  # noqa: BLE001 - job retry boundary captures and retries failures.
            last_error = error
            if attempt < attempts - 1 and delay_seconds:
                time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def run_refresh_plan(
    plan: RefreshPlan,
    *,
    data_update: Callable[[], list[Any]],
    model_refresh: Callable[[], Any],
    dashboard_export: Callable[[list[Any], Any], dict],
    notify: Callable[[dict], None],
) -> dict:
    with JobLock(plan.lock_path):
        data_attempts = {"count": 0}

        def counted_data_update() -> list[Any]:
            data_attempts["count"] += 1
            return data_update()

        manifests = retry(counted_data_update, attempts=plan.attempts, delay_seconds=plan.delay_seconds)
        model_comparison = model_refresh()
        dashboard_payload = dashboard_export(manifests, model_comparison)
        summary = {
            "name": plan.name,
            "status": "ok",
            "attempts": data_attempts["count"],
            "artifacts": len(manifests),
            "dashboard_artifacts": len(dashboard_payload.get("artifacts", [])),
        }
        notify(summary)
        return summary

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class RefreshPlan:
    name: str
    lock_path: Path
    attempts: int = 1
    delay_seconds: float = 0.0


def _validate_qlib_manifests(manifests: list[dict]) -> None:
    for manifest in manifests:
        source = manifest.get("source", "")
        if not str(source).startswith("qlib"):
            raise ValueError("scheduled refresh outputs must be qlib-derived manifests")


def run_refresh_plan(
    plan: RefreshPlan,
    *,
    data_refresh: Callable[[], list[dict]],
    dashboard_export: Callable[[list[dict]], dict],
) -> dict:
    plan.lock_path.parent.mkdir(parents=True, exist_ok=True)
    if plan.lock_path.exists():
        raise RuntimeError(f"refresh lock already exists: {plan.lock_path}")
    plan.lock_path.write_text(plan.name, encoding="utf-8")
    attempts = 0
    try:
        last_error: Exception | None = None
        for attempt in range(1, plan.attempts + 1):
            attempts = attempt
            try:
                manifests = data_refresh()
                _validate_qlib_manifests(manifests)
                export = dashboard_export(manifests)
                return {"status": "ok", "attempts": attempts, "artifacts": len(manifests), "export": export}
            except Exception as exc:  # pragma: no cover - exercised by caller-specific retry tests.
                last_error = exc
                if attempt < plan.attempts:
                    time.sleep(plan.delay_seconds)
        raise RuntimeError(f"blocked: qlib refresh failed: {last_error}") from last_error
    finally:
        plan.lock_path.unlink(missing_ok=True)

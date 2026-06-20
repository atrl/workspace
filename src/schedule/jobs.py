from __future__ import annotations

import time
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


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
    dashboard_export: Callable[..., dict],
    model_refresh: Callable[[list[dict]], Any] | None = None,
    notify: Callable[[dict], None] | None = None,
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
                model_results = model_refresh(manifests) if model_refresh else []
                export = _call_dashboard_export(dashboard_export, manifests, model_results)
                summary = {
                    "status": "ok",
                    "attempts": attempts,
                    "artifacts": len(manifests),
                    "model_results": len(model_results),
                    "export": export,
                }
                if notify:
                    notify(summary)
                return summary
            except Exception as exc:  # pragma: no cover - exercised by caller-specific retry tests.
                last_error = exc
                if attempt < plan.attempts:
                    time.sleep(plan.delay_seconds)
        raise RuntimeError(f"blocked: qlib refresh failed: {last_error}") from last_error
    finally:
        plan.lock_path.unlink(missing_ok=True)


def _call_dashboard_export(callback: Callable[..., dict], manifests: list[dict], model_results: Any) -> dict:
    params = inspect.signature(callback).parameters
    if len(params) >= 2:
        return callback(manifests, model_results)
    return callback(manifests)

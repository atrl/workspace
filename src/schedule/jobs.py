from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RefreshPlan:
    name: str
    lock_path: Path
    attempts: int = 1
    delay_seconds: float = 0.0
    state_path: Path | None = None


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
    signal_refresh: Callable[..., Any] | None = None,
    notify: Callable[[dict], Any] | None = None,
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
                manifests = _as_list(data_refresh())
                _validate_qlib_manifests(manifests)
                model_results = _as_list(model_refresh(manifests)) if model_refresh else []
                trade_notifications = (
                    _as_list(_call_signal_refresh(signal_refresh, manifests, model_results)) if signal_refresh else []
                )
                export = _call_dashboard_export(dashboard_export, manifests, model_results)
                summary = {
                    "status": "ok",
                    "attempts": attempts,
                    "artifacts": len(manifests),
                    "model_results": len(model_results),
                    "trade_notifications": len(trade_notifications),
                    "data_manifest_details": _jsonable(manifests),
                    "model_result_details": _jsonable(model_results),
                    "trade_notification_details": _jsonable(trade_notifications),
                    "export": export,
                }
                if notify:
                    summary["notification_result"] = notify(summary)
                _write_state(plan, summary)
                return summary
            except Exception as exc:  # pragma: no cover - exercised by caller-specific retry tests.
                last_error = exc
                if attempt < plan.attempts:
                    time.sleep(plan.delay_seconds)
        _write_state(plan, {"status": "blocked", "attempts": attempts, "error": str(last_error)})
        raise RuntimeError(f"blocked: qlib refresh failed: {last_error}") from last_error
    finally:
        plan.lock_path.unlink(missing_ok=True)


def _call_dashboard_export(callback: Callable[..., dict], manifests: list[dict], model_results: Any) -> dict:
    params = inspect.signature(callback).parameters
    if len(params) >= 2:
        return callback(manifests, model_results)
    return callback(manifests)


def _call_signal_refresh(callback: Callable[..., Any], manifests: list[dict], model_results: Any) -> Any:
    params = inspect.signature(callback).parameters
    if len(params) >= 2:
        return callback(manifests, model_results)
    return callback(model_results)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _write_state(plan: RefreshPlan, payload: dict[str, Any]) -> None:
    if plan.state_path is None:
        return
    plan.state_path.parent.mkdir(parents=True, exist_ok=True)
    plan.state_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value

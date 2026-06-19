from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from src.analysis import backfill_analysis_manifest, scan_analysis_cards
from src.api import (
    PaperBrokerAdapter,
    PaperMarketDataAdapter,
    assert_paper_sim_mode,
    build_target_position_intent_from_dashboard,
    execute_paper_intent,
    write_intent_ledger,
)
from src.core import production_manifest_content_hash
from src.data import build_canonical_snapshot, publish_canonical_snapshot
from src.dashboard import build_dashboard_view_model, write_dashboard_html, write_model_matrix_dashboard_export
from src.model import run_factor_feature_set_matrix

from .jobs import JobResult, ScheduleJob, run_job_once


@dataclass(frozen=True)
class DataUpdatePipelineConfig:
    name: str
    dataset_key: str
    dataset_name: str
    universe: tuple[str, ...]
    output_dir: Path
    registry_path: Path
    calendar_path: str
    source_paths: tuple[str, ...]
    start: str
    end: str
    as_of: str
    date_column: str = "date"
    published_at: str | None = None
    notify_details: bool = False
    retries: int = 0


@dataclass(frozen=True)
class ModelDashboardPipelineConfig:
    name: str
    card_dir: Path
    dashboard_output_path: Path
    start: str
    end: str
    train_end: str
    model_names: tuple[str, ...] = ("ridge_mse", "random_forest_clf", "hist_gradient_mse")
    top_n_factors: int = 5
    dashboard_top_n: int = 10
    attribution_card_dir: Path | None = None
    production_health_path: Path | None = None
    dashboard_page_output_path: Path | None = None
    dashboard_page_title: str = "SOXL Dashboard"
    analysis_manifest_preflight: bool = False
    analysis_manifest_auto_backfill: bool = False
    analysis_manifest_dataset_name: str = "schedule_dashboard_refresh"
    analysis_manifest_universe: tuple[str, ...] = ("SOXL",)
    analysis_manifest_calendar_path: str = ""
    notify_details: bool = False
    retries: int = 0


@dataclass(frozen=True)
class ProductionHealthCheckConfig:
    name: str
    registry_path: Path
    strategy_id: str
    output_path: Path | None = None
    required_paths: tuple[str, ...] = (
        "notebook",
        "summary",
        "dashboard_export",
        "intent_ledger",
        "paper_execution_result",
        "data_registry",
        "analysis_manifest",
    )
    notify_details: bool = False
    retries: int = 0


@dataclass(frozen=True)
class PaperExecutionPipelineConfig:
    name: str
    dashboard_export_path: Path
    data_registry_path: Path
    dataset_key: str
    instrument: str
    intent_ledger_path: Path
    paper_execution_result_path: Path
    mode: str = "paper"
    min_position: float = 0.0
    max_position: float = 1.0
    price_field: str = "close"
    as_of: str | None = None
    initial_cash: float = 10000.0
    initial_shares: int = 0
    notify_details: bool = False
    retries: int = 0


MatrixRunner = Callable[..., Any]
DashboardExporter = Callable[..., dict[str, Any]]
DashboardPageExporter = Callable[..., dict[str, Any]]
FrameProvider = Callable[[DataUpdatePipelineConfig], pd.DataFrame]
CalendarProvider = Callable[[DataUpdatePipelineConfig], pd.DatetimeIndex | list[str] | tuple[str, ...]]


def _default_frame_provider(config: DataUpdatePipelineConfig) -> pd.DataFrame:
    if not config.source_paths:
        raise ValueError("source_paths must contain at least one CSV path for default data update")
    path = Path(config.source_paths[0])
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def _default_calendar_provider(config: DataUpdatePipelineConfig) -> pd.DatetimeIndex:
    path = Path(config.calendar_path)
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DatetimeIndex(pd.to_datetime(values), name="date")


def _data_update_attempt(
    config: DataUpdatePipelineConfig,
    *,
    frame_provider: FrameProvider,
    calendar_provider: CalendarProvider,
) -> dict[str, Any]:
    frame = frame_provider(config)
    calendar = calendar_provider(config)
    snapshot = build_canonical_snapshot(
        frame,
        name=config.dataset_name,
        universe=config.universe,
        calendar=calendar,
        calendar_path=config.calendar_path,
        source_paths=config.source_paths,
        start=config.start,
        end=config.end,
        as_of=config.as_of,
        date_column=config.date_column,
    )
    manifest = publish_canonical_snapshot(
        snapshot,
        output_dir=config.output_dir,
        dataset_key=config.dataset_key,
        registry_path=config.registry_path,
        published_at=config.published_at,
    )
    metadata = manifest["dataset"].get("metadata", {})
    return {
        "schema_version": "schedule.data_update_result.v1",
        "dataset_key": config.dataset_key,
        "dataset_name": config.dataset_name,
        "status": manifest.get("status", ""),
        "manifest_path": str(Path(config.output_dir) / manifest["paths"]["manifest"]),
        "manifest_hash": manifest["manifest_hash"],
        "dataset_schema_hash": manifest["dataset_schema_hash"],
        "registry_path": str(config.registry_path),
        "start": manifest["dataset"].get("start", ""),
        "end": manifest["dataset"].get("end", ""),
        "as_of": metadata.get("as_of", ""),
        "row_count": int(metadata.get("row_count", 0) or 0),
        "column_count": int(metadata.get("column_count", 0) or 0),
        "frame_schema_hash": str(metadata.get("frame_schema_hash", "")),
        "dropped_non_calendar_rows": int(metadata.get("dropped_non_calendar_rows", 0) or 0),
        "published_at": manifest.get("published_at", ""),
    }


def _data_update_task_with_retries(
    config: DataUpdatePipelineConfig,
    *,
    frame_provider: FrameProvider,
    calendar_provider: CalendarProvider,
) -> dict[str, Any]:
    attempts = 0
    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        attempts = attempt + 1
        try:
            payload = _data_update_attempt(
                config,
                frame_provider=frame_provider,
                calendar_provider=calendar_provider,
            )
            payload["attempts"] = attempts
            return payload
        except Exception as exc:
            last_error = exc
            if attempt >= config.retries:
                break
    raise RuntimeError(str(last_error) if last_error else "data update failed")


def _data_update_notification(job_name: str, status: str, payload: dict[str, Any]) -> str:
    if status != "success":
        return f"{job_name}:{status}"
    return (
        f"{job_name}:success"
        f" | dataset={payload.get('dataset_key', '')}"
        f" | rows={payload.get('row_count', 0)}"
        f" | as_of={payload.get('as_of', '')}"
        f" | attempts={payload.get('attempts', '')}"
        f" | manifest={payload.get('manifest_path', '')}"
    )


def run_data_update_once(
    config: DataUpdatePipelineConfig,
    *,
    lock_dir: str | Path,
    notify: Callable[[str], None] | None = None,
    frame_provider: FrameProvider = _default_frame_provider,
    calendar_provider: CalendarProvider = _default_calendar_provider,
) -> JobResult:
    job = ScheduleJob(name=config.name, task="data_update", cron="manual")
    latest_payload: dict[str, Any] = {}

    def task() -> dict[str, Any]:
        nonlocal latest_payload
        latest_payload = _data_update_task_with_retries(
            config,
            frame_provider=frame_provider,
            calendar_provider=calendar_provider,
        )
        return latest_payload

    def notify_with_details(message: str) -> None:
        if notify is None:
            return
        if not config.notify_details:
            notify(message)
            return
        _, _, status = message.partition(":")
        notify(_data_update_notification(config.name, status, latest_payload))

    return run_job_once(
        job,
        task=task,
        lock_dir=lock_dir,
        notify=notify_with_details if notify is not None else None,
    )


def _resolve_declared_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def _production_health_attempt(config: ProductionHealthCheckConfig) -> dict[str, Any]:
    registry_path = Path(config.registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    strategy_entry = registry.get("strategies", {}).get(config.strategy_id)
    if not isinstance(strategy_entry, dict):
        raise ValueError(f"production registry missing strategy_id: {config.strategy_id}")
    current = strategy_entry.get("current")
    if not isinstance(current, dict):
        raise ValueError(f"production registry current pointer missing: {config.strategy_id}")

    manifest_path = _resolve_declared_path(str(current.get("manifest_path", "")), base_dir=registry_path.parent)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent
    manifest_hash = str(manifest.get("manifest_hash", ""))
    current_hash = str(current.get("manifest_hash", ""))
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}

    missing_required_paths: list[str] = []
    resolved_paths: dict[str, str] = {}
    for key in config.required_paths:
        declared = paths.get(key)
        if not declared:
            missing_required_paths.append(str(key))
            continue
        resolved = _resolve_declared_path(str(declared), base_dir=manifest_dir)
        resolved_paths[str(key)] = str(resolved)
        if not resolved.exists():
            missing_required_paths.append(str(key))

    computed_manifest_hash = production_manifest_content_hash(manifest)
    registry_hash_matches = not current_hash or current_hash == manifest_hash
    content_hash_matches = not manifest_hash or computed_manifest_hash == manifest_hash
    health_status = "healthy"
    if not registry_hash_matches or not content_hash_matches:
        health_status = "manifest_hash_mismatch"
    elif missing_required_paths:
        health_status = "missing_required_paths"

    payload = {
        "schema_version": "schedule.production_health.v1",
        "strategy_id": config.strategy_id,
        "health_status": health_status,
        "is_healthy": health_status == "healthy",
        "registry_path": str(registry_path),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_hash,
        "registry_manifest_hash": current_hash,
        "computed_manifest_hash": computed_manifest_hash,
        "manifest_hash_matches_registry": registry_hash_matches,
        "manifest_hash_matches_content": content_hash_matches,
        "artifact_id": str(manifest.get("artifact_id", "")),
        "artifact_type": str(manifest.get("artifact_type", "")),
        "mode": str(metadata.get("mode", "")),
        "status": str(metadata.get("status", "")),
        "live_broker_enabled": bool(metadata.get("live_broker_enabled", False)),
        "paper_execution_pipeline_schema_version": str(metadata.get("paper_execution_pipeline_schema_version", "")),
        "paper_execution_job_status": str(metadata.get("paper_execution_job_status", "")),
        "published_at": str(manifest.get("published_at") or current.get("published_at", "")),
        "start": str(current.get("start", "")),
        "end": str(current.get("end", "")),
        "required_path_count": int(len(config.required_paths)),
        "checked_path_count": int(len(config.required_paths)),
        "missing_required_paths": missing_required_paths,
        "resolved_paths": resolved_paths,
        "dashboard_export_path": resolved_paths.get("dashboard_export", ""),
        "intent_ledger_path": resolved_paths.get("intent_ledger", ""),
        "paper_execution_result_path": resolved_paths.get("paper_execution_result", ""),
        "output_path": str(config.output_path) if config.output_path is not None else "",
    }
    return payload


def _production_health_task_with_retries(config: ProductionHealthCheckConfig) -> dict[str, Any]:
    attempts = 0
    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        attempts = attempt + 1
        try:
            payload = _production_health_attempt(config)
            payload["attempts"] = attempts
            if config.output_path is not None:
                output = Path(config.output_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt >= config.retries:
                break
    raise RuntimeError(str(last_error) if last_error else "production health check failed")


def _production_health_notification(job_name: str, status: str, payload: dict[str, Any]) -> str:
    if status != "success":
        return f"{job_name}:{status}"
    paper_execution_schema = str(payload.get("paper_execution_pipeline_schema_version", ""))
    paper_execution_status = str(payload.get("paper_execution_job_status", ""))
    paper_execution = ""
    if paper_execution_schema or paper_execution_status:
        paper_execution = f" | paper_execution={paper_execution_schema}/{paper_execution_status}"
    return (
        f"{job_name}:success"
        f" | strategy={payload.get('strategy_id', '')}"
        f" | health={payload.get('health_status', '')}"
        f" | missing={len(payload.get('missing_required_paths', []))}"
        f"{paper_execution}"
        f" | manifest={payload.get('manifest_path', '')}"
    )


def run_production_health_check_once(
    config: ProductionHealthCheckConfig,
    *,
    lock_dir: str | Path,
    notify: Callable[[str], None] | None = None,
) -> JobResult:
    job = ScheduleJob(name=config.name, task="production_health_check", cron="manual")
    latest_payload: dict[str, Any] = {}

    def task() -> dict[str, Any]:
        nonlocal latest_payload
        latest_payload = _production_health_task_with_retries(config)
        return latest_payload

    def notify_with_details(message: str) -> None:
        if notify is None:
            return
        if not config.notify_details:
            notify(message)
            return
        _, _, status = message.partition(":")
        notify(_production_health_notification(config.name, status, latest_payload))

    return run_job_once(
        job,
        task=task,
        lock_dir=lock_dir,
        notify=notify_with_details if notify is not None else None,
    )


def _snapshot_payload(snapshot: Any) -> dict[str, Any]:
    return {
        "cash": float(snapshot.cash),
        "shares": int(snapshot.shares),
        "price": float(snapshot.price),
        "account_value": float(snapshot.account_value),
        "position": float(snapshot.position),
    }


def _read_dashboard_export(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dashboard export must be a JSON object")
    return payload


def _paper_execution_attempt(config: PaperExecutionPipelineConfig) -> dict[str, Any]:
    assert_paper_sim_mode(config.mode, surface="paper execution pipeline")

    dashboard_payload = _read_dashboard_export(Path(config.dashboard_export_path))
    market_data = PaperMarketDataAdapter.from_registry(
        config.data_registry_path,
        dataset_key=config.dataset_key,
        as_of=config.as_of,
        price_field=config.price_field,
    )
    quote = market_data.latest_quote(config.instrument)
    intent = build_target_position_intent_from_dashboard(
        dashboard_payload,
        instrument=config.instrument,
        min_position=config.min_position,
        max_position=config.max_position,
        mode=config.mode,
    )
    broker = PaperBrokerAdapter(cash=config.initial_cash, shares=config.initial_shares)
    result = execute_paper_intent(intent, broker=broker, price=quote.price)

    write_intent_ledger(config.intent_ledger_path, intent=intent, result=result)
    output = Path(config.paper_execution_result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = intent.metadata or {}
    return {
        "schema_version": "schedule.paper_execution.v1",
        "status": result.status,
        "live_broker_enabled": False,
        "mode": config.mode,
        "guardrail": intent.guardrail,
        "instrument": intent.instrument,
        "dataset_key": config.dataset_key,
        "dashboard_export_path": str(config.dashboard_export_path),
        "data_registry_path": str(config.data_registry_path),
        "intent_ledger_path": str(config.intent_ledger_path),
        "paper_execution_result_path": str(config.paper_execution_result_path),
        "source_model": intent.source_model,
        "source_artifact": intent.source_artifact,
        "target_position": float(intent.target_position),
        "execution_gate": str(metadata.get("execution_gate", "")),
        "quote": quote.to_dict(),
        "broker_name": result.broker_name,
        "snapshot": _snapshot_payload(result.snapshot),
    }


def _paper_execution_task_with_retries(config: PaperExecutionPipelineConfig) -> dict[str, Any]:
    attempts = 0
    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        attempts = attempt + 1
        try:
            payload = _paper_execution_attempt(config)
            payload["attempts"] = attempts
            return payload
        except Exception as exc:
            last_error = exc
            if attempt >= config.retries:
                break
    raise RuntimeError(str(last_error) if last_error else "paper execution failed")


def _paper_execution_notification(job_name: str, status: str, payload: dict[str, Any]) -> str:
    if status != "success":
        return f"{job_name}:{status}"
    quote = payload.get("quote") if isinstance(payload.get("quote"), dict) else {}
    snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}
    return (
        f"{job_name}:success"
        f" | mode={payload.get('mode', '')}"
        f" | gate={payload.get('execution_gate', '')}"
        f" | model={payload.get('source_model', '')}"
        f" | target={float(payload.get('target_position', 0.0) or 0.0):g}"
        f" | quote={quote.get('instrument', '')}@{quote.get('date', '')}/"
        f"{float(quote.get('price', 0.0) or 0.0):g}"
        f" | shares={int(snapshot.get('shares', 0) or 0)}"
        f" | ledger={payload.get('intent_ledger_path', '')}"
    )


def run_paper_execution_once(
    config: PaperExecutionPipelineConfig,
    *,
    lock_dir: str | Path,
    notify: Callable[[str], None] | None = None,
) -> JobResult:
    job = ScheduleJob(name=config.name, task="paper_execution", cron="manual")
    latest_payload: dict[str, Any] = {}

    def task() -> dict[str, Any]:
        nonlocal latest_payload
        latest_payload = _paper_execution_task_with_retries(config)
        return latest_payload

    def notify_with_details(message: str) -> None:
        if notify is None:
            return
        if not config.notify_details:
            notify(message)
            return
        _, _, status = message.partition(":")
        notify(_paper_execution_notification(config.name, status, latest_payload))

    return run_job_once(
        job,
        task=task,
        lock_dir=lock_dir,
        notify=notify_with_details if notify is not None else None,
    )


def _default_matrix_runner(
    *,
    card_dir: Path,
    start: str,
    end: str,
    train_end: str,
    model_names: Iterable[str],
    top_n_factors: int,
) -> Any:
    return run_factor_feature_set_matrix(
        card_dir=card_dir,
        start=start,
        end=end,
        train_end=train_end,
        model_names=tuple(model_names),
        top_n_factors=top_n_factors,
    )


def _default_dashboard_exporter(
    card_dir: Path,
    *,
    output_path: Path,
    top_n: int,
    attribution_card_dir: Path | None = None,
    production_health_path: Path | None = None,
) -> dict[str, Any]:
    return write_model_matrix_dashboard_export(
        card_dir,
        output_path=output_path,
        top_n=top_n,
        attribution_card_dir=attribution_card_dir,
        production_health_path=production_health_path,
    )


def _default_dashboard_page_exporter(
    dashboard_payload: dict[str, Any],
    *,
    output_path: Path,
    title: str,
) -> dict[str, Any]:
    view_model = build_dashboard_view_model(dashboard_payload)
    html = write_dashboard_html(view_model, output_path=output_path, title=title)
    return {
        "schema_version": "dashboard.static_page.v1",
        "output_path": str(output_path),
        "view_model_schema_version": str(view_model.get("schema_version", "")),
        "model_card_count": int(len(view_model.get("model_cards", []))),
        "html_size_bytes": len(html.encode("utf-8")),
    }


def _read_manifest_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "manifest_path": str(path),
        "manifest_hash": str(payload.get("manifest_hash", "")),
        "manifest_producer": str(payload.get("producer", "")),
    }


def _analysis_manifest_preflight(config: ModelDashboardPipelineConfig) -> dict[str, Any]:
    if not config.analysis_manifest_preflight:
        return {}

    card = Path(config.card_dir)
    scan = scan_analysis_cards(card)
    row = scan.iloc[0].to_dict() if not scan.empty else {}
    public_count = int(row.get("public_sidecar_count", 0) or 0)
    private_count = int(row.get("private_sidecar_count", 0) or 0)
    status = str(row.get("status", "missing_analysis_sidecars"))
    payload: dict[str, Any] = {
        "status": status,
        "action": "skip",
        "card_path": str(card),
        "public_sidecar_count": public_count,
        "private_sidecar_count": private_count,
    }

    manifest_path = card / "analysis_manifest.json"
    if status == "already_manifested":
        payload.update({"status": "ready", "action": "keep_manifest"})
        payload.update(_read_manifest_summary(manifest_path))
        return payload

    if status != "backfillable":
        payload["action"] = "skip"
        return payload

    if not config.analysis_manifest_auto_backfill:
        payload["status"] = "needs_manifest"
        payload["action"] = "review_backfill"
        return payload

    if not config.analysis_manifest_calendar_path:
        raise ValueError("analysis_manifest_calendar_path is required when analysis_manifest_auto_backfill=True")

    manifest = backfill_analysis_manifest(
        card,
        dataset_name=config.analysis_manifest_dataset_name,
        universe=config.analysis_manifest_universe,
        calendar_path=config.analysis_manifest_calendar_path,
        data_paths=(str(card),),
    )
    payload.update(
        {
            "status": "backfilled",
            "action": "backfilled",
            "manifest_path": str(manifest_path),
            "manifest_hash": str(manifest.get("manifest_hash", "")),
            "manifest_producer": str(manifest.get("producer", "")),
        }
    )
    return payload


def _prefixed_preflight_payload(preflight: dict[str, Any]) -> dict[str, Any]:
    return {f"analysis_manifest_preflight_{key}": value for key, value in preflight.items()}


def _top_failure_payload(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    failure_modes = dashboard_payload.get("failure_modes")
    if not isinstance(failure_modes, list) or not failure_modes:
        return {}
    top = failure_modes[0] if isinstance(failure_modes[0], dict) else {}
    if not top:
        return {}
    return {
        "top_failure_mode": str(top.get("failure_mode", "")),
        "top_failure_segment": str(top.get("segment", "")),
        "top_failure_model": str(top.get("model", "")),
        "top_failure_severity": float(top.get("severity_score", 0.0) or 0.0),
    }


def _top_review_payload(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    review_rows = dashboard_payload.get("model_review_briefing")
    if not isinstance(review_rows, list) or not review_rows:
        return {}
    sorted_rows = sorted(
        [row for row in review_rows if isinstance(row, dict)],
        key=lambda row: float(row.get("priority_rank", 999999) or 999999),
    )
    if not sorted_rows:
        return {}
    top = sorted_rows[0]
    return {
        "top_review_signal": str(top.get("review_signal", "")),
        "top_review_decision_hint": str(top.get("decision_hint", "")),
        "top_review_segment": str(top.get("segment", "")),
        "top_review_model": str(top.get("model", "")),
        "top_review_priority_rank": int(float(top.get("priority_rank", 0) or 0)),
    }


def _backtest_contract_payload(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    backtest = dashboard_payload.get("backtest")
    if not isinstance(backtest, dict):
        return {}
    config = backtest.get("config")
    splits = backtest.get("walk_forward_splits")
    if not isinstance(config, dict):
        return {}
    cost_model = config.get("cost_model")
    if not isinstance(cost_model, dict):
        cost_model = {}
    walk_forward_count = len(splits) if isinstance(splits, list) else 0
    return {
        "backtest_schema_version": str(config.get("schema_version", "")),
        "backtest_start": str(config.get("start", "")),
        "backtest_end": str(config.get("end", "")),
        "backtest_open_cost_bps": float(cost_model.get("open_cost_bps", 0.0) or 0.0),
        "backtest_close_cost_bps": float(cost_model.get("close_cost_bps", 0.0) or 0.0),
        "backtest_slippage_bps": float(cost_model.get("slippage_bps", 0.0) or 0.0),
        "walk_forward_fold_count": int(walk_forward_count),
    }


def _analysis_overview_payload(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    overview = dashboard_payload.get("analysis_overview")
    if not isinstance(overview, dict):
        return {}
    gaps = overview.get("largest_relative_return_spread_segments")
    top_gap = gaps[0] if isinstance(gaps, list) and gaps and isinstance(gaps[0], dict) else {}
    top_weakness = overview.get("top_weakness")
    if not isinstance(top_weakness, dict):
        top_weakness = {}
    return {
        "analysis_overview_schema_version": str(overview.get("schema_version", "")),
        "analysis_overview_baseline_model": str(overview.get("baseline_model", "")),
        "analysis_overview_top_weakness_model": str(top_weakness.get("model", "")),
        "analysis_overview_top_weakness_segment": str(top_weakness.get("segment", "")),
        "analysis_overview_top_weakness_decision_hint": str(top_weakness.get("decision_hint", "")),
        "analysis_overview_top_gap_segment": str(top_gap.get("segment", "")),
        "analysis_overview_top_gap_best_model": str(top_gap.get("best_relative_return_model", "")),
        "analysis_overview_top_gap_worst_model": str(top_gap.get("worst_relative_return_model", "")),
        "analysis_overview_top_gap_spread": float(top_gap.get("relative_return_pct_spread", 0.0) or 0.0),
    }


def _promotion_payload(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    objective = dashboard_payload.get("objective")
    if not isinstance(objective, dict):
        return {}
    rows = objective.get("promotion_scores")
    if not isinstance(rows, list) or not rows:
        return {}
    promotion_rows = [row for row in rows if isinstance(row, dict)]
    if not promotion_rows:
        return {}
    ordered = sorted(
        promotion_rows,
        key=lambda row: (
            int(float(row.get("rank", 999999) or 999999)),
            -float(row.get("promotion_score", 0.0) or 0.0),
        ),
    )
    top = ordered[0]
    return {
        "top_promotion_model": str(top.get("model", "")),
        "top_promotion_decision": str(top.get("promotion_decision", "")),
        "top_promotion_score": float(top.get("promotion_score", 0.0) or 0.0),
        "blocked_promotion_count": int(
            sum(1 for row in promotion_rows if str(row.get("promotion_decision", "")) == "blocked")
        ),
    }


def _model_run_coverage_payload(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    coverage = dashboard_payload.get("model_run_coverage")
    if not isinstance(coverage, dict):
        return {}
    expected = int(coverage.get("expected_model_count", coverage.get("ledger_model_count", 0)) or 0)
    trained = int(coverage.get("trained_model_count", 0) or 0)
    backtested = int(coverage.get("backtested_model_count", 0) or 0)
    missing = coverage.get("missing_models")
    missing_models = [str(value) for value in missing] if isinstance(missing, list) else []
    if expected <= 0 and trained <= 0 and backtested <= 0 and not missing_models:
        return {}
    return {
        "model_run_schema_version": str(coverage.get("schema_version", "")),
        "model_run_expected_model_count": expected,
        "model_run_trained_model_count": trained,
        "model_run_backtested_model_count": backtested,
        "model_run_all_completed": bool(coverage.get("all_models_completed", False)),
        "model_run_missing_model_count": int(len(missing_models)),
        "model_run_missing_models": ",".join(missing_models),
    }


def _requested_model_contract_payload(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    contract = dashboard_payload.get("requested_model_contract")
    if not isinstance(contract, dict):
        return {}
    requested = int(contract.get("requested_model_count", 0) or 0)
    completed = int(contract.get("completed_requested_model_count", 0) or 0)
    missing_values: list[str] = []
    for key in ("missing_from_registry", "missing_from_ledger"):
        values = contract.get(key)
        if isinstance(values, list):
            missing_values.extend(str(value) for value in values)
    incomplete = contract.get("incomplete_requested_models")
    incomplete_models = [str(value) for value in incomplete] if isinstance(incomplete, list) else []
    missing_models = list(dict.fromkeys(missing_values))
    if requested <= 0 and completed <= 0 and not missing_models and not incomplete_models:
        return {}
    return {
        "requested_model_schema_version": str(contract.get("schema_version", "")),
        "requested_model_count": requested,
        "requested_model_registered_count": int(contract.get("registered_model_count", 0) or 0),
        "requested_model_ledger_count": int(contract.get("ledger_model_count", 0) or 0),
        "requested_model_completed_count": completed,
        "requested_model_all_completed": bool(contract.get("all_requested_models_completed", False)),
        "requested_model_missing_count": int(len(missing_models)),
        "requested_model_incomplete_count": int(len(incomplete_models)),
        "requested_model_missing_models": ",".join(missing_models),
        "requested_model_incomplete_models": ",".join(incomplete_models),
    }


def _run_pipeline_attempt(
    config: ModelDashboardPipelineConfig,
    *,
    matrix_runner: MatrixRunner,
    dashboard_exporter: DashboardExporter,
    dashboard_page_exporter: DashboardPageExporter,
) -> dict[str, Any]:
    matrix_result = matrix_runner(
        card_dir=config.card_dir,
        start=config.start,
        end=config.end,
        train_end=config.train_end,
        model_names=config.model_names,
        top_n_factors=config.top_n_factors,
    )
    preflight = _analysis_manifest_preflight(config)
    dashboard_kwargs: dict[str, Any] = {
        "output_path": config.dashboard_output_path,
        "top_n": config.dashboard_top_n,
    }
    if config.attribution_card_dir is not None:
        dashboard_kwargs["attribution_card_dir"] = config.attribution_card_dir
    if config.production_health_path is not None:
        dashboard_kwargs["production_health_path"] = config.production_health_path
    dashboard_payload = dashboard_exporter(config.card_dir, **dashboard_kwargs)
    page_payload: dict[str, Any] = {}
    if config.dashboard_page_output_path is not None:
        page_payload = dashboard_page_exporter(
            dashboard_payload,
            output_path=config.dashboard_page_output_path,
            title=config.dashboard_page_title,
        )
    matrix_payload = matrix_result.summary if hasattr(matrix_result, "summary") else dict(matrix_result or {})
    manifest_source = dashboard_payload.get("manifest_source")
    manifest_payload = manifest_source if isinstance(manifest_source, dict) else {}
    return {
        "matrix_card": str(config.card_dir),
        "dashboard_export": str(config.dashboard_output_path),
        "dashboard_schema_version": str(dashboard_payload.get("schema_version", "")),
        "dashboard_artifact_id": str(dashboard_payload.get("artifact_id", "")),
        "dashboard_manifest_kind": str(manifest_payload.get("kind", "")),
        "dashboard_manifest_path": str(manifest_payload.get("path", "")),
        "dashboard_manifest_hash": str(manifest_payload.get("manifest_hash", "")),
        "dashboard_manifest_producer": str(manifest_payload.get("producer", "")),
        "dashboard_page": str(config.dashboard_page_output_path) if config.dashboard_page_output_path is not None else "",
        "dashboard_page_schema_version": str(page_payload.get("schema_version", "")),
        "dashboard_view_model_schema_version": str(page_payload.get("view_model_schema_version", "")),
        "dashboard_page_model_card_count": int(page_payload.get("model_card_count", 0) or 0),
        "top_model": str((dashboard_payload.get("top_models") or [{}])[0].get("model", "")),
        "attribution_card": str(config.attribution_card_dir) if config.attribution_card_dir is not None else "",
        "attribution_status": str(dashboard_payload.get("attribution", {}).get("status", "")),
        "matrix_status": str(matrix_payload.get("status", "")),
        "variant_count": int(matrix_payload.get("variant_count", 0) or 0),
        **_top_failure_payload(dashboard_payload),
        **_top_review_payload(dashboard_payload),
        **_backtest_contract_payload(dashboard_payload),
        **_analysis_overview_payload(dashboard_payload),
        **_promotion_payload(dashboard_payload),
        **_model_run_coverage_payload(dashboard_payload),
        **_requested_model_contract_payload(dashboard_payload),
        **_prefixed_preflight_payload(preflight),
    }


def _task_with_retries(
    config: ModelDashboardPipelineConfig,
    *,
    matrix_runner: MatrixRunner,
    dashboard_exporter: DashboardExporter,
    dashboard_page_exporter: DashboardPageExporter,
) -> dict[str, Any]:
    attempts = 0
    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        attempts = attempt + 1
        try:
            payload = _run_pipeline_attempt(
                config,
                matrix_runner=matrix_runner,
                dashboard_exporter=dashboard_exporter,
                dashboard_page_exporter=dashboard_page_exporter,
            )
            payload["attempts"] = attempts
            return payload
        except Exception as exc:
            last_error = exc
            if attempt >= config.retries:
                break
    raise RuntimeError(str(last_error) if last_error else "pipeline failed")


def _dashboard_refresh_notification(job_name: str, status: str, payload: dict[str, Any]) -> str:
    if status != "success":
        return f"{job_name}:{status}"
    message = (
        f"{job_name}:success"
        f" | top_model={payload.get('top_model', '')}"
        f" | dashboard_manifest={payload.get('dashboard_manifest_kind', '')}"
    )
    if "analysis_manifest_preflight_status" in payload:
        message += f" | analysis_preflight={payload.get('analysis_manifest_preflight_status', '')}"
    message += f" | attempts={payload.get('attempts', '')} | export={payload.get('dashboard_export', '')}"
    if payload.get("top_failure_mode"):
        message += f" | top_failure={payload.get('top_failure_mode')}@{payload.get('top_failure_segment', '')}"
    if payload.get("top_review_signal"):
        message += (
            f" | top_review={payload.get('top_review_signal')}:{payload.get('top_review_decision_hint', '')}"
            f"@{payload.get('top_review_segment', '')}"
        )
    if payload.get("backtest_schema_version"):
        message += (
            f" | backtest={payload.get('backtest_schema_version')} folds={payload.get('walk_forward_fold_count', 0)}"
            f" slippage_bps={payload.get('backtest_slippage_bps', 0.0)}"
        )
    if payload.get("analysis_overview_top_gap_segment"):
        message += (
            f" | regime_gap={payload.get('analysis_overview_top_gap_segment')}:"
            f"{payload.get('analysis_overview_top_gap_worst_model', '')}->"
            f"{payload.get('analysis_overview_top_gap_best_model', '')}"
            f"/spread={payload.get('analysis_overview_top_gap_spread', 0.0):g}"
        )
    if payload.get("top_promotion_model"):
        message += (
            f" | promotion={payload.get('top_promotion_decision', '')}:{payload.get('top_promotion_model', '')}"
            f" blocked={payload.get('blocked_promotion_count', 0)}"
        )
    if payload.get("model_run_schema_version"):
        expected = payload.get("model_run_expected_model_count", 0)
        message += (
            f" | run_coverage={payload.get('model_run_trained_model_count', 0)}/{expected}"
            f" backtested={payload.get('model_run_backtested_model_count', 0)}/{expected}"
            f" missing={payload.get('model_run_missing_model_count', 0)}"
        )
    if payload.get("requested_model_schema_version"):
        message += (
            f" | requested_models={payload.get('requested_model_completed_count', 0)}/"
            f"{payload.get('requested_model_count', 0)}"
            f" missing={payload.get('requested_model_missing_count', 0)}"
            f" incomplete={payload.get('requested_model_incomplete_count', 0)}"
        )
    return message


def run_model_dashboard_refresh_once(
    config: ModelDashboardPipelineConfig,
    *,
    lock_dir: str | Path,
    notify: Callable[[str], None] | None = None,
    matrix_runner: MatrixRunner = _default_matrix_runner,
    dashboard_exporter: DashboardExporter = _default_dashboard_exporter,
    dashboard_page_exporter: DashboardPageExporter = _default_dashboard_page_exporter,
) -> JobResult:
    job = ScheduleJob(name=config.name, task="model_dashboard_refresh", cron="manual")
    latest_payload: dict[str, Any] = {}

    def task() -> dict[str, Any]:
        nonlocal latest_payload
        latest_payload = _task_with_retries(
            config,
            matrix_runner=matrix_runner,
            dashboard_exporter=dashboard_exporter,
            dashboard_page_exporter=dashboard_page_exporter,
        )
        return latest_payload

    def notify_with_details(message: str) -> None:
        if notify is None:
            return
        if not config.notify_details:
            notify(message)
            return
        _, _, status = message.partition(":")
        notify(_dashboard_refresh_notification(config.name, status, latest_payload))

    return run_job_once(
        job,
        task=task,
        lock_dir=lock_dir,
        notify=notify_with_details if notify is not None else None,
    )

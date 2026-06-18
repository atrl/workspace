from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from src.dashboard import build_dashboard_view_model, write_dashboard_html, write_model_matrix_dashboard_export
from src.model import run_factor_feature_set_matrix

from .jobs import JobResult, ScheduleJob, run_job_once


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
    dashboard_page_output_path: Path | None = None
    dashboard_page_title: str = "SOXL Dashboard"
    retries: int = 0


MatrixRunner = Callable[..., Any]
DashboardExporter = Callable[..., dict[str, Any]]
DashboardPageExporter = Callable[..., dict[str, Any]]


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
) -> dict[str, Any]:
    return write_model_matrix_dashboard_export(
        card_dir,
        output_path=output_path,
        top_n=top_n,
        attribution_card_dir=attribution_card_dir,
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
    dashboard_kwargs: dict[str, Any] = {
        "output_path": config.dashboard_output_path,
        "top_n": config.dashboard_top_n,
    }
    if config.attribution_card_dir is not None:
        dashboard_kwargs["attribution_card_dir"] = config.attribution_card_dir
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
    return run_job_once(
        job,
        task=lambda: _task_with_retries(
            config,
            matrix_runner=matrix_runner,
            dashboard_exporter=dashboard_exporter,
            dashboard_page_exporter=dashboard_page_exporter,
        ),
        lock_dir=lock_dir,
        notify=notify,
    )

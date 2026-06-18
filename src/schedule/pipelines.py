from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from src.analysis import backfill_analysis_manifest, scan_analysis_cards
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
    analysis_manifest_preflight: bool = False
    analysis_manifest_auto_backfill: bool = False
    analysis_manifest_dataset_name: str = "schedule_dashboard_refresh"
    analysis_manifest_universe: tuple[str, ...] = ("SOXL",)
    analysis_manifest_calendar_path: str = ""
    notify_details: bool = False
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

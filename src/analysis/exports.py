from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.core import ArtifactManifest, DatasetSnapshot

from .failure_modes import build_failure_modes
from .qlib_ext import RegimeWindowEffectAnalysis, build_analysis_overview, build_model_review_briefing


ANALYSIS_REPORT_SCHEMA_VERSION = "analysis.report.v1"
ANALYSIS_REPORT_EXPORT_SCHEMA_VERSION = "analysis.report.export.v1"

DEFAULT_ANALYSIS_TABLE_FILENAMES = {
    "market_regimes": "market_regimes.csv",
    "market_regime_counts": "market_regime_counts.csv",
    "market_regime_daily_backtest": "market_regime_daily_backtest.csv",
    "model_comparison": "model_comparison.csv",
    "regime_metrics": "regime_metrics.csv",
    "window_metrics": "window_metrics.csv",
    "window_regime_metrics": "window_regime_metrics.csv",
    "qlib_return_frame": "qlib_return_frame.csv",
    "qlib_risk_metrics": "qlib_risk_metrics.csv",
    "segment_metrics": "segment_metrics.csv",
    "segment_effect_matrix": "segment_effect_matrix.csv",
    "model_effect_deltas": "model_effect_deltas.csv",
    "model_regime_window_diagnostics": "model_regime_window_diagnostics.csv",
    "model_review_briefing": "model_review_briefing.csv",
    "failure_modes": "failure_modes.csv",
    "qlib_portfolio_report": "qlib_portfolio_report.csv",
    "qlib_risk_analysis": "qlib_risk_analysis.csv",
    "qlib_positions": "qlib_positions.csv",
    "qlib_orders": "qlib_orders.csv",
    "qlib_indicator_analysis": "qlib_indicator_analysis.csv",
}


@dataclass(frozen=True)
class AnalysisReport:
    artifact_id: str
    summary: dict[str, Any]
    tables: Mapping[str, pd.DataFrame]
    artifact_type: str = "analysis_report"
    schema_version: str = ANALYSIS_REPORT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def table_counts(self) -> dict[str, int]:
        return {name: int(len(table)) for name, table in sorted(self.tables.items())}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return _json_safe(value.item())
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _analysis_layers(summary: Mapping[str, Any], table_names: list[str]) -> list[str]:
    existing = [str(value) for value in summary.get("analysis_layers", [])]
    layers = list(existing)
    for name in table_names:
        if name not in layers:
            layers.append(name)
    return layers


def build_regime_window_analysis_report(
    *,
    artifact_id: str,
    effect_analysis: RegimeWindowEffectAnalysis,
    model_comparison: pd.DataFrame | None = None,
    regime_metrics: pd.DataFrame | None = None,
    window_metrics: pd.DataFrame | None = None,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AnalysisReport:
    model_review_briefing = build_model_review_briefing(effect_analysis)
    tables: dict[str, pd.DataFrame] = {
        "qlib_return_frame": effect_analysis.qlib_return_frame,
        "qlib_risk_metrics": effect_analysis.qlib_risk_metrics,
        "segment_metrics": effect_analysis.segment_metrics,
        "segment_effect_matrix": effect_analysis.segment_effect_matrix,
        "window_regime_metrics": effect_analysis.window_regime_metrics,
        "model_effect_deltas": effect_analysis.model_effect_deltas,
        "model_regime_window_diagnostics": effect_analysis.model_regime_window_diagnostics,
        "model_review_briefing": model_review_briefing,
        "failure_modes": build_failure_modes(effect_analysis.model_effect_deltas),
    }
    if model_comparison is not None:
        tables["model_comparison"] = model_comparison
    if regime_metrics is not None:
        tables["regime_metrics"] = regime_metrics
    if window_metrics is not None:
        tables["window_metrics"] = window_metrics

    table_names = [name for name in DEFAULT_ANALYSIS_TABLE_FILENAMES if name in tables]
    summary_payload = dict(summary or {})
    summary_payload["schema_version"] = ANALYSIS_REPORT_SCHEMA_VERSION
    summary_payload["artifact_id"] = artifact_id
    summary_payload["artifact_type"] = "analysis_report"
    summary_payload.setdefault("status", "analysis_report_completed")
    summary_payload["analysis_layers"] = _analysis_layers(summary_payload, table_names)
    summary_payload["analysis_table_counts"] = {name: int(len(tables[name])) for name in table_names}
    summary_payload.setdefault("analysis_summary", effect_analysis.summary)
    summary_payload.setdefault("window_names", effect_analysis.summary.get("window_names", []))
    summary_payload.setdefault("window_definitions", effect_analysis.summary.get("window_definitions", []))
    summary_payload.setdefault(
        "analysis_overview",
        build_analysis_overview(effect_analysis, model_review_briefing=model_review_briefing),
    )

    return AnalysisReport(
        artifact_id=artifact_id,
        summary=_json_safe(summary_payload),
        tables={name: tables[name] for name in table_names},
        metadata=_json_safe(dict(metadata or {})),
    )


def _default_filename(table_name: str, table_filenames: Mapping[str, str] | None) -> str:
    if table_filenames and table_name in table_filenames:
        return table_filenames[table_name]
    if table_name in DEFAULT_ANALYSIS_TABLE_FILENAMES:
        return DEFAULT_ANALYSIS_TABLE_FILENAMES[table_name]
    return f"{table_name}.csv"


def write_analysis_report(
    report: AnalysisReport,
    output_dir: str | Path,
    *,
    dataset: DatasetSnapshot | None = None,
    summary_filename: str = "analysis_summary.json",
    manifest_filename: str = "analysis_manifest.json",
    table_filenames: Mapping[str, str] | None = None,
    manifest_metadata: Mapping[str, Any] | None = None,
    extra_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    table_counts = report.table_counts()
    summary_payload = dict(report.summary)
    summary_payload.setdefault("schema_version", report.schema_version)
    summary_payload.setdefault("artifact_id", report.artifact_id)
    summary_payload.setdefault("artifact_type", report.artifact_type)
    summary_payload.setdefault("analysis_layers", list(report.tables.keys()))
    summary_payload["analysis_table_counts"] = table_counts

    paths: dict[str, str] = {"summary": summary_filename}
    (output / summary_filename).write_text(
        json.dumps(_json_safe(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for table_name, table in report.tables.items():
        filename = _default_filename(table_name, table_filenames)
        table.to_csv(output / filename, index=False)
        paths[table_name] = filename
    paths.update({str(name): str(path) for name, path in dict(extra_paths or {}).items()})

    manifest_path: str | None = None
    if dataset is not None:
        metadata = {
            "analysis_report_schema_version": report.schema_version,
            "table_counts": table_counts,
            **report.metadata,
            **dict(manifest_metadata or {}),
        }
        manifest = ArtifactManifest(
            artifact_id=report.artifact_id,
            artifact_type=report.artifact_type,
            producer="src.analysis.exports",
            schema_version="artifact.v1",
            dataset=dataset,
            paths=paths,
            metadata=_json_safe(metadata),
        )
        payload = manifest.to_dict()
        payload["manifest_hash"] = manifest.manifest_hash()
        (output / manifest_filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path = manifest_filename

    result_paths = dict(paths)
    if manifest_path is not None:
        result_paths["manifest"] = manifest_path
    return {
        "schema_version": ANALYSIS_REPORT_EXPORT_SCHEMA_VERSION,
        "artifact_id": report.artifact_id,
        "artifact_type": report.artifact_type,
        "paths": result_paths,
        "table_counts": table_counts,
        "manifest": manifest_path,
    }

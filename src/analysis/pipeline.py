from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pandas as pd

from .exports import AnalysisReport, build_regime_window_analysis_report
from .market_report import build_market_regime_analysis_report
from .qlib_ext import analyze_model_effects_by_regime_and_window
from .qlib_recorder import build_qlib_recorder_regime_window_analysis_report, load_qlib_recorder_analysis
from .regimes import MarketRegimeConfig
from .reports import compare_models, compute_regime_metrics, compute_window_metrics


ANALYSIS_SPEC_SCHEMA_VERSION = "analysis.spec.v1"


@dataclass(frozen=True)
class AnalysisSpec:
    artifact_id: str
    baseline_model: str | None = None
    windows: Mapping[str, tuple[str, str]] | None = None
    market_regime_config: MarketRegimeConfig | None = None
    freq: str = "day"
    source_type: str = "daily_backtest"
    summary_note: str = ""
    schema_version: str = ANALYSIS_SPEC_SCHEMA_VERSION

    def normalized_windows(self) -> dict[str, tuple[str, str]]:
        return {str(name): (str(start), str(end)) for name, (start, end) in dict(self.windows or {}).items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "baseline_model": self.baseline_model,
            "windows": {name: [start, end] for name, (start, end) in self.normalized_windows().items()},
            "market_regime_config": asdict(self.market_regime_config) if self.market_regime_config is not None else None,
            "freq": self.freq,
            "source_type": self.source_type,
            "summary_note": self.summary_note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisSpec":
        raw_windows = dict(payload.get("windows") or {})
        windows = {str(name): (str(value[0]), str(value[1])) for name, value in raw_windows.items()}
        raw_config = payload.get("market_regime_config")
        return cls(
            artifact_id=str(payload["artifact_id"]),
            baseline_model=str(payload["baseline_model"]) if payload.get("baseline_model") is not None else None,
            windows=windows,
            market_regime_config=MarketRegimeConfig(**dict(raw_config)) if raw_config is not None else None,
            freq=str(payload.get("freq", "day")),
            source_type=str(payload.get("source_type", "daily_backtest")),
            summary_note=str(payload.get("summary_note", "")),
            schema_version=str(payload.get("schema_version", ANALYSIS_SPEC_SCHEMA_VERSION)),
        )

    def summary_payload(self) -> dict[str, Any]:
        return {
            "analysis_pipeline_schema_version": self.schema_version,
            "baseline_model": self.baseline_model,
            "source_type": self.source_type,
            "summary_note": self.summary_note,
            "analysis_spec": self.to_dict(),
        }


def _daily_backtest_report(
    *,
    spec: AnalysisSpec,
    daily_backtest: pd.DataFrame,
    summary: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> AnalysisReport:
    windows = spec.normalized_windows()
    effect = analyze_model_effects_by_regime_and_window(
        daily_backtest,
        windows=windows,
        baseline_model=spec.baseline_model,
        freq=spec.freq,
    )
    return build_regime_window_analysis_report(
        artifact_id=spec.artifact_id,
        effect_analysis=effect,
        model_comparison=compare_models(daily_backtest),
        regime_metrics=compute_regime_metrics(daily_backtest),
        window_metrics=compute_window_metrics(daily_backtest, windows=windows),
        summary={
            "status": "analysis_pipeline_completed",
            **spec.summary_payload(),
            **dict(summary or {}),
        },
        metadata={
            "source_type": spec.source_type,
            "analysis_pipeline_schema_version": spec.schema_version,
            **dict(metadata or {}),
        },
    )


def _retag_pipeline_report(
    report: AnalysisReport,
    *,
    spec: AnalysisSpec,
    summary: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> AnalysisReport:
    report_summary = dict(report.summary)
    source_status = str(report_summary.get("status", ""))
    report_summary.update(
        {
            "status": "analysis_pipeline_completed",
            "source_report_status": source_status,
            **spec.summary_payload(),
            **dict(summary or {}),
        }
    )
    report_metadata = {
        **dict(report.metadata),
        "source_type": spec.source_type,
        "analysis_pipeline_schema_version": spec.schema_version,
        **dict(metadata or {}),
    }
    return AnalysisReport(
        artifact_id=report.artifact_id,
        summary=report_summary,
        tables=report.tables,
        artifact_type=report.artifact_type,
        schema_version=report.schema_version,
        metadata=report_metadata,
    )


def _load_recorder_map(sources: Mapping[str, object]) -> dict[str, object]:
    return {str(model): load_qlib_recorder_analysis(source, model_name=str(model)) for model, source in sources.items()}


def build_model_regime_comparison_report(
    *,
    spec: AnalysisSpec,
    daily_backtest: pd.DataFrame | None = None,
    market_frame: pd.DataFrame | None = None,
    qlib_recorders: Mapping[str, object] | None = None,
    regime_frame: pd.DataFrame | None = None,
    market_regime_config: MarketRegimeConfig | None = None,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> AnalysisReport:
    if spec.source_type == "daily_backtest":
        if daily_backtest is None:
            raise ValueError("daily_backtest is required when source_type='daily_backtest'")
        return _daily_backtest_report(spec=spec, daily_backtest=daily_backtest, summary=summary, metadata=metadata)
    if spec.source_type == "market_panel":
        if daily_backtest is None:
            raise ValueError("daily_backtest is required when source_type='market_panel'")
        if market_frame is None:
            raise ValueError("market_frame is required when source_type='market_panel'")
        report = build_market_regime_analysis_report(
            artifact_id=spec.artifact_id,
            daily_backtest=daily_backtest,
            market_frame=market_frame,
            windows=spec.normalized_windows(),
            baseline_model=spec.baseline_model,
            config=market_regime_config or spec.market_regime_config,
            summary=spec.summary_payload(),
            metadata={
                "source_type": spec.source_type,
                "analysis_pipeline_schema_version": spec.schema_version,
            },
            freq=spec.freq,
        )
        return _retag_pipeline_report(report, spec=spec, summary=summary, metadata=metadata)
    if spec.source_type == "qlib_recorder":
        if not qlib_recorders:
            raise ValueError("qlib_recorders is required when source_type='qlib_recorder'")
        report = build_qlib_recorder_regime_window_analysis_report(
            artifact_id=spec.artifact_id,
            recorder_analyses=_load_recorder_map(qlib_recorders),
            regime_frame=regime_frame,
            windows=spec.normalized_windows(),
            baseline_model=spec.baseline_model,
            summary=spec.summary_payload(),
            metadata={
                "source_type": spec.source_type,
                "analysis_pipeline_schema_version": spec.schema_version,
            },
            freq=spec.freq,
        )
        return _retag_pipeline_report(report, spec=spec, summary=summary, metadata=metadata)
    raise ValueError(f"unsupported analysis source_type: {spec.source_type}")

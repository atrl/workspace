from .attribution import ModelAttributionReport, explain_model_artifact
from .exports import AnalysisReport, build_regime_window_analysis_report, write_analysis_report
from .failure_modes import build_failure_modes, render_failure_modes_markdown, write_failure_modes_markdown
from .manifest_backfill import backfill_analysis_manifest, backfill_legacy_analysis_cards, scan_analysis_cards
from .qlib_recorder import QlibRecorderAnalysis, build_qlib_recorder_analysis_report, load_qlib_recorder_analysis
from .reports import compare_models, compute_regime_metrics, compute_window_metrics, compute_window_regime_metrics
from .qlib_ext import (
    RegimeWindowEffectAnalysis,
    analyze_model_effects_by_regime_and_window,
    build_qlib_return_frame,
    compare_segment_model_effects,
    compute_qlib_risk_metrics,
)

__all__ = [
    "AnalysisReport",
    "QlibRecorderAnalysis",
    "RegimeWindowEffectAnalysis",
    "ModelAttributionReport",
    "analyze_model_effects_by_regime_and_window",
    "backfill_analysis_manifest",
    "backfill_legacy_analysis_cards",
    "build_failure_modes",
    "build_qlib_return_frame",
    "build_qlib_recorder_analysis_report",
    "build_regime_window_analysis_report",
    "compare_models",
    "compare_segment_model_effects",
    "compute_qlib_risk_metrics",
    "compute_regime_metrics",
    "compute_window_metrics",
    "compute_window_regime_metrics",
    "explain_model_artifact",
    "load_qlib_recorder_analysis",
    "render_failure_modes_markdown",
    "write_analysis_report",
    "write_failure_modes_markdown",
    "scan_analysis_cards",
]

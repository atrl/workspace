from .attribution import (
    ModelAttributionCard,
    ModelAttributionReport,
    build_model_attribution_card,
    explain_model_artifact,
    write_model_attribution_card,
)
from .exports import AnalysisReport, build_regime_window_analysis_report, write_analysis_report
from .failure_modes import build_failure_modes, render_failure_modes_markdown, write_failure_modes_markdown
from .manifest_backfill import backfill_analysis_manifest, backfill_legacy_analysis_cards, scan_analysis_cards
from .market_report import build_market_regime_analysis_report
from .pipeline import ANALYSIS_SPEC_SCHEMA_VERSION, AnalysisSpec, build_model_regime_comparison_report
from .qlib_recorder import (
    QlibRecorderAnalysis,
    build_qlib_recorder_analysis_report,
    build_qlib_recorder_regime_window_analysis_report,
    load_qlib_recorder_analysis,
)
from .reports import compare_models, compute_regime_metrics, compute_window_metrics, compute_window_regime_metrics
from .windows import build_analysis_window_table, build_analysis_windows
from .qlib_ext import (
    RegimeWindowEffectAnalysis,
    analyze_model_effects_by_regime_and_window,
    build_analysis_overview,
    build_model_difference_explanations,
    build_model_review_briefing,
    build_model_regime_window_diagnostics,
    build_qlib_return_frame,
    build_segment_effect_matrix,
    compare_segment_model_effects,
    compute_qlib_risk_metrics,
    render_model_review_briefing_markdown,
)
from .regimes import MarketRegimeConfig, attach_market_regimes, build_market_regime_frame

__all__ = [
    "AnalysisReport",
    "AnalysisSpec",
    "ANALYSIS_SPEC_SCHEMA_VERSION",
    "QlibRecorderAnalysis",
    "RegimeWindowEffectAnalysis",
    "ModelAttributionReport",
    "ModelAttributionCard",
    "MarketRegimeConfig",
    "analyze_model_effects_by_regime_and_window",
    "attach_market_regimes",
    "backfill_analysis_manifest",
    "backfill_legacy_analysis_cards",
    "build_failure_modes",
    "build_analysis_overview",
    "build_analysis_window_table",
    "build_analysis_windows",
    "build_model_review_briefing",
    "build_model_regime_window_diagnostics",
    "build_model_attribution_card",
    "build_model_difference_explanations",
    "build_market_regime_analysis_report",
    "build_model_regime_comparison_report",
    "build_market_regime_frame",
    "build_qlib_return_frame",
    "build_qlib_recorder_analysis_report",
    "build_qlib_recorder_regime_window_analysis_report",
    "build_regime_window_analysis_report",
    "build_segment_effect_matrix",
    "compare_models",
    "compare_segment_model_effects",
    "compute_qlib_risk_metrics",
    "compute_regime_metrics",
    "compute_window_metrics",
    "compute_window_regime_metrics",
    "explain_model_artifact",
    "load_qlib_recorder_analysis",
    "render_failure_modes_markdown",
    "render_model_review_briefing_markdown",
    "write_analysis_report",
    "write_model_attribution_card",
    "write_failure_modes_markdown",
    "scan_analysis_cards",
]

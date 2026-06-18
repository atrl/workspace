from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .exports import AnalysisReport, build_regime_window_analysis_report
from .qlib_ext import analyze_model_effects_by_regime_and_window
from .regimes import MARKET_REGIME_SCHEMA_VERSION, MarketRegimeConfig, attach_market_regimes, build_market_regime_frame
from .reports import compare_models, compute_regime_metrics, compute_window_metrics


def _market_regime_counts(regime_frame: pd.DataFrame) -> pd.DataFrame:
    if regime_frame.empty:
        return pd.DataFrame(columns=["regime", "row_count", "row_share"])
    counts = regime_frame["regime"].astype(str).value_counts().rename_axis("regime").reset_index(name="row_count")
    total = float(counts["row_count"].sum())
    counts["row_share"] = counts["row_count"].astype(float) / total if total > 0 else 0.0
    return counts.sort_values(["row_count", "regime"], ascending=[False, True]).reset_index(drop=True)


def build_market_regime_analysis_report(
    *,
    artifact_id: str,
    daily_backtest: pd.DataFrame,
    market_frame: pd.DataFrame,
    windows: Mapping[str, tuple[str, str]] | None = None,
    baseline_model: str | None = None,
    config: MarketRegimeConfig | None = None,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    freq: str = "day",
) -> AnalysisReport:
    """Rebuild a model analysis report using regimes derived from a market panel."""

    market_regimes = build_market_regime_frame(market_frame, config=config)
    reclassified = attach_market_regimes(daily_backtest, market_regimes, overwrite=True)
    counts = _market_regime_counts(market_regimes)
    effect = analyze_model_effects_by_regime_and_window(
        reclassified,
        windows=windows,
        baseline_model=baseline_model,
        freq=freq,
    )

    summary_payload = {
        "status": "market_regime_analysis_completed",
        "market_regime_schema_version": MARKET_REGIME_SCHEMA_VERSION,
        "market_regime_count": int(market_regimes["regime"].nunique()),
        "market_regime_rows": int(len(market_regimes)),
        "source_daily_row_count": int(len(daily_backtest)),
        "source_daily_model_count": int(daily_backtest["model"].nunique()) if "model" in daily_backtest.columns else 0,
        "market_regime_counts": counts.to_dict(orient="records"),
        **dict(summary or {}),
    }
    report = build_regime_window_analysis_report(
        artifact_id=artifact_id,
        effect_analysis=effect,
        model_comparison=compare_models(reclassified),
        regime_metrics=compute_regime_metrics(reclassified),
        window_metrics=compute_window_metrics(reclassified, windows=windows or {}),
        summary=summary_payload,
        metadata={
            "source_type": "market_regime_analysis",
            "market_regime_schema_version": MARKET_REGIME_SCHEMA_VERSION,
            **dict(metadata or {}),
        },
    )

    tables = dict(report.tables)
    tables["market_regimes"] = market_regimes
    tables["market_regime_counts"] = counts
    tables["market_regime_daily_backtest"] = reclassified

    ordered = {
        "market_regimes": tables["market_regimes"],
        "market_regime_counts": tables["market_regime_counts"],
        "market_regime_daily_backtest": tables["market_regime_daily_backtest"],
    }
    for name, table in tables.items():
        if name not in ordered:
            ordered[name] = table

    report_summary = dict(report.summary)
    layers = list(report_summary.get("analysis_layers", []))
    for name in ordered:
        if name not in layers:
            layers.append(name)
    report_summary["analysis_layers"] = layers
    report_summary["analysis_table_counts"] = {name: int(len(table)) for name, table in ordered.items()}

    return AnalysisReport(
        artifact_id=report.artifact_id,
        summary=report_summary,
        tables=ordered,
        artifact_type=report.artifact_type,
        schema_version=report.schema_version,
        metadata=report.metadata,
    )

from __future__ import annotations

import json

import pandas as pd

from src.analysis import (
    analyze_model_effects_by_regime_and_window,
    build_regime_window_analysis_report,
    compare_models,
    compute_regime_metrics,
    compute_window_metrics,
    write_analysis_report,
)
from src.core import DatasetSnapshot


def _daily_frame() -> pd.DataFrame:
    dates = list(pd.date_range("2026-01-01", periods=8, freq="D")) * 2
    regimes = ["bull", "bull", "bull", "neutral", "stress", "stress", "stress", "downtrend"] * 2
    return pd.DataFrame(
        {
            "date": dates,
            "model": ["ridge_mse"] * 8 + ["gcn_mse"] * 8,
            "regime": regimes,
            "nav": [
                1.00,
                1.02,
                1.04,
                1.03,
                1.01,
                0.99,
                1.00,
                1.01,
                1.00,
                1.04,
                1.08,
                1.07,
                1.02,
                0.96,
                0.97,
                0.98,
            ],
            "benchmark_nav": [1.0, 1.01, 1.02, 1.02, 1.00, 0.99, 1.00, 1.00] * 2,
            "position": [0.2, 0.3, 0.5, 0.1, 0.4, 0.2, 0.1, 0.2] * 2,
            "turnover": [0.0, 0.1, 0.2, 0.4, 0.3, 0.1, 0.1, 0.2] * 2,
        }
    )


def test_analysis_report_writes_standard_sidecars_summary_and_manifest(tmp_path):
    windows = {"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")}
    daily = _daily_frame()
    effect = analyze_model_effects_by_regime_and_window(daily, windows=windows, baseline_model="ridge_mse")
    report = build_regime_window_analysis_report(
        artifact_id="analysis_report_export_v1",
        effect_analysis=effect,
        model_comparison=compare_models(daily),
        regime_metrics=compute_regime_metrics(daily),
        window_metrics=compute_window_metrics(daily, windows=windows),
        summary={
            "status": "analysis_report_completed",
            "note": "中文 review 入口：统一分析报告写出 qlib、regime、window 和模型差异表。",
        },
    )
    dataset = DatasetSnapshot(
        name="unit_daily",
        universe=("SOXL", "SOXX"),
        start="2026-01-01",
        end="2026-01-08",
        calendar_path="data/qlib/soxl/calendars/day.txt",
        data_paths=("unit.csv",),
    )

    result = write_analysis_report(report, tmp_path, dataset=dataset)

    assert result["schema_version"] == "analysis.report.export.v1"
    assert result["paths"]["summary"] == "analysis_summary.json"
    assert result["paths"]["qlib_risk_metrics"] == "qlib_risk_metrics.csv"
    assert result["paths"]["model_effect_deltas"] == "model_effect_deltas.csv"
    assert result["paths"]["segment_effect_matrix"] == "segment_effect_matrix.csv"
    assert result["paths"]["model_regime_window_diagnostics"] == "model_regime_window_diagnostics.csv"
    assert result["paths"]["model_review_briefing"] == "model_review_briefing.csv"
    assert result["table_counts"]["window_regime_metrics"] >= 2
    assert result["table_counts"]["model_regime_window_diagnostics"] == 2
    assert result["table_counts"]["model_review_briefing"] >= 1

    expected_files = {
        "analysis_summary.json",
        "analysis_manifest.json",
        "model_comparison.csv",
        "regime_metrics.csv",
        "window_metrics.csv",
        "window_regime_metrics.csv",
        "qlib_return_frame.csv",
        "qlib_risk_metrics.csv",
        "segment_metrics.csv",
        "segment_effect_matrix.csv",
        "model_effect_deltas.csv",
        "model_regime_window_diagnostics.csv",
        "model_review_briefing.csv",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})

    summary = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "analysis.report.v1"
    assert summary["artifact_id"] == "analysis_report_export_v1"
    assert summary["status"] == "analysis_report_completed"
    assert "qlib_risk_metrics" in summary["analysis_layers"]
    assert summary["analysis_summary"]["baseline_model"] == "ridge_mse"
    assert summary["analysis_table_counts"]["model_effect_deltas"] == len(effect.model_effect_deltas)
    assert summary["analysis_table_counts"]["model_regime_window_diagnostics"] == len(
        effect.model_regime_window_diagnostics
    )
    assert "model_review_briefing" in summary["analysis_layers"]
    assert "model_regime_window_diagnostics" in summary["analysis_layers"]

    briefing = pd.read_csv(tmp_path / "model_review_briefing.csv")
    assert {"review_signal", "decision_hint", "review_note"}.issubset(briefing.columns)
    assert "weakness" in set(briefing["review_signal"])
    assert "强于基准" in "\n".join(briefing["review_note"].astype(str).tolist())

    diagnostics = pd.read_csv(tmp_path / "model_regime_window_diagnostics.csv")
    assert diagnostics.loc[diagnostics["model"] == "gcn_mse", "diagnostic_signal"].iloc[0] == "stress_fragile"
    assert "压力" in diagnostics.loc[diagnostics["model"] == "gcn_mse", "review_note"].iloc[0]

    manifest = json.loads((tmp_path / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "analysis_report"
    assert manifest["producer"] == "src.analysis.exports"
    assert manifest["paths"]["summary"] == "analysis_summary.json"
    assert manifest["paths"]["model_regime_window_diagnostics"] == "model_regime_window_diagnostics.csv"
    assert manifest["paths"]["model_review_briefing"] == "model_review_briefing.csv"
    assert manifest["metadata"]["analysis_report_schema_version"] == "analysis.report.v1"
    assert len(manifest["manifest_hash"]) == 64

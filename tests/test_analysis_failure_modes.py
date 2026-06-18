from __future__ import annotations

import json

import pandas as pd

from src.analysis import (
    analyze_model_effects_by_regime_and_window,
    build_failure_modes,
    build_regime_window_analysis_report,
    compare_models,
    compute_regime_metrics,
    compute_window_metrics,
    render_failure_modes_markdown,
    write_analysis_report,
    write_failure_modes_markdown,
)
from src.core import DatasetSnapshot
from src.dashboard import load_model_matrix_dashboard


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


def test_build_failure_modes_prioritizes_regime_window_underperformance_with_chinese_notes(tmp_path):
    analysis = analyze_model_effects_by_regime_and_window(
        _daily_frame(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
        baseline_model="ridge_mse",
    )

    failure_modes = build_failure_modes(analysis.model_effect_deltas, top_n=5)

    assert {
        "schema_version",
        "model",
        "baseline_model",
        "segment_type",
        "segment",
        "failure_mode",
        "severity_score",
        "review_note",
    }.issubset(failure_modes.columns)
    assert not failure_modes.empty
    assert failure_modes["schema_version"].nunique() == 1
    assert failure_modes.iloc[0]["failure_mode"] in {"return_stability_lag", "drawdown_expansion"}
    assert "stress" in failure_modes.iloc[0]["segment"]
    assert "基准" in failure_modes.iloc[0]["review_note"]
    assert list(failure_modes["severity_score"]) == sorted(failure_modes["severity_score"], reverse=True)

    markdown = render_failure_modes_markdown(failure_modes, title="SOXL 模型失败模式")
    assert "## 中文结论" in markdown
    assert "stress" in markdown
    assert "gcn_mse" in markdown

    path = write_failure_modes_markdown(failure_modes, tmp_path / "failure_modes.md", title="SOXL 模型失败模式")
    assert path.name == "failure_modes.md"
    assert "SOXL 模型失败模式" in path.read_text(encoding="utf-8")


def test_analysis_report_exports_failure_modes_table_and_manifest_path(tmp_path):
    windows = {"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")}
    daily = _daily_frame()
    effect = analyze_model_effects_by_regime_and_window(daily, windows=windows, baseline_model="ridge_mse")
    report = build_regime_window_analysis_report(
        artifact_id="analysis_failure_modes_v1",
        effect_analysis=effect,
        model_comparison=compare_models(daily),
        regime_metrics=compute_regime_metrics(daily),
        window_metrics=compute_window_metrics(daily, windows=windows),
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

    assert result["paths"]["failure_modes"] == "failure_modes.csv"
    assert result["paths"]["segment_effect_matrix"] == "segment_effect_matrix.csv"
    assert result["paths"]["model_review_briefing"] == "model_review_briefing.csv"
    assert result["table_counts"]["failure_modes"] >= 1
    assert result["table_counts"]["segment_effect_matrix"] >= 1
    assert result["table_counts"]["model_review_briefing"] >= 1
    failure_modes = pd.read_csv(tmp_path / "failure_modes.csv")
    segment_effect_matrix = pd.read_csv(tmp_path / "segment_effect_matrix.csv")
    model_review_briefing = pd.read_csv(tmp_path / "model_review_briefing.csv")
    assert "review_note" in failure_modes.columns
    assert "stress" in set(failure_modes["segment"])
    assert "best_relative_return_model" in segment_effect_matrix.columns
    assert "review_signal" in model_review_briefing.columns

    summary = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    assert "failure_modes" in summary["analysis_layers"]
    assert "segment_effect_matrix" in summary["analysis_layers"]
    assert "model_review_briefing" in summary["analysis_layers"]
    assert summary["analysis_table_counts"]["failure_modes"] == len(failure_modes)
    assert summary["analysis_table_counts"]["segment_effect_matrix"] == len(segment_effect_matrix)
    assert summary["analysis_table_counts"]["model_review_briefing"] == len(model_review_briefing)

    manifest = json.loads((tmp_path / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["paths"]["failure_modes"] == "failure_modes.csv"
    assert manifest["paths"]["segment_effect_matrix"] == "segment_effect_matrix.csv"
    assert manifest["paths"]["model_review_briefing"] == "model_review_briefing.csv"
    assert manifest["metadata"]["table_counts"]["failure_modes"] == len(failure_modes)
    assert manifest["metadata"]["table_counts"]["segment_effect_matrix"] == len(segment_effect_matrix)
    assert manifest["metadata"]["table_counts"]["model_review_briefing"] == len(model_review_briefing)

    dashboard = load_model_matrix_dashboard(tmp_path)
    assert dashboard["manifest_source"]["kind"] == "analysis_manifest"
    assert "failure_modes" in dashboard["artifacts"]
    assert dashboard["failure_modes"][0]["failure_mode"] == "return_stability_lag"
    assert "stress" in dashboard["failure_modes"][0]["segment"]

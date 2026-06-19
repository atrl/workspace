from __future__ import annotations

import pandas as pd

from src.analysis import (
    analyze_model_effects_by_regime_and_window,
    build_analysis_overview,
    build_model_difference_explanations,
    build_model_regime_window_diagnostics,
    build_model_review_briefing,
    build_qlib_return_frame,
    build_regime_window_analysis_report,
    build_segment_effect_matrix,
    compute_qlib_risk_metrics,
    compute_window_regime_metrics,
    render_model_review_briefing_markdown,
)


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
            "vix": [17, 18, 19, 24, 35, 32, 31, 22] * 2,
            "trend": [0.02, 0.03, 0.04, 0.00, -0.05, -0.06, -0.04, -0.02] * 2,
            "drawdown": [0.0, 0.0, 0.0, -0.01, -0.12, -0.14, -0.10, -0.04] * 2,
        }
    )


def test_build_qlib_return_frame_preserves_model_date_regime_and_excess_returns():
    returns = build_qlib_return_frame(_daily_frame())

    assert {"date", "model", "regime", "strategy_return", "benchmark_return", "excess_return"}.issubset(
        returns.columns
    )
    first_gcn = returns.loc[returns["model"] == "gcn_mse"].iloc[0]
    assert first_gcn["strategy_return"] == 0.0

    second_gcn = returns.loc[returns["model"] == "gcn_mse"].iloc[1]
    assert abs(float(second_gcn["strategy_return"]) - 0.04) < 1e-12
    assert abs(float(second_gcn["excess_return"]) - 0.03) < 1e-12


def test_compute_qlib_risk_metrics_segments_all_regime_and_window_returns():
    risk = compute_qlib_risk_metrics(
        _daily_frame(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
    )

    assert {"model", "segment_type", "segment", "return_kind", "annualized_return", "information_ratio"}.issubset(
        risk.columns
    )
    assert {"all", "regime", "window", "window_regime"}.issubset(set(risk["segment_type"]))
    assert {"strategy_return", "excess_return"}.issubset(set(risk["return_kind"]))

    gcn_stress = risk.loc[
        (risk["model"] == "gcn_mse")
        & (risk["segment_type"] == "regime")
        & (risk["segment"] == "stress")
        & (risk["return_kind"] == "strategy_return")
    ].iloc[0]
    assert float(gcn_stress["annualized_return"]) < 0.0

    gcn_stress_window = risk.loc[
        (risk["model"] == "gcn_mse")
        & (risk["segment_type"] == "window_regime")
        & (risk["segment"] == "stress_window::stress")
        & (risk["return_kind"] == "strategy_return")
    ].iloc[0]
    assert gcn_stress_window["window"] == "stress_window"
    assert gcn_stress_window["regime"] == "stress"
    assert float(gcn_stress_window["annualized_return"]) < 0.0


def test_compute_window_regime_metrics_compares_model_effects_inside_time_and_regime_segments():
    metrics = compute_window_regime_metrics(
        _daily_frame(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
    )

    assert {"window", "regime", "segment", "model", "return_pct", "stability_score"}.issubset(metrics.columns)
    assert "early::bull" in set(metrics["segment"])
    assert "stress_window::stress" in set(metrics["segment"])

    gcn_stress = metrics.loc[
        (metrics["model"] == "gcn_mse") & (metrics["segment"] == "stress_window::stress")
    ].iloc[0]
    ridge_stress = metrics.loc[
        (metrics["model"] == "ridge_mse") & (metrics["segment"] == "stress_window::stress")
    ].iloc[0]
    assert float(gcn_stress["return_pct"]) < float(ridge_stress["return_pct"])


def test_analyze_model_effects_by_regime_and_window_explains_segment_deltas():
    analysis = analyze_model_effects_by_regime_and_window(
        _daily_frame(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
        baseline_model="ridge_mse",
    )

    assert analysis.summary["baseline_model"] == "ridge_mse"
    assert analysis.summary["window_regime_count"] >= 2
    assert not analysis.qlib_risk_metrics.empty
    assert not analysis.window_regime_metrics.empty

    deltas = analysis.model_effect_deltas
    assert "window_regime" in set(deltas["segment_type"])
    gcn_bull = deltas.loc[
        (deltas["model"] == "gcn_mse") & (deltas["segment_type"] == "regime") & (deltas["segment"] == "bull")
    ].iloc[0]
    assert gcn_bull["regime"] == "bull"
    gcn_stress = deltas.loc[
        (deltas["model"] == "gcn_mse") & (deltas["segment_type"] == "regime") & (deltas["segment"] == "stress")
    ].iloc[0]
    gcn_stress_window = deltas.loc[
        (deltas["model"] == "gcn_mse")
        & (deltas["segment_type"] == "window_regime")
        & (deltas["segment"] == "stress_window::stress")
    ].iloc[0]

    assert float(gcn_bull["relative_return_pct_delta"]) > 0.0
    assert float(gcn_stress["relative_return_pct_delta"]) < 0.0
    assert gcn_stress_window["window"] == "stress_window"
    assert gcn_stress_window["regime"] == "stress"
    assert "弱于基准" in gcn_stress["diagnosis"]
    assert "stress" in analysis.summary["weakest_segments"][0]["segment"]


def test_build_segment_effect_matrix_summarizes_winners_losers_and_baseline_gaps():
    matrix = build_segment_effect_matrix(
        _daily_frame(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
        baseline_model="ridge_mse",
    )

    assert {
        "schema_version",
        "segment_type",
        "segment",
        "baseline_model",
        "best_relative_return_model",
        "worst_relative_return_model",
        "relative_return_pct_spread",
        "baseline_relative_return_pct",
        "best_relative_return_vs_baseline_delta",
        "best_stability_model",
        "worst_stability_model",
        "stability_score_spread",
        "highest_drawdown_model",
        "review_note",
    }.issubset(matrix.columns)
    assert matrix["schema_version"].nunique() == 1
    assert "window_regime" in set(matrix["segment_type"])

    stress_window = matrix.loc[
        (matrix["segment_type"] == "window_regime") & (matrix["segment"] == "stress_window::stress")
    ].iloc[0]
    assert stress_window["best_relative_return_model"] == "ridge_mse"
    assert stress_window["worst_relative_return_model"] == "gcn_mse"
    assert float(stress_window["relative_return_pct_spread"]) > 0.0
    assert float(stress_window["best_relative_return_vs_baseline_delta"]) == 0.0
    assert "stress_window::stress" in stress_window["review_note"]

    bull = matrix.loc[(matrix["segment_type"] == "regime") & (matrix["segment"] == "bull")].iloc[0]
    assert bull["best_relative_return_model"] == "gcn_mse"
    assert float(bull["best_relative_return_vs_baseline_delta"]) > 0.0
    assert "强于基准" in bull["review_note"]


def test_build_model_difference_explanations_summarizes_segment_winners_losers_in_chinese():
    analysis = analyze_model_effects_by_regime_and_window(
        _daily_frame(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
        baseline_model="ridge_mse",
    )

    explanations = build_model_difference_explanations(analysis, max_rows=20)

    assert {
        "schema_version",
        "priority_rank",
        "segment_type",
        "segment",
        "baseline_model",
        "winner_model",
        "loser_model",
        "relative_return_pct_spread",
        "winner_vs_baseline_delta",
        "explanation_signal",
        "decision_hint",
        "review_note",
    }.issubset(explanations.columns)
    assert explanations["schema_version"].nunique() == 1
    assert explanations["schema_version"].iloc[0] == "analysis.model_difference_explanation.v1"
    assert list(explanations["priority_rank"]) == list(range(1, len(explanations) + 1))

    stress = explanations.loc[explanations["segment"] == "stress_window::stress"].iloc[0]
    assert stress["winner_model"] == "ridge_mse"
    assert stress["loser_model"] == "gcn_mse"
    assert stress["explanation_signal"] == "baseline_defends"
    assert stress["decision_hint"] == "block_or_rework_loser"
    assert "中文 review" in stress["review_note"]
    assert "gcn_mse" in stress["review_note"]
    assert float(stress["relative_return_pct_spread"]) > 0.0

    bull = explanations.loc[(explanations["segment_type"] == "regime") & (explanations["segment"] == "bull")].iloc[0]
    assert bull["winner_model"] == "gcn_mse"
    assert bull["explanation_signal"] == "segment_winner"
    assert bull["decision_hint"] == "promote_segment_candidate"
    assert "强于基准" in bull["review_note"]


def test_regime_window_analysis_report_exports_model_difference_explanations_table():
    analysis = analyze_model_effects_by_regime_and_window(
        _daily_frame(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
        baseline_model="ridge_mse",
    )

    report = build_regime_window_analysis_report(
        artifact_id="unit_difference_explanations",
        effect_analysis=analysis,
    )

    assert "model_difference_explanations" in report.tables
    assert "model_difference_explanations" in report.summary["analysis_layers"]
    assert report.summary["analysis_table_counts"]["model_difference_explanations"] >= 1
    assert report.tables["model_difference_explanations"]["schema_version"].iloc[0] == (
        "analysis.model_difference_explanation.v1"
    )


def test_build_model_review_briefing_prioritizes_regime_window_model_differences_with_chinese_notes():
    analysis = analyze_model_effects_by_regime_and_window(
        _daily_frame(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
        baseline_model="ridge_mse",
    )

    briefing = build_model_review_briefing(analysis, max_rows=6)

    assert {
        "schema_version",
        "priority_rank",
        "model",
        "baseline_model",
        "segment_type",
        "segment",
        "review_signal",
        "decision_hint",
        "review_note",
    }.issubset(briefing.columns)
    assert not briefing.empty
    assert briefing["schema_version"].nunique() == 1
    assert list(briefing["priority_rank"]) == list(range(1, len(briefing) + 1))
    assert "weakness" in set(briefing["review_signal"])
    assert "strength" in set(briefing["review_signal"])

    stress = briefing.loc[(briefing["model"] == "gcn_mse") & briefing["segment"].str.contains("stress")].iloc[0]
    assert stress["review_signal"] == "weakness"
    assert stress["decision_hint"] == "block_or_rework"
    assert "弱于基准" in stress["review_note"]

    markdown = render_model_review_briefing_markdown(briefing, title="SOXL 模型差异 Review")
    assert "## 中文结论" in markdown
    assert "是否晋升" in markdown
    assert "stress" in markdown
    assert "gcn_mse" in markdown


def test_build_analysis_overview_summarizes_regime_window_model_gaps_for_dashboard_and_schedule():
    analysis = analyze_model_effects_by_regime_and_window(
        _daily_frame(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
        baseline_model="ridge_mse",
    )

    overview = build_analysis_overview(analysis)

    assert overview["schema_version"] == "analysis.overview.v1"
    assert overview["baseline_model"] == "ridge_mse"
    assert overview["model_count"] == 2
    assert overview["regime_count"] == 4
    assert overview["window_count"] == 2
    assert overview["review_signal_counts"]["weakness"] >= 1
    assert overview["review_signal_counts"]["strength"] >= 1
    assert overview["top_weakness"]["model"] == "gcn_mse"
    assert "stress" in overview["top_weakness"]["segment"]
    assert overview["top_weakness"]["decision_hint"] == "block_or_rework"
    assert overview["top_strength"]["model"] == "gcn_mse"
    assert overview["largest_relative_return_spread_segments"][0]["segment"]
    assert overview["largest_relative_return_spread_segments"][0]["best_relative_return_model"]
    assert overview["largest_relative_return_spread_segments"][0]["worst_relative_return_model"]
    assert overview["largest_relative_return_spread_segments"][0]["relative_return_pct_spread"] > 0.0
    assert "regime/window" in overview["dashboard_note"]


def test_build_model_regime_window_diagnostics_profiles_stability_across_segments():
    analysis = analyze_model_effects_by_regime_and_window(
        _daily_frame(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
        baseline_model="ridge_mse",
    )

    diagnostics = build_model_regime_window_diagnostics(analysis)

    assert {
        "schema_version",
        "model",
        "baseline_model",
        "evaluated_segment_count",
        "positive_segment_count",
        "negative_segment_count",
        "weakness_segment_count",
        "strength_segment_count",
        "window_regime_negative_count",
        "stress_negative_count",
        "best_segment",
        "worst_segment",
        "diagnostic_signal",
        "review_note",
    }.issubset(diagnostics.columns)
    assert diagnostics["schema_version"].nunique() == 1
    assert set(diagnostics["model"]) == {"ridge_mse", "gcn_mse"}

    baseline = diagnostics.loc[diagnostics["model"] == "ridge_mse"].iloc[0]
    assert baseline["diagnostic_signal"] == "baseline"
    assert baseline["baseline_model"] == "ridge_mse"

    gcn = diagnostics.loc[diagnostics["model"] == "gcn_mse"].iloc[0]
    assert int(gcn["evaluated_segment_count"]) >= 4
    assert int(gcn["strength_segment_count"]) >= 1
    assert int(gcn["weakness_segment_count"]) >= 1
    assert int(gcn["window_regime_negative_count"]) >= 1
    assert int(gcn["stress_negative_count"]) >= 1
    assert gcn["diagnostic_signal"] == "stress_fragile"
    assert "stress" in gcn["worst_segment"]
    assert "压力" in gcn["review_note"]
    assert analysis.model_regime_window_diagnostics.equals(diagnostics)
    assert analysis.summary["model_regime_window_diagnostic_count"] == len(diagnostics)
    assert analysis.summary["top_regime_fragile_models"][0]["model"] == "gcn_mse"

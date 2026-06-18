from __future__ import annotations

import pandas as pd

from src.analysis import (
    analyze_model_effects_by_regime_and_window,
    build_qlib_return_frame,
    compute_qlib_risk_metrics,
    compute_window_regime_metrics,
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

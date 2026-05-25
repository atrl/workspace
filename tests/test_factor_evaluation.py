from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_signed_te_reversion import (
    compute_time_series_ic,
    run_long_flat_backtest,
)
from scripts.evaluate_signed_te_stress_gate import build_gated_signal
from src.soxl_quant.app.factor_diagnostics import (
    build_factor_gate_summary,
    render_factor_gate_summary,
)


def test_time_series_ic_scores_positive_forward_alignment():
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    feature = pd.Series([1, -1, 1, -1, 1, -1, 1, -1], index=index, dtype=float)
    returns = pd.Series([0.0, 0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02], index=index)

    result = compute_time_series_ic(feature, returns, horizon=1)

    assert result.n_days == 7
    assert result.ic_mean > 0.0
    assert result.ic_positive_ratio > 0.70


def test_long_flat_backtest_uses_signal_after_previous_close():
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    signal = pd.Series([0.0, 2.0, 2.0, -1.0, -1.0, 2.0], index=index)
    returns = pd.Series([0.0, 0.10, 0.10, -0.10, -0.10, 0.10], index=index)

    result = run_long_flat_backtest(signal, returns, threshold=0.0)

    expected_daily = pd.Series([0.0, 0.0, 0.10, -0.10, 0.0, 0.0], index=index)
    expected_equity = (1.0 + expected_daily).cumprod()
    assert np.allclose(result.equity.to_numpy(), expected_equity.to_numpy())
    assert result.num_signals == 3
    assert result.trades == 2


def test_build_gated_signal_keeps_only_release_days_above_thresholds():
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    signed_te = pd.Series([0.2, 0.8, 1.2, 1.4, 0.9, 1.8], index=index)
    vix_divergence = pd.Series([0.0, 0.2, 0.6, 0.1, 0.7, 0.8], index=index)
    regime_risk = pd.Series([0.8, 0.75, 0.70, 0.72, 0.65, 0.60], index=index)

    signal = build_gated_signal(
        signed_te=signed_te,
        vix_divergence=vix_divergence,
        regime_risk=regime_risk,
        te_threshold=1.0,
        vix_divergence_threshold=0.5,
        regime_risk_threshold=0.7,
        regime_risk_drop_5_threshold=0.1,
    )

    expected = pd.Series([0.0, 0.0, 1.2, 0.0, 0.0, 1.8], index=index)
    assert np.allclose(signal.to_numpy(), expected.to_numpy())


def test_factor_gate_summary_separates_pass_redundant_and_weak_features():
    ic_df = pd.DataFrame(
        {
            "feature_name": ["strong", "redundant", "weak"],
            "horizon": [20, 20, 20],
            "n_days": [100, 100, 100],
            "ic_mean": [0.041, 0.052, 0.010],
            "ic_std": [0.10, 0.12, 0.11],
            "ic_t": [4.1, 4.3, 0.9],
            "ic_positive_ratio": [0.62, 0.64, 0.49],
            "pearson_corr": [0.10, 0.12, 0.01],
        }
    )
    backtest_df = pd.DataFrame(
        {
            "feature_name": ["strong", "redundant", "weak"],
            "direction": ["long", "long", "long"],
            "threshold": [0.5, 0.5, 0.5],
            "total_return": [0.20, 0.30, 0.02],
            "sharpe": [0.80, 0.90, 0.10],
            "max_drawdown": [-0.20, -0.25, -0.15],
            "profit_factor": [1.35, 1.40, 1.05],
            "signal_accuracy": [0.58, 0.59, 0.51],
            "num_signals": [40, 45, 20],
            "trades": [8, 9, 4],
        }
    )
    corr_df = pd.DataFrame(
        {
            "feature_name": ["strong", "redundant", "weak"],
            "max_abs_core_correlation": [0.30, 0.91, 0.20],
            "max_abs_core_correlation_feature": ["mom_5", "mom_5", "vol_21"],
        }
    )

    summary = build_factor_gate_summary(ic_df, backtest_df, corr_df).set_index("feature_name")

    assert bool(summary.loc["strong", "full_gate"]) is True
    assert summary.loc["strong", "primary_failure"] == "pass"
    assert bool(summary.loc["redundant", "full_gate"]) is False
    assert summary.loc["redundant", "primary_failure"] == "correlation_redundant"
    assert bool(summary.loc["weak", "full_gate"]) is False
    assert summary.loc["weak", "primary_failure"] == "weak_direction_and_execution"

    review = render_factor_gate_summary(summary.reset_index())
    assert "Factor Gate Summary" in review
    assert "correlation_redundant" in review

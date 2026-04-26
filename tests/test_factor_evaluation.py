from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_signed_te_reversion import (
    compute_time_series_ic,
    run_long_flat_backtest,
)
from scripts.evaluate_signed_te_stress_gate import build_gated_signal


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

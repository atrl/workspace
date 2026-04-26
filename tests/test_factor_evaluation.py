from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_signed_te_reversion import (
    compute_time_series_ic,
    run_long_flat_backtest,
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

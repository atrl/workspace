from __future__ import annotations

import pandas as pd

from src.soxl_quant.backtest import _benchmark_metrics_from_price_series


def test_benchmark_metrics_from_price_series_use_warmup_tradable_anchor():
    dates = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.Series([100.0, 200.0, 200.0, 100.0, 100.0], index=dates)

    metrics = _benchmark_metrics_from_price_series(
        prices,
        effective_test_seg=("2024-01-02", "2024-01-08"),
        strategy_return=-5.0,
        lookback=2,
    )

    assert metrics["bnh_return"] == -50.0
    assert metrics["relative_outperformance"] == 45.0
    assert metrics["benchmark_start_date"] == "2024-01-04"
    assert metrics["benchmark_end_date"] == "2024-01-05"
    assert metrics["benchmark_warmup_rows"] == 2

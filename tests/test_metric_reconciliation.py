from __future__ import annotations

import pandas as pd

from src.soxl_quant.core.metric_reconciliation import compute_relative_metric_variants


def test_relative_metric_variants_detect_warmup_gate_flip() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.Series([100.0, 200.0, 200.0, 100.0, 100.0], index=dates)

    metrics = compute_relative_metric_variants(
        strategy_return=-5.0,
        price_series=prices,
        requested_seg=("2024-01-02", "2024-01-08"),
        effective_seg=("2024-01-02", "2024-01-08"),
        lookback=2,
        target_relative_return=30.0,
    )

    assert metrics["calendar_buy_and_hold_return"] == 0.0
    assert metrics["tradable_buy_and_hold_return"] == -50.0
    assert metrics["calendar_relative_return"] == -5.0
    assert metrics["tradable_relative_return"] == 45.0
    assert metrics["first_tradable_date"] == "2024-01-04"
    assert metrics["warmup_days"] == 2
    assert metrics["gate_flip"] is True
    assert metrics["mismatch_reason"] == "warmup_benchmark_alignment_gate_flip"

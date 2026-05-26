from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.sequence_risk_objective import (
    SequenceRiskPathConfig,
    apply_sequence_risk_mask,
    fold_local_sequence_risk_thresholds,
    score_sequence_risk_path,
)


def test_fold_local_thresholds_use_only_supplied_train_dates():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=6, freq="B"),
            "anchor_slope_180": [-0.30, -0.20, -0.10, 0.50, 0.60, 0.70],
            "anchor_gap_180": [-0.40, -0.30, -0.20, 0.80, 0.90, 1.00],
            "trend_gap_63": [0.10, 0.20, 0.30, 9.00, 10.00, 11.00],
        }
    )

    thresholds = fold_local_sequence_risk_thresholds(
        frame,
        frame["datetime"].iloc[:3],
    )

    assert thresholds["anchor_slope_180"]["threshold"] == pd.Series([-0.30, -0.20, -0.10]).quantile(0.25)
    assert thresholds["anchor_gap_180"]["threshold"] == pd.Series([-0.40, -0.30, -0.20]).quantile(0.25)
    assert thresholds["trend_gap_63"]["threshold"] == pd.Series([0.10, 0.20, 0.30]).quantile(0.75)


def test_sequence_risk_mask_and_path_penalty_capture_drawdown_breaches():
    daily = pd.DataFrame(
        {
            "datetime": pd.date_range("2022-01-03", periods=4, freq="B"),
            "strategy_nav": [100.0, 96.0, 82.0, 86.0],
            "buyhold_nav": [100.0, 90.0, 80.0, 82.0],
            "position": [0.05, 0.18, 0.22, 0.08],
            "anchor_slope_180": [0.02, -0.04, -0.05, 0.03],
            "anchor_gap_180": [0.10, -0.20, -0.30, 0.05],
            "trend_gap_63": [0.02, 0.21, 0.25, 0.03],
        }
    )
    thresholds = {
        "anchor_slope_180": {"direction": "low", "threshold": -0.01, "quantile": 0.25},
        "anchor_gap_180": {"direction": "low", "threshold": -0.12, "quantile": 0.25},
        "trend_gap_63": {"direction": "high", "threshold": 0.18, "quantile": 0.75},
    }

    mask, hit_count = apply_sequence_risk_mask(daily, thresholds, mode="any")
    metrics = score_sequence_risk_path(
        daily,
        thresholds=thresholds,
        mode="any",
        config=SequenceRiskPathConfig(target_max_dd_pct=15.0, sequence_position_reference=0.10),
    )

    assert mask.tolist() == [False, True, True, False]
    assert hit_count.tolist() == [0, 3, 3, 0]
    assert metrics["sequence_rows"] == 2
    assert metrics["drawdown_breach_rows"] == 1
    assert metrics["sequence_breach_capture"] == 1.0
    assert metrics["sequence_path_penalty"] > 0.0

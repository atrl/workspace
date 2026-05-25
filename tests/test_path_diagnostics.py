from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.path_diagnostics import (
    compare_interval_feature_deltas,
    find_worst_drawdown_intervals,
    make_backtest_daily_frame,
)


def test_make_backtest_daily_frame_aligns_positions_and_features():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    result = {
        "portfolio": pd.DataFrame({"account": [100.0, 110.0, 99.0]}, index=index),
        "position_history": [0.20, 0.50, 0.30, 0.10],
    }
    features = pd.DataFrame({"vol_stress": [0.1, 0.8, 0.9]}, index=index)

    daily = make_backtest_daily_frame(result, label="candidate", feature_frame=features)

    assert daily["datetime"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert daily["label"].tolist() == ["candidate", "candidate", "candidate"]
    assert daily["position"].round(2).tolist() == [0.20, 0.50, 0.30]
    assert daily["nav"].round(2).tolist() == [1.0, 1.1, 0.99]
    assert daily["daily_return"].round(4).tolist() == [0.0, 0.1, -0.1]
    assert daily["drawdown"].round(4).tolist() == [0.0, 0.0, 0.1]
    assert daily["vol_stress"].tolist() == [0.1, 0.8, 0.9]


def test_find_worst_drawdown_intervals_returns_peak_to_trough_path():
    daily = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "account": [100.0, 125.0, 100.0, 130.0],
            "position": [0.2, 0.7, 0.7, 0.3],
        }
    )

    intervals = find_worst_drawdown_intervals(daily, top_n=1, min_depth=0.05)

    row = intervals.iloc[0]
    assert row["peak_date"] == "2024-01-02"
    assert row["trough_date"] == "2024-01-03"
    assert row["rows"] == 2
    assert row["max_dd"] == 20.0
    assert row["return"] == -20.0
    assert row["avg_pos"] == 0.7


def test_compare_interval_feature_deltas_ranks_stress_features_and_position_gap():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    candidate = pd.DataFrame(
        {
            "datetime": dates,
            "account": [100.0, 112.0, 84.0, 90.0, 96.0, 98.0],
            "position": [0.3, 0.7, 0.7, 0.4, 0.3, 0.3],
            "vol_stress": [0.1, 5.0, 5.5, 0.3, 0.2, 0.1],
            "trend_support": [1.4, 0.8, 0.7, 1.0, 1.3, 1.2],
        }
    )
    baseline = pd.DataFrame(
        {
            "datetime": dates,
            "account": [100.0, 110.0, 104.5, 108.0, 110.0, 112.0],
            "position": [0.3, 0.35, 0.35, 0.35, 0.3, 0.3],
        }
    )
    intervals = pd.DataFrame(
        {
            "rank": [1],
            "peak_date": ["2024-01-02"],
            "trough_date": ["2024-01-03"],
        }
    )

    deltas = compare_interval_feature_deltas(
        candidate_daily=candidate,
        baseline_daily=baseline,
        intervals=intervals,
        feature_cols=["vol_stress", "trend_support"],
    )

    by_feature = deltas.set_index("feature")
    assert by_feature.index.tolist()[0] == "vol_stress"
    assert by_feature.loc["vol_stress", "interval_mean"] > by_feature.loc["vol_stress", "reference_mean"]
    assert by_feature.loc["trend_support", "delta_z"] < 0.0
    assert by_feature.loc["vol_stress", "candidate_avg_pos"] == 0.7
    assert by_feature.loc["vol_stress", "baseline_avg_pos"] == 0.35
    assert by_feature.loc["vol_stress", "position_delta"] == 0.35

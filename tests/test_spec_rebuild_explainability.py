from __future__ import annotations

import pandas as pd

from src.analysis import feature_importance_table, permutation_importance, window_sensitivity


def test_tabular_permutation_importance_and_feature_importance_table_are_ranked():
    frame = pd.DataFrame(
        {
            "x_signal": [1.0, 2.0, 3.0, 4.0],
            "x_noise": [5.0, 5.0, 5.0, 5.0],
            "target": [2.0, 4.0, 6.0, 8.0],
        }
    )

    def predict_fn(features: pd.DataFrame) -> pd.Series:
        return features["x_signal"] * 2.0

    importance = permutation_importance(
        frame,
        feature_cols=["x_signal", "x_noise"],
        target_col="target",
        predict_fn=predict_fn,
    )
    table = feature_importance_table({"x_noise": 0.0, "x_signal": 10.0})

    assert importance["feature"].tolist() == ["x_signal", "x_noise"]
    assert importance.loc[0, "importance"] > 0
    assert importance.loc[1, "importance"] == 0
    assert table.to_dict("records") == [
        {"feature": "x_signal", "importance": 10.0, "rank": 1},
        {"feature": "x_noise", "importance": 0.0, "rank": 2},
    ]


def test_window_sensitivity_reports_sequence_window_contribution():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "signal": [1.0, 2.0, -1.0, 4.0],
        }
    )

    sensitivity = window_sensitivity(
        frame,
        value_col="signal",
        windows={
            "early": ("2026-01-01", "2026-01-02"),
            "late": ("2026-01-03", "2026-01-04"),
        },
    )

    assert sensitivity.to_dict("records") == [
        {"window": "early", "start": "2026-01-01", "end": "2026-01-02", "contribution": 3.0, "share": 0.5},
        {"window": "late", "start": "2026-01-03", "end": "2026-01-04", "contribution": 3.0, "share": 0.5},
    ]

from __future__ import annotations

import pandas as pd

import src.backtest as backtest_module
from src.backtest import walk_forward_slices


def test_walk_forward_and_stress_window_metrics_include_risk_exposure_and_benchmark():
    result = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=6, freq="D"),
            "nav": [1.0, 1.10, 1.05, 1.20, 1.18, 1.30],
            "benchmark_nav": [1.0, 1.05, 1.04, 1.10, 1.08, 1.12],
            "turnover": [0.0, 0.2, 0.4, 0.1, 0.3, 0.2],
            "position": [0.0, 0.5, 1.0, 1.0, 0.6, 0.8],
        }
    )
    folds = walk_forward_slices(result["date"], train_size=2, test_size=2)
    assert hasattr(backtest_module, "walk_forward_metrics")
    assert hasattr(backtest_module, "stress_window_metrics")

    fold_metrics = backtest_module.walk_forward_metrics(result, folds=folds, benchmark_nav_col="benchmark_nav")
    stress_metrics = backtest_module.stress_window_metrics(
        result,
        windows={"shock": ("2026-01-04", "2026-01-07")},
        benchmark_nav_col="benchmark_nav",
    )

    expected_cols = {
        "window",
        "start",
        "end",
        "return_pct",
        "max_drawdown_pct",
        "sharpe",
        "turnover",
        "exposure",
        "benchmark_return_pct",
        "excess_return_pct",
    }
    assert expected_cols.issubset(fold_metrics.columns)
    assert fold_metrics["window"].tolist() == ["fold_1", "fold_2"]
    assert fold_metrics.loc[0, "turnover"] == 0.5
    assert fold_metrics.loc[0, "exposure"] == 1.0
    assert stress_metrics["window"].tolist() == ["shock"]
    assert stress_metrics.loc[0, "max_drawdown_pct"] < 0.0

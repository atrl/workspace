from __future__ import annotations

import pandas as pd

from src.analysis import compare_models, compute_regime_metrics
from src.backtest import (
    BacktestConfig,
    CostModel,
    PositionPolicy,
    build_walk_forward_splits,
    run_prediction_backtest,
)


def _predictions() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "model": ["ridge_mse"] * 10 + ["gcn_mse"] * 10,
            "prediction": [0.01, 0.02, -0.01, 0.03, 0.04, -0.03, -0.02, 0.01, 0.02, 0.03]
            + [0.02, 0.04, -0.02, 0.05, 0.05, -0.04, -0.04, 0.02, 0.03, 0.04],
            "target_return": [0.0, 0.02, -0.01, 0.03, 0.04, -0.05, -0.02, 0.01, 0.02, 0.03] * 2,
        }
    )


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "close": [100, 102, 101, 104, 108, 102, 100, 101, 103, 106],
            "vix": [18, 19, 22, 25, 28, 38, 35, 30, 24, 20],
        }
    )


def test_prediction_backtest_contract_accounts_for_costs_and_exports_analysis_frame():
    config = BacktestConfig(
        start="2026-01-01",
        end="2026-01-10",
        benchmark="SOXL",
        position_policy=PositionPolicy(min_position=0.0, max_position=1.0, neutral_position=0.5, lag=1),
        cost_model=CostModel(open_cost_bps=5.0, close_cost_bps=5.0, slippage_bps=2.0),
        stress_windows={"stress_window": ("2026-01-06", "2026-01-07")},
    )

    result = run_prediction_backtest(_predictions(), price_frame=_prices(), config=config)

    assert result.schema_version == "backtest.result.v1"
    assert result.config.schema_version == "backtest.config.v1"
    assert result.daily["cost_return"].abs().sum() > 0.0
    assert (result.daily["net_strategy_return"] <= result.daily["gross_strategy_return"] + 1e-12).all()
    assert set(result.daily["stress_window"].dropna()) == {"stress_window"}
    assert result.model_summary.iloc[0]["model"] in {"gcn_mse", "ridge_mse"}
    assert {"return_pct", "max_drawdown_pct", "avg_turnover", "utility_score"}.issubset(result.model_summary.columns)

    analysis_frame = result.to_analysis_frame()
    assert {"date", "model", "nav", "benchmark_nav", "position", "turnover", "vix", "trend", "drawdown"}.issubset(
        analysis_frame.columns
    )
    regime_metrics = compute_regime_metrics(analysis_frame)
    comparison = compare_models(analysis_frame)

    assert not regime_metrics.empty
    assert comparison.iloc[0]["rank"] == 1


def test_walk_forward_splits_are_lookback_only_and_schema_stable():
    splits = build_walk_forward_splits(
        pd.date_range("2026-01-01", periods=12, freq="D"),
        train_size=5,
        test_size=2,
        step_size=2,
    )

    assert [split.fold for split in splits] == [1, 2, 3]
    assert splits[0].train_start == "2026-01-01"
    assert splits[0].train_end == "2026-01-05"
    assert splits[0].test_start == "2026-01-06"
    assert splits[0].test_end == "2026-01-07"
    assert all(split.train_end < split.test_start for split in splits)
    assert {split.schema_version for split in splits} == {"backtest.walk_forward_split.v1"}

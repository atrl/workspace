from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.checkpoint_daily_export import (
    build_checkpoint_candidate_daily_export,
    metrics_from_official_backtest,
)


def test_build_checkpoint_candidate_daily_export_rebases_and_writes_path(tmp_path):
    dates = pd.bdate_range("2020-01-02", periods=4)
    backtest = {
        "portfolio": pd.DataFrame({"account": [1_000_000.0, 1_100_000.0, 1_050_000.0, 1_200_000.0]}, index=dates),
        "return": 20.0,
        "sharpe": 1.25,
        "max_dd": 4.55,
        "bnh_return": 50.0,
        "relative_outperformance": -30.0,
        "position_history": [0.0, 0.3, 0.3, 0.1],
        "trade_signals": [(dates[1], 110.0, 0.3, "buy"), (dates[3], 150.0, 0.1, "sell")],
        "effective_test_seg": ("2020-01-02", "2020-01-07"),
    }
    prices = pd.Series([100.0, 110.0, 90.0, 150.0], index=dates)

    export = build_checkpoint_candidate_daily_export(
        backtest_result=backtest,
        price_series=prices,
        output_dir=tmp_path,
        fold="wf_2020",
        candidate="ppo_retained_iter_4",
        candidate_kind="ppo_retained",
    )

    assert export.row["daily_path"] == str(export.path)
    assert export.row["daily_path_rows"] == 4
    assert export.row["daily_path_start"] == "2020-01-02"
    assert export.row["daily_path_end"] == "2020-01-07"
    assert export.row["test_return"] == 20.0
    assert export.row["test_relative_return"] == -30.0

    daily = pd.read_csv(export.path)
    assert daily["datetime"].tolist() == ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    assert daily["strategy_nav"].tolist() == [1_000_000.0, 1_100_000.0, 1_050_000.0, 1_200_000.0]
    assert daily["buyhold_nav"].tolist() == [1_000_000.0, 1_100_000.0, 900_000.0, 1_500_000.0]
    assert daily["position"].tolist() == [0.0, 0.3, 0.3, 0.1]
    assert daily["action"].tolist() == ["HOLD", "BUY", "HOLD", "SELL"]


def test_metrics_from_official_backtest_prefers_qlib_backtest_metrics():
    metrics = metrics_from_official_backtest(
        split_metrics={"return": -99.0, "sharpe": -1.0, "max_dd": 99.0},
        backtest_result={
            "return": 12.5,
            "sharpe": 0.8,
            "max_dd": 18.2,
            "bnh_return": 40.0,
            "relative_outperformance": -27.5,
        },
    )

    assert metrics["return"] == 12.5
    assert metrics["sharpe"] == 0.8
    assert metrics["max_dd"] == 18.2
    assert metrics["buy_and_hold_return"] == 40.0
    assert metrics["relative_return"] == -27.5

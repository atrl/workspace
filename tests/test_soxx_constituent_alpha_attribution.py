from __future__ import annotations

import numpy as np
import pandas as pd


def _synthetic_frames(rows: int = 18) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    soxx_returns = np.array([0.0, 0.01, -0.005, 0.012, -0.008, 0.006] * 3)[:rows]
    soxx_open = 100.0 * np.cumprod(1.0 + soxx_returns)
    soxl_open = [100.0]
    for ret in soxx_returns[1:]:
        soxl_open.append(soxl_open[-1] * (1.0 + 3.0 * ret))
    index_frame = pd.DataFrame(
        {
            "datetime": dates,
            "soxx_open": soxx_open,
            "soxl_open": soxl_open,
        }
    )

    frames = []
    for instrument, scale, weight in [("AAA", 1.2, 60.0), ("BBB", 0.7, 40.0)]:
        frames.append(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "instrument": instrument,
                    "adj_close": 50.0 * np.cumprod(1.0 + soxx_returns * scale),
                    "weight_pct": weight,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), index_frame


def test_constituent_attribution_panel_lags_signals_and_fits_beta_on_train_only():
    import src.analysis as analysis
    from src.analysis.soxx_constituent_attribution import build_constituent_attribution_panel

    assert {"AnalysisReport", "build_analysis_report_from_qlib_recorder"}.issubset(analysis.__all__)

    constituents, index_frame = _synthetic_frames()
    result = build_constituent_attribution_panel(
        constituents,
        index_frame,
        train=("2024-01-02", "2024-01-15"),
        lookback=2,
        lag_days=1,
        horizon=5,
    )

    aaa = result.constituents.loc[result.constituents["instrument"].eq("AAA")].reset_index(drop=True)
    prices = constituents.loc[constituents["instrument"].eq("AAA"), "adj_close"].reset_index(drop=True)
    expected = prices.pct_change(2, fill_method=None).shift(1)

    assert np.allclose(aaa["raw_signal"].fillna(0.0), expected.fillna(0.0))
    assert set(result.betas) == {"AAA", "BBB"}
    assert result.daily["datetime"].max() == index_frame["datetime"].max()
    assert {"weighted_raw", "weighted_residual", "equal_raw", "breadth_positive"}.issubset(result.daily)


def test_attribution_metrics_orient_signals_from_train_and_map_daily_reset_leverage():
    from src.analysis.soxx_constituent_attribution import (
        build_constituent_attribution_panel,
        evaluate_constituent_attribution,
    )

    constituents, index_frame = _synthetic_frames(rows=18)
    result = build_constituent_attribution_panel(
        constituents,
        index_frame,
        train=("2024-01-02", "2024-01-15"),
        lookback=2,
        lag_days=1,
        horizon=5,
    )
    metrics = evaluate_constituent_attribution(
        result,
        segments={
            "train": ("2024-01-02", "2024-01-15"),
            "validation": ("2024-01-16", "2024-01-19"),
            "test": ("2024-01-22", "2024-01-25"),
        },
    )

    assert set(metrics) == {"individual", "aggregate", "leverage", "stability", "nonoverlap"}
    assert set(metrics["aggregate"]["target"]) == {"soxx_5d", "soxl_5d"}
    assert metrics["aggregate"].loc[metrics["aggregate"]["segment"].eq("train"), "train_direction"].notna().all()
    assert set(metrics["nonoverlap"]["offset"]) == set(range(5))
    assert set(metrics["nonoverlap"]["target"]) == {"soxx_5d", "soxl_5d"}
    assert metrics["nonoverlap"]["overlap_purged"].all()
    leverage = metrics["leverage"].loc[metrics["leverage"]["segment"].eq("train")].iloc[0]
    assert abs(float(leverage["tracking_residual_mean_pct"])) < 1e-10
    assert abs(float(leverage["tracking_rmse_pct"])) < 1e-10


def test_execution_censoring_reports_zero_raw_targets_hidden_by_position_floor():
    from src.analysis.soxx_constituent_attribution import summarize_execution_censoring

    dates = pd.date_range("2026-01-02", periods=6, freq="B")
    positions = pd.DataFrame(
        {
            "datetime": dates,
            "candidate": "selected",
            "segment": "test",
            "probability": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "raw_target": [0.0, 0.0, 0.0, 0.0, 0.1, 0.2],
            "target_position": [0.015] * 6,
            "turnover": [0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
            "selected": True,
        }
    )
    labels = pd.DataFrame(
        {
            "datetime": dates,
            "soxl_5d": [0.03, 0.02, 0.01, -0.01, -0.02, -0.03],
        }
    )

    metrics = summarize_execution_censoring(positions, labels)

    row = metrics.iloc[0]
    assert np.isclose(row["raw_zero_ratio"], 4.0 / 6.0)
    assert row["position_floor"] == 0.015
    assert row["position_floor_ratio"] == 1.0
    assert row["position_distinct_count"] == 1
    assert row["probability_rank_correlation"] < 0.0

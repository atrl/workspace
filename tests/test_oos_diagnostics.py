from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.oos_diagnostics import (
    compute_feature_differences,
    find_poor_windows,
    render_oos_diagnostic_review,
)


def test_find_poor_windows_ranks_negative_excess_and_drawdown():
    daily = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=6, freq="B"),
            "strategy_nav": [100.0, 102.0, 101.0, 96.0, 91.0, 80.0],
            "buyhold_nav": [100.0, 105.0, 109.0, 112.0, 116.0, 118.0],
            "position": [0.6, 0.6, 0.5, 0.4, 0.3, 0.3],
        }
    )

    windows = find_poor_windows(daily, window_size=3, top_n=2)

    assert windows["start"].tolist() == ["2024-01-04", "2024-01-03"]
    assert windows.iloc[0]["end"] == "2024-01-08"
    assert windows.iloc[0]["excess_return"] < -20.0
    assert windows.iloc[0]["max_dd"] > 1.0
    assert windows.iloc[0]["bad_score"] > windows.iloc[1]["bad_score"]


def test_compute_feature_differences_compares_bad_windows_to_reference():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    features = pd.DataFrame(
        {
            "trend_pressure": [0.1, 0.2, 0.3, 1.5, 1.6, 1.7],
            "vol_stress": [2.0, 2.1, 2.2, 0.2, 0.1, 0.0],
        },
        index=dates,
    )
    poor_windows = pd.DataFrame(
        {
            "start": ["2024-01-04"],
            "end": ["2024-01-08"],
            "bad_score": [25.0],
        }
    )

    diffs = compute_feature_differences(features, poor_windows, top_n=2)

    by_feature = diffs.set_index("feature")

    assert set(by_feature.index) == {"trend_pressure", "vol_stress"}
    assert by_feature.loc["trend_pressure", "bad_mean"] > by_feature.loc["trend_pressure", "reference_mean"]
    assert by_feature.loc["trend_pressure", "delta_z"] > 1.0
    assert by_feature.loc["vol_stress", "delta_z"] < -1.0


def test_render_oos_diagnostic_review_includes_thresholds_sources_and_next_actions():
    windows = pd.DataFrame(
        {
            "start": ["2024-01-04"],
            "end": ["2024-01-08"],
            "strategy_return": [-5.0],
            "buyhold_return": [12.0],
            "excess_return": [-17.0],
            "max_dd": [8.0],
            "avg_pos": [0.4],
            "bad_score": [25.0],
        }
    )
    diffs = pd.DataFrame(
        {
            "feature": ["trend_pressure"],
            "bad_mean": [1.6],
            "reference_mean": [0.9],
            "delta_z": [1.4],
            "spearman_to_excess": [-0.5],
        }
    )

    review = render_oos_diagnostic_review(
        profile="demo_profile",
        train_end="2018-12-31",
        oos_start="2020-01-02",
        oos_end="2026-05-21",
        target_excess=30.0,
        target_max_dd=15.0,
        windows=windows,
        feature_differences=diffs,
        external_references=[
            {
                "label": "Deflated Sharpe Ratio",
                "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
                "note": "multiple-testing correction",
            }
        ],
    )

    assert "# OOS Poor-Window Diagnostic: demo_profile" in review
    assert "target_excess=30.00%" in review
    assert "target_max_dd=15.00%" in review
    assert "Deflated Sharpe Ratio" in review
    assert "trend_pressure" in review
    assert "Next Actions" in review

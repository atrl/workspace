from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.soxl_quant.app.oos_diagnostics import (
    aggregate_feature_differences,
    aggregate_poor_windows,
    classify_poor_window,
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


def test_classify_poor_window_separates_underparticipation_and_drawdown():
    under = pd.Series(
        {
            "buyhold_return": 120.0,
            "excess_return": -90.0,
            "max_dd": 8.0,
            "avg_pos": 0.12,
        }
    )
    drawdown = pd.Series(
        {
            "buyhold_return": -15.0,
            "excess_return": -10.0,
            "max_dd": 24.0,
            "avg_pos": 0.72,
        }
    )

    assert classify_poor_window(under) == "underparticipation"
    assert classify_poor_window(drawdown) == "drawdown_control"


def test_classify_poor_window_treats_partial_exposure_in_strong_rebound_as_underparticipation():
    rebound = pd.Series(
        {
            "buyhold_return": 220.0,
            "excess_return": -150.0,
            "max_dd": 16.0,
            "avg_pos": 0.52,
        }
    )

    assert classify_poor_window(rebound) == "underparticipation"


def test_aggregate_poor_windows_clusters_overlapping_profiles():
    left = pd.DataFrame(
        {
            "start": ["2026-02-10", "2025-04-08"],
            "end": ["2026-05-11", "2025-07-09"],
            "strategy_return": [1.7, 82.8],
            "buyhold_return": [200.6, 223.0],
            "excess_return": [-198.9, -140.2],
            "max_dd": [6.4, 16.3],
            "avg_pos": [0.14, 0.55],
            "bad_score": [205.3, 156.5],
        }
    )
    right = pd.DataFrame(
        {
            "start": ["2026-02-20", "2024-08-01"],
            "end": ["2026-05-20", "2024-10-30"],
            "strategy_return": [-3.1, -18.0],
            "buyhold_return": [158.1, -2.0],
            "excess_return": [-161.2, -16.0],
            "max_dd": [6.4, 27.0],
            "avg_pos": [0.05, 0.70],
            "bad_score": [167.6, 43.0],
        }
    )

    clusters = aggregate_poor_windows({"left": left, "right": right})

    assert clusters.iloc[0]["cluster_start"] == "2026-02-10"
    assert clusters.iloc[0]["cluster_end"] == "2026-05-20"
    assert clusters.iloc[0]["profile_count"] == 2
    assert clusters.iloc[0]["window_count"] == 2
    assert clusters.iloc[0]["dominant_failure_type"] == "underparticipation"
    assert clusters["cluster_id"].is_unique


def test_aggregate_feature_differences_keeps_consistent_direction():
    left = pd.DataFrame(
        {
            "feature": ["mom_21", "vol_21", "noise"],
            "delta_z": [0.8, 1.2, 0.1],
            "spearman_to_excess": [-0.2, 0.1, 0.0],
        }
    )
    right = pd.DataFrame(
        {
            "feature": ["mom_21", "vol_21", "noise"],
            "delta_z": [0.6, -1.0, -0.1],
            "spearman_to_excess": [-0.3, 0.2, 0.0],
        }
    )

    summary = aggregate_feature_differences({"left": left, "right": right}, top_n=3)
    by_feature = summary.set_index("feature")

    assert by_feature.loc["mom_21", "profile_count"] == 2
    assert by_feature.loc["mom_21", "direction_consistency"] == 1.0
    assert by_feature.loc["vol_21", "direction_consistency"] == 0.0
    assert by_feature.index[0] == "mom_21"


def test_aggregate_oos_script_writes_cluster_and_feature_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "demo"
        source.mkdir()
        pd.DataFrame(
            {
                "start": ["2026-02-10"],
                "end": ["2026-05-11"],
                "strategy_return": [1.7],
                "buyhold_return": [200.6],
                "excess_return": [-198.9],
                "max_dd": [6.4],
                "avg_pos": [0.14],
                "bad_score": [205.3],
            }
        ).to_csv(source / "poor_windows.csv", index=False)
        pd.DataFrame(
            {
                "feature": ["mom_21"],
                "delta_z": [0.8],
                "spearman_to_excess": [-0.2],
            }
        ).to_csv(source / "feature_differences.csv", index=False)
        output = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/aggregate_oos_poor_windows.py",
                "--input-dir",
                str(source),
                "--label",
                "demo",
                "--output-dir",
                str(output),
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )

        assert result.returncode == 0, result.stderr
        assert "oos_aggregate" in result.stdout
        assert (output / "poor_window_clusters.csv").exists()
        assert (output / "feature_delta_clusters.csv").exists()
        assert pd.read_csv(output / "poor_window_clusters.csv").iloc[0]["dominant_failure_type"] == "underparticipation"

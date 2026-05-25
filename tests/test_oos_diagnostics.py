from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.soxl_quant.app.oos_diagnostics import (
    aggregate_feature_differences,
    aggregate_poor_windows,
    attribute_gate_gaps,
    classify_poor_window,
    compute_feature_differences,
    find_poor_windows,
    normalise_gate_metrics_from_selection,
    render_gate_gap_review,
    render_oos_diagnostic_review,
    summarize_gate_gap_attribution,
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


def test_normalise_gate_metrics_from_selection_handles_reward_and_selector_shapes():
    reward_selection = {
        "metrics": {
            "test_proxy": {
                "return": 619.7,
                "buy_and_hold_return": 2542.0,
                "relative_return": -1470.0,
                "max_dd": 43.6,
                "avg_pos": 0.38,
            },
            "walkforward": [
                {
                    "label": "wf_2023_2024",
                    "return": -3.2,
                    "buy_and_hold_return": -40.0,
                    "relative_return": 37.0,
                    "max_dd": 15.2,
                    "avg_pos": 0.16,
                }
            ],
        }
    }
    selector_selection = {
        "selected_metrics": {
            "strict_oos": {
                "return": 1667.8,
                "buy_and_hold_return": 816.5,
                "relative_return": 851.3,
                "max_dd": 43.5,
                "avg_pos": 0.45,
            }
        }
    }

    reward_metrics = normalise_gate_metrics_from_selection(reward_selection)
    selector_metrics = normalise_gate_metrics_from_selection(selector_selection)

    assert sorted(reward_metrics) == ["test_proxy", "wf_2023_2024"]
    assert reward_metrics["wf_2023_2024"]["relative_return"] == 37.0
    assert selector_metrics["strict_oos"]["max_dd"] == 43.5


def test_normalise_gate_metrics_from_selection_handles_top_level_slices_and_pbo_rows():
    selection = {
        "test_proxy": {
            "return": 496.8,
            "buy_and_hold_return": 2542.0,
            "relative_return": -2107.5,
            "max_dd": 44.2,
            "avg_pos": 0.27,
        },
        "slices": {
            "test_2020": {
                "return": 51.0,
                "buy_and_hold_return": 355.7,
                "relative_return": -304.7,
                "max_dd": 21.4,
                "avg_pos": 0.30,
            }
        },
        "walk_forward": {
            "wf_2020": {
                "test_proxy": {
                    "return": 50.1,
                    "buy_and_hold_return": 355.7,
                    "relative_return": -304.7,
                    "max_dd": 23.1,
                    "avg_pos": 0.30,
                }
            }
        },
        "window_best_rows": [
            {
                "window": "wf_2022",
                "best_oos_relative_return": 65.6,
                "best_oos_max_dd": 30.7,
            }
        ],
    }

    metrics = normalise_gate_metrics_from_selection(selection)

    assert sorted(metrics) == ["test_2020", "test_proxy", "wf_2020", "wf_2022"]
    assert metrics["wf_2020"]["relative_return"] == -304.7
    assert metrics["wf_2022"]["relative_return"] == 65.6
    assert metrics["wf_2022"]["max_dd"] == 30.7


def test_attribute_gate_gaps_classifies_near_miss_joint_tradeoff_and_summary():
    rows = attribute_gate_gaps(
        {
            "state_reward": {
                "test_proxy": {
                    "return": 619.7,
                    "buy_and_hold_return": 2542.0,
                    "relative_return": -1470.0,
                    "max_dd": 43.6,
                    "avg_pos": 0.38,
                },
                "wf_2023_2024": {
                    "return": -3.2,
                    "buy_and_hold_return": -40.0,
                    "relative_return": 37.0,
                    "max_dd": 15.2,
                    "avg_pos": 0.16,
                },
                "wf_2022": {
                    "return": -23.3,
                    "buy_and_hold_return": -76.6,
                    "relative_return": 45.1,
                    "max_dd": 29.1,
                    "avg_pos": 0.24,
                },
            }
        },
        target_relative_return=30.0,
        target_max_dd=15.0,
    )
    by_window = rows.set_index("window")

    assert by_window.loc["test_proxy", "failure_type"] == "joint_tradeoff"
    assert by_window.loc["wf_2023_2024", "failure_type"] == "near_miss_drawdown"
    assert by_window.loc["wf_2022", "failure_type"] == "drawdown_control"

    summary = summarize_gate_gap_attribution(rows)
    assert summary.iloc[0]["profile"] == "state_reward"
    assert summary.iloc[0]["dominant_failure_type"] == "joint_tradeoff"
    assert summary.iloc[0]["recommended_next_mechanism"] == "walk_forward_weighted_multi_objective_selection"

    review = render_gate_gap_review(rows, summary)
    assert "Gate-Gap Failure Attribution" in review
    assert "state_reward" in review
    assert "walk_forward_weighted_multi_objective_selection" in review


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
        selection_path = tmp_path / "selection.json"
        selection_path.write_text(
            """
            {
              "metrics": {
                "test_proxy": {
                  "return": 619.7,
                  "buy_and_hold_return": 2542.0,
                  "relative_return": -1470.0,
                  "max_dd": 43.6,
                  "avg_pos": 0.38
                }
              }
            }
            """,
            encoding="utf-8",
        )
        output = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/aggregate_oos_poor_windows.py",
                "--input-dir",
                str(source),
                "--label",
                "demo",
                "--selection-json",
                str(selection_path),
                "--selection-label",
                "demo_selection",
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
        assert (output / "gate_gap_attribution.csv").exists()
        assert (output / "gate_gap_profile_summary.csv").exists()
        assert pd.read_csv(output / "poor_window_clusters.csv").iloc[0]["dominant_failure_type"] == "underparticipation"
        assert pd.read_csv(output / "gate_gap_attribution.csv").iloc[0]["failure_type"] == "joint_tradeoff"

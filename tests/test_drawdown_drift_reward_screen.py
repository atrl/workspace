from __future__ import annotations

import json

import pandas as pd

from src.soxl_quant.app.drawdown_drift_reward_screen import (
    DEFAULT_RECIPE_GRID,
    render_drawdown_drift_reward_screen_review,
    screen_drawdown_drift_reward_recipes,
    summarize_drawdown_drift_reward_screen,
)


def test_drawdown_drift_screen_selects_joint_recipe_before_training():
    daily_paths = {
        "short_volshock_plain__wf_2022": pd.DataFrame(
            {
                "datetime": pd.date_range("2022-01-03", periods=5, freq="B"),
                "strategy_nav": [100.0, 98.0, 82.0, 85.0, 84.0],
                "buyhold_nav": [100.0, 96.0, 80.0, 78.0, 76.0],
                "position": [0.08, 0.18, 0.22, 0.12, 0.09],
            }
        ),
        "short_volshock_plain__wf_2023_2024": pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-02", periods=6, freq="B"),
                "strategy_nav": [100.0, 99.0, 101.0, 102.0, 103.0, 104.0],
                "buyhold_nav": [100.0, 103.0, 108.0, 114.0, 120.0, 124.0],
                "position": [0.12, 0.14, 0.16, 0.18, 0.19, 0.20],
            }
        ),
    }
    gate_gaps = pd.DataFrame(
        [
            {
                "profile": "short_volshock_plain",
                "window": "wf_2022",
                "failure_type": "drawdown_control",
                "relative_gap": 0.0,
                "max_dd_gap": 7.2,
                "gate_pass": False,
            },
            {
                "profile": "short_volshock_plain",
                "window": "wf_2023_2024",
                "failure_type": "joint_tradeoff",
                "relative_gap": 25.0,
                "max_dd_gap": 1.7,
                "gate_pass": False,
            },
        ]
    )
    recipe_grid = pd.DataFrame(
        [
            {
                "recipe": "drawdown_only",
                "drawdown_penalty": 3.0,
                "drawdown_threshold_pct": 15.0,
                "position_threshold": 0.10,
                "trend_floor_penalty": 0.0,
                "trend_floor_position": 0.35,
                "trend_lookback": 2,
                "min_buyhold_return_pct": 2.0,
            },
            {
                "recipe": "joint_drawdown_trend",
                "drawdown_penalty": 3.0,
                "drawdown_threshold_pct": 15.0,
                "position_threshold": 0.10,
                "trend_floor_penalty": 2.0,
                "trend_floor_position": 0.35,
                "trend_lookback": 2,
                "min_buyhold_return_pct": 2.0,
            },
            {
                "recipe": "trend_only",
                "drawdown_penalty": 0.0,
                "drawdown_threshold_pct": 15.0,
                "position_threshold": 0.10,
                "trend_floor_penalty": 2.0,
                "trend_floor_position": 0.35,
                "trend_lookback": 2,
                "min_buyhold_return_pct": 2.0,
            },
        ]
    )

    scored = screen_drawdown_drift_reward_recipes(
        daily_paths,
        gate_gaps=gate_gaps,
        recipe_grid=recipe_grid,
        selected_candidate="short_volshock_plain",
    )
    summary = summarize_drawdown_drift_reward_screen(scored)
    review = render_drawdown_drift_reward_screen_review(scored, summary)

    top = scored.iloc[0]
    patch = json.loads(str(top["reward_objective_patch_json"]))
    assert top["recipe"] == "joint_drawdown_trend"
    assert bool(top["recipe_ready"]) is True
    assert top["drawdown_failure_capture"] >= 0.50
    assert top["trend_underparticipation_capture"] >= 0.50
    assert summary["decision"] == "drawdown_drift_reward_recipe_ready_for_short_training_screen"
    assert summary["training_allowed_next"] is True
    assert patch["drawdown_drift_reward_enabled"] is True
    assert patch["trend_preservation_floor_enabled"] is True
    assert "joint_drawdown_trend" in review


def test_drawdown_drift_screen_blocks_one_sided_recipe():
    daily_paths = {
        "short_volshock_plain__wf_2022": pd.DataFrame(
            {
                "datetime": pd.date_range("2022-01-03", periods=4, freq="B"),
                "strategy_nav": [100.0, 99.0, 83.0, 84.0],
                "buyhold_nav": [100.0, 97.0, 82.0, 81.0],
                "position": [0.12, 0.16, 0.20, 0.09],
            }
        )
    }
    gate_gaps = pd.DataFrame(
        [
            {
                "profile": "short_volshock_plain",
                "window": "wf_2022",
                "failure_type": "drawdown_control",
                "relative_gap": 0.0,
                "max_dd_gap": 2.0,
                "gate_pass": False,
            }
        ]
    )
    recipe_grid = pd.DataFrame(
        [
            {
                "recipe": "drawdown_only",
                "drawdown_penalty": 2.0,
                "drawdown_threshold_pct": 15.0,
                "position_threshold": 0.10,
                "trend_floor_penalty": 0.0,
                "trend_floor_position": 0.35,
                "trend_lookback": 2,
                "min_buyhold_return_pct": 2.0,
            }
        ]
    )

    scored = screen_drawdown_drift_reward_recipes(
        daily_paths,
        gate_gaps=gate_gaps,
        recipe_grid=recipe_grid,
        selected_candidate="short_volshock_plain",
    )
    summary = summarize_drawdown_drift_reward_screen(scored)

    assert bool(scored.iloc[0]["recipe_ready"]) is False
    assert summary["decision"] == "drawdown_drift_reward_recipe_no_training_signal"
    assert summary["training_allowed_next"] is False


def test_drawdown_drift_screen_does_not_penalize_flat_no_position_drawdown_tail():
    daily_paths = {
        "candidate__wf_2020": pd.DataFrame(
            {
                "datetime": pd.date_range("2020-01-02", periods=7, freq="B"),
                "strategy_nav": [100.0, 98.0, 84.0, 80.0, 80.5, 81.0, 82.0],
                "buyhold_nav": [100.0, 104.0, 110.0, 114.0, 120.0, 126.0, 130.0],
                "position": [0.20, 0.25, 0.30, 0.35, 0.0, 0.0, 0.0],
            }
        )
    }
    gate_gaps = pd.DataFrame(
        [
            {
                "profile": "candidate",
                "window": "wf_2020",
                "failure_type": "joint_tradeoff",
                "relative_gap": 30.0,
                "max_dd_gap": 5.0,
                "gate_pass": False,
            }
        ]
    )
    recipe_grid = pd.DataFrame(
        [
            {
                "recipe": "joint_active_exposure",
                "drawdown_penalty": 2.0,
                "drawdown_threshold_pct": 15.0,
                "position_threshold": 0.10,
                "trend_floor_penalty": 2.0,
                "trend_floor_position": 0.35,
                "trend_lookback": 1,
                "min_buyhold_return_pct": 2.0,
            }
        ]
    )

    scored = screen_drawdown_drift_reward_recipes(
        daily_paths,
        gate_gaps=gate_gaps,
        recipe_grid=recipe_grid,
        selected_candidate="candidate",
    )
    summary = summarize_drawdown_drift_reward_screen(scored)
    top = scored.iloc[0]

    assert bool(top["recipe_ready"]) is True
    assert top["drawdown_failure_capture"] == 1.0
    assert top["raw_drawdown_failure_capture"] < 0.50
    assert top["inactive_drawdown_breach_rows"] == 3
    assert summary["decision"] == "drawdown_drift_reward_recipe_ready_for_short_training_screen"


def test_default_grid_includes_short_volshock_low_position_threshold():
    thresholds = {float(row["position_threshold"]) for row in DEFAULT_RECIPE_GRID}

    assert 0.05 in thresholds

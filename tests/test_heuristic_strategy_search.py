from __future__ import annotations

import pandas as pd


def _demo_panel() -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=8, freq="B")
    return pd.DataFrame(
        {
            "datetime": dates,
            "soxl_close": [100.0, 80.0, 82.0, 84.0, 90.0, 100.0, 112.0, 125.0],
            "soxx_close": [100.0, 98.0, 99.0, 101.0, 103.0, 108.0, 114.0, 120.0],
            "vix": [28.0, 31.0, 27.0, 22.0, 18.0, 16.0, 15.0, 14.0],
        }
    )


def test_heuristic_strategy_metrics_require_non_empty_windows():
    from src.soxl_quant.app.heuristic_strategy_search import (
        HeuristicStrategyConfig,
        evaluate_heuristic_strategy,
    )

    result = evaluate_heuristic_strategy(
        _demo_panel(),
        HeuristicStrategyConfig(),
        windows={"missing": ("2025-01-01", "2025-12-31")},
    )

    assert result.gate_pass is False
    assert result.window_metrics["missing"].rows == 0
    assert "empty_window" in result.window_metrics["missing"].failures


def test_heuristic_strategy_can_beat_buyhold_with_drawdown_cap_on_demo_panel():
    from src.soxl_quant.app.heuristic_strategy_search import (
        HeuristicStrategyConfig,
        evaluate_heuristic_strategy,
    )

    result = evaluate_heuristic_strategy(
        _demo_panel(),
        HeuristicStrategyConfig(
            momentum_lookback=2,
            soxx_momentum_lookback=2,
            vix_threshold=24.0,
            bull_position=1.0,
            risk_position=0.0,
            cash_position=0.0,
            drawdown_start=0.05,
            drawdown_full=0.10,
            drawdown_min_position=0.0,
        ),
        windows={"demo": ("2022-01-03", "2022-01-12")},
        max_drawdown_pct=20.0,
    )

    metrics = result.window_metrics["demo"]
    assert result.gate_pass is True
    assert metrics.relative_return_pct > 0
    assert metrics.max_drawdown_pct < 20.0
    assert result.daily["position"].max() == 1.0


def test_screen_heuristic_strategies_ranks_passing_configs_first():
    from src.soxl_quant.app.heuristic_strategy_search import (
        HeuristicStrategyConfig,
        screen_heuristic_strategies,
    )

    configs = [
        HeuristicStrategyConfig(
            momentum_lookback=2,
            soxx_momentum_lookback=2,
            vix_threshold=24.0,
            bull_position=1.0,
            risk_position=0.0,
            cash_position=0.0,
        ),
        HeuristicStrategyConfig(
            momentum_lookback=2,
            soxx_momentum_lookback=2,
            vix_threshold=10.0,
            bull_position=0.0,
            risk_position=0.0,
            cash_position=0.0,
        ),
    ]

    rows = screen_heuristic_strategies(
        _demo_panel(),
        configs,
        windows={"demo": ("2022-01-03", "2022-01-12")},
    )

    assert rows.iloc[0]["gate_pass"] is True
    assert rows.iloc[0]["config_id"] == 0
    assert rows.iloc[0]["demo_relative_return_pct"] > rows.iloc[1]["demo_relative_return_pct"]


def test_default_heuristic_grid_includes_recoverable_drawdown_budgets():
    from src.soxl_quant.app.heuristic_strategy_search import default_heuristic_config_grid

    configs = default_heuristic_config_grid()

    assert any(config.drawdown_min_position > 0 for config in configs)
    assert any(config.drawdown_start >= 0.95 and config.drawdown_full >= 0.95 for config in configs)
    assert any(config.require_soxx_confirmation is False for config in configs)

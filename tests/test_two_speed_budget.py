from __future__ import annotations

from src.soxl_quant.policy.two_speed_budget import (
    TwoSpeedBudgetConfig,
    compose_budget_usage_position,
    compose_two_speed_position,
    infer_mode_from_slow_position,
    maybe_rebalance_position,
)


def test_compose_two_speed_position_caps_fast_signal_by_mode_slack():
    cfg = TwoSpeedBudgetConfig(mode_slack={"flat_wait": 0.05, "trend_hold": 0.35})

    assert compose_two_speed_position(0.20, 0.80, "flat_wait", cfg) == 0.25
    assert compose_two_speed_position(0.40, 0.80, "trend_hold", cfg) == 0.75


def test_compose_two_speed_position_keeps_fast_derisking():
    cfg = TwoSpeedBudgetConfig(mode_slack={"defensive_hold": 0.10})

    assert compose_two_speed_position(0.45, 0.20, "defensive_hold", cfg) == 0.20


def test_maybe_rebalance_position_filters_small_moves_and_honors_lock():
    cfg = TwoSpeedBudgetConfig(trade_threshold=0.04, trade_lock_days=3)

    assert maybe_rebalance_position(0.50, 0.53, 4, cfg) == 0.50
    assert maybe_rebalance_position(0.50, 0.60, 2, cfg) == 0.50
    assert maybe_rebalance_position(0.50, 0.60, 3, cfg) == 0.60


def test_infer_mode_from_slow_position_maps_defensive_bands():
    assert infer_mode_from_slow_position(0.18) == "flat_wait"
    assert infer_mode_from_slow_position(0.32) == "defensive_hold"
    assert infer_mode_from_slow_position(0.48) == "crash_rebuild"
    assert infer_mode_from_slow_position(0.70) == "trend_hold"


def test_compose_budget_usage_position_maps_fast_signal_inside_mode_band():
    cfg = TwoSpeedBudgetConfig(
        mode_bands={"flat_wait": (0.0, 0.25), "trend_hold": (0.45, 0.95)}
    )

    assert compose_budget_usage_position(0.80, "flat_wait", cfg) == 0.20
    assert compose_budget_usage_position(0.80, "trend_hold", cfg) == 0.85


def test_compose_budget_usage_position_preserves_full_derisking():
    cfg = TwoSpeedBudgetConfig(mode_bands={"crash_rebuild": (0.25, 0.65)})

    assert compose_budget_usage_position(0.0, "crash_rebuild", cfg) == 0.25


if __name__ == "__main__":
    test_compose_two_speed_position_caps_fast_signal_by_mode_slack()
    test_compose_two_speed_position_keeps_fast_derisking()
    test_maybe_rebalance_position_filters_small_moves_and_honors_lock()
    test_infer_mode_from_slow_position_maps_defensive_bands()
    test_compose_budget_usage_position_maps_fast_signal_inside_mode_band()
    test_compose_budget_usage_position_preserves_full_derisking()
    print("test_two_speed_budget: OK")

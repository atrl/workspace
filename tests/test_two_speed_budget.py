from __future__ import annotations

import pandas as pd

from src.soxl_quant.policy.two_speed_budget import (
    BullParticipationGuardConfig,
    EquityDrawdownControlConfig,
    RecoveryReentryGuardConfig,
    TwoSpeedBudgetConfig,
    compose_budget_usage_policy_frame,
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


def test_compose_budget_usage_policy_frame_keeps_audit_columns():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    slow = pd.DataFrame({"datetime": dates, "slow_position": [0.20, 0.50, 0.70]})
    fast = pd.DataFrame({"datetime": dates, "fast_position": [1.00, 0.00, 0.50]})
    cfg = TwoSpeedBudgetConfig(trade_threshold=0.0, trade_lock_days=0)

    out = compose_budget_usage_policy_frame(slow, fast, cfg)

    assert out["mode"].tolist() == ["flat_wait", "crash_rebuild", "trend_hold"]
    assert out["raw_two_speed_position"].round(4).tolist() == [0.25, 0.25, 0.70]
    assert out["position"].round(4).tolist() == [0.25, 0.25, 0.70]
    assert out["fast_usage"].tolist() == [1.0, 0.0, 0.5]


def test_compose_budget_usage_policy_frame_applies_bull_participation_guard():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    slow = pd.DataFrame({"datetime": dates, "slow_position": [0.20] * 5})
    fast = pd.DataFrame({"datetime": dates, "fast_position": [0.70] * 5})
    price = pd.Series([100.0, 103.0, 106.0, 109.0, 112.0], index=dates, name="price")
    cfg = TwoSpeedBudgetConfig(trade_threshold=0.0, trade_lock_days=0)

    out = compose_budget_usage_policy_frame(
        slow,
        fast,
        cfg,
        price_series=price,
        bull_guard=BullParticipationGuardConfig(
            enabled=True,
            ma_window=2,
            momentum_window=1,
            high_window=2,
            min_periods=1,
            min_momentum=0.01,
            max_drawdown=0.02,
        ),
    )

    assert out["bull_participation_guard"].tolist() == [False, True, True, True, True]
    assert out["base_two_speed_position"].round(4).tolist() == [0.175] * 5
    assert out["raw_two_speed_position"].round(4).tolist() == [0.175, 0.7, 0.7, 0.7, 0.7]
    assert out["position"].round(4).tolist() == [0.175, 0.7, 0.7, 0.7, 0.7]


def test_recovery_reentry_guard_lifts_flat_actor_only_in_high_vol_rebound():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    slow = pd.DataFrame({"datetime": dates, "slow_position": [0.0] * 5})
    fast = pd.DataFrame({"datetime": dates, "fast_position": [0.0] * 5})
    features = pd.DataFrame(
        {
            "datetime": dates,
            "mom_10": [0.01, 0.09, 0.12, 0.04, 0.11],
            "mom_21": [0.02, 0.15, 0.18, 0.16, 0.17],
            "relative_strength_21": [0.3, 1.1, 1.2, 1.3, 0.4],
            "vol_of_vol_21": [0.05, 0.13, 0.15, 0.16, 0.17],
            "vix_percentile_252": [0.40, 0.70, 0.75, 0.95, 0.72],
        }
    )
    cfg = TwoSpeedBudgetConfig(trade_threshold=0.0, trade_lock_days=0)

    out = compose_budget_usage_policy_frame(
        slow,
        fast,
        cfg,
        feature_frame=features,
        recovery_guard=RecoveryReentryGuardConfig(
            enabled=True,
            min_target=0.35,
            max_target=0.45,
            max_base_target=0.20,
            min_mom_10=0.08,
            min_mom_21=0.12,
            min_relative_strength_21=1.0,
            min_vol_of_vol_21=0.10,
            max_vix_percentile_252=0.85,
        ),
    )

    assert out["recovery_reentry_guard"].tolist() == [False, True, True, False, False]
    assert out["base_two_speed_position"].round(4).tolist() == [0.0] * 5
    assert out["raw_two_speed_position"].round(4).tolist() == [0.0, 0.35, 0.35, 0.0, 0.0]
    assert out["position"].round(4).tolist() == [0.0, 0.35, 0.35, 0.0, 0.0]


def test_recovery_reentry_guard_requires_long_anchor_confirmation_when_configured():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    slow = pd.DataFrame({"datetime": dates, "slow_position": [0.0] * 3})
    fast = pd.DataFrame({"datetime": dates, "fast_position": [0.0] * 3})
    features = pd.DataFrame(
        {
            "datetime": dates,
            "mom_10": [0.12, 0.12, 0.12],
            "mom_21": [0.18, 0.18, 0.18],
            "relative_strength_21": [1.4, 1.4, 1.4],
            "vol_of_vol_21": [0.13, 0.13, 0.13],
            "vix_percentile_252": [0.70, 0.70, 0.70],
            "below_anchor_ratio_63": [1.0, 0.10, 0.10],
            "anchor_gap_180": [-0.30, 0.20, 0.80],
        }
    )

    out = compose_budget_usage_policy_frame(
        slow,
        fast,
        TwoSpeedBudgetConfig(trade_threshold=0.0, trade_lock_days=0),
        feature_frame=features,
        recovery_guard=RecoveryReentryGuardConfig(
            enabled=True,
            min_target=0.35,
            max_base_target=0.25,
            min_mom_10=0.08,
            min_mom_21=0.12,
            min_relative_strength_21=1.0,
            min_vol_of_vol_21=0.10,
            max_vix_percentile_252=0.85,
            max_below_anchor_ratio_63=0.20,
            min_anchor_gap_180=0.50,
        ),
    )

    assert out["recovery_reentry_guard"].tolist() == [False, False, True]
    assert out["position"].round(4).tolist() == [0.0, 0.0, 0.35]


def test_equity_drawdown_control_caps_target_and_relaxes_sells_only():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    slow = pd.DataFrame({"datetime": dates, "slow_position": [0.70] * 4})
    fast = pd.DataFrame({"datetime": dates, "fast_position": [1.00] * 4})
    price = pd.Series([100.0, 80.0, 80.0, 80.0], index=dates, name="price")

    out = compose_budget_usage_policy_frame(
        slow,
        fast,
        TwoSpeedBudgetConfig(trade_threshold=0.0, trade_lock_days=5),
        price_series=price,
        drawdown_control=EquityDrawdownControlConfig(
            enabled=True,
            start_drawdown=0.10,
            full_drawdown=0.15,
            min_target=0.20,
            sell_trade_threshold=0.0,
            sell_trade_lock_days=0,
        ),
    )

    assert out["raw_two_speed_position"].round(4).tolist() == [0.95, 0.20, 0.20, 0.20]
    assert out["position"].round(4).tolist() == [0.95, 0.20, 0.20, 0.20]
    assert out["equity_drawdown_control"].tolist() == [False, True, True, True]
    assert out["equity_drawdown"].round(4).tolist() == [0.0, 0.19, 0.19, 0.19]
    assert out["effective_trade_lock_days"].tolist() == [5, 0, 5, 5]


def test_equity_drawdown_control_can_require_broken_anchor_and_high_volatility():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    slow = pd.DataFrame({"datetime": dates, "slow_position": [0.70] * 4})
    fast = pd.DataFrame({"datetime": dates, "fast_position": [1.00] * 4})
    price = pd.Series([100.0, 80.0, 80.0, 80.0], index=dates, name="price")
    features = pd.DataFrame(
        {
            "datetime": dates,
            "mom_10": [0.0, -0.05, -0.05, -0.05],
            "mom_21": [0.0, 0.18, -0.05, -0.05],
            "relative_strength_21": [0.0, 1.1, 0.4, 0.4],
            "vol_of_vol_21": [0.0, 0.12, 0.12, 0.12],
            "vix_percentile_252": [0.0, 0.70, 0.70, 0.70],
            "below_anchor_ratio_63": [0.0, 0.05, 0.75, 0.75],
            "anchor_gap_180": [0.0, 1.20, -0.20, -0.20],
        }
    )

    out = compose_budget_usage_policy_frame(
        slow,
        fast,
        TwoSpeedBudgetConfig(trade_threshold=0.0, trade_lock_days=5),
        price_series=price,
        feature_frame=features,
        drawdown_control=EquityDrawdownControlConfig(
            enabled=True,
            start_drawdown=0.10,
            full_drawdown=0.15,
            min_target=0.20,
            sell_trade_threshold=0.0,
            sell_trade_lock_days=0,
            max_anchor_gap_180=0.0,
            min_below_anchor_ratio_63=0.50,
            min_vix_percentile_252=0.60,
            max_mom_21=0.05,
        ),
    )

    assert out["equity_drawdown_control"].tolist() == [False, False, True, True]
    assert out["raw_two_speed_position"].round(4).tolist() == [0.95, 0.95, 0.20, 0.20]
    assert out["position"].round(4).tolist() == [0.95, 0.95, 0.20, 0.20]


if __name__ == "__main__":
    test_compose_two_speed_position_caps_fast_signal_by_mode_slack()
    test_compose_two_speed_position_keeps_fast_derisking()
    test_maybe_rebalance_position_filters_small_moves_and_honors_lock()
    test_infer_mode_from_slow_position_maps_defensive_bands()
    test_compose_budget_usage_position_maps_fast_signal_inside_mode_band()
    test_compose_budget_usage_position_preserves_full_derisking()
    test_compose_budget_usage_policy_frame_keeps_audit_columns()
    test_compose_budget_usage_policy_frame_applies_bull_participation_guard()
    test_recovery_reentry_guard_lifts_flat_actor_only_in_high_vol_rebound()
    test_recovery_reentry_guard_requires_long_anchor_confirmation_when_configured()
    test_equity_drawdown_control_caps_target_and_relaxes_sells_only()
    test_equity_drawdown_control_can_require_broken_anchor_and_high_volatility()
    print("test_two_speed_budget: OK")

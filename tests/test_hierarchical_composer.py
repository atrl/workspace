from __future__ import annotations

import numpy as np

from src.soxl_quant.config import ModeRouterConfig, load_profile
from src.soxl_quant.data import FEATURE_COLUMNS, STATIC_INDEX
from src.soxl_quant.policy.hierarchical_composer import (
    HierarchicalComposerConfig,
    HierarchicalComposerState,
    hierarchical_budget_target_position,
    next_hierarchical_state,
)


def _obs(**overrides: float) -> np.ndarray:
    obs = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)
    defaults = {
        "regime_risk": 0.35,
        "realized_vol_10": 0.35,
        "trend_gap_21": 0.04,
        "anchor_gap_180": 0.03,
        "cum_drop_30": -0.02,
        "relative_strength_21": 0.10,
        "vix_zscore": 0.0,
        "downside_ratio_10": 0.30,
        "anchor_slope_180": 0.02,
        "vol_coef_21": 0.40,
        "anchor_gap_velocity_10": 0.02,
        "trend_gap_21_velocity": 0.01,
        "regime_risk_velocity_5": -0.02,
        "drawdown_streak_63": 0.0,
        "below_anchor_ratio_63": 0.0,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        obs[STATIC_INDEX[key]] = value
    return obs


def test_next_hierarchical_state_keeps_mode_between_refreshes():
    router_cfg = ModeRouterConfig(enabled=True)
    composer_cfg = HierarchicalComposerConfig(slow_update_every=3, use_learned_router=False)

    state = HierarchicalComposerState()
    state = next_hierarchical_state(_obs(regime_risk=0.80), state, router_cfg, composer_cfg)
    assert state.active_mode == "flat_wait"

    hold_state = next_hierarchical_state(_obs(regime_risk=0.20, anchor_gap_180=0.05), state, router_cfg, composer_cfg)
    assert hold_state.active_mode == "flat_wait"
    assert hold_state.days_since_refresh == 1

    refresh_state = next_hierarchical_state(_obs(regime_risk=0.20, anchor_gap_180=0.05), hold_state, router_cfg, composer_cfg)
    assert refresh_state.active_mode == "trend_hold"
    assert refresh_state.days_since_refresh == 0


def test_hierarchical_budget_target_position_uses_active_mode_band():
    profile = load_profile("risk_budget_usage_v1")
    bull_obs = _obs()
    action = 10

    flat_wait_target = hierarchical_budget_target_position(
        action,
        bull_obs,
        active_mode="flat_wait",
        profile=profile,
    )
    trend_target = hierarchical_budget_target_position(
        action,
        bull_obs,
        active_mode="trend_hold",
        profile=profile,
    )

    assert round(flat_wait_target, 4) == 0.25
    assert round(trend_target, 4) == 0.95


if __name__ == "__main__":
    test_next_hierarchical_state_keeps_mode_between_refreshes()
    test_hierarchical_budget_target_position_uses_active_mode_band()
    print("test_hierarchical_composer: OK")

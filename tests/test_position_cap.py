from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.data import FEATURE_COLUMNS, STATIC_INDEX
from src.soxl_quant.policy.positioning import action_to_position, drawdown_position_cap, hybrid_overlay_position
from src.soxl_quant.settings import EXTRA_OBS_DIM, N_ACTIONS, PositionCapConfig


def make_obs(cum_drop_30: float, regime_risk: float = 0.2, anchor_gap_180: float = 0.03, anchor_slope_180: float = 0.01) -> np.ndarray:
    obs = np.zeros(len(FEATURE_COLUMNS) + EXTRA_OBS_DIM, dtype=np.float32)
    obs[STATIC_INDEX["cum_drop_30"]] = cum_drop_30
    obs[STATIC_INDEX["regime_risk"]] = regime_risk
    obs[STATIC_INDEX["anchor_gap_180"]] = anchor_gap_180
    obs[STATIC_INDEX["anchor_slope_180"]] = anchor_slope_180
    return obs


def test_drawdown_position_cap_thresholds():
    assert drawdown_position_cap(0.05) == 1.0
    assert drawdown_position_cap(0.10) == 1.0
    assert abs(drawdown_position_cap(0.20) - 0.70) < 1e-9
    assert drawdown_position_cap(0.30) == 0.40
    assert drawdown_position_cap(0.45) == 0.40


def test_action_to_position_applies_hard_cap_from_cumulative_drop():
    high_action = N_ACTIONS - 1
    calm_obs = make_obs(cum_drop_30=-0.05)
    stressed_obs = make_obs(cum_drop_30=-0.20)
    crash_obs = make_obs(cum_drop_30=-0.35)

    calm_pos = action_to_position(high_action, calm_obs)
    stressed_pos = action_to_position(high_action, stressed_obs)
    crash_pos = action_to_position(high_action, crash_obs)

    assert abs(calm_pos - 1.0) < 1e-6
    assert abs(stressed_pos - 0.70) < 1e-6
    assert abs(crash_pos - 0.40) < 1e-6
    assert calm_pos > stressed_pos > crash_pos


def test_drawdown_position_cap_respects_delayed_start_config():
    config = PositionCapConfig(start_drawdown=0.15, full_drawdown=0.30, min_cap=0.40)

    assert drawdown_position_cap(0.12, config=config) == 1.0
    assert abs(drawdown_position_cap(0.20, config=config) - 0.80) < 1e-9
    assert drawdown_position_cap(0.35, config=config) == 0.40


def test_drawdown_position_cap_supports_slower_non_linear_ramp():
    linear = PositionCapConfig(start_drawdown=0.10, full_drawdown=0.30, min_cap=0.40, curve_power=1.0)
    slower = PositionCapConfig(start_drawdown=0.10, full_drawdown=0.30, min_cap=0.40, curve_power=2.0)

    linear_cap = drawdown_position_cap(0.20, config=linear)
    slower_cap = drawdown_position_cap(0.20, config=slower)

    assert abs(linear_cap - 0.70) < 1e-9
    assert abs(slower_cap - 0.85) < 1e-9
    assert slower_cap > linear_cap


def test_hybrid_overlay_position_clips_delta_inside_current_band():
    obs = make_obs(cum_drop_30=-0.02, regime_risk=0.2, anchor_gap_180=0.06, anchor_slope_180=0.01)
    pos = hybrid_overlay_position(2, obs, delta_levels=np.asarray([-0.5, 0.0, 0.5], dtype=np.float32))
    assert 0.4 <= pos <= 1.0


if __name__ == "__main__":
    test_drawdown_position_cap_thresholds()
    test_action_to_position_applies_hard_cap_from_cumulative_drop()
    test_drawdown_position_cap_respects_delayed_start_config()
    test_drawdown_position_cap_supports_slower_non_linear_ramp()
    test_hybrid_overlay_position_clips_delta_inside_current_band()
    print("test_position_cap: OK")

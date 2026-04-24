from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..config import PositionBudgetConfig, PositionCapConfig, StrategyConfig
from ..data import STATIC_INDEX
from ..settings import DEFAULT_ACTION_LEVELS

BASE_ACTION_LEVELS = np.asarray(DEFAULT_ACTION_LEVELS, dtype=np.float32)
DEFAULT_POSITION_CAP = PositionCapConfig()
DEFAULT_POSITION_BUDGET = PositionBudgetConfig()


def action_levels_for_profile(profile: Optional[StrategyConfig]) -> np.ndarray:
    if profile is None:
        return BASE_ACTION_LEVELS
    return np.asarray(profile.model.action_levels, dtype=np.float32)


def rule_based_position(obs: np.ndarray) -> float:
    floor, cap = action_band_from_obs(obs)
    return (floor + cap) / 2.0


def _apply_mode_band(
    floor: float,
    cap: float,
    position_band: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    if position_band is None:
        return floor, cap
    band_floor, band_cap = float(position_band[0]), float(position_band[1])
    floor = max(0.0, band_floor)
    cap = min(cap, band_cap)
    if cap < floor:
        cap = floor
    return floor, cap


def hybrid_overlay_position(
    action: int,
    obs: np.ndarray,
    config: Optional[PositionCapConfig] = None,
    delta_levels: Optional[np.ndarray] = None,
    hard_floor: Optional[float] = None,
    mode_band: Optional[Tuple[float, float]] = None,
) -> float:
    levels = np.asarray(delta_levels if delta_levels is not None else (-0.15, 0.0, 0.15), dtype=np.float32)
    clipped = int(np.clip(action, 0, len(levels) - 1))
    delta = float(levels[clipped])
    floor, cap = action_band_from_obs(obs, hard_floor=hard_floor)
    floor, cap = _apply_mode_band(floor, cap, mode_band)
    cumulative_drop = max(-float(np.nan_to_num(obs[STATIC_INDEX["cum_drop_30"]])), 0.0)
    cap = min(cap, drawdown_position_cap(cumulative_drop, obs=obs, config=config))
    if mode_band is not None and cap < floor:
        cap = floor
    base = float(np.clip(rule_based_position(obs), floor, cap))
    return float(np.clip(base + delta, floor, cap))


def action_band_from_obs(obs: np.ndarray, hard_floor: Optional[float] = None) -> Tuple[float, float]:
    regime_risk = float(np.nan_to_num(obs[STATIC_INDEX["regime_risk"]]))
    anchor_gap = float(np.nan_to_num(obs[STATIC_INDEX["anchor_gap_180"]]))
    anchor_slope = float(np.nan_to_num(obs[STATIC_INDEX["anchor_slope_180"]]))
    cum_drop_30 = float(np.nan_to_num(obs[STATIC_INDEX["cum_drop_30"]]))
    vix_zscore = float(np.nan_to_num(obs[STATIC_INDEX["vix_zscore"]]))
    vol_coef = float(np.nan_to_num(obs[STATIC_INDEX["vol_coef_21"]]))
    anchor_gap_vel = float(np.nan_to_num(obs[STATIC_INDEX["anchor_gap_velocity_10"]]))
    trend_gap_vel = float(np.nan_to_num(obs[STATIC_INDEX["trend_gap_21_velocity"]]))
    regime_risk_vel = float(np.nan_to_num(obs[STATIC_INDEX["regime_risk_velocity_5"]]))

    floor = hard_floor if hard_floor is not None else 0.40
    cap = 0.95

    if anchor_gap > 0.02 and anchor_slope >= 0:
        floor = 0.50
        cap = 1.00
    if regime_risk > 0.68 or (anchor_gap < -0.04 and anchor_slope < 0):
        floor = 0.30
        cap = 0.70
    if cum_drop_30 < -0.18 and vix_zscore < 2.8:
        floor = max(floor, 0.35)
        cap = min(cap, 0.70)
    if vol_coef > 1.45:
        floor = max(0.30, floor - 0.05)
        cap = min(cap, 0.90)
    if anchor_gap_vel > 0.015 and cum_drop_30 < -0.08 and regime_risk_vel < 0.0:
        floor = max(floor, 0.40)
        cap = max(cap, 0.90)
    elif anchor_gap_vel < -0.01 and trend_gap_vel < -0.005 and cum_drop_30 < -0.05:
        floor = max(floor, 0.25)
        cap = min(cap, 0.65)
    elif regime_risk_vel > 0.08 and regime_risk > 0.50:
        floor = max(floor, 0.30)
        cap = min(cap, 0.70)

    drawdown_streak = float(np.nan_to_num(obs[STATIC_INDEX["drawdown_streak_63"]]))
    below_anchor_ratio = float(np.nan_to_num(obs[STATIC_INDEX["below_anchor_ratio_63"]]))
    if (
        drawdown_streak > 0.30
        and below_anchor_ratio > 0.50
        and anchor_gap < -0.02
        and regime_risk > 0.45
    ):
        floor = 0.25
        cap = min(cap, 0.35)

    return floor, cap


def drawdown_position_cap(
    cumulative_drop: float,
    obs: Optional[np.ndarray] = None,
    config: Optional[PositionCapConfig] = None,
) -> float:
    cfg = config or DEFAULT_POSITION_CAP
    effective_min_cap = float(cfg.min_cap)
    if obs is not None and len(obs) > 0:
        realized_vol = float(np.nan_to_num(obs[STATIC_INDEX["realized_vol_10"]]))
        anchor_gap = float(np.nan_to_num(obs[STATIC_INDEX["anchor_gap_180"]]))
        trend_gap = float(np.nan_to_num(obs[STATIC_INDEX["trend_gap_21"]]))
        regime_risk = float(np.nan_to_num(obs[STATIC_INDEX["regime_risk"]]))
        if (
            realized_vol < cfg.bull_vol_threshold
            and anchor_gap > cfg.bull_anchor_threshold
            and trend_gap > 0.0
        ):
            effective_min_cap = max(effective_min_cap, float(cfg.bull_min_cap))
        if regime_risk > cfg.bear_regime_threshold:
            effective_min_cap = min(effective_min_cap, float(cfg.bear_min_cap))

    drop = max(float(cumulative_drop), 0.0)
    if drop <= cfg.start_drawdown:
        return 1.0
    if drop >= cfg.full_drawdown:
        return float(effective_min_cap)
    span = max(cfg.full_drawdown - cfg.start_drawdown, 1e-6)
    scale = (drop - cfg.start_drawdown) / span
    curve_power = max(float(getattr(cfg, "curve_power", 1.0)), 1e-6)
    shaped_scale = scale**curve_power
    return float(1.0 - shaped_scale * (1.0 - effective_min_cap))


def max_position_for_budget(
    current_drawdown: float,
    config: Optional[PositionBudgetConfig] = None,
) -> float:
    cfg = config or DEFAULT_POSITION_BUDGET
    consumed = max(float(current_drawdown), 0.0) / max(float(cfg.budget), 1e-6)
    for start, end, max_pos in cfg.thresholds:
        if float(start) <= consumed < float(end):
            return float(max_pos)
    return float(cfg.thresholds[-1][2])


def apply_position_budget(
    target_position: float,
    obs: np.ndarray,
    current_drawdown: float,
    config: Optional[PositionBudgetConfig] = None,
) -> float:
    cfg = config or DEFAULT_POSITION_BUDGET
    if not cfg.enabled:
        return float(target_position)
    regime_risk = float(np.nan_to_num(obs[STATIC_INDEX["regime_risk"]]))
    if regime_risk < float(cfg.regime_threshold):
        return float(target_position)
    return float(min(target_position, max_position_for_budget(current_drawdown, cfg)))


def action_to_position(
    action: int,
    obs: np.ndarray,
    config: Optional[PositionCapConfig] = None,
    action_levels: Optional[np.ndarray] = None,
    hard_floor: Optional[float] = None,
    mode_band: Optional[Tuple[float, float]] = None,
) -> float:
    levels = action_levels if action_levels is not None else BASE_ACTION_LEVELS
    clipped = int(np.clip(action, 0, len(levels) - 1))
    base = float(levels[clipped])
    floor, cap = action_band_from_obs(obs, hard_floor=hard_floor)
    floor, cap = _apply_mode_band(floor, cap, mode_band)
    cumulative_drop = max(-float(np.nan_to_num(obs[STATIC_INDEX["cum_drop_30"]])), 0.0)
    cap = min(cap, drawdown_position_cap(cumulative_drop, obs=obs, config=config))
    if mode_band is not None and cap < floor:
        cap = floor
    return float(np.clip(floor + base * (cap - floor), 0.0, 1.0))


def action_to_budget_usage_position(
    action: int,
    obs: np.ndarray,
    config: Optional[PositionCapConfig] = None,
    action_levels: Optional[np.ndarray] = None,
    hard_floor: Optional[float] = None,
    mode_band: Optional[Tuple[float, float]] = None,
) -> float:
    levels = action_levels if action_levels is not None else BASE_ACTION_LEVELS
    clipped = int(np.clip(action, 0, len(levels) - 1))
    usage = float(np.clip(levels[clipped], 0.0, 1.0))
    floor, cap = action_band_from_obs(obs, hard_floor=hard_floor)
    floor, cap = _apply_mode_band(floor, cap, mode_band)
    cumulative_drop = max(-float(np.nan_to_num(obs[STATIC_INDEX["cum_drop_30"]])), 0.0)
    cap = min(cap, drawdown_position_cap(cumulative_drop, obs=obs, config=config))
    if mode_band is not None and cap < floor:
        cap = floor
    return float(np.clip(floor + usage * (cap - floor), 0.0, 1.0))


def budget_usage_to_position(
    usage: float,
    obs: np.ndarray,
    config: Optional[PositionCapConfig] = None,
    hard_floor: Optional[float] = None,
    mode_band: Optional[Tuple[float, float]] = None,
) -> float:
    clipped_usage = float(np.clip(usage, 0.0, 1.0))
    floor, cap = action_band_from_obs(obs, hard_floor=hard_floor)
    floor, cap = _apply_mode_band(floor, cap, mode_band)
    cumulative_drop = max(-float(np.nan_to_num(obs[STATIC_INDEX["cum_drop_30"]])), 0.0)
    cap = min(cap, drawdown_position_cap(cumulative_drop, obs=obs, config=config))
    if mode_band is not None and cap < floor:
        cap = floor
    return float(np.clip(floor + clipped_usage * (cap - floor), 0.0, 1.0))


def target_position_for_profile(
    action: int,
    obs: np.ndarray,
    config: Optional[PositionCapConfig] = None,
    action_levels: Optional[np.ndarray] = None,
    hard_floor: Optional[float] = None,
    use_hybrid_overlay: bool = False,
    delta_levels: Optional[np.ndarray] = None,
    mode_band: Optional[Tuple[float, float]] = None,
    position_budget_config: Optional[PositionBudgetConfig] = None,
    current_drawdown: float = 0.0,
    position_mapping: str = "default",
) -> float:
    if position_mapping == "budget_usage":
        if delta_levels is not None:
            action_levels = np.linspace(0.0, 1.0, len(delta_levels), dtype=np.float32)
        target = action_to_budget_usage_position(
            action,
            obs,
            config=config,
            action_levels=action_levels,
            hard_floor=hard_floor,
            mode_band=mode_band,
        )
    elif use_hybrid_overlay:
        target = hybrid_overlay_position(
            action,
            obs,
            config=config,
            delta_levels=delta_levels,
            hard_floor=hard_floor,
            mode_band=mode_band,
        )
    else:
        target = action_to_position(
            action,
            obs,
            config=config,
            action_levels=action_levels,
            hard_floor=hard_floor,
            mode_band=mode_band,
        )
    return apply_position_budget(target, obs, current_drawdown, position_budget_config)


__all__ = [
    "BASE_ACTION_LEVELS",
    "DEFAULT_POSITION_CAP",
    "DEFAULT_POSITION_BUDGET",
    "action_band_from_obs",
    "action_levels_for_profile",
    "budget_usage_to_position",
    "action_to_budget_usage_position",
    "action_to_position",
    "apply_position_budget",
    "drawdown_position_cap",
    "hybrid_overlay_position",
    "max_position_for_budget",
    "rule_based_position",
    "target_position_for_profile",
]

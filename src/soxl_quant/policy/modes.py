from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..config import ModeRouterConfig
from ..data import STATIC_INDEX


@dataclass(frozen=True)
class ModeState:
    name: str = "flat_wait"
    days_in_mode: int = 0


# Lazy singleton for learned router (loaded on first use)
_learned_router: Optional["LearnedModeRouter"] = None


def _get_learned_router(model_path: Optional[str]) -> "LearnedModeRouter":
    global _learned_router
    if _learned_router is None:
        from .models.mode_classifier import LearnedModeRouter
        if model_path is None or model_path == "":
            model_path = "runtime/model/card/mode_classifier/mode_classifier.pt"
        _learned_router = LearnedModeRouter(model_path)
    return _learned_router


def classify_rule_mode(obs: np.ndarray) -> str:
    regime_risk = float(np.nan_to_num(obs[STATIC_INDEX["regime_risk"]]))
    realized_vol = float(np.nan_to_num(obs[STATIC_INDEX["realized_vol_10"]]))
    trend_gap = float(np.nan_to_num(obs[STATIC_INDEX["trend_gap_21"]]))
    anchor_gap = float(np.nan_to_num(obs[STATIC_INDEX["anchor_gap_180"]]))
    cum_drop_30 = float(np.nan_to_num(obs[STATIC_INDEX["cum_drop_30"]]))
    relative_strength = float(np.nan_to_num(obs[STATIC_INDEX["relative_strength_21"]]))
    vix_zscore = float(np.nan_to_num(obs[STATIC_INDEX["vix_zscore"]]))
    downside_ratio = float(np.nan_to_num(obs[STATIC_INDEX["downside_ratio_10"]]))

    if regime_risk >= 0.72 or (
        regime_risk >= 0.60 and realized_vol >= 0.60 and anchor_gap <= -0.03 and trend_gap <= 0.0
    ):
        return "flat_wait"
    if (
        cum_drop_30 <= -0.12
        and regime_risk <= 0.58
        and trend_gap >= 0.02
        and relative_strength >= 0.08
        and vix_zscore <= 0.80
        and downside_ratio <= 0.55
    ):
        return "crash_rebuild"
    if (
        regime_risk <= 0.40
        and realized_vol <= 0.45
        and anchor_gap >= 0.02
        and trend_gap >= 0.02
    ):
        return "trend_hold"
    return "defensive_hold"


def classify_learned_mode(obs: np.ndarray, model_path: Optional[str] = None) -> str:
    """Classify mode using the learned classifier."""
    router = _get_learned_router(model_path)
    return router.classify(obs, STATIC_INDEX)


def _min_dwell_days(mode: str, cfg: ModeRouterConfig) -> int:
    if mode == "trend_hold":
        return int(cfg.trend_hold_min_dwell_days)
    if mode == "defensive_hold":
        return int(cfg.defensive_hold_min_dwell_days)
    if mode == "crash_rebuild":
        return int(cfg.crash_rebuild_min_dwell_days)
    return int(cfg.flat_wait_min_dwell_days)


def next_mode(obs: np.ndarray, state: Optional[ModeState], cfg: ModeRouterConfig) -> ModeState:
    if getattr(cfg, "use_learned", False):
        target = classify_learned_mode(obs, getattr(cfg, "learned_model_path", None))
    else:
        target = classify_rule_mode(obs)
    if state is None:
        return ModeState(name=target, days_in_mode=1)
    if state.name == target:
        return ModeState(name=state.name, days_in_mode=state.days_in_mode + 1)
    if state.days_in_mode < _min_dwell_days(state.name, cfg):
        return ModeState(name=state.name, days_in_mode=state.days_in_mode + 1)
    return ModeState(name=target, days_in_mode=1)


def route_mode(obs: np.ndarray, state: Optional[ModeState], cfg: ModeRouterConfig) -> ModeState:
    if not cfg.enabled:
        return ModeState(name="flat_wait", days_in_mode=1)
    return next_mode(obs, state, cfg)


def mode_band(mode: str, cfg: ModeRouterConfig) -> Tuple[float, float]:
    if mode == "trend_hold":
        return (float(cfg.trend_hold_floor), float(cfg.trend_hold_cap))
    if mode == "defensive_hold":
        return (float(cfg.defensive_hold_floor), float(cfg.defensive_hold_cap))
    if mode == "crash_rebuild":
        return (float(cfg.crash_rebuild_floor), float(cfg.crash_rebuild_cap))
    return (float(cfg.flat_wait_floor), float(cfg.flat_wait_cap))


def mode_threshold_multiplier(mode: Optional[str], direction: str, cfg: ModeRouterConfig) -> float:
    if mode == "trend_hold":
        return (
            float(cfg.trend_hold_buy_threshold_multiplier)
            if direction == "buy"
            else float(cfg.trend_hold_sell_threshold_multiplier)
        )
    if mode == "defensive_hold":
        return (
            float(cfg.defensive_hold_buy_threshold_multiplier)
            if direction == "buy"
            else float(cfg.defensive_hold_sell_threshold_multiplier)
        )
    if mode == "crash_rebuild":
        return (
            float(cfg.crash_rebuild_buy_threshold_multiplier)
            if direction == "buy"
            else float(cfg.crash_rebuild_sell_threshold_multiplier)
        )
    if mode == "flat_wait":
        return (
            float(cfg.flat_wait_buy_threshold_multiplier)
            if direction == "buy"
            else float(cfg.flat_wait_sell_threshold_multiplier)
        )
    return 1.0


__all__ = [
    "ModeState",
    "classify_rule_mode",
    "mode_band",
    "mode_threshold_multiplier",
    "next_mode",
    "route_mode",
]

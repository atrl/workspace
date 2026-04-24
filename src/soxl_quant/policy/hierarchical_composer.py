from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ..config import ModeRouterConfig, StrategyConfig
from .modes import ModeState, classify_learned_mode, classify_rule_mode, mode_band
from .positioning import target_position_for_profile


@dataclass(frozen=True)
class HierarchicalComposerConfig:
    slow_update_every: int = 5
    use_learned_router: bool = False


@dataclass(frozen=True)
class HierarchicalComposerState:
    active_mode: str = "flat_wait"
    router_state: Optional[ModeState] = None
    days_since_refresh: int = 0


def _router_config(cfg: ModeRouterConfig, composer_cfg: HierarchicalComposerConfig) -> ModeRouterConfig:
    return replace(cfg, use_learned=bool(composer_cfg.use_learned_router))


def _classify_mode(obs: np.ndarray, cfg: ModeRouterConfig, composer_cfg: HierarchicalComposerConfig) -> str:
    router_cfg = _router_config(cfg, composer_cfg)
    if getattr(router_cfg, "use_learned", False):
        return classify_learned_mode(obs, getattr(router_cfg, "learned_model_path", None))
    return classify_rule_mode(obs)


def next_hierarchical_state(
    obs: np.ndarray,
    state: Optional[HierarchicalComposerState],
    router_cfg: ModeRouterConfig,
    composer_cfg: HierarchicalComposerConfig,
) -> HierarchicalComposerState:
    refresh_every = max(int(composer_cfg.slow_update_every), 1)
    current = state or HierarchicalComposerState()
    should_refresh = current.router_state is None or current.days_since_refresh + 2 >= refresh_every
    if should_refresh:
        active_mode = _classify_mode(obs, router_cfg, composer_cfg)
        return HierarchicalComposerState(
            active_mode=active_mode,
            router_state=ModeState(name=active_mode, days_in_mode=1),
            days_since_refresh=0,
        )

    router_state = current.router_state or ModeState(name=current.active_mode, days_in_mode=1)
    return HierarchicalComposerState(
        active_mode=current.active_mode,
        router_state=ModeState(name=router_state.name, days_in_mode=router_state.days_in_mode + 1),
        days_since_refresh=current.days_since_refresh + 1,
    )


def _scale_band_by_vol(
    band: tuple[float, float],
    obs: np.ndarray,
    router_cfg: ModeRouterConfig,
) -> tuple[float, float]:
    """Scale mode band width based on realized volatility.

    High vol -> compress band (reduce risk exposure).
    Low vol -> expand band (increase participation).
    """
    if not getattr(router_cfg, "use_dynamic_bands", False):
        return band
    from ..data import STATIC_INDEX
    realized_vol = float(np.nan_to_num(obs[STATIC_INDEX["realized_vol_10"]]))
    baseline = float(router_cfg.dynamic_band_vol_baseline)
    sensitivity = float(router_cfg.dynamic_band_vol_sensitivity)
    min_scale = float(router_cfg.dynamic_band_min_scale)
    max_scale = float(router_cfg.dynamic_band_max_scale)
    scale = 1.0 - (realized_vol - baseline) * sensitivity
    scale = max(min_scale, min(max_scale, scale))
    floor, cap = float(band[0]), float(band[1])
    width = cap - floor
    centre = (floor + cap) / 2.0
    new_width = width * scale
    new_floor = max(0.0, centre - new_width / 2.0)
    new_cap = min(1.0, centre + new_width / 2.0)
    return (new_floor, new_cap)


def hierarchical_budget_target_position(
    action: int,
    obs: np.ndarray,
    active_mode: str,
    profile: StrategyConfig,
    current_drawdown: float = 0.0,
) -> float:
    router_cfg = profile.mode_router
    if router_cfg and getattr(router_cfg, "enabled", False):
        active_band = mode_band(active_mode, router_cfg)
        active_band = _scale_band_by_vol(active_band, obs, router_cfg)
    else:
        active_band = None
    return target_position_for_profile(
        action,
        obs,
        config=profile.position_cap,
        action_levels=np.asarray(profile.model.action_levels, dtype=np.float32),
        hard_floor=profile.model.hard_floor,
        use_hybrid_overlay=profile.model.use_hybrid_overlay,
        delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.delta_levels else None,
        mode_band=active_band,
        position_budget_config=profile.position_budget,
        current_drawdown=current_drawdown,
        position_mapping=profile.model.position_mapping,
    )


__all__ = [
    "HierarchicalComposerConfig",
    "HierarchicalComposerState",
    "hierarchical_budget_target_position",
    "next_hierarchical_state",
]

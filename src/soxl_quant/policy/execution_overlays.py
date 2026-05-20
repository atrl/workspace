from __future__ import annotations

from copy import deepcopy
from typing import Any


POST_RELEASE_CAP_VARIANT = "selective_release_recovery_post_release_cap_v1"


SELECTIVE_RELEASE_OVERLAY_CONFIGS: dict[str, dict[str, Any] | None] = {
    "baseline": None,
    "selective_release_cicc_v1": {
        "type": "selective_release",
        "boost": 0.05,
        "max_target": 0.62,
        "max_regime_risk": 0.62,
        "cooldown_threshold": 0.25,
        "vix_threshold": 0.04,
        "volume_sync_threshold": 0.68,
    },
    "selective_release_tight_v1": {
        "type": "selective_release",
        "boost": 0.05,
        "max_target": 0.60,
        "max_regime_risk": 0.58,
        "cooldown_threshold": 0.30,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.72,
    },
    "selective_release_recovery_v1": {
        "type": "selective_release",
        "boost": 0.08,
        "max_target": 0.65,
        "max_regime_risk": 0.60,
        "cooldown_threshold": 0.20,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.70,
    },
    "selective_release_recovery_threshold_bypass_v1": {
        "type": "selective_release",
        "boost": 0.08,
        "max_target": 0.65,
        "max_regime_risk": 0.60,
        "cooldown_threshold": 0.20,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.70,
        "rebalance_threshold_override": 0.0,
    },
    "selective_release_recovery_target_lift_v1": {
        "type": "selective_release",
        "boost": 0.08,
        "max_target": 0.65,
        "max_regime_risk": 0.60,
        "cooldown_threshold": 0.20,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.70,
        "min_target_delta": 0.08,
    },
    "selective_release_recovery_trade_lock_bypass_v1": {
        "type": "selective_release",
        "boost": 0.08,
        "max_target": 0.65,
        "max_regime_risk": 0.60,
        "cooldown_threshold": 0.20,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.70,
        "rebalance_threshold_override": 0.0,
        "trade_lock_override": 0,
    },
    "selective_release_recovery_trade_lock_short_v1": {
        "type": "selective_release",
        "boost": 0.08,
        "max_target": 0.65,
        "max_regime_risk": 0.60,
        "cooldown_threshold": 0.20,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.70,
        "rebalance_threshold_override": 0.0,
        "trade_lock_cap": 1,
    },
    "selective_release_recovery_trade_lock_guarded_short_v1": {
        "type": "selective_release",
        "boost": 0.08,
        "max_target": 0.65,
        "max_regime_risk": 0.60,
        "cooldown_threshold": 0.20,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.70,
        "rebalance_threshold_override": 0.0,
        "trade_lock_cap": 1,
        "max_current_drawdown": 0.03,
        "min_release_exposure_gap": 0.10,
    },
    POST_RELEASE_CAP_VARIANT: {
        "type": "selective_release",
        "boost": 0.08,
        "max_target": 0.65,
        "max_regime_risk": 0.60,
        "cooldown_threshold": 0.20,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.70,
        "rebalance_threshold_override": 0.0,
        "trade_lock_cap": 1,
        "max_current_drawdown": 0.03,
        "min_release_exposure_gap": 0.10,
        "post_release_state_days": 90,
        "post_release_drawdown_threshold": 0.04,
        "post_release_target_cap": 0.55,
        "post_release_rebalance_threshold_override": 0.0,
        "post_release_trade_lock_override": 0,
    },
}


def overlay_config(name: str) -> dict[str, Any]:
    if name == "baseline":
        raise ValueError("baseline has no overlay config")
    if name not in SELECTIVE_RELEASE_OVERLAY_CONFIGS:
        raise KeyError(f"unknown execution overlay '{name}'")
    config = SELECTIVE_RELEASE_OVERLAY_CONFIGS[name]
    if config is None:
        raise ValueError(f"execution overlay '{name}' has no config")
    return deepcopy(config)


def overlay_variants() -> list[tuple[str, dict[str, Any] | None]]:
    variants: list[tuple[str, dict[str, Any] | None]] = []
    for name, config in SELECTIVE_RELEASE_OVERLAY_CONFIGS.items():
        variants.append((name, deepcopy(config) if config is not None else None))
    return variants

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


POST_RELEASE_CAP_VARIANT = "selective_release_recovery_post_release_cap_v1"
SOXX_RELATIVE_WEAKNESS_CAP_VARIANT = "soxx_relative_weakness_cap_v1"


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

SOXX_RELATIVE_OVERLAY_CONFIGS: dict[str, dict[str, Any] | None] = {
    "baseline": None,
    SOXX_RELATIVE_WEAKNESS_CAP_VARIANT: {
        "type": "soxx_relative_weakness_cap",
        "min_regime_risk": 0.45,
        "max_regime_risk": 0.90,
        "soxx_bias_max": -0.02,
        "price_ratio_mom_max": 0.0,
        "signed_te_reversion_max": 0.50,
        "target_cap": 0.35,
    },
}


def _row_float(row: Any, key: str) -> float | None:
    if not hasattr(row, "get"):
        return None
    value = row.get(key, None)
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_float):
        return None
    return value_float


def soxx_relative_weakness_ready(row: Any, regime_risk: float, overlay_config: dict[str, Any]) -> bool:
    """Detect SOXX-led weakness where a fixed actor should be capped, not boosted."""
    min_regime = float(overlay_config.get("min_regime_risk", float("-inf")))
    max_regime = float(overlay_config.get("max_regime_risk", float("inf")))
    if not (min_regime <= float(regime_risk) <= max_regime):
        return False

    soxx_bias = _row_float(row, "SOXX_BIAS_10")
    price_ratio_mom = _row_float(row, "PRICE_RATIO_MOM")
    signed_te_reversion = _row_float(row, "SIGNED_TE_REVERSION_10")
    if soxx_bias is None or price_ratio_mom is None or signed_te_reversion is None:
        return False

    if soxx_bias > float(overlay_config.get("soxx_bias_max", -0.02)):
        return False
    if price_ratio_mom > float(overlay_config.get("price_ratio_mom_max", 0.0)):
        return False
    if signed_te_reversion >= float(overlay_config.get("signed_te_reversion_max", 0.50)):
        return False
    return True


def apply_soxx_relative_weakness_cap(
    target_pos: float,
    row: Any,
    regime_risk: float,
    overlay_config: dict[str, Any],
) -> float:
    if not soxx_relative_weakness_ready(row, regime_risk, overlay_config):
        return float(target_pos)
    return min(float(target_pos), float(overlay_config.get("target_cap", 0.35)))


def _config_float(config: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in config:
            continue
        try:
            value = float(config[key])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    return None


def sequence_risk_action_cap_ready(row: Any, overlay_config: dict[str, Any]) -> bool:
    """Detect a fold-local sequence-risk state for a default-off action cap."""

    slope_threshold = _config_float(overlay_config, "sequence_risk_anchor_slope_max", "anchor_slope_max")
    gap_threshold = _config_float(overlay_config, "sequence_risk_anchor_gap_max", "anchor_gap_max")
    trend_threshold = _config_float(overlay_config, "sequence_risk_trend_gap_min", "trend_gap_min")
    if slope_threshold is None or gap_threshold is None or trend_threshold is None:
        return False

    slope = _row_float(row, "anchor_slope_180")
    anchor_gap = _row_float(row, "anchor_gap_180")
    trend_gap = _row_float(row, "trend_gap_63")
    if slope is None or anchor_gap is None or trend_gap is None:
        return False

    hit_count = int(slope <= slope_threshold)
    hit_count += int(anchor_gap <= gap_threshold)
    hit_count += int(trend_gap >= trend_threshold)
    return hit_count >= int(overlay_config.get("sequence_risk_min_feature_hits", 2))


def apply_sequence_risk_action_cap(
    target_pos: float,
    row: Any,
    overlay_config: dict[str, Any],
) -> float:
    if not sequence_risk_action_cap_ready(row, overlay_config):
        return float(target_pos)
    return min(float(target_pos), float(overlay_config.get("target_cap", 0.10)))


def short_volshock_action_cap_ready(row: Any, overlay_config: dict[str, Any]) -> bool:
    """Detect a fold-local short-volatility shock state for an action cap."""

    realized_threshold = _config_float(overlay_config, "realized_vol_21_min", "realized_vol_21_q80")
    vol_of_vol_threshold = _config_float(overlay_config, "vol_of_vol_21_min", "vol_of_vol_21_q80")
    downside_threshold = _config_float(overlay_config, "downside_semivol_21_min", "downside_semivol_21_q80")
    if realized_threshold is None or vol_of_vol_threshold is None or downside_threshold is None:
        return False

    realized_vol = _row_float(row, "realized_vol_21")
    vol_of_vol = _row_float(row, "vol_of_vol_21")
    downside_semivol = _row_float(row, "downside_semivol_21")
    if realized_vol is None or vol_of_vol is None or downside_semivol is None:
        return False

    hit_count = int(realized_vol >= realized_threshold)
    hit_count += int(vol_of_vol >= vol_of_vol_threshold)
    hit_count += int(downside_semivol >= downside_threshold)
    return hit_count >= int(overlay_config.get("short_volshock_min_feature_hits", 1))


def apply_short_volshock_action_cap(
    target_pos: float,
    row: Any,
    overlay_config: dict[str, Any],
) -> float:
    if not short_volshock_action_cap_ready(row, overlay_config):
        return float(target_pos)
    return min(float(target_pos), float(overlay_config.get("target_cap", 0.05)))


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


def soxx_overlay_config(name: str) -> dict[str, Any]:
    if name == "baseline":
        raise ValueError("baseline has no overlay config")
    if name not in SOXX_RELATIVE_OVERLAY_CONFIGS:
        raise KeyError(f"unknown SOXX relative execution overlay '{name}'")
    config = SOXX_RELATIVE_OVERLAY_CONFIGS[name]
    if config is None:
        raise ValueError(f"SOXX relative execution overlay '{name}' has no config")
    return deepcopy(config)


def soxx_overlay_variants() -> list[tuple[str, dict[str, Any] | None]]:
    variants: list[tuple[str, dict[str, Any] | None]] = []
    for name, config in SOXX_RELATIVE_OVERLAY_CONFIGS.items():
        variants.append((name, deepcopy(config) if config is not None else None))
    return variants

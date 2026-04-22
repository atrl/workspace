from __future__ import annotations

from typing import Optional

from ..config import ExecutionConfig, ExecutionRegimeConfig, ModeRouterConfig
from .modes import mode_threshold_multiplier

DEFAULT_EXECUTION = ExecutionConfig()
DEFAULT_MODE_ROUTER = ModeRouterConfig()


def _regime_context(
    regime_risk: float,
    realized_vol: float,
    trend_strength: float,
    cum_drop_30: float,
    relative_strength: float,
    vix_zscore: float,
    downside_ratio_10: float,
) -> dict[str, float]:
    return {
        "regime_risk": float(regime_risk),
        "realized_vol": float(realized_vol),
        "trend_strength": float(trend_strength),
        "cum_drop_30": float(cum_drop_30),
        "relative_strength": float(relative_strength),
        "vix_zscore": float(vix_zscore),
        "downside_ratio_10": float(downside_ratio_10),
    }


def _matches_custom_regime(rule: ExecutionRegimeConfig, context: dict[str, float]) -> bool:
    for key, expected in rule.conditions.items():
        if key.endswith("_min"):
            metric = key[:-4]
            if context.get(metric, 0.0) < expected:
                return False
            continue
        if key.endswith("_max"):
            metric = key[:-4]
            if context.get(metric, 0.0) > expected:
                return False
            continue
        if context.get(key, 0.0) != expected:
            return False
    return True


def execution_regime(
    regime_risk: float,
    realized_vol: float,
    trend_strength: float,
    cum_drop_30: float = 0.0,
    relative_strength: float = 0.0,
    vix_zscore: float = 0.0,
    downside_ratio_10: float = 0.0,
    config: Optional[ExecutionConfig] = None,
) -> str:
    cfg = config or DEFAULT_EXECUTION
    context = _regime_context(
        regime_risk=regime_risk,
        realized_vol=realized_vol,
        trend_strength=trend_strength,
        cum_drop_30=cum_drop_30,
        relative_strength=relative_strength,
        vix_zscore=vix_zscore,
        downside_ratio_10=downside_ratio_10,
    )
    if (
        regime_risk > cfg.stress_regime_threshold
        or realized_vol > cfg.stress_vol_threshold
        or trend_strength > cfg.stress_trend_threshold
    ):
        return "stress"
    if (
        regime_risk < cfg.quiet_regime_threshold
        and realized_vol < cfg.quiet_vol_threshold
        and trend_strength > cfg.bull_trend_threshold
    ):
        return "bull"
    for custom_regime in cfg.custom_regimes:
        if _matches_custom_regime(custom_regime, context):
            return custom_regime.name
    return "base"


def _custom_regime(cfg: ExecutionConfig, regime: str) -> Optional[ExecutionRegimeConfig]:
    for item in cfg.custom_regimes:
        if item.name == regime:
            return item
    return None


def execution_trade_lock_days(config: Optional[ExecutionConfig], regime: str) -> int:
    cfg = config or DEFAULT_EXECUTION
    if regime == "bull" and cfg.bull_trade_lock_days >= 0:
        return int(cfg.bull_trade_lock_days)
    if regime == "stress" and cfg.stress_trade_lock_days >= 0:
        return int(cfg.stress_trade_lock_days)
    custom = _custom_regime(cfg, regime)
    if custom is not None and custom.trade_lock_days >= 0:
        return int(custom.trade_lock_days)
    return int(cfg.trade_lock_days)


def execution_buy_cooldown_days(
    cum_drop_30: float,
    config: Optional[ExecutionConfig],
    regime: str,
) -> int:
    cfg = config or DEFAULT_EXECUTION
    if cum_drop_30 < cfg.deep_drop_threshold:
        return min(
            cfg.deep_drop_cooldown_max,
            max(cfg.deep_drop_cooldown_min, int(abs(cum_drop_30) * cfg.deep_drop_cooldown_scale)),
        )
    if regime == "bull" and cfg.bull_buy_cooldown_days >= 0:
        return int(cfg.bull_buy_cooldown_days)
    if regime == "stress" and cfg.stress_buy_cooldown_days >= 0:
        return int(cfg.stress_buy_cooldown_days)
    custom = _custom_regime(cfg, regime)
    if custom is not None and custom.buy_cooldown_days >= 0:
        return int(custom.buy_cooldown_days)
    return int(cfg.buy_cooldown_days)


def rebalance_threshold(
    regime_risk: float,
    realized_vol: float,
    trend_strength: float,
    cum_drop_30: float = 0.0,
    relative_strength: float = 0.0,
    vix_zscore: float = 0.0,
    downside_ratio_10: float = 0.0,
    direction: str = "neutral",
    mode: str | None = None,
    config: Optional[ExecutionConfig] = None,
    mode_router_config: Optional[ModeRouterConfig] = None,
) -> float:
    cfg = config or DEFAULT_EXECUTION
    mode_cfg = mode_router_config or DEFAULT_MODE_ROUTER
    regime = execution_regime(
        regime_risk,
        realized_vol,
        trend_strength,
        cum_drop_30=cum_drop_30,
        relative_strength=relative_strength,
        vix_zscore=vix_zscore,
        downside_ratio_10=downside_ratio_10,
        config=cfg,
    )
    threshold = float(cfg.base_threshold)
    if (
        regime_risk < cfg.quiet_regime_threshold
        and realized_vol < cfg.quiet_vol_threshold
        and trend_strength < cfg.quiet_trend_threshold
    ):
        threshold = float(cfg.quiet_threshold)
    if (
        regime_risk > cfg.stress_regime_threshold
        or realized_vol > cfg.stress_vol_threshold
        or trend_strength > cfg.stress_trend_threshold
    ):
        threshold = float(cfg.stress_threshold)
    if direction == "buy":
        threshold *= float(cfg.buy_threshold_multiplier)
    elif direction == "sell":
        threshold *= float(cfg.sell_threshold_multiplier)
    if regime == "bull":
        if direction == "buy":
            threshold *= float(cfg.bull_buy_threshold_multiplier)
        elif direction == "sell":
            threshold *= float(cfg.bull_sell_threshold_multiplier)
    elif regime == "stress":
        if direction == "buy":
            threshold *= float(cfg.stress_buy_threshold_multiplier)
        elif direction == "sell":
            threshold *= float(cfg.stress_sell_threshold_multiplier)
    else:
        custom = _custom_regime(cfg, regime)
        if custom is not None:
            if direction == "buy":
                threshold *= float(custom.buy_threshold_multiplier)
            elif direction == "sell":
                threshold *= float(custom.sell_threshold_multiplier)
    threshold *= mode_threshold_multiplier(mode, direction, mode_cfg)
    return threshold


__all__ = [
    "DEFAULT_EXECUTION",
    "execution_buy_cooldown_days",
    "execution_regime",
    "execution_trade_lock_days",
    "rebalance_threshold",
]

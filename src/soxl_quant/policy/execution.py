from __future__ import annotations

from typing import Optional

from ..config import ExecutionConfig

DEFAULT_EXECUTION = ExecutionConfig()


def execution_regime(
    regime_risk: float,
    realized_vol: float,
    trend_strength: float,
    config: Optional[ExecutionConfig] = None,
) -> str:
    cfg = config or DEFAULT_EXECUTION
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
    return "base"


def execution_trade_lock_days(config: Optional[ExecutionConfig], regime: str) -> int:
    cfg = config or DEFAULT_EXECUTION
    if regime == "bull" and cfg.bull_trade_lock_days >= 0:
        return int(cfg.bull_trade_lock_days)
    if regime == "stress" and cfg.stress_trade_lock_days >= 0:
        return int(cfg.stress_trade_lock_days)
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
    return int(cfg.buy_cooldown_days)


def rebalance_threshold(
    regime_risk: float,
    realized_vol: float,
    trend_strength: float,
    direction: str = "neutral",
    config: Optional[ExecutionConfig] = None,
) -> float:
    cfg = config or DEFAULT_EXECUTION
    regime = execution_regime(regime_risk, realized_vol, trend_strength, config=cfg)
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
    return threshold


__all__ = [
    "DEFAULT_EXECUTION",
    "execution_buy_cooldown_days",
    "execution_regime",
    "execution_trade_lock_days",
    "rebalance_threshold",
]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


DEFAULT_MODE_SLACK = {
    "trend_hold": 0.35,
    "crash_rebuild": 0.25,
    "defensive_hold": 0.10,
    "flat_wait": 0.05,
}


@dataclass(frozen=True)
class TwoSpeedBudgetConfig:
    """Fixed-signal two-speed composition settings.

    The slow layer defines a risk anchor. The fast layer may participate above
    that anchor only by the configured per-mode slack.
    """

    mode_slack: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_MODE_SLACK))
    trade_threshold: float = 0.04
    trade_lock_days: int = 5


def infer_mode_from_slow_position(position: float) -> str:
    """Infer a coarse mode from the slow layer when explicit mode rows are absent."""

    pos = max(0.0, min(float(position), 1.0))
    if pos <= 0.25:
        return "flat_wait"
    if pos <= 0.40:
        return "defensive_hold"
    if pos <= 0.60:
        return "crash_rebuild"
    return "trend_hold"


def compose_two_speed_position(
    slow_position: float,
    fast_position: float,
    mode: str,
    config: TwoSpeedBudgetConfig | None = None,
) -> float:
    """Combine slow risk budget and fast participation signal into a target position."""

    cfg = config or TwoSpeedBudgetConfig()
    slow = max(0.0, min(float(slow_position), 1.0))
    fast = max(0.0, min(float(fast_position), 1.0))
    slack = max(float(cfg.mode_slack.get(mode, 0.0)), 0.0)
    return round(max(0.0, min(fast, slow + slack, 1.0)), 10)


def maybe_rebalance_position(
    current_position: float,
    target_position: float,
    days_since_trade: int,
    config: TwoSpeedBudgetConfig | None = None,
) -> float:
    """Suppress low-value micro moves and changes during the trade lock."""

    cfg = config or TwoSpeedBudgetConfig()
    current = max(0.0, min(float(current_position), 1.0))
    target = max(0.0, min(float(target_position), 1.0))
    if abs(target - current) < float(cfg.trade_threshold):
        return current
    if int(days_since_trade) < int(cfg.trade_lock_days):
        return current
    return target


__all__ = [
    "DEFAULT_MODE_SLACK",
    "TwoSpeedBudgetConfig",
    "compose_two_speed_position",
    "infer_mode_from_slow_position",
    "maybe_rebalance_position",
]

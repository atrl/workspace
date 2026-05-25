from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownBudgetOverlayConfig:
    """Execution-layer cap for a fixed daily target-position path."""

    enabled: bool = True
    start_drawdown: float = 0.10
    full_drawdown: float = 0.20
    min_position: float = 0.25
    trade_threshold: float = 0.0
    trade_lock_days: int = 0
    sell_trade_threshold: float | None = 0.0
    sell_trade_lock_days: int | None = 0

    def __post_init__(self) -> None:
        if float(self.start_drawdown) < 0.0:
            raise ValueError("start_drawdown must be non-negative")
        if float(self.full_drawdown) < float(self.start_drawdown):
            raise ValueError("full_drawdown must be >= start_drawdown")
        if not 0.0 <= float(self.min_position) <= 1.0:
            raise ValueError("min_position must be within [0, 1]")
        if float(self.trade_threshold) < 0.0:
            raise ValueError("trade_threshold must be non-negative")
        if int(self.trade_lock_days) < 0:
            raise ValueError("trade_lock_days must be non-negative")


def _target_cap(drawdown: float, config: DrawdownBudgetOverlayConfig) -> float:
    if not config.enabled:
        return 1.0
    dd = max(float(drawdown), 0.0)
    start = max(float(config.start_drawdown), 0.0)
    full = max(float(config.full_drawdown), start)
    min_position = max(0.0, min(float(config.min_position), 1.0))
    if dd < start:
        return 1.0
    if full <= start or dd >= full:
        return min_position
    usage = (dd - start) / max(full - start, 1e-12)
    return max(min_position, 1.0 - usage * (1.0 - min_position))


def _normalise_position_frame(frame: pd.DataFrame, position_col: str) -> pd.DataFrame:
    if list(frame.columns).count("datetime") != 1:
        raise ValueError("frame must contain exactly one `datetime` column")
    position_count = list(frame.columns).count(position_col)
    if position_count == 0:
        raise ValueError(f"frame missing `{position_col}` column")
    if position_count > 1:
        raise ValueError(f"frame has duplicate `{position_col}` columns")
    out = frame[["datetime", position_col]].rename(columns={position_col: "base_position"}).copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["base_position"] = pd.to_numeric(out["base_position"], errors="coerce").clip(0.0, 1.0)
    return out.dropna(subset=["datetime", "base_position"]).sort_values("datetime").reset_index(drop=True)


def _price_frame(price_series: pd.Series) -> pd.DataFrame:
    price = price_series.copy()
    if not isinstance(price.index, pd.DatetimeIndex):
        price.index = pd.to_datetime(price.index)
    out = price.rename("price").reset_index()
    out.columns = ["datetime", "price"]
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    return out.dropna(subset=["datetime", "price"]).sort_values("datetime")


def _rebalance_position(
    current_position: float,
    target_position: float,
    days_since_trade: int,
    *,
    threshold: float,
    lock_days: int,
) -> float:
    current = max(0.0, min(float(current_position), 1.0))
    target = max(0.0, min(float(target_position), 1.0))
    if abs(target - current) < float(threshold):
        return current
    if int(days_since_trade) < int(lock_days):
        return current
    return target


def apply_drawdown_budget_overlay(
    frame: pd.DataFrame,
    price_series: pd.Series,
    config: DrawdownBudgetOverlayConfig | None = None,
    *,
    position_col: str = "position",
) -> pd.DataFrame:
    """Apply a trailing equity-drawdown budget to a fixed target path.

    The overlay updates account equity using the position that was active before
    each close, then uses that realized drawdown to cap the target for the next
    holding interval. This keeps the counterfactual auditable and avoids using
    future returns when choosing today's output position.
    """

    cfg = config or DrawdownBudgetOverlayConfig()
    positions = _normalise_position_frame(frame, position_col)
    scored = positions.merge(_price_frame(price_series), on="datetime", how="inner")
    if len(scored) < 2:
        raise ValueError("not enough overlapping position and price rows")
    scored = scored.sort_values("datetime").reset_index(drop=True)

    current = float(scored["base_position"].iloc[0])
    days_since_trade = int(cfg.trade_lock_days)
    nav = 1.0
    peak_nav = 1.0
    previous_price: float | None = None
    output_positions: list[float] = []
    raw_positions: list[float] = []
    navs: list[float] = []
    drawdowns: list[float] = []
    caps: list[float] = []
    active_flags: list[bool] = []
    effective_thresholds: list[float] = []
    effective_locks: list[int] = []

    for _, row in scored.iterrows():
        price = float(row["price"])
        if previous_price is not None and previous_price > 0:
            nav *= 1.0 + current * (price / previous_price - 1.0)
        peak_nav = max(peak_nav, nav)
        drawdown = max(1.0 - nav / max(peak_nav, 1e-12), 0.0)
        cap = _target_cap(drawdown, cfg)
        raw_target = min(float(row["base_position"]), cap)
        active = bool(cfg.enabled and cap < 1.0)

        threshold = float(cfg.trade_threshold)
        lock_days = int(cfg.trade_lock_days)
        if active and raw_target < current:
            if cfg.sell_trade_threshold is not None:
                threshold = float(cfg.sell_trade_threshold)
            if cfg.sell_trade_lock_days is not None:
                lock_days = max(int(cfg.sell_trade_lock_days), 0)
        target = _rebalance_position(
            current,
            raw_target,
            days_since_trade,
            threshold=threshold,
            lock_days=lock_days,
        )
        if abs(target - current) >= threshold:
            days_since_trade = 0
        else:
            days_since_trade += 1

        output_positions.append(target)
        raw_positions.append(raw_target)
        navs.append(nav)
        drawdowns.append(drawdown)
        caps.append(cap)
        active_flags.append(active)
        effective_thresholds.append(threshold)
        effective_locks.append(lock_days)
        current = target
        previous_price = price

    out = scored.drop(columns=["price"]).copy()
    out["raw_position"] = raw_positions
    out["position"] = np.asarray(output_positions, dtype=float).clip(0.0, 1.0)
    out["equity_nav"] = navs
    out["equity_drawdown"] = drawdowns
    out["drawdown_budget_cap"] = caps
    out["drawdown_budget_active"] = active_flags
    out["effective_trade_threshold"] = effective_thresholds
    out["effective_trade_lock_days"] = effective_locks
    return out


__all__ = [
    "DrawdownBudgetOverlayConfig",
    "apply_drawdown_budget_overlay",
]

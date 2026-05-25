from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

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


def _max_drawdown(values: pd.Series) -> float:
    nav = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(nav) < 2:
        return 0.0
    peak = nav.cummax()
    return abs(float((nav / peak - 1.0).min()) * 100.0)


def _active_position_nav(frame: pd.DataFrame, price_series: pd.Series, *, position_col: str) -> pd.DataFrame:
    if "datetime" not in frame.columns:
        raise ValueError("frame missing `datetime` column")
    if position_col not in frame.columns:
        raise ValueError(f"frame missing `{position_col}` column")
    positions = frame[["datetime", position_col]].copy()
    positions["datetime"] = pd.to_datetime(positions["datetime"])
    positions[position_col] = pd.to_numeric(positions[position_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    scored = positions.merge(_price_frame(price_series), on="datetime", how="inner")
    if len(scored) < 2:
        raise ValueError("not enough overlapping position and price rows")
    scored = scored.sort_values("datetime").reset_index(drop=True)
    price = scored["price"].to_numpy(dtype=float)
    position = scored[position_col].to_numpy(dtype=float)
    returns = np.zeros(len(scored), dtype=float)
    returns[1:] = np.divide(price[1:], price[:-1], out=np.ones(len(price) - 1), where=price[:-1] != 0.0) - 1.0
    active_position = np.r_[position[0], position[:-1]]
    scored["strategy_nav"] = np.cumprod(1.0 + active_position * returns)
    scored["buyhold_nav"] = price / price[0] if price[0] else 1.0
    scored["active_position"] = active_position
    return scored


def drawdown_budget_overlay_metrics(
    frame: pd.DataFrame,
    price_series: pd.Series,
    *,
    position_col: str = "position",
    base_position_col: str = "base_position",
    target_relative_return: float = 30.0,
    target_max_dd: float = 15.0,
) -> dict[str, float | int | bool]:
    """Compute target-gate metrics for a drawdown-budget overlay path."""

    scored = _active_position_nav(frame, price_series, position_col=position_col)
    strategy = scored["strategy_nav"]
    buyhold = scored["buyhold_nav"]
    strategy_return = float((strategy.iloc[-1] / strategy.iloc[0] - 1.0) * 100.0)
    buyhold_return = float((buyhold.iloc[-1] / buyhold.iloc[0] - 1.0) * 100.0)
    position = pd.to_numeric(scored[position_col], errors="coerce").fillna(0.0)
    if base_position_col in frame.columns:
        base = frame[["datetime", base_position_col]].copy()
        base["datetime"] = pd.to_datetime(base["datetime"])
        scored = scored.merge(base, on="datetime", how="left")
        base_position = pd.to_numeric(scored[base_position_col], errors="coerce").fillna(position)
    else:
        base_position = position
    changed_days = int((position.sub(base_position).abs() > 1e-9).sum())
    active_days = int(pd.Series(frame.get("drawdown_budget_active", False)).fillna(False).astype(bool).sum())
    relative_return = strategy_return - buyhold_return
    max_dd = _max_drawdown(strategy)
    relative_gap = max(0.0, float(target_relative_return) - relative_return)
    drawdown_gap = max(0.0, max_dd - float(target_max_dd))
    turnover = float(position.diff().abs().fillna(0.0).sum())
    robust_score = relative_gap**2 + 4.0 * drawdown_gap**2 + 0.02 * turnover
    return {
        "rows": int(len(scored)),
        "return": strategy_return,
        "buy_and_hold_return": buyhold_return,
        "relative_return": float(relative_return),
        "max_dd": float(max_dd),
        "avg_pos": float(position.mean()) if len(position) else 0.0,
        "turnover": turnover,
        "active_days": active_days,
        "changed_days": changed_days,
        "relative_gap": float(relative_gap),
        "drawdown_gap": float(drawdown_gap),
        "target_gate": bool(relative_gap == 0.0 and drawdown_gap == 0.0),
        "robust_score": float(robust_score),
    }


def screen_drawdown_budget_overlay_configs(
    frame: pd.DataFrame,
    price_series: pd.Series,
    configs: Iterable[DrawdownBudgetOverlayConfig],
    *,
    fold: str,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    target_relative_return: float = 30.0,
    target_max_dd: float = 15.0,
    position_col: str = "position",
) -> pd.DataFrame:
    """Score drawdown-budget overlay configs using only a fold train window."""

    start = pd.Timestamp(train_start).normalize()
    end = pd.Timestamp(train_end).normalize()
    if end < start:
        raise ValueError("train_end is before train_start")
    full_frame = frame.copy()
    full_frame["datetime"] = pd.to_datetime(full_frame["datetime"])
    full_rows = int(len(full_frame))
    rows: list[dict[str, object]] = []
    for candidate_id, config in enumerate(configs):
        overlay = apply_drawdown_budget_overlay(
            full_frame,
            price_series,
            config,
            position_col=position_col,
        )
        dates = pd.to_datetime(overlay["datetime"]).dt.normalize()
        train_overlay = overlay.loc[(dates >= start) & (dates <= end)].copy()
        if train_overlay.empty:
            continue
        metrics = drawdown_budget_overlay_metrics(
            train_overlay,
            price_series,
            position_col="position",
            target_relative_return=target_relative_return,
            target_max_dd=target_max_dd,
        )
        rows.append(
            {
                "fold": str(fold),
                "candidate_id": int(candidate_id),
                **asdict(config),
                "full_rows": full_rows,
                **{f"train_{key}": value for key, value in metrics.items()},
                "target_gate": bool(metrics["target_gate"]),
                "robust_score": float(metrics["robust_score"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["target_gate", "robust_score", "candidate_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


__all__ = [
    "DrawdownBudgetOverlayConfig",
    "apply_drawdown_budget_overlay",
    "drawdown_budget_overlay_metrics",
    "screen_drawdown_budget_overlay_configs",
]

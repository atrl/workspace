from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def apply_position_sizing_policy(
    raw_target: pd.Series,
    *,
    min_position: float = 0.0,
    max_position: float = 1.0,
    rebalance_threshold: float = 0.0,
    max_daily_change: float | None = None,
    min_hold_days: int = 1,
) -> pd.Series:
    """Convert a raw target series into a constrained target-position series."""

    if min_position > max_position:
        raise ValueError("min_position cannot exceed max_position")
    if rebalance_threshold < 0:
        raise ValueError("rebalance_threshold cannot be negative")
    if max_daily_change is not None and max_daily_change <= 0:
        raise ValueError("max_daily_change must be positive when provided")
    if min_hold_days <= 0:
        raise ValueError("min_hold_days must be positive")

    clipped = raw_target.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(
        lower=float(min_position),
        upper=float(max_position),
    )
    values: list[float] = []
    current = 0.0
    days_since_trade = int(min_hold_days)
    for desired in clipped.to_numpy(dtype="float64"):
        next_position = float(desired)
        diff = next_position - current
        if abs(diff) < float(rebalance_threshold):
            next_position = current
        elif days_since_trade < int(min_hold_days):
            next_position = current
        elif max_daily_change is not None:
            step = float(np.clip(diff, -float(max_daily_change), float(max_daily_change)))
            next_position = current + step

        next_position = float(np.clip(next_position, float(min_position), float(max_position)))
        if abs(next_position - current) > 1e-12:
            days_since_trade = 1
        else:
            days_since_trade += 1
        current = next_position
        values.append(current)
    return pd.Series(values, index=raw_target.index, name="target_position")


def volatility_scaled_target(
    raw_target: pd.Series,
    realized_vol: pd.Series,
    *,
    target_vol: float,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
) -> pd.Series:
    """Scale raw target exposure by realized volatility, capped to a stable range."""

    if target_vol <= 0:
        raise ValueError("target_vol must be positive")
    if min_scale < 0 or max_scale <= 0 or min_scale > max_scale:
        raise ValueError("invalid volatility scale bounds")
    vol = realized_vol.astype("float64").replace([np.inf, -np.inf], np.nan)
    scale = (float(target_vol) / vol).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    scale = scale.clip(lower=float(min_scale), upper=float(max_scale))
    return raw_target.astype("float64") * scale


def _max_drawdown_pct(strategy_returns: pd.Series) -> float:
    if strategy_returns.empty:
        return 0.0
    equity = (1.0 + strategy_returns.astype("float64")).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def _holding_period_stats(target_position: pd.Series) -> tuple[int, int, float]:
    deltas = target_position.astype("float64").diff().fillna(target_position.astype("float64")).to_numpy()
    trade_mask = np.abs(deltas) > 1e-9
    trade_count = int(trade_mask.sum())
    next_day_reversal_count = int(
        sum(
            1
            for previous, current in zip(deltas[:-1], deltas[1:])
            if abs(previous) > 1e-9 and abs(current) > 1e-9 and np.sign(previous) != np.sign(current)
        )
    )
    trade_indices = np.flatnonzero(trade_mask)
    if len(trade_indices) < 2:
        avg_holding_days = float(len(target_position)) if len(target_position) else 0.0
    else:
        avg_holding_days = float(np.diff(trade_indices).mean())
    return trade_count, next_day_reversal_count, avg_holding_days


def evaluate_position_sizing_policies(
    frame: pd.DataFrame,
    *,
    policies: list[dict[str, Any]],
    raw_target_col: str = "raw_target",
    return_col: str = "forward_return",
    realized_vol_col: str | None = None,
    benchmark_return_col: str | None = None,
    cost_per_turnover: float = 0.0003,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate target-position policies on an aligned daily return frame."""

    if frame.empty:
        raise ValueError("position sizing frame is empty")
    if not policies:
        raise ValueError("at least one position sizing policy is required")
    required = {raw_target_col, return_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"position sizing frame missing columns: {missing}")
    if realized_vol_col is not None and realized_vol_col not in frame.columns:
        raise ValueError(f"position sizing frame missing realized_vol_col: {realized_vol_col}")
    if benchmark_return_col is not None and benchmark_return_col not in frame.columns:
        raise ValueError(f"position sizing frame missing benchmark_return_col: {benchmark_return_col}")

    data = frame.copy()
    data = data.sort_values("datetime").reset_index(drop=True) if "datetime" in data.columns else data.reset_index(drop=True)
    raw_target = data[raw_target_col].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    forward_return = data[return_col].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    metric_rows: list[dict[str, Any]] = []
    position_rows: list[pd.DataFrame] = []
    for policy in policies:
        candidate = str(policy["candidate"])
        candidate_raw_target = raw_target.copy()
        target_vol = _float_or_none(policy.get("target_vol"))
        policy_realized_vol_col = policy.get("realized_vol_col", realized_vol_col)
        if target_vol is not None:
            if policy_realized_vol_col is None:
                raise ValueError(f"policy {candidate!r} requires realized_vol_col for target_vol")
            candidate_raw_target = volatility_scaled_target(
                candidate_raw_target,
                data[str(policy_realized_vol_col)],
                target_vol=target_vol,
                min_scale=float(policy.get("min_vol_scale", 0.0)),
                max_scale=float(policy.get("max_vol_scale", 1.0)),
            )

        target_position = apply_position_sizing_policy(
            candidate_raw_target,
            min_position=float(policy.get("min_position", 0.0)),
            max_position=float(policy.get("max_position", 1.0)),
            rebalance_threshold=float(policy.get("rebalance_threshold", 0.0)),
            max_daily_change=_float_or_none(policy.get("max_daily_change")),
            min_hold_days=int(policy.get("min_hold_days", 1)),
        )
        turnover = target_position.diff().abs().fillna(target_position.abs())
        strategy_returns = target_position * forward_return - turnover * float(cost_per_turnover)
        return_pct = float(((1.0 + strategy_returns).prod() - 1.0) * 100.0)
        benchmark_return_pct = None
        if benchmark_return_col is not None:
            benchmark_return_pct = float(((1.0 + data[benchmark_return_col].astype("float64")).prod() - 1.0) * 100.0)
        trade_count, next_day_reversal_count, avg_holding_days = _holding_period_stats(target_position)

        metric_rows.append(
            {
                "candidate": candidate,
                "return_pct": return_pct,
                "benchmark_return_pct": benchmark_return_pct,
                "max_drawdown_pct": _max_drawdown_pct(strategy_returns),
                "avg_abs_position": float(target_position.abs().mean()),
                "max_position": float(target_position.max()),
                "min_position": float(target_position.min()),
                "days_above_100": int((target_position > 1.0 + 1e-9).sum()),
                "days_below_0": int((target_position < -1e-9).sum()),
                "total_turnover": float(turnover.sum()),
                "trade_count": trade_count,
                "next_day_reversal_count": next_day_reversal_count,
                "avg_holding_days": avg_holding_days,
                "cost_per_turnover": float(cost_per_turnover),
                "params": {
                    key: value
                    for key, value in policy.items()
                    if key not in {"candidate"}
                },
            }
        )

        position_frame = pd.DataFrame(
            {
                "candidate": candidate,
                "target_position": target_position,
                "raw_target": raw_target,
                "turnover": turnover,
                "strategy_return": strategy_returns,
            }
        )
        if "datetime" in data.columns:
            position_frame.insert(0, "datetime", pd.to_datetime(data["datetime"]).dt.normalize())
        position_rows.append(position_frame)

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics["params"] = metrics["params"].map(lambda value: value if isinstance(value, str) else str(value))
    positions = pd.concat(position_rows, ignore_index=True) if position_rows else pd.DataFrame()
    return metrics.sort_values("return_pct", ascending=False).reset_index(drop=True), positions

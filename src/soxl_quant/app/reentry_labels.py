from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReentryOracleLabelConfig:
    """Future-aware re-entry labels for no-promotion BC diagnostics."""

    horizon_days: int = 21
    max_base_position: float = 0.35
    target_position: float = 0.65
    min_forward_return: float = 0.25
    max_forward_drawdown: float = 0.15

    def __post_init__(self) -> None:
        if int(self.horizon_days) <= 0:
            raise ValueError("horizon_days must be positive")
        for name in ("max_base_position", "target_position"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if float(self.max_forward_drawdown) < 0.0:
            raise ValueError("max_forward_drawdown must be non-negative")


def _normalise_daily(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "position", "price"}
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(f"daily replay missing required columns: {missing}")
    out = daily[["datetime", "position", "price"]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["position"] = out["position"].astype(float).clip(0.0, 1.0)
    out["price"] = out["price"].astype(float)
    out = out.dropna(subset=["datetime", "position", "price"]).sort_values("datetime").reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("daily replay needs at least two priced rows")
    return out


def _forward_metrics(prices: np.ndarray, horizon_days: int) -> tuple[np.ndarray, np.ndarray]:
    horizon = int(horizon_days)
    forward_return = np.full(len(prices), np.nan, dtype=float)
    forward_drawdown = np.full(len(prices), np.nan, dtype=float)
    for idx, start_price in enumerate(prices):
        end_idx = idx + horizon
        if end_idx >= len(prices) or start_price <= 0.0:
            continue
        path = prices[idx + 1 : end_idx + 1] / start_price - 1.0
        forward_return[idx] = prices[end_idx] / start_price - 1.0
        forward_drawdown[idx] = max(0.0, -float(np.min(path)))
    return forward_return, forward_drawdown


def build_reentry_oracle_label_frame(
    daily: pd.DataFrame,
    config: ReentryOracleLabelConfig | None = None,
) -> pd.DataFrame:
    """Build future-aware re-entry labels from a daily replay.

    These labels intentionally use forward returns and must stay in diagnostic
    or training-label artifacts. They are not promotion evidence by themselves.
    """

    cfg = config or ReentryOracleLabelConfig()
    out = _normalise_daily(daily)
    prices = out["price"].to_numpy(dtype=float)
    forward_return, forward_drawdown = _forward_metrics(prices, cfg.horizon_days)
    out["base_position"] = out["position"].astype(float)
    out["forward_return"] = forward_return
    out["forward_drawdown"] = forward_drawdown
    out["has_forward_window"] = np.isfinite(forward_return) & np.isfinite(forward_drawdown)
    out["oracle_reentry"] = (
        out["has_forward_window"]
        & (out["base_position"] <= float(cfg.max_base_position))
        & (out["forward_return"] >= float(cfg.min_forward_return))
        & (out["forward_drawdown"] <= float(cfg.max_forward_drawdown))
    )
    out["expert_position"] = np.where(
        out["oracle_reentry"],
        np.maximum(out["base_position"], float(cfg.target_position)),
        out["base_position"],
    ).clip(0.0, 1.0)
    out["label_lift"] = out["expert_position"] - out["base_position"]
    return out


__all__ = ["ReentryOracleLabelConfig", "build_reentry_oracle_label_frame"]

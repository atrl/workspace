from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Close-to-close triple-barrier labels for diagnostic action targets."""

    horizon_days: int = 21
    upper_return: float = 0.20
    lower_return: float = 0.10

    def __post_init__(self) -> None:
        if int(self.horizon_days) <= 0:
            raise ValueError("horizon_days must be positive")
        if float(self.upper_return) <= 0.0:
            raise ValueError("upper_return must be positive")
        if float(self.lower_return) <= 0.0:
            raise ValueError("lower_return must be positive")


@dataclass(frozen=True)
class BarrierActionLabelConfig:
    """Map triple-barrier outcomes to staged re-entry/de-risk actions."""

    barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    max_reentry_position: float = 0.35
    min_derisk_position: float = 0.50
    reentry_position: float = 0.75
    derisk_position: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_reentry_position",
            "min_derisk_position",
            "reentry_position",
            "derisk_position",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


def _normalise_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "price"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"daily frame missing required columns: {missing}")
    out = frame[["datetime", "price"]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["datetime", "price"]).sort_values("datetime").reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("daily frame needs at least two priced rows")
    return out


def compute_triple_barrier_outcomes(
    daily: pd.DataFrame,
    config: TripleBarrierConfig | None = None,
) -> pd.DataFrame:
    """Compute first-hit upper/lower barrier outcomes from close prices.

    Labels are diagnostic training targets: `1` means the upper return barrier
    was hit first, `-1` means the lower drawdown barrier was hit first, and `0`
    means neither barrier was reached before the vertical horizon.
    """

    cfg = config or TripleBarrierConfig()
    out = _normalise_price_frame(daily)
    prices = out["price"].to_numpy(dtype=float)
    horizon = int(cfg.horizon_days)
    upper = float(cfg.upper_return)
    lower = float(cfg.lower_return)

    labels = np.zeros(len(out), dtype=int)
    hits = np.full(len(out), "none", dtype=object)
    hit_offsets = np.zeros(len(out), dtype=int)
    has_forward = np.zeros(len(out), dtype=bool)
    forward_return = np.full(len(out), np.nan, dtype=float)
    max_forward_return = np.full(len(out), np.nan, dtype=float)
    max_forward_drawdown = np.full(len(out), np.nan, dtype=float)

    for idx, start_price in enumerate(prices):
        end_idx = idx + horizon
        if end_idx >= len(prices) or start_price <= 0.0:
            continue
        path = prices[idx + 1 : end_idx + 1] / start_price - 1.0
        has_forward[idx] = True
        forward_return[idx] = float(path[-1])
        max_forward_return[idx] = float(np.max(path))
        max_forward_drawdown[idx] = max(0.0, -float(np.min(path)))
        for offset, value in enumerate(path, start=1):
            if value >= upper:
                labels[idx] = 1
                hits[idx] = "upper"
                hit_offsets[idx] = int(offset)
                break
            if value <= -lower:
                labels[idx] = -1
                hits[idx] = "lower"
                hit_offsets[idx] = int(offset)
                break

    out["barrier_label"] = labels
    out["barrier_hit"] = hits
    out["hit_offset"] = hit_offsets
    out["has_forward_window"] = has_forward
    out["forward_return"] = forward_return
    out["max_forward_return"] = max_forward_return
    out["max_forward_drawdown"] = max_forward_drawdown
    return out


def build_barrier_action_label_frame(
    daily: pd.DataFrame,
    config: BarrierActionLabelConfig | None = None,
) -> pd.DataFrame:
    """Build staged action labels from first-hit barrier outcomes."""

    cfg = config or BarrierActionLabelConfig()
    required = {"datetime", "position", "price"}
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(f"daily replay missing required columns: {missing}")

    base = daily[["datetime", "position", "price"]].copy()
    base["datetime"] = pd.to_datetime(base["datetime"])
    base["position"] = pd.to_numeric(base["position"], errors="coerce").clip(0.0, 1.0)
    base["price"] = pd.to_numeric(base["price"], errors="coerce")
    base = base.dropna(subset=["datetime", "position", "price"]).sort_values("datetime").reset_index(drop=True)
    outcomes = compute_triple_barrier_outcomes(base[["datetime", "price"]], cfg.barrier)
    out = base.merge(
        outcomes.drop(columns=["price"]),
        on="datetime",
        how="inner",
    )
    out["base_position"] = out["position"].astype(float)
    out["reentry_label"] = (
        out["has_forward_window"]
        & (out["barrier_label"] == 1)
        & (out["base_position"] <= float(cfg.max_reentry_position))
    )
    out["derisk_label"] = (
        out["has_forward_window"]
        & (out["barrier_label"] == -1)
        & (out["base_position"] >= float(cfg.min_derisk_position))
    )
    out["action_label"] = np.select(
        [out["reentry_label"], out["derisk_label"]],
        [1, -1],
        default=0,
    ).astype(int)
    out["expert_position"] = out["base_position"]
    out.loc[out["reentry_label"], "expert_position"] = np.maximum(
        out.loc[out["reentry_label"], "base_position"],
        float(cfg.reentry_position),
    )
    out.loc[out["derisk_label"], "expert_position"] = np.minimum(
        out.loc[out["derisk_label"], "base_position"],
        float(cfg.derisk_position),
    )
    out["expert_position"] = out["expert_position"].astype(float).clip(0.0, 1.0)
    out["label_delta"] = out["expert_position"] - out["base_position"]
    return out


__all__ = [
    "BarrierActionLabelConfig",
    "TripleBarrierConfig",
    "build_barrier_action_label_frame",
    "compute_triple_barrier_outcomes",
]

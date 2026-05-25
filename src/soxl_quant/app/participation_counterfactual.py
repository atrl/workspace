from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ReentryParticipationOverlayConfig:
    """Feature-gated replay overlay for under-participation diagnostics."""

    enabled: bool = False
    min_target: float = 0.60
    max_target: float = 0.75
    max_base_position: float = 0.55
    min_mom_21: float = 0.12
    min_relative_strength_21: float = 1.0
    min_vol_of_vol_21: float = 0.10
    min_tracking_error_vol_21: float | None = None
    max_vix_percentile_252: float | None = 0.85
    max_equity_drawdown: float | None = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.min_target) <= 1.0:
            raise ValueError("min_target must be within [0, 1]")
        if not 0.0 <= float(self.max_target) <= 1.0:
            raise ValueError("max_target must be within [0, 1]")
        if float(self.min_target) > float(self.max_target):
            raise ValueError("min_target must be <= max_target")
        if not 0.0 <= float(self.max_base_position) <= 1.0:
            raise ValueError("max_base_position must be within [0, 1]")
        if self.max_equity_drawdown is not None and float(self.max_equity_drawdown) < 0.0:
            raise ValueError("max_equity_drawdown must be non-negative when configured")


def _normalise_feature_frame(feature_frame: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
    frame = feature_frame.copy()
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    elif isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
        if "instrument" in frame.columns:
            frame = frame.drop(columns=["instrument"])
        if "datetime" not in frame.columns:
            raise ValueError("feature_frame multi-index must include `datetime`")
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    else:
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"])

    required = list(dict.fromkeys(required_columns))
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"feature_frame missing reentry participation columns: {missing}")
    return frame[["datetime", *required]].dropna(subset=["datetime"]).sort_values("datetime")


def _required_feature_columns(config: ReentryParticipationOverlayConfig) -> list[str]:
    required = ["mom_21", "relative_strength_21", "vol_of_vol_21"]
    if config.min_tracking_error_vol_21 is not None:
        required.append("tracking_error_vol_21")
    if config.max_vix_percentile_252 is not None:
        required.append("vix_percentile_252")
    return required


def _feature_ready(row: pd.Series, config: ReentryParticipationOverlayConfig) -> bool:
    if pd.isna(row["mom_21"]) or float(row["mom_21"]) < float(config.min_mom_21):
        return False
    if pd.isna(row["relative_strength_21"]) or (
        float(row["relative_strength_21"]) < float(config.min_relative_strength_21)
    ):
        return False
    if pd.isna(row["vol_of_vol_21"]) or float(row["vol_of_vol_21"]) < float(config.min_vol_of_vol_21):
        return False
    if config.min_tracking_error_vol_21 is not None and (
        pd.isna(row["tracking_error_vol_21"])
        or float(row["tracking_error_vol_21"]) < float(config.min_tracking_error_vol_21)
    ):
        return False
    if config.max_vix_percentile_252 is not None and (
        pd.isna(row["vix_percentile_252"])
        or float(row["vix_percentile_252"]) > float(config.max_vix_percentile_252)
    ):
        return False
    return True


def apply_reentry_participation_overlay(
    daily: pd.DataFrame,
    feature_frame: pd.DataFrame,
    config: ReentryParticipationOverlayConfig | None = None,
    *,
    position_col: str = "position",
) -> pd.DataFrame:
    """Apply a same-day feature-gated participation floor to a daily replay.

    The result is intended for offline counterfactual scoring. It preserves the
    daily replay shape, keeps the original position in `base_position`, and
    writes the adjusted position back to `position`.
    """

    cfg = config or ReentryParticipationOverlayConfig()
    if "datetime" not in daily.columns:
        raise ValueError("daily frame missing `datetime` column")
    if position_col not in daily.columns:
        raise ValueError(f"daily frame missing `{position_col}` column")
    if cfg.max_equity_drawdown is not None and "price" not in daily.columns:
        raise ValueError("daily frame missing `price` column required by max_equity_drawdown")

    frame = daily.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["base_position"] = frame[position_col].astype(float).clip(0.0, 1.0)

    required_features = _required_feature_columns(cfg)
    features = _normalise_feature_frame(feature_frame, required_features)
    frame = frame.drop(columns=[col for col in required_features if col in frame.columns])
    frame = frame.merge(features, on="datetime", how="left")
    frame = frame.sort_values("datetime").reset_index(drop=True)
    if frame[required_features].notna().any(axis=1).sum() == 0:
        raise ValueError("daily frame and feature_frame have no overlapping non-null feature rows")

    current_position = 0.0
    previous_price: float | None = None
    nav = 1.0
    peak_nav = 1.0
    positions: list[float] = []
    feature_flags: list[bool] = []
    equity_allowed_flags: list[bool] = []
    ready_flags: list[bool] = []
    lifted_flags: list[bool] = []
    navs: list[float] = []
    drawdowns: list[float] = []

    for _, row in frame.iterrows():
        if cfg.max_equity_drawdown is not None:
            price = float(row["price"])
            if previous_price is not None and previous_price > 0.0:
                nav *= 1.0 + current_position * (price / previous_price - 1.0)
            peak_nav = max(peak_nav, nav)
            equity_drawdown = max(1.0 - nav / max(peak_nav, 1e-12), 0.0)
            equity_allowed = equity_drawdown <= float(cfg.max_equity_drawdown)
        else:
            price = None
            equity_drawdown = 0.0
            equity_allowed = True

        base = float(row["base_position"])
        feature_ready = bool(cfg.enabled and base <= float(cfg.max_base_position) and _feature_ready(row, cfg))
        ready = bool(feature_ready and equity_allowed)
        target = base
        if ready:
            floor_target = min(float(cfg.max_target), max(base, float(cfg.min_target)))
            target = max(base, floor_target)

        target = max(0.0, min(float(target), 1.0))
        positions.append(target)
        feature_flags.append(feature_ready)
        equity_allowed_flags.append(bool(equity_allowed))
        ready_flags.append(ready)
        lifted_flags.append(bool(target > base + 1e-12))
        navs.append(float(nav))
        drawdowns.append(float(equity_drawdown))

        current_position = target
        if price is not None:
            previous_price = price

    out = frame.copy()
    out["position"] = positions
    out["reentry_participation_feature_ready"] = feature_flags
    out["reentry_participation_equity_allowed"] = equity_allowed_flags
    out["reentry_participation_ready"] = ready_flags
    out["reentry_participation_lifted"] = lifted_flags
    out["equity_nav"] = navs
    out["equity_drawdown"] = drawdowns
    return out


__all__ = [
    "ReentryParticipationOverlayConfig",
    "apply_reentry_participation_overlay",
]

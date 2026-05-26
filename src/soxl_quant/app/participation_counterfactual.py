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


@dataclass(frozen=True)
class TrendParticipationGuardConfig:
    """High-trend participation lift with volatility and path-drawdown guards."""

    enabled: bool = False
    min_target: float = 0.45
    max_target: float = 0.65
    max_base_position: float = 0.35
    min_anchor_gap_180: float | None = 0.50
    min_mom_63: float | None = 0.40
    min_trend_slope_63: float | None = 0.10
    min_trend_gap_63: float | None = 0.10
    max_vix_percentile_252: float | None = 0.45
    max_realized_vol_63: float | None = 1.00
    min_drawdown_63: float | None = -0.25
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


def _required_trend_feature_columns(config: TrendParticipationGuardConfig) -> list[str]:
    required: list[str] = []
    if config.min_anchor_gap_180 is not None:
        required.append("anchor_gap_180")
    if config.min_mom_63 is not None:
        required.append("mom_63")
    if config.min_trend_slope_63 is not None:
        required.append("trend_slope_63")
    if config.min_trend_gap_63 is not None:
        required.append("trend_gap_63")
    if config.max_vix_percentile_252 is not None:
        required.append("vix_percentile_252")
    if config.max_realized_vol_63 is not None:
        required.append("realized_vol_63")
    if config.min_drawdown_63 is not None:
        required.append("drawdown_63")
    return required


def _meets_min(row: pd.Series, column: str, threshold: float | None) -> bool:
    if threshold is None:
        return True
    if pd.isna(row[column]):
        return False
    return float(row[column]) >= float(threshold)


def _meets_max(row: pd.Series, column: str, threshold: float | None) -> bool:
    if threshold is None:
        return True
    if pd.isna(row[column]):
        return False
    return float(row[column]) <= float(threshold)


def _trend_signal_ready(row: pd.Series, config: TrendParticipationGuardConfig) -> bool:
    return (
        _meets_min(row, "anchor_gap_180", config.min_anchor_gap_180)
        and _meets_min(row, "mom_63", config.min_mom_63)
        and _meets_min(row, "trend_slope_63", config.min_trend_slope_63)
        and _meets_min(row, "trend_gap_63", config.min_trend_gap_63)
    )


def _trend_guard_ready(row: pd.Series, config: TrendParticipationGuardConfig) -> bool:
    return (
        _meets_max(row, "vix_percentile_252", config.max_vix_percentile_252)
        and _meets_max(row, "realized_vol_63", config.max_realized_vol_63)
        and _meets_min(row, "drawdown_63", config.min_drawdown_63)
    )


def _price_proxy(frame: pd.DataFrame, price_col: str) -> pd.Series:
    if price_col in frame.columns:
        return pd.to_numeric(frame[price_col], errors="coerce")
    if "buyhold_nav" in frame.columns:
        return pd.to_numeric(frame["buyhold_nav"], errors="coerce")
    raise ValueError(f"daily frame missing `{price_col}` or `buyhold_nav` price proxy")


def _starting_nav(frame: pd.DataFrame) -> float:
    if "strategy_nav" in frame.columns:
        nav = pd.to_numeric(frame["strategy_nav"], errors="coerce").dropna()
        if not nav.empty:
            return float(nav.iloc[0])
    return 1.0


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


def apply_trend_participation_guard_overlay(
    daily: pd.DataFrame,
    feature_frame: pd.DataFrame,
    config: TrendParticipationGuardConfig | None = None,
    *,
    position_col: str = "position",
    price_col: str = "price",
) -> pd.DataFrame:
    """Apply a high-trend participation floor with volatility/drawdown guards.

    This is an offline counterfactual helper. It keeps the original position in
    `base_position`, preserves original `strategy_nav` as `base_strategy_nav`,
    writes the adjusted target to `position`, and recomputes `strategy_nav`
    using prior-day adjusted positions.
    """

    cfg = config or TrendParticipationGuardConfig()
    if "datetime" not in daily.columns:
        raise ValueError("daily frame missing `datetime` column")
    if position_col not in daily.columns:
        raise ValueError(f"daily frame missing `{position_col}` column")

    frame = daily.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["base_position"] = pd.to_numeric(frame[position_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "strategy_nav" in frame.columns:
        frame["base_strategy_nav"] = pd.to_numeric(frame["strategy_nav"], errors="coerce")

    required_features = _required_trend_feature_columns(cfg)
    features = _normalise_feature_frame(feature_frame, required_features)
    frame = frame.drop(columns=[col for col in required_features if col in frame.columns])
    frame = frame.merge(features, on="datetime", how="left")
    frame = frame.sort_values("datetime").reset_index(drop=True)
    if required_features and frame[required_features].notna().any(axis=1).sum() == 0:
        raise ValueError("daily frame and feature_frame have no overlapping non-null feature rows")

    price = _price_proxy(frame, price_col)
    if price.isna().all():
        raise ValueError("daily frame price proxy is all NaN")
    price = price.ffill().bfill()
    start_nav = _starting_nav(frame)
    current_position = 0.0
    previous_price: float | None = None
    nav = float(start_nav)
    peak_nav = float(start_nav)

    positions: list[float] = []
    trend_flags: list[bool] = []
    guard_flags: list[bool] = []
    equity_flags: list[bool] = []
    ready_flags: list[bool] = []
    lifted_flags: list[bool] = []
    navs: list[float] = []
    drawdowns: list[float] = []

    for idx, row in frame.iterrows():
        row_price = float(price.iloc[idx])
        if previous_price is not None and previous_price > 0.0:
            nav *= 1.0 + current_position * (row_price / previous_price - 1.0)
        peak_nav = max(peak_nav, nav)
        equity_drawdown = max(1.0 - nav / max(peak_nav, 1e-12), 0.0)
        equity_allowed = (
            True
            if cfg.max_equity_drawdown is None
            else equity_drawdown <= float(cfg.max_equity_drawdown)
        )

        base = float(row["base_position"])
        trend_ready = bool(cfg.enabled and base <= float(cfg.max_base_position) and _trend_signal_ready(row, cfg))
        guard_ready = bool(_trend_guard_ready(row, cfg))
        ready = bool(trend_ready and guard_ready and equity_allowed)
        target = base
        if ready:
            target = max(base, min(float(cfg.max_target), max(base, float(cfg.min_target))))
        target = max(0.0, min(float(target), 1.0))

        positions.append(target)
        trend_flags.append(trend_ready)
        guard_flags.append(guard_ready)
        equity_flags.append(bool(equity_allowed))
        ready_flags.append(ready)
        lifted_flags.append(bool(target > base + 1e-12))
        navs.append(float(nav))
        drawdowns.append(float(equity_drawdown))

        current_position = target
        previous_price = row_price

    out = frame.copy()
    out["position"] = positions
    out["strategy_nav"] = navs
    if "buyhold_nav" not in out.columns:
        out["buyhold_nav"] = start_nav * price / float(price.iloc[0])
    out["trend_participation_signal_ready"] = trend_flags
    out["trend_participation_guard_ready"] = guard_flags
    out["trend_participation_equity_allowed"] = equity_flags
    out["trend_participation_ready"] = ready_flags
    out["trend_participation_lifted"] = lifted_flags
    out["equity_nav"] = navs
    out["equity_drawdown"] = drawdowns
    return out


__all__ = [
    "ReentryParticipationOverlayConfig",
    "TrendParticipationGuardConfig",
    "apply_reentry_participation_overlay",
    "apply_trend_participation_guard_overlay",
]

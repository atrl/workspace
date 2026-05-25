from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd


DEFAULT_MODE_SLACK = {
    "trend_hold": 0.35,
    "crash_rebuild": 0.25,
    "defensive_hold": 0.10,
    "flat_wait": 0.05,
}

DEFAULT_MODE_BANDS = {
    "trend_hold": (0.45, 0.95),
    "crash_rebuild": (0.25, 0.65),
    "defensive_hold": (0.10, 0.45),
    "flat_wait": (0.00, 0.25),
}


@dataclass(frozen=True)
class TwoSpeedBudgetConfig:
    """Fixed-signal two-speed composition settings.

    The slow layer defines a risk anchor. The fast layer may participate above
    that anchor only by the configured per-mode slack.
    """

    mode_slack: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_MODE_SLACK))
    mode_bands: Mapping[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_MODE_BANDS))
    trade_threshold: float = 0.04
    trade_lock_days: int = 5


@dataclass(frozen=True)
class BullParticipationGuardConfig:
    enabled: bool = False
    ma_window: int = 50
    momentum_window: int = 20
    high_window: int = 50
    min_periods: int = 20
    min_momentum: float = 0.03
    max_drawdown: float = 0.08


@dataclass(frozen=True)
class RecoveryReentryGuardConfig:
    enabled: bool = False
    min_target: float = 0.35
    max_target: float = 0.55
    max_base_target: float = 0.25
    min_mom_10: float = 0.08
    min_mom_21: float = 0.12
    min_relative_strength_21: float = 1.0
    min_vol_of_vol_21: float = 0.10
    max_vix_percentile_252: float = 0.85
    max_below_anchor_ratio_63: float = 1.0
    min_anchor_gap_180: float = -10.0


@dataclass(frozen=True)
class DeepCapitulationReentryGuardConfig:
    enabled: bool = False
    min_target: float = 0.30
    max_target: float = 0.45
    max_base_target: float = 0.20
    max_mom_5: float = 0.05
    max_drawdown_63: float = -0.58


@dataclass(frozen=True)
class EquityDrawdownControlConfig:
    enabled: bool = False
    start_drawdown: float = 0.10
    full_drawdown: float = 0.18
    min_target: float = 0.25
    sell_trade_threshold: float | None = 0.0
    sell_trade_lock_days: int | None = 0
    max_anchor_gap_180: float | None = None
    min_below_anchor_ratio_63: float | None = None
    min_vix_percentile_252: float | None = None
    min_vol_of_vol_21: float | None = None
    max_mom_21: float | None = None


RECOVERY_REENTRY_FEATURE_COLUMNS = (
    "mom_10",
    "mom_21",
    "relative_strength_21",
    "vol_of_vol_21",
    "vix_percentile_252",
    "below_anchor_ratio_63",
    "anchor_gap_180",
)

RECOVERY_REENTRY_REQUIRED_BASE_COLUMNS = (
    "mom_10",
    "mom_21",
    "relative_strength_21",
    "vol_of_vol_21",
    "vix_percentile_252",
)

DEEP_CAPITULATION_FEATURE_COLUMNS = ("mom_5", "drawdown_63")

ALL_OPTIONAL_FEATURE_COLUMNS = (
    *RECOVERY_REENTRY_FEATURE_COLUMNS,
    *DEEP_CAPITULATION_FEATURE_COLUMNS,
)


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


def compose_budget_usage_position(
    fast_position: float,
    mode: str,
    config: TwoSpeedBudgetConfig | None = None,
) -> float:
    """Treat the fast signal as usage ratio inside the mode risk budget."""

    cfg = config or TwoSpeedBudgetConfig()
    usage = max(0.0, min(float(fast_position), 1.0))
    floor, cap = cfg.mode_bands.get(mode, (0.0, 1.0))
    floor = max(0.0, min(float(floor), 1.0))
    cap = max(floor, min(float(cap), 1.0))
    return round(floor + usage * (cap - floor), 10)


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


def equity_drawdown_target_cap(drawdown: float, config: EquityDrawdownControlConfig | None = None) -> float:
    """Linearly cap exposure as realized strategy drawdown deepens."""

    cfg = config or EquityDrawdownControlConfig()
    if not cfg.enabled:
        return 1.0
    dd = max(float(drawdown), 0.0)
    start = max(float(cfg.start_drawdown), 0.0)
    full = max(float(cfg.full_drawdown), start)
    min_target = max(0.0, min(float(cfg.min_target), 1.0))
    if dd < start:
        return 1.0
    if full <= start or dd >= full:
        return min_target
    usage = (dd - start) / max(full - start, 1e-12)
    return max(min_target, 1.0 - usage * (1.0 - min_target))


def equity_drawdown_uses_feature_filters(config: EquityDrawdownControlConfig | None = None) -> bool:
    cfg = config or EquityDrawdownControlConfig()
    return any(
        value is not None
        for value in (
            cfg.max_anchor_gap_180,
            cfg.min_below_anchor_ratio_63,
            cfg.min_vix_percentile_252,
            cfg.min_vol_of_vol_21,
            cfg.max_mom_21,
        )
    )


def _equity_drawdown_required_feature_columns(config: EquityDrawdownControlConfig) -> set[str]:
    required: set[str] = set()
    if config.max_anchor_gap_180 is not None:
        required.add("anchor_gap_180")
    if config.min_below_anchor_ratio_63 is not None:
        required.add("below_anchor_ratio_63")
    if config.min_vix_percentile_252 is not None:
        required.add("vix_percentile_252")
    if config.min_vol_of_vol_21 is not None:
        required.add("vol_of_vol_21")
    if config.max_mom_21 is not None:
        required.add("mom_21")
    return required


def _recovery_reentry_required_feature_columns(config: RecoveryReentryGuardConfig) -> set[str]:
    required = set(RECOVERY_REENTRY_REQUIRED_BASE_COLUMNS)
    if float(config.max_below_anchor_ratio_63) < 1.0:
        required.add("below_anchor_ratio_63")
    if float(config.min_anchor_gap_180) > -10.0:
        required.add("anchor_gap_180")
    return required


def equity_drawdown_regime_ready(row: pd.Series, config: EquityDrawdownControlConfig | None = None) -> bool:
    """Check optional feature filters for account-equity drawdown control."""

    cfg = config or EquityDrawdownControlConfig()
    if not equity_drawdown_uses_feature_filters(cfg):
        return True
    if cfg.max_anchor_gap_180 is not None and float(row["anchor_gap_180"]) > float(cfg.max_anchor_gap_180):
        return False
    if cfg.min_below_anchor_ratio_63 is not None and (
        float(row["below_anchor_ratio_63"]) < float(cfg.min_below_anchor_ratio_63)
    ):
        return False
    if cfg.min_vix_percentile_252 is not None and float(row["vix_percentile_252"]) < float(cfg.min_vix_percentile_252):
        return False
    if cfg.min_vol_of_vol_21 is not None and float(row["vol_of_vol_21"]) < float(cfg.min_vol_of_vol_21):
        return False
    if cfg.max_mom_21 is not None and float(row["mom_21"]) > float(cfg.max_mom_21):
        return False
    return True


def _position_column(frame: pd.DataFrame, preferred: str) -> str:
    if preferred in frame.columns:
        return preferred
    if "position" in frame.columns:
        return "position"
    raise ValueError(f"frame missing `{preferred}` or `position` column")


def _price_frame(price_series: pd.Series) -> pd.DataFrame:
    price = price_series.copy()
    if not isinstance(price.index, pd.DatetimeIndex):
        price.index = pd.to_datetime(price.index)
    price_df = price.rename("price").reset_index()
    price_df.columns = ["datetime", "price"]
    price_df["datetime"] = pd.to_datetime(price_df["datetime"])
    return price_df.dropna(subset=["price"])


def _normalise_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    required_columns: set[str] | None = None,
) -> pd.DataFrame:
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
    required = set(required_columns or ())
    missing = [col for col in sorted(required) if col not in frame.columns]
    if missing:
        raise ValueError(f"feature_frame missing required columns: {missing}")
    for column in ALL_OPTIONAL_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    columns = ["datetime", *dict.fromkeys(ALL_OPTIONAL_FEATURE_COLUMNS)]
    return frame[columns].dropna(subset=["datetime"]).sort_values("datetime")


def _bull_guard_indicators(
    prices: pd.Series,
    guard: BullParticipationGuardConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma_window = max(int(guard.ma_window), 1)
    momentum_window = max(int(guard.momentum_window), 1)
    high_window = max(int(guard.high_window), 1)
    min_periods = max(int(guard.min_periods), 1)
    moving_average = prices.rolling(ma_window, min_periods=min(min_periods, ma_window)).mean()
    momentum = prices / prices.shift(momentum_window) - 1.0
    drawdown = prices / prices.rolling(high_window, min_periods=min(min_periods, high_window)).max() - 1.0
    return moving_average, momentum, drawdown


def bull_participation_guard_snapshot(
    price_series: pd.Series,
    guard: BullParticipationGuardConfig,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, float | bool]:
    """Evaluate the price-confirmed bull guard at one runtime date."""

    price_frame = _price_frame(price_series).sort_values("datetime").reset_index(drop=True)
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        price_frame = price_frame[price_frame["datetime"] <= as_of_ts].reset_index(drop=True)
    if price_frame.empty:
        raise ValueError("price_series has no rows available for bull participation guard")

    prices = price_frame["price"].astype(float)
    moving_average, momentum, drawdown = _bull_guard_indicators(prices, guard)
    idx = len(price_frame) - 1
    price = float(prices.iloc[idx])
    ma = float(moving_average.iloc[idx])
    mom = float(momentum.iloc[idx])
    dd = float(drawdown.iloc[idx])
    ready = bool(
        guard.enabled
        and pd.notna(ma)
        and pd.notna(mom)
        and pd.notna(dd)
        and price > ma
        and mom >= float(guard.min_momentum)
        and dd >= -float(guard.max_drawdown)
    )
    return {
        "enabled": bool(guard.enabled),
        "ready": ready,
        "price": price,
        "moving_average": ma,
        "momentum": mom,
        "drawdown": dd,
    }


def compose_budget_usage_policy_frame(
    slow: pd.DataFrame,
    fast: pd.DataFrame,
    config: TwoSpeedBudgetConfig | None = None,
    *,
    price_series: pd.Series | None = None,
    bull_guard: BullParticipationGuardConfig | None = None,
    feature_frame: pd.DataFrame | None = None,
    recovery_guard: RecoveryReentryGuardConfig | None = None,
    deep_capitulation_guard: DeepCapitulationReentryGuardConfig | None = None,
    drawdown_control: EquityDrawdownControlConfig | None = None,
) -> pd.DataFrame:
    """Compose two daily policy paths into an auditable budget-usage path.

    The slow path owns the coarse risk mode. The fast path is interpreted as a
    0..1 usage ratio inside that mode's allowed band.
    """

    cfg = config or TwoSpeedBudgetConfig()
    slow_pos_col = _position_column(slow, "slow_position")
    fast_pos_col = _position_column(fast, "fast_position")
    frame = slow[["datetime", slow_pos_col]].rename(columns={slow_pos_col: "slow_position"}).merge(
        fast[["datetime", fast_pos_col]].rename(columns={fast_pos_col: "fast_position"}),
        on="datetime",
        how="inner",
    )
    if frame.empty:
        raise ValueError("slow/fast policy frames have no overlapping datetimes")
    frame["datetime"] = pd.to_datetime(frame["datetime"])

    guard_cfg = bull_guard or BullParticipationGuardConfig()
    recovery_cfg = recovery_guard or RecoveryReentryGuardConfig()
    deep_cap_cfg = deep_capitulation_guard or DeepCapitulationReentryGuardConfig()
    drawdown_cfg = drawdown_control or EquityDrawdownControlConfig()
    price_required = bool(guard_cfg.enabled or drawdown_cfg.enabled)
    if price_required:
        if price_series is None:
            raise ValueError("price_series is required when bull_participation_guard or drawdown_control is enabled")
        frame = frame.merge(_price_frame(price_series), on="datetime", how="inner")
        frame = frame.sort_values("datetime").reset_index(drop=True)
        if frame.empty:
            raise ValueError("policy frames and price series have no overlapping datetimes")
    if guard_cfg.enabled:
        prices = frame["price"].astype(float)
        moving_average, momentum, drawdown = _bull_guard_indicators(prices, guard_cfg)
    else:
        moving_average = pd.Series([float("nan")] * len(frame), index=frame.index)
        momentum = pd.Series([float("nan")] * len(frame), index=frame.index)
        drawdown = pd.Series([float("nan")] * len(frame), index=frame.index)

    feature_filters_required = bool(
        recovery_cfg.enabled
        or deep_cap_cfg.enabled
        or equity_drawdown_uses_feature_filters(drawdown_cfg)
    )
    if feature_filters_required:
        if feature_frame is None:
            raise ValueError("feature_frame is required when recovery_reentry_guard or drawdown feature filters are enabled")
        required_feature_columns: set[str] = set()
        if recovery_cfg.enabled:
            required_feature_columns.update(_recovery_reentry_required_feature_columns(recovery_cfg))
        if deep_cap_cfg.enabled:
            required_feature_columns.update(DEEP_CAPITULATION_FEATURE_COLUMNS)
        if equity_drawdown_uses_feature_filters(drawdown_cfg):
            required_feature_columns.update(_equity_drawdown_required_feature_columns(drawdown_cfg))
        frame = frame.merge(
            _normalise_feature_frame(feature_frame, required_columns=required_feature_columns),
            on="datetime",
            how="inner",
        )
        frame = frame.sort_values("datetime").reset_index(drop=True)
        if frame.empty:
            raise ValueError("policy frames and feature frame have no overlapping datetimes")
        if guard_cfg.enabled:
            prices = frame["price"].astype(float)
            moving_average, momentum, drawdown = _bull_guard_indicators(prices, guard_cfg)
        else:
            moving_average = pd.Series([float("nan")] * len(frame), index=frame.index)
            momentum = pd.Series([float("nan")] * len(frame), index=frame.index)
            drawdown = pd.Series([float("nan")] * len(frame), index=frame.index)

    current = 0.0
    days_since_trade = cfg.trade_lock_days
    nav = 1.0
    peak_nav = 1.0
    previous_price: float | None = None
    modes: list[str] = []
    base_targets: list[float] = []
    raw_targets: list[float] = []
    targets: list[float] = []
    fast_usage: list[float] = []
    guard_flags: list[bool] = []
    recovery_flags: list[bool] = []
    deep_cap_flags: list[bool] = []
    equity_navs: list[float] = []
    equity_drawdowns: list[float] = []
    equity_caps: list[float] = []
    drawdown_flags: list[bool] = []
    drawdown_regime_flags: list[bool] = []
    effective_thresholds: list[float] = []
    effective_locks: list[int] = []
    for _, row in frame.iterrows():
        if drawdown_cfg.enabled:
            price = float(row["price"])
            if previous_price is not None and previous_price > 0:
                nav *= 1.0 + current * (price / previous_price - 1.0)
            peak_nav = max(peak_nav, nav)
            equity_drawdown = max(1.0 - nav / max(peak_nav, 1e-12), 0.0)
            equity_cap = equity_drawdown_target_cap(equity_drawdown, drawdown_cfg)
        else:
            equity_drawdown = 0.0
            equity_cap = 1.0
        slow_pos = float(row["slow_position"])
        usage = max(0.0, min(float(row["fast_position"]), 1.0))
        mode = infer_mode_from_slow_position(slow_pos)
        base_raw = compose_budget_usage_position(usage, mode, cfg)
        guard_ready = False
        if guard_cfg.enabled:
            idx = row.name
            guard_ready = bool(
                float(row["price"]) > moving_average.iloc[idx]
                and momentum.iloc[idx] >= float(guard_cfg.min_momentum)
                and drawdown.iloc[idx] >= -float(guard_cfg.max_drawdown)
            )
        raw = max(base_raw, usage) if guard_ready else base_raw
        recovery_ready = False
        if recovery_cfg.enabled and raw <= float(recovery_cfg.max_base_target):
            recovery_ready = bool(
                float(row["mom_10"]) >= float(recovery_cfg.min_mom_10)
                and float(row["mom_21"]) >= float(recovery_cfg.min_mom_21)
                and float(row["relative_strength_21"]) >= float(recovery_cfg.min_relative_strength_21)
                and float(row["vol_of_vol_21"]) >= float(recovery_cfg.min_vol_of_vol_21)
                and float(row["vix_percentile_252"]) <= float(recovery_cfg.max_vix_percentile_252)
                and float(row["below_anchor_ratio_63"]) <= float(recovery_cfg.max_below_anchor_ratio_63)
                and float(row["anchor_gap_180"]) >= float(recovery_cfg.min_anchor_gap_180)
            )
            if recovery_ready:
                raw = min(max(raw, float(recovery_cfg.min_target)), float(recovery_cfg.max_target))
        deep_cap_ready = False
        if deep_cap_cfg.enabled and raw <= float(deep_cap_cfg.max_base_target):
            deep_cap_ready = bool(
                float(row["mom_5"]) <= float(deep_cap_cfg.max_mom_5)
                and float(row["drawdown_63"]) <= float(deep_cap_cfg.max_drawdown_63)
            )
            if deep_cap_ready:
                raw = min(max(raw, float(deep_cap_cfg.min_target)), float(deep_cap_cfg.max_target))
        drawdown_regime_ready = bool(
            drawdown_cfg.enabled and equity_drawdown_regime_ready(row, drawdown_cfg)
        )
        drawdown_ready = bool(drawdown_cfg.enabled and equity_cap < 1.0 and drawdown_regime_ready)
        if drawdown_ready:
            raw = min(raw, equity_cap)
        effective_trade_threshold = float(cfg.trade_threshold)
        effective_trade_lock_days = int(cfg.trade_lock_days)
        if drawdown_ready and raw < current:
            if drawdown_cfg.sell_trade_threshold is not None:
                effective_trade_threshold = float(drawdown_cfg.sell_trade_threshold)
            if drawdown_cfg.sell_trade_lock_days is not None:
                effective_trade_lock_days = max(int(drawdown_cfg.sell_trade_lock_days), 0)
        effective_cfg = TwoSpeedBudgetConfig(
            mode_slack=cfg.mode_slack,
            mode_bands=cfg.mode_bands,
            trade_threshold=effective_trade_threshold,
            trade_lock_days=effective_trade_lock_days,
        )
        target = maybe_rebalance_position(current, raw, days_since_trade, effective_cfg)
        if abs(target - current) >= effective_trade_threshold:
            days_since_trade = 0
        else:
            days_since_trade += 1
        current = target
        if drawdown_cfg.enabled:
            previous_price = float(row["price"])
        modes.append(mode)
        base_targets.append(base_raw)
        raw_targets.append(raw)
        targets.append(target)
        fast_usage.append(usage)
        guard_flags.append(guard_ready)
        recovery_flags.append(recovery_ready)
        deep_cap_flags.append(deep_cap_ready)
        equity_navs.append(nav)
        equity_drawdowns.append(equity_drawdown)
        equity_caps.append(equity_cap)
        drawdown_flags.append(drawdown_ready)
        drawdown_regime_flags.append(drawdown_regime_ready)
        effective_thresholds.append(effective_trade_threshold)
        effective_locks.append(effective_trade_lock_days)

    out = frame.drop(columns=["price"], errors="ignore").copy()
    out["mode"] = modes
    out["fast_usage"] = fast_usage
    if guard_cfg.enabled or recovery_cfg.enabled or deep_cap_cfg.enabled:
        out["base_two_speed_position"] = base_targets
    out["raw_two_speed_position"] = raw_targets
    out["position"] = targets
    if guard_cfg.enabled:
        out["bull_participation_guard"] = guard_flags
        out["bull_guard_ma"] = moving_average
        out["bull_guard_momentum"] = momentum
        out["bull_guard_drawdown"] = drawdown
    if recovery_cfg.enabled:
        out["recovery_reentry_guard"] = recovery_flags
    if deep_cap_cfg.enabled:
        out["deep_capitulation_reentry_guard"] = deep_cap_flags
    if drawdown_cfg.enabled:
        out["equity_nav"] = equity_navs
        out["equity_drawdown"] = equity_drawdowns
        out["equity_drawdown_cap"] = equity_caps
        out["equity_drawdown_regime_ready"] = drawdown_regime_flags
        out["equity_drawdown_control"] = drawdown_flags
        out["effective_trade_threshold"] = effective_thresholds
        out["effective_trade_lock_days"] = effective_locks
    return out


__all__ = [
    "DEFAULT_MODE_BANDS",
    "DEFAULT_MODE_SLACK",
    "BullParticipationGuardConfig",
    "bull_participation_guard_snapshot",
    "DeepCapitulationReentryGuardConfig",
    "EquityDrawdownControlConfig",
    "equity_drawdown_regime_ready",
    "equity_drawdown_target_cap",
    "equity_drawdown_uses_feature_filters",
    "RecoveryReentryGuardConfig",
    "TwoSpeedBudgetConfig",
    "compose_budget_usage_policy_frame",
    "compose_budget_usage_position",
    "compose_two_speed_position",
    "infer_mode_from_slow_position",
    "maybe_rebalance_position",
]

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .quantbench_screen import QuantBenchGate, screen_quantbench_candidates, summary_record_from_payload
from .quantbench_supervised import (
    RiskAwarePositionConfig,
    fit_quantbench_supervised_models,
    prediction_rank_position_series,
    quantbench_risk_overlay_configs,
    risk_aware_position_series,
)


DEFAULT_WINDOWS: dict[str, tuple[str, str]] = {
    "test_2020": ("2020-01-01", "2020-12-31"),
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
}


@dataclass(frozen=True)
class QuantBenchPathSelectorConfig:
    """Macro-risk path selector for QuantBench-style ensemble robustness checks."""

    name: str
    defensive_path: str
    offensive_path: str
    neutral_path: str | None = None
    recovery_path: str | None = None
    stress_floor: float = 0.10
    stress_cap: float = 0.20
    neutral_floor: float = 0.0
    neutral_cap: float = 0.45
    bull_floor: float = 0.55
    bull_cap: float = 0.95
    recovery_floor: float = 0.30
    recovery_cap: float = 0.65
    stress_risk_threshold: float = 0.68
    stress_drop_threshold: float = -0.20
    stress_drawdown_threshold: float = -0.25
    bull_risk_threshold: float = 0.35
    bull_momentum_threshold: float = 0.04
    bull_trend_threshold: float = 0.02
    bull_drawdown_floor: float = -0.08
    recovery_risk_threshold: float = 0.45
    recovery_momentum_threshold: float = 0.06
    recovery_trend_threshold: float = 0.00
    recovery_drawdown_min: float = -0.25
    recovery_drawdown_max: float = -0.04
    portfolio_drawdown_trigger: float = 0.0
    portfolio_drawdown_cap: float = 0.08
    portfolio_drawdown_release: float = 0.06
    portfolio_reentry_floor: float = 0.0
    portfolio_reentry_cap: float = 0.55
    portfolio_reentry_max_drawdown: float = 1.0
    portfolio_drawdown_budget_start: float = 0.0
    portfolio_drawdown_budget_full: float = 0.0
    portfolio_drawdown_budget_start_cap: float = 1.0
    portfolio_drawdown_budget_min_cap: float = 0.0
    portfolio_drawdown_budget_stress_floor: float = 0.0
    portfolio_drawdown_budget_bull_release_cap: float = 0.0
    portfolio_drawdown_budget_bull_release_max_drawdown: float = 1.0
    portfolio_drawdown_budget_bull_release_ma_window: int = 200
    portfolio_drawdown_budget_bull_release_momentum_window: int = 126
    portfolio_drawdown_budget_bull_release_high_window: int = 126
    portfolio_drawdown_budget_bull_release_momentum_threshold: float = 0.10
    portfolio_drawdown_budget_bull_release_drawdown_floor: float = -0.12
    portfolio_drawdown_budget_bull_release_volatility_window: int = 20
    portfolio_drawdown_budget_bull_release_volatility_max: float = 0.0
    portfolio_drawdown_budget_anchor_reset_drawdown: float = 0.0
    portfolio_drawdown_budget_anchor_reset_max_drawdown: float = 0.0
    portfolio_drawdown_budget_anchor_reset_streak_days: int = 0
    confirmed_reentry_floor: float = 0.0
    confirmed_reentry_cap: float = 0.0
    confirmed_reentry_ma_window: int = 50
    confirmed_reentry_momentum_window: int = 20
    confirmed_reentry_high_window: int = 50
    confirmed_reentry_momentum_threshold: float = 0.03
    confirmed_reentry_drawdown_floor: float = -0.08
    confirmed_reentry_volatility_window: int = 20
    confirmed_reentry_volatility_max: float = 0.0
    hazard_threshold: float = 0.0
    hazard_cap: float = 0.12
    tail_shock_day_threshold: float = 0.0
    tail_shock_rolling_2d_threshold: float = 0.0
    tail_shock_rolling_3d_threshold: float = 0.0
    tail_shock_cooldown_days: int = 0
    tail_shock_cap: float = 1.0
    tail_shock_stress_floor: float = 0.0
    volatility_target: float = 0.0
    volatility_stress_target: float | None = None
    volatility_neutral_target: float | None = None
    volatility_recovery_target: float | None = None
    volatility_bull_target: float | None = None
    volatility_lookback: int = 20
    volatility_min_periods: int = 10
    volatility_floor: float = 0.0
    volatility_max_scale: float = 1.0


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _as_position_series(series: pd.Series, name: str) -> pd.Series:
    out = series.copy()
    out.index = pd.to_datetime(out.index)
    out = out.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        raise ValueError(f"path `{name}` has no finite positions")
    return out.clip(0.0, 1.0).sort_index()


def _path_position_frame(paths: Mapping[str, pd.Series]) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one path is required")
    frame = pd.concat(
        {
            name: _as_position_series(series, name)
            for name, series in paths.items()
        },
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        raise ValueError("paths have no overlapping finite dates")
    return frame.sort_index()


def _risk_column(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return frame[column].astype(float).replace([np.inf, -np.inf], np.nan).fillna(default)


def _require_path(paths: pd.DataFrame, name: str | None, role: str) -> str:
    if name is None:
        raise ValueError(f"{role} path is required")
    if name not in paths.columns:
        raise ValueError(f"{role} path `{name}` is missing from paths")
    return name


def select_quantbench_path_ensemble(
    paths: Mapping[str, pd.Series],
    risk_frame: pd.DataFrame,
    config: QuantBenchPathSelectorConfig,
) -> pd.DataFrame:
    """Select among daily position paths using laggable macro-risk state."""
    path_frame = _path_position_frame(paths)
    risk = risk_frame.copy()
    risk.index = pd.to_datetime(risk.index)
    risk = risk.sort_index().reindex(path_frame.index)

    defensive = _require_path(path_frame, config.defensive_path, "defensive")
    offensive = _require_path(path_frame, config.offensive_path, "offensive")
    neutral = config.neutral_path or defensive
    recovery = config.recovery_path or offensive
    _require_path(path_frame, neutral, "neutral")
    _require_path(path_frame, recovery, "recovery")

    regime_risk = _risk_column(risk, "regime_risk", 0.0)
    cum_drop_30 = _risk_column(risk, "cum_drop_30", 0.0)
    drawdown_63 = _risk_column(risk, "drawdown_63", 0.0)
    mom_21 = _risk_column(risk, "mom_21", 0.0)
    trend_gap_63 = _risk_column(risk, "trend_gap_63", 0.0)

    stress_mask = (
        (regime_risk >= float(config.stress_risk_threshold))
        | (cum_drop_30 <= float(config.stress_drop_threshold))
        | (drawdown_63 <= float(config.stress_drawdown_threshold))
    )
    bull_mask = (
        ~stress_mask
        & (regime_risk <= float(config.bull_risk_threshold))
        & (mom_21 >= float(config.bull_momentum_threshold))
        & (trend_gap_63 >= float(config.bull_trend_threshold))
        & (drawdown_63 >= float(config.bull_drawdown_floor))
    )
    recovery_mask = (
        ~stress_mask
        & ~bull_mask
        & (regime_risk <= float(config.recovery_risk_threshold))
        & (mom_21 >= float(config.recovery_momentum_threshold))
        & (trend_gap_63 >= float(config.recovery_trend_threshold))
        & (drawdown_63 >= float(config.recovery_drawdown_min))
        & (drawdown_63 <= float(config.recovery_drawdown_max))
    )
    selected_path = pd.Series(neutral, index=path_frame.index, dtype=object)
    selected_path.loc[stress_mask] = defensive
    selected_path.loc[bull_mask] = offensive
    selected_path.loc[recovery_mask] = recovery

    macro_regime = pd.Series("neutral", index=path_frame.index, dtype=object)
    macro_regime.loc[stress_mask] = "stress"
    macro_regime.loc[bull_mask] = "bull"
    macro_regime.loc[recovery_mask] = "recovery"

    position = path_frame[neutral].clip(float(config.neutral_floor), float(config.neutral_cap)).copy()
    position.loc[stress_mask] = path_frame.loc[stress_mask, defensive].clip(
        float(config.stress_floor),
        float(config.stress_cap),
    )
    position.loc[bull_mask] = path_frame.loc[bull_mask, offensive].clip(
        float(config.bull_floor),
        float(config.bull_cap),
    )
    position.loc[recovery_mask] = path_frame.loc[recovery_mask, recovery].clip(
        float(config.recovery_floor),
        float(config.recovery_cap),
    )

    out = pd.DataFrame(
        {
            "datetime": path_frame.index,
            "position": position.clip(0.0, 1.0).astype(float),
            "selected_path": selected_path,
            "macro_regime": macro_regime,
            "regime_risk": regime_risk,
            "cum_drop_30": cum_drop_30,
            "drawdown_63": drawdown_63,
            "mom_21": mom_21,
            "trend_gap_63": trend_gap_63,
        },
        index=path_frame.index,
    )
    for name in path_frame.columns:
        out[f"{name}_position"] = path_frame[name].astype(float)
    return out.reset_index(drop=True)


def backtest_position_path(
    close: pd.Series,
    positions: pd.Series,
    *,
    cost: float = 0.0008,
) -> dict[str, float]:
    close = close.astype(float).sort_index()
    position = positions.astype(float).reindex(close.index).ffill().fillna(0.0).clip(0.0, 1.0)
    returns = close.pct_change().fillna(0.0)
    exec_position = position.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_returns = exec_position * returns - turnover * float(cost)
    equity = (1.0 + strategy_returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    buy_hold = close / close.iloc[0]
    return_pct = (float(equity.iloc[-1]) - 1.0) * 100.0
    bnh_return_pct = (float(buy_hold.iloc[-1]) - 1.0) * 100.0
    return {
        "return": round(return_pct, 10),
        "sharpe": (
            float(strategy_returns.mean() / strategy_returns.std(ddof=0) * np.sqrt(252))
            if len(strategy_returns) > 1 and float(strategy_returns.std(ddof=0)) > 0.0
            else 0.0
        ),
        "max_dd": round(abs(float(drawdown.min())) * 100.0, 10),
        "avg_pos": float(exec_position.mean()),
        "buy_and_hold_return": round(bnh_return_pct, 10),
        "relative_return": round(return_pct - bnh_return_pct, 10),
        "turnover": float(turnover.sum()),
        "trades": int((turnover > 1e-8).sum()),
    }


def apply_quantbench_ensemble_drawdown_guard(
    daily: pd.DataFrame,
    close: pd.Series,
    config: QuantBenchPathSelectorConfig,
    *,
    cost: float = 0.0008,
) -> pd.DataFrame:
    """Apply a no-lookahead account-equity drawdown governor to selected paths."""
    out = daily.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)
    out["raw_position"] = out["position"].astype(float).clip(0.0, 1.0)
    budget_enabled = (
        float(config.portfolio_drawdown_budget_start) > 0.0
        and float(config.portfolio_drawdown_budget_full) > float(config.portfolio_drawdown_budget_start)
    )
    if float(config.portfolio_drawdown_trigger) <= 0.0 and not budget_enabled:
        out["portfolio_drawdown_guard"] = False
        out["portfolio_drawdown"] = 0.0
        return out

    price = close.astype(float).copy()
    price.index = pd.to_datetime(price.index)
    price = price.sort_index()
    returns = price.pct_change().fillna(0.0)
    confirmed_reentry_enabled = float(config.confirmed_reentry_floor) > 0.0
    confirmed_frame = pd.DataFrame(index=pd.DatetimeIndex(out["datetime"]))
    if confirmed_reentry_enabled:
        lagged_close = price.shift(1)
        ma_window = max(int(config.confirmed_reentry_ma_window), 1)
        momentum_window = max(int(config.confirmed_reentry_momentum_window), 1)
        high_window = max(int(config.confirmed_reentry_high_window), 1)
        vol_window = max(int(config.confirmed_reentry_volatility_window), 1)
        confirmed_frame["confirmed_reentry_price_above_ma"] = (
            lagged_close > lagged_close.rolling(ma_window, min_periods=ma_window).mean()
        )
        confirmed_frame["confirmed_reentry_momentum"] = lagged_close / lagged_close.shift(momentum_window) - 1.0
        confirmed_frame["confirmed_reentry_rolling_high_drawdown"] = (
            lagged_close / lagged_close.rolling(high_window, min_periods=high_window).max() - 1.0
        )
        confirmed_frame["confirmed_reentry_realized_volatility"] = (
            returns.shift(1).rolling(vol_window, min_periods=vol_window).std(ddof=0) * float(np.sqrt(252.0))
        )
        confirmed_frame = confirmed_frame.reindex(pd.DatetimeIndex(out["datetime"]))
        confirmed_reentry_series = (
            confirmed_frame["confirmed_reentry_price_above_ma"].fillna(False).astype(bool)
            & (
                confirmed_frame["confirmed_reentry_momentum"].fillna(-np.inf)
                >= float(config.confirmed_reentry_momentum_threshold)
            )
            & (
                confirmed_frame["confirmed_reentry_rolling_high_drawdown"].fillna(-np.inf)
                >= float(config.confirmed_reentry_drawdown_floor)
            )
        )
        if float(config.confirmed_reentry_volatility_max) > 0.0:
            confirmed_reentry_series &= (
                confirmed_frame["confirmed_reentry_realized_volatility"].fillna(np.inf)
                <= float(config.confirmed_reentry_volatility_max)
            )
        confirmed_reentry_by_date = confirmed_reentry_series.to_dict()
    else:
        confirmed_reentry_by_date = {}
    bull_release_enabled = (
        budget_enabled
        and float(config.portfolio_drawdown_budget_bull_release_cap) > 0.0
        and float(config.portfolio_drawdown_budget_bull_release_max_drawdown) > 0.0
    )
    bull_release_frame = pd.DataFrame(index=pd.DatetimeIndex(out["datetime"]))
    if bull_release_enabled:
        lagged_close = price.shift(1)
        ma_window = max(int(config.portfolio_drawdown_budget_bull_release_ma_window), 1)
        momentum_window = max(int(config.portfolio_drawdown_budget_bull_release_momentum_window), 1)
        high_window = max(int(config.portfolio_drawdown_budget_bull_release_high_window), 1)
        vol_window = max(int(config.portfolio_drawdown_budget_bull_release_volatility_window), 1)
        bull_release_frame["portfolio_drawdown_budget_bull_release_price_above_ma"] = (
            lagged_close > lagged_close.rolling(ma_window, min_periods=ma_window).mean()
        )
        bull_release_frame["portfolio_drawdown_budget_bull_release_momentum"] = (
            lagged_close / lagged_close.shift(momentum_window) - 1.0
        )
        bull_release_frame["portfolio_drawdown_budget_bull_release_rolling_high_drawdown"] = (
            lagged_close / lagged_close.rolling(high_window, min_periods=high_window).max() - 1.0
        )
        bull_release_frame["portfolio_drawdown_budget_bull_release_realized_volatility"] = (
            returns.shift(1).rolling(vol_window, min_periods=vol_window).std(ddof=0) * float(np.sqrt(252.0))
        )
        bull_release_frame = bull_release_frame.reindex(pd.DatetimeIndex(out["datetime"]))
        bull_release_series = (
            bull_release_frame["portfolio_drawdown_budget_bull_release_price_above_ma"].fillna(False).astype(bool)
            & (
                bull_release_frame["portfolio_drawdown_budget_bull_release_momentum"].fillna(-np.inf)
                >= float(config.portfolio_drawdown_budget_bull_release_momentum_threshold)
            )
            & (
                bull_release_frame["portfolio_drawdown_budget_bull_release_rolling_high_drawdown"].fillna(-np.inf)
                >= float(config.portfolio_drawdown_budget_bull_release_drawdown_floor)
            )
        )
        if float(config.portfolio_drawdown_budget_bull_release_volatility_max) > 0.0:
            bull_release_series &= (
                bull_release_frame["portfolio_drawdown_budget_bull_release_realized_volatility"].fillna(np.inf)
                <= float(config.portfolio_drawdown_budget_bull_release_volatility_max)
            )
        bull_release_by_date = bull_release_series.to_dict()
    else:
        bull_release_by_date = {}
    anchor_reset_enabled = (
        bull_release_enabled
        and float(config.portfolio_drawdown_budget_anchor_reset_max_drawdown) > 0.0
        and int(config.portfolio_drawdown_budget_anchor_reset_streak_days) > 0
    )
    equity = 1.0
    peak = 1.0
    budget_peak = 1.0
    previous_target = 0.0
    guard_active = False
    bull_confirmation_streak = 0
    guarded: list[float] = []
    guard_flags: list[bool] = []
    drawdowns: list[float] = []
    confirmed_flags: list[bool] = []
    budget_caps: list[float] = []
    budget_flags: list[bool] = []
    budget_drawdowns: list[float] = []
    bull_release_flags: list[bool] = []
    anchor_reset_flags: list[bool] = []
    anchor_reset_streaks: list[int] = []
    for row in out.itertuples(index=False):
        date = pd.Timestamp(row.datetime)
        day_return = float(returns.loc[date]) if date in returns.index else 0.0
        equity *= 1.0 + previous_target * day_return
        peak = max(peak, equity)
        budget_peak = max(budget_peak, equity)
        drawdown = 1.0 - equity / peak if peak > 0.0 else 0.0
        raw_target = float(row.raw_position)
        tail_shock_active = bool(getattr(row, "tail_shock_guard", False))
        anchor_reset = False
        if anchor_reset_enabled:
            if (
                not tail_shock_active
                and str(row.macro_regime) in {"bull", "recovery"}
                and drawdown <= float(config.portfolio_drawdown_budget_anchor_reset_max_drawdown)
                and bool(bull_release_by_date.get(date, False))
                and raw_target > 0.0
            ):
                bull_confirmation_streak += 1
            else:
                bull_confirmation_streak = 0
            if bull_confirmation_streak >= int(config.portfolio_drawdown_budget_anchor_reset_streak_days):
                reset_drawdown = float(np.clip(float(config.portfolio_drawdown_budget_anchor_reset_drawdown), 0.0, 0.99))
                reset_peak = equity / max(1.0 - reset_drawdown, 1e-12)
                if reset_peak < budget_peak:
                    budget_peak = reset_peak
                    anchor_reset = True
        else:
            bull_confirmation_streak = 0
        budget_drawdown = 1.0 - equity / budget_peak if budget_peak > 0.0 else 0.0
        if float(config.portfolio_drawdown_trigger) > 0.0:
            if drawdown >= float(config.portfolio_drawdown_trigger):
                guard_active = True
            elif drawdown <= float(config.portfolio_drawdown_release):
                guard_active = False
        else:
            guard_active = False

        confirmed_reentry = bool(confirmed_reentry_by_date.get(date, False))
        bull_release = False
        can_reenter = (
            guard_active
            and str(row.macro_regime) in {"bull", "recovery"}
            and config.portfolio_reentry_floor > 0.0
            and (drawdown <= float(config.portfolio_reentry_max_drawdown) or confirmed_reentry)
        )
        if can_reenter:
            reentry_floor = (
                float(config.confirmed_reentry_floor)
                if confirmed_reentry and float(config.confirmed_reentry_floor) > 0.0
                else float(config.portfolio_reentry_floor)
            )
            reentry_cap = (
                float(config.confirmed_reentry_cap)
                if confirmed_reentry and float(config.confirmed_reentry_cap) > 0.0
                else float(config.portfolio_reentry_cap)
            )
            target = min(
                max(raw_target, reentry_floor),
                reentry_cap,
            )
        elif guard_active:
            target = min(raw_target, float(config.portfolio_drawdown_cap))
        else:
            target = raw_target

        if budget_enabled:
            severity = float(
                np.clip(
                    (budget_drawdown - float(config.portfolio_drawdown_budget_start))
                    / (float(config.portfolio_drawdown_budget_full) - float(config.portfolio_drawdown_budget_start)),
                    0.0,
                    1.0,
                )
            )
            budget_cap = (1.0 - severity) * float(config.portfolio_drawdown_budget_start_cap) + severity * float(
                config.portfolio_drawdown_budget_min_cap
            )
            budget_cap = float(np.clip(budget_cap, 0.0, 1.0))
            budget_guard = budget_drawdown >= float(config.portfolio_drawdown_budget_start)
            if budget_guard:
                target = min(target, budget_cap)
                stress_floor = float(config.portfolio_drawdown_budget_stress_floor)
                if stress_floor > 0.0 and str(row.macro_regime) == "stress" and raw_target > 0.0:
                    target = max(target, min(raw_target, stress_floor))
                if (
                    bull_release_enabled
                    and not tail_shock_active
                    and str(row.macro_regime) in {"bull", "recovery"}
                    and drawdown <= float(config.portfolio_drawdown_budget_bull_release_max_drawdown)
                    and bool(bull_release_by_date.get(date, False))
                    and raw_target > 0.0
                ):
                    target = max(
                        target,
                        min(raw_target, float(config.portfolio_drawdown_budget_bull_release_cap)),
                    )
                    bull_release = True
        else:
            budget_cap = 1.0
            budget_guard = False
            budget_drawdown = drawdown

        equity -= abs(target - previous_target) * float(cost)
        peak = max(peak, equity)
        budget_peak = max(budget_peak, equity)
        guarded.append(float(np.clip(target, 0.0, 1.0)))
        guard_flags.append(bool(guard_active))
        drawdowns.append(float(drawdown))
        confirmed_flags.append(bool(confirmed_reentry))
        budget_caps.append(float(budget_cap))
        budget_flags.append(bool(budget_guard))
        budget_drawdowns.append(float(budget_drawdown))
        bull_release_flags.append(bool(bull_release))
        anchor_reset_flags.append(bool(anchor_reset))
        anchor_reset_streaks.append(int(bull_confirmation_streak))
        previous_target = target

    out["position"] = guarded
    out["portfolio_drawdown_guard"] = guard_flags
    out["portfolio_drawdown"] = drawdowns
    if budget_enabled:
        out["portfolio_drawdown_budget_cap"] = budget_caps
        out["portfolio_drawdown_budget_guard"] = budget_flags
        out["portfolio_drawdown_budget_drawdown"] = budget_drawdowns
        if bull_release_enabled:
            out["portfolio_drawdown_budget_bull_release"] = bull_release_flags
            for column in bull_release_frame.columns:
                out[column] = bull_release_frame[column].reset_index(drop=True).to_numpy()
        if anchor_reset_enabled:
            out["portfolio_drawdown_budget_anchor_reset"] = anchor_reset_flags
            out["portfolio_drawdown_budget_anchor_reset_streak"] = anchor_reset_streaks
    if confirmed_reentry_enabled:
        out["confirmed_reentry"] = confirmed_flags
        for column in confirmed_frame.columns:
            out[column] = confirmed_frame[column].reset_index(drop=True).to_numpy()
    return out


def apply_quantbench_hazard_guard(
    daily: pd.DataFrame,
    hazard_score: pd.Series | None,
    config: QuantBenchPathSelectorConfig,
) -> pd.DataFrame:
    """Cap selected targets when a trained future-drawdown hazard is elevated."""
    out = daily.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)
    if hazard_score is None or float(config.hazard_threshold) <= 0.0:
        out["hazard_score"] = np.nan
        out["hazard_guard"] = False
        return out

    hazard = hazard_score.astype(float).copy()
    hazard.index = pd.to_datetime(hazard.index)
    aligned = hazard.reindex(pd.DatetimeIndex(out["datetime"])).fillna(0.0).clip(0.0, 1.0)
    guard = aligned >= float(config.hazard_threshold)
    out["hazard_score"] = aligned.to_numpy(dtype=float)
    out["hazard_guard"] = guard.to_numpy(dtype=bool)
    out.loc[out["hazard_guard"], "position"] = out.loc[out["hazard_guard"], "position"].astype(float).clip(
        upper=float(config.hazard_cap)
    )
    return out


def apply_quantbench_tail_shock_guard(
    daily: pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    config: QuantBenchPathSelectorConfig,
) -> pd.DataFrame:
    """Cap selected targets for a cooldown period after lagged SOXL tail-loss shocks."""
    out = daily.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)
    out["tail_shock_raw_position"] = out["position"].astype(float).clip(0.0, 1.0)

    cooldown_days = max(int(config.tail_shock_cooldown_days), 0)
    enabled = (
        cooldown_days > 0
        and float(config.tail_shock_cap) < 1.0
        and (
            float(config.tail_shock_day_threshold) < 0.0
            or float(config.tail_shock_rolling_2d_threshold) < 0.0
            or float(config.tail_shock_rolling_3d_threshold) < 0.0
        )
    )
    if not enabled:
        out["tail_shock_lag_1d_return"] = np.nan
        out["tail_shock_lag_2d_return"] = np.nan
        out["tail_shock_lag_3d_return"] = np.nan
        out["tail_shock_trigger"] = False
        out["tail_shock_guard"] = False
        out["tail_shock_cooldown_remaining"] = 0
        return out

    price = _daily_price_series(close)
    returns = price.pct_change().fillna(0.0)
    lag_1d = returns.shift(1)
    lag_2d = (price / price.shift(2) - 1.0).shift(1)
    lag_3d = (price / price.shift(3) - 1.0).shift(1)
    dates = pd.DatetimeIndex(out["datetime"])
    out["tail_shock_lag_1d_return"] = lag_1d.reindex(dates).reset_index(drop=True).fillna(0.0).to_numpy(dtype=float)
    out["tail_shock_lag_2d_return"] = lag_2d.reindex(dates).reset_index(drop=True).fillna(0.0).to_numpy(dtype=float)
    out["tail_shock_lag_3d_return"] = lag_3d.reindex(dates).reset_index(drop=True).fillna(0.0).to_numpy(dtype=float)

    trigger = pd.Series(False, index=out.index, dtype=bool)
    if float(config.tail_shock_day_threshold) < 0.0:
        trigger |= out["tail_shock_lag_1d_return"] <= float(config.tail_shock_day_threshold)
    if float(config.tail_shock_rolling_2d_threshold) < 0.0:
        trigger |= out["tail_shock_lag_2d_return"] <= float(config.tail_shock_rolling_2d_threshold)
    if float(config.tail_shock_rolling_3d_threshold) < 0.0:
        trigger |= out["tail_shock_lag_3d_return"] <= float(config.tail_shock_rolling_3d_threshold)

    guarded: list[float] = []
    guard_flags: list[bool] = []
    remaining_values: list[int] = []
    remaining = 0
    cap = float(np.clip(float(config.tail_shock_cap), 0.0, 1.0))
    stress_floor = float(np.clip(float(config.tail_shock_stress_floor), 0.0, 1.0))
    for idx, row in enumerate(out.itertuples(index=False)):
        if bool(trigger.iloc[idx]):
            remaining = cooldown_days
        active = remaining > 0
        raw_target = float(row.tail_shock_raw_position)
        target = raw_target
        if active:
            target = min(target, cap)
            if stress_floor > 0.0 and str(row.macro_regime) == "stress" and raw_target > 0.0:
                target = max(target, min(raw_target, stress_floor))
            remaining -= 1
        guarded.append(float(np.clip(target, 0.0, 1.0)))
        guard_flags.append(bool(active))
        remaining_values.append(int(remaining))

    out["position"] = guarded
    out["tail_shock_trigger"] = trigger.to_numpy(dtype=bool)
    out["tail_shock_guard"] = guard_flags
    out["tail_shock_cooldown_remaining"] = remaining_values
    return out


def apply_quantbench_volatility_budget_guard(
    daily: pd.DataFrame,
    close: pd.Series,
    config: QuantBenchPathSelectorConfig,
) -> pd.DataFrame:
    """Scale selected targets by a lagged realized-volatility risk budget."""
    out = daily.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)
    out["volatility_budget_raw_position"] = out["position"].astype(float).clip(0.0, 1.0)

    base_target = float(config.volatility_target)
    if base_target <= 0.0:
        out["realized_volatility"] = np.nan
        out["volatility_budget_target"] = np.nan
        out["volatility_budget_scale"] = 1.0
        out["volatility_budget_guard"] = False
        return out

    price = close.astype(float).copy()
    price.index = pd.to_datetime(price.index)
    returns = price.sort_index().pct_change().fillna(0.0)
    lookback = max(int(config.volatility_lookback), 1)
    min_periods = min(max(int(config.volatility_min_periods), 1), lookback)
    realized_vol = returns.shift(1).rolling(lookback, min_periods=min_periods).std(ddof=0) * float(np.sqrt(252.0))
    aligned_vol = realized_vol.reindex(pd.DatetimeIndex(out["datetime"])).astype(float)

    def target_for_regime(regime: object) -> float:
        label = str(regime)
        if label == "stress" and config.volatility_stress_target is not None:
            return float(config.volatility_stress_target)
        if label == "bull" and config.volatility_bull_target is not None:
            return float(config.volatility_bull_target)
        if label == "recovery" and config.volatility_recovery_target is not None:
            return float(config.volatility_recovery_target)
        if label == "neutral" and config.volatility_neutral_target is not None:
            return float(config.volatility_neutral_target)
        return base_target

    target = pd.Series([target_for_regime(regime) for regime in out["macro_regime"]], index=out.index, dtype=float)
    raw = out["volatility_budget_raw_position"].astype(float)
    finite_budget = target > 0.0
    finite_vol = aligned_vol.reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    scale = pd.Series(1.0, index=out.index, dtype=float)
    guard = finite_budget & finite_vol.notna() & (finite_vol > target)
    scale.loc[guard] = (target.loc[guard] / finite_vol.loc[guard]).clip(
        lower=0.0,
        upper=max(float(config.volatility_max_scale), 0.0),
    )
    budgeted = raw * scale
    floor = max(float(config.volatility_floor), 0.0)
    if floor > 0.0:
        budgeted.loc[guard & (raw > 0.0)] = np.maximum(
            budgeted.loc[guard & (raw > 0.0)],
            np.minimum(raw.loc[guard & (raw > 0.0)], floor),
        )
    out.loc[guard, "position"] = np.minimum(raw.loc[guard], budgeted.loc[guard]).clip(0.0, 1.0)
    out["realized_volatility"] = finite_vol.to_numpy(dtype=float)
    out["volatility_budget_target"] = target.to_numpy(dtype=float)
    out["volatility_budget_scale"] = scale.to_numpy(dtype=float)
    out["volatility_budget_guard"] = guard.to_numpy(dtype=bool)
    return out


def _future_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    values: list[float] = []
    prices = close.astype(float).to_numpy()
    for idx, current in enumerate(prices):
        if idx + int(horizon) >= len(prices) or not np.isfinite(current) or current <= 0.0:
            values.append(float("nan"))
            continue
        path = prices[idx + 1 : idx + int(horizon) + 1]
        values.append(float(np.nanmin(path) / current - 1.0) if len(path) else float("nan"))
    return pd.Series(values, index=close.index)


def _fit_hazard_classifier(train: pd.DataFrame, feature_columns: list[str]):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=15,
            l2_regularization=0.10,
            random_state=0,
        ),
    )
    model.fit(train[feature_columns], train["drawdown_hazard_label"].astype(int))
    return model


def _hazard_feature_label_frame(
    features: pd.DataFrame,
    close: pd.Series,
    *,
    horizon: int,
    drawdown_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    feature_frame = features.copy().replace([np.inf, -np.inf], np.nan)
    feature_frame.index = pd.to_datetime(feature_frame.index)
    close_series = close.astype(float).sort_index()
    close_series.index = pd.to_datetime(close_series.index)
    common_index = feature_frame.index.intersection(close_series.index)
    feature_frame = feature_frame.loc[common_index]
    close_series = close_series.loc[common_index]
    future_dd = _future_drawdown(close_series, int(horizon))
    label = (future_dd <= float(drawdown_threshold)).astype(float).where(future_dd.notna())

    frame = feature_frame.copy()
    frame["drawdown_hazard_label"] = label
    frame = frame.dropna()
    feature_columns = [column for column in feature_frame.columns if column in frame.columns]
    return frame, feature_columns


def fit_quantbench_drawdown_hazard_model(
    features: pd.DataFrame,
    close: pd.Series,
    *,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    horizon: int = 20,
    drawdown_threshold: float = -0.18,
) -> tuple[pd.Series, dict[str, Any]]:
    """Train a non-leaking future-drawdown classifier and return OOS hazard scores."""
    frame, feature_columns = _hazard_feature_label_frame(
        features,
        close,
        horizon=horizon,
        drawdown_threshold=drawdown_threshold,
    )
    train = frame.loc[pd.Timestamp(train_start) : pd.Timestamp(train_end)]
    test = frame.loc[pd.Timestamp(test_start) : pd.Timestamp(test_end)]
    if train.empty or test.empty:
        raise ValueError("hazard train and test windows must both contain rows")
    if train["drawdown_hazard_label"].nunique(dropna=True) < 2:
        raise ValueError("hazard train window requires both drawdown and non-drawdown labels")

    model = _fit_hazard_classifier(train, feature_columns)
    hazard = pd.Series(
        model.predict_proba(test[feature_columns])[:, 1],
        index=test.index,
        name="drawdown_hazard",
    ).clip(0.0, 1.0)
    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(train["drawdown_hazard_label"].mean()),
        "test_positive_rate": float(test["drawdown_hazard_label"].mean()),
        "horizon": int(horizon),
        "drawdown_threshold": float(drawdown_threshold),
        "mean_hazard": float(hazard.mean()),
    }
    return hazard, metrics


def fit_quantbench_rolling_drawdown_hazard_model(
    features: pd.DataFrame,
    close: pd.Series,
    *,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    horizon: int = 20,
    drawdown_threshold: float = -0.18,
    min_train_rows: int = 504,
    refit_step_rows: int = 63,
    train_lookback_rows: int | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Predict drawdown hazard with expanding/rolling refits and label-lag safety."""
    frame, feature_columns = _hazard_feature_label_frame(
        features,
        close,
        horizon=horizon,
        drawdown_threshold=drawdown_threshold,
    )
    if frame.empty:
        raise ValueError("rolling hazard feature frame is empty")
    index = frame.index.sort_values()
    test_index = index[(index >= pd.Timestamp(test_start)) & (index <= pd.Timestamp(test_end))]
    if test_index.empty:
        raise ValueError("rolling hazard test window has no rows")

    hazards: list[pd.Series] = []
    folds: list[dict[str, Any]] = []
    step = max(int(refit_step_rows), 1)
    horizon_rows = max(int(horizon), 1)
    min_rows = max(int(min_train_rows), 1)
    for block_start_pos in range(0, len(test_index), step):
        block_index = test_index[block_start_pos : block_start_pos + step]
        predict_start = block_index[0]
        predict_end = block_index[-1]
        predict_start_pos = int(index.get_loc(predict_start))
        label_cutoff_pos = predict_start_pos - horizon_rows
        if label_cutoff_pos <= 0:
            continue
        train_end = index[label_cutoff_pos - 1]
        train = frame.loc[:train_end].copy()
        if train_lookback_rows is not None and int(train_lookback_rows) > 0 and len(train) > int(train_lookback_rows):
            train = train.iloc[-int(train_lookback_rows) :]
        if len(train) < min_rows or train["drawdown_hazard_label"].nunique(dropna=True) < 2:
            continue
        predict = frame.loc[block_index]
        model = _fit_hazard_classifier(train, feature_columns)
        block_hazard = pd.Series(
            model.predict_proba(predict[feature_columns])[:, 1],
            index=predict.index,
            name="drawdown_hazard",
        ).clip(0.0, 1.0)
        hazards.append(block_hazard)
        folds.append(
            {
                "predict_start": str(pd.Timestamp(predict_start).date()),
                "predict_end": str(pd.Timestamp(predict_end).date()),
                "train_start": str(pd.Timestamp(train.index.min()).date()),
                "train_end": str(pd.Timestamp(train.index.max()).date()),
                "label_cutoff": str(pd.Timestamp(train_end).date()),
                "train_rows": int(len(train)),
                "predict_rows": int(len(predict)),
                "train_positive_rate": float(train["drawdown_hazard_label"].mean()),
            }
        )

    if not hazards:
        raise ValueError("rolling hazard could not produce any folds")
    hazard = pd.concat(hazards).sort_index()
    metrics = {
        "fold_count": int(len(folds)),
        "test_rows": int(len(hazard)),
        "horizon": int(horizon),
        "drawdown_threshold": float(drawdown_threshold),
        "min_train_rows": int(min_train_rows),
        "refit_step_rows": int(refit_step_rows),
        "train_lookback_rows": None if train_lookback_rows is None else int(train_lookback_rows),
        "mean_hazard": float(hazard.mean()),
        "folds": folds,
    }
    return hazard, metrics


def _payload_for_daily(
    daily: pd.DataFrame,
    close: pd.Series,
    *,
    profile: str,
    config: QuantBenchPathSelectorConfig,
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] | None,
    cost: float,
) -> dict[str, Any]:
    frame = daily.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    positions = pd.Series(frame["position"].to_numpy(dtype=float), index=frame["datetime"])
    close = close.astype(float).sort_index()
    common = close.index.intersection(positions.index)
    if len(common) < 2:
        raise ValueError(f"ensemble `{profile}` has fewer than two overlapping close rows")
    payload = {
        "profile": profile,
        "model_family": "quantbench_path_ensemble",
        "ensemble_config": asdict(config),
        "test_proxy": backtest_position_path(close.loc[common], positions.loc[common], cost=cost),
        "slices": {},
    }
    for name, (start, end) in (windows or DEFAULT_WINDOWS).items():
        window_close = close.loc[start:end]
        window_pos = positions.loc[start:end]
        if len(window_close.index.intersection(window_pos.index)) < 2:
            continue
        payload["slices"][name] = backtest_position_path(window_close, window_pos, cost=cost)
    return payload


def build_quantbench_path_ensemble_payloads(
    close: pd.Series,
    paths: Mapping[str, pd.Series],
    risk_frame: pd.DataFrame,
    *,
    configs: Iterable[QuantBenchPathSelectorConfig],
    hazard_score: pd.Series | None = None,
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] | None = None,
    cost: float = 0.0008,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    payloads: list[dict[str, Any]] = []
    daily_by_profile: dict[str, pd.DataFrame] = {}
    for config in tuple(configs):
        daily = select_quantbench_path_ensemble(paths, risk_frame, config)
        daily = apply_quantbench_hazard_guard(daily, hazard_score, config)
        if float(config.volatility_target) > 0.0:
            daily = apply_quantbench_volatility_budget_guard(daily, close, config)
        if max(int(config.tail_shock_cooldown_days), 0) > 0:
            daily = apply_quantbench_tail_shock_guard(daily, close, config)
        daily = apply_quantbench_ensemble_drawdown_guard(daily, close, config, cost=cost)
        payload = _payload_for_daily(
            daily,
            close,
            profile=config.name,
            config=config,
            windows=windows,
            cost=cost,
        )
        payloads.append(payload)
        daily_by_profile[config.name] = daily
    return payloads, daily_by_profile


def build_quantbench_path_ensemble_report(
    close: pd.Series,
    paths: Mapping[str, pd.Series],
    risk_frame: pd.DataFrame,
    *,
    card_dir: Path,
    root: Path | None = None,
    configs: Iterable[QuantBenchPathSelectorConfig],
    hazard_score: pd.Series | None = None,
    gate: QuantBenchGate = QuantBenchGate(),
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] | None = None,
    cost: float = 0.0008,
) -> dict[str, Any]:
    """Persist a QuantBench-compatible macro path-ensemble screen."""
    root = root.resolve() if root is not None else None
    card_dir = card_dir.resolve()
    card_dir.mkdir(parents=True, exist_ok=True)

    payloads, daily_by_profile = build_quantbench_path_ensemble_payloads(
        close,
        paths,
        risk_frame,
        configs=configs,
        hazard_score=hazard_score,
        windows=windows,
        cost=cost,
    )
    records = [
        summary_record_from_payload(payload, source_path=f"generated:{payload['profile']}")
        for payload in payloads
    ]
    screen_rows = screen_quantbench_candidates(records, gate=gate)
    eligible_rows = [row for row in screen_rows if row["eligible"]]

    payloads_path = card_dir / "ensemble_payloads.json"
    screen_path = card_dir / "ensemble_candidate_screen.csv"
    summary_path = card_dir / "ensemble_summary.json"
    payloads_path.write_text(json.dumps(payloads, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    _write_rows_csv(screen_path, screen_rows)
    for profile, daily in daily_by_profile.items():
        daily.to_csv(card_dir / f"{profile}_soxl_daily.csv", index=False)

    def rel(path: Path) -> str:
        return _display_path(path, root=root)

    report = {
        "paper": {
            "title": "QuantBench: Benchmarking AI Methods for Quantitative Investment",
            "arxiv_pdf": "https://arxiv.org/pdf/2504.18600",
            "mapped_claim": "seed/model ensemble robustness with macro-risk regime selection",
        },
        "data": {
            "path_count": len(paths),
            "selector_count": len(payloads),
            "start": str(pd.to_datetime(close.index).min().date()),
            "end": str(pd.to_datetime(close.index).max().date()),
        },
        "gate": gate.__dict__,
        "artifacts": {
            "summary_json": rel(summary_path),
            "ensemble_candidate_screen_csv": rel(screen_path),
            "ensemble_payloads_json": rel(payloads_path),
            "daily_csvs": [rel(card_dir / f"{profile}_soxl_daily.csv") for profile in daily_by_profile],
        },
        "screen": {
            "records": len(records),
            "eligible_count": len(eligible_rows),
            "eligible_profiles": [row["profile"] for row in eligible_rows],
            "best_rows": screen_rows[:10],
        },
        "decision": "eligible_ensemble_candidates_found"
        if eligible_rows
        else "no_ensemble_model_satisfies_quantbench_soxl_20dd_gate",
        "next_action": (
            "Promote only an eligible ensemble; otherwise inspect regime audit columns "
            "and adjust reusable path selection logic before another screen."
        ),
    }
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    return report


def _daily_index_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    if isinstance(frame.index, pd.MultiIndex) and "instrument" in frame.index.names:
        frame = frame.droplevel("instrument")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _daily_price_series(close: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(close, pd.DataFrame):
        if "price" in close.columns:
            close = close["price"]
        elif "close" in close.columns:
            close = close["close"]
        else:
            close = close.iloc[:, 0]
    series = close.copy()
    if isinstance(series.index, pd.MultiIndex) and "instrument" in series.index.names:
        series = series.droplevel("instrument")
    series.index = pd.to_datetime(series.index)
    return series.astype(float).sort_index()


def _supervised_path_name(profile_prefix: str, model_name: str, risk_config: RiskAwarePositionConfig | None = None) -> str:
    suffix = f"_{risk_config.name}" if risk_config is not None else ""
    return f"{profile_prefix}_{model_name}{suffix}"


def build_supervised_path_library(
    predictions: Mapping[str, pd.Series],
    *,
    risk_frame: pd.DataFrame,
    profile_prefix: str = "quantbench_supervised",
    risk_configs: Iterable[RiskAwarePositionConfig] = (),
    min_position: float = 0.10,
    max_position: float = 0.90,
) -> dict[str, pd.Series]:
    """Create reusable daily position paths from supervised model predictions."""
    paths: dict[str, pd.Series] = {}
    for model_name, prediction in predictions.items():
        paths[_supervised_path_name(profile_prefix, model_name)] = prediction_rank_position_series(
            prediction,
            min_position=min_position,
            max_position=max_position,
        )
        for risk_config in tuple(risk_configs):
            paths[_supervised_path_name(profile_prefix, model_name, risk_config)] = risk_aware_position_series(
                prediction.dropna(),
                risk_frame=risk_frame,
                config=risk_config,
                min_position=min_position,
                max_position=max_position,
            )
    return paths


def default_quantbench_path_selector_configs(
    profile_prefix: str = "quantbench_supervised",
) -> list[QuantBenchPathSelectorConfig]:
    """Return a bounded default screen of defensive/offensive supervised path pairs."""

    def path(model_name: str, overlay: str | None = None) -> str:
        suffix = f"_{overlay}" if overlay else ""
        return f"{profile_prefix}_{model_name}{suffix}"

    return [
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_lowdd_mlp_v1",
            defensive_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            neutral_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            offensive_path=path("mlp_mse"),
            recovery_path=path("mlp_mse"),
            stress_floor=0.10,
            stress_cap=0.16,
            neutral_cap=0.32,
            bull_floor=0.65,
            bull_cap=0.92,
            recovery_floor=0.35,
            recovery_cap=0.60,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_lowdd_ridge_v1",
            defensive_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            neutral_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            offensive_path=path("ridge_mse"),
            recovery_path=path("ridge_mse"),
            stress_floor=0.10,
            stress_cap=0.16,
            neutral_cap=0.32,
            bull_floor=0.60,
            bull_cap=0.90,
            recovery_floor=0.32,
            recovery_cap=0.58,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_extra_logistic_v1",
            defensive_path=path("extra_trees_rank", "macro_budget_lowdd_v1"),
            neutral_path=path("extra_trees_rank", "macro_budget_lowdd_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.18,
            neutral_cap=0.35,
            bull_floor=0.62,
            bull_cap=0.92,
            recovery_floor=0.35,
            recovery_cap=0.62,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_defensive_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.20,
            neutral_cap=0.40,
            bull_floor=0.62,
            bull_cap=0.92,
            recovery_floor=0.35,
            recovery_cap=0.62,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_guarded_rf_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.14,
            neutral_cap=0.20,
            bull_floor=0.70,
            bull_cap=0.98,
            recovery_floor=0.50,
            recovery_cap=0.85,
            stress_risk_threshold=0.92,
            stress_drop_threshold=-0.35,
            stress_drawdown_threshold=-0.45,
            bull_risk_threshold=0.50,
            bull_momentum_threshold=0.00,
            bull_trend_threshold=-0.05,
            bull_drawdown_floor=-0.30,
            recovery_risk_threshold=0.75,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.08,
            recovery_drawdown_min=-0.45,
            recovery_drawdown_max=-0.02,
            portfolio_drawdown_trigger=0.15,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.05,
            portfolio_reentry_floor=0.40,
            portfolio_reentry_cap=0.65,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_guarded_hist_logistic_v1",
            defensive_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            neutral_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.14,
            neutral_cap=0.20,
            bull_floor=0.70,
            bull_cap=0.98,
            recovery_floor=0.50,
            recovery_cap=0.85,
            stress_risk_threshold=0.92,
            stress_drop_threshold=-0.35,
            stress_drawdown_threshold=-0.45,
            bull_risk_threshold=0.50,
            bull_momentum_threshold=0.00,
            bull_trend_threshold=-0.05,
            bull_drawdown_floor=-0.30,
            recovery_risk_threshold=0.75,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.08,
            recovery_drawdown_min=-0.45,
            recovery_drawdown_max=-0.02,
            portfolio_drawdown_trigger=0.15,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.05,
            portfolio_reentry_floor=0.40,
            portfolio_reentry_cap=0.65,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_vol_budget_rf_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.05,
            stress_cap=0.18,
            neutral_cap=0.35,
            bull_floor=0.55,
            bull_cap=0.95,
            recovery_floor=0.30,
            recovery_cap=0.70,
            stress_risk_threshold=0.68,
            stress_drop_threshold=-0.14,
            stress_drawdown_threshold=-0.18,
            bull_risk_threshold=0.42,
            bull_momentum_threshold=0.02,
            bull_trend_threshold=0.00,
            bull_drawdown_floor=-0.15,
            recovery_risk_threshold=0.60,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.04,
            recovery_drawdown_min=-0.30,
            recovery_drawdown_max=-0.04,
            portfolio_drawdown_trigger=0.10,
            portfolio_drawdown_cap=0.10,
            portfolio_drawdown_release=0.04,
            portfolio_reentry_floor=0.30,
            portfolio_reentry_cap=0.55,
            portfolio_reentry_max_drawdown=0.18,
            confirmed_reentry_floor=0.30,
            confirmed_reentry_cap=0.45,
            confirmed_reentry_ma_window=50,
            confirmed_reentry_momentum_window=21,
            confirmed_reentry_high_window=50,
            confirmed_reentry_momentum_threshold=0.03,
            confirmed_reentry_drawdown_floor=-0.10,
            confirmed_reentry_volatility_window=20,
            confirmed_reentry_volatility_max=1.50,
            hazard_threshold=0.85,
            hazard_cap=0.05,
            volatility_target=0.30,
            volatility_stress_target=0.16,
            volatility_neutral_target=0.26,
            volatility_recovery_target=0.38,
            volatility_bull_target=0.58,
            volatility_lookback=20,
            volatility_min_periods=10,
            volatility_floor=0.05,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_confirmed_reentry_rf_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.20,
            neutral_cap=0.40,
            bull_floor=0.55,
            bull_cap=0.95,
            recovery_floor=0.30,
            recovery_cap=0.75,
            stress_risk_threshold=0.68,
            stress_drop_threshold=-0.14,
            stress_drawdown_threshold=-0.18,
            bull_risk_threshold=0.45,
            bull_momentum_threshold=0.02,
            bull_trend_threshold=-0.02,
            bull_drawdown_floor=-0.18,
            recovery_risk_threshold=0.65,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.05,
            recovery_drawdown_min=-0.32,
            recovery_drawdown_max=-0.04,
            portfolio_drawdown_trigger=0.10,
            portfolio_drawdown_cap=0.12,
            portfolio_drawdown_release=0.04,
            portfolio_reentry_floor=0.30,
            portfolio_reentry_cap=0.55,
            portfolio_reentry_max_drawdown=0.18,
            confirmed_reentry_floor=0.30,
            confirmed_reentry_cap=0.40,
            confirmed_reentry_ma_window=50,
            confirmed_reentry_momentum_window=21,
            confirmed_reentry_high_window=50,
            confirmed_reentry_momentum_threshold=0.10,
            confirmed_reentry_drawdown_floor=-0.10,
            confirmed_reentry_volatility_window=20,
            confirmed_reentry_volatility_max=1.00,
            hazard_threshold=0.90,
            hazard_cap=0.10,
            volatility_target=0.35,
            volatility_stress_target=0.30,
            volatility_neutral_target=0.32,
            volatility_recovery_target=0.45,
            volatility_bull_target=0.75,
            volatility_lookback=20,
            volatility_min_periods=10,
            volatility_floor=0.10,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_drawdown_budget_guarded_rf_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.14,
            neutral_cap=0.20,
            bull_floor=0.70,
            bull_cap=0.98,
            recovery_floor=0.50,
            recovery_cap=0.85,
            stress_risk_threshold=0.92,
            stress_drop_threshold=-0.35,
            stress_drawdown_threshold=-0.45,
            bull_risk_threshold=0.50,
            bull_momentum_threshold=0.00,
            bull_trend_threshold=-0.05,
            bull_drawdown_floor=-0.30,
            recovery_risk_threshold=0.75,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.08,
            recovery_drawdown_min=-0.45,
            recovery_drawdown_max=-0.02,
            portfolio_drawdown_trigger=0.15,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.05,
            portfolio_reentry_floor=0.40,
            portfolio_reentry_cap=0.65,
            portfolio_drawdown_budget_start=0.06,
            portfolio_drawdown_budget_full=0.16,
            portfolio_drawdown_budget_start_cap=0.45,
            portfolio_drawdown_budget_min_cap=0.02,
            portfolio_drawdown_budget_stress_floor=0.10,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_tail_shock_budget_rf_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.14,
            neutral_cap=0.20,
            bull_floor=0.70,
            bull_cap=0.98,
            recovery_floor=0.50,
            recovery_cap=0.85,
            stress_risk_threshold=0.92,
            stress_drop_threshold=-0.35,
            stress_drawdown_threshold=-0.45,
            bull_risk_threshold=0.50,
            bull_momentum_threshold=0.00,
            bull_trend_threshold=-0.05,
            bull_drawdown_floor=-0.30,
            recovery_risk_threshold=0.75,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.08,
            recovery_drawdown_min=-0.45,
            recovery_drawdown_max=-0.02,
            portfolio_drawdown_trigger=0.15,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.05,
            portfolio_reentry_floor=0.40,
            portfolio_reentry_cap=0.65,
            portfolio_drawdown_budget_start=0.04,
            portfolio_drawdown_budget_full=0.12,
            portfolio_drawdown_budget_start_cap=0.35,
            portfolio_drawdown_budget_min_cap=0.02,
            portfolio_drawdown_budget_stress_floor=0.10,
            tail_shock_day_threshold=-0.04,
            tail_shock_rolling_2d_threshold=-0.075,
            tail_shock_rolling_3d_threshold=-0.12,
            tail_shock_cooldown_days=3,
            tail_shock_cap=0.05,
            tail_shock_stress_floor=0.10,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_tail_shock_bull_release_rf_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.14,
            neutral_cap=0.20,
            bull_floor=0.70,
            bull_cap=0.98,
            recovery_floor=0.50,
            recovery_cap=0.85,
            stress_risk_threshold=0.92,
            stress_drop_threshold=-0.35,
            stress_drawdown_threshold=-0.45,
            bull_risk_threshold=0.50,
            bull_momentum_threshold=0.00,
            bull_trend_threshold=-0.05,
            bull_drawdown_floor=-0.30,
            recovery_risk_threshold=0.75,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.08,
            recovery_drawdown_min=-0.45,
            recovery_drawdown_max=-0.02,
            portfolio_drawdown_trigger=0.15,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.05,
            portfolio_reentry_floor=0.40,
            portfolio_reentry_cap=0.65,
            portfolio_drawdown_budget_start=0.04,
            portfolio_drawdown_budget_full=0.12,
            portfolio_drawdown_budget_start_cap=0.35,
            portfolio_drawdown_budget_min_cap=0.04,
            portfolio_drawdown_budget_stress_floor=0.14,
            portfolio_drawdown_budget_bull_release_cap=0.20,
            portfolio_drawdown_budget_bull_release_max_drawdown=0.12,
            portfolio_drawdown_budget_bull_release_ma_window=200,
            portfolio_drawdown_budget_bull_release_momentum_window=126,
            portfolio_drawdown_budget_bull_release_high_window=126,
            portfolio_drawdown_budget_bull_release_momentum_threshold=0.10,
            portfolio_drawdown_budget_bull_release_drawdown_floor=-0.12,
            portfolio_drawdown_budget_bull_release_volatility_window=20,
            portfolio_drawdown_budget_bull_release_volatility_max=0.80,
            tail_shock_day_threshold=-0.04,
            tail_shock_rolling_2d_threshold=-0.075,
            tail_shock_rolling_3d_threshold=-0.12,
            tail_shock_cooldown_days=3,
            tail_shock_cap=0.05,
            tail_shock_stress_floor=0.10,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_tail_shock_anchor_reset_rf_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.14,
            neutral_cap=0.20,
            bull_floor=0.70,
            bull_cap=0.98,
            recovery_floor=0.50,
            recovery_cap=0.85,
            stress_risk_threshold=0.92,
            stress_drop_threshold=-0.35,
            stress_drawdown_threshold=-0.45,
            bull_risk_threshold=0.50,
            bull_momentum_threshold=0.00,
            bull_trend_threshold=-0.05,
            bull_drawdown_floor=-0.30,
            recovery_risk_threshold=0.75,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.08,
            recovery_drawdown_min=-0.45,
            recovery_drawdown_max=-0.02,
            portfolio_drawdown_trigger=0.15,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.05,
            portfolio_reentry_floor=0.40,
            portfolio_reentry_cap=0.65,
            portfolio_drawdown_budget_start=0.04,
            portfolio_drawdown_budget_full=0.12,
            portfolio_drawdown_budget_start_cap=0.35,
            portfolio_drawdown_budget_min_cap=0.04,
            portfolio_drawdown_budget_stress_floor=0.14,
            portfolio_drawdown_budget_bull_release_cap=0.30,
            portfolio_drawdown_budget_bull_release_max_drawdown=0.20,
            portfolio_drawdown_budget_bull_release_ma_window=200,
            portfolio_drawdown_budget_bull_release_momentum_window=126,
            portfolio_drawdown_budget_bull_release_high_window=126,
            portfolio_drawdown_budget_bull_release_momentum_threshold=0.10,
            portfolio_drawdown_budget_bull_release_drawdown_floor=-0.12,
            portfolio_drawdown_budget_bull_release_volatility_window=20,
            portfolio_drawdown_budget_bull_release_volatility_max=0.80,
            portfolio_drawdown_budget_anchor_reset_drawdown=0.08,
            portfolio_drawdown_budget_anchor_reset_max_drawdown=0.20,
            portfolio_drawdown_budget_anchor_reset_streak_days=1,
            tail_shock_day_threshold=-0.04,
            tail_shock_rolling_2d_threshold=-0.075,
            tail_shock_rolling_3d_threshold=-0.12,
            tail_shock_cooldown_days=3,
            tail_shock_cap=0.05,
            tail_shock_stress_floor=0.10,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_sequence_anchor_reset_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("sequence_lag_ridge_mse"),
            recovery_path=path("sequence_lag_ridge_mse"),
            stress_floor=0.10,
            stress_cap=0.14,
            neutral_cap=0.20,
            bull_floor=0.70,
            bull_cap=0.98,
            recovery_floor=0.50,
            recovery_cap=0.85,
            stress_risk_threshold=0.92,
            stress_drop_threshold=-0.35,
            stress_drawdown_threshold=-0.45,
            bull_risk_threshold=0.50,
            bull_momentum_threshold=0.00,
            bull_trend_threshold=-0.05,
            bull_drawdown_floor=-0.30,
            recovery_risk_threshold=0.75,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.08,
            recovery_drawdown_min=-0.45,
            recovery_drawdown_max=-0.02,
            portfolio_drawdown_trigger=0.15,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.05,
            portfolio_reentry_floor=0.40,
            portfolio_reentry_cap=0.65,
            portfolio_drawdown_budget_start=0.04,
            portfolio_drawdown_budget_full=0.12,
            portfolio_drawdown_budget_start_cap=0.35,
            portfolio_drawdown_budget_min_cap=0.04,
            portfolio_drawdown_budget_stress_floor=0.14,
            portfolio_drawdown_budget_bull_release_cap=0.30,
            portfolio_drawdown_budget_bull_release_max_drawdown=0.20,
            portfolio_drawdown_budget_bull_release_ma_window=200,
            portfolio_drawdown_budget_bull_release_momentum_window=126,
            portfolio_drawdown_budget_bull_release_high_window=126,
            portfolio_drawdown_budget_bull_release_momentum_threshold=0.10,
            portfolio_drawdown_budget_bull_release_drawdown_floor=-0.12,
            portfolio_drawdown_budget_bull_release_volatility_window=20,
            portfolio_drawdown_budget_bull_release_volatility_max=0.80,
            portfolio_drawdown_budget_anchor_reset_drawdown=0.08,
            portfolio_drawdown_budget_anchor_reset_max_drawdown=0.20,
            portfolio_drawdown_budget_anchor_reset_streak_days=1,
            tail_shock_day_threshold=-0.04,
            tail_shock_rolling_2d_threshold=-0.075,
            tail_shock_rolling_3d_threshold=-0.12,
            tail_shock_cooldown_days=3,
            tail_shock_cap=0.05,
            tail_shock_stress_floor=0.10,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_hazard_rf_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.14,
            neutral_cap=0.20,
            bull_floor=0.70,
            bull_cap=0.98,
            recovery_floor=0.50,
            recovery_cap=0.85,
            stress_risk_threshold=0.92,
            stress_drop_threshold=-0.35,
            stress_drawdown_threshold=-0.45,
            bull_risk_threshold=0.50,
            bull_momentum_threshold=0.00,
            bull_trend_threshold=-0.05,
            bull_drawdown_floor=-0.30,
            recovery_risk_threshold=0.75,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.08,
            recovery_drawdown_min=-0.45,
            recovery_drawdown_max=-0.02,
            portfolio_drawdown_trigger=0.15,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.05,
            portfolio_reentry_floor=0.40,
            portfolio_reentry_cap=0.65,
            hazard_threshold=0.80,
            hazard_cap=0.02,
        ),
    ]


def _config_paths_available(config: QuantBenchPathSelectorConfig, paths: Mapping[str, pd.Series]) -> bool:
    required = {
        config.defensive_path,
        config.offensive_path,
        config.neutral_path or config.defensive_path,
        config.recovery_path or config.offensive_path,
    }
    return required <= set(paths)


def build_quantbench_supervised_ensemble_report(
    features: pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    *,
    card_dir: Path,
    root: Path | None = None,
    horizon: int = 20,
    model_names: Iterable[str] | None = None,
    feature_packs: Iterable[str] = (),
    train_start: str | pd.Timestamp = "2010-03-11",
    train_end: str | pd.Timestamp = "2018-12-31",
    test_start: str | pd.Timestamp = "2020-01-01",
    test_end: str | pd.Timestamp = "2026-12-31",
    profile_prefix: str = "quantbench_supervised",
    risk_configs: Iterable[RiskAwarePositionConfig] | None = None,
    ensemble_configs: Iterable[QuantBenchPathSelectorConfig] | None = None,
    gate: QuantBenchGate = QuantBenchGate(),
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] | None = None,
    min_position: float = 0.10,
    max_position: float = 0.90,
    cost: float = 0.0008,
    hazard_enabled: bool = True,
    hazard_mode: str = "rolling",
    hazard_drawdown_threshold: float = -0.18,
    hazard_cap_threshold: float | None = None,
    hazard_refit_step_rows: int = 63,
    hazard_train_lookback_rows: int | None = 1260,
    hazard_min_train_rows: int = 756,
) -> dict[str, Any]:
    """Train supervised paths and run the src-owned macro path ensemble screen."""
    root = root.resolve() if root is not None else None
    card_dir = card_dir.resolve()
    card_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = _daily_index_frame(features).replace([np.inf, -np.inf], np.nan)
    close_series = _daily_price_series(close)
    common_index = feature_frame.index.intersection(close_series.index)
    feature_frame = feature_frame.loc[common_index]
    close_series = close_series.loc[common_index]
    if feature_frame.empty or close_series.empty:
        raise ValueError("features and close must have overlapping rows")

    from .quantbench_objectives import build_soxl_supervised_targets

    targets = build_soxl_supervised_targets(close_series, horizons=(horizon,))
    predictions, objective_metrics = fit_quantbench_supervised_models(
        feature_frame,
        targets,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        horizon=horizon,
        model_names=model_names,
    )
    risk_config_tuple = tuple(quantbench_risk_overlay_configs() if risk_configs is None else risk_configs)
    paths = build_supervised_path_library(
        predictions,
        risk_frame=feature_frame,
        profile_prefix=profile_prefix,
        risk_configs=risk_config_tuple,
        min_position=min_position,
        max_position=max_position,
    )
    config_tuple = tuple(
        default_quantbench_path_selector_configs(profile_prefix=profile_prefix)
        if ensemble_configs is None
        else ensemble_configs
    )
    config_tuple = tuple(config for config in config_tuple if _config_paths_available(config, paths))
    if not config_tuple:
        raise ValueError("no ensemble selector configs match the generated supervised path library")

    hazard_score = None
    hazard_metrics: dict[str, Any] = {}
    if bool(hazard_enabled):
        if hazard_mode == "rolling":
            hazard_score, hazard_metrics = fit_quantbench_rolling_drawdown_hazard_model(
                feature_frame,
                close_series,
                test_start=test_start,
                test_end=test_end,
                horizon=horizon,
                drawdown_threshold=hazard_drawdown_threshold,
                min_train_rows=hazard_min_train_rows,
                refit_step_rows=hazard_refit_step_rows,
                train_lookback_rows=hazard_train_lookback_rows,
            )
        elif hazard_mode == "static":
            hazard_score, hazard_metrics = fit_quantbench_drawdown_hazard_model(
                feature_frame,
                close_series,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                horizon=horizon,
                drawdown_threshold=hazard_drawdown_threshold,
            )
        else:
            raise ValueError("hazard_mode must be 'rolling' or 'static'")
        hazard_metrics["mode"] = str(hazard_mode)
        if hazard_cap_threshold is not None:
            config_tuple = tuple(
                QuantBenchPathSelectorConfig(**{**asdict(config), "hazard_threshold": float(hazard_cap_threshold)})
                if config.hazard_threshold > 0.0
                else config
                for config in config_tuple
            )

    report = build_quantbench_path_ensemble_report(
        close_series,
        paths,
        feature_frame,
        card_dir=card_dir,
        root=root,
        configs=config_tuple,
        hazard_score=hazard_score,
        gate=gate,
        windows=windows,
        cost=cost,
    )
    path_rows = []
    for name in sorted(paths):
        parts = name.removeprefix(f"{profile_prefix}_").split("_macro_budget_", maxsplit=1)
        path_rows.append(
            {
                "path": name,
                "model_name": parts[0],
                "risk_overlay": f"macro_budget_{parts[1]}" if len(parts) == 2 else "",
            }
        )
    path_library_path = card_dir / "ensemble_path_library.csv"
    objective_metrics_path = card_dir / "ensemble_objective_metrics.json"
    hazard_metrics_path = card_dir / "ensemble_hazard_metrics.json"
    _write_rows_csv(path_library_path, path_rows)
    objective_metrics_path.write_text(
        json.dumps(objective_metrics, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    hazard_metrics_path.write_text(
        json.dumps(hazard_metrics, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    def rel(path: Path) -> str:
        return _display_path(path, root=root)

    report["data"].update(
        {
            "feature_count": int(feature_frame.shape[1]),
            "feature_packs": list(feature_packs),
            "model_names": list(predictions.keys()),
            "risk_overlays": [config.name for config in risk_config_tuple],
            "horizon": int(horizon),
            "hazard_enabled": bool(hazard_enabled),
            "hazard_mode": str(hazard_mode),
            "hazard_drawdown_threshold": float(hazard_drawdown_threshold),
            "hazard_refit_step_rows": int(hazard_refit_step_rows),
            "hazard_train_lookback_rows": (
                None if hazard_train_lookback_rows is None else int(hazard_train_lookback_rows)
            ),
            "hazard_min_train_rows": int(hazard_min_train_rows),
            "train_window": [str(pd.Timestamp(train_start).date()), str(pd.Timestamp(train_end).date())],
            "test_window": [str(pd.Timestamp(test_start).date()), str(pd.Timestamp(test_end).date())],
        }
    )
    report["artifacts"]["ensemble_path_library_csv"] = rel(path_library_path)
    report["artifacts"]["ensemble_objective_metrics_json"] = rel(objective_metrics_path)
    report["artifacts"]["ensemble_hazard_metrics_json"] = rel(hazard_metrics_path)
    summary_path = card_dir / "ensemble_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "DEFAULT_WINDOWS",
    "QuantBenchPathSelectorConfig",
    "apply_quantbench_ensemble_drawdown_guard",
    "apply_quantbench_hazard_guard",
    "apply_quantbench_tail_shock_guard",
    "apply_quantbench_volatility_budget_guard",
    "backtest_position_path",
    "build_quantbench_supervised_ensemble_report",
    "build_quantbench_path_ensemble_payloads",
    "build_quantbench_path_ensemble_report",
    "build_supervised_path_library",
    "default_quantbench_path_selector_configs",
    "fit_quantbench_drawdown_hazard_model",
    "fit_quantbench_rolling_drawdown_hazard_model",
    "select_quantbench_path_ensemble",
]

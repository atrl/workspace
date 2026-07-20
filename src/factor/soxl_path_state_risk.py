"""Causal SOXL path-state features with horizon-specific forward labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


PATH_STATE_FEATURES = [
    "trend_permission",
    "fast_turbulence_risk",
    "structural_chop_risk",
]

PATH_STATE_BASELINE_FEATURES = [
    "range_pos_252",
    "low_trend_efficiency_pressure_20",
    "sign_change_density_pressure_20",
    "realized_volatility_rank_20",
    "price_only_score",
]

PATH_STATE_LABELS = ["fast_tail_5d", "structural_break_20d"]

PATH_STATE_FORWARD_METRICS = [
    "forward_terminal_5",
    "forward_mae_5",
    "forward_mfe_5",
    "forward_terminal_20",
    "forward_mae_20",
    "forward_mfe_20",
]

_REQUIRED = {"instrument", "datetime", "close"}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).div(denominator.astype(float).replace(0.0, np.nan))


def _rolling_percentile(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        current = values[-1]
        finite = values[np.isfinite(values)]
        if not np.isfinite(current) or not len(finite):
            return np.nan
        return float(np.mean(finite <= current))

    return series.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def structural_state_age(state: pd.Series) -> pd.Series:
    """Return the causal consecutive-run age of a boolean state."""

    ages: list[int] = []
    age = 0
    for active in state.fillna(False).astype(bool):
        age = age + 1 if active else 0
        ages.append(age)
    return pd.Series(ages, index=state.index, dtype=int)


def forward_path_metrics(
    close: pd.Series,
    *,
    horizon: int,
    adverse_threshold: float,
    terminal_threshold: float,
) -> pd.DataFrame:
    """Measure the path strictly after each current close."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    terminal = np.full(len(values), np.nan)
    mae = np.full(len(values), np.nan)
    mfe = np.full(len(values), np.nan)
    label = np.full(len(values), np.nan)
    for index in range(max(0, len(values) - horizon)):
        current = values[index]
        path_values = values[index + 1 : index + horizon + 1]
        if not np.isfinite(current) or current <= 0.0 or len(path_values) != horizon or not np.isfinite(path_values).all():
            continue
        path = path_values / current - 1.0
        terminal[index] = round(float(path[-1]), 12)
        mae[index] = round(float(path.min()), 12)
        mfe[index] = round(float(path.max()), 12)
        label[index] = float(mae[index] <= adverse_threshold and terminal[index] <= terminal_threshold)
    return pd.DataFrame(
        {
            f"forward_terminal_{horizon}": terminal,
            f"forward_mae_{horizon}": mae,
            f"forward_mfe_{horizon}": mfe,
            f"forward_label_{horizon}": label,
        },
        index=close.index,
    )


def _asset_path_state(
    asset: pd.DataFrame,
    *,
    normalization_window: int,
    min_periods: int,
) -> pd.DataFrame:
    out = asset.sort_values("datetime").copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    log_return = np.log(_safe_divide(close, close.shift(1)))
    absolute_path = log_return.abs().rolling(20, min_periods=20).sum()
    signed_efficiency = _safe_divide(np.log(_safe_divide(close, close.shift(20))), absolute_path).clip(-1.0, 1.0)
    realized_volatility = log_return.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    volatility_rank = _rolling_percentile(
        realized_volatility,
        window=normalization_window,
        min_periods=min_periods,
    ).clip(0.0, 1.0)
    rolling_high = close.rolling(normalization_window, min_periods=min_periods).max()
    rolling_low = close.rolling(normalization_window, min_periods=min_periods).min()
    range_position = _safe_divide(close - rolling_low, rolling_high - rolling_low).clip(0.0, 1.0)

    signs = np.sign(log_return)
    sign_flip = signs.ne(signs.shift()).astype(float)
    sign_flip[log_return.isna() | log_return.shift().isna()] = np.nan
    sign_flip_density = sign_flip.rolling(20, min_periods=20).mean().clip(0.0, 1.0)

    trend_up_strength = ((signed_efficiency - 0.15) / 0.20).clip(0.0, 1.0)
    ordinary_vol_permission = ((0.75 - volatility_rank) / 0.20).clip(0.0, 1.0)
    fast_vol_stress = ((volatility_rank - 0.67) / 0.13).clip(0.0, 1.0)
    structural_state = range_position.ge(0.70) & signed_efficiency.abs().le(0.20) & volatility_rank.ge(0.67)
    state_age = structural_state_age(structural_state)
    high_level_gate = ((range_position - 0.60) / 0.10).clip(0.0, 1.0)
    chop_gate = ((0.25 - signed_efficiency.abs()) / 0.10).clip(0.0, 1.0)
    structural_vol_stress = ((volatility_rank - 0.60) / 0.15).clip(0.0, 1.0)
    persistence_score = (state_age.astype(float) / 4.0).clip(0.0, 1.0)

    out["signed_efficiency_20"] = signed_efficiency
    out["rv_20"] = realized_volatility
    out["rv_rank_20_252"] = volatility_rank
    out["range_pos_252"] = range_position
    out["low_trend_efficiency_pressure_20"] = 1.0 - signed_efficiency.abs()
    out["sign_change_density_pressure_20"] = sign_flip_density
    out["realized_volatility_rank_20"] = volatility_rank
    out["price_only_score"] = out[
        [
            "range_pos_252",
            "low_trend_efficiency_pressure_20",
            "sign_change_density_pressure_20",
            "realized_volatility_rank_20",
        ]
    ].mean(axis=1, skipna=False)
    out["trend_permission"] = (trend_up_strength * ordinary_vol_permission).clip(0.0, 1.0)
    out["fast_turbulence_risk"] = (trend_up_strength * fast_vol_stress).clip(0.0, 1.0)
    out["structural_chop_state"] = structural_state.astype(float)
    out["structural_chop_age"] = state_age
    out["fast_turbulence_event"] = (signed_efficiency.ge(0.25) & volatility_rank.ge(0.75)).astype(float)
    out["structural_chop_event"] = state_age.ge(4).astype(float)
    out["structural_chop_risk"] = (
        high_level_gate * chop_gate * structural_vol_stress * persistence_score
    ).clip(0.0, 1.0)

    fast = forward_path_metrics(
        close,
        horizon=5,
        adverse_threshold=-0.10,
        terminal_threshold=-0.05,
    )
    structural = forward_path_metrics(
        close,
        horizon=20,
        adverse_threshold=-0.20,
        terminal_threshold=-0.10,
    )
    for column in fast:
        out[column] = fast[column].to_numpy()
    for column in structural:
        out[column] = structural[column].to_numpy()
    out["fast_tail_5d"] = out.pop("forward_label_5")
    out["structural_break_20d"] = out.pop("forward_label_20")
    return out


def build_soxl_path_state_frame(
    frame: pd.DataFrame,
    *,
    normalization_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Build causal state features and future path labels per instrument."""

    missing = sorted(_REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"path-state frame missing columns: {missing}")
    if normalization_window < 20 or not 20 <= min_periods <= normalization_window:
        raise ValueError("normalization_window and min_periods are inconsistent")
    working = frame.copy()
    working["instrument"] = working["instrument"].astype(str).str.upper()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="raise").dt.normalize()
    if working.duplicated(["instrument", "datetime"]).any():
        raise ValueError("path-state frame contains duplicate instrument/datetime rows")
    pieces = [
        _asset_path_state(asset, normalization_window=normalization_window, min_periods=min_periods)
        for _, asset in working.groupby("instrument", sort=False)
    ]
    result = pd.concat(pieces, ignore_index=True).sort_values(["instrument", "datetime"]).reset_index(drop=True)
    result.attrs["feature_cols"] = PATH_STATE_FEATURES
    result.attrs["label_cols"] = PATH_STATE_LABELS
    result.attrs["normalization_window"] = int(normalization_window)
    result.attrs["min_periods"] = int(min_periods)
    result.attrs["timing"] = "close_t_features_predict_strictly_after_close_t"
    return result

"""Point-in-time price-volume pressure factors for long 3x equity ETFs."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_ONLY_BASELINE_FEATURES = [
    "range_pos_252",
    "low_trend_efficiency_pressure_20",
    "sign_change_density_pressure_20",
    "high_chop_price_pressure",
]

PRICE_VOLUME_PRESSURE_FEATURES = [
    "effort_result_pressure_20",
    "volume_weighted_clv_pressure_20",
    "failed_breakout_pressure_20",
    "volume_sign_flip_pressure_20",
    "signed_volume_selling_pressure_20",
    "price_volume_divergence_pressure_20",
    "leverage_residual_pressure_20",
    "realized_leverage_shortfall_pressure_60",
]

ALL_FACTOR_FEATURES = PRICE_ONLY_BASELINE_FEATURES + PRICE_VOLUME_PRESSURE_FEATURES

_REQUIRED_COLUMNS = {
    "instrument",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "underlying_close",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).div(denominator.astype(float).replace(0.0, np.nan))


def _rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        current = values[-1]
        finite = values[np.isfinite(values)]
        if not np.isfinite(current) or len(finite) == 0:
            return np.nan
        return float(np.mean(finite <= current))

    return series.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def _asset_factor_frame(
    asset: pd.DataFrame,
    *,
    normalization_window: int,
    min_periods: int,
    label_horizon: int,
) -> pd.DataFrame:
    out = asset.sort_values("datetime").copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").clip(lower=0.0)
    underlying = pd.to_numeric(out["underlying_close"], errors="coerce")

    returns = close.pct_change(fill_method=None)
    underlying_returns = underlying.pct_change(fill_method=None)
    rolling_volume = volume.rolling(20, min_periods=min(min_periods, 20)).median()
    relative_volume = _safe_divide(volume, rolling_volume).clip(0.0, 10.0)

    rolling_high = close.rolling(normalization_window, min_periods=min_periods).max()
    rolling_low = close.rolling(normalization_window, min_periods=min_periods).min()
    range_pos = _safe_divide(close - rolling_low, rolling_high - rolling_low).clip(0.0, 1.0)

    path_length = close.diff().abs().rolling(20, min_periods=min(min_periods, 20)).sum()
    trend_efficiency = _safe_divide(close.diff(20).abs(), path_length).clip(0.0, 1.0)
    signs = np.sign(returns)
    sign_flip = signs.ne(signs.shift()).astype(float)
    sign_flip[returns.isna() | returns.shift().isna()] = np.nan
    sign_flip_density = sign_flip.rolling(20, min_periods=min(min_periods, 20)).mean()
    realized_vol = returns.rolling(20, min_periods=min(min_periods, 20)).std(ddof=0) * np.sqrt(252.0)

    effort = (returns.abs() * relative_volume).rolling(
        20, min_periods=min(min_periods, 20)
    ).sum()
    net_result = close.pct_change(20, fill_method=None).abs()
    result_floor = returns.rolling(20, min_periods=min(min_periods, 20)).std(ddof=0) * 0.25
    effort_result = _safe_divide(effort, net_result + result_floor)

    intraday_range = (high - low).replace(0.0, np.nan)
    clv = _safe_divide(2.0 * close - high - low, intraday_range).clip(-1.0, 1.0)
    weighted_clv = _safe_divide(
        (clv * relative_volume).rolling(20, min_periods=min(min_periods, 20)).sum(),
        relative_volume.rolling(20, min_periods=min(min_periods, 20)).sum(),
    )

    prior_high = high.shift(1).rolling(20, min_periods=min(min_periods, 20)).max()
    failed_breakout = (
        high.gt(prior_high)
        & close.lt(prior_high)
        & prior_high.notna()
    ).astype(float)
    close_off_high = _safe_divide(high - close, intraday_range).clip(0.0, 1.0)
    failed_breakout_score = (failed_breakout * close_off_high * relative_volume).rolling(
        20, min_periods=min(min_periods, 20)
    ).mean()

    volume_sign_flip = _safe_divide(
        (sign_flip * relative_volume).rolling(20, min_periods=min(min_periods, 20)).sum(),
        relative_volume.rolling(20, min_periods=min(min_periods, 20)).sum(),
    )
    signed_volume = _safe_divide(
        (np.sign(returns) * relative_volume).rolling(20, min_periods=min(min_periods, 20)).sum(),
        relative_volume.rolling(20, min_periods=min(min_periods, 20)).sum(),
    )
    ret_20 = close.pct_change(20, fill_method=None)
    price_state = _rolling_percentile(ret_20, normalization_window, min_periods)
    volume_state = _rolling_percentile(signed_volume, normalization_window, min_periods)
    price_volume_divergence = price_state - volume_state

    residual = returns - 3.0 * underlying_returns
    positive_underlying_weight = 1.0 + underlying_returns.gt(0.0).astype(float)
    leverage_residual_pressure = _safe_divide(
        (-residual * relative_volume * positive_underlying_weight).rolling(
            20, min_periods=min(min_periods, 20)
        ).sum(),
        (relative_volume * positive_underlying_weight).rolling(
            20, min_periods=min(min_periods, 20)
        ).sum(),
    )

    beta_window = 60
    covariance = returns.rolling(beta_window, min_periods=min(min_periods, beta_window)).cov(underlying_returns)
    underlying_variance = underlying_returns.rolling(
        beta_window, min_periods=min(min_periods, beta_window)
    ).var(ddof=1)
    realized_beta = _safe_divide(covariance, underlying_variance)
    leverage_shortfall = (3.0 - realized_beta).clip(lower=0.0)
    volume_conditioned_shortfall = leverage_shortfall * relative_volume.rolling(
        20, min_periods=min(min_periods, 20)
    ).mean()

    raw_pressure = {
        "range_pos_252": range_pos,
        "low_trend_efficiency_pressure_20": -trend_efficiency,
        "sign_change_density_pressure_20": sign_flip_density,
        "effort_result_pressure_20": effort_result,
        "volume_weighted_clv_pressure_20": -weighted_clv,
        "failed_breakout_pressure_20": failed_breakout_score,
        "volume_sign_flip_pressure_20": volume_sign_flip,
        "signed_volume_selling_pressure_20": -signed_volume,
        "price_volume_divergence_pressure_20": price_volume_divergence,
        "leverage_residual_pressure_20": leverage_residual_pressure,
        "realized_leverage_shortfall_pressure_60": volume_conditioned_shortfall,
    }

    normalized: dict[str, pd.Series] = {}
    for name, values in raw_pressure.items():
        normalized[name] = _rolling_percentile(values, normalization_window, min_periods)
    # Sparse event pressure needs a true zero floor. A conventional empirical
    # percentile would rank an all-zero history at 1.0, reversing the intended
    # semantics before the first failed breakout occurs.
    normalized["failed_breakout_pressure_20"] = normalized[
        "failed_breakout_pressure_20"
    ].where(failed_breakout_score.gt(0.0), 0.0)

    normalized["high_chop_price_pressure"] = pd.concat(
        [
            normalized["range_pos_252"],
            normalized["low_trend_efficiency_pressure_20"],
            normalized["sign_change_density_pressure_20"],
            _rolling_percentile(realized_vol, normalization_window, min_periods),
        ],
        axis=1,
    ).mean(axis=1, skipna=False)

    # The row at t represents a decision before the t session. Every factor is
    # therefore delayed by one complete trading session.
    for name in ALL_FACTOR_FEATURES:
        out[name] = normalized[name].shift(1).clip(0.0, 1.0)

    out["high_level_state"] = out["range_pos_252"].ge(0.70).astype(float)
    out.loc[out["range_pos_252"].isna(), "high_level_state"] = np.nan
    out["breakdown_label"] = _forward_breakdown_label(
        close,
        returns=returns,
        horizon=label_horizon,
    )
    return out


def _forward_breakdown_label(
    close: pd.Series,
    *,
    returns: pd.Series,
    horizon: int,
) -> np.ndarray:
    close_values = close.to_numpy(dtype=float)
    annualized_sigma = (
        returns.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    ).to_numpy(dtype=float)
    labels = np.full(len(close_values), np.nan)
    usable_end = max(0, len(close_values) - horizon - 1)
    for index in range(usable_end):
        sigma = annualized_sigma[index]
        if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(close_values[index]):
            continue
        path = close_values[index + 1 : index + horizon + 1] / close_values[index] - 1.0
        if len(path) != horizon or not np.isfinite(path).all():
            continue
        scale = sigma * np.sqrt(horizon / 252.0)
        terminal = path[-1]
        maximum_adverse_excursion = float(path.min())
        labels[index] = float(
            maximum_adverse_excursion <= -scale and terminal <= -0.5 * scale
        )
    return labels


def build_multi_3x_price_volume_frame(
    panel: pd.DataFrame,
    *,
    label_horizon: int = 20,
    normalization_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Build asset-local, lagged factor scores and a forward breakdown label."""

    missing = sorted(_REQUIRED_COLUMNS.difference(panel.columns))
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    if label_horizon < 1:
        raise ValueError("label_horizon must be positive")
    if normalization_window < 2:
        raise ValueError("normalization_window must be at least 2")
    if min_periods < 2 or min_periods > normalization_window:
        raise ValueError("min_periods must be between 2 and normalization_window")

    working = panel.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="raise")
    if working.duplicated(["instrument", "datetime"]).any():
        raise ValueError("panel contains duplicate instrument/datetime rows")
    pieces = [
        _asset_factor_frame(
            asset,
            normalization_window=normalization_window,
            min_periods=min_periods,
            label_horizon=label_horizon,
        )
        for _, asset in working.groupby("instrument", sort=False)
    ]
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["datetime", "instrument"])
        .reset_index(drop=True)
    )

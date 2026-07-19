"""Transparent, point-in-time factor families for SOXL breakdown diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_PATH_EXHAUSTION_FEATURES = [
    "failed_breakout_persistence_10",
    "high_level_duration_60",
    "efficiency_collapse_20",
    "upside_progress_per_range_pressure_20",
    "downside_range_skew_20",
    "close_location_decay_5",
]

LEVERAGE_FRAGILITY_FEATURES = [
    "cumulative_leverage_residual_pressure_20",
    "up_day_leverage_shortfall_20",
    "asymmetric_beta_pressure_60",
    "volatility_drag_pressure_20",
    "relative_strength_breakdown_20",
]

FIXED_BELLWETHERS = ("NVDA", "AVGO", "AMD", "TSM", "ASML")

STRUCTURE_FEATURES = [
    "fixed_bellwether_residual_dispersion_20",
    "fixed_bellwether_leadership_concentration_20",
    "fixed_bellwether_breadth_divergence_20",
    "fixed_bellwether_new_high_participation_decay_20",
    "fixed_bellwether_downside_breadth_5",
]

_PAIR_REQUIRED = {
    "instrument",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "underlying_close",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).div(denominator.astype(float).replace(0.0, np.nan))


def _rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        current = values[-1]
        finite = values[np.isfinite(values)]
        if not np.isfinite(current) or not len(finite):
            return np.nan
        return float(np.mean(finite <= current))

    return series.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def _conditional_beta(
    dependent: pd.Series,
    independent: pd.Series,
    condition: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    x = independent.where(condition)
    y = dependent.where(condition)
    count = condition.astype(float).rolling(window, min_periods=1).sum()
    xy = (x * y).rolling(window, min_periods=1).sum()
    xx = x.pow(2).rolling(window, min_periods=1).sum()
    beta = _safe_divide(xy, xx)
    return beta.where(count >= max(8, min_periods // 3))


def _asset_exhaustion_fragility(
    asset: pd.DataFrame,
    *,
    normalization_window: int,
    min_periods: int,
) -> pd.DataFrame:
    out = asset.sort_values("datetime").copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    underlying = pd.to_numeric(out["underlying_close"], errors="coerce")
    returns = close.pct_change(fill_method=None)
    underlying_returns = underlying.pct_change(fill_method=None)

    rolling_high = close.rolling(normalization_window, min_periods=min_periods).max()
    rolling_low = close.rolling(normalization_window, min_periods=min_periods).min()
    range_position = _safe_divide(close - rolling_low, rolling_high - rolling_low).clip(0.0, 1.0)

    prior_high = high.shift(1).rolling(20, min_periods=min(20, min_periods)).max()
    failed_breakout = (high.gt(prior_high) & close.lt(prior_high) & prior_high.notna()).astype(float)
    failed_persistence = failed_breakout.rolling(10, min_periods=min(5, min_periods)).mean()
    high_duration = range_position.ge(0.70).astype(float).rolling(
        60, min_periods=min(20, min_periods)
    ).mean()

    absolute_path = returns.abs().rolling(20, min_periods=min(20, min_periods)).sum()
    trend_efficiency = _safe_divide(close.pct_change(20, fill_method=None).abs(), absolute_path).clip(0.0, 1.0)
    efficiency_collapse = 1.0 - trend_efficiency
    upside_progress = close.pct_change(20, fill_method=None).clip(lower=0.0)
    upside_progress_pressure = 1.0 - _safe_divide(upside_progress, absolute_path).clip(0.0, 1.0)

    session_range = _safe_divide(high - low, close.shift(1)).clip(lower=0.0)
    downside_range = session_range.where(returns.lt(0.0), 0.0).rolling(
        20, min_periods=min(20, min_periods)
    ).sum()
    total_range = session_range.rolling(20, min_periods=min(20, min_periods)).sum()
    downside_range_skew = _safe_divide(downside_range, total_range).clip(0.0, 1.0)

    close_location = _safe_divide(close - low, high - low).clip(0.0, 1.0)
    close_location_decay = (
        close_location.rolling(20, min_periods=min(20, min_periods)).mean()
        - close_location.rolling(5, min_periods=min(5, min_periods)).mean()
    ).clip(lower=0.0)

    leverage_residual = returns - 3.0 * underlying_returns
    cumulative_residual_pressure = -leverage_residual.rolling(
        20, min_periods=min(20, min_periods)
    ).sum()
    realized_multiple = _safe_divide(returns, underlying_returns)
    up_day_shortfall = (3.0 - realized_multiple).clip(lower=0.0).where(underlying_returns.gt(0.0))
    up_day_shortfall = up_day_shortfall.rolling(20, min_periods=max(3, min(8, min_periods))).mean()

    up_beta = _conditional_beta(
        returns,
        underlying_returns,
        underlying_returns.gt(0.0),
        window=60,
        min_periods=min_periods,
    )
    down_beta = _conditional_beta(
        returns,
        underlying_returns,
        underlying_returns.lt(0.0),
        window=60,
        min_periods=min_periods,
    )
    asymmetric_beta = (down_beta - up_beta).clip(lower=0.0)

    etf_path = close.pct_change(20, fill_method=None)
    underlying_path = underlying.pct_change(20, fill_method=None)
    volatility_drag = (3.0 * underlying_path - etf_path).clip(lower=0.0)
    relative_strength = np.log(close) - 3.0 * np.log(underlying)
    relative_strength_breakdown = (-relative_strength.diff(20)).clip(lower=0.0)

    raw = {
        "failed_breakout_persistence_10": failed_persistence,
        "high_level_duration_60": high_duration,
        "efficiency_collapse_20": efficiency_collapse,
        "upside_progress_per_range_pressure_20": upside_progress_pressure,
        "downside_range_skew_20": downside_range_skew,
        "close_location_decay_5": close_location_decay,
        "cumulative_leverage_residual_pressure_20": cumulative_residual_pressure,
        "up_day_leverage_shortfall_20": up_day_shortfall,
        "asymmetric_beta_pressure_60": asymmetric_beta,
        "volatility_drag_pressure_20": volatility_drag,
        "relative_strength_breakdown_20": relative_strength_breakdown,
    }
    for name, values in raw.items():
        normalized = _rolling_percentile(values, normalization_window, min_periods)
        if name == "failed_breakout_persistence_10":
            normalized = normalized.where(values.gt(0.0), 0.0)
        out[name] = normalized.shift(1).clip(0.0, 1.0)
    return out


def build_exhaustion_fragility_frame(
    panel: pd.DataFrame,
    *,
    normalization_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Build lagged asset-local price-path and leverage-fragility factors."""

    missing = sorted(_PAIR_REQUIRED.difference(panel.columns))
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    if normalization_window < 2 or not 2 <= min_periods <= normalization_window:
        raise ValueError("normalization_window and min_periods are inconsistent")
    working = panel.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="raise")
    if working.duplicated(["instrument", "datetime"]).any():
        raise ValueError("panel contains duplicate instrument/datetime rows")
    pieces = [
        _asset_exhaustion_fragility(
            asset,
            normalization_window=normalization_window,
            min_periods=min_periods,
        )
        for _, asset in working.groupby("instrument", sort=False)
    ]
    return pd.concat(pieces, ignore_index=True).sort_values(["datetime", "instrument"]).reset_index(drop=True)


def build_fixed_bellwether_structure_frame(
    bellwether_panel: pd.DataFrame,
    soxx_frame: pd.DataFrame,
    *,
    normalization_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Build a lagged equal-weight proxy from a frozen five-name bellwether list."""

    required = {"instrument", "datetime", "close"}
    missing = sorted(required.difference(bellwether_panel.columns))
    if missing:
        raise ValueError(f"bellwether panel missing columns: {missing}")
    if not {"datetime", "soxx_close"}.issubset(soxx_frame.columns):
        raise ValueError("soxx frame must contain datetime and soxx_close")
    panel = bellwether_panel.copy()
    panel["instrument"] = panel["instrument"].astype(str).str.upper()
    panel["datetime"] = pd.to_datetime(panel["datetime"], errors="raise").dt.normalize()
    available = set(panel["instrument"].unique())
    missing_symbols = sorted(set(FIXED_BELLWETHERS).difference(available))
    if missing_symbols:
        raise ValueError(f"fixed bellwether symbols missing: {missing_symbols}")
    panel = panel.loc[panel["instrument"].isin(FIXED_BELLWETHERS)]
    if panel.duplicated(["instrument", "datetime"]).any():
        raise ValueError("bellwether panel contains duplicate instrument/datetime rows")

    prices = panel.pivot(index="datetime", columns="instrument", values="close").reindex(columns=FIXED_BELLWETHERS)
    prices = prices.apply(pd.to_numeric, errors="coerce")
    returns = prices.pct_change(fill_method=None)
    equal_return = returns.mean(axis=1, skipna=False)
    residual_dispersion = returns.sub(equal_return, axis=0).std(axis=1, ddof=0).rolling(
        20, min_periods=min(20, min_periods)
    ).mean()
    # Use absolute return contribution so concentration remains defined during
    # synchronized selloffs; positive-only contribution created long all-null
    # runs precisely in the stressed regimes this factor is meant to describe.
    absolute_contribution = returns.abs()
    leadership = absolute_contribution.max(axis=1).div(
        absolute_contribution.sum(axis=1).replace(0.0, np.nan)
    ).rolling(
        20, min_periods=min(20, min_periods)
    ).mean()
    ret20 = prices.pct_change(20, fill_method=None)
    breadth20 = ret20.gt(0.0).mean(axis=1)

    soxx = soxx_frame.copy()
    soxx["datetime"] = pd.to_datetime(soxx["datetime"], errors="raise").dt.normalize()
    soxx = pd.to_numeric(soxx.set_index("datetime")["soxx_close"], errors="coerce").reindex(prices.index)
    soxx_positive_state = soxx.pct_change(20, fill_method=None).clip(lower=0.0)
    breadth_divergence = soxx_positive_state * (1.0 - breadth20)

    trailing_high = prices.shift(1).rolling(60, min_periods=min(40, min_periods)).max()
    new_high_participation = prices.ge(trailing_high).mean(axis=1)
    participation_peak = new_high_participation.rolling(20, min_periods=min(10, min_periods)).max()
    participation_decay = (participation_peak - new_high_participation).clip(lower=0.0)
    downside_breadth = prices.pct_change(5, fill_method=None).lt(0.0).mean(axis=1)

    raw = {
        "fixed_bellwether_residual_dispersion_20": residual_dispersion,
        "fixed_bellwether_leadership_concentration_20": leadership,
        "fixed_bellwether_breadth_divergence_20": breadth_divergence,
        "fixed_bellwether_new_high_participation_decay_20": participation_decay,
        "fixed_bellwether_downside_breadth_5": downside_breadth,
    }
    out = pd.DataFrame({"datetime": prices.index})
    for name, values in raw.items():
        out[name] = _rolling_percentile(values, normalization_window, min_periods).shift(1).clip(0.0, 1.0).to_numpy()
    out.attrs["membership_contract"] = "fixed_bellwether_proxy_not_historical_constituents"
    out.attrs["fixed_bellwethers"] = list(FIXED_BELLWETHERS)
    return out.reset_index(drop=True)

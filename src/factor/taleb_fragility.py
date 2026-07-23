from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


EMPIRICAL_TAIL_FEATURES = (
    "soxx_tail_xi_504",
    "soxx_tail_xi_instability",
    "soxx_es_var_ratio_504",
    "soxx_tail_loss_concentration_252",
    "soxx_exceedance_intensity_60",
    "soxx_exceedance_ewm",
    "soxx_downside_semivariance_share_20",
    "soxx_jump_ratio_20",
)

IMPLIED_TAIL_FEATURES = (
    "vix9d_vix3m_z_126",
    "vvix_vix_z_126",
    "skew_z_252",
    "tail_risk_premium_9d",
)

STRUCTURAL_FRAGILITY_FEATURES = (
    "soxl_fragility_curvature_504",
    "soxl_downside_beta_gap_252",
    "soxx_rebalance_drag_pressure_20",
    "soxl_levered_decay_acceleration_5",
)

TALEB_FRAGILITY_FEATURES = (
    *EMPIRICAL_TAIL_FEATURES,
    *IMPLIED_TAIL_FEATURES,
    *STRUCTURAL_FRAGILITY_FEATURES,
)

FEATURE_DEFINITIONS: Mapping[str, str] = {
    "soxx_tail_xi_504": (
        "rolling 504d lower-tail Hill xi using the largest 10% losses"
    ),
    "soxx_tail_xi_instability": (
        "absolute disagreement between rolling 252d and 756d Hill xi"
    ),
    "soxx_es_var_ratio_504": (
        "rolling 504d absolute expected-shortfall to VaR ratio at 5%"
    ),
    "soxx_tail_loss_concentration_252": (
        "five largest rolling 252d SOXX losses divided by all losses"
    ),
    "soxx_exceedance_intensity_60": (
        "rolling 60d frequency below the lagged rolling 252d 5% quantile"
    ),
    "soxx_exceedance_ewm": (
        "20d-span exponentially weighted lagged-tail exceedance intensity"
    ),
    "soxx_downside_semivariance_share_20": (
        "rolling 20d negative squared returns divided by total squared returns"
    ),
    "soxx_jump_ratio_20": (
        "rolling 20d non-negative daily RV minus bipower variation share"
    ),
    "vix9d_vix3m_z_126": (
        "rolling 126d z-score of VIX9D divided by VIX3M"
    ),
    "vvix_vix_z_126": (
        "rolling 126d z-score of VVIX divided by VIX"
    ),
    "skew_z_252": "rolling 252d z-score of Cboe SKEW",
    "tail_risk_premium_9d": (
        "VIX9D annualized variance minus rolling 9d SOXX realized variance"
    ),
    "soxl_fragility_curvature_504": (
        "rolling downside loss acceleration across 0-1, 1-2, and >2 sigma "
        "SOXX shock buckets"
    ),
    "soxl_downside_beta_gap_252": (
        "rolling 252d SOXL beta on negative SOXX days minus unconditional beta"
    ),
    "soxx_rebalance_drag_pressure_20": (
        "3 times rolling 20d SOXX squared-return sum, the 3x path drag proxy"
    ),
    "soxl_levered_decay_acceleration_5": (
        "five-day change in rolling 20d SOXL return minus 3x SOXX return"
    ),
}


def _normalize_external(
    frame: pd.DataFrame, *, value_col: str, output_col: str
) -> pd.DataFrame:
    date_columns = [
        column
        for column in frame.columns
        if column.lower() in {"date", "datetime", "observation_date"}
    ]
    if not date_columns:
        raise ValueError(f"{output_col} frame requires a date column")
    if value_col not in frame:
        raise ValueError(f"{output_col} frame missing {value_col}")
    result = frame.loc[:, [date_columns[0], value_col]].copy()
    result["datetime"] = pd.to_datetime(
        result[date_columns[0]], errors="raise"
    ).dt.normalize()
    result[output_col] = pd.to_numeric(
        result[value_col], errors="coerce"
    )
    return (
        result.loc[:, ["datetime", output_col]]
        .dropna(subset=["datetime"])
        .sort_values("datetime")
        .groupby("datetime", as_index=False)
        .last()
    )


def _rolling_zscore(
    values: pd.Series, *, window: int, min_periods: int
) -> pd.Series:
    rolling = values.rolling(window, min_periods=min_periods)
    return (values - rolling.mean()).div(
        rolling.std(ddof=1).replace(0.0, np.nan)
    )


def _hill_xi(values: np.ndarray) -> float:
    losses = -np.asarray(values, dtype=float)
    losses = losses[np.isfinite(losses) & (losses > 0.0)]
    if len(losses) < 40:
        return float("nan")
    ordered = np.sort(losses)[::-1]
    k = int(np.clip(round(len(ordered) * 0.10), 15, 50))
    if len(ordered) <= k:
        return float("nan")
    threshold = ordered[k]
    if threshold <= 0.0:
        return float("nan")
    xi = np.log(ordered[:k] / threshold).mean()
    return float(np.clip(xi, 0.0, 1.5))


def _es_var_ratio(values: np.ndarray) -> float:
    returns = np.asarray(values, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 100:
        return float("nan")
    var = float(np.quantile(returns, 0.05))
    tail = returns[returns <= var]
    if not len(tail) or var >= 0.0:
        return float("nan")
    return float(np.clip(abs(tail.mean()) / abs(var), 1.0, 5.0))


def _loss_concentration(values: np.ndarray) -> float:
    losses = np.maximum(-np.asarray(values, dtype=float), 0.0)
    losses = losses[np.isfinite(losses)]
    total = losses.sum()
    if len(losses) < 50 or total <= 0.0:
        return float("nan")
    largest = np.sort(losses)[-min(5, len(losses)) :]
    return float(np.clip(largest.sum() / total, 0.0, 1.0))


def _rolling_beta(
    asset: pd.Series, benchmark: pd.Series, *, window: int
) -> tuple[pd.Series, pd.Series]:
    covariance = asset.rolling(window, min_periods=126).cov(benchmark)
    variance = benchmark.rolling(window, min_periods=126).var(ddof=1)
    unconditional = covariance.div(variance.replace(0.0, np.nan))

    downside_asset = asset.where(benchmark.lt(0.0))
    downside_benchmark = benchmark.where(benchmark.lt(0.0))
    downside_covariance = downside_asset.rolling(
        window, min_periods=63
    ).cov(downside_benchmark)
    downside_variance = downside_benchmark.rolling(
        window, min_periods=63
    ).var(ddof=1)
    downside = downside_covariance.div(
        downside_variance.replace(0.0, np.nan)
    )
    return unconditional, downside


def _fragility_curvature(
    soxl_return: pd.Series,
    soxx_return: pd.Series,
    *,
    window: int = 504,
    min_periods: int = 252,
) -> pd.Series:
    values = np.full(len(soxx_return), np.nan, dtype=float)
    soxl_values = soxl_return.to_numpy(dtype=float)
    soxx_values = soxx_return.to_numpy(dtype=float)
    for end in range(min_periods - 1, len(soxx_values)):
        start = max(0, end - window + 1)
        x = soxx_values[start : end + 1]
        y = soxl_values[start : end + 1]
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(x) < min_periods:
            continue
        scale = np.std(x, ddof=1)
        loss_scale = np.std(y, ddof=1)
        if scale <= 0.0 or loss_scale <= 0.0:
            continue
        z = x / scale
        losses = -y
        buckets = (
            losses[(z < 0.0) & (z >= -1.0)],
            losses[(z < -1.0) & (z >= -2.0)],
            losses[z < -2.0],
        )
        if any(len(bucket) < 5 for bucket in buckets):
            continue
        bucket_means = [float(bucket.mean()) for bucket in buckets]
        values[end] = (
            bucket_means[2] - 2.0 * bucket_means[1] + bucket_means[0]
        ) / loss_scale
    return pd.Series(values, index=soxx_return.index)


def build_taleb_fragility_feature_frame(
    *,
    daily: pd.DataFrame,
    vix9d: pd.DataFrame,
    vix3m: pd.DataFrame,
    vvix: pd.DataFrame,
    skew: pd.DataFrame,
) -> pd.DataFrame:
    """Build point-in-time empirical, implied, and structural tail features."""

    required = {
        "datetime",
        "soxl_close",
        "soxx_close",
        "vix",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily frame missing columns: {missing}")
    result = daily.copy()
    result["datetime"] = pd.to_datetime(
        result["datetime"], errors="raise"
    ).dt.normalize()
    result = result.sort_values("datetime").drop_duplicates(
        "datetime", keep="last"
    )
    for source, value_col, output_col in (
        (vix9d, "CLOSE", "vix9d"),
        (vix3m, "CLOSE", "vix3m"),
        (vvix, "VVIX", "vvix"),
        (skew, "SKEW", "skew"),
    ):
        result = result.merge(
            _normalize_external(
                source, value_col=value_col, output_col=output_col
            ),
            on="datetime",
            how="left",
        )
    for column in ("vix9d", "vix3m", "vvix", "skew"):
        result[column] = result[column].ffill()

    soxx_return = pd.to_numeric(
        result["soxx_close"], errors="coerce"
    ).pct_change(fill_method=None)
    soxl_return = pd.to_numeric(
        result["soxl_close"], errors="coerce"
    ).pct_change(fill_method=None)

    xi_252 = soxx_return.rolling(252, min_periods=126).apply(
        _hill_xi, raw=True
    )
    xi_504 = soxx_return.rolling(504, min_periods=252).apply(
        _hill_xi, raw=True
    )
    xi_756 = soxx_return.rolling(756, min_periods=378).apply(
        _hill_xi, raw=True
    )
    result["soxx_tail_xi_504"] = xi_504
    result["soxx_tail_xi_instability"] = (xi_252 - xi_756).abs()
    result["soxx_es_var_ratio_504"] = soxx_return.rolling(
        504, min_periods=252
    ).apply(_es_var_ratio, raw=True)
    result["soxx_tail_loss_concentration_252"] = soxx_return.rolling(
        252, min_periods=126
    ).apply(_loss_concentration, raw=True)

    lagged_quantile = soxx_return.shift(1).rolling(
        252, min_periods=126
    ).quantile(0.05)
    exceedance = soxx_return.lt(lagged_quantile).astype(float)
    exceedance.loc[lagged_quantile.isna()] = np.nan
    result["soxx_exceedance_intensity_60"] = exceedance.rolling(
        60, min_periods=20
    ).mean()
    result["soxx_exceedance_ewm"] = exceedance.ewm(
        span=20, min_periods=20, adjust=False
    ).mean()

    squared = soxx_return.pow(2)
    downside_squared = soxx_return.clip(upper=0.0).pow(2)
    result["soxx_downside_semivariance_share_20"] = (
        downside_squared.rolling(20, min_periods=10).sum().div(
            squared.rolling(20, min_periods=10).sum().replace(0.0, np.nan)
        )
    )
    realized_variation = squared.rolling(20, min_periods=10).sum()
    bipower_variation = (
        (np.pi / 2.0)
        * soxx_return.abs()
        * soxx_return.shift(1).abs()
    ).rolling(20, min_periods=10).sum()
    result["soxx_jump_ratio_20"] = (
        (realized_variation - bipower_variation)
        .clip(lower=0.0)
        .div(realized_variation.replace(0.0, np.nan))
        .clip(0.0, 1.0)
    )

    vix9d_vix3m = result["vix9d"].div(
        result["vix3m"].replace(0.0, np.nan)
    )
    vvix_vix = result["vvix"].div(
        pd.to_numeric(result["vix"], errors="coerce").replace(0.0, np.nan)
    )
    result["vix9d_vix3m_z_126"] = _rolling_zscore(
        vix9d_vix3m, window=126, min_periods=63
    )
    result["vvix_vix_z_126"] = _rolling_zscore(
        vvix_vix, window=126, min_periods=63
    )
    result["skew_z_252"] = _rolling_zscore(
        result["skew"], window=252, min_periods=126
    )
    realized_volatility_9d = soxx_return.rolling(
        9, min_periods=5
    ).std(ddof=1) * np.sqrt(252.0)
    result["tail_risk_premium_9d"] = (
        result["vix9d"].div(100.0).pow(2)
        - realized_volatility_9d.pow(2)
    )

    result["soxl_fragility_curvature_504"] = _fragility_curvature(
        soxl_return, soxx_return
    )
    beta, downside_beta = _rolling_beta(
        soxl_return, soxx_return, window=252
    )
    result["soxl_downside_beta_gap_252"] = downside_beta - beta
    result["soxx_rebalance_drag_pressure_20"] = (
        3.0 * squared.rolling(20, min_periods=10).sum()
    )
    levered_decay = (
        pd.to_numeric(result["soxl_close"], errors="coerce")
        .pct_change(20, fill_method=None)
        - 3.0
        * pd.to_numeric(result["soxx_close"], errors="coerce").pct_change(
            20, fill_method=None
        )
    )
    result["soxl_levered_decay_acceleration_5"] = levered_decay.diff(5)

    result.loc[:, TALEB_FRAGILITY_FEATURES] = result.loc[
        :, TALEB_FRAGILITY_FEATURES
    ].replace([np.inf, -np.inf], np.nan)
    output = result.loc[:, ["datetime", *TALEB_FRAGILITY_FEATURES]].copy()
    output.attrs["feature_definitions"] = dict(FEATURE_DEFINITIONS)
    output.attrs["point_in_time"] = True
    return output

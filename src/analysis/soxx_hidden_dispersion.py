from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .soxx_constituent_attribution import build_constituent_attribution_panel


HIDDEN_DISPERSION_FEATURES = (
    "residual_dispersion_5",
    "range_energy_5",
    "cancellation_ratio_5",
    "correlation_breakdown_20",
    "breadth_entropy_5",
    "leadership_turnover_5",
    "abnormal_volume_breadth_5",
    "dispersion_vix_gap",
    "weighted_residual_x_dispersion_vix_gap",
)


@dataclass(frozen=True)
class HiddenDispersionPanel:
    constituents: pd.DataFrame
    daily: pd.DataFrame
    betas: dict[str, float]
    train: tuple[str, str]
    lag_days: int
    horizon: int


def build_hidden_dispersion_panel(
    constituent_frame: pd.DataFrame,
    index_frame: pd.DataFrame,
    *,
    train: tuple[str, str],
    lag_days: int = 1,
    horizon: int = 5,
) -> HiddenDispersionPanel:
    required_constituent = {
        "datetime",
        "instrument",
        "open",
        "high",
        "low",
        "adj_close",
        "volume",
        "weight_pct",
    }
    required_index = {"datetime", "soxx_open", "soxl_open", "vix"}
    if missing := sorted(required_constituent - set(constituent_frame.columns)):
        raise ValueError(f"hidden dispersion constituent frame missing columns: {missing}")
    if missing := sorted(required_index - set(index_frame.columns)):
        raise ValueError(f"hidden dispersion index frame missing columns: {missing}")
    if lag_days < 1 or horizon < 1:
        raise ValueError("lag_days and horizon must be positive")

    index_data = index_frame.copy()
    index_data["datetime"] = pd.to_datetime(index_data["datetime"]).dt.normalize()
    index_data = index_data.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    for column in ["soxx_open", "soxl_open", "vix"]:
        index_data[column] = pd.to_numeric(index_data[column], errors="coerce")
    index_data["soxx_return"] = index_data["soxx_open"].pct_change(fill_method=None)
    index_data["soxx_5d"] = (
        index_data["soxx_open"].shift(-(horizon + 1)) / index_data["soxx_open"].shift(-1) - 1.0
    )
    index_data["soxl_5d"] = (
        index_data["soxl_open"].shift(-(horizon + 1)) / index_data["soxl_open"].shift(-1) - 1.0
    )
    index_data["soxx_abs_5d"] = index_data["soxx_5d"].abs()

    constituents = constituent_frame.copy()
    constituents["datetime"] = pd.to_datetime(constituents["datetime"]).dt.normalize()
    constituents["instrument"] = constituents["instrument"].astype(str).str.upper()
    numeric_columns = ["open", "high", "low", "adj_close", "volume", "weight_pct"]
    for column in numeric_columns:
        constituents[column] = pd.to_numeric(constituents[column], errors="coerce")
    constituents = constituents.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    constituents["return"] = constituents.groupby("instrument", sort=False)["open"].pct_change(fill_method=None)
    constituents["range"] = np.log(
        constituents["high"].where(constituents["high"].gt(0.0))
        / constituents["low"].where(constituents["low"].gt(0.0))
    )
    volume_group = constituents.groupby("instrument", sort=False)["volume"]
    volume_mean = volume_group.transform(lambda values: values.rolling(20, min_periods=10).mean())
    volume_std = volume_group.transform(lambda values: values.rolling(20, min_periods=10).std(ddof=0))
    constituents["volume_z"] = (constituents["volume"] - volume_mean) / volume_std.replace(0.0, np.nan)
    constituents = constituents.merge(index_data[["datetime", "soxx_return"]], on="datetime", how="left")

    train_start, train_end = pd.Timestamp(train[0]), pd.Timestamp(train[1])
    betas: dict[str, float] = {}
    residual_parts: list[pd.DataFrame] = []
    for instrument, group in constituents.groupby("instrument", sort=True):
        train_values = group.loc[
            group["datetime"].between(train_start, train_end),
            ["return", "soxx_return"],
        ].dropna()
        market_variance = float(train_values["soxx_return"].var()) if len(train_values) > 1 else 0.0
        beta = (
            float(train_values["return"].cov(train_values["soxx_return"]) / market_variance)
            if market_variance > 1e-12
            else 0.0
        )
        betas[str(instrument)] = beta
        part = group.copy()
        part["residual_return"] = part["return"] - beta * part["soxx_return"]
        residual_parts.append(part)
    constituents = pd.concat(residual_parts, ignore_index=True).sort_values(["datetime", "instrument"])

    daily_components = (
        constituents.groupby("datetime", sort=True)
        .apply(_daily_components, include_groups=False)
        .reset_index()
    )
    return_pivot = constituents.pivot(index="datetime", columns="instrument", values="return").sort_index()
    daily_components = daily_components.merge(
        _cross_sectional_dynamics(return_pivot).reset_index(),
        on="datetime",
        how="left",
    )
    daily_components["residual_dispersion_5"] = daily_components["residual_dispersion"].rolling(
        5, min_periods=3
    ).mean()
    daily_components["range_energy_5"] = daily_components["range_energy"].rolling(5, min_periods=3).mean()
    daily_components["cancellation_ratio_5"] = daily_components["cancellation_ratio"].rolling(
        5, min_periods=3
    ).mean()
    daily_components["breadth_entropy_5"] = daily_components["breadth_entropy"].rolling(
        5, min_periods=3
    ).mean()
    daily_components["abnormal_volume_breadth_5"] = daily_components["abnormal_volume_breadth"].rolling(
        5, min_periods=3
    ).mean()

    lagged_columns = [
        "residual_dispersion_5",
        "range_energy_5",
        "cancellation_ratio_5",
        "correlation_breakdown_20",
        "breadth_entropy_5",
        "leadership_turnover_5",
        "abnormal_volume_breadth_5",
    ]
    daily_components[lagged_columns] = daily_components[lagged_columns].shift(lag_days)
    daily = index_data.merge(daily_components[["datetime", *lagged_columns]], on="datetime", how="left")

    attribution = build_constituent_attribution_panel(
        constituents[["datetime", "instrument", "adj_close", "weight_pct"]],
        index_data[["datetime", "soxx_open", "soxl_open"]],
        train=train,
        lookback=20,
        lag_days=lag_days,
        horizon=horizon,
    )
    daily = daily.merge(attribution.daily[["datetime", "weighted_residual"]], on="datetime", how="left")
    vix_abs_change = np.log(daily["vix"].where(daily["vix"].gt(0.0))).diff(5).abs().shift(lag_days)
    daily["dispersion_vix_gap"] = _expanding_z(daily["residual_dispersion_5"]) - _expanding_z(vix_abs_change)
    daily["weighted_residual_x_dispersion_vix_gap"] = (
        daily["weighted_residual"] * daily["dispersion_vix_gap"]
    )
    return HiddenDispersionPanel(
        constituents=constituents.reset_index(drop=True),
        daily=daily.reset_index(drop=True),
        betas=betas,
        train=train,
        lag_days=int(lag_days),
        horizon=int(horizon),
    )


def _daily_components(group: pd.DataFrame) -> pd.Series:
    returns = pd.to_numeric(group["return"], errors="coerce").dropna()
    residuals = pd.to_numeric(group["residual_return"], errors="coerce").dropna()
    ranges = pd.to_numeric(group["range"], errors="coerce").dropna()
    volume_z = pd.to_numeric(group["volume_z"], errors="coerce").dropna()
    absolute_mean = float(returns.abs().mean()) if len(returns) else 0.0
    cancellation = 0.0 if absolute_mean <= 1e-12 else 1.0 - abs(float(returns.mean())) / absolute_mean
    positive_share = float(returns.gt(0.0).mean()) if len(returns) else 0.0
    entropy = _binary_entropy(positive_share)
    return pd.Series(
        {
            "residual_dispersion": float(residuals.std(ddof=0)) if len(residuals) >= 2 else np.nan,
            "range_energy": float(np.sqrt(np.mean(np.square(ranges)))) if len(ranges) else np.nan,
            "cancellation_ratio": float(np.clip(cancellation, 0.0, 1.0)),
            "breadth_entropy": entropy,
            "abnormal_volume_breadth": float(volume_z.gt(1.5).mean()) if len(volume_z) else 0.0,
            "available_count": int(len(returns)),
        }
    )


def _binary_entropy(positive_share: float) -> float:
    p = float(np.clip(positive_share, 0.0, 1.0))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0))


def _cross_sectional_dynamics(return_pivot: pd.DataFrame) -> pd.DataFrame:
    ranks = return_pivot.rank(axis=1, method="average", pct=True)
    leadership_corr = pd.Series(np.nan, index=return_pivot.index, dtype="float64")
    lagged_ranks = ranks.shift(5)
    for idx in range(5, len(ranks)):
        paired = pd.concat([ranks.iloc[idx], lagged_ranks.iloc[idx]], axis=1).dropna()
        if len(paired) >= 2 and paired.iloc[:, 0].nunique() > 1 and paired.iloc[:, 1].nunique() > 1:
            leadership_corr.iloc[idx] = float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman"))
    leadership_turnover = ((1.0 - leadership_corr) / 2.0).clip(0.0, 1.0)
    correlation_breakdown = pd.Series(np.nan, index=return_pivot.index, dtype="float64")
    for idx in range(19, len(return_pivot)):
        corr = return_pivot.iloc[idx - 19 : idx + 1].corr(min_periods=10)
        values = corr.to_numpy(dtype="float64")
        upper = values[np.triu_indices_from(values, k=1)]
        finite = upper[np.isfinite(upper)]
        mean_correlation = float(finite.mean()) if len(finite) else 1.0
        correlation_breakdown.iloc[idx] = float(np.clip(1.0 - mean_correlation, 0.0, 2.0))
    return pd.DataFrame(
        {
            "correlation_breakdown_20": correlation_breakdown,
            "leadership_turnover_5": leadership_turnover,
        },
        index=return_pivot.index,
    )


def _expanding_z(values: pd.Series, *, min_periods: int = 20) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.expanding(min_periods=min_periods).mean()
    std = numeric.expanding(min_periods=min_periods).std(ddof=0)
    return (numeric - mean) / std.replace(0.0, np.nan)

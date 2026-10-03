from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.data.soxx_leveraged_training import (
    ACTUAL_2X_TRAINING_PRODUCTS,
    synthetic_daily_reset_return,
)


PIT_TAIL_SOURCE = "qlib.data.D.features"
PIT_TAIL_RAW_FIELDS = [
    "$open",
    "$high",
    "$low",
    "$close",
    "$adj_close",
    "$volume",
    "$pit_weight_pct",
    "$pit_available",
    "$pit_member",
    "$instrument_type",
    "$leverage_multiple",
    "$is_actual_2x",
]
CONSTITUENT_MODEL_FEATURES = [
    "ret1",
    "mom5",
    "mom20",
    "mom60",
    "reversal1",
    "vol10",
    "vol20",
    "downside_vol20",
    "skew20",
    "kurt60",
    "drawdown20",
    "drawdown60",
    "gap1",
    "intraday1",
    "volume_z20",
    "range_pct",
    "mom5_rank",
    "mom20_rank",
    "vol20_rank",
    "drawdown20_rank",
    "volume_z20_rank",
]


def load_soxx_pit_leveraged_training_frame(
    config: QlibProviderConfig,
    *,
    instruments: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    frame = load_qlib_feature_frame(
        config,
        instruments=[str(item).upper() for item in instruments],
        fields=PIT_TAIL_RAW_FIELDS,
        start=start,
        end=end,
    ).reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    return frame.sort_values(["instrument", "datetime"]).reset_index(
        drop=True
    )


def build_constituent_2x_tail_training_frame(
    qlib_frame: pd.DataFrame,
    *,
    horizon: int = 5,
    synthetic_leverage: float = 2.0,
    annual_fee_bps: float = 95.0,
    crash_drawdown: float = -0.12,
    crash_terminal: float = -0.15,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("constituent 2x tail horizon must be positive")
    frame = qlib_frame.copy()
    frame = frame.loc[
        pd.to_numeric(
            frame["$instrument_type"], errors="coerce"
        ).eq(0.0)
    ].copy()
    if frame.empty:
        raise ValueError("no constituent rows in leveraged training provider")
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame = frame.sort_values(["instrument", "datetime"]).reset_index(
        drop=True
    )

    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("instrument", sort=False):
        part = group.copy()
        price = pd.to_numeric(
            part.get("$adj_close", part["$close"]),
            errors="coerce",
        )
        close = pd.to_numeric(part["$close"], errors="coerce")
        open_price = pd.to_numeric(part["$open"], errors="coerce")
        high = pd.to_numeric(part["$high"], errors="coerce")
        low = pd.to_numeric(part["$low"], errors="coerce")
        volume = pd.to_numeric(part["$volume"], errors="coerce")
        ret1 = price.pct_change(fill_method=None)
        part["ret1"] = ret1
        part["mom5"] = price.pct_change(5, fill_method=None)
        part["mom20"] = price.pct_change(20, fill_method=None)
        part["mom60"] = price.pct_change(60, fill_method=None)
        part["reversal1"] = -ret1
        part["vol10"] = ret1.rolling(10, min_periods=6).std(ddof=0)
        part["vol20"] = ret1.rolling(20, min_periods=10).std(ddof=0)
        downside = ret1.clip(upper=0.0)
        part["downside_vol20"] = np.sqrt(
            downside.pow(2).rolling(20, min_periods=10).mean()
        )
        part["skew20"] = ret1.rolling(20, min_periods=10).skew()
        part["kurt60"] = ret1.rolling(60, min_periods=30).kurt()
        part["drawdown20"] = (
            price / price.rolling(20, min_periods=10).max() - 1.0
        )
        part["drawdown60"] = (
            price / price.rolling(60, min_periods=20).max() - 1.0
        )
        part["gap1"] = open_price / close.shift(1) - 1.0
        part["intraday1"] = close / open_price - 1.0
        log_volume = np.log1p(volume.clip(lower=0.0))
        part["volume_z20"] = (
            log_volume - log_volume.rolling(20, min_periods=10).mean()
        ) / (
            log_volume.rolling(20, min_periods=10).std(ddof=0) + 1e-8
        )
        part["range_pct"] = (high - low) / close.replace(0.0, np.nan)

        synthetic_return = synthetic_daily_reset_return(
            ret1,
            leverage_multiple=synthetic_leverage,
            annual_fee_bps=annual_fee_bps,
        )
        future_steps = [
            synthetic_return.shift(-step)
            for step in range(1, horizon + 1)
        ]
        underlying_future_steps = [
            ret1.shift(-step) for step in range(1, horizon + 1)
        ]
        future_matrix = np.column_stack(
            [step.to_numpy(dtype=float) for step in future_steps]
        )
        underlying_future_matrix = np.column_stack(
            [
                step.to_numpy(dtype=float)
                for step in underlying_future_steps
            ]
        )
        future_matrix = np.where(
            np.isfinite(future_matrix),
            future_matrix,
            np.nan,
        )
        log_steps = np.log1p(future_matrix)
        cumulative = np.expm1(
            np.nancumsum(log_steps, axis=1)
        )
        full_horizon = np.isfinite(future_matrix).all(axis=1)
        terminal = cumulative[:, -1]
        minimum = np.nanmin(cumulative, axis=1)
        underlying_terminal = np.expm1(
            np.nansum(np.log1p(underlying_future_matrix), axis=1)
        )
        terminal[~full_horizon] = np.nan
        minimum[~full_horizon] = np.nan
        underlying_terminal[~full_horizon] = np.nan
        part["underlying_forward_return"] = underlying_terminal
        part["synthetic_2x_forward_return"] = terminal
        part["synthetic_2x_excess_1x"] = (
            terminal - underlying_terminal
        )
        part["synthetic_2x_upside_label"] = np.where(
            full_horizon,
            (terminal > underlying_terminal).astype(float),
            np.nan,
        )
        part["synthetic_2x_forward_min_return"] = minimum
        part["synthetic_2x_crash_label"] = np.where(
            full_horizon,
            (
                (minimum <= crash_drawdown)
                | (terminal <= crash_terminal)
            ).astype(float),
            np.nan,
        )
        part["synthetic_2x_tail_severity"] = np.where(
            full_horizon,
            np.maximum(-minimum, 0.0),
            np.nan,
        )
        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    for feature in (
        "mom5",
        "mom20",
        "vol20",
        "drawdown20",
        "volume_z20",
    ):
        result[f"{feature}_rank"] = result.groupby(
            "datetime", sort=False
        )[feature].rank(pct=True)
    pit_available = pd.to_numeric(
        result["$pit_available"], errors="coerce"
    ).fillna(0.0)
    pit_member = pd.to_numeric(
        result["$pit_member"], errors="coerce"
    ).fillna(0.0)
    pit_weight = pd.to_numeric(
        result["$pit_weight_pct"], errors="coerce"
    ).fillna(0.0)
    result["training_scope"] = np.where(
        pit_available.gt(0.5),
        "pit_member",
        "union_backfill_pretrain",
    )
    result["training_eligible"] = np.where(
        pit_available.gt(0.5),
        pit_member.gt(0.5),
        True,
    )
    result["training_weight"] = np.where(
        pit_available.gt(0.5),
        0.5 + np.sqrt(pit_weight.clip(lower=0.0) / 10.0),
        0.25,
    )
    result.attrs["source"] = PIT_TAIL_SOURCE
    result.attrs["feature_cols"] = list(CONSTITUENT_MODEL_FEATURES)
    result.attrs["target_cols"] = [
        "underlying_forward_return",
        "synthetic_2x_forward_return",
        "synthetic_2x_excess_1x",
        "synthetic_2x_upside_label",
        "synthetic_2x_forward_min_return",
        "synthetic_2x_crash_label",
        "synthetic_2x_tail_severity",
    ]
    result.attrs["synthetic_2x"] = {
        "leverage_multiple": synthetic_leverage,
        "annual_fee_bps": annual_fee_bps,
        "horizon": horizon,
        "crash_drawdown": crash_drawdown,
        "crash_terminal": crash_terminal,
        "execution_status": "training_label_only_not_tradable_instrument",
    }
    return result.sort_values(
        ["datetime", "instrument"]
    ).reset_index(drop=True)


def build_daily_soxx_tail_factor_frame(
    constituent_frame: pd.DataFrame,
    *,
    lag_days: int = 1,
    systemic_window: int = 20,
    tail_window: int = 20,
) -> pd.DataFrame:
    if lag_days < 1:
        raise ValueError("SOXX daily tail factors require lag_days >= 1")
    frame = constituent_frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    pit_available = pd.to_numeric(
        frame["$pit_available"], errors="coerce"
    ).fillna(0.0)
    pit_member = pd.to_numeric(
        frame["$pit_member"], errors="coerce"
    ).fillna(0.0)
    frame["_aggregate_eligible"] = np.where(
        pit_available.gt(0.5),
        pit_member.gt(0.5),
        True,
    )
    eligible = frame.loc[frame["_aggregate_eligible"]].copy()
    if eligible.empty:
        raise ValueError("no eligible constituent rows for tail factors")
    weights = pd.to_numeric(
        eligible["$pit_weight_pct"], errors="coerce"
    ).fillna(0.0)
    eligible["_weight"] = np.where(
        weights.gt(0.0),
        weights,
        1.0,
    )

    daily_rows: list[dict[str, float | str | pd.Timestamp]] = []
    for date, group in eligible.groupby("datetime", sort=True):
        ret1 = pd.to_numeric(group["ret1"], errors="coerce")
        valid = ret1.notna()
        returns = ret1.loc[valid]
        group_weights = pd.to_numeric(
            group.loc[valid, "_weight"], errors="coerce"
        ).clip(lower=0.0)
        normalized = (
            group_weights / group_weights.sum()
            if group_weights.sum() > 0.0
            else pd.Series(
                1.0 / max(len(returns), 1),
                index=returns.index,
            )
        )
        daily_rows.append(
            {
                "datetime": date,
                "soxx_pit_tail_breadth_ret1_pos": float(
                    returns.gt(0.0).mean()
                ),
                "soxx_pit_tail_downside_frac_m3": float(
                    returns.lt(-0.03).mean()
                ),
                "soxx_pit_tail_downside_frac_m5": float(
                    returns.lt(-0.05).mean()
                ),
                "soxx_pit_tail_q05_ret1": float(
                    returns.quantile(0.05)
                ),
                "soxx_pit_tail_q10_ret1": float(
                    returns.quantile(0.10)
                ),
                "soxx_pit_tail_dispersion_ret1": float(
                    returns.std(ddof=0)
                ),
                "soxx_pit_tail_weighted_ret1": float(
                    (returns * normalized).sum()
                ),
                "soxx_pit_tail_breadth_mom20_pos": float(
                    pd.to_numeric(
                        group["mom20"], errors="coerce"
                    )
                    .gt(0.0)
                    .mean()
                ),
                "soxx_pit_tail_breadth_dd20_m10": float(
                    pd.to_numeric(
                        group["drawdown20"], errors="coerce"
                    )
                    .lt(-0.10)
                    .mean()
                ),
                "soxx_pit_tail_avg_downside_vol20": float(
                    pd.to_numeric(
                        group["downside_vol20"], errors="coerce"
                    ).mean()
                ),
                "soxx_pit_tail_weight_hhi": float(
                    np.square(normalized).sum()
                ),
                "soxx_pit_tail_available_count": float(len(group)),
                "soxx_pit_tail_pit_fraction": float(
                    pd.to_numeric(
                        group["$pit_available"], errors="coerce"
                    )
                    .fillna(0.0)
                    .gt(0.5)
                    .mean()
                ),
            }
        )
    daily = pd.DataFrame(daily_rows).sort_values("datetime")

    return_matrix = (
        eligible.pivot_table(
            index="datetime",
            columns="instrument",
            values="ret1",
            aggfunc="last",
        )
        .sort_index()
        .reindex(pd.DatetimeIndex(daily["datetime"]))
    )
    daily["soxx_pit_tail_absorption20"] = _rolling_absorption_ratio(
        return_matrix,
        window=systemic_window,
    ).to_numpy()
    daily["soxx_pit_tail_avg_corr20"] = _rolling_average_correlation(
        return_matrix,
        window=systemic_window,
    ).to_numpy()
    daily["soxx_pit_tail_hill20"] = _rolling_left_tail_hill(
        return_matrix,
        window=tail_window,
    ).to_numpy()
    base_features = [
        column for column in daily.columns if column != "datetime"
    ]
    daily[base_features] = daily[base_features].shift(lag_days)
    feature_cols = [
        f"{column}_lag{lag_days}" for column in base_features
    ]
    daily = daily.rename(
        columns=dict(zip(base_features, feature_cols, strict=True))
    )
    daily.attrs["source"] = PIT_TAIL_SOURCE
    daily.attrs["feature_cols"] = feature_cols
    daily.attrs["lag_days"] = lag_days
    daily.attrs["systemic_window"] = systemic_window
    daily.attrs["tail_window"] = tail_window
    daily.attrs["pit_policy"] = (
        "SEC filing-date eligible weights; union backfill before first filing"
    )
    return daily.reset_index(drop=True)


def actual_2x_calibration_metrics(
    qlib_frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = qlib_frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    prices = frame.pivot_table(
        index="datetime",
        columns="instrument",
        values="$adj_close",
        aggfunc="last",
    ).sort_index()
    returns = prices.pct_change(fill_method=None)
    rows: list[dict[str, Any]] = []
    for product, metadata in ACTUAL_2X_TRAINING_PRODUCTS.items():
        underlying = str(metadata["underlying_ticker"])
        if product not in returns or underlying not in returns:
            continue
        pair = pd.concat(
            [
                returns[product].rename("actual"),
                returns[underlying].rename("underlying"),
            ],
            axis=1,
        ).dropna()
        effective_date = metadata.get(
            "two_x_effective_date", metadata.get("inception_date")
        )
        if effective_date:
            pair = pair.loc[pair.index >= pd.Timestamp(effective_date)]
        if pair.empty:
            continue
        synthetic = synthetic_daily_reset_return(pair["underlying"])
        residual = pair["actual"] - synthetic
        rows.append(
            {
                "product": product,
                "underlying": underlying,
                "observations": int(len(pair)),
                "start_date": pair.index.min().date().isoformat(),
                "end_date": pair.index.max().date().isoformat(),
                "daily_rank_correlation": float(
                    pair["actual"].corr(synthetic, method="spearman")
                ),
                "tracking_residual_mean_bps": float(
                    residual.mean() * 10_000.0
                ),
                "tracking_residual_rmse_bps": float(
                    np.sqrt(np.mean(np.square(residual))) * 10_000.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _rolling_absorption_ratio(
    returns: pd.DataFrame,
    *,
    window: int,
) -> pd.Series:
    values: list[float] = []
    for end in range(len(returns)):
        start = max(0, end - window + 1)
        sample = returns.iloc[start : end + 1].dropna(axis=1, thresh=5)
        sample = sample.dropna(axis=0, thresh=max(3, sample.shape[1] // 2))
        sample = sample.fillna(sample.mean())
        if sample.shape[0] < 5 or sample.shape[1] < 3:
            values.append(np.nan)
            continue
        covariance = np.cov(sample.to_numpy(), rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        total = float(np.maximum(eigenvalues, 0.0).sum())
        values.append(
            np.nan
            if total <= 0.0
            else float(max(eigenvalues[-1], 0.0) / total)
        )
    return pd.Series(values, index=returns.index)


def _rolling_average_correlation(
    returns: pd.DataFrame,
    *,
    window: int,
) -> pd.Series:
    values: list[float] = []
    for end in range(len(returns)):
        sample = returns.iloc[max(0, end - window + 1) : end + 1]
        correlation = sample.corr(min_periods=5).to_numpy()
        if correlation.shape[0] < 3:
            values.append(np.nan)
            continue
        upper = correlation[np.triu_indices_from(correlation, k=1)]
        values.append(
            float(np.nanmean(upper))
            if np.isfinite(upper).any()
            else np.nan
        )
    return pd.Series(values, index=returns.index)


def _rolling_left_tail_hill(
    returns: pd.DataFrame,
    *,
    window: int,
    quantile: float = 0.10,
) -> pd.Series:
    values: list[float] = []
    for end in range(len(returns)):
        sample = returns.iloc[max(0, end - window + 1) : end + 1]
        losses = (-sample.to_numpy().ravel()).astype(float)
        losses = losses[np.isfinite(losses) & (losses > 0.0)]
        if len(losses) < 30:
            values.append(np.nan)
            continue
        threshold = float(np.quantile(losses, 1.0 - quantile))
        tail = losses[losses >= threshold]
        if threshold <= 0.0 or len(tail) < 5:
            values.append(np.nan)
            continue
        hill_xi = float(np.mean(np.log(tail / threshold)))
        values.append(hill_xi)
    return pd.Series(values, index=returns.index)

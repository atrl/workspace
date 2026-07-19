"""Transferable, point-in-time factors for high-level 3x ETF chop regimes."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


PATH_LABELS = {
    "benign_chop": 0,
    "breakout": 1,
    "breakdown": 2,
    "whipsaw": 3,
}

TRANSFERABLE_FEATURES = [
    "ret_5_pctile252",
    "ret_20_pctile252",
    "ret_60_pctile252",
    "range_pos_252",
    "dist_high_252",
    "drawdown_60_pctile252",
    "trend_efficiency_20_pctile252",
    "sign_change_density_20_pctile252",
    "rv_20_pctile252",
    "downside_semivariance_20_pctile252",
    "tracking_error_20_pctile252",
    "tracking_error_expansion_pctile252",
    "realized_vix_gap",
    "range_vix_gap",
    "abs_vix_change_5_pctile252",
    "cross_asset_dispersion",
    "cross_asset_positive_breadth",
    "cross_asset_downside_breadth",
    "cohort_leadership_turnover",
    "high_chop_intensity",
]

_REQUIRED_COLUMNS = {
    "instrument",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "underlying_close",
    "vix",
}


def _rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        if len(values) == 0 or not np.isfinite(values[-1]):
            return np.nan
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return np.nan
        return float(np.mean(finite <= values[-1]))

    return series.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0.0, np.nan))


def _asset_features(
    asset: pd.DataFrame,
    *,
    normalization_window: int,
    min_periods: int,
) -> pd.DataFrame:
    out = asset.sort_values("datetime").copy()
    close = out["close"].astype(float)
    underlying = out["underlying_close"].astype(float)
    returns = close.pct_change(fill_method=None)
    underlying_returns = underlying.pct_change(fill_method=None)
    vix = out["vix"].astype(float)

    ret_5 = close.pct_change(5, fill_method=None)
    ret_20 = close.pct_change(20, fill_method=None)
    ret_60 = close.pct_change(60, fill_method=None)
    rolling_high = close.rolling(normalization_window, min_periods=min_periods).max()
    rolling_low = close.rolling(normalization_window, min_periods=min_periods).min()
    rolling_range = rolling_high - rolling_low
    drawdown_60 = close.div(close.rolling(60, min_periods=min(min_periods, 60)).max()).sub(1.0)

    net_move_20 = close.diff(20).abs()
    path_20 = close.diff().abs().rolling(20, min_periods=min(min_periods, 20)).sum()
    trend_efficiency_20 = _safe_divide(net_move_20, path_20)
    signs = np.sign(returns)
    sign_change = signs.ne(signs.shift()).astype(float)
    sign_change[returns.isna() | returns.shift().isna()] = np.nan
    sign_density_20 = sign_change.rolling(20, min_periods=min(min_periods, 20)).mean()

    rv_20 = returns.rolling(20, min_periods=min(min_periods, 20)).std(ddof=0) * np.sqrt(252.0)
    downside_semivariance_20 = returns.clip(upper=0.0).pow(2).rolling(
        20, min_periods=min(min_periods, 20)
    ).mean()
    tracking_residual = returns - 3.0 * underlying_returns
    tracking_error_20 = tracking_residual.rolling(
        20, min_periods=min(min_periods, 20)
    ).std(ddof=0) * np.sqrt(252.0)
    tracking_error_base = tracking_residual.rolling(
        min(normalization_window, 60), min_periods=min(min_periods, 20)
    ).std(ddof=0) * np.sqrt(252.0)
    tracking_error_expansion = _safe_divide(tracking_error_20, tracking_error_base).sub(1.0)

    intraday_range = _safe_divide(out["high"].astype(float) - out["low"].astype(float), close)
    range_20 = intraday_range.rolling(20, min_periods=min(min_periods, 20)).mean()
    vix_level_pctile = _rolling_percentile(vix, normalization_window, min_periods)

    raw = pd.DataFrame(index=out.index)
    raw["ret_5_pctile252"] = _rolling_percentile(ret_5, normalization_window, min_periods)
    raw["ret_20_pctile252"] = _rolling_percentile(ret_20, normalization_window, min_periods)
    raw["ret_60_pctile252"] = _rolling_percentile(ret_60, normalization_window, min_periods)
    raw["range_pos_252"] = _safe_divide(close - rolling_low, rolling_range).clip(0.0, 1.0)
    raw["dist_high_252"] = close.div(rolling_high).sub(1.0)
    raw["drawdown_60_pctile252"] = _rolling_percentile(drawdown_60, normalization_window, min_periods)
    raw["trend_efficiency_20_pctile252"] = _rolling_percentile(
        trend_efficiency_20, normalization_window, min_periods
    )
    raw["sign_change_density_20_pctile252"] = _rolling_percentile(
        sign_density_20, normalization_window, min_periods
    )
    raw["rv_20_pctile252"] = _rolling_percentile(rv_20, normalization_window, min_periods)
    raw["downside_semivariance_20_pctile252"] = _rolling_percentile(
        downside_semivariance_20, normalization_window, min_periods
    )
    raw["tracking_error_20_pctile252"] = _rolling_percentile(
        tracking_error_20, normalization_window, min_periods
    )
    raw["tracking_error_expansion_pctile252"] = _rolling_percentile(
        tracking_error_expansion, normalization_window, min_periods
    )
    raw["realized_vix_gap"] = raw["rv_20_pctile252"] - vix_level_pctile
    raw["range_vix_gap"] = _rolling_percentile(range_20, normalization_window, min_periods) - vix_level_pctile
    raw["abs_vix_change_5_pctile252"] = _rolling_percentile(
        vix.pct_change(5, fill_method=None).abs(), normalization_window, min_periods
    )

    # Every model-visible asset feature is delayed by one complete session.
    for column in raw.columns:
        out[column] = raw[column].shift(1)
    out["_lagged_return"] = returns.shift(1)
    out["_lagged_ret_5"] = ret_5.shift(1)
    out["_lagged_sigma_20"] = rv_20.shift(1)
    return out


def _add_cross_asset_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    by_date = out.groupby("datetime", sort=True)
    out["cross_asset_dispersion"] = by_date["_lagged_return"].transform(lambda values: values.std(ddof=0))
    out["cross_asset_positive_breadth"] = by_date["_lagged_return"].transform(lambda values: (values > 0).mean())
    downside_threshold = by_date["_lagged_return"].transform(lambda values: values.quantile(0.25))
    out["cross_asset_downside_breadth"] = (
        out["_lagged_return"].le(downside_threshold).groupby(out["datetime"]).transform("mean")
    )

    rank = by_date["_lagged_ret_5"].rank(method="average", pct=True)
    leaders = rank.ge(0.75)
    leader_matrix = (
        out.assign(_leader=leaders.astype(float))
        .pivot(index="datetime", columns="instrument", values="_leader")
        .sort_index()
    )
    turnover = leader_matrix.diff().abs().mean(axis=1)
    out["cohort_leadership_turnover"] = out["datetime"].map(turnover)

    near_high = out["range_pos_252"].clip(0.0, 1.0)
    low_efficiency = 1.0 - out["trend_efficiency_20_pctile252"].clip(0.0, 1.0)
    noisy = out["rv_20_pctile252"].clip(0.0, 1.0)
    quiet_vix = 1.0 - out["abs_vix_change_5_pctile252"].clip(0.0, 1.0)
    out["high_chop_intensity"] = pd.concat(
        [near_high, low_efficiency, noisy, quiet_vix], axis=1
    ).mean(axis=1, skipna=False).clip(0.0, 1.0)
    return out


def _add_path_labels(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, asset in frame.groupby("instrument", sort=False):
        out = asset.sort_values("datetime").copy()
        close = out["close"].astype(float).to_numpy()
        sigma = out["_lagged_sigma_20"].to_numpy()
        labels = np.full(len(out), np.nan)
        mfe = np.full(len(out), np.nan)
        mae = np.full(len(out), np.nan)
        terminal = np.full(len(out), np.nan)
        usable_end = max(0, len(out) - horizon - 1)
        for index in range(usable_end):
            path = close[index + 1 : index + horizon + 1] / close[index] - 1.0
            if len(path) != horizon or not np.isfinite(sigma[index]) or sigma[index] <= 0.0:
                continue
            scale = sigma[index] * np.sqrt(horizon / 252.0)
            mfe[index] = np.nanmax(path)
            mae[index] = np.nanmin(path)
            terminal[index] = path[-1]
            if mfe[index] >= 0.75 * scale and mae[index] <= -0.75 * scale:
                labels[index] = PATH_LABELS["whipsaw"]
            elif mae[index] <= -1.0 * scale and terminal[index] <= -0.5 * scale:
                labels[index] = PATH_LABELS["breakdown"]
            elif mfe[index] >= 1.0 * scale and mae[index] > -0.75 * scale:
                labels[index] = PATH_LABELS["breakout"]
            else:
                labels[index] = PATH_LABELS["benign_chop"]
        out["forward_mfe"] = mfe
        out["forward_mae"] = mae
        out["forward_terminal_return"] = terminal
        out["path_label"] = labels
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True).sort_values(["datetime", "instrument"]).reset_index(drop=True)


def build_multi_3x_high_chop_frame(
    panel: pd.DataFrame,
    *,
    label_horizon: int = 20,
    normalization_window: int = 252,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Build transferable factors and mutually exclusive forward path labels.

    Model features use only information available before the row's session.
    Forward fields are research labels and must never be included in inference.
    """

    missing = sorted(_REQUIRED_COLUMNS.difference(panel.columns))
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    if label_horizon < 1:
        raise ValueError("label_horizon must be positive")
    if normalization_window < 2:
        raise ValueError("normalization_window must be at least 2")
    effective_min = min_periods if min_periods is not None else max(20, normalization_window // 2)
    if effective_min < 2 or effective_min > normalization_window:
        raise ValueError("min_periods must be in [2, normalization_window]")

    working = panel.copy()
    working["datetime"] = pd.to_datetime(working["datetime"])
    if working.duplicated(["instrument", "datetime"]).any():
        raise ValueError("panel contains duplicate instrument/datetime rows")
    pieces = [
        _asset_features(asset, normalization_window=normalization_window, min_periods=effective_min)
        for _, asset in working.groupby("instrument", sort=False)
    ]
    featured = _add_cross_asset_features(pd.concat(pieces, ignore_index=True))
    return _add_path_labels(featured, label_horizon)


def purge_segment_tail(
    frame: pd.DataFrame,
    *,
    end: str | pd.Timestamp,
    horizon: int = 20,
    columns: Iterable[str] = ("path_label",),
) -> pd.DataFrame:
    """Blank labels whose forward path would cross a segment boundary."""

    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    boundary = pd.Timestamp(end)
    for _, positions in out.loc[out["datetime"] <= boundary].groupby("instrument").groups.items():
        tail = list(positions)[-horizon:]
        out.loc[tail, list(columns)] = np.nan
    return out

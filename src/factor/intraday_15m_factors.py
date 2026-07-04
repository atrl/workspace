from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame


INTRADAY_15M_FACTOR_SOURCE = "qlib.data.D.features"
INTRADAY_15M_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$bar_index", "$bar_count"]
INTRADAY_15M_LIMITATIONS = [
    "15min bars are aggregated to daily features and lagged before open-to-open model use.",
    "Intraday aggregates describe the prior completed session; they are not same-day execution signals.",
]
_DAILY_AGGREGATE_COLUMNS = [
    "ret",
    "range",
    "realized_vol",
    "late_ret",
    "close_location",
    "volume_total",
    "volume_z20",
    "bar_coverage",
]


def build_intraday_15m_factor_frame(
    config: QlibProviderConfig,
    *,
    instruments: Iterable[str],
    start: str,
    end: str,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Build daily-safe 15-minute intraday structure factors from a qlib provider."""

    if config.freq.lower() != "15min":
        raise ValueError("intraday 15m factors require QlibProviderConfig(freq='15min')")
    if lag_days < 1:
        raise ValueError("intraday 15m factors must use lag_days >= 1 for open-to-open execution")
    instrument_list = [str(instrument).upper() for instrument in instruments]
    if not instrument_list:
        raise ValueError("intraday 15m factor frame requires at least one instrument")

    qlib_frame = load_qlib_feature_frame(
        config,
        instruments=instrument_list,
        fields=INTRADAY_15M_FIELDS,
        start=start,
        end=end,
    )
    panel = qlib_frame.reset_index().copy()
    panel["instrument"] = panel["instrument"].astype(str).str.upper()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["date"] = panel["datetime"].dt.normalize()
    panel = panel.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    for field in INTRADAY_15M_FIELDS:
        panel[field] = pd.to_numeric(panel[field], errors="coerce")

    calendar = pd.Index(sorted(panel["date"].dropna().unique()), name="datetime")
    feature_frame = pd.DataFrame({"datetime": calendar})
    raw_daily_by_symbol: dict[str, pd.DataFrame] = {}
    for instrument in instrument_list:
        symbol = instrument.lower()
        daily = _daily_intraday_aggregates(panel.loc[panel["instrument"].eq(instrument)], calendar=calendar)
        raw_daily_by_symbol[instrument] = daily
        daily_features = pd.DataFrame({"datetime": calendar})
        raw_to_feature = {
            "ret": f"intraday15m_{symbol}_ret_lag{lag_days}",
            "range": f"intraday15m_{symbol}_range_lag{lag_days}",
            "realized_vol": f"intraday15m_{symbol}_realized_vol_lag{lag_days}",
            "late_ret": f"intraday15m_{symbol}_late_ret_lag{lag_days}",
            "close_location": f"intraday15m_{symbol}_close_location_lag{lag_days}",
            "volume_z20": f"intraday15m_{symbol}_volume_z20_lag{lag_days}",
            "bar_coverage": f"intraday15m_{symbol}_bar_coverage_lag{lag_days}",
        }
        for raw_col, feature_col in raw_to_feature.items():
            daily_features[feature_col] = daily[raw_col].shift(lag_days).to_numpy()
        feature_frame = feature_frame.merge(daily_features, on="datetime", how="left")

    cross_features = _cross_asset_intraday_features(raw_daily_by_symbol, calendar=calendar, lag_days=lag_days)
    if not cross_features.empty:
        feature_frame = feature_frame.merge(cross_features, on="datetime", how="left")

    feature_cols = [column for column in feature_frame.columns if column != "datetime"]
    feature_frame.attrs["feature_cols"] = feature_cols
    feature_frame.attrs["source"] = INTRADAY_15M_FACTOR_SOURCE
    feature_frame.attrs["provider_uri"] = config.resolved_provider_uri.as_posix()
    feature_frame.attrs["instruments"] = instrument_list
    feature_frame.attrs["freq"] = config.freq
    feature_frame.attrs["lag_days"] = int(lag_days)
    feature_frame.attrs["fields"] = INTRADAY_15M_FIELDS
    feature_frame.attrs["limitations"] = INTRADAY_15M_LIMITATIONS
    return feature_frame


def _daily_intraday_aggregates(panel: pd.DataFrame, *, calendar: pd.Index) -> pd.DataFrame:
    rows = []
    for date, group in panel.groupby("date", sort=True):
        rows.append({"datetime": date, **_aggregate_session(group)})
    if rows:
        daily = pd.DataFrame(rows).set_index("datetime").sort_index()
    else:
        daily = pd.DataFrame(index=calendar, columns=_DAILY_AGGREGATE_COLUMNS)
    daily = daily.reindex(calendar)
    volume_mean = daily["volume_total"].rolling(20, min_periods=5).mean()
    daily["volume_z20"] = daily["volume_total"] / (volume_mean + 1e-12) - 1.0
    return daily


def _aggregate_session(group: pd.DataFrame) -> dict[str, float]:
    group = group.sort_values("datetime")
    open_ = _first_valid(group["$open"])
    close = _last_valid(group["$close"])
    high = _max_valid(group["$high"])
    low = _min_valid(group["$low"])
    close_series = pd.to_numeric(group["$close"], errors="coerce").dropna()
    bar_returns = close_series.pct_change(fill_method=None).dropna()
    late_ret = np.nan
    if len(close_series) >= 2:
        late_start = close_series.iloc[max(0, len(close_series) // 2)]
        late_ret = _safe_ratio(close, late_start) - 1.0
    range_ = _safe_ratio(high, low) - 1.0
    close_location = np.nan
    if np.isfinite(high) and np.isfinite(low) and high > low and np.isfinite(close):
        close_location = float((close - low) / (high - low))
    expected_bar_count = _max_valid(group["$bar_count"])
    observed_bar_count = float(group["$close"].notna().sum())
    bar_coverage = np.nan
    if np.isfinite(expected_bar_count) and expected_bar_count > 0:
        bar_coverage = float(observed_bar_count / expected_bar_count)
    return {
        "ret": _safe_ratio(close, open_) - 1.0,
        "range": range_,
        "realized_vol": float(bar_returns.std(ddof=0)) if len(bar_returns) >= 1 else np.nan,
        "late_ret": late_ret,
        "close_location": close_location,
        "volume_total": float(pd.to_numeric(group["$volume"], errors="coerce").sum()),
        "bar_coverage": bar_coverage,
    }


def _cross_asset_intraday_features(
    raw_daily_by_symbol: dict[str, pd.DataFrame],
    *,
    calendar: pd.Index,
    lag_days: int,
) -> pd.DataFrame:
    if "SOXL" not in raw_daily_by_symbol or "SOXX" not in raw_daily_by_symbol:
        return pd.DataFrame()
    soxl = raw_daily_by_symbol["SOXL"]
    soxx = raw_daily_by_symbol["SOXX"]
    cross = pd.DataFrame({"datetime": calendar})
    cross[f"intraday15m_soxl_minus_3x_soxx_ret_lag{lag_days}"] = (
        soxl["ret"] - 3.0 * soxx["ret"]
    ).shift(lag_days).to_numpy()
    cross[f"intraday15m_range_ratio_soxl_soxx_lag{lag_days}"] = (
        soxl["range"] / (soxx["range"] + 1e-12)
    ).shift(lag_days).to_numpy()
    cross[f"intraday15m_realized_vol_ratio_soxl_soxx_lag{lag_days}"] = (
        soxl["realized_vol"] / (soxx["realized_vol"] + 1e-12)
    ).shift(lag_days).to_numpy()
    return cross


def _first_valid(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def _last_valid(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else np.nan


def _max_valid(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else np.nan


def _min_valid(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if not values.empty else np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return np.nan
    return float(numerator / denominator)

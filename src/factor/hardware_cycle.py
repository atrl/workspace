from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.data.hardware_cycle import (
    CLOUD_GPU_DAILY_PROXY_COUNT_FIELDS,
    CLOUD_GPU_DAILY_PROXY_FIELDS,
    FREE_GPU_PRICE_FIELDS,
    HARDWARE_COUNT_FIELDS,
    HARDWARE_CYCLE_INSTRUMENT,
    HARDWARE_DAILY_DERIVED_FIELDS,
    HARDWARE_LEVEL_FIELDS,
    HARDWARE_SALES_PRICE_FIELDS,
)


HARDWARE_CYCLE_FACTOR_SOURCE = "qlib.data.D.features"
HARDWARE_CYCLE_LIMITATIONS = [
    "cloud_gpu_daily_* features are daily public-market proxy indices from trading-day closes, not direct rental quote history.",
    "free_cloud_gpu_* features are direct no-auth daily rental quote snapshots when available; missing dates remain missing before factor lagging.",
    "free_cloud_gpu_* prices are cloud rental or instance-hour quotes, not GPU hardware sales prices.",
    "hardware_sales_* features are official monthly FRED/BLS producer sales price indices carried as observed/stale daily state, not direct daily quotes.",
    "storage_hdd_* and networking_* features are daily listed-equity supply-chain proxies, not product-level hard-disk spot prices.",
    "Cloud GPU direct pricing history starts later than market proxy history and is exposed only as diagnostic state-provenance fields.",
    "All hardware-cycle features are shifted before open-to-open model use.",
]
CLOUD_GPU_STATE_SUFFIXES = ("_available", "_source_observed", "_is_carried", "_staleness_days")
FREE_GPU_PRICE_STATE_SUFFIXES = ("_available", "_source_observed", "_provider_count", "_quote_count")
DEFAULT_CLOUD_GPU_FIELDS = (
    "cloud_gpu_h100_on_demand",
    "cloud_gpu_h200_on_demand",
    "cloud_gpu_a100_on_demand",
    "cloud_gpu_l40s_on_demand",
    "cloud_gpu_rtx_4090_on_demand",
    "cloud_gpu_modern_on_demand",
    "cloud_gpu_last_released_on_demand",
    "cloud_gpu_legacy_on_demand",
)


def _zscore(series: pd.Series, *, window: int, min_periods: int | None = None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    min_obs = min_periods or min(window, 20)
    rolling = numeric.rolling(window, min_periods=min_obs)
    return (numeric - rolling.mean()) / rolling.std(ddof=0).replace(0.0, np.nan)


def _log_return(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    return np.log(numeric / numeric.shift(window)).replace([np.inf, -np.inf], np.nan)


def _available_fields(config: QlibProviderConfig, *, start: str, end: str) -> list[str]:
    manifest_path = config.resolved_provider_uri / "qlib_store_manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [str(field).removeprefix("$") for field in manifest.get("fields", [])]
    return [
        *HARDWARE_LEVEL_FIELDS,
        *HARDWARE_COUNT_FIELDS,
        *CLOUD_GPU_DAILY_PROXY_FIELDS,
        *HARDWARE_DAILY_DERIVED_FIELDS,
        *DEFAULT_CLOUD_GPU_FIELDS,
        *FREE_GPU_PRICE_FIELDS,
        *HARDWARE_SALES_PRICE_FIELDS,
    ]


def build_hardware_cycle_factor_frame(
    config: QlibProviderConfig,
    *,
    start: str,
    end: str,
    proxy_return_windows: Iterable[int] = (5, 20, 60),
    zscore_windows: Iterable[int] = (90,),
    cloud_return_windows: Iterable[int] = (21,),
    lag_days: int = 1,
) -> pd.DataFrame:
    """Build lagged hardware-cycle factors from the hardware qlib provider."""

    if lag_days < 1:
        raise ValueError("hardware-cycle factors must use lag_days >= 1 for open-to-open execution")
    fields = _available_fields(config, start=start, end=end)
    query_fields = [f"${field}" for field in fields]
    qlib_frame = load_qlib_feature_frame(
        config,
        instruments=[HARDWARE_CYCLE_INSTRUMENT],
        fields=query_fields,
        start=start,
        end=end,
    )
    panel = qlib_frame.reset_index().copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"]).dt.normalize()
    panel = panel.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    panel = panel.rename(columns={field: field.removeprefix("$") for field in query_fields if field in panel.columns})
    for field in fields:
        if field in panel:
            panel[field] = pd.to_numeric(panel[field], errors="coerce")

    feature_data: dict[str, pd.Series] = {"datetime": panel["datetime"]}
    feature_cols: list[str] = []
    for field in HARDWARE_LEVEL_FIELDS:
        if field not in panel:
            continue
        for window in tuple(int(item) for item in proxy_return_windows):
            col = f"hardware_{field.removesuffix('_index')}_ret{window}_lag{lag_days}"
            feature_data[col] = _log_return(panel[field], window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        for window in tuple(int(item) for item in zscore_windows):
            col = f"hardware_{field.removesuffix('_index')}_z{window}_lag{lag_days}"
            feature_data[col] = _zscore(panel[field], window=window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)

    for field in HARDWARE_COUNT_FIELDS:
        if field not in panel:
            continue
        col = f"hardware_{field}_lag{lag_days}"
        feature_data[col] = panel[field].shift(lag_days).fillna(0.0)
        feature_cols.append(col)

    for field in CLOUD_GPU_DAILY_PROXY_FIELDS:
        if field not in panel:
            continue
        feature_base = field.removesuffix("_index")
        for window in tuple(int(item) for item in proxy_return_windows):
            col = f"hardware_{feature_base}_ret{window}_lag{lag_days}"
            feature_data[col] = _log_return(panel[field], window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        for window in tuple(int(item) for item in zscore_windows):
            col = f"hardware_{feature_base}_z{window}_lag{lag_days}"
            feature_data[col] = _zscore(panel[field], window=window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        availability_col = f"hardware_{feature_base}_available_lag{lag_days}"
        feature_data[availability_col] = panel[field].notna().astype("float64").shift(lag_days).fillna(0.0)
        feature_cols.append(availability_col)

    for field in CLOUD_GPU_DAILY_PROXY_COUNT_FIELDS:
        if field not in panel:
            continue
        col = f"hardware_{field}_lag{lag_days}"
        feature_data[col] = panel[field].shift(lag_days).fillna(0.0)
        feature_cols.append(col)

    for field in HARDWARE_DAILY_DERIVED_FIELDS:
        if field not in panel:
            continue
        feature_base = field.removesuffix("_index")
        for window in tuple(int(item) for item in proxy_return_windows):
            col = f"hardware_{feature_base}_ret{window}_lag{lag_days}"
            feature_data[col] = _log_return(panel[field], window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        for window in tuple(int(item) for item in zscore_windows):
            col = f"hardware_{feature_base}_z{window}_lag{lag_days}"
            feature_data[col] = _zscore(panel[field], window=window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        availability_col = f"hardware_{feature_base}_available_lag{lag_days}"
        feature_data[availability_col] = panel[field].notna().astype("float64").shift(lag_days).fillna(0.0)
        feature_cols.append(availability_col)

    free_gpu_fields = [field for field in fields if _is_free_gpu_price_field(field) and field in panel]
    for field in free_gpu_fields:
        for window in tuple(int(item) for item in proxy_return_windows):
            col = f"hardware_{field}_ret{window}_lag{lag_days}"
            feature_data[col] = _log_return(panel[field], window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        for window in tuple(int(item) for item in zscore_windows):
            col = f"hardware_{field}_z{window}_lag{lag_days}"
            feature_data[col] = _zscore(panel[field], window=window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        for state_suffix in ("available", "source_observed", "provider_count", "quote_count"):
            state_field = f"{field}_{state_suffix}"
            if state_field not in panel:
                continue
            state_col = f"hardware_{state_field}_lag{lag_days}"
            feature_data[state_col] = pd.to_numeric(panel[state_field], errors="coerce").shift(lag_days).fillna(0.0)
            feature_cols.append(state_col)

    hardware_sales_fields = [field for field in fields if _is_hardware_sales_price_field(field) and field in panel]
    for field in hardware_sales_fields:
        for window in tuple(int(item) for item in cloud_return_windows):
            col = f"hardware_{field}_ret{window}_lag{lag_days}"
            feature_data[col] = _log_return(panel[field], window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        for window in tuple(int(item) for item in zscore_windows):
            col = f"hardware_{field}_z{window}_lag{lag_days}"
            feature_data[col] = _zscore(panel[field], window=window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        for state_suffix in ("available", "source_observed", "is_carried", "staleness_days"):
            state_field = f"{field}_{state_suffix}"
            if state_field not in panel:
                continue
            state_col = f"hardware_{state_field}_lag{lag_days}"
            feature_data[state_col] = pd.to_numeric(panel[state_field], errors="coerce").shift(lag_days).fillna(0.0)
            feature_cols.append(state_col)

    cloud_fields = [field for field in fields if _is_cloud_gpu_level_field(field) and field in panel]
    for field in cloud_fields:
        available_field = f"{field}_available"
        available = panel[available_field] if available_field in panel else panel[field].notna().astype("float64")
        for window in tuple(int(item) for item in cloud_return_windows):
            col = f"hardware_{field}_ret{window}_lag{lag_days}"
            feature_data[col] = _log_return(panel[field], window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        for window in tuple(int(item) for item in zscore_windows):
            col = f"hardware_{field}_z{window}_lag{lag_days}"
            feature_data[col] = _zscore(panel[field], window=window).shift(lag_days).fillna(0.0)
            feature_cols.append(col)
        availability_col = f"hardware_{field}_available_lag{lag_days}"
        feature_data[availability_col] = pd.to_numeric(available, errors="coerce").shift(lag_days).fillna(0.0)
        feature_cols.append(availability_col)
        for state_suffix in ("source_observed", "is_carried", "staleness_days"):
            state_field = f"{field}_{state_suffix}"
            if state_field not in panel:
                continue
            state_col = f"hardware_{state_field}_lag{lag_days}"
            feature_data[state_col] = pd.to_numeric(panel[state_field], errors="coerce").shift(lag_days).fillna(0.0)
            feature_cols.append(state_col)

    feature_frame = pd.DataFrame(feature_data).replace([np.inf, -np.inf], np.nan)
    feature_frame.attrs["feature_cols"] = feature_cols
    feature_frame.attrs["source"] = HARDWARE_CYCLE_FACTOR_SOURCE
    feature_frame.attrs["provider_uri"] = config.resolved_provider_uri.as_posix()
    feature_frame.attrs["instrument"] = HARDWARE_CYCLE_INSTRUMENT
    feature_frame.attrs["lag_days"] = int(lag_days)
    feature_frame.attrs["limitations"] = HARDWARE_CYCLE_LIMITATIONS
    return feature_frame


def _is_cloud_gpu_level_field(field: str) -> bool:
    return (
        field.startswith("cloud_gpu_")
        and field not in CLOUD_GPU_DAILY_PROXY_FIELDS
        and field not in CLOUD_GPU_DAILY_PROXY_COUNT_FIELDS
        and not field.endswith(CLOUD_GPU_STATE_SUFFIXES)
    )


def _is_free_gpu_price_field(field: str) -> bool:
    return (
        field.startswith("free_cloud_gpu_")
        and field.endswith("_price")
        and not field.endswith(FREE_GPU_PRICE_STATE_SUFFIXES)
    )


def _is_hardware_sales_price_field(field: str) -> bool:
    return field in HARDWARE_SALES_PRICE_FIELDS

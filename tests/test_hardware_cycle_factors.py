from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig
from src.data.hardware_cycle import HARDWARE_CYCLE_INSTRUMENT, build_hardware_cycle_qlib_store_from_frame
from src.factor.hardware_cycle import build_hardware_cycle_factor_frame


def test_hardware_cycle_factor_frame_lags_model_features(tmp_path: Path):
    dates = pd.date_range("2024-07-01", periods=8, freq="D")
    frame = pd.DataFrame(
        {
            "instrument": [HARDWARE_CYCLE_INSTRUMENT] * len(dates),
            "datetime": dates,
            "ai_accelerator_index": [100, 100, 100, 200, 200, 200, 200, 200],
            "memory_storage_index": [100, 100, 100, 100, 100, 100, 100, 100],
            "cloud_capex_index": [100, 101, 102, 103, 104, 105, 106, 107],
            "ai_server_index": [100, 98, 97, 99, 101, 102, 103, 104],
            "cloud_gpu_h100_on_demand": [7.0, 7.0, 7.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            "cloud_gpu_h100_on_demand_available": [1] * len(dates),
        }
    )
    manifest = build_hardware_cycle_qlib_store_from_frame(
        frame,
        provider_uri=tmp_path / "hardware_cycle",
        as_of="2024-07-08",
    )

    factors = build_hardware_cycle_factor_frame(
        QlibProviderConfig(provider_uri=Path(manifest["provider_uri"]), region="us", market="all", kernels=1),
        start="2024-07-01",
        end="2024-07-08",
        proxy_return_windows=(1,),
        zscore_windows=(3,),
        cloud_return_windows=(1,),
        lag_days=1,
    )

    row_jump_day = factors.loc[factors["datetime"].eq(pd.Timestamp("2024-07-04"))].iloc[0]
    row_after_jump = factors.loc[factors["datetime"].eq(pd.Timestamp("2024-07-05"))].iloc[0]
    assert row_jump_day["hardware_ai_accelerator_ret1_lag1"] == 0.0
    assert np.isclose(row_after_jump["hardware_ai_accelerator_ret1_lag1"], np.log(2.0))
    assert row_jump_day["hardware_cloud_gpu_h100_on_demand_ret1_lag1"] == 0.0
    assert np.isclose(row_after_jump["hardware_cloud_gpu_h100_on_demand_ret1_lag1"], np.log(6.0 / 7.0))
    assert "hardware_ai_accelerator_z3_lag1" in factors.attrs["feature_cols"]


def test_hardware_cycle_factor_frame_lags_cloud_state_features(tmp_path: Path):
    dates = pd.date_range("2024-07-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "instrument": [HARDWARE_CYCLE_INSTRUMENT] * len(dates),
            "datetime": dates,
            "ai_accelerator_index": [100, 101, 102, 103, 104, 105],
            "memory_storage_index": [100, 100, 101, 101, 102, 102],
            "cloud_capex_index": [100, 100, 100, 101, 101, 101],
            "ai_server_index": [100, 99, 100, 101, 102, 103],
            "cloud_gpu_h100_on_demand": [7.0, 7.0, 7.0, 6.8, 6.8, 6.8],
            "cloud_gpu_h100_on_demand_available": [1, 1, 1, 1, 1, 1],
            "cloud_gpu_h100_on_demand_source_observed": [1, 0, 0, 1, 0, 0],
            "cloud_gpu_h100_on_demand_is_carried": [0, 1, 1, 0, 1, 1],
            "cloud_gpu_h100_on_demand_staleness_days": [0, 1, 2, 0, 1, 2],
        }
    )
    manifest = build_hardware_cycle_qlib_store_from_frame(
        frame,
        provider_uri=tmp_path / "hardware_cycle",
        as_of="2024-07-06",
    )

    factors = build_hardware_cycle_factor_frame(
        QlibProviderConfig(provider_uri=Path(manifest["provider_uri"]), region="us", market="all", kernels=1),
        start="2024-07-01",
        end="2024-07-06",
        proxy_return_windows=(1,),
        zscore_windows=(3,),
        cloud_return_windows=(1,),
        lag_days=1,
    )

    feature_cols = set(factors.attrs["feature_cols"])
    assert "hardware_cloud_gpu_h100_on_demand_source_observed_lag1" in feature_cols
    assert "hardware_cloud_gpu_h100_on_demand_is_carried_lag1" in feature_cols
    assert "hardware_cloud_gpu_h100_on_demand_staleness_days_lag1" in feature_cols

    row_after_update = factors.loc[factors["datetime"].eq(pd.Timestamp("2024-07-02"))].iloc[0]
    row_after_carried = factors.loc[factors["datetime"].eq(pd.Timestamp("2024-07-03"))].iloc[0]
    assert row_after_update["hardware_cloud_gpu_h100_on_demand_source_observed_lag1"] == 1.0
    assert row_after_update["hardware_cloud_gpu_h100_on_demand_is_carried_lag1"] == 0.0
    assert row_after_update["hardware_cloud_gpu_h100_on_demand_staleness_days_lag1"] == 0.0
    assert row_after_carried["hardware_cloud_gpu_h100_on_demand_source_observed_lag1"] == 0.0
    assert row_after_carried["hardware_cloud_gpu_h100_on_demand_is_carried_lag1"] == 1.0
    assert row_after_carried["hardware_cloud_gpu_h100_on_demand_staleness_days_lag1"] == 1.0


def test_hardware_cycle_factor_frame_treats_cloud_gpu_daily_proxies_as_daily_indices(tmp_path: Path):
    dates = pd.date_range("2024-07-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "instrument": [HARDWARE_CYCLE_INSTRUMENT] * len(dates),
            "datetime": dates,
            "ai_accelerator_index": [100, 101, 102, 103, 104, 105],
            "memory_storage_index": [100, 100, 101, 101, 102, 102],
            "cloud_capex_index": [100, 100, 100, 101, 101, 101],
            "ai_server_index": [100, 99, 100, 101, 102, 103],
            "cloud_gpu_daily_hopper_proxy_index": [100, 100, 110, 110, 121, 121],
            "cloud_gpu_h100_on_demand": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            "cloud_gpu_h100_on_demand_available": [1, 1, 1, 1, 1, 1],
        }
    )
    manifest = build_hardware_cycle_qlib_store_from_frame(
        frame,
        provider_uri=tmp_path / "hardware_cycle",
        as_of="2024-07-06",
    )

    factors = build_hardware_cycle_factor_frame(
        QlibProviderConfig(provider_uri=Path(manifest["provider_uri"]), region="us", market="all", kernels=1),
        start="2024-07-01",
        end="2024-07-06",
        proxy_return_windows=(1,),
        zscore_windows=(3,),
        cloud_return_windows=(1,),
        lag_days=1,
    )

    feature_cols = set(factors.attrs["feature_cols"])
    assert "hardware_cloud_gpu_daily_hopper_proxy_ret1_lag1" in feature_cols
    assert "hardware_cloud_gpu_daily_hopper_proxy_z3_lag1" in feature_cols
    assert "hardware_cloud_gpu_daily_hopper_proxy_available_lag1" in feature_cols
    assert "hardware_cloud_gpu_daily_hopper_proxy_index_ret1_lag1" not in feature_cols
    row_after_daily_move = factors.loc[factors["datetime"].eq(pd.Timestamp("2024-07-04"))].iloc[0]
    assert np.isclose(row_after_daily_move["hardware_cloud_gpu_daily_hopper_proxy_ret1_lag1"], np.log(110.0 / 100.0))


def test_hardware_cycle_factor_frame_lags_free_daily_gpu_price_observations(tmp_path: Path):
    dates = pd.date_range("2026-03-27", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "instrument": [HARDWARE_CYCLE_INSTRUMENT] * len(dates),
            "datetime": dates,
            "ai_accelerator_index": [100, 101, 102, 103, 104, 105],
            "memory_storage_index": [100, 100, 101, 101, 102, 102],
            "cloud_capex_index": [100, 100, 100, 101, 101, 101],
            "ai_server_index": [100, 99, 100, 101, 102, 103],
            "free_cloud_gpu_h100_price": [4.0, 3.9, 3.8, float("nan"), 3.7, float("nan")],
            "free_cloud_gpu_h100_price_available": [1, 1, 1, 0, 1, 0],
            "free_cloud_gpu_h100_price_source_observed": [1, 1, 1, 0, 1, 0],
            "free_cloud_gpu_h100_price_provider_count": [3, 3, 4, 0, 4, 0],
            "free_cloud_gpu_h100_price_quote_count": [5, 5, 6, 0, 6, 0],
        }
    )
    manifest = build_hardware_cycle_qlib_store_from_frame(
        frame,
        provider_uri=tmp_path / "hardware_cycle",
        as_of="2026-04-01",
    )

    factors = build_hardware_cycle_factor_frame(
        QlibProviderConfig(provider_uri=Path(manifest["provider_uri"]), region="us", market="all", kernels=1),
        start="2026-03-27",
        end="2026-04-01",
        proxy_return_windows=(1,),
        zscore_windows=(3,),
        cloud_return_windows=(1,),
        lag_days=1,
    )

    feature_cols = set(factors.attrs["feature_cols"])
    assert "hardware_free_cloud_gpu_h100_price_ret1_lag1" in feature_cols
    assert "hardware_free_cloud_gpu_h100_price_available_lag1" in feature_cols
    assert "hardware_free_cloud_gpu_h100_price_provider_count_lag1" in feature_cols
    row = factors.loc[factors["datetime"].eq(pd.Timestamp("2026-03-29"))].iloc[0]
    assert np.isclose(row["hardware_free_cloud_gpu_h100_price_ret1_lag1"], np.log(3.9 / 4.0))
    assert row["hardware_free_cloud_gpu_h100_price_available_lag1"] == 1.0

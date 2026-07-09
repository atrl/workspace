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

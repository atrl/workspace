from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig
from src.data.macro_state import MACRO_STATE_INSTRUMENT, build_macro_state_qlib_store_from_frame
from src.factor.macro_state import (
    BROAD_DOLLAR_FEATURES,
    RATE_VOL_FEATURES,
    REAL_RATE_FEATURES,
    build_macro_state_factor_frame,
)


def test_macro_state_factor_frame_uses_qlib_release_mapped_state(tmp_path: Path):
    dates = pd.date_range("2024-01-02", periods=300, freq="B")
    trend = np.arange(len(dates), dtype="float64")
    frame = pd.DataFrame(
        {
            "instrument": MACRO_STATE_INSTRUMENT,
            "datetime": dates,
            "dgs2": 4.0 + trend * 0.001,
            "dgs10": 4.5 + trend * 0.0015,
            "dfii10": 1.7 + trend * 0.0005,
            "t5y5y_inflation_proxy": 2.1 + trend * 0.0002,
            "dtwexbgs": 115.0 + trend * 0.02,
        }
    )
    for field in ["dgs2", "dgs10", "dfii10", "t5y5y_inflation_proxy", "dtwexbgs"]:
        frame[f"{field}_source_observed"] = 1.0
        frame[f"{field}_is_carried"] = 0.0
        frame[f"{field}_staleness_days"] = 0.0
        frame[f"{field}_publication_lag_days"] = 1.0 if field != "dtwexbgs" else 2.0
        frame[f"{field}_observation_epoch_days"] = (dates - pd.Timestamp("1970-01-01")).days
    manifest = build_macro_state_qlib_store_from_frame(
        frame,
        provider_uri=tmp_path / "macro_state",
        as_of=dates[-1].date().isoformat(),
    )
    factors = build_macro_state_factor_frame(
        QlibProviderConfig(provider_uri=Path(manifest["provider_uri"]), region="us", market="all", kernels=1),
        start=dates[0].date().isoformat(),
        end=dates[-1].date().isoformat(),
    )

    expected = {*REAL_RATE_FEATURES, *BROAD_DOLLAR_FEATURES, *RATE_VOL_FEATURES}
    assert set(factors.attrs["feature_cols"]) == expected
    assert factors[list(expected)].notna().all().all()
    last = factors.iloc[-1]
    assert last["macro_dfii10_chg5"] > 0
    assert last["macro_dtwexbgs_ret20"] > 0
    assert last["macro_curve_10y2y_chg20"] > 0

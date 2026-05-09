from __future__ import annotations

import pandas as pd

from scripts.update_market_data import build_local_surface_frames


def test_build_local_surface_frames_derives_raw_and_external_views():
    master = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-24", "2026-04-27"]),
            "soxl_open": [30.0, 31.0],
            "soxl_high": [31.0, 32.0],
            "soxl_low": [29.0, 30.0],
            "soxl_close": [30.5, 31.5],
            "soxl_volume": [1_000, 2_000],
            "soxx_open": [200.0, 201.0],
            "soxx_high": [202.0, 203.0],
            "soxx_low": [198.0, 199.0],
            "soxx_close": [201.0, 202.0],
            "soxx_volume": [3_000, 4_000],
            "vix": [25.0, 24.5],
        }
    )

    frames = build_local_surface_frames(master)

    assert sorted(frames) == [
        "data/raw/external/VIX.csv",
        "data/raw/symbols/SOXL.csv",
        "data/raw/symbols/SOXX.csv",
    ]

    soxl = frames["data/raw/symbols/SOXL.csv"]
    assert list(soxl.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert soxl.iloc[-1].to_dict() == {
        "date": "2026-04-27",
        "open": 31.0,
        "high": 32.0,
        "low": 30.0,
        "close": 31.5,
        "volume": 2000,
    }

    soxx = frames["data/raw/symbols/SOXX.csv"]
    assert soxx.iloc[0].to_dict() == {
        "date": "2026-04-24",
        "open": 200.0,
        "high": 202.0,
        "low": 198.0,
        "close": 201.0,
        "volume": 3000,
    }

    vix = frames["data/raw/external/VIX.csv"]
    assert list(vix.columns) == ["DATE", "CLOSE"]
    assert vix.iloc[-1].to_dict() == {"DATE": "2026-04-27", "CLOSE": 24.5}

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts import update_market_data
from scripts.update_market_data import build_local_surface_frames, update_master_csv


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


def test_update_master_csv_polygon_provider_uses_cboe_vix():
    with TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "panel.csv"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-05-15"]),
                "soxl_open": [140.0],
                "soxl_high": [145.0],
                "soxl_low": [138.0],
                "soxl_close": [142.0],
                "soxl_volume": [10_000],
                "soxx_open": [250.0],
                "soxx_high": [252.0],
                "soxx_low": [248.0],
                "soxx_close": [251.0],
                "soxx_volume": [20_000],
                "vix": [20.0],
            }
        ).to_csv(csv_path, index=False)

        calls = []

        def fake_fetch_polygon(ticker: str, start: str | None = None):
            calls.append((ticker, start))
            prefix = ticker.lower()
            volume = [11_000, 12_000] if prefix == "soxl" else [21_000, 22_000]
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-05-18", "2026-05-19"]),
                    "open": [150.0, 151.0],
                    "high": [152.0, 153.0],
                    "low": [149.0, 150.0],
                    "close": [151.0, 152.0],
                    "volume": volume,
                }
            )

        def fake_fetch_cboe_vix(start: str | None = None):
            calls.append(("CBOE_VIX", start))
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-05-18", "2026-05-19"]),
                    "vix": [18.5, 18.0],
                }
            )

        original_fetch_polygon = update_market_data._fetch_polygon
        original_fetch_cboe_vix = update_market_data._fetch_cboe_vix
        update_market_data._fetch_polygon = fake_fetch_polygon
        update_market_data._fetch_cboe_vix = fake_fetch_cboe_vix
        try:
            result = update_master_csv(csv_path=csv_path, dry_run=False, provider="polygon")
        finally:
            update_market_data._fetch_polygon = original_fetch_polygon
            update_market_data._fetch_cboe_vix = original_fetch_cboe_vix

        assert result["new_rows"] == 2
        assert result["last_date"] == "2026-05-19"
        assert calls == [
            ("SOXL", "2026-05-05"),
            ("SOXX", "2026-05-05"),
            ("CBOE_VIX", "2026-05-05"),
        ]

        updated = pd.read_csv(csv_path, parse_dates=["date"])
        assert updated["date"].dt.strftime("%Y-%m-%d").tolist() == [
            "2026-05-15",
            "2026-05-18",
            "2026-05-19",
        ]
        assert updated.iloc[-1]["vix"] == 18.0

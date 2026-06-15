from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts import update_market_data
from scripts import update_data_registry
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


def test_build_local_surface_frames_can_extend_raw_symbols_beyond_vix_overlap():
    master = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-12"]),
            "soxl_open": [222.22],
            "soxl_high": [242.40],
            "soxl_low": [214.7406],
            "soxl_close": [234.68],
            "soxl_volume": [48_833_409],
            "soxx_open": [584.64],
            "soxx_high": [602.69],
            "soxx_low": [578.53],
            "soxx_close": [596.25],
            "soxx_volume": [9_057_945],
            "vix": [17.68],
        }
    )
    symbol_frames = {
        "SOXL": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-12", "2026-06-15"]),
                "open": [222.22, 265.91],
                "high": [242.40, 272.18],
                "low": [214.7406, 261.6019],
                "close": [234.68, 271.44],
                "volume": [48_833_409, 24_818_910],
            }
        ),
        "SOXX": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-12", "2026-06-15"]),
                "open": [584.64, 623.25],
                "high": [602.69, 627.47],
                "low": [578.53, 618.45],
                "close": [596.25, 626.77],
                "volume": [9_057_945, 4_381_522],
            }
        ),
    }

    frames = build_local_surface_frames(master, symbol_frames=symbol_frames)

    soxl = frames["data/raw/symbols/SOXL.csv"]
    soxx = frames["data/raw/symbols/SOXX.csv"]
    vix = frames["data/raw/external/VIX.csv"]
    assert soxl["date"].tolist() == ["2026-06-12", "2026-06-15"]
    assert soxl.iloc[-1]["close"] == 271.44
    assert soxx.iloc[-1]["close"] == 626.77
    assert vix["DATE"].tolist() == ["2026-06-12"]


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


def test_normalize_futu_kline_frame_maps_ohlcv_fields():
    frame = pd.DataFrame(
        {
            "time_key": ["2026-06-15 00:00:00"],
            "open": [265.91],
            "high": [272.18],
            "low": [261.6019],
            "close": [271.44],
            "volume": [24_818_910],
            "turnover": [6_619_473_000.0],
        }
    )

    result = update_market_data._normalize_futu_kline_frame(frame)

    assert result.to_dict("records") == [
        {
            "date": pd.Timestamp("2026-06-15"),
            "open": 265.91,
            "high": 272.18,
            "low": 261.6019,
            "close": 271.44,
            "volume": 24_818_910,
        }
    ]


def test_update_master_csv_futu_provider_uses_futu_daily_and_cboe_vix():
    with TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "panel.csv"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-12"]),
                "soxl_open": [222.22],
                "soxl_high": [242.40],
                "soxl_low": [214.7406],
                "soxl_close": [234.68],
                "soxl_volume": [50_419_140],
                "soxx_open": [584.64],
                "soxx_high": [602.69],
                "soxx_low": [578.53],
                "soxx_close": [596.25],
                "soxx_volume": [9_947_173],
                "vix": [17.68],
            }
        ).to_csv(csv_path, index=False)

        calls = []

        def fake_fetch_futu_daily(code: str, start: str | None = None):
            calls.append((code, start))
            if code == "US.SOXL":
                return pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-06-15"]),
                        "open": [265.91],
                        "high": [272.18],
                        "low": [261.6019],
                        "close": [271.44],
                        "volume": [24_818_910],
                    }
                )
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-06-15"]),
                    "open": [623.25],
                    "high": [627.47],
                    "low": [618.45],
                    "close": [626.77],
                    "volume": [4_381_522],
                }
            )

        def fake_fetch_cboe_vix(start: str | None = None):
            calls.append(("CBOE_VIX", start))
            return pd.DataFrame({"date": pd.to_datetime(["2026-06-15"]), "vix": [16.95]})

        original_fetch_futu_daily = update_market_data._fetch_futu_daily
        original_fetch_cboe_vix = update_market_data._fetch_cboe_vix
        update_market_data._fetch_futu_daily = fake_fetch_futu_daily
        update_market_data._fetch_cboe_vix = fake_fetch_cboe_vix
        try:
            result = update_master_csv(csv_path=csv_path, dry_run=False, provider="futu")
        finally:
            update_market_data._fetch_futu_daily = original_fetch_futu_daily
            update_market_data._fetch_cboe_vix = original_fetch_cboe_vix

        assert result["new_rows"] == 1
        assert result["last_date"] == "2026-06-15"
        assert calls == [
            ("US.SOXL", "2026-06-02"),
            ("US.SOXX", "2026-06-02"),
            ("CBOE_VIX", "2026-06-02"),
        ]

        updated = pd.read_csv(csv_path, parse_dates=["date"])
        assert updated["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-06-12", "2026-06-15"]
        assert updated.iloc[-1]["soxl_close"] == 271.44
        assert updated.iloc[-1]["soxx_close"] == 626.77
        assert updated.iloc[-1]["vix"] == 16.95


def test_update_master_csv_reports_provider_leg_coverage_when_vix_lags():
    with TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "panel.csv"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-05-19"]),
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

        def fake_fetch_polygon(ticker: str, start: str | None = None):
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-05-19", "2026-05-20"]),
                    "open": [150.0, 151.0],
                    "high": [152.0, 153.0],
                    "low": [149.0, 150.0],
                    "close": [151.0, 152.0],
                    "volume": [11_000, 12_000],
                }
            )

        def fake_fetch_cboe_vix(start: str | None = None):
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-05-19"]),
                    "vix": [18.0],
                }
            )

        original_fetch_polygon = update_market_data._fetch_polygon
        original_fetch_cboe_vix = update_market_data._fetch_cboe_vix
        update_market_data._fetch_polygon = fake_fetch_polygon
        update_market_data._fetch_cboe_vix = fake_fetch_cboe_vix
        try:
            result = update_master_csv(csv_path=csv_path, dry_run=True, provider="polygon")
        finally:
            update_market_data._fetch_polygon = original_fetch_polygon
            update_market_data._fetch_cboe_vix = original_fetch_cboe_vix

        assert result["source_ranges"] == {
            "SOXL": {"rows": 2, "start_date": "2026-05-19", "last_date": "2026-05-20"},
            "SOXX": {"rows": 2, "start_date": "2026-05-19", "last_date": "2026-05-20"},
            "VIX": {"rows": 1, "start_date": "2026-05-19", "last_date": "2026-05-19"},
        }
        assert result["overlap_last_date"] == "2026-05-19"


def test_update_master_csv_normalizes_provider_dates_before_merge():
    with TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "panel.csv"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-05-19"]),
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

        def fake_fetch_market_frames(provider: str, start: str | None):
            assert provider == "yfinance"
            assert start == "2026-05-09"
            soxl = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-05-20 00:00:00-04:00"]),
                    "open": [150.0],
                    "high": [152.0],
                    "low": [149.0],
                    "close": [151.0],
                    "volume": [11_000],
                }
            )
            soxx = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-05-20"]),
                    "open": [500.0],
                    "high": [502.0],
                    "low": [498.0],
                    "close": [501.0],
                    "volume": [21_000],
                }
            )
            vix = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-05-20 16:00:00"]),
                    "vix": [17.0],
                }
            )
            return soxl, soxx, vix

        original_fetch_market_frames = update_market_data._fetch_market_frames
        update_market_data._fetch_market_frames = fake_fetch_market_frames
        try:
            result = update_master_csv(csv_path=csv_path, dry_run=False, provider="yfinance")
        finally:
            update_market_data._fetch_market_frames = original_fetch_market_frames

        assert result["new_rows"] == 1
        assert result["last_date"] == "2026-05-20"
        assert result["overlap_last_date"] == "2026-05-20"

        updated = pd.read_csv(csv_path, parse_dates=["date"])
        assert updated["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-05-19", "2026-05-20"]
        assert updated.iloc[-1]["soxl_close"] == 151.0
        assert updated.iloc[-1]["soxx_close"] == 501.0
        assert updated.iloc[-1]["vix"] == 17.0


def test_data_registry_check_reports_stale_canonical_panel_end():
    stale_registry = {
        "symbols": {"SOXL": {"end": "2026-05-19"}},
        "processed_panels": {"soxl_soxx_vix_full": {"end": "2026-05-19"}},
        "qlib": {"SOXL": {"exists": True}},
    }
    current_registry = {
        "symbols": {"SOXL": {"end": "2026-05-20"}},
        "processed_panels": {"soxl_soxx_vix_full": {"end": "2026-05-20"}},
        "qlib": {"SOXL": {"exists": True}},
    }

    failures = update_data_registry.check_registry(stale_registry, current_registry=current_registry)

    assert "stale registry processed_panels.soxl_soxx_vix_full.end: stored=2026-05-19 current=2026-05-20" in failures

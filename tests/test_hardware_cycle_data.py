from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.data.hardware_cycle import (
    HARDWARE_CYCLE_INSTRUMENT,
    HARDWARE_PROXY_SYMBOLS,
    build_hardware_cycle_daily_frame,
    build_hardware_cycle_qlib_store_from_frame,
    extract_aimultiple_gpu_index_series,
    fetch_hardware_proxy_price_frames,
)


def test_extract_aimultiple_gpu_index_series_from_embedded_chart():
    html = r'''
    <script>
    self.__next_f.push([1,"{\"id\":\"nivo-test\",\"type\":\"chart-line\",\"data\":{\"data\":[{\"id\":\"Modern\",\"category\":\"modern\",\"billing_type\":\"ON_DEMAND\",\"data\":[{\"x\":\"2024-07-01\",\"y\":2.5},{\"x\":\"2024-08-01\",\"y\":2.75}]},{\"id\":\"H100\",\"gpu_model\":\"nvidia-h100\",\"data\":[{\"x\":\"2024-07-01\",\"y\":7.1}]}]}}"]);
    </script>
    '''

    frame = extract_aimultiple_gpu_index_series(html)

    assert set(frame["field"]) == {
        "cloud_gpu_modern_on_demand",
        "cloud_gpu_h100_on_demand",
    }
    assert frame.loc[frame["field"].eq("cloud_gpu_modern_on_demand"), "value"].tolist() == [2.5, 2.75]
    assert frame["source"].eq("aimultiple_cloud_gpu_index").all()


def test_build_hardware_cycle_daily_frame_keeps_cloud_index_availability():
    dates = pd.date_range("2024-06-28", "2024-07-05", freq="D")
    proxy_frames = _proxy_frames(dates)
    cloud_gpu = pd.DataFrame(
        [
            {"datetime": "2024-07-01", "field": "cloud_gpu_h100_on_demand", "value": 7.0},
            {"datetime": "2024-08-01", "field": "cloud_gpu_h100_on_demand", "value": 6.5},
        ]
    )

    frame = build_hardware_cycle_daily_frame(
        proxy_frames,
        cloud_gpu,
        calendar=[date.date().isoformat() for date in dates],
    )

    assert frame["instrument"].unique().tolist() == [HARDWARE_CYCLE_INSTRUMENT]
    assert frame.loc[frame["datetime"].eq(pd.Timestamp("2024-06-30")), "cloud_gpu_h100_on_demand_available"].iloc[0] == 0.0
    assert frame.loc[frame["datetime"].eq(pd.Timestamp("2024-07-01")), "cloud_gpu_h100_on_demand_available"].iloc[0] == 1.0
    assert frame.loc[frame["datetime"].eq(pd.Timestamp("2024-07-05")), "cloud_gpu_h100_on_demand"].iloc[0] == 7.0
    assert frame["ai_accelerator_index"].notna().all()
    assert frame["memory_storage_index"].notna().all()
    assert frame["cloud_capex_index"].notna().all()


def test_build_hardware_cycle_daily_frame_marks_cloud_state_updates_and_staleness():
    dates = pd.date_range("2024-06-28", "2024-07-08", freq="B")
    cloud_gpu = pd.DataFrame(
        [
            {"datetime": "2024-07-01", "field": "cloud_gpu_h100_on_demand", "value": 7.0},
            {"datetime": "2024-07-06", "field": "cloud_gpu_h100_on_demand", "value": 6.8},
        ]
    )

    frame = build_hardware_cycle_daily_frame(
        _proxy_frames(dates),
        cloud_gpu,
        calendar=[date.date().isoformat() for date in dates],
    )

    before = frame.loc[frame["datetime"].eq(pd.Timestamp("2024-06-28"))].iloc[0]
    first_update = frame.loc[frame["datetime"].eq(pd.Timestamp("2024-07-01"))].iloc[0]
    carried = frame.loc[frame["datetime"].eq(pd.Timestamp("2024-07-02"))].iloc[0]
    weekend_update = frame.loc[frame["datetime"].eq(pd.Timestamp("2024-07-08"))].iloc[0]

    assert before["cloud_gpu_h100_on_demand_available"] == 0.0
    assert before["cloud_gpu_h100_on_demand_source_observed"] == 0.0
    assert before["cloud_gpu_h100_on_demand_is_carried"] == 0.0
    assert pd.isna(before["cloud_gpu_h100_on_demand_staleness_days"])
    assert first_update["cloud_gpu_h100_on_demand"] == 7.0
    assert first_update["cloud_gpu_h100_on_demand_available"] == 1.0
    assert first_update["cloud_gpu_h100_on_demand_source_observed"] == 1.0
    assert first_update["cloud_gpu_h100_on_demand_is_carried"] == 0.0
    assert first_update["cloud_gpu_h100_on_demand_staleness_days"] == 0.0
    assert carried["cloud_gpu_h100_on_demand"] == 7.0
    assert carried["cloud_gpu_h100_on_demand_source_observed"] == 0.0
    assert carried["cloud_gpu_h100_on_demand_is_carried"] == 1.0
    assert carried["cloud_gpu_h100_on_demand_staleness_days"] == 1.0
    assert weekend_update["cloud_gpu_h100_on_demand"] == 6.8
    assert weekend_update["cloud_gpu_h100_on_demand_source_observed"] == 1.0
    assert weekend_update["cloud_gpu_h100_on_demand_is_carried"] == 0.0
    assert weekend_update["cloud_gpu_h100_on_demand_staleness_days"] == 0.0


def test_build_hardware_cycle_qlib_store_from_frame_reads_back(tmp_path: Path):
    dates = pd.date_range("2024-07-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "instrument": [HARDWARE_CYCLE_INSTRUMENT] * len(dates),
            "datetime": dates,
            "ai_accelerator_index": [100, 101, 102, 103, 104, 105],
            "memory_storage_index": [100, 99, 98, 99, 100, 101],
            "cloud_capex_index": [100, 100, 101, 101, 102, 102],
            "ai_server_index": [100, 110, 105, 106, 107, 108],
            "cloud_gpu_h100_on_demand": [float("nan"), 7.0, 7.0, 7.0, 6.9, 6.9],
            "cloud_gpu_h100_on_demand_available": [0, 1, 1, 1, 1, 1],
        }
    )

    manifest = build_hardware_cycle_qlib_store_from_frame(
        frame,
        provider_uri=tmp_path / "hardware_cycle",
        as_of="2024-07-06",
    )
    readback = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=Path(manifest["provider_uri"]), region="us", market="all", kernels=1),
        instruments=[HARDWARE_CYCLE_INSTRUMENT],
        fields=["$ai_accelerator_index", "$cloud_gpu_h100_on_demand_available"],
        start="2024-07-01",
        end="2024-07-06",
    )

    assert manifest["status"] == "ok"
    assert readback.reset_index()["$ai_accelerator_index"].dropna().tolist() == [100, 101, 102, 103, 104, 105]


def test_fetch_hardware_proxy_price_frames_skips_unavailable_optional_symbol(monkeypatch):
    def fake_fetch(symbol: str, *, start: str, end: str) -> pd.DataFrame:
        if symbol == "PSTG":
            raise RuntimeError("yahoo chart failed for PSTG")
        return pd.DataFrame(
            {
                "date": pd.date_range(start, periods=2, freq="D"),
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000, 1100],
            }
        )

    monkeypatch.setattr("src.data.hardware_cycle._fetch_yahoo_chart_ohlcv", fake_fetch)

    frames = fetch_hardware_proxy_price_frames(
        start="2024-07-01",
        end="2024-07-02",
        symbols=["NVDA", "PSTG", "NTAP"],
    )

    assert sorted(frames) == ["NTAP", "NVDA"]


def _proxy_frames(dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    return {
        symbol: pd.DataFrame(
            {
                "date": dates,
                "close": [100.0 + idx + offset for idx in range(len(dates))],
                "volume": [1000 + idx for idx in range(len(dates))],
            }
        )
        for offset, symbol in enumerate(HARDWARE_PROXY_SYMBOLS)
    }

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
    extract_bls_hardware_sales_price_payload,
    extract_fred_hardware_sales_price_series,
    extract_freellm_gpu_price_snapshot,
    fetch_bls_hardware_sales_price_series,
    fetch_fred_hardware_sales_price_series,
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


def test_build_hardware_cycle_daily_frame_adds_trading_day_cloud_gpu_proxy_features():
    dates = pd.date_range("2024-07-01", "2024-07-10", freq="B")
    cloud_gpu = pd.DataFrame(
        [
            {"datetime": "2024-07-01", "field": "cloud_gpu_h100_on_demand", "value": 7.0},
        ]
    )

    frame = build_hardware_cycle_daily_frame(
        _proxy_frames(dates),
        cloud_gpu,
        calendar=[date.date().isoformat() for date in dates],
    )

    assert "cloud_gpu_daily_hopper_proxy_index" in frame.columns
    assert "cloud_gpu_daily_server_supply_proxy_index" in frame.columns
    assert "cloud_gpu_daily_supply_demand_pressure_index" in frame.columns
    assert frame["cloud_gpu_daily_hopper_proxy_index"].notna().all()
    assert frame["cloud_gpu_daily_hopper_proxy_index"].nunique() > 1
    assert frame["cloud_gpu_daily_supply_demand_pressure_index"].nunique() > 1
    assert "cloud_gpu_daily_hopper_proxy_index_is_carried" not in frame.columns
    assert frame.loc[frame["datetime"].eq(pd.Timestamp("2024-07-02")), "cloud_gpu_h100_on_demand_is_carried"].iloc[0] == 1.0


def test_build_hardware_cycle_daily_frame_adds_storage_and_interconnect_daily_proxy_features():
    dates = pd.date_range("2024-07-01", "2024-07-10", freq="B")

    frame = build_hardware_cycle_daily_frame(
        _proxy_frames(dates),
        cloud_gpu_index=pd.DataFrame(),
        calendar=[date.date().isoformat() for date in dates],
    )

    assert "storage_hdd_manufacturer_index" in frame.columns
    assert "storage_nand_ssd_index" in frame.columns
    assert "networking_interconnect_index" in frame.columns
    assert "storage_hdd_cloud_demand_pressure_index" in frame.columns
    assert "storage_hdd_vs_nand_index" in frame.columns
    assert "networking_cloud_demand_pressure_index" in frame.columns
    assert frame["storage_hdd_manufacturer_index"].notna().all()
    assert frame["storage_hdd_cloud_demand_pressure_index"].nunique() > 1


def test_extract_freellm_gpu_price_snapshot_builds_daily_direct_price_indices():
    rows = [
        {"provider": "lambda", "product": "NVIDIA H100 SXM", "price_per_hour_usd": 3.99},
        {"provider": "crusoe", "product": "NVIDIA H100", "price_per_hour_usd": 3.90},
        {"provider": "coreweave", "product": "NVIDIA HGX B200", "price_per_hour_usd": 8.60},
        {"provider": "runpod", "product": "NVIDIA RTX 4090", "secure_price": "0.44"},
        {"provider": "empty", "product": "NVIDIA H100", "price_per_hour_usd": None},
    ]

    frame = extract_freellm_gpu_price_snapshot(rows, snapshot_date="2026-03-28")

    h100 = frame.loc[frame["field"].eq("free_cloud_gpu_h100_price")].iloc[0]
    assert h100["datetime"] == pd.Timestamp("2026-03-28")
    assert h100["source"] == "freellm_gpu_cloud_pricing_history"
    assert h100["provider_count"] == 2
    assert h100["quote_count"] == 2
    assert h100["value"] == 3.945
    assert set(frame["field"]) == {
        "free_cloud_gpu_h100_price",
        "free_cloud_gpu_b200_price",
        "free_cloud_gpu_rtx_4090_price",
    }


def test_build_hardware_cycle_daily_frame_adds_free_daily_gpu_rental_price_observations():
    dates = pd.date_range("2026-03-27", "2026-04-02", freq="B")
    free_prices = pd.DataFrame(
        [
            {"datetime": "2026-03-27", "field": "free_cloud_gpu_h100_price", "value": 3.90},
            {"datetime": "2026-03-28", "field": "free_cloud_gpu_h100_price", "value": 3.80},
            {"datetime": "2026-03-30", "field": "free_cloud_gpu_h100_price", "value": 3.70},
        ]
    )

    frame = build_hardware_cycle_daily_frame(
        _proxy_frames(dates),
        cloud_gpu_index=pd.DataFrame(),
        free_gpu_price_history=free_prices,
        calendar=[date.date().isoformat() for date in dates],
    )

    friday = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-03-27"))].iloc[0]
    monday = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-03-30"))].iloc[0]
    tuesday = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-03-31"))].iloc[0]
    assert friday["free_cloud_gpu_h100_price"] == 3.90
    assert friday["free_cloud_gpu_h100_price_source_observed"] == 1.0
    assert monday["free_cloud_gpu_h100_price"] == 3.70
    assert monday["free_cloud_gpu_h100_price_source_observed"] == 1.0
    assert monday["free_cloud_gpu_h100_price_quote_count"] == 0.0
    assert pd.isna(tuesday["free_cloud_gpu_h100_price"])
    assert tuesday["free_cloud_gpu_h100_price_available"] == 0.0


def test_extract_fred_hardware_sales_price_series_marks_official_sales_price_index():
    csv_text = "observation_date,PCU334112334112\n2026-01-01,70.918\n2026-02-01,72.266\n2026-03-01,.\n"

    frame = extract_fred_hardware_sales_price_series(
        csv_text,
        series_id="PCU334112334112",
        field="hardware_sales_storage_device_ppi",
    )

    assert frame["field"].unique().tolist() == ["hardware_sales_storage_device_ppi"]
    assert frame["datetime"].tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]
    assert frame["value"].tolist() == [70.918, 72.266]
    assert frame["source"].eq("fred_bls_hardware_sales_price_index").all()
    assert frame["series_id"].eq("PCU334112334112").all()


def test_fetch_fred_hardware_sales_price_series_keeps_successful_series_when_one_request_fails(monkeypatch):
    class FakeResponse:
        text = "observation_date,GOOD\n2026-01-01,100.0\n2026-02-01,101.0\n"

        def raise_for_status(self) -> None:
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.headers = {}

        def get(self, url: str, *, params: dict[str, str], timeout: int) -> FakeResponse:
            if params["id"] == "BAD":
                raise RuntimeError("timeout")
            return FakeResponse()

    monkeypatch.setattr("src.data.hardware_cycle.requests.Session", FakeSession)

    frame = fetch_fred_hardware_sales_price_series(
        series_map={
            "hardware_sales_good_ppi": {"series_id": "GOOD"},
            "hardware_sales_bad_ppi": {"series_id": "BAD"},
        }
    )

    assert frame["field"].unique().tolist() == ["hardware_sales_good_ppi"]
    assert frame["value"].tolist() == [100.0, 101.0]
    assert frame.attrs["errors"] == {"hardware_sales_bad_ppi": "timeout"}


def test_extract_bls_hardware_sales_price_payload_marks_official_sales_price_index():
    payload = {
        "Results": {
            "series": [
                {
                    "seriesID": "PCU334112334112",
                    "data": [
                        {"year": "2026", "period": "M02", "value": "72.266"},
                        {"year": "2026", "period": "M01", "value": "70.918"},
                        {"year": "2026", "period": "M13", "value": "74.0"},
                    ],
                }
            ]
        }
    }

    frame = extract_bls_hardware_sales_price_payload(
        payload,
        series_id_to_field={"PCU334112334112": "hardware_sales_storage_device_ppi"},
    )

    assert frame["field"].unique().tolist() == ["hardware_sales_storage_device_ppi"]
    assert frame["datetime"].tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]
    assert frame["value"].tolist() == [70.918, 72.266]
    assert frame["source"].eq("bls_public_api_hardware_sales_price_index").all()


def test_fetch_bls_hardware_sales_price_series_batches_series_and_year_chunks(monkeypatch):
    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class FakeSession:
        def __init__(self) -> None:
            self.headers = {}
            self.requests: list[dict[str, object]] = []

        def post(self, url: str, *, json: dict[str, object], timeout: int) -> FakeResponse:
            self.requests.append(json)
            start_year = str(json["startyear"])
            return FakeResponse(
                {
                    "status": "REQUEST_SUCCEEDED",
                    "Results": {
                        "series": [
                            {
                                "seriesID": "PCU334112334112",
                                "data": [{"year": start_year, "period": "M01", "value": "100.0"}],
                            },
                            {
                                "seriesID": "WPU1154",
                                "data": [{"year": start_year, "period": "M01", "value": "200.0"}],
                            },
                        ]
                    },
                }
            )

    fake_session = FakeSession()
    monkeypatch.setattr("src.data.hardware_cycle.requests.Session", lambda: fake_session)

    frame = fetch_bls_hardware_sales_price_series(
        series_map={
            "hardware_sales_storage_device_ppi": {"series_id": "PCU334112334112"},
            "hardware_sales_computer_peripheral_parts_ppi": {"series_id": "WPU1154"},
        },
        start_year=2000,
        end_year=2012,
    )

    assert [request["startyear"] for request in fake_session.requests] == ["2000", "2010"]
    assert set(frame["field"]) == {
        "hardware_sales_storage_device_ppi",
        "hardware_sales_computer_peripheral_parts_ppi",
    }
    assert frame["datetime"].min() == pd.Timestamp("2000-01-01")
    assert frame["datetime"].max() == pd.Timestamp("2010-01-01")


def test_build_hardware_cycle_daily_frame_adds_official_sales_price_state_with_staleness():
    dates = pd.date_range("2026-01-30", "2026-02-05", freq="B")
    sales_prices = pd.DataFrame(
        [
            {"datetime": "2026-02-01", "field": "hardware_sales_storage_device_ppi", "value": 72.266},
        ]
    )

    frame = build_hardware_cycle_daily_frame(
        _proxy_frames(dates),
        cloud_gpu_index=pd.DataFrame(),
        hardware_sales_price_index=sales_prices,
        calendar=[date.date().isoformat() for date in dates],
    )

    before = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-01-30"))].iloc[0]
    observed = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-02-02"))].iloc[0]
    carried = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-02-03"))].iloc[0]
    assert pd.isna(before["hardware_sales_storage_device_ppi"])
    assert before["hardware_sales_storage_device_ppi_available"] == 0.0
    assert observed["hardware_sales_storage_device_ppi"] == 72.266
    assert observed["hardware_sales_storage_device_ppi_source_observed"] == 1.0
    assert observed["hardware_sales_storage_device_ppi_is_carried"] == 0.0
    assert observed["hardware_sales_storage_device_ppi_staleness_days"] == 0.0
    assert carried["hardware_sales_storage_device_ppi"] == 72.266
    assert carried["hardware_sales_storage_device_ppi_source_observed"] == 0.0
    assert carried["hardware_sales_storage_device_ppi_is_carried"] == 1.0
    assert carried["hardware_sales_storage_device_ppi_staleness_days"] == 1.0


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

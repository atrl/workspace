from __future__ import annotations

from pathlib import Path

import builtins
import pandas as pd
import pytest


def test_normalize_intraday_bars_emits_canonical_new_york_schema():
    from src.soxl_quant.app.intraday_data import normalize_intraday_bars

    raw = pd.DataFrame(
        [
            {
                "time_key": "2026-06-04 09:30:00",
                "open": "10.0",
                "high": "10.5",
                "low": "9.9",
                "close": "10.2",
                "volume": "1000",
                "turnover": "10200",
            }
        ]
    )

    result = normalize_intraday_bars(
        raw,
        provider="futu",
        symbol="SOXL",
        interval="1m",
        timestamp_column="time_key",
        source_tz="America/New_York",
        adjusted=True,
    )

    assert list(result.columns) == [
        "provider",
        "symbol",
        "interval",
        "timestamp",
        "date",
        "session",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "source_tz",
        "adjusted",
        "fetched_at",
    ]
    row = result.iloc[0]
    assert row["provider"] == "futu"
    assert row["symbol"] == "SOXL"
    assert row["interval"] == "1m"
    assert str(row["timestamp"]) == "2026-06-04 09:30:00-04:00"
    assert row["date"] == "2026-06-04"
    assert row["session"] == "regular"
    assert row["open"] == 10.0
    assert row["volume"] == 1000.0


def test_polygon_provider_normalizes_aggregate_payload(monkeypatch: pytest.MonkeyPatch):
    from src.soxl_quant.app import intraday_data

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "t": int(pd.Timestamp("2026-06-04 13:30:00", tz="UTC").timestamp() * 1000),
                        "o": 10,
                        "h": 11,
                        "l": 9,
                        "c": 10.5,
                        "v": 1234,
                    }
                ]
            }

    calls: list[tuple[str, dict]] = []

    def fake_get(url, params, timeout):
        calls.append((url, params))
        return FakeResponse()

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    monkeypatch.setattr(intraday_data.requests, "get", fake_get)

    provider = intraday_data.PolygonIntradayProvider()
    result = provider.fetch("SOXL", start="2026-06-04", end="2026-06-04", interval="1m")

    assert calls[0][0].endswith("/v2/aggs/ticker/SOXL/range/1/minute/2026-06-04/2026-06-04")
    assert calls[0][1]["adjusted"] == "true"
    assert result.iloc[0]["provider"] == "polygon"
    assert str(result.iloc[0]["timestamp"]) == "2026-06-04 09:30:00-04:00"


def test_build_canonical_intraday_panel_prefers_first_provider_and_flags_disagreement():
    from src.soxl_quant.app.intraday_data import build_canonical_intraday_panel, normalize_intraday_bars

    futu = normalize_intraday_bars(
        pd.DataFrame(
            [{"datetime": "2026-06-04 09:30:00", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100}]
        ),
        provider="futu",
        symbol="SOXL",
        interval="1m",
        timestamp_column="datetime",
    )
    polygon = normalize_intraday_bars(
        pd.DataFrame(
            [{"datetime": "2026-06-04 09:30:00", "open": 10, "high": 11, "low": 9, "close": 10.02, "volume": 100}]
        ),
        provider="polygon",
        symbol="SOXL",
        interval="1m",
        timestamp_column="datetime",
    )

    result = build_canonical_intraday_panel([futu, polygon], provider_priority=("futu", "polygon"))

    assert len(result) == 1
    assert result.iloc[0]["provider"] == "futu"
    assert result.iloc[0]["source_count"] == 2
    assert result.iloc[0]["close_disagreement_abs"] == pytest.approx(0.02)


def test_fetch_intraday_cli_dry_run_does_not_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import fetch_intraday_data
    from src.soxl_quant.app import intraday_data

    seen_kwargs = {}

    class FakeProvider:
        def __init__(self, *args, **kwargs):
            seen_kwargs.update(kwargs)

        def fetch(self, symbol: str, start: str, end: str, interval: str):
            return intraday_data.normalize_intraday_bars(
                pd.DataFrame(
                    [{"datetime": "2026-06-04 09:30:00", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]
                ),
                provider="futu",
                symbol=symbol,
                interval=interval,
                timestamp_column="datetime",
            )

    monkeypatch.setattr(fetch_intraday_data, "ROOT", tmp_path)
    monkeypatch.setattr(fetch_intraday_data, "FutuIntradayProvider", FakeProvider)

    result = fetch_intraday_data.main(
        [
            "--providers",
            "futu",
            "--symbols",
            "SOXL",
            "--start",
            "2026-06-04",
            "--end",
            "2026-06-04",
            "--dry-run",
            "--futu-timeout-sec",
            "7",
        ]
    )

    assert result["rows"] == 1
    assert seen_kwargs["connect_timeout_sec"] == 7
    assert not (tmp_path / "data").exists()


def test_prepare_futu_log_home_points_sdk_home_to_writable_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from src.soxl_quant.app.intraday_data import _prepare_futu_log_home

    monkeypatch.setenv("HOME", "/original/home")

    previous = _prepare_futu_log_home(tmp_path / "futu_home")

    assert previous == "/original/home"
    assert Path.home() == tmp_path / "futu_home"
    assert (tmp_path / "futu_home").exists()


def test_futu_import_permission_error_is_not_reported_as_missing_sdk(monkeypatch: pytest.MonkeyPatch):
    from src.soxl_quant.app.intraday_data import _import_futu_sdk

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "futu":
            raise PermissionError("cannot write futu log")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="failed to import futu-api.*PermissionError"):
        _import_futu_sdk()


def test_intraday_daily_feature_builder_lags_features_by_one_day(tmp_path: Path):
    from src.soxl_quant.app.intraday_data import build_intraday_daily_features

    bars = pd.DataFrame(
        [
            {
                "provider": "futu",
                "symbol": "SOXL",
                "interval": "1m",
                "timestamp": pd.Timestamp("2026-06-04 09:30", tz="America/New_York"),
                "date": "2026-06-04",
                "session": "regular",
                "open": 10.0,
                "high": 11.0,
                "low": 10.0,
                "close": 10.5,
                "volume": 100.0,
                "turnover": 1050.0,
                "source_tz": "America/New_York",
                "adjusted": True,
                "fetched_at": "2026-06-05T00:00:00Z",
            },
            {
                "provider": "futu",
                "symbol": "SOXL",
                "interval": "1m",
                "timestamp": pd.Timestamp("2026-06-04 15:59", tz="America/New_York"),
                "date": "2026-06-04",
                "session": "regular",
                "open": 10.5,
                "high": 12.0,
                "low": 10.3,
                "close": 11.5,
                "volume": 300.0,
                "turnover": 3450.0,
                "source_tz": "America/New_York",
                "adjusted": True,
                "fetched_at": "2026-06-05T00:00:00Z",
            },
        ]
    )

    out = build_intraday_daily_features(bars, output_path=tmp_path / "features.csv")

    assert out.iloc[0]["date"] == "2026-06-05"
    assert out.iloc[0]["INTRADAY_SOXL_BAR_COUNT_1M"] == 2
    assert out.iloc[0]["INTRADAY_SOXL_VOLUME_CONCENTRATION_CLOSE_30M"] == 0.75
    assert (tmp_path / "features.csv").exists()


def test_intraday_microstructure_feature_pack_reads_lagged_daily_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from src.soxl_quant import data as soxl_data

    cache = tmp_path / "intraday_features_soxl_1m.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-05",
                "INTRADAY_SOXL_BAR_COUNT_1M": 2,
                "INTRADAY_SOXL_OPEN_RANGE_RETURN_30M": 0.05,
                "INTRADAY_SOXL_CLOSE_LOCATION": 0.75,
                "INTRADAY_SOXL_REALIZED_VOL_1M": 0.01,
                "INTRADAY_SOXL_VOLUME_CONCENTRATION_CLOSE_30M": 0.75,
            }
        ]
    ).to_csv(cache, index=False)
    monkeypatch.setattr(soxl_data, "EXTERNAL_DATA_DIR", tmp_path)

    index = pd.to_datetime(["2026-06-04", "2026-06-05"])
    registry = soxl_data._feature_pack_registry()
    payload = registry["intraday_microstructure_v1"]["builder"]({"index": index})

    assert "INTRADAY_SOXL_CLOSE_LOCATION" in soxl_data.feature_columns_for_packs(["intraday_microstructure_v1"])
    assert payload["INTRADAY_SOXL_CLOSE_LOCATION"].loc[pd.Timestamp("2026-06-04")] == 0.0
    assert payload["INTRADAY_SOXL_CLOSE_LOCATION"].loc[pd.Timestamp("2026-06-05")] == 0.75

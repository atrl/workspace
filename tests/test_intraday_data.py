from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts import fetch_intraday_data
from src.soxl_quant.app.intraday_data import (
    FutuIntradayProvider,
    PolygonIntradayProvider,
    canonical_intraday_path,
    intraday_raw_path,
    normalize_futu_history_kline,
    normalize_polygon_aggregates,
)


def _ny_ms(value: str) -> int:
    return int(pd.Timestamp(value, tz="America/New_York").tz_convert("UTC").timestamp() * 1000)


def test_normalize_futu_history_kline_filters_regular_session_and_keeps_canonical_schema():
    raw = pd.DataFrame(
        {
            "code": ["US.SOXL", "US.SOXL", "US.SOXL"],
            "time_key": [
                "2024-01-02 09:29:00",
                "2024-01-02 09:30:00",
                "2024-01-02 16:00:00",
            ],
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.1, 11.1, 12.1],
            "volume": [100, 200, 300],
            "turnover": [1000.0, 2200.0, 3600.0],
        }
    )

    frame = normalize_futu_history_kline(
        raw,
        symbol="SOXL",
        interval="1m",
        session="regular",
        adjusted=True,
        fetched_at=pd.Timestamp("2026-06-05T00:00:00Z"),
    )

    assert frame["timestamp"].dt.tz.zone == "America/New_York"
    assert frame["timestamp"].dt.strftime("%H:%M:%S").tolist() == ["09:30:00"]
    assert frame.iloc[0].to_dict() == {
        "provider": "futu",
        "symbol": "SOXL",
        "interval": "1m",
        "timestamp": pd.Timestamp("2024-01-02 09:30:00", tz="America/New_York"),
        "date": "2024-01-02",
        "session": "regular",
        "open": 11.0,
        "high": 11.5,
        "low": 10.5,
        "close": 11.1,
        "volume": 200.0,
        "turnover": 2200.0,
        "source_tz": "America/New_York",
        "adjusted": True,
        "fetched_at": pd.Timestamp("2026-06-05T00:00:00Z"),
    }


def test_normalize_polygon_aggregates_uses_eastern_timestamps_and_adjustment_flag():
    payload = {
        "ticker": "SOXL",
        "adjusted": True,
        "results": [
            {"t": _ny_ms("2024-01-02 09:30:00"), "o": 20.0, "h": 20.5, "l": 19.5, "c": 20.2, "v": 500},
            {"t": _ny_ms("2024-01-02 16:01:00"), "o": 21.0, "h": 21.5, "l": 20.5, "c": 21.2, "v": 600},
        ],
    }

    frame = normalize_polygon_aggregates(
        payload,
        symbol="SOXL",
        interval="1m",
        session="regular",
        fetched_at=pd.Timestamp("2026-06-05T00:00:00Z"),
    )

    assert frame["timestamp"].dt.tz.zone == "America/New_York"
    assert frame["timestamp"].dt.strftime("%H:%M:%S").tolist() == ["09:30:00"]
    assert frame.iloc[0]["provider"] == "polygon"
    assert frame.iloc[0]["adjusted"] is True
    assert frame.iloc[0]["turnover"] == 0.0


def test_futu_provider_pages_mocked_history_kline_and_closes_context():
    class FakeContext:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            self.closed = False

        def request_history_kline(self, **kwargs):
            self.calls.append(kwargs)
            frame = pd.DataFrame(
                {
                    "time_key": ["2024-01-02 09:30:00"],
                    "open": [10.0],
                    "high": [10.5],
                    "low": [9.5],
                    "close": [10.1],
                    "volume": [100],
                    "turnover": [1000.0],
                }
            )
            if kwargs.get("page_req_key") is None:
                return 0, frame, b"next"
            return 0, frame.assign(time_key=["2024-01-02 09:31:00"]), None

        def close(self):
            self.closed = True

    fake_context = FakeContext()

    class FakeFutu:
        RET_OK = 0

        class KLType:
            K_1M = "K_1M"

        class AuType:
            QFQ = "qfq"

        class KL_FIELD:
            ALL = "ALL"

        class Session:
            RTH = "RTH"
            ALL = "ALL"

        @staticmethod
        def OpenQuoteContext(**kwargs):
            fake_context.kwargs = kwargs
            return fake_context

    provider = FutuIntradayProvider(sdk=FakeFutu)
    frame = provider.fetch("SOXL", interval="1m", start="2024-01-02", end="2024-01-02", session="regular")

    assert len(frame) == 2
    assert fake_context.closed is True
    assert fake_context.calls[0]["code"] == "US.SOXL"
    assert fake_context.calls[0]["ktype"] == "K_1M"
    assert fake_context.calls[0]["session"] == "RTH"
    assert fake_context.calls[1]["page_req_key"] == b"next"


def test_polygon_provider_pages_mocked_aggregate_responses():
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, params=None, timeout=None):
            self.calls.append((url, params, timeout))
            if len(self.calls) == 1:
                return FakeResponse(
                    {
                        "ticker": "SOXL",
                        "adjusted": True,
                        "results": [
                            {
                                "t": _ny_ms("2024-01-02 09:30:00"),
                                "o": 20.0,
                                "h": 20.5,
                                "l": 19.5,
                                "c": 20.1,
                                "v": 100,
                            }
                        ],
                        "next_url": "https://api.polygon.io/next-page",
                    }
                )
            return FakeResponse(
                {
                    "ticker": "SOXL",
                    "adjusted": True,
                    "results": [
                        {
                            "t": _ny_ms("2024-01-02 09:31:00"),
                            "o": 20.1,
                            "h": 20.6,
                            "l": 20.0,
                            "c": 20.4,
                            "v": 120,
                        }
                    ],
                }
            )

    session_obj = FakeSession()
    provider = PolygonIntradayProvider(api_key="secret", session_obj=session_obj)
    frame = provider.fetch("SOXL", interval="1m", start="2024-01-02", end="2024-01-02", session="regular")

    assert frame["timestamp"].dt.strftime("%H:%M:%S").tolist() == ["09:30:00", "09:31:00"]
    assert session_obj.calls[0][0].endswith("/v2/aggs/ticker/SOXL/range/1/minute/2024-01-02/2024-01-02")
    assert session_obj.calls[0][1]["adjusted"] == "true"
    assert session_obj.calls[0][1]["apiKey"] == "secret"
    assert session_obj.calls[1][0] == "https://api.polygon.io/next-page"


def test_fetch_intraday_data_dry_run_does_not_write_outputs():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        code = fetch_intraday_data.main(
            [
                "--providers",
                "futu,polygon",
                "--symbols",
                "SOXL,SOXX",
                "--interval",
                "1m",
                "--start",
                "2024-01-02",
                "--end",
                "2024-01-02",
                "--session",
                "regular",
                "--output-root",
                str(root),
                "--dry-run",
            ]
        )

        assert code == 0
        assert not (root / "data/raw/intraday").exists()
        assert not canonical_intraday_path(root=root).exists()
        assert intraday_raw_path("futu", "SOXL", "1m", "2024-01-02", "2024-01-02", "regular", root=root).suffix == ".parquet"


def test_fetch_intraday_data_rejects_unknown_provider():
    code = fetch_intraday_data.main(
        [
            "--providers",
            "futu,bad",
            "--symbols",
            "SOXL",
            "--interval",
            "1m",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-02",
            "--session",
            "regular",
            "--dry-run",
        ]
    )

    assert code == 2

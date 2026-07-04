from __future__ import annotations

import pandas as pd
import requests

from src.data.intraday import (
    fetch_polygon_aggregate_bars,
    normalize_polygon_aggregate_frame,
    polygon_intraday_to_qlib_frame,
)


def test_polygon_aggregates_normalize_to_ny_bar_end_canonical_schema():
    raw = pd.DataFrame(
        [
            {
                "t": 1767364200000,  # 2026-01-02 09:30:00 America/New_York
                "o": 100.0,
                "h": 102.0,
                "l": 99.0,
                "c": 101.0,
                "v": 12345,
                "vw": 100.5,
                "n": 77,
            },
            {
                "t": 1767365100000,  # 2026-01-02 09:45:00 America/New_York
                "o": 101.0,
                "h": 103.0,
                "l": 100.0,
                "c": 102.5,
                "v": 23456,
                "vw": 101.75,
                "n": 88,
            },
        ]
    )

    normalized = normalize_polygon_aggregate_frame(
        raw,
        symbol="qqq",
        multiplier=15,
        fetched_at="2026-07-04T00:00:00+00:00",
    )

    assert normalized["provider"].tolist() == ["polygon", "polygon"]
    assert normalized["symbol"].tolist() == ["QQQ", "QQQ"]
    assert normalized["interval"].tolist() == ["15m", "15m"]
    assert normalized["date"].tolist() == ["2026-01-02", "2026-01-02"]
    assert normalized["session"].tolist() == ["regular", "regular"]
    assert normalized["timestamp"].tolist() == [
        pd.Timestamp("2026-01-02 09:45:00", tz="America/New_York"),
        pd.Timestamp("2026-01-02 10:00:00", tz="America/New_York"),
    ]
    assert normalized["turnover"].tolist() == [12345 * 100.5, 23456 * 101.75]
    assert normalized["transactions"].tolist() == [77, 88]
    assert normalized["adjusted"].tolist() == [True, True]
    assert normalized["fetched_at"].tolist() == [
        "2026-07-04T00:00:00+00:00",
        "2026-07-04T00:00:00+00:00",
    ]


def test_polygon_intraday_frame_can_feed_qlib_15min_store_builder(tmp_path):
    normalized = normalize_polygon_aggregate_frame(
        pd.DataFrame(
            [
                {"t": 1767364200000, "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 1000, "vw": 100.1},
                {"t": 1767365100000, "o": 101.0, "h": 103.0, "l": 100.0, "c": 102.0, "v": 1200, "vw": 101.2},
            ]
        ),
        symbol="NVDA",
        multiplier=15,
    )

    qlib_frame = polygon_intraday_to_qlib_frame(normalized)

    assert qlib_frame[["instrument", "symbol", "interval"]].drop_duplicates().to_dict("records") == [
        {"instrument": "NVDA", "symbol": "NVDA", "interval": "15min"}
    ]
    assert qlib_frame["datetime"].tolist() == [
        pd.Timestamp("2026-01-02 09:45:00", tz="America/New_York"),
        pd.Timestamp("2026-01-02 10:00:00", tz="America/New_York"),
    ]
    assert qlib_frame["bar_index"].tolist() == [0, 1]
    assert qlib_frame["bar_count"].tolist() == [2, 2]
    assert {"open", "high", "low", "close", "volume", "turnover", "vwap", "transactions"}.issubset(
        qlib_frame.columns
    )


def test_polygon_15min_regular_session_excludes_bar_ending_at_0930():
    normalized = normalize_polygon_aggregate_frame(
        pd.DataFrame(
            [
                {"t": 1767363300000, "o": 99.0, "h": 100.0, "l": 98.5, "c": 99.5, "v": 900, "vw": 99.2},
                {"t": 1767364200000, "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 1000, "vw": 100.1},
            ]
        ),
        symbol="QQQ",
        multiplier=15,
    )

    assert normalized["timestamp"].tolist() == [
        pd.Timestamp("2026-01-02 09:30:00", tz="America/New_York"),
        pd.Timestamp("2026-01-02 09:45:00", tz="America/New_York"),
    ]
    assert normalized["session"].tolist() == ["premarket", "regular"]

    qlib_frame = polygon_intraday_to_qlib_frame(normalized)

    assert qlib_frame["datetime"].tolist() == [pd.Timestamp("2026-01-02 09:45:00", tz="America/New_York")]
    assert qlib_frame["bar_index"].tolist() == [0]


def test_polygon_fetch_timeout_returns_fail_closed_audit(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(requests, "get", raise_timeout)

    frame, audit = fetch_polygon_aggregate_bars(
        "SMH",
        start="2025-01-02",
        end="2026-07-02",
        api_key="test-key",
        max_retries=1,
    )

    assert frame.empty
    assert audit == {
        "symbol": "SMH",
        "status": "blocked_provider_timeout",
        "http_status": 0,
        "row_count": 0,
        "page_count": 0,
        "message": "read timed out",
    }

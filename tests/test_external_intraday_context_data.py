from __future__ import annotations

import pandas as pd
import requests

from src.data.intraday import (
    fetch_polygon_aggregate_bars,
    normalize_polygon_aggregate_frame,
    polygon_intraday_to_qlib_frame,
)
from src.workflow import intraday_external_context_data as external_context_workflow


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
    assert audit["symbol"] == "SMH"
    assert audit["status"] == "blocked_provider_timeout"
    assert audit["http_status"] == 0
    assert audit["row_count"] == 0
    assert audit["page_count"] == 0
    assert audit["message"] == "read timed out"
    assert audit["requested_start"] == "2025-01-02"
    assert audit["requested_end"] == "2026-07-02"
    assert audit["coverage_start"] == ""
    assert audit["coverage_end"] == ""
    assert audit["provider_history_truncated"] is True


def test_polygon_fetch_marks_provider_history_truncation(monkeypatch):
    class Response:
        status_code = 200
        headers = {}

        @staticmethod
        def json():
            return {
                "status": "OK",
                "results": [
                    {
                        "t": int(pd.Timestamp("2024-07-05 13:30:00", tz="UTC").timestamp() * 1000),
                        "o": 490.0,
                        "h": 491.0,
                        "l": 489.0,
                        "c": 490.5,
                        "v": 1000,
                        "vw": 490.25,
                        "n": 10,
                    }
                ],
            }

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())

    frame, audit = fetch_polygon_aggregate_bars(
        "QQQ",
        start="2019-01-02",
        end="2026-07-02",
        api_key="test-key",
    )

    assert not frame.empty
    assert audit["status"] == "ok"
    assert audit["requested_start"] == "2019-01-02"
    assert audit["requested_end"] == "2026-07-02"
    assert audit["coverage_start"] == "2024-07-05 09:45:00"
    assert audit["coverage_end"] == "2024-07-05 09:45:00"
    assert audit["provider_history_truncated"] is True


def test_polygon_fetch_preserves_partial_rows_when_later_page_hits_rate_limit(monkeypatch):
    class FirstResponse:
        status_code = 200
        headers = {}

        @staticmethod
        def json():
            return {
                "status": "OK",
                "results": [
                    {
                        "t": int(pd.Timestamp("2024-07-05 13:30:00", tz="UTC").timestamp() * 1000),
                        "o": 490.0,
                        "h": 491.0,
                        "l": 489.0,
                        "c": 490.5,
                        "v": 1000,
                        "vw": 490.25,
                        "n": 10,
                    }
                ],
                "next_url": "https://api.polygon.io/next",
            }

    class RateLimitResponse:
        status_code = 429
        headers = {}

        @staticmethod
        def json():
            return {"status": "ERROR", "message": "rate limit"}

    responses = iter([FirstResponse(), RateLimitResponse()])
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: next(responses))

    frame, audit = fetch_polygon_aggregate_bars(
        "QQQ",
        start="2019-01-02",
        end="2026-07-02",
        api_key="test-key",
        max_retries=0,
    )

    assert not frame.empty
    assert audit["status"] == "ok_partial_provider_rate_limit"
    assert audit["row_count"] == 1
    assert audit["http_status"] == 429
    assert audit["page_count"] == 1
    assert audit["coverage_start"] == "2024-07-05 09:45:00"
    assert audit["provider_history_truncated"] is True


def test_external_context_symbol_fetch_uses_fast_fail_rate_limit_policy(monkeypatch, tmp_path):
    calls = []

    def fake_fetch(symbol, **kwargs):
        calls.append({"symbol": symbol, **kwargs})
        return pd.DataFrame(), {
            "symbol": symbol,
            "status": "blocked_provider_rate_limit",
            "http_status": 429,
            "row_count": 0,
            "page_count": 0,
            "message": "rate limit",
            "requested_start": kwargs["start"],
            "requested_end": kwargs["end"],
            "coverage_start": "",
            "coverage_end": "",
            "provider_history_truncated": True,
        }

    monkeypatch.setattr(external_context_workflow, "fetch_polygon_aggregate_bars", fake_fetch)
    spec = external_context_workflow.IntradayExternalContextDataSpec(
        root=tmp_path,
        requested_symbols=("QQQ",),
    )

    source_audit, normalized = external_context_workflow._fetch_requested_symbols(spec, api_key="test-key")

    assert normalized.empty
    assert source_audit["status"].tolist() == ["blocked_provider_rate_limit"]
    assert calls[0]["start"] == "2019-01-02"
    assert calls[0]["max_retries"] == 0
    assert calls[0]["retry_sleep_seconds"] == 0.0


def test_external_context_symbol_fetch_persists_and_reuses_raw_cache(monkeypatch, tmp_path):
    normalized = normalize_polygon_aggregate_frame(
        pd.DataFrame(
            [
                {
                    "t": int(pd.Timestamp("2024-07-05 13:30:00", tz="UTC").timestamp() * 1000),
                    "o": 490.0,
                    "h": 491.0,
                    "l": 489.0,
                    "c": 490.5,
                    "v": 1000,
                    "vw": 490.25,
                    "n": 10,
                }
            ]
        ),
        symbol="QQQ",
        multiplier=15,
        fetched_at="2026-07-04T00:00:00+00:00",
    )
    calls = []

    def fake_fetch(symbol, **kwargs):
        calls.append(symbol)
        return normalized, {
            "symbol": symbol,
            "status": "ok_partial_provider_rate_limit",
            "http_status": 429,
            "row_count": len(normalized),
            "page_count": 1,
            "message": "rate limit",
            "requested_start": kwargs["start"],
            "requested_end": kwargs["end"],
            "coverage_start": "2024-07-05 09:45:00",
            "coverage_end": "2024-07-05 09:45:00",
            "provider_history_truncated": True,
        }

    monkeypatch.setattr(external_context_workflow, "fetch_polygon_aggregate_bars", fake_fetch)
    spec = external_context_workflow.IntradayExternalContextDataSpec(
        root=tmp_path,
        requested_symbols=("QQQ",),
    )

    source_audit, cached_frame = external_context_workflow._fetch_requested_symbols(spec, api_key="test-key")

    assert calls == ["QQQ"]
    assert not cached_frame.empty
    assert source_audit["status"].tolist() == ["ok_partial_provider_rate_limit"]
    assert external_context_workflow._raw_cache_path(spec, "QQQ").exists()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("cache should avoid a second Polygon call")

    monkeypatch.setattr(external_context_workflow, "fetch_polygon_aggregate_bars", fail_if_called)

    cached_audit, cached_again = external_context_workflow._fetch_requested_symbols(spec, api_key="test-key")

    assert not cached_again.empty
    assert cached_audit["status"].tolist() == ["ok_cached"]
    assert cached_audit["row_count"].tolist() == [1]
    assert cached_audit["provider_history_truncated"].tolist() == [True]

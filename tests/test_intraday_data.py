from __future__ import annotations

import pandas as pd
import pytest


def _minute_rows(date: str, *, symbol: str = "SOXL", base: float = 100.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    times = ["09:31", "09:32", "10:00", "15:31", "16:00"]
    closes = [base + 1.0, base + 2.0, base + 3.0, base - 2.0, base + 4.0]
    for idx, (hhmm, close) in enumerate(zip(times, closes, strict=True), start=1):
        rows.append(
            {
                "provider": "futu",
                "symbol": symbol,
                "interval": "1m",
                "timestamp": pd.Timestamp(f"{date} {hhmm}", tz="America/New_York"),
                "date": date,
                "session": "regular",
                "open": base if idx == 1 else close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100 * idx,
                "turnover": close * 100 * idx,
                "source_tz": "America/New_York",
                "adjusted": True,
                "fetched_at": "2026-07-03T00:00:00+00:00",
            }
        )
    return rows


def test_intraday_merge_replaces_overlap_by_symbol_timestamp():
    from src.data.intraday import merge_intraday_bars

    existing = pd.DataFrame(_minute_rows("2026-07-01", base=100.0))
    fetched = pd.DataFrame(
        [
            {
                **_minute_rows("2026-07-01", base=100.0)[0],
                "close": 111.0,
                "volume": 999,
            },
            *_minute_rows("2026-07-02", base=120.0),
        ]
    )

    merged = merge_intraday_bars(existing, fetched)

    overlap = merged.loc[merged["timestamp"] == pd.Timestamp("2026-07-01 09:31", tz="America/New_York")]
    assert len(overlap) == 1
    assert overlap["close"].item() == 111.0
    assert overlap["volume"].item() == 999
    assert len(merged) == len(existing) + len(_minute_rows("2026-07-02", base=120.0))


def test_resample_intraday_bars_returns_qlib_ready_frequency_rows():
    from src.data.intraday import resample_intraday_bars

    minute_frame = pd.DataFrame(_minute_rows("2026-07-02", base=100.0))

    bars = resample_intraday_bars(minute_frame, freq="5min", symbols=["SOXL"])

    first = bars.iloc[0]
    assert first["instrument"] == "SOXL"
    assert first["interval"] == "5min"
    assert first["datetime"] == pd.Timestamp("2026-07-02 09:35", tz="America/New_York")
    assert first["open"] == pytest.approx(100.0)
    assert first["close"] == pytest.approx(102.0)
    assert first["bar_index"] == 0
    assert first["bar_count"] == len(bars)
    assert {"open", "high", "low", "close", "volume", "turnover"}.issubset(bars.columns)


def test_build_t1_intraday_features_aligns_to_next_qlib_calendar_day():
    from src.data.intraday import build_t1_intraday_feature_frame

    minute_frame = pd.DataFrame(
        [
            *_minute_rows("2026-07-02", base=100.0),
            *_minute_rows("2026-07-06", base=120.0),
        ]
    )
    calendar = ["2026-07-02", "2026-07-06", "2026-07-07"]

    features = build_t1_intraday_feature_frame(minute_frame, calendar=calendar, symbol="SOXL")

    assert features["date"].tolist() == ["2026-07-06", "2026-07-07"]
    first = features.iloc[0]
    assert first["INTRADAY_SOXL_BAR_COUNT_1M"] == 5
    assert first["INTRADAY_SOXL_OPEN_RANGE_RETURN_30M"] == pytest.approx(0.03)
    assert first["INTRADAY_SOXL_AMPLITUDE"] == pytest.approx(105.0 / 97.0 - 1.0)
    assert first["INTRADAY_SOXL_DRAWDOWN_FROM_OPEN"] == pytest.approx(-0.03)
    assert "INTRADAY_SOXL_LATE_RETURN_30M" in features.columns
    assert "INTRADAY_SOXL_CLOSE_VWAP_DEVIATION" in features.columns

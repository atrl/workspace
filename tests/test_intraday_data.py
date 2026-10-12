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


def test_minute_audit_detects_old_holes_without_rejecting_early_close_or_live_session():
    from src.data.intraday_quality import audit_minute_coverage, plan_minute_repairs

    schedule = pd.DataFrame({
        "open": pd.to_datetime(["2026-07-24 13:30Z", "2026-07-27 13:30Z", "2026-07-28 13:30Z"]),
        "close": pd.to_datetime(["2026-07-24 17:00Z", "2026-07-27 20:00Z", "2026-07-28 20:00Z"]),
    }, index=pd.to_datetime(["2026-07-24", "2026-07-27", "2026-07-28"]))
    rows = []
    for day, count in [("2026-07-24", 210), ("2026-07-27", 61), ("2026-07-28", 61)]:
        for stamp in pd.date_range(f"{day} 09:31", periods=count, freq="min", tz="America/New_York"):
            rows.append(dict(symbol="SOXL", timestamp=stamp, open=100, high=101,
                             low=99, close=100, volume=0, turnover=0))
    frame = pd.DataFrame(rows)
    audit = audit_minute_coverage(frame, schedule, symbols=["SOXL", "SOXX"],
        timestamp_convention="right_edge", observed_at=pd.Timestamp("2026-07-28 10:31", tz="America/New_York"))
    indexed = audit.set_index(["symbol", "date"])
    assert indexed.loc[("SOXL", "2026-07-24"), "training_eligible"]
    assert indexed.loc[("SOXL", "2026-07-27"), "missing_minutes"] == 329
    assert indexed.loc[("SOXX", "2026-07-27"), "missing_minutes"] == 390
    assert indexed.loc[("SOXL", "2026-07-28"), "status"] == "in_progress"
    repairs = plan_minute_repairs(audit)
    assert set(repairs.start) == {"2026-07-24", "2026-07-27"}
    with pytest.raises(ValueError, match="right_edge"):
        audit_minute_coverage(frame, schedule, symbols=["SOXL"],
            timestamp_convention="unknown", observed_at=pd.Timestamp("2026-07-28 21:00Z"))


def test_native_minute_variance_uses_session_open_and_matched_five_minute_path(tmp_path):
    import numpy as np
    from src.data import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame
    from src.data.intraday_quality import audit_minute_coverage
    from src.analysis.intraday_volatility import minute_volatility_reports

    stamps = pd.date_range("2026-07-27 09:31", periods=390, freq="min", tz="America/New_York")
    rows = []
    for symbol, step in [("SOXX", 0.001), ("SOXL", 0.003)]:
        prices = 100 * np.exp(step * np.arange(391))
        for i, stamp in enumerate(stamps):
            rows.append(dict(symbol=symbol, timestamp=stamp, open=prices[i], close=prices[i + 1],
                             high=prices[i + 1], low=prices[i], volume=100))
    raw = pd.DataFrame(rows)
    schedule = pd.DataFrame({"open": pd.to_datetime(["2026-07-27 13:30Z"]),
        "close": pd.to_datetime(["2026-07-27 20:00Z"])}, index=pd.to_datetime(["2026-07-27"]))
    coverage = audit_minute_coverage(raw, schedule, symbols=["SOXL", "SOXX"],
        timestamp_convention="right_edge", observed_at=pd.Timestamp("2026-07-27 21:00Z"))
    provider = tmp_path / "minute_provider"
    fields = ["open", "high", "low", "close", "volume"]
    build_qlib_store_from_frame(raw.rename(columns={"symbol": "instrument", "timestamp": "datetime"}),
        provider_uri=provider, freq="1min", field_map={"$" + name: name for name in fields},
        as_of="2026-07-27", source="analytical constant-log-return fixture")
    features = load_qlib_feature_frame(QlibProviderConfig(provider, freq="1min"),
        instruments=["SOXL", "SOXX"], fields=["$" + name for name in fields],
        start="2026-07-27 09:31", end="2026-07-27 16:00")
    reports = minute_volatility_reports(features, coverage, segments=[("opening", 570, 600), ("rest", 600, 960)])
    soxx = reports["daily"].set_index("instrument").loc["SOXX"]
    assert soxx.realized_vol_pct == pytest.approx(100 * np.sqrt(390 * 0.001 ** 2), rel=1e-4)
    assert soxx.rv1_to_rv5_ratio == pytest.approx(0.2, rel=1e-4)
    opening = reports["profile"].set_index(["instrument", "segment"]).loc[("SOXX", "opening")]
    assert opening.mean_variance_share == pytest.approx(30 / 390, rel=1e-4)
    features.attrs.clear()
    with pytest.raises(ValueError, match="native Qlib"):
        minute_volatility_reports(features, coverage, segments=[("full", 570, 960)])

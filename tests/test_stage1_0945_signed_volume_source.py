import pandas as pd

from src.data.stage1_0945_signed_volume_source import build_daily_prefix


def _prefix() -> pd.DataFrame:
    clock = pd.date_range("2025-01-02 09:31", periods=15, freq="min")
    close = [100.0, 101.0, 100.0] + [100.0] * 12
    volume = [9.0, 1.0, 3.0] + [0.0] * 12
    return pd.DataFrame({"datetime": clock, "instrument": "SOXX",
                         "$close": close, "$volume": volume, "$price_valid": 1.0})


def test_signed_volume_uses_only_completed_signed_bars():
    frame = _prefix()
    row = build_daily_prefix(frame, pd.DatetimeIndex(["2025-01-02"])).iloc[0]
    assert row.reason == "ok"
    assert row.signed_volume_pressure == -0.5
    assert row.opening_return_0931_0945 == 0.0
    assert row.unsigned_volume_14bars == 4.0
    assert row.source_last_bar == pd.Timestamp("2025-01-02 09:45")
    assert row.decision_at == pd.Timestamp("2025-01-02 09:46")


def test_missing_or_invalid_prefix_stays_unknown():
    full = _prefix()
    missing = full.loc[full.datetime.ne(pd.Timestamp("2025-01-02 09:40"))]
    flagged = full.copy()
    flagged.loc[flagged.datetime.eq(pd.Timestamp("2025-01-02 09:40")), "$price_valid"] = 0.0
    for frame, reason in ((missing, "missing_or_duplicate_prefix_bar"),
                          (flagged, "invalid_close_volume_or_price_flag")):
        row = build_daily_prefix(frame, pd.DatetimeIndex(["2025-01-02"])).iloc[0]
        assert row.reason == reason
        assert pd.isna(row.signed_volume_pressure)

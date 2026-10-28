import pandas as pd
import pytest

from src.backtest.event_guard_clock import guard_slot


def test_event_clock_uses_completed_bar_then_two_minute_fill():
    day = pd.Timestamp("2024-07-01")
    slot = guard_slot(day + pd.Timedelta(hours=9, minutes=48),
        session_open=day + pd.Timedelta(hours=9, minutes=30),
        session_close=day + pd.Timedelta(hours=16),
        original_open_fill=day + pd.Timedelta(hours=9, minutes=47))
    assert slot.decision_at == day + pd.Timedelta(hours=9, minutes=49)
    assert slot.fill_at == day + pd.Timedelta(hours=9, minutes=50)
    assert guard_slot(day + pd.Timedelta(hours=9, minutes=47),
        session_open=day + pd.Timedelta(hours=9, minutes=30),
        session_close=day + pd.Timedelta(hours=16),
        original_open_fill=day + pd.Timedelta(hours=9, minutes=47)) is None


def test_half_day_guard_requires_a_fill_before_actual_close():
    day = pd.Timestamp("2024-11-29")
    common = dict(session_open=day + pd.Timedelta(hours=9, minutes=30),
                  session_close=day + pd.Timedelta(hours=13),
                  original_open_fill=day + pd.Timedelta(hours=9, minutes=47))
    assert guard_slot(day + pd.Timedelta(hours=12, minutes=57), **common).fill_at == day + pd.Timedelta(hours=12, minutes=59)
    assert guard_slot(day + pd.Timedelta(hours=12, minutes=58), **common) is None


def test_guard_rejects_cross_session_or_timezone_ambiguity():
    day = pd.Timestamp("2024-11-29")
    with pytest.raises(ValueError, match="calendar differ"):
        guard_slot(day + pd.Timedelta(hours=10),
            session_open=day + pd.Timedelta(days=1, hours=9, minutes=30),
            session_close=day + pd.Timedelta(days=1, hours=13))

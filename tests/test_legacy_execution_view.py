"""Native last-observed interval and immutable execution-calendar boundary."""
from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.data.legacy_execution_view import CANONICAL_PROVIDER, EXECUTION_PROVIDER, load_legacy_execution_view


def test_execution_boundary_adds_no_observation_and_rejects_tampering(tmp_path):
    from qlib.data import D
    from qlib.backtest.utils import TradeCalendarManager

    root = Path(__file__).resolve().parents[1]
    for relative in (CANONICAL_PROVIDER, EXECUTION_PROVIDER):
        shutil.copytree(root / relative, tmp_path / relative)
    provider, audit = load_legacy_execution_view(tmp_path)
    assert provider == tmp_path / EXECUTION_PROVIDER
    assert audit["observed_calendar_rows"] == 4162
    assert audit["future_calendar_rows"] == 4163
    assert audit["native_rows"] == 8324
    observed = pd.DatetimeIndex(D.calendar(freq="day", future=False))
    future = pd.DatetimeIndex(D.calendar(freq="day", future=True))
    assert future[:-1].equals(observed)
    assert future[-1] == pd.Timestamp("2026-09-28")
    prices = D.features(["SOXL", "SOXX"], ["$close"], start_time="2026-09-25", end_time="2026-09-28", freq="day")
    assert len(prices) == 2
    assert prices.index.get_level_values("datetime").max() == pd.Timestamp("2026-09-25")
    calendar = TradeCalendarManager(freq="day", start_time="2026-09-25", end_time="2026-09-25")
    assert calendar.get_step_time() == (pd.Timestamp("2026-09-25"), pd.Timestamp("2026-09-27 23:59:59"))
    boundary_file = provider / "calendars/day_future.txt"
    boundary_file.write_text(boundary_file.read_text().replace("2026-09-28", "2026-09-29"))
    with pytest.raises(ValueError, match="boundary calendar changed"):
        load_legacy_execution_view(tmp_path)

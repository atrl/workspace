from __future__ import annotations

import pandas as pd
import pytest

from src.data.macro_state import build_macro_state_daily_frame, fetch_fred_macro_state_series


def test_macro_state_daily_frame_applies_trading_day_availability_lags():
    calendar = pd.date_range("2026-01-05", periods=6, freq="B")
    rows = []
    for field, value in [("dgs2", 4.0), ("dgs10", 4.5), ("dfii10", 1.9), ("t5y5y_inflation_proxy", 2.3)]:
        rows.append({"datetime": "2026-01-05", "field": field, "value": value})
    rows.append({"datetime": "2026-01-05", "field": "dtwexbgs", "value": 120.0})

    frame = build_macro_state_daily_frame(pd.DataFrame(rows), calendar=calendar)

    monday = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-01-05"))].iloc[0]
    tuesday = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-01-06"))].iloc[0]
    wednesday = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-01-07"))].iloc[0]
    thursday = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-01-08"))].iloc[0]
    assert pd.isna(monday["dgs10"])
    assert tuesday["dgs10"] == 4.5
    assert tuesday["dgs10_source_observed"] == 1.0
    assert tuesday["dgs10_publication_lag_days"] == 1.0
    assert pd.isna(tuesday["dtwexbgs"])
    assert wednesday["dtwexbgs"] == 120.0
    assert wednesday["dtwexbgs_publication_lag_days"] == 2.0
    assert thursday["dtwexbgs_is_carried"] == 1.0
    assert thursday["dtwexbgs_staleness_days"] == 1.0


def test_fetch_fred_macro_state_series_fails_closed_if_required_series_missing():
    class FakeResponse:
        def __init__(self, series_id: str):
            self.text = f"observation_date,{series_id}\n2026-01-05,4.0\n"

        def raise_for_status(self) -> None:
            return None

    class FakeSession:
        def get(self, url, *, params, timeout, headers):
            if params["id"] == "DFII10":
                raise RuntimeError("source unavailable")
            return FakeResponse(params["id"])

    with pytest.raises(RuntimeError, match="required FRED macro-state series unavailable"):
        fetch_fred_macro_state_series(start="2026-01-01", end="2026-01-31", session=FakeSession())

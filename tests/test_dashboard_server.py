from __future__ import annotations

import pandas as pd

from scripts.dashboard_server import _build_daily_rows, _response_window_from_rows


def test_build_daily_rows_include_market_ohlc_for_rerun_kline():
    idx = pd.to_datetime(["2026-06-05", "2026-06-08"])
    account = pd.Series([1_000_000.0, 1_010_000.0], index=idx)
    price = pd.Series([182.54, 211.44], index=idx)
    market = {
        "2026-06-05": {
            "open": 180.0,
            "high": 184.0,
            "low": 178.0,
            "close": 182.54,
            "volume": 123,
            "ma180": 90.0,
        },
        "2026-06-08": {
            "open": 190.0,
            "high": 212.0,
            "low": 188.0,
            "close": 211.44,
            "volume": 456,
            "ma180": 91.0,
        },
    }

    rows = _build_daily_rows(
        account=account,
        price=price,
        position_history=[0.0, 0.5],
        trade_signals=[(idx[1], 211.44, 0.5, "buy")],
        market_lookup=market,
    )

    assert rows[0]["open"] == 180.0
    assert rows[0]["high"] == 184.0
    assert rows[0]["low"] == 178.0
    assert rows[0]["close"] == 182.54
    assert rows[0]["volume"] == 123
    assert rows[1]["action"] == "BUY"
    assert rows[1]["position"] == 0.5


def test_response_window_from_rows_uses_actual_row_dates():
    rows = [
        {"datetime": "2026-06-05"},
        {"datetime": "2026-06-10"},
    ]

    window = _response_window_from_rows(rows, "2026-06-05", "2026-06-11")

    assert window == {"start": "2026-06-05", "end": "2026-06-10", "rows": 2}

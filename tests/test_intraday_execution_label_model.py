from __future__ import annotations

import pandas as pd
import pytest

from src.workflow.intraday_execution_label_model import build_execution_label_frame


ADVERSE_EVENT_RISK_COL = "adverse_event_risk_target"
CRASH_EVENT_TARGET_COL = "crash_event_target"
OPEN_BUY_DUMP_TARGET_COL = "open_buy_dump_target"


def test_build_execution_label_frame_uses_next_bar_entry_and_triple_barrier():
    raw = pd.DataFrame(
        [
            {
                "instrument": "SOXL",
                "datetime": "2026-01-05 09:45:00",
                "$open": 100.0,
                "$high": 101.0,
                "$low": 99.5,
                "$close": 100.5,
                "$volume": 1000.0,
                "$turnover": 100000.0,
                "$bar_index": 0.0,
                "$bar_count": 5.0,
            },
            {
                "instrument": "SOXL",
                "datetime": "2026-01-05 10:00:00",
                "$open": 101.0,
                "$high": 101.5,
                "$low": 100.0,
                "$close": 101.2,
                "$volume": 1200.0,
                "$turnover": 121200.0,
                "$bar_index": 1.0,
                "$bar_count": 5.0,
            },
            {
                "instrument": "SOXL",
                "datetime": "2026-01-05 10:15:00",
                "$open": 102.0,
                "$high": 106.0,
                "$low": 101.5,
                "$close": 105.0,
                "$volume": 3000.0,
                "$turnover": 315000.0,
                "$bar_index": 2.0,
                "$bar_count": 5.0,
            },
            {
                "instrument": "SOXL",
                "datetime": "2026-01-05 10:30:00",
                "$open": 105.0,
                "$high": 105.2,
                "$low": 104.0,
                "$close": 104.5,
                "$volume": 1500.0,
                "$turnover": 156750.0,
                "$bar_index": 3.0,
                "$bar_count": 5.0,
            },
            {
                "instrument": "SOXL",
                "datetime": "2026-01-05 10:45:00",
                "$open": 104.5,
                "$high": 105.0,
                "$low": 103.0,
                "$close": 103.5,
                "$volume": 1300.0,
                "$turnover": 134550.0,
                "$bar_index": 4.0,
                "$bar_count": 5.0,
            },
            {
                "instrument": "SOXX",
                "datetime": "2026-01-05 09:45:00",
                "$open": 50.0,
                "$high": 50.3,
                "$low": 49.9,
                "$close": 50.1,
                "$volume": 500.0,
                "$turnover": 25050.0,
                "$bar_index": 0.0,
                "$bar_count": 5.0,
            },
            {
                "instrument": "SOXX",
                "datetime": "2026-01-05 10:00:00",
                "$open": 50.2,
                "$high": 50.4,
                "$low": 50.0,
                "$close": 50.3,
                "$volume": 600.0,
                "$turnover": 30180.0,
                "$bar_index": 1.0,
                "$bar_count": 5.0,
            },
            {
                "instrument": "SOXX",
                "datetime": "2026-01-05 10:15:00",
                "$open": 50.4,
                "$high": 51.0,
                "$low": 50.2,
                "$close": 50.9,
                "$volume": 800.0,
                "$turnover": 40720.0,
                "$bar_index": 2.0,
                "$bar_count": 5.0,
            },
            {
                "instrument": "SOXX",
                "datetime": "2026-01-05 10:30:00",
                "$open": 50.9,
                "$high": 51.0,
                "$low": 50.7,
                "$close": 50.8,
                "$volume": 700.0,
                "$turnover": 35560.0,
                "$bar_index": 3.0,
                "$bar_count": 5.0,
            },
            {
                "instrument": "SOXX",
                "datetime": "2026-01-05 10:45:00",
                "$open": 50.8,
                "$high": 50.9,
                "$low": 50.4,
                "$close": 50.5,
                "$volume": 650.0,
                "$turnover": 32825.0,
                "$bar_index": 4.0,
                "$bar_count": 5.0,
            },
        ]
    ).set_index(["instrument", "datetime"])

    frame, feature_cols, label_audit = build_execution_label_frame(
        raw,
        profit_take=0.025,
        stop_loss=0.015,
        horizon_bars=3,
        cost_bps=0.0,
    )

    assert "entry_day_worst_return" in frame.columns
    assert OPEN_BUY_DUMP_TARGET_COL in frame.columns
    assert CRASH_EVENT_TARGET_COL in frame.columns
    assert ADVERSE_EVENT_RISK_COL in frame.columns
    first = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-01-05 09:45:00"))].iloc[0]
    assert first["entry_datetime"] == pd.Timestamp("2026-01-05 10:00:00")
    assert first["triple_barrier_outcome"] == "profit"
    assert first["target_positive"] == 1
    assert first["target_return"] == (105.0 / 101.0 - 1.0)
    assert first["execution_improvement_return"] == (105.0 / 101.0 - 1.0) - (105.0 / 100.0 - 1.0)
    assert "soxl_open_gap_from_prior_close" in feature_cols
    assert "soxl_realized_vol_expansion" in feature_cols
    assert "soxl_volume_price_absorption" in feature_cols
    assert label_audit["label_policy"] == "triple_barrier_1h_2h_execution_improvement"


def test_build_execution_label_frame_records_entry_day_adverse_event_targets():
    rows = []
    for instrument, base in [("SOXL", 100.0), ("SOXX", 50.0)]:
        for index, (time, open_, high, low, close) in enumerate(
            [
                ("2026-01-05 09:45:00", base, base + 1.0, base - 0.5, base + 0.5),
                ("2026-01-05 10:00:00", base + 1.0, base + 1.5, base, base + 1.2),
                ("2026-01-05 10:15:00", base + 1.2, base + 1.3, base * 0.965, base * 0.97),
                ("2026-01-05 10:30:00", base * 0.97, base * 0.975, base * 0.91, base * 0.92),
                ("2026-01-05 10:45:00", base * 0.92, base * 0.93, base * 0.90, base * 0.91),
            ]
        ):
            rows.append(
                {
                    "instrument": instrument,
                    "datetime": time,
                    "$open": open_,
                    "$high": high,
                    "$low": low,
                    "$close": close,
                    "$volume": 1000.0 + index,
                    "$turnover": 100000.0,
                    "$bar_index": float(index),
                    "$bar_count": 5.0,
                }
            )
    raw = pd.DataFrame(rows).set_index(["instrument", "datetime"])

    frame, _, label_audit = build_execution_label_frame(
        raw,
        profit_take=0.20,
        stop_loss=0.20,
        horizon_bars=4,
        cost_bps=0.0,
    )

    first = frame.loc[frame["datetime"].eq(pd.Timestamp("2026-01-05 09:45:00"))].iloc[0]
    assert first["entry_datetime"] == pd.Timestamp("2026-01-05 10:00:00")
    assert first["entry_day_worst_return"] == pytest.approx(90.0 / 101.0 - 1.0)
    assert first[OPEN_BUY_DUMP_TARGET_COL] == 1
    assert first[CRASH_EVENT_TARGET_COL] == 1
    assert first[ADVERSE_EVENT_RISK_COL] == 1
    assert label_audit["adverse_event_policy"]["entry_timing"] == "next_15m_open"
    assert label_audit["adverse_event_policy"]["open_buy_dump_threshold"] == -0.03
    assert label_audit["adverse_event_policy"]["uses_same_bar_future"] is False

"""Causal clock guard for the research-only event replay entrypoint."""

import pandas as pd
import pytest
import json
from pathlib import Path

from src.backtest.stage1_event_episode import replay_event_full_path


def _tariff():
    root = Path(__file__).resolve().parents[1]
    source = root / "research/ibkr_tiered_minute_diagnosis_20261013/fee_schedule.json"
    return json.loads(source.read_text())["tariff"]


def test_event_replay_rejects_unavailable_minute_signal_before_account_work():
    day = pd.Timestamp("2022-06-01")
    event = {
        "trade_date": day,
        "bar_end_at": day + pd.Timedelta(hours=10),
        "decision_at": day + pd.Timedelta(hours=10, minutes=2),
        "fill_at": day + pd.Timedelta(hours=10, minutes=3),
        "desired_weights": [0.1, 0.5, 0.4],
        "adjusted_fill_prices": [10.0, 100.0],
        "raw_fill_prices": [10.0, 100.0],
    }
    with pytest.raises(ValueError, match="event clock differs"):
        replay_event_full_path(pd.DataFrame({"trade_date": [day]}),
            models=None, plan={"initial_cash": 100000, "tariff": {}},
            slippage_bps=0, event=event)


def test_single_funded_event_changes_only_the_event_branch():
    day = pd.Timestamp("2022-06-01")
    frame = pd.DataFrame({
        "signal_date": [day - pd.Timedelta(days=1)],
        "trade_date": [day], "exit_date": [day + pd.Timedelta(days=1)],
        "known_soxl": [10.], "known_soxx": [100.],
        "soxl_open": [10.], "soxx_open": [100.],
        "raw_soxl_open": [10.], "raw_soxx_open": [100.],
        "soxl_recovery": [10.], "soxx_recovery": [100.],
        "raw_soxl_recovery": [10.], "raw_soxx_recovery": [100.],
        "close_soxl": [10.], "close_soxx": [100.],
        "soxl_exit_open": [10.], "soxx_exit_open": [100.],
    })

    class Hold:
        def target(self, row, q, cash, known, peak):
            return None

    options = {"models": [Hold()], "plan": {"initial_cash": 100000, "tariff": _tariff()},
        "slippage_bps": 0}
    baseline = replay_event_full_path(frame, **options)
    event = {"trade_date": day,
        "bar_end_at": day + pd.Timedelta(hours=9, minutes=58),
        "decision_at": day + pd.Timedelta(hours=9, minutes=59),
        "fill_at": day + pd.Timedelta(hours=10),
        "desired_weights": [0., .1, .9],
        "adjusted_fill_prices": [10., 100.],
        "raw_fill_prices": [10., 100.]}
    candidate = replay_event_full_path(frame, event=event, **options)
    assert baseline["orders"].empty
    assert baseline["replay"].ending_soxx_quantity.iloc[0] == 0
    assert len(candidate["orders"]) == 1
    assert candidate["orders"].slot.iloc[0] == "event"
    assert candidate["orders"].symbol.iloc[0] == "SOXX"
    assert candidate["replay"].ending_soxx_quantity.iloc[0] > 0
    assert candidate["replay"].native_expected_nav.iloc[0] < 100000

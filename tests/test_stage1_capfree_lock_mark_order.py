"""The next-session open mark must follow, not precede, its early bars."""
from types import SimpleNamespace
from math import isclose

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees
from src.backtest.stage1_capfree_lock_episode import replay_capfree_lock_episode


def test_early_next_session_loss_precedes_new_peak():
    days = pd.bdate_range("2019-12-19", periods=5)
    frame = pd.DataFrame({
        "trade_date": days,
        "close_soxl": 100.0,
        "close_soxx": 100.0,
        "soxl_exit_open": 100.0,
        "soxx_exit_open": [120.0, 100.0, 100.0, 100.0, 100.0],
    })
    entries = []
    for i, day in enumerate(days):
        hours = ("09:48", "09:49", "09:50", "09:51", "16:00") if i == 0 else (
            "09:31", "09:47", "09:48", "16:00")
        for hour in hours:
            soxx = 80.0 if i == 1 and hour in ("09:31", "09:47") else 100.0
            entries.append((pd.Timestamp(f"{day.date()} {hour}"), {
                "SOXL_open": 100.0,
                "SOXX_open": 100.0,
                "SOXL_close": 100.0,
                "SOXX_close": soxx,
                "SOXL_raw_open": 100.0,
                "SOXX_raw_open": 100.0,
                "SOXL_price_valid": 1.0,
                "SOXX_price_valid": 1.0,
            }))
    quotes = pd.DataFrame([item for _, item in entries], index=[stamp for stamp, _ in entries])
    state = SimpleNamespace(q=np.array([0.0, 1.0]), cash=0.0, peak=100.0,
                            fees=TieredFees({}, slippage_bps=0))
    checkpoint = SimpleNamespace(index=0, state=state)
    start = days[0]
    event = {
        "trade_date": start,
        "bar_end_at": start + pd.Timedelta(hours=9, minutes=48),
        "decision_at": start + pd.Timedelta(hours=9, minutes=49),
        "fill_at": start + pd.Timedelta(hours=9, minutes=50),
        "desired_weights": np.array([0.0, 1.0, 0.0]),
        "adjusted_fill_prices": np.array([100.0, 100.0]),
        "raw_fill_prices": np.array([100.0, 100.0]),
    }
    result = replay_capfree_lock_episode(frame, checkpoint, quotes, event=event)
    assert result["event_status"] == "no_whole_share_fill"
    assert isclose(result["HOLD"]["terminal_available_mark_max_drawdown"], 0.2, abs_tol=1e-12)
    assert isclose(result["EVENT"]["terminal_available_mark_max_drawdown"], 0.2, abs_tol=1e-12)
    assert result["HOLD"]["path"].iloc[0].next_open_nav == 120.0
    assert isclose(result["HOLD"]["path"].iloc[1].available_mark_max_drawdown, 0.2, abs_tol=1e-12)


def test_fee_loss_is_marked_at_event_fill():
    days = pd.bdate_range("2019-12-19", periods=5)
    frame = pd.DataFrame({"trade_date": days, "close_soxl": 100.0,
                          "close_soxx": 100.0, "soxl_exit_open": 100.0,
                          "soxx_exit_open": 100.0})
    stamps = []
    for offset, day in enumerate(days):
        hours = ("09:48", "09:49", "09:50", "09:51", "16:00") if offset == 0 else (
            "09:31", "09:47", "09:48", "16:00")
        stamps.extend(pd.Timestamp(f"{day.date()} {hour}") for hour in hours)
    quotes = pd.DataFrame({key: 100.0 for key in (
        "SOXL_open", "SOXX_open", "SOXL_close", "SOXX_close",
        "SOXL_raw_open", "SOXX_raw_open")}, index=stamps)
    quotes["SOXL_price_valid"] = quotes["SOXX_price_valid"] = 1.0
    tariff = {"tiers": [{"through_shares": None, "per_share": 0.01}],
              "minimum_per_order": 0.5, "maximum_order_fraction": 1.0,
              "venue_per_share": 0.0, "venue_subdollar_fraction": 0.0,
              "clearing_per_share": 0.0, "clearing_maximum_fraction": 1.0,
              "sec_sale_fraction": 0.0, "taf_per_sold_share": 0.0,
              "taf_maximum_per_trade": 0.0, "cat_per_share": 0.0,
              "commission_pass_through_fraction": 0.0}
    checkpoint = SimpleNamespace(index=0, state=SimpleNamespace(
        q=np.array([0.0, 0.0]), cash=200.0, peak=200.0,
        fees=TieredFees(tariff, slippage_bps=0)))
    start = days[0]
    event = {"trade_date": start,
             "bar_end_at": start + pd.Timedelta(hours=9, minutes=48),
             "decision_at": start + pd.Timedelta(hours=9, minutes=49),
             "fill_at": start + pd.Timedelta(hours=9, minutes=50),
             "desired_weights": np.array([0.0, 0.5, 0.5]),
             "adjusted_fill_prices": np.array([100.0, 100.0]),
             "raw_fill_prices": np.array([100.0, 100.0])}
    result = replay_capfree_lock_episode(frame, checkpoint, quotes, event=event)
    assert result["EVENT"]["orders"][0]["raw_shares"] == 1
    assert result["EVENT"]["orders"][0]["total"] == 0.5
    assert isclose(result["EVENT"]["terminal_available_mark_max_drawdown"], 0.0025, abs_tol=1e-12)
    assert result["HOLD"]["terminal_available_mark_max_drawdown"] == 0.0

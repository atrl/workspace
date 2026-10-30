"""Five-session cap-free lock pilot from one dated Stage1 after-open checkpoint.

This is a mechanical branch reference, not a trained allocator or native Qlib
candidate account. It handles one early event in an explicit no-recovery year.
Both HOLD and EVENT stop old b472 actions at the same event clock.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import settle_tiered_event


def replay_capfree_lock_pilot(frame, checkpoint, minute_bars, *, event, horizon_sessions=5):
    """Return independent HOLD/EVENT paths with own shares, fees and peaks."""
    required = {"trade_date", "bar_end_at", "decision_at", "fill_at", "desired_weights",
                "adjusted_fill_prices", "raw_fill_prices"}
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("blocked: frozen event fields differ")
    if horizon_sessions != 5 or checkpoint.index < 0 or checkpoint.index + 5 > len(frame):
        raise ValueError("blocked: fixed five-session terminal is unavailable")
    row0 = frame.iloc[checkpoint.index]
    day = pd.Timestamp(row0.trade_date)
    bar_end, decision, fill = (pd.Timestamp(event[k]) for k in
                                ("bar_end_at", "decision_at", "fill_at"))
    if (pd.Timestamp(event["trade_date"]) != day or bar_end.normalize() != day or
            decision != bar_end + pd.Timedelta(minutes=1) or
            fill != bar_end + pd.Timedelta(minutes=2) or
            fill <= day + pd.Timedelta(hours=9, minutes=47) or
            fill >= day + pd.Timedelta(hours=10, minutes=32)):
        raise ValueError("blocked: pilot requires a causal pre-recovery event")
    desired = np.asarray(event["desired_weights"], float)
    adjusted = np.asarray(event["adjusted_fill_prices"], float)
    raw = np.asarray(event["raw_fill_prices"], float)
    if (desired.shape != (3,) or adjusted.shape != (2,) or raw.shape != (2,) or
            not np.isfinite([*desired, *adjusted, *raw]).all() or
            desired.min() < 0 or abs(desired.sum() - 1) > 1e-10 or
            adjusted.min() <= 0 or raw.min() <= 0):
        raise ValueError("blocked: event target or price is invalid")
    required_quotes = {"SOXL_close", "SOXX_close", "SOXL_price_valid", "SOXX_price_valid"}
    if not isinstance(minute_bars, pd.DataFrame) or not required_quotes <= set(minute_bars):
        raise ValueError("blocked: Qlib pair minute marks missing")
    if minute_bars.index.tz is not None or not minute_bars.index.is_unique:
        raise ValueError("blocked: minute marks must be unique local ET")
    if bar_end not in minute_bars.index:
        raise ValueError("blocked: completed decision bar is absent")
    decision_price = minute_bars.loc[bar_end, ["SOXL_close", "SOXX_close"]].to_numpy(float)
    before = deepcopy(checkpoint.state)
    decision_nav = float(before.q @ decision_price + before.cash)
    if decision_nav <= 0 or not np.isfinite(decision_nav):
        raise ValueError("blocked: decision NAV invalid")
    results = {}
    for arm in ("HOLD", "EVENT"):
        state = deepcopy(before)
        state.peak = max(float(state.peak), decision_nav)
        event_orders = []
        if arm == "EVENT":
            trade = settle_tiered_event(state.q, state.cash, desired=desired,
                adjusted_prices=adjusted, raw_prices=raw, stamp=fill,
                fees=state.fees)
            state.q, state.cash = trade["quantities"], float(trade["cash"])
            event_orders = trade["orders"]
            if (len(event_orders) != 1 or event_orders[0]["sell"] or
                    event_orders[0]["symbol"] != "SOXX" or state.cash < -1e-8):
                raise ValueError("blocked: fixed SOXX arm was not one funded whole-share buy")
        rows = []
        monitored = max(0.0, 1.0 - decision_nav / state.peak)
        for offset in range(5):
            row = frame.iloc[checkpoint.index + offset]
            session = pd.Timestamp(row.trade_date)
            marks = minute_bars.loc[minute_bars.index.normalize() == session]
            if offset == 0:
                marks = marks.loc[marks.index >= fill + pd.Timedelta(minutes=1)]
            if marks.empty or not marks[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all().all():
                raise ValueError("blocked: incomplete five-session Qlib minute marks")
            prices = marks[["SOXL_close", "SOXX_close"]].to_numpy(float)
            for price in prices:
                nav = float(state.q @ price + state.cash)
                if nav <= 0 or not np.isfinite(nav):
                    raise ValueError("blocked: nonpositive marked NAV")
                state.peak = max(float(state.peak), nav)
                monitored = max(monitored, 1.0 - nav / state.peak)
            close_price = row[["close_soxl", "close_soxx"]].to_numpy(float)
            exit_price = row[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
            close_nav = float(state.q @ close_price + state.cash)
            if abs(close_nav - float(state.q @ prices[-1] + state.cash)) > 1e-6:
                raise ValueError("blocked: Qlib final minute differs from account close plane")
            state.peak = max(float(state.peak), close_nav)
            next_open_nav = float(state.q @ exit_price + state.cash)
            rows.append({"trade_date": session, "next_open_nav": next_open_nav,
                         "close_nav": close_nav, "ending_cash": float(state.cash),
                         "soxl_quantity": float(state.q[0]),
                         "soxx_quantity": float(state.q[1]),
                         "minute_close_max_drawdown": float(monitored),
                         "event_fee_usd": float(sum(o["total"] for o in event_orders)) if offset == 0 else 0.0})
        results[arm] = {"path": pd.DataFrame(rows), "orders": event_orders,
                        "ending_quantities": state.q.copy(),
                        "ending_cash": float(state.cash),
                        "ending_peak": float(state.peak),
                        "monthly_fee_shares": deepcopy(state.fees.monthly_shares)}
    results["decision_nav"] = decision_nav
    return results

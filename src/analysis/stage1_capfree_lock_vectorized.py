"""Vectorized five-session cap-free lock values for one known prefill state.

The parent state and its peak must be known before the first source minute.
The compiled Qlib price plane is chronological, including next opens at their
actual 09:47 position and the fifth terminal open after the last close.
This computes source-conditioned labels, not a trained live policy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import settle_tiered_event
from src.backtest.stage1_fixed_share_event import (
    fixed_buy_intent as make_fixed_buy_intent,
    settle_fixed_buy,
)


@dataclass(frozen=True)
class LockPricePlane:
    stamps: np.ndarray
    prices: np.ndarray
    quotes: pd.DataFrame
    first_day: pd.Timestamp
    terminal_price: np.ndarray


def compile_lock_price_plane(frame_window: pd.DataFrame, quotes: pd.DataFrame) -> LockPricePlane:
    if len(frame_window) != 5 or not isinstance(quotes.index, pd.DatetimeIndex) or quotes.index.tz is not None:
        raise ValueError("blocked: five-session Qlib local-ET price plane required")
    required = {"SOXL_open", "SOXX_open", "SOXL_close", "SOXX_close",
                "SOXL_raw_open", "SOXX_raw_open", "SOXL_price_valid", "SOXX_price_valid"}
    if not required <= set(quotes) or not quotes.index.is_unique:
        raise ValueError("blocked: Qlib pair quote schema differs")
    frame_window = frame_window.reset_index(drop=True)
    stamps, prices = [], []
    for offset, row in frame_window.iterrows():
        day = pd.Timestamp(row.trade_date)
        bars = quotes.loc[quotes.index.normalize() == day]
        if offset == 0:
            bars = bars.loc[bars.index >= day + pd.Timedelta(hours=9, minutes=48)]
        if bars.empty or not bars[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all().all():
            raise ValueError("blocked: incomplete five-session minute source")
        if offset == 0:
            pieces = [(bars.index, bars[["SOXL_close", "SOXX_close"]].to_numpy(float))]
        else:
            boundary = day + pd.Timedelta(hours=9, minutes=47)
            early = bars.loc[bars.index <= boundary]
            late = bars.loc[bars.index > boundary]
            if early.empty or late.empty:
                raise ValueError("blocked: next-open minute boundary absent")
            previous = frame_window.iloc[offset - 1]
            open_price = previous[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
            pieces = [(early.index, early[["SOXL_close", "SOXX_close"]].to_numpy(float)),
                      ([boundary], open_price.reshape(1, 2)),
                      (late.index, late[["SOXL_close", "SOXX_close"]].to_numpy(float))]
        for times, values in pieces:
            stamps.extend(pd.DatetimeIndex(times).to_numpy(dtype="datetime64[ns]"))
            prices.extend(values)
    final = frame_window.iloc[-1]
    terminal = final[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
    terminal_stamp = pd.Timestamp(final.exit_date) + pd.Timedelta(hours=9, minutes=47)
    stamps.append(terminal_stamp.to_datetime64())
    prices.append(terminal)
    stamp_array, price_array = np.asarray(stamps, dtype="datetime64[ns]"), np.asarray(prices, float)
    if not np.isfinite(price_array).all() or price_array.min() <= 0 or np.any(stamp_array[1:] < stamp_array[:-1]):
        raise ValueError("blocked: chronological price plane invalid")
    return LockPricePlane(stamp_array, price_array, quotes,
                          pd.Timestamp(frame_window.iloc[0].trade_date), terminal)


def _path_risk(q, cash, peak, initial_drawdown, prices):
    nav = prices @ q + cash
    if not np.isfinite(nav).all() or np.min(nav) <= 0:
        raise ValueError("blocked: lock NAV path invalid")
    running_peak = np.maximum(float(peak), np.maximum.accumulate(nav))
    max_drawdown = max(float(initial_drawdown), float(np.max(1.0 - nav / running_peak)))
    return float(nav[-1]), max_drawdown, float(running_peak[-1])


def value_lock_event(plane: LockPricePlane, *, bar_end_at, fill_at,
                     quantities, cash, starting_peak, fees, desired_weights=None,
                     fixed_intent=None, start_at=None):
    """Settle one arbitrary simplex target and return HOLD/EVENT path values."""
    bar_end, fill = pd.Timestamp(bar_end_at), pd.Timestamp(fill_at)
    if bar_end.normalize() != plane.first_day or fill != bar_end + pd.Timedelta(minutes=2):
        raise ValueError("blocked: event clock differs")
    q = np.asarray(quantities, float)
    cash = float(cash)
    if q.shape != (2,) or q.min() < -1e-8 or cash < -1e-8 or starting_peak <= 0:
        raise ValueError("blocked: pre-event account state invalid")
    day = plane.first_day
    start = (pd.Timestamp(start_at) if start_at is not None else
             day + pd.Timedelta(hours=9, minutes=48))
    if (start.normalize() != day or
            start < day + pd.Timedelta(hours=9, minutes=48) or start > bar_end):
        raise ValueError("blocked: pre-event state availability clock differs")
    times = plane.stamps
    decision_index = int(np.searchsorted(times, bar_end.to_datetime64(), side="left"))
    fill_index = int(np.searchsorted(times, fill.to_datetime64(), side="left"))
    if (decision_index >= len(times) or fill_index >= len(times) or
            times[decision_index] != bar_end.to_datetime64() or
            times[fill_index] != fill.to_datetime64() or
            fill_index != decision_index + 2):
        raise ValueError("blocked: two completed prefill minute marks absent")
    start_index = int(np.searchsorted(times, start.to_datetime64(), side="left"))
    if start_index >= len(times) or times[start_index] != start.to_datetime64():
        raise ValueError("blocked: pre-event state start bar absent")
    prior_nav = plane.prices[start_index:decision_index + 1] @ q + cash
    decision_nav = float(prior_nav[-1])
    if (desired_weights is None) == (fixed_intent is None):
        raise ValueError("blocked: choose exactly one event order contract")
    if fixed_intent is not None:
        symbol = fixed_intent.get("symbol") if isinstance(fixed_intent, dict) else None
        field = symbol + "_raw_close" if symbol in ("SOXL", "SOXX") else ""
        if field not in plane.quotes or bar_end not in plane.quotes.index:
            raise ValueError("blocked: completed raw-close source absent")
        raw_close = float(plane.quotes.loc[bar_end, field])
        expected = make_fixed_buy_intent(symbol=symbol, decision_nav=decision_nav,
            raw_close=raw_close, cash=cash, fees=fees,
            decision_at=bar_end + pd.Timedelta(minutes=1),
            nominal_beta_increment=float(fixed_intent["nominal_beta_increment"]))
        if (fixed_intent["status"] != expected["status"] or
                fixed_intent["raw_shares"] != expected["raw_shares"] or
                abs(float(fixed_intent["decision_nav"]) - decision_nav) > 1e-7 or
                abs(float(fixed_intent["decision_raw_close"]) - raw_close) > 1e-8):
            raise ValueError("blocked: fixed buy differs from decision-known source")
    decision_peak = max(float(starting_peak), float(np.max(prior_nav)))
    prefill_nav = plane.prices[decision_index:fill_index + 1] @ q + cash
    prefill_peaks = np.maximum(decision_peak, np.maximum.accumulate(prefill_nav))
    prefill_drawdown = float(np.max(1.0 - prefill_nav / prefill_peaks))
    peak_at_fill = float(prefill_peaks[-1])
    quote_stamp = fill + pd.Timedelta(minutes=1)
    if quote_stamp not in plane.quotes.index:
        raise ValueError("blocked: Qlib next-bar fill quote absent")
    fillbar = plane.quotes.loc[quote_stamp]
    adjusted = fillbar[["SOXL_open", "SOXX_open"]].to_numpy(float)
    raw = fillbar[["SOXL_raw_open", "SOXX_raw_open"]].to_numpy(float)
    future_prices = plane.prices[fill_index + 1:]
    if len(future_prices) == 0:
        raise ValueError("blocked: no postfill path")
    result = {"decision_nav": decision_nav, "fill_at": fill,
              "label_available_at": pd.Timestamp(times[-1]) + pd.Timedelta(minutes=2)}
    for arm in ("HOLD", "EVENT"):
        shares, balance = q.copy(), cash
        branch_fees = deepcopy(fees)
        orders = []
        status = "hold"
        if arm == "EVENT":
            if fixed_intent is not None:
                trade = settle_fixed_buy(shares, balance, intent=fixed_intent,
                    adjusted_fill_prices=adjusted, raw_fill_prices=raw,
                    fill_at=fill, fees=branch_fees)
                status = trade["status"]
            else:
                trade = settle_tiered_event(shares, balance,
                    desired=np.asarray(desired_weights, float),
                    adjusted_prices=adjusted, raw_prices=raw,
                    stamp=fill, fees=branch_fees)
                status = "funded" if trade["orders"] else "no_whole_share_fill"
            shares, balance, orders = trade["quantities"], float(trade["cash"]), trade["orders"]
        fill_nav = float(shares @ adjusted + balance)
        if fill_nav <= 0:
            raise ValueError("blocked: fill NAV nonpositive")
        fill_peak = max(peak_at_fill, fill_nav)
        fill_drawdown = max(prefill_drawdown, 1.0 - fill_nav / fill_peak)
        terminal_nav, max_drawdown, ending_peak = _path_risk(
            shares, balance, fill_peak, fill_drawdown, future_prices)
        result[arm] = {"terminal_nav": terminal_nav, "max_drawdown": max_drawdown,
                       "ending_peak": ending_peak, "quantities": shares,
                       "cash": balance, "orders": orders,
                       "status": status,
                       "fee_usd": float(sum(o["total"] for o in orders)),
                       "monthly_fee_shares": deepcopy(branch_fees.monthly_shares)}
    return result

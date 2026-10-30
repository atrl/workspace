"""Source and whole-share support for five-session cap-free minute lock labels.

Decision-known intentions are counted before future fills. Retrospective funded
and path-source masks are separate, and no future economic value is consumed.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.backtest.event_guard_clock import guard_slot
from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event


def _position(position):
    return np.array([position.get_stock_amount(s) if position.check_stock(s) else 0.
                     for s in ("SOXL", "SOXX")], float), float(position.get_cash())


def _funded(q, cash, mark, fill, raw, fees, stamp, delta_beta):
    nav = float(q @ mark + cash)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("blocked: decision NAV invalid")
    weights = np.r_[q * mark, cash] / nav
    if weights[2] < delta_beta - 1e-10:
        return False, False
    results = []
    for symbol, change in (("SOXL", np.array([delta_beta / 3, 0., -delta_beta / 3])),
                           ("SOXX", np.array([0., delta_beta, -delta_beta]))):
        target = weights + change
        trade = settle_tiered_event(q, cash, desired=target,
            adjusted_prices=fill, raw_prices=raw, stamp=stamp, fees=deepcopy(fees))
        orders = trade["orders"]
        results.append(bool(len(orders) == 1 and orders[0]["symbol"] == symbol and
                            not orders[0]["sell"] and trade["cash"] >= -1e-8))
    return tuple(results)


def audit_capfree_lock_support(frame, book, quotes, saved_labels, *, tariff,
                               delta_beta=.10, horizon_sessions=5):
    """Return per-day source/funding counts without returning minute matrices."""
    frame = frame.reset_index(drop=True)
    path = book["independent_replay"].reset_index(drop=True)
    if len(frame) != len(path) or not frame.trade_date.equals(path.trade_date):
        raise ValueError("blocked: parent source calendar differs")
    if horizon_sessions != 5 or delta_beta != .10:
        raise ValueError("blocked: frozen action or horizon differs")
    if not isinstance(quotes.index, pd.DatetimeIndex) or quotes.index.tz is not None:
        raise ValueError("blocked: Qlib minute clock must be local naive ET")
    fields = ["SOXL_close", "SOXX_close", "SOXL_open", "SOXX_open",
              "SOXL_raw_open", "SOXX_raw_open", "SOXL_price_valid", "SOXX_price_valid"]
    if not set(fields) <= set(quotes) or not quotes.index.is_unique:
        raise ValueError("blocked: Qlib minute pair schema differs")
    by_day = {pd.Timestamp(day): part for day, part in quotes.groupby(quotes.index.normalize(), sort=True)}
    source_days = set(pd.DatetimeIndex(frame.trade_date))
    if not source_days <= set(by_day):
        raise ValueError("blocked: Qlib minute source misses a parent day")
    label_keys = saved_labels[["trade_date", "bar_end_at", "soxl_funded", "soxx_funded"]].copy()
    label_keys["trade_date"] = pd.to_datetime(label_keys.trade_date)
    label_keys["bar_end_at"] = pd.to_datetime(label_keys.bar_end_at)
    if label_keys.duplicated(["trade_date", "bar_end_at"]).any():
        raise ValueError("blocked: saved label minute identity duplicated")
    saved = {(pd.Timestamp(r.trade_date), pd.Timestamp(r.bar_end_at)):
             (bool(r.soxl_funded), bool(r.soxx_funded))
             for r in label_keys.itertuples(index=False)}
    orders = book["orders"].sort_values("stamp")
    orders_by_day = {pd.Timestamp(day): part for day, part in orders.groupby("trade_date", sort=True)}
    fee_book = TieredFees(tariff, slippage_bps=0)
    fee_snapshots = {}
    for row in frame.itertuples(index=False):
        day = pd.Timestamp(row.trade_date)
        current = orders_by_day.get(day, pd.DataFrame())
        for order in current.itertuples(index=False):
            if order.slot == "open":
                fee_book.commit(int(order.raw_shares), stamp=order.stamp,
                                cap_applied=bool(order.cap_applied))
        early = deepcopy(fee_book)
        for order in current.itertuples(index=False):
            if order.slot == "recovery":
                fee_book.commit(int(order.raw_shares), stamp=order.stamp,
                                cap_applied=bool(order.cap_applied))
        fee_snapshots[day] = (early, deepcopy(fee_book))
    full_valid = {day: bool(part[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all().all())
                  for day, part in by_day.items()}
    rows = []
    for index, row in frame.iterrows():
        day = pd.Timestamp(row.trade_date)
        asset = by_day[day]
        close = asset.index.max()
        current = path.iloc[index]
        p0 = book["native_positions"].get(day + pd.Timedelta(hours=9, minutes=47))
        p1 = book["native_positions"].get(day + pd.Timedelta(hours=10, minutes=32))
        if p0 is None or p1 is None:
            raise ValueError("blocked: dated parent minute position absent")
        q0, c0 = _position(p0)
        q1, c1 = _position(p1)
        if not bool(current.recovery_action) and (not np.array_equal(q0, q1) or abs(c0-c1)>1e-7):
            raise ValueError("blocked: parent post-10:32 state changed without recovery")
        maturity = index + horizon_sessions <= len(frame)
        future_days = [pd.Timestamp(v) for v in frame.trade_date.iloc[index:index+horizon_sessions]]
        future_valid = maturity and all(full_valid[d] for d in future_days[1:])
        counts = dict(total_clock_slots=0, ambiguous_clock=0, decision_source_valid=0,
                      cash_intents=0, mature_path_source_intents=0,
                      fill_source_valid=0, soxl_funded=0, soxx_funded=0,
                      paired_funded=0, no_whole_share_pair=0,
                      saved_funding_reused=0, newly_settled_funding=0)
        first = day + pd.Timedelta(hours=9, minutes=48)
        valid = asset[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all(axis=1)
        invalid_prefix = (~valid.loc[valid.index >= first]).cumsum()
        for stamp in asset.index:
            if stamp < first:
                continue
            slot = guard_slot(stamp, session_open=day + pd.Timedelta(hours=9, minutes=30),
                              session_close=close,
                              original_open_fill=day + pd.Timedelta(hours=9, minutes=47))
            if slot is None:
                continue
            counts["total_clock_slots"] += 1
            recovery_clock = day + pd.Timedelta(hours=10, minutes=32)
            if not (slot.fill_at < recovery_clock or stamp > recovery_clock):
                counts["ambiguous_clock"] += 1
                continue
            if not bool(valid.loc[stamp]) or int(invalid_prefix.loc[stamp]) > 0:
                continue
            counts["decision_source_valid"] += 1
            late = stamp > recovery_clock
            q, cash = (q1, c1) if late else (q0, c0)
            mark = asset.loc[stamp, ["SOXL_close", "SOXX_close"]].to_numpy(float)
            nav = float(q @ mark + cash)
            if nav <= 0 or not np.isfinite(nav):
                raise ValueError("blocked: parent minute decision NAV invalid")
            if cash / nav < delta_beta - 1e-10:
                continue
            counts["cash_intents"] += 1
            fill_stamp = slot.fill_at + pd.Timedelta(minutes=1)
            if fill_stamp not in asset.index or not bool(valid.loc[fill_stamp]):
                continue
            counts["fill_source_valid"] += 1
            # The label path must be complete after the future fill and across
            # the remaining four sessions; this is never a live decision mask.
            current_suffix = valid.loc[valid.index >= fill_stamp]
            if not (future_valid and bool(current_suffix.all())):
                continue
            counts["mature_path_source_intents"] += 1
            key = (day, stamp)
            if key in saved:
                soxl, soxx = saved[key]
                counts["saved_funding_reused"] += 1
            else:
                fill = asset.loc[fill_stamp, ["SOXL_open", "SOXX_open"]].to_numpy(float)
                raw = asset.loc[fill_stamp, ["SOXL_raw_open", "SOXX_raw_open"]].to_numpy(float)
                soxl, soxx = _funded(q, cash, mark, fill, raw,
                    fee_snapshots[day][1 if late else 0], slot.fill_at, delta_beta)
                counts["newly_settled_funding"] += 1
            counts["soxl_funded"] += int(soxl)
            counts["soxx_funded"] += int(soxx)
            counts["paired_funded"] += int(soxl and soxx)
            counts["no_whole_share_pair"] += int(not (soxl and soxx))
        rows.append({"trade_date": day, "year": int(day.year),
                     "recovery_action": bool(current.recovery_action),
                     "five_session_mature": bool(maturity),
                     "future_four_full_source_valid": bool(future_valid),
                     "session_close": close, **counts})
    result = pd.DataFrame(rows)
    if len(result) != len(frame):
        raise ValueError("blocked: source daily audit calendar incomplete")
    return result

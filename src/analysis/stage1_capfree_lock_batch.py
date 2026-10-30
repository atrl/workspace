"""One frozen all-minute five-session HOLD/SOXL/SOXX label pass.

Every decision-known intent remains in the output. Fill legality and future
price coverage only determine retrospective label status. No model is fitted.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd

from src.analysis.stage1_capfree_lock_vectorized import compile_lock_price_plane, value_lock_event
from src.backtest.event_guard_clock import guard_slot
from src.backtest.ibkr_tiered import TieredFees


def _position(pos):
    return np.array([pos.get_stock_amount(s) if pos.check_stock(s) else 0.
                     for s in ("SOXL", "SOXX")], float), float(pos.get_cash())


def _fee_snapshots(frame, orders, tariff):
    by_day = {pd.Timestamp(day): part for day, part in orders.groupby("trade_date", sort=True)}
    fees, snapshots = TieredFees(tariff, slippage_bps=0), {}
    for row in frame.itertuples(index=False):
        day = pd.Timestamp(row.trade_date)
        current = by_day.get(day, pd.DataFrame())
        for order in current.itertuples(index=False):
            if order.slot == "open":
                fees.commit(int(order.raw_shares), stamp=order.stamp,
                            cap_applied=bool(order.cap_applied))
        early = deepcopy(fees)
        for order in current.itertuples(index=False):
            if order.slot == "recovery":
                fees.commit(int(order.raw_shares), stamp=order.stamp,
                            cap_applied=bool(order.cap_applied))
        snapshots[day] = (early, deepcopy(fees))
    return snapshots


def _opening_peaks(frame, book):
    targets = book["targets"].reset_index(drop=True)
    if len(targets) != len(frame) or not targets.trade_date.equals(frame.trade_date):
        raise ValueError("blocked: target/price calendars differ")
    peak = float(targets.iloc[0].known_equity)
    values, maximum_error = [], 0.0
    positions = book["native_positions"]
    for i, row in frame.iterrows():
        day = pd.Timestamp(row.trade_date)
        known = float(targets.iloc[i].known_equity)
        peak = max(peak, known)
        error = abs(max(0., 1. - known / peak) - float(targets.iloc[i].known_drawdown))
        maximum_error = max(maximum_error, error)
        if error > 1e-8:
            raise ValueError("blocked: dated opening high-water state differs")
        values.append(peak)
        close_stamp = max(stamp for stamp in positions if stamp.normalize() == day)
        q, cash = _position(positions[close_stamp])
        close_price = row[["close_soxl", "close_soxx"]].to_numpy(float)
        peak = max(peak, float(q @ close_price + cash))
    return np.array(values, float), maximum_error


def build_capfree_lock_labels(frame, book, quotes, perday_support, *, tariff,
                              maximum_intent_rows=400000, maximum_seconds=900):
    frame = frame.reset_index(drop=True)
    path = book["independent_replay"].reset_index(drop=True)
    if len(frame) != len(path) or not frame.trade_date.equals(path.trade_date):
        raise ValueError("blocked: dated parent calendar differs")
    support = perday_support.reset_index(drop=True)
    if len(support) != len(frame) or not pd.to_datetime(support.trade_date).equals(frame.trade_date):
        raise ValueError("blocked: source-support calendar differs")
    fee_book = _fee_snapshots(frame, book["orders"], tariff)
    peaks, peak_error = _opening_peaks(frame, book)
    by_day = {pd.Timestamp(day): part for day, part in quotes.groupby(quotes.index.normalize(), sort=True)}
    if set(frame.trade_date) - set(by_day):
        raise ValueError("blocked: Qlib minute source misses parent session")
    result = []
    started = perf_counter()
    counts = {"intent_rows":0, "mature_path_rows":0, "soxl_funded":0,
              "soxx_funded":0, "paired_funded":0, "unknown_path":0,
              "terminal_unknown":0, "no_exact_whole_share_pair":0}
    for index, row in frame.iterrows():
        day = pd.Timestamp(row.trade_date)
        asset = by_day[day]
        source = support.iloc[index]
        p0 = book["native_positions"].get(day + pd.Timedelta(hours=9, minutes=47))
        p1 = book["native_positions"].get(day + pd.Timedelta(hours=10, minutes=32))
        if p0 is None or p1 is None:
            raise ValueError("blocked: dated parent event state absent")
        q0, cash0 = _position(p0)
        q1, cash1 = _position(p1)
        first = day + pd.Timedelta(hours=9, minutes=48)
        recovery_clock = day + pd.Timedelta(hours=10, minutes=32)
        valid = asset[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all(axis=1)
        invalid_prefix = (~valid.loc[valid.index >= first]).cumsum()
        early = asset.loc[(asset.index >= first) & (asset.index <= recovery_clock)]
        if not early.empty and bool(valid.loc[early.index].all()):
            early_nav = early[["SOXL_close", "SOXX_close"]].to_numpy(float) @ q0 + cash0
            recovery_peak = max(float(peaks[index]), float(np.max(early_nav)))
        else:
            recovery_peak = float("nan")
        late = asset.loc[asset.index > recovery_clock]
        early_peak_series = pd.Series(np.maximum(float(peaks[index]),
            np.maximum.accumulate(early[["SOXL_close", "SOXX_close"]].to_numpy(float) @ q0 + cash0)) if len(early) else [], index=early.index)
        late_peak_series = pd.Series(np.maximum(recovery_peak,
            np.maximum.accumulate(late[["SOXL_close", "SOXX_close"]].to_numpy(float) @ q1 + cash1)) if len(late) else [], index=late.index)
        maturity = bool(source.five_session_mature)
        future_valid = bool(source.future_four_full_source_valid)
        plane = None
        if source.mature_path_source_intents > 0:
            plane = compile_lock_price_plane(frame.iloc[index:index + 5],
                quotes.loc[(quotes.index.normalize().isin(frame.trade_date.iloc[index:index + 5]))])
        day_counts = {"intent_rows":0, "mature_path_rows":0, "paired_funded":0}
        for stamp in asset.index:
            slot = guard_slot(stamp, session_open=day + pd.Timedelta(hours=9, minutes=30),
                session_close=asset.index.max(), original_open_fill=day + pd.Timedelta(hours=9, minutes=47)) if stamp >= first else None
            if slot is None or not (slot.fill_at < recovery_clock or stamp > recovery_clock):
                continue
            if not bool(valid.loc[stamp]) or int(invalid_prefix.loc[stamp]) > 0:
                continue
            is_late = stamp > recovery_clock
            q, cash = (q1, cash1) if is_late else (q0, cash0)
            mark = asset.loc[stamp, ["SOXL_close", "SOXX_close"]].to_numpy(float)
            decision_nav = float(q @ mark + cash)
            if decision_nav <= 0 or not np.isfinite(decision_nav):
                raise ValueError("blocked: decision NAV invalid")
            cash_weight = cash / decision_nav
            if cash_weight < .10 - 1e-10:
                continue
            counts["intent_rows"] += 1
            day_counts["intent_rows"] += 1
            if counts["intent_rows"] > maximum_intent_rows:
                raise ValueError("blocked: frozen intent row budget exceeded")
            decision_peak = float(late_peak_series.loc[stamp] if is_late else early_peak_series.loc[stamp])
            decision_drawdown = max(0., 1. - decision_nav / decision_peak)
            terminal = frame.iloc[index + 4] if maturity else None
            label_at = (pd.Timestamp(terminal.exit_date) + pd.Timedelta(hours=9, minutes=49)) if maturity else pd.NaT
            record = {"trade_date":day,"bar_end_at":stamp,"decision_at":slot.decision_at,
                      "fill_at":slot.fill_at,"label_available_at":label_at,
                      "decision_nav":decision_nav,"decision_cash_weight":cash_weight,
                      "decision_own_drawdown":decision_drawdown,"parent_risk_state":int(path.iloc[index].risk_state),
                      "phase":"after_1032" if is_late else "before_1032",
                      "status":"unknown","soxl_funded":False,"soxx_funded":False,
                      "soxl_raw_shares":0,"soxx_raw_shares":0,"soxl_fee_usd":0.,"soxx_fee_usd":0.,
                      "hold_terminal_nav":np.nan,"hold_max_drawdown":np.nan,
                      "soxl_value":np.nan,"soxx_value":np.nan,"relative_value_soxl_minus_soxx":np.nan,
                      "soxl_tail_damage":np.nan,"soxx_tail_damage":np.nan}
            if not maturity:
                record["status"]="terminal_unknown"
                counts["terminal_unknown"] += 1
                result.append(record)
                continue
            fill_stamp = slot.fill_at + pd.Timedelta(minutes=1)
            path_valid = (future_valid and plane is not None and
                          fill_stamp in asset.index and bool(valid.loc[fill_stamp]) and
                          bool(valid.loc[valid.index >= fill_stamp].all()))
            if not path_valid:
                record["status"]="path_source_unknown"
                counts["unknown_path"] += 1
                result.append(record)
                continue
            counts["mature_path_rows"] += 1
            day_counts["mature_path_rows"] += 1
            starting_peak = recovery_peak if is_late else float(peaks[index])
            if not np.isfinite(starting_peak):
                raise ValueError("blocked: event start peak unknown")
            start_at = day + pd.Timedelta(hours=10, minutes=33) if is_late else first
            fees = fee_book[day][1 if is_late else 0]
            weights = np.r_[q * mark, cash] / decision_nav
            targets = {"SOXL":weights + np.array([.10 / 3, 0., -.10 / 3]),
                       "SOXX":weights + np.array([0., .10, -.10])}
            arms = {}
            for symbol, target in targets.items():
                arms[symbol] = value_lock_event(plane, bar_end_at=stamp, fill_at=slot.fill_at,
                    quantities=q, cash=cash, starting_peak=starting_peak, fees=fees,
                    desired_weights=target, start_at=start_at)
            hold = arms["SOXL"]["HOLD"]
            other_hold = arms["SOXX"]["HOLD"]
            if (abs(hold["terminal_nav"] - other_hold["terminal_nav"]) > 1e-7 or
                    abs(hold["max_drawdown"] - other_hold["max_drawdown"]) > 1e-10 or
                    abs(arms["SOXL"]["decision_nav"] - decision_nav) > 1e-7 or
                    abs(arms["SOXX"]["decision_nav"] - decision_nav) > 1e-7):
                raise ValueError("blocked: same-state HOLD or decision identity differs")
            record["hold_terminal_nav"] = hold["terminal_nav"]
            record["hold_max_drawdown"] = hold["max_drawdown"]
            for symbol in ("SOXL", "SOXX"):
                branch = arms[symbol]["EVENT"]
                orders = branch["orders"]
                funded = bool(len(orders) == 1 and orders[0]["symbol"] == symbol and
                              not orders[0]["sell"] and branch["cash"] >= -1e-8)
                record[symbol.lower() + "_funded"] = funded
                if funded:
                    record[symbol.lower() + "_raw_shares"] = int(orders[0]["raw_shares"])
                    record[symbol.lower() + "_fee_usd"] = float(branch["fee_usd"])
                    record[symbol.lower() + "_value"] = (branch["terminal_nav"] - hold["terminal_nav"]) / decision_nav
                    record[symbol.lower() + "_tail_damage"] = branch["max_drawdown"] - hold["max_drawdown"]
                    counts[symbol.lower() + "_funded"] += 1
            if record["soxl_funded"] and record["soxx_funded"]:
                record["status"]="paired_funded"
                record["relative_value_soxl_minus_soxx"] = record["soxl_value"] - record["soxx_value"]
                counts["paired_funded"] += 1
                day_counts["paired_funded"] += 1
            else:
                record["status"]="no_exact_whole_share_pair"
                counts["no_exact_whole_share_pair"] += 1
            result.append(record)
        source_keys = {"intent_rows": "cash_intents",
                       "mature_path_rows": "mature_path_source_intents",
                       "paired_funded": "paired_funded"}
        for key, source_key in source_keys.items():
            if day_counts[key] != int(source[source_key]):
                raise ValueError("blocked: full label day/source-support counts differ")
        if index % 100 == 0:
            print("label_progress", index, counts["intent_rows"], round(perf_counter() - started, 1), flush=True)
        if perf_counter() - started > maximum_seconds:
            raise ValueError("blocked: frozen batch wall-time budget exceeded")
    labels = pd.DataFrame(result)
    if len(labels) != counts["intent_rows"]:
        raise ValueError("blocked: label row inventory differs")
    return labels, {"counts":counts,"opening_peak_max_error":peak_error,
                    "elapsed_seconds":perf_counter() - started}

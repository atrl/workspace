"""Same-state five-session lock episodes for causal minute interventions.

The dated Stage1 controller supplies the state before the first event. Both
branches then use the same cap-free lock rule: no b472 action for five sessions,
long-only whole-share Tiered settlement, own cash and peak. Source recovery at
10:32 may precede a late event; the adjacent ambiguous bars are rejected.
This is a label engine, not a trained receiver or native candidate account.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.backtest.event_guard_clock import guard_slot
from src.backtest.ibkr_tiered import settle_tiered_event
from src.backtest.stage1_fixed_share_event import fixed_buy_intent, settle_fixed_buy


_QUOTE_FIELDS = ("SOXL_open", "SOXX_open", "SOXL_close", "SOXX_close",
                 "SOXL_raw_open", "SOXX_raw_open", "SOXL_price_valid", "SOXX_price_valid")


def replay_capfree_lock_episode(frame, checkpoint, quotes, *, event,
                                recovery_state=None, horizon_sessions=5):
    """Replay arbitrary funded simplex target versus same-clock HOLD lock.

    ``recovery_state`` is the verified native parent position and already
    executed recovery orders at 10:32, supplied only when the event is later.
    An event order can round to no fill; that remains an explicit episode.
    """
    required = {"trade_date", "bar_end_at", "decision_at", "fill_at",
                "adjusted_fill_prices", "raw_fill_prices"}
    if not isinstance(event, dict) or (set(event) not in
            (required | {"desired_weights"}, required | {"fixed_buy_intent"})):
        raise ValueError("blocked: event contract differs")
    fixed_mode = "fixed_buy_intent" in event
    if horizon_sessions != 5 or checkpoint.index < 0 or checkpoint.index + 5 > len(frame):
        raise ValueError("blocked: five-session terminal unavailable")
    if not isinstance(quotes, pd.DataFrame) or not set(_QUOTE_FIELDS) <= set(quotes):
        raise ValueError("blocked: complete Qlib minute plane required")
    if not isinstance(quotes.index, pd.DatetimeIndex) or quotes.index.tz is not None or not quotes.index.is_unique:
        raise ValueError("blocked: minute clock must be unique naive ET")
    first = frame.iloc[checkpoint.index]
    day = pd.Timestamp(first.trade_date)
    bar_end, decision, fill = (pd.Timestamp(event[k]) for k in
                                ("bar_end_at", "decision_at", "fill_at"))
    first_quotes = quotes.loc[quotes.index.normalize() == day]
    if first_quotes.empty:
        raise ValueError("blocked: event session quotes absent")
    session_close = first_quotes.index.max()
    slot = guard_slot(bar_end, session_open=day + pd.Timedelta(hours=9, minutes=30),
                      session_close=session_close,
                      original_open_fill=day + pd.Timedelta(hours=9, minutes=47))
    if (pd.Timestamp(event["trade_date"]) != day or slot is None or
            slot.decision_at != decision or slot.fill_at != fill or
            bar_end not in first_quotes.index or
            fill + pd.Timedelta(minutes=1) not in first_quotes.index):
        raise ValueError("blocked: event clock or Qlib fill bar differs")
    recovery_clock = day + pd.Timedelta(hours=10, minutes=32)
    early = fill < recovery_clock
    late = bar_end > recovery_clock
    if not early and not late:
        raise ValueError("blocked: 10:32 recovery/event ordering is ambiguous")
    if late != (recovery_state is not None):
        raise ValueError("blocked: late event needs verified post-recovery parent state")
    adjusted = np.asarray(event["adjusted_fill_prices"], float)
    raw = np.asarray(event["raw_fill_prices"], float)
    if (adjusted.shape != (2,) or raw.shape != (2,) or
            not np.isfinite([*adjusted, *raw]).all() or adjusted.min() <= 0 or raw.min() <= 0):
        raise ValueError("blocked: fill prices invalid")
    if not fixed_mode:
        desired = np.asarray(event["desired_weights"], float)
        if (desired.shape != (3,) or not np.isfinite(desired).all() or
                desired.min() < -1e-10 or abs(float(desired.sum()) - 1) > 1e-8):
            raise ValueError("blocked: long-only simplex invalid")
    fill_quote = first_quotes.loc[fill + pd.Timedelta(minutes=1)]
    source_adjusted = fill_quote[["SOXL_open", "SOXX_open"]].to_numpy(float)
    source_raw = fill_quote[["SOXL_raw_open", "SOXX_raw_open"]].to_numpy(float)
    if not np.allclose(adjusted, source_adjusted, rtol=0, atol=1e-8) or not np.allclose(raw, source_raw, rtol=0, atol=1e-8):
        raise ValueError("blocked: event fill differs from Qlib source")
    prefix = first_quotes.loc[(first_quotes.index >= day + pd.Timedelta(hours=9, minutes=48)) &
                              (first_quotes.index <= bar_end)]
    if prefix.empty or not prefix[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all().all():
        raise ValueError("blocked: completed predecision minute history absent")
    before = deepcopy(checkpoint.state)
    before_peak = float(before.peak)
    if not np.isfinite([*before.q, before.cash, before_peak]).all() or before.cash < -1e-8:
        raise ValueError("blocked: checkpoint account state invalid")
    post_1032_state_spliced = False
    for stamp, mark in prefix.iterrows():
        if late and not post_1032_state_spliced and stamp > recovery_clock:
            if not isinstance(recovery_state, dict) or set(recovery_state) != {"quantities", "cash", "orders"}:
                raise ValueError("blocked: post-recovery source contract differs")
            q = np.asarray(recovery_state["quantities"], float)
            cash = float(recovery_state["cash"])
            if q.shape != (2,) or not np.isfinite([*q, cash]).all() or q.min() < -1e-8 or cash < -1e-8:
                raise ValueError("blocked: post-recovery position invalid")
            for order in recovery_state["orders"]:
                if pd.Timestamp(order["stamp"]) != recovery_clock or order["slot"] != "recovery":
                    raise ValueError("blocked: recovery order clock differs")
                before.fees.commit(int(order["raw_shares"]), stamp=order["stamp"],
                                   cap_applied=bool(order["cap_applied"]))
            before.q, before.cash = q.copy(), cash
            post_1032_state_spliced = True
        price = mark[["SOXL_close", "SOXX_close"]].to_numpy(float)
        nav = float(before.q @ price + before.cash)
        if nav <= 0 or not np.isfinite(nav):
            raise ValueError("blocked: predecision NAV invalid")
        before.peak = max(float(before.peak), nav)
    if late and not post_1032_state_spliced:
        raise ValueError("blocked: no post-recovery minute before late decision")
    decision_price = first_quotes.loc[bar_end, ["SOXL_close", "SOXX_close"]].to_numpy(float)
    decision_nav = float(before.q @ decision_price + before.cash)
    if decision_nav <= 0:
        raise ValueError("blocked: event decision NAV invalid")
    if fixed_mode:
        intent = event["fixed_buy_intent"]
        symbol = intent.get("symbol") if isinstance(intent, dict) else None
        field = symbol + "_raw_close" if symbol in ("SOXL", "SOXX") else ""
        if field not in first_quotes:
            raise ValueError("blocked: decision raw close source absent")
        raw_close = float(first_quotes.loc[bar_end, field])
        expected = fixed_buy_intent(symbol=symbol, decision_nav=decision_nav,
            raw_close=raw_close, cash=before.cash, fees=before.fees,
            decision_at=decision,
            nominal_beta_increment=float(intent["nominal_beta_increment"]))
        if (intent["status"] != expected["status"] or
                intent["raw_shares"] != expected["raw_shares"] or
                abs(float(intent["decision_nav"]) - decision_nav) > 1e-7 or
                abs(float(intent["decision_raw_close"]) - raw_close) > 1e-8):
            raise ValueError("blocked: fixed buy intent differs from decision-known source")
    prefill_drawdown = max(0.0, 1.0 - decision_nav / float(before.peak))
    prefill_marks = first_quotes.loc[(first_quotes.index > bar_end) &
                                     (first_quotes.index <= fill)]
    if len(prefill_marks) != 2 or not prefill_marks[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all().all():
        raise ValueError("blocked: two valid prefill minute marks required")
    for _, mark in prefill_marks.iterrows():
        price = mark[["SOXL_close", "SOXX_close"]].to_numpy(float)
        nav = float(before.q @ price + before.cash)
        if nav <= 0 or not np.isfinite(nav):
            raise ValueError("blocked: prefill NAV invalid")
        before.peak = max(float(before.peak), nav)
        prefill_drawdown = max(prefill_drawdown, 1.0 - nav / float(before.peak))
    results = {}
    for arm in ("HOLD", "EVENT"):
        state = deepcopy(before)
        orders = []
        status = "hold"
        if arm == "EVENT":
            if fixed_mode:
                settled = settle_fixed_buy(state.q, state.cash, intent=intent,
                    adjusted_fill_prices=adjusted, raw_fill_prices=raw,
                    fill_at=fill, fees=state.fees)
                status = settled["status"]
            else:
                settled = settle_tiered_event(state.q, state.cash, desired=desired,
                    adjusted_prices=adjusted, raw_prices=raw, stamp=fill, fees=state.fees)
                status = "funded" if settled["orders"] else "no_whole_share_fill"
            state.q, state.cash = settled["quantities"], float(settled["cash"])
            orders = settled["orders"]
        if state.q.min() < -1e-8 or state.cash < -1e-8:
            raise ValueError("blocked: event branch borrowed or shorted")
        monitored = prefill_drawdown
        fill_nav = float(state.q @ adjusted + state.cash)
        if fill_nav <= 0 or not np.isfinite(fill_nav):
            raise ValueError("blocked: event fill NAV invalid")
        state.peak = max(float(state.peak), fill_nav)
        monitored = max(monitored, 1.0 - fill_nav / float(state.peak))
        rows = []
        for offset in range(5):
            row = frame.iloc[checkpoint.index + offset]
            session = pd.Timestamp(row.trade_date)
            bars = quotes.loc[quotes.index.normalize() == session]
            if offset == 0:
                bars = bars.loc[bars.index >= fill + pd.Timedelta(minutes=1)]
            if bars.empty or not bars[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all().all():
                raise ValueError("blocked: invalid or incomplete Qlib lock-session marks")
            prices = bars[["SOXL_close", "SOXX_close"]].to_numpy(float)

            def mark_price(price):
                nonlocal monitored
                nav = float(state.q @ price + state.cash)
                if nav <= 0 or not np.isfinite(nav):
                    raise ValueError("blocked: nonpositive lock-session NAV")
                state.peak = max(float(state.peak), nav)
                monitored = max(monitored, 1.0 - nav / float(state.peak))

            if offset == 0:
                for price in prices:
                    mark_price(price)
            else:
                # The prior row's next-open mark occurs at this session's
                # 09:47, after its earlier minute bars and before 09:48.
                boundary = session + pd.Timedelta(hours=9, minutes=47)
                early = bars.index <= boundary
                if not early.any() or early.all():
                    raise ValueError("blocked: next-open minute boundary absent")
                for price in prices[early]:
                    mark_price(price)
                previous = frame.iloc[checkpoint.index + offset - 1]
                previous_next_open = previous[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
                mark_price(previous_next_open)
                for price in prices[~early]:
                    mark_price(price)
            close_price = row[["close_soxl", "close_soxx"]].to_numpy(float)
            close_nav = float(state.q @ close_price + state.cash)
            if abs(close_nav - float(state.q @ prices[-1] + state.cash)) > 1e-6:
                raise ValueError("blocked: terminal Qlib minute/account close price differs")
            exit_price = row[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
            next_open_nav = float(state.q @ exit_price + state.cash)
            rows.append({"trade_date": session, "next_open_nav": next_open_nav,
                         "close_nav": close_nav, "ending_cash": float(state.cash),
                         "soxl_quantity": float(state.q[0]), "soxx_quantity": float(state.q[1]),
                         "available_mark_max_drawdown": float(monitored),
                         "event_fee_usd": float(sum(o["total"] for o in orders)) if offset == 0 else 0.0})
        # The fifth next-open is the terminal mark and follows the last close.
        mark_price(exit_price)
        results[arm] = {"path": pd.DataFrame(rows), "orders": orders,
                        "status": status,
                        "ending_quantities": state.q.copy(), "ending_cash": float(state.cash),
                        "ending_peak": float(state.peak),
                        "terminal_available_mark_max_drawdown": float(monitored),
                        "monthly_fee_shares": deepcopy(state.fees.monthly_shares)}
    results.update({"decision_nav": decision_nav, "prefill_quantities": before.q.copy(),
                    "prefill_cash": float(before.cash), "prefill_peak": float(before.peak),
                    "post_1032_parent_state_spliced": post_1032_state_spliced,
                    "event_status": results["EVENT"]["status"],
                    "session_close": session_close})
    return results

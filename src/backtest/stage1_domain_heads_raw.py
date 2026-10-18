"""Stage1 decisions on one cost scenario's physical-share cash account.

This is the frozen research clock (09:45 known, 09:47 fill, 10:32 recovery),
not a reproduction of the live release clock.  The source target stream is
supplied by the caller; the original opening HOLD decision and recovery model
are evaluated against this account's own state.  Native Qlib validation remains
the authority for the returned prescribed orders and cash-account evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees
from src.backtest.raw_corporate_policy import IncomeLedger, SYMBOLS, settle_raw
from src.workflow.stage1_open_action_value import (
    prepare_on_policy_open,
    prepare_on_policy_recovery,
)


def _validate_grid(frame, point_prices):
    if frame.empty:
        raise ValueError("blocked: empty Stage1 source target frame")
    dates = pd.DatetimeIndex(frame.trade_date)
    exits = pd.DatetimeIndex(frame.exit_date)
    signals = pd.DatetimeIndex(frame.signal_date)
    if (dates.has_duplicates or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize())
            or not exits.equals(exits.normalize())
            or not (signals < dates).all()
            or not (dates < exits).all()
            or not exits[:-1].equals(dates[1:])):
        raise ValueError("blocked: Stage1 source clocks are not contiguous ordered intervals")
    marks = point_prices.loc[:, list(SYMBOLS)]
    if (marks.index.has_duplicates or not marks.index.is_monotonic_increasing
            or not np.isfinite(marks.to_numpy(float)).all()
            or not marks.gt(0).all().all()):
        raise ValueError("blocked: invalid raw Stage1 point prices")
    expected = []
    for day in dates:
        stamps = marks.index[marks.index.normalize() == day]
        required = pd.DatetimeIndex([day + pd.Timedelta(clock)
                                     for clock in ("9h45min", "9h47min", "10h32min")])
        if len(stamps) != 4 or not stamps[:3].equals(required) or stamps[-1] <= required[-1]:
            raise ValueError("blocked: Stage1 requires known/open/recovery/actual-close raw points")
        expected.extend(stamps)
    expected.extend(exits[-1] + pd.Timedelta(clock) for clock in ("9h45min", "9h47min"))
    if not marks.index.equals(pd.DatetimeIndex(expected)):
        raise ValueError("blocked: Stage1 terminal or point-grid differs; exclude native calendar guard")
    return dates, exits[-1], marks


def replay_stage1_raw(
    frame, *, point_prices, schedule, plan, controller, recovery_spec,
    action_spec, selection, slippage_bps,
):
    """Return one raw path accepted by ``native_raw_validate``.

    Invoke separately for each arm and cost.  No state or fee counter is shared
    between invocations, and no composition transform or cross-cost vote is
    applied.  ``frame`` contains the arm's target weights plus the frozen
    Stage1 risk and first-hour context; future fill prices are not used by the
    opening decision.  Receivables count in NAV/cash-like weight but cannot fund
    purchases before the supplied corporate-action schedule pays them.
    """
    dates, terminal, prices = _validate_grid(frame, point_prices)
    if not callable(getattr(controller.get("frozen_open_action_value_model"), "predict_proba", None)):
        raise ValueError("blocked: the frozen Stage1 opening action model is required")
    if "frozen_recovery_model" not in controller:
        raise ValueError("blocked: frozen Stage1 recovery model identity is missing")
    initial_cash = float(plan["initial_cash"])
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("blocked: invalid Stage1 initial cash")
    if "slippage_bps" in plan and slippage_bps not in plan["slippage_bps"]:
        raise ValueError("blocked: cost scenario is outside the frozen plan")
    fees = TieredFees(plan["tariff"], slippage_bps=slippage_bps)
    income = IncomeLedger()
    q, cash, peak = np.zeros(2), initial_cash, initial_cash
    state = clock = 0
    previous_known = pending = None
    history, records, orders, points, targets, trace = [], [], [], [], [], []
    endpoints = {}
    rows = {pd.Timestamp(row.trade_date): row for _, row in frame.iterrows()}

    def mark(stamp, quote):
        nav = float(q @ quote + cash + income.value)
        points.append({"stamp": stamp, "q_soxl": q[0], "q_soxx": q[1],
                       "cash": cash, "receivable": income.value, "nav": nav})
        return nav

    for day in [*dates, terminal]:
        q, cash, _ = income.apply(day, q, cash, schedule)
        known_stamp = day + pd.Timedelta("9h45min")
        open_stamp = day + pd.Timedelta("9h47min")
        known = prices.loc[known_stamp].to_numpy(float)
        opening = prices.loc[open_stamp].to_numpy(float)
        known_nav = mark(known_stamp, known)
        peak = max(peak, known_nav)
        pre_open_nav = float(q @ opening + cash + income.value)
        endpoints[open_stamp] = pre_open_nav
        if pending is not None:
            ending = np.r_[q * opening, cash + income.value] / pre_open_nav
            pending.update(
                strategy_return=pre_open_nav / pending.pop("start_nav") - 1,
                equity=pre_open_nav / initial_cash, native_expected_nav=pre_open_nav,
                ending_soxl_quantity=q[0], ending_soxx_quantity=q[1],
                ending_cash=cash + income.value, ending_spendable_cash=cash,
                ending_receivable=income.value,
                **dict(zip(("ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"), ending)),
            )
            records.append(pending)
            pending = None
        if day == terminal:
            mark(open_stamp, opening)
            break

        if previous_known is not None:
            history.append(known_nav / previous_known - 1)
        previous_known = known_nav
        row = rows[day]
        pretrade = np.r_[q * known, cash + income.value] / known_nav
        opened = prepare_on_policy_open(
            row, pretrade_weights=pretrade, equity=known_nav, peak=peak,
            risk_state=state, risk_clock=clock, history=history,
            recovery_spec=recovery_spec,
            model=controller["frozen_open_action_value_model"], action_spec=action_spec,
        )
        state, clock = opened["risk_state"], opened["risk_clock"]
        desired = opened["desired_open_weights"] if opened["execute_open_target"] else None
        target = np.asarray(desired if desired is not None else pretrade, float)
        targets.append({
            "signal_date": row.signal_date, "trade_date": day,
            "decision_at": day + pd.Timedelta("9h46min"),
            "known_equity": known_nav, "known_drawdown": opened["drawdown"],
            "execute_open_target": bool(opened["execute_open_target"]),
            **dict(zip(("soxl_weight", "soxx_weight", "cash_weight"), target)),
        })
        q, cash, opened_orders = settle_raw(
            q, cash, income.value, desired=desired, prices=opening, stamp=open_stamp, fees=fees,
        )
        session_orders = [{**order, "slot": "open", "trade_date": day} for order in opened_orders]
        mark(open_stamp, opening)

        recovered = prepare_on_policy_recovery(
            row, open_state=opened, recovery_spec=recovery_spec,
            model=controller["frozen_recovery_model"], selected_model=selection,
        )
        recovery_stamp = day + pd.Timedelta("10h32min")
        recovery_prices = prices.loc[recovery_stamp].to_numpy(float)
        if recovered["recovery_action"]:
            q, cash, recovery_orders = settle_raw(
                q, cash, income.value, desired=recovered["target_weights"],
                prices=recovery_prices, stamp=recovery_stamp, fees=fees,
            )
            session_orders.extend({**order, "slot": "recovery", "trade_date": day}
                                  for order in recovery_orders)
        mark(recovery_stamp, recovery_prices)
        day_prices = prices.loc[prices.index.normalize() == day]
        close_stamp = day_prices.index[-1]
        close_nav = mark(close_stamp, day_prices.iloc[-1].to_numpy(float))
        peak = max(peak, close_nav)
        total = sum(order["total"] for order in session_orders)
        slippage = sum(order["slippage"] for order in session_orders)
        orders.extend(session_orders)
        pending = {
            "signal_date": row.signal_date, "trade_date": day, "exit_date": row.exit_date,
            "start_nav": pre_open_nav,
            "turnover": sum(order["notional"] for order in session_orders) / pre_open_nav,
            "fees": total / pre_open_nav,
            "broker_and_external_fees_usd": total - slippage,
            "slippage_reserve_usd": slippage,
            "portfolio_drawdown": opened["drawdown"],
            "monitored_drawdown": max(opened["drawdown"], max(0., 1 - close_nav / peak)),
            "risk_state": state, "risk_clock": clock,
            "current_beta_cap": opened["current_beta_cap"],
            "execute_open_target": bool(opened["execute_open_target"]),
            "recovery_action": bool(recovered["recovery_action"]),
        }
        trace.append({
            "trade_date": day, "slippage_bps": slippage_bps,
            "known_nav": known_nav, "risk_state": state, "risk_clock": clock,
            "current_beta_cap": opened["current_beta_cap"],
            "execute_open_target": bool(opened["execute_open_target"]),
            "recovery_probability": recovered["recovery_probability"],
            "recovery_action": bool(recovered["recovery_action"]),
            "spendable_cash": cash, "receivable": income.value,
        })

    return {
        "replay": pd.DataFrame(records), "orders": pd.DataFrame(orders),
        "points": pd.DataFrame(points), "targets": pd.DataFrame(targets),
        "corporate_receipts": pd.DataFrame(income.receipts),
        "monthly_fee_shares": fees.monthly_shares, "endpoints": endpoints,
        "trace": pd.DataFrame(trace),
    }

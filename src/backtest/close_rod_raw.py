"""New close-only permission with unchanged Stage1 morning decisions.

This is the frozen research clock (09:45 known, 09:47 fill, 10:32 recovery),
not a reproduction of the live release clock.  The source target stream is
supplied by the caller; the original opening HOLD decision and recovery model
are evaluated against this account's own state.  Native Qlib validation remains
the authority for the returned prescribed orders and cash-account evidence.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees
from src.workflow.intraday_recovery_direct_allocation import project_legal_weights
from src.backtest.raw_corporate_policy import IncomeLedger, SYMBOLS, settle_raw
from src.workflow.stage1_open_action_value import (
    prepare_on_policy_open,
    prepare_on_policy_recovery,
)


def _validate_grid(frame, point_prices, late_inputs):
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
    if not late_inputs.index.equals(dates):
        raise ValueError("late input dates must exactly match account dates")
    flags = late_inputs[["regular_session", "source_ready", "rod_known", "rod_positive"]]
    if not flags.isin([False, True]).all().all() or (flags.rod_positive & ~flags.rod_known).any():
        raise ValueError("late flags must be explicit known binary values")
    expected = []
    for day in dates:
        stamps = marks.index[marks.index.normalize() == day]
        required = pd.DatetimeIndex([day + pd.Timedelta(clock)
                                     for clock in ("9h45min", "9h47min", "10h32min")])
        close = pd.Timestamp(late_inputs.loc[day, "actual_close_at"])
        extra = [day + pd.Timedelta(minutes=minutes) for minutes in (930, 932, 959)] if late_inputs.loc[day, "regular_session"] else []
        full = pd.DatetimeIndex([*required, *extra, close])
        if not stamps.equals(full):
            raise ValueError("close-permission price grid differs from frozen session clocks")
        expected.extend(stamps)
    expected.extend(exits[-1] + pd.Timedelta(clock) for clock in ("9h45min", "9h47min"))
    if not marks.index.equals(pd.DatetimeIndex(expected)):
        raise ValueError("blocked: Stage1 terminal or point-grid differs; exclude native calendar guard")
    return dates, exits[-1], marks



def _envelope(q, cash, receivable, prices, reserve, *, cap, raw_risk, spec):
    nav = float(q @ prices + cash + receivable)
    if not np.isfinite(nav) or nav <= 0:
        return False
    weights = np.r_[q * prices, cash + receivable] / nav
    return bool(cash >= reserve - 1e-8 and cash >= -1e-8 and q.min() >= 0
        and 3*weights[0] + weights[1] <= cap + 1e-12
        and weights[0] <= spec.maximum_soxl_weight + 1e-12
        and weights[2] >= spec.minimum_cash_weight - 1e-12
        and weights[:2].sum() <= raw_risk + 1e-12)


def prepare_close_permission(row, *, opened, q, cash, receivable, known_prices,
                             peak, history, recovery_spec, selection, source, require_ROD):
    nav = float(q @ known_prices + cash + receivable)
    drawdown = max(0., 1-nav/peak)
    raw_beta = 3*float(row.soxl_weight) + float(row.soxx_weight)
    raw_risk = float(row.soxl_weight) + float(row.soxx_weight)
    cap = min(raw_beta, float(selection["action_beta"]))
    emergency = max(opened["drawdown"], drawdown) >= recovery_spec.emergency_drawdown
    if emergency:
        cap = min(cap, recovery_spec.emergency_beta)
    trailing = len(history) >= recovery_spec.trailing_loss_days and np.prod(
        1+np.asarray(history[-recovery_spec.trailing_loss_days:]))-1 <= -recovery_spec.trailing_loss_trigger
    if trailing:
        cap = min(cap, recovery_spec.trailing_loss_beta)
    conditions = [(bool(source.source_ready), "source_unavailable"),
        (np.isfinite(row.alpha_probability) and row.alpha_probability >= selection["minimum_source_alpha_probability"], "source_alpha"),
        (drawdown >= recovery_spec.recovery_eligibility_drawdown, "late_drawdown"),
        (cap > opened["current_beta_cap"] + 1e-12, "no_new_permission_room"),
        (not require_ROD or (bool(source.rod_known) and bool(source.rod_positive)), "ROD_gate")]
    reason = next((reason for passed, reason in conditions if not passed), "eligible")
    requested = np.zeros(2, dtype=np.int64)
    if reason == "eligible":
        target = project_legal_weights(float(row.soxl_weight), float(row.soxx_weight), cap,
            minimum_cash_weight=recovery_spec.minimum_cash_weight, maximum_soxl_weight=recovery_spec.maximum_soxl_weight)
        delta = target[:2] * nav / known_prices - q
        sales = np.floor(np.maximum(-delta, 0.) + 1e-10).astype(np.int64)
        if sales.any():
            reason = "baseline_sale_required"
        else:
            requested = np.floor(np.maximum(delta, 0.) + 1e-10).astype(np.int64)
            if not requested.any():
                reason = "no_whole_share_increment"
    return {"requested_buy": requested, "reason": reason, "late_cap": cap,
            "raw_risk": raw_risk, "raw_beta": raw_beta, "late_drawdown": drawdown,
            "source_ready": bool(source.source_ready), "ROD_gate_required": bool(require_ROD),
            "ROD_known": bool(source.rod_known), "ROD_positive": bool(source.rod_positive),
            "new_emergency_clamp": bool(emergency), "trailing_cap_active": bool(trailing)}


def _unwind_reserve(fees, lot, prices, stamp):
    copied, reserve = deepcopy(fees), 0.
    for i in range(2):
        quote = copied.quote(shares=int(lot[i]), raw_price=prices[i], sell=True, stamp=stamp)
        reserve += quote["total"]
        copied.commit(int(lot[i]), stamp=stamp, cap_applied=quote["cap_applied"])
    return reserve


def fill_increment(q, cash, receivable, *, requested, prices, reserve_prices, fees,
                   stamp, exit_stamp, cap, raw_risk, spec):
    """Only truncate known integer buys; preserve baseline inventory and full envelope."""
    q, lot, orders = np.asarray(q, float).copy(), np.zeros(2, dtype=np.int64), []
    requested = np.asarray(requested)
    if requested.shape != (2,) or (requested < 0).any() or not np.equal(requested, np.floor(requested)).all():
        raise ValueError("invalid precommitted buy quantity")
    for i, symbol in enumerate(SYMBOLS):
        for amount in range(int(requested[i]), 0, -1):
            quote = fees.quote(shares=amount, raw_price=prices[i], sell=False, stamp=stamp)
            candidate_q, candidate_lot = q.copy(), lot.copy()
            candidate_q[i] += amount; candidate_lot[i] += amount
            candidate_cash = cash - quote["notional"] - quote["total"]
            preview = deepcopy(fees)
            preview.commit(amount, stamp=stamp, cap_applied=quote["cap_applied"])
            reserve = _unwind_reserve(preview, candidate_lot, reserve_prices, exit_stamp)
            if not _envelope(candidate_q, candidate_cash, receivable, prices, reserve,
                             cap=cap, raw_risk=raw_risk, spec=spec):
                continue
            q, cash, lot = candidate_q, float(candidate_cash), candidate_lot
            fees.commit(amount, stamp=stamp, cap_applied=quote["cap_applied"])
            orders.append({"symbol": symbol, "sell": False, "raw_shares": amount,
                "adjusted_amount": amount, "raw_price": float(prices[i]), "adjusted_price": float(prices[i]),
                "stamp": pd.Timestamp(stamp), **quote})
            break
    if (lot > requested).any():
        raise ValueError("arrival prices increased the known-time request")
    return q, cash, orders, lot


def close_increment(q, cash, *, lot, prices, fees, stamp):
    """Sell exact executed incremental shares; never restore target weights or wealth."""
    q, orders = np.asarray(q, float).copy(), []
    for i, symbol in enumerate(SYMBOLS):
        amount = int(lot[i])
        if amount < 0 or amount > q[i]:
            raise ValueError("incremental inventory missing at scheduled exit")
        if not amount:
            continue
        quote = fees.quote(shares=amount, raw_price=prices[i], sell=True, stamp=stamp)
        q[i] -= amount; cash += quote["notional"] - quote["total"]
        fees.commit(amount, stamp=stamp, cap_applied=quote["cap_applied"])
        orders.append({"symbol": symbol, "sell": True, "raw_shares": amount,
            "adjusted_amount": amount, "raw_price": float(prices[i]), "adjusted_price": float(prices[i]),
            "stamp": pd.Timestamp(stamp), **quote})
    if q.min() < 0 or cash < -1e-7:
        raise ValueError("temporary exit is unfunded")
    return q, float(cash), orders

def replay_close_permission_raw(
    frame, *, point_prices, schedule, plan, controller, recovery_spec,
    action_spec, selection, slippage_bps, late_inputs, require_ROD,
):
    """Return one raw path accepted by ``native_raw_validate``.

    Invoke separately for each arm and cost.  No state or fee counter is shared
    between invocations, and no composition transform or cross-cost vote is
    applied.  ``frame`` contains the arm's target weights plus the frozen
    Stage1 risk and first-hour context; future fill prices are not used by the
    opening decision.  Receivables count in NAV/cash-like weight but cannot fund
    purchases before the supplied corporate-action schedule pays them.
    """
    dates, terminal, prices = _validate_grid(frame, point_prices, late_inputs)
    if float(selection["action_beta"]) != .6 or float(selection["minimum_source_alpha_probability"]) != .25:
        raise ValueError("frozen late ceiling/alpha parameters changed")
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
    endpoints, late_events = {}, []
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
        late_record = {"trade_date": day, "slippage_bps": slippage_bps, "late_action": False,
                       "lot_closed": True, "entry_shares_soxl": 0, "entry_shares_soxx": 0,
                       "roundtrip_pnl_usd": 0., "reason": "current_half_day"}
        if late_inputs.loc[day, "regular_session"]:
            late_known_stamp = day + pd.Timedelta(minutes=930)
            entry_stamp, exit_stamp = day + pd.Timedelta(minutes=932), day + pd.Timedelta(minutes=959)
            late_known = prices.loc[late_known_stamp].to_numpy(float)
            mark(late_known_stamp, late_known)  # valuation only: never advance/reset peak or risk clock
            before_q, before_cash, before_receivable = q.copy(), float(cash), income.value
            request = prepare_close_permission(row, opened=opened, q=q, cash=cash, receivable=income.value,
                known_prices=late_known, peak=peak, history=history, recovery_spec=recovery_spec,
                selection=selection, source=late_inputs.loc[day], require_ROD=require_ROD)
            desired = request["requested_buy"]
            # Fund and constrain the known-time request before any arrival price is used.
            _, _, known_orders, known_lot = fill_increment(q, cash, income.value,
                requested=desired, prices=late_known, reserve_prices=late_known,
                fees=deepcopy(fees), stamp=entry_stamp, exit_stamp=exit_stamp,
                cap=request["late_cap"], raw_risk=request["raw_risk"], spec=recovery_spec)
            entry_prices = prices.loc[entry_stamp].to_numpy(float)
            q, cash, added_orders, lot = fill_increment(q, cash, income.value,
                requested=known_lot, prices=entry_prices, reserve_prices=late_known,
                fees=fees, stamp=entry_stamp, exit_stamp=exit_stamp,
                cap=request["late_cap"], raw_risk=request["raw_risk"], spec=recovery_spec)
            session_orders.extend({**order, "slot": "late_entry", "trade_date": day} for order in added_orders)
            mark(entry_stamp, entry_prices)
            exit_prices = prices.loc[exit_stamp].to_numpy(float)
            q, cash, exit_orders = close_increment(q, cash, lot=lot, prices=exit_prices, fees=fees, stamp=exit_stamp)
            session_orders.extend({**order, "slot": "late_exit", "trade_date": day} for order in exit_orders)
            mark(exit_stamp, exit_prices)
            np.testing.assert_array_equal(q, before_q)
            if income.value != before_receivable:
                raise ValueError("temporary lot changed receivables")
            expected_cash = before_cash - sum(o["notional"] + o["total"] for o in added_orders)
            expected_cash += sum(o["notional"] - o["total"] for o in exit_orders)
            if not np.isclose(cash, expected_cash, rtol=1e-12, atol=1e-8):
                raise ValueError("temporary lot cash conservation failed")
            late_record = {"trade_date": day, "slippage_bps": slippage_bps,
                **{key: value for key, value in request.items() if key != "requested_buy"},
                "requested_soxl": int(desired[0]), "requested_soxx": int(desired[1]),
                "known_funded_soxl": int(known_lot[0]), "known_funded_soxx": int(known_lot[1]),
                "entry_shares_soxl": int(lot[0]), "entry_shares_soxx": int(lot[1]),
                "baseline_soxl": int(before_q[0]), "baseline_soxx": int(before_q[1]),
                "after_exit_soxl": int(q[0]), "after_exit_soxx": int(q[1]),
                "cash_before": before_cash, "cash_after": cash, "receivable": before_receivable,
                "late_action": bool(lot.sum()), "lot_closed": True,
                "roundtrip_pnl_usd": cash - before_cash}
        late_events.append(late_record)
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
            "late_action": late_record["late_action"],
            "late_roundtrip_pnl_usd": late_record["roundtrip_pnl_usd"],
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
        "trace": pd.DataFrame(trace), "late_events": pd.DataFrame(late_events),
    }

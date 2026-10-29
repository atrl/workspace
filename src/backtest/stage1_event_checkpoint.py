"""Checkpointed 20-session Stage1 event replay on the dated development source.

The parent prefix is replayed once to capture the exact after-opening state.
Both HOLD and a single event then continue with independent cash, holdings,
monthly Tiered counters, risk state, and dated action/recovery models.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event
from src.workflow.stage1_open_action_value import prepare_on_policy_open, prepare_on_policy_recovery


@dataclass
class _State:
    q: np.ndarray
    cash: float
    peak: float
    risk_state: int
    risk_clock: int
    history: list[float]
    previous_known: float | None
    fees: TieredFees


@dataclass
class EpisodeCheckpoint:
    index: int
    state: _State
    opening_nav: float
    opened: dict
    opening_orders: list[dict]
    target: dict


def _open(row, state: _State, *, controller, recovery_spec, action_spec):
    known = row[["known_soxl", "known_soxx"]].to_numpy(float)
    executable = row[["soxl_open", "soxx_open"]].to_numpy(float)
    known_nav = float(state.q @ known + state.cash)
    state.peak = max(state.peak, known_nav)
    if state.previous_known is not None:
        state.history.append(known_nav / state.previous_known - 1.)
    state.previous_known = known_nav
    pretrade = np.r_[state.q * known, state.cash] / known_nav
    opening_nav = float(state.q @ executable + state.cash)
    opened = prepare_on_policy_open(row, pretrade_weights=pretrade,
        equity=known_nav, peak=state.peak, risk_state=state.risk_state,
        risk_clock=state.risk_clock, history=state.history,
        recovery_spec=recovery_spec,
        model=controller["frozen_open_action_value_model"], action_spec=action_spec)
    state.risk_state, state.risk_clock = opened["risk_state"], opened["risk_clock"]
    desired = opened["desired_open_weights"] if opened["execute_open_target"] else None
    target = {"signal_date": row.signal_date, "trade_date": row.trade_date,
        "decision_at": row.trade_date + pd.Timedelta(hours=9, minutes=46),
        "known_equity": known_nav, "known_drawdown": opened["drawdown"],
        "soxl_weight": desired[0] if desired is not None else pretrade[0],
        "soxx_weight": desired[1] if desired is not None else pretrade[1],
        "cash_weight": desired[2] if desired is not None else pretrade[2]}
    traded = settle_tiered_event(state.q, state.cash, desired=desired,
        adjusted_prices=executable,
        raw_prices=row[["raw_soxl_open", "raw_soxx_open"]],
        stamp=row.trade_date + pd.Timedelta(hours=9, minutes=47), fees=state.fees)
    state.q, state.cash = traded["quantities"], traded["cash"]
    orders = [{**order, "slot": "open", "trade_date": row.trade_date}
              for order in traded["orders"]]
    return EpisodeCheckpoint(-1, state, opening_nav, opened, orders, target)


def _apply_event(row, state: _State, event: dict, session_orders: list[dict]):
    traded = settle_tiered_event(state.q, state.cash,
        desired=np.asarray(event["desired_weights"], float),
        adjusted_prices=np.asarray(event["adjusted_fill_prices"], float),
        raw_prices=np.asarray(event["raw_fill_prices"], float),
        stamp=pd.Timestamp(event["fill_at"]), fees=state.fees)
    state.q, state.cash = traded["quantities"], traded["cash"]
    session_orders.extend({**order, "slot": "event", "trade_date": row.trade_date}
                          for order in traded["orders"])


def _finish(row, checkpoint: EpisodeCheckpoint, *, controller, recovery_spec,
            selection, initial_cash: float, event: dict | None):
    state = checkpoint.state
    opened = checkpoint.opened
    session_orders = list(checkpoint.opening_orders)
    recovery_stamp = row.trade_date + pd.Timedelta(hours=10, minutes=32)
    if event is not None and pd.Timestamp(event["fill_at"]) < recovery_stamp:
        _apply_event(row, state, event, session_orders)
    recovered = prepare_on_policy_recovery(row, open_state=opened,
        recovery_spec=recovery_spec,
        model=controller["frozen_recovery_model"], selected_model=selection)
    if recovered["recovery_action"]:
        traded = settle_tiered_event(state.q, state.cash,
            desired=recovered["target_weights"],
            adjusted_prices=row[["soxl_recovery", "soxx_recovery"]],
            raw_prices=row[["raw_soxl_recovery", "raw_soxx_recovery"]],
            stamp=recovery_stamp, fees=state.fees)
        state.q, state.cash = traded["quantities"], traded["cash"]
        session_orders.extend({**order, "slot": "recovery", "trade_date": row.trade_date}
                              for order in traded["orders"])
    if event is not None and pd.Timestamp(event["fill_at"]) >= recovery_stamp:
        _apply_event(row, state, event, session_orders)
    closing = row[["close_soxl", "close_soxx"]].to_numpy(float)
    exiting = row[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
    close_nav = float(state.q @ closing + state.cash)
    state.peak = max(state.peak, close_nav)
    monitored = max(opened["drawdown"], max(0., 1. - close_nav / state.peak))
    ending_nav = float(state.q @ exiting + state.cash)
    ending = np.r_[state.q * exiting, state.cash] / ending_nav
    cost = sum(order["total"] for order in session_orders)
    slipped = sum(order["slippage"] for order in session_orders)
    record = {"signal_date": row.signal_date, "trade_date": row.trade_date,
        "exit_date": row.exit_date,
        "strategy_return": ending_nav / checkpoint.opening_nav - 1.,
        "equity": ending_nav / initial_cash, "native_expected_nav": ending_nav,
        "turnover": sum(order["notional"] for order in session_orders) / checkpoint.opening_nav,
        "fees": cost / checkpoint.opening_nav,
        "broker_and_external_fees_usd": cost - slipped,
        "slippage_reserve_usd": slipped,
        "portfolio_drawdown": opened["drawdown"],
        "monitored_drawdown": monitored,
        "risk_state": state.risk_state, "risk_clock": state.risk_clock,
        "current_beta_cap": opened["current_beta_cap"],
        "execute_open_target": bool(opened["execute_open_target"]),
        "recovery_action": bool(recovered["recovery_action"]),
        "ending_soxl_quantity": state.q[0], "ending_soxx_quantity": state.q[1],
        "ending_cash": state.cash,
        "ending_soxl_weight": ending[0], "ending_soxx_weight": ending[1],
        "ending_cash_weight": ending[2]}
    return record, session_orders


def capture_after_open(frame: pd.DataFrame, *, event_day, controller_by_date,
                       recovery_spec, action_spec, selection, plan,
                       slippage_bps=0) -> EpisodeCheckpoint:
    """Replay the parent once to the requested day's completed opening trade."""
    days = pd.DatetimeIndex(frame.trade_date)
    event_day = pd.Timestamp(event_day)
    matches = np.flatnonzero(days == event_day)
    if len(matches) != 1:
        raise ValueError("blocked: checkpoint day absent or duplicated")
    event_index = int(matches[0])
    initial_cash = float(plan["initial_cash"])
    state = _State(np.zeros(2), initial_cash, initial_cash, 0, 0, [], None,
                   TieredFees(plan["tariff"], slippage_bps=slippage_bps))
    for index in range(event_index + 1):
        row = frame.iloc[index]
        active = controller_by_date(pd.Timestamp(row.trade_date))
        opened = _open(row, state, controller=active,
                       recovery_spec=recovery_spec, action_spec=action_spec)
        opened.index = int(index)
        if index == event_index:
            return deepcopy(opened)
        _finish(row, opened, controller=active, recovery_spec=recovery_spec,
                selection=selection, initial_cash=initial_cash, event=None)
    raise AssertionError("unreachable checkpoint capture")


def resume_episode(frame: pd.DataFrame, checkpoint: EpisodeCheckpoint, *,
                   controller_by_date, recovery_spec, action_spec, selection,
                   plan, event: dict | None, horizon_sessions: int = 20):
    """Run one current event and following sessions from an independent state."""
    if not 1 <= horizon_sessions <= 20 or checkpoint.index + horizon_sessions > len(frame):
        raise ValueError("blocked: episode horizon is absent")
    if event is not None:
        required = {"trade_date", "bar_end_at", "decision_at", "fill_at", "desired_weights",
                    "adjusted_fill_prices", "raw_fill_prices"}
        if not isinstance(event, dict) or set(event) != required:
            raise ValueError("blocked: event branch requires frozen clock/target/fill contract")
        day = pd.Timestamp(frame.iloc[checkpoint.index].trade_date)
        bar_end = pd.Timestamp(event["bar_end_at"])
        desired = np.asarray(event["desired_weights"], float)
        adjusted = np.asarray(event["adjusted_fill_prices"], float)
        raw = np.asarray(event["raw_fill_prices"], float)
        if (pd.Timestamp(event["trade_date"]) != day or
                bar_end.normalize() != day or
                pd.Timestamp(event["decision_at"]) != bar_end + pd.Timedelta(minutes=1) or
                pd.Timestamp(event["fill_at"]) != bar_end + pd.Timedelta(minutes=2) or
                pd.Timestamp(event["fill_at"]) <= day + pd.Timedelta(hours=9, minutes=47) or
                pd.Timestamp(event["fill_at"]) >= day + pd.Timedelta(hours=16)):
            raise ValueError("blocked: checkpoint event clock differs")
        if (desired.shape != (3,) or adjusted.shape != (2,) or raw.shape != (2,) or
                not np.isfinite(desired).all() or desired.min() < -1e-10 or
                abs(desired.sum() - 1.) > 1e-8 or
                not np.isfinite(adjusted).all() or not np.isfinite(raw).all() or
                min(adjusted.min(), raw.min()) <= 0):
            raise ValueError("blocked: checkpoint event target/fill invalid")
    current = deepcopy(checkpoint)
    rows, orders, targets = [], [], []
    initial_cash = float(plan["initial_cash"])
    for index in range(checkpoint.index, checkpoint.index + horizon_sessions):
        row = frame.iloc[index]
        active = controller_by_date(pd.Timestamp(row.trade_date))
        opened = current if index == checkpoint.index else _open(row, current.state,
            controller=active, recovery_spec=recovery_spec, action_spec=action_spec)
        record, submitted = _finish(row, opened, controller=active,
            recovery_spec=recovery_spec, selection=selection,
            initial_cash=initial_cash,
            event=event if index == checkpoint.index else None)
        rows.append(record)
        orders.extend(submitted)
        targets.append(opened.target)
    return {"replay": pd.DataFrame(rows), "orders": pd.DataFrame(orders),
            "targets": pd.DataFrame(targets),
            "monthly_fee_shares": deepcopy(current.state.fees.monthly_shares),
            "ending_quantities": current.state.q.copy(),
            "ending_cash": float(current.state.cash),
            "ending_peak": float(current.state.peak),
            "ending_risk_state": int(current.state.risk_state),
            "ending_risk_clock": int(current.state.risk_clock)}

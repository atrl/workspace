"""Causal OOF teacher shadow and fixed-reset local action labels for P2.

The adapter reuses the frozen nonlearned opening risk projection and the
project's physical raw-share, income-ledger, and Tiered-fee primitives. It has
no provider reads, model fitting, or account replay beyond one prescribed
training reference path.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees
from src.backtest.raw_corporate_policy import IncomeLedger, SYMBOLS, settle_raw
from src.workflow.stage1_open_action_value import (
    ACTION_FEATURES,
    Stage1OpenActionValueSpec,
    prepare_on_policy_open,
)
from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec,
    mark_to_market_weights,
)


def _session_date(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    if stamp != stamp.normalize():
        raise ValueError("blocked: trade dates must be timezone-naive session dates")
    return stamp


def _at(day: pd.Timestamp, clock: str) -> pd.Timestamp:
    return day + pd.Timedelta(clock)


def _price(point_prices: pd.DataFrame, day: pd.Timestamp, clock: str) -> np.ndarray:
    stamp = _at(day, clock)
    if stamp not in point_prices.index:
        raise ValueError(f"needs_data: required physical mark is missing at {stamp}")
    values = point_prices.loc[stamp, list(SYMBOLS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"needs_data: required physical mark is invalid at {stamp}")
    return values


def clocked_smooth_targets(
    prediction: pd.DataFrame,
    timing: pd.DataFrame,
    *,
    spec,
    candidate,
) -> pd.DataFrame:
    """Apply the original EMA/rebalance rule with explicit H1 feedback clocks.

    A row's H1 return updates virtual holdings only before the following row's
    target. Thus its own H1 return cannot influence its 09:46 target.
    """
    weights = ["soxl_weight", "soxx_weight", "cash_weight"]
    returns = ["soxl_forward_open_return", "soxx_forward_open_return"]
    required = set(weights + ["tail_probability", "alpha_probability"])
    if not required.issubset(prediction.columns):
        raise ValueError("blocked: original five-output OOF prediction schema differs")
    if not set(returns + ["trade_date", "return_available_at"]).issubset(timing.columns):
        raise ValueError("blocked: original H1 return timing schema differs")
    pred = prediction.copy()
    clock = timing.copy()
    if not pred.index.equals(clock.index) or pred.index.has_duplicates:
        raise ValueError("blocked: OOF prediction/timing indices differ")
    trade_dates = pd.DatetimeIndex([_session_date(value) for value in clock.trade_date])
    order = np.argsort(trade_dates.to_numpy(), kind="stable")
    if not np.array_equal(order, np.arange(len(order))) or trade_dates.has_duplicates:
        raise ValueError("blocked: OOF trade-date sequence is not strictly chronological")
    smooth = pred.loc[:, weights].ewm(span=(candidate.smoothing_span or spec.smoothing_span),
                                      adjust=False).mean()
    threshold = candidate.rebalance_threshold
    if threshold is None:
        threshold = spec.rebalance_threshold
    accepted = np.empty((len(pred), 3), dtype=np.float64)
    current = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    decisions = []
    for i, (signal_date, values) in enumerate(smooth.iterrows()):
        target = values.to_numpy(dtype=np.float64)
        if i:
            previous = clock.iloc[i - 1]
            available = pd.Timestamp(previous.return_available_at)
            decision_at = trade_dates[i].tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=46)
            if available.tzinfo is None or available > decision_at:
                raise ValueError("blocked: previous H1 return is not mature by the current 09:46 target")
            prior_return = previous.loc[returns].to_numpy(dtype=np.float64)
            if not np.isfinite(prior_return).all() or (prior_return <= -1).any():
                raise ValueError("blocked: clocked virtual H1 return is invalid")
            current = mark_to_market_weights(accepted[i - 1], np.r_[prior_return, 0.0])
        turnover = float(np.abs(target[:2] - current[:2]).sum())
        executed = target if turnover >= float(threshold) else current
        accepted[i] = executed
        if not np.isfinite(executed).all() or (executed < 0).any() or not np.isclose(executed.sum(), 1.0):
            raise ValueError("blocked: original smoother produced an invalid target")
        decisions.append({"signal_date": pd.Timestamp(signal_date), "trade_date": trade_dates[i],
                          "decision_at": trade_dates[i].tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=46),
                          "smoothed_turnover": turnover,
                          **dict(zip(weights, executed)),
                          "tail_probability": float(pred.iloc[i].tail_probability),
                          "alpha_probability": float(pred.iloc[i].alpha_probability),
                          "return_available_at": pd.Timestamp(clock.iloc[i].return_available_at)})
    result = pd.DataFrame(decisions).set_index("trade_date").sort_index()
    return result


def _copy_state(q, cash, income: IncomeLedger, fees: TieredFees) -> dict[str, Any]:
    return {"q": np.asarray(q, dtype=np.float64).copy(), "cash": float(cash),
            "income": deepcopy(income), "fees": deepcopy(fees)}


def run_training_shadow(
    teacher: pd.DataFrame,
    *,
    point_prices: pd.DataFrame,
    schedule: dict,
    sessions: pd.DatetimeIndex,
    recovery_spec: IntradayRecoveryDirectAllocationSpec,
    action_spec: Stage1OpenActionValueSpec,
    tariff: dict,
    initial_cash: float = 100_000.0,
    slippage_bps: float = 2.0,
    start_trade_date: str = "2018-10-12",
) -> dict[str, Any]:
    """Run the sole continuous physical teacher path at the frozen 2bp cost."""
    if not isinstance(teacher, pd.DataFrame) or "trade_date" not in teacher:
        raise ValueError("blocked: teacher frame must include actual trade_date")
    required = {"signal_date", "exit_date", "soxl_weight", "soxx_weight", "cash_weight",
                "tail_probability", "alpha_probability"}
    if not required.issubset(teacher.columns):
        raise ValueError("blocked: common OOF teacher frame is incomplete")
    data = teacher.copy()
    data["trade_date"] = data.trade_date.map(_session_date)
    data["exit_date"] = data.exit_date.map(_session_date)
    data = data.loc[data.trade_date >= pd.Timestamp(start_trade_date)].sort_values("trade_date")
    if data.empty or data.trade_date.duplicated().any():
        raise ValueError("needs_data: prescribed teacher trade grid is empty or duplicated")
    last_trade = data.trade_date.iloc[-1]
    terminal = data.exit_date.iloc[-1]
    session_dates = pd.DatetimeIndex(sessions)
    if session_dates.tz is not None:
        session_dates = session_dates.tz_localize(None)
    session_dates = session_dates.normalize()
    expected = session_dates[(session_dates >= data.trade_date.iloc[0]) & (session_dates <= terminal)]
    actual = pd.DatetimeIndex([*data.trade_date.tolist(), terminal])
    if not expected.equals(actual):
        raise ValueError("needs_data: teacher/shadow grid has a missing or nonadjacent session")
    if terminal > pd.Timestamp("2022-12-30"):
        raise ValueError("blocked: P2 teacher terminal extends beyond the frozen training period")
    point_prices = point_prices.copy().sort_index()
    if point_prices.index.has_duplicates or point_prices.index.tz is not None:
        raise ValueError("blocked: raw physical point grid must be unique and timezone-naive")
    if not {"SOXL", "SOXX"}.issubset(point_prices.columns):
        raise ValueError("blocked: physical point prices lack SOXL/SOXX")

    q = np.zeros(2, dtype=np.float64)
    cash = float(initial_cash)
    income = IncomeLedger()
    fees = TieredFees(tariff, slippage_bps=slippage_bps)
    peak = float(initial_cash)
    previous_known = None
    history: list[float] = []
    state = clock = 0
    rows_by_day = {row.trade_date: row for row in data.itertuples(index=False)}
    all_days = [*data.trade_date.tolist(), terminal]
    for day in all_days:
        day_grid = point_prices.index[point_prices.index.normalize() == day]
        required = pd.DatetimeIndex([_at(day, "9h45min"), _at(day, "9h47min"), _at(day, "10h32min")])
        if (len(day_grid) != 4 or not required.isin(day_grid).all()
                or day_grid[-1].hour not in (13, 16) or day_grid[-1].minute != 0
                or day_grid[-1].date() != day.date()):
            raise ValueError("needs_data: sparse native point grid must contain exactly three decision marks and the scheduled close")
    decisions, branch_contexts, targets, orders, points, receipts, trace = [], {}, [], [], [], [], []
    endpoints: dict[pd.Timestamp, float] = {}
    pending: dict[str, Any] | None = None
    replay = []

    def record_point(stamp: pd.Timestamp, prices: np.ndarray) -> float:
        value = float(q @ prices + cash + income.value)
        points.append({"stamp": stamp, "q_soxl": q[0], "q_soxx": q[1], "cash": cash,
                       "receivable": income.value, "nav": value})
        return value

    for day in all_days:
        q, cash, applied = income.apply(day, q, cash, schedule)
        receipts.extend(applied)
        known = _price(point_prices, day, "9h45min")
        nav = record_point(_at(day, "9h45min"), known)
        peak = max(peak, nav)
        if previous_known is not None:
            history.append(nav / previous_known - 1.0)
        previous_known = nav
        opening = _price(point_prices, day, "9h47min")
        prebranch_nav = float(q @ opening + cash + income.value)
        endpoints[_at(day, "9h47min")] = prebranch_nav
        if pending is not None:
            prior = pending
            if day != prior["exit_date"]:
                raise ValueError("needs_data: teacher exit date differs from next shadow session")
            ending = prebranch_nav
            replay.append({**prior, "native_expected_nav": ending,
                           "strategy_return": ending / prior["start_nav"] - 1.0,
                           "equity": ending / initial_cash})
            pending = None

        row = rows_by_day.get(day)
        if row is not None:
            pretrade = np.r_[q * known, cash + income.value] / nav
            opened = prepare_on_policy_open(
                pd.Series(row._asdict()), pretrade_weights=pretrade, equity=nav, peak=peak,
                risk_state=state, risk_clock=clock, history=history,
                recovery_spec=recovery_spec, model=None, action_spec=action_spec,
            )
            state, clock = int(opened["risk_state"]), int(opened["risk_clock"])
            if not opened["execute_open_target"]:
                raise ValueError("blocked: model-free training teacher did not execute its target")
            if not np.array_equal(opened["open_weights"], opened["desired_open_weights"]):
                raise ValueError("blocked: teacher opening target differs from risk-projected target")
            action_features = {name: float(opened["features"][name]) for name in ACTION_FEATURES}
            finite_features = bool(np.isfinite(list(action_features.values())).all())
            guards = {
                "mandatory_derisk": bool(opened["mandatory_derisk"]),
                "large_transition": bool(opened["large_transition"]),
                "mandatory_position_repair": bool(opened["mandatory_position_repair"]),
                "nonfinite_original22": not finite_features,
            }
            eligible = not any(guards.values())
            target = np.asarray(opened["desired_open_weights"], dtype=np.float64).copy()
            if not np.isfinite(target).all() or target.min() < 0 or not np.isclose(target.sum(), 1.0):
                raise ValueError("blocked: deterministic teacher target is not a simplex")
            snapshot = _copy_state(q, cash, income, fees)
            common_prebranch_nav = prebranch_nav
            q_before, cash_before = q.copy(), float(cash)
            q, cash, fills = settle_raw(q, cash, income.value, desired=target,
                prices=opening, stamp=_at(day, "9h47min"), fees=fees)
            orders.extend({**fill, "slot": "open", "trade_date": day} for fill in fills)
            branch_contexts[day] = {"state": snapshot, "target": target.copy(),
                "opening_prices": opening.copy(), "prebranch_nav": common_prebranch_nav,
                "eligible": eligible, "guards": guards, "action_features": action_features,
                "signal_date": pd.Timestamp(row.signal_date), "exit_date": row.exit_date,
                "decision_at": day.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=46),
                "prefix_mask": int(getattr(row, "prefix_mask", 0)),
                "path_area_signed": float(getattr(row, "path_area_signed", np.nan)),
                **{name: float(getattr(row, name, np.nan)) for name in _path_control_names()},
            }
            target_record = {"trade_date": day, "signal_date": pd.Timestamp(row.signal_date),
                "decision_at": branch_contexts[day]["decision_at"],
                "known_nav": nav, "prebranch_nav_0947": prebranch_nav,
                "drawdown": opened["drawdown"], "risk_state": state, "risk_clock": clock,
                "current_beta_cap": opened["current_beta_cap"],
                "mandatory_derisk": guards["mandatory_derisk"],
                "large_transition": guards["large_transition"],
                "mandatory_position_repair": guards["mandatory_position_repair"],
                "eligible": eligible, "eligibility_reasons": [key for key, value in guards.items() if value],
                "prefix_mask": branch_contexts[day]["prefix_mask"],
                "path_area_signed": branch_contexts[day]["path_area_signed"],
                **{name: branch_contexts[day][name] for name in _path_control_names()},
                "source_target_soxl_weight": float(row.soxl_weight),
                "source_target_soxx_weight": float(row.soxx_weight),
                "source_target_cash_weight": float(row.cash_weight),
                **{name: float(target[i]) for i, name in enumerate(("soxl_weight", "soxx_weight", "cash_weight"))},
                "projected_target_soxl_weight": float(target[0]),
                "projected_target_soxx_weight": float(target[1]),
                "projected_target_cash_weight": float(target[2]),
                "pretrade_soxl_weight": float(pretrade[0]),
                "pretrade_soxx_weight": float(pretrade[1]),
                "pretrade_cash_weight": float(pretrade[2]),
                "prebranch_nav_0947": prebranch_nav,
                "filled_order_count": len(fills),
                "filled_notional_usd": float(sum(fill["notional"] for fill in fills)),
                "total_order_fees_usd": float(sum(fill["total"] for fill in fills)),
                "slippage_reserve_usd": float(sum(fill["slippage"] for fill in fills)),
                "postfill_q_soxl": float(q[0]), "postfill_q_soxx": float(q[1]),
                "postfill_cash": float(cash), "postfill_receivable": income.value,
                "postfill_nav": float(q @ opening + cash + income.value),
                "postfill_soxl_weight": float(q[0] * opening[0] / (q @ opening + cash + income.value)),
                "postfill_soxx_weight": float(q[1] * opening[1] / (q @ opening + cash + income.value)),
                "postfill_cash_weight": float((cash + income.value) / (q @ opening + cash + income.value)),
                **action_features}
            decisions.append(target_record)
            targets.append({key: target_record[key] for key in ("trade_date", "signal_date", "decision_at",
                "known_nav", "current_beta_cap", "soxl_weight", "soxx_weight", "cash_weight")})
            trace.append({"trade_date": day, "signal_date": pd.Timestamp(row.signal_date),
                "decision_at": branch_contexts[day]["decision_at"],
                "known_nav_0945": nav, "prebranch_nav_0947": prebranch_nav,
                "drawdown": opened["drawdown"], "risk_state": state, "risk_clock": clock,
                "current_beta_cap": opened["current_beta_cap"],
                "source_target_soxl_weight": float(row.soxl_weight),
                "source_target_soxx_weight": float(row.soxx_weight),
                "source_target_cash_weight": float(row.cash_weight),
                "projected_target_soxl_weight": float(target[0]),
                "projected_target_soxx_weight": float(target[1]),
                "projected_target_cash_weight": float(target[2]),
                "pretrade_soxl_weight": float(pretrade[0]),
                "pretrade_soxx_weight": float(pretrade[1]),
                "pretrade_cash_weight": float(pretrade[2]),
                "mandatory_derisk": guards["mandatory_derisk"],
                "large_transition": guards["large_transition"],
                "mandatory_position_repair": guards["mandatory_position_repair"],
                "eligible": eligible, "eligibility_reasons": target_record["eligibility_reasons"],
                "prefix_mask": target_record["prefix_mask"],
                "path_area_signed": target_record["path_area_signed"],
                "filled_order_count": len(fills),
                "filled_notional_usd": target_record["filled_notional_usd"],
                "total_order_fees_usd": target_record["total_order_fees_usd"],
                "slippage_reserve_usd": target_record["slippage_reserve_usd"],
                "pretrade_q_soxl": float(q_before[0]), "pretrade_q_soxx": float(q_before[1]),
                "pretrade_cash_usd": cash_before, "pretrade_receivable_usd": income.value,
                "postfill_q_soxl": float(q[0]), "postfill_q_soxx": float(q[1]),
                "postfill_cash_usd": float(cash), "postfill_receivable_usd": income.value,
                "postfill_nav_0947": target_record["postfill_nav"],
                "postfill_soxl_weight": target_record["postfill_soxl_weight"],
                "postfill_soxx_weight": target_record["postfill_soxx_weight"],
                "postfill_cash_weight": target_record["postfill_cash_weight"]})
            pending = {"signal_date": pd.Timestamp(row.signal_date), "trade_date": day,
                       "exit_date": row.exit_date, "start_nav": prebranch_nav}

        # The native provider contains the same sparse raw marks for every
        # shadow session, including the no-order terminal valuation date.
        record_point(_at(day, "9h47min"), opening)
        recovery = _price(point_prices, day, "10h32min")
        record_point(_at(day, "10h32min"), recovery)
        close_matches = point_prices.index[(point_prices.index.normalize() == day)
                                           & (point_prices.index.hour >= 13)]
        if not len(close_matches):
            raise ValueError("needs_data: scheduled session-close point is missing")
        close_at = close_matches[-1]
        close_prices = point_prices.loc[close_at, list(SYMBOLS)].to_numpy(dtype=np.float64)
        close_nav = record_point(close_at, close_prices)
        peak = max(peak, close_nav)
    if pending is not None:
        raise ValueError("needs_data: final teacher target has no native exit valuation")
    decision_frame = pd.DataFrame(decisions).set_index("trade_date").sort_index()
    if len(decision_frame) != len(data):
        raise ValueError("blocked: shadow did not retain every prescribed teacher session")
    return {"replay": pd.DataFrame(replay), "orders": pd.DataFrame(orders),
            "points": pd.DataFrame(points), "targets": pd.DataFrame(targets),
            "decisions": decision_frame, "trace": pd.DataFrame(trace).set_index("trade_date"),
            "corporate_receipts": pd.DataFrame(receipts),
            "monthly_fee_shares": dict(fees.monthly_shares), "endpoints": endpoints,
            "_branch_contexts": branch_contexts,
            "audit": {"status": "python_shadow_complete_pending_native", "sessions": len(data),
                      "points": len(points), "orders": len(orders), "slippage_bps": float(slippage_bps),
                      "initial_cash_usd": float(initial_cash), "continuous_physical_account": True,
                      "historical_PIT_qualified": False}}


def _path_control_names() -> tuple[str, ...]:
    return ("path_control_X", "path_control_Y", "path_control_X_squared", "path_control_XY",
            "path_control_Y_squared", "path_control_Qx", "path_control_Qxy", "path_control_Qy")


def _branch_after_reset(
    initial: dict[str, Any],
    *,
    execute: bool,
    current_target: np.ndarray,
    current_day: pd.Timestamp,
    next_day: pd.Timestamp,
    next_target: np.ndarray,
    next_prices: np.ndarray,
    schedule: dict,
    sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    state = _copy_state(initial["q"], initial["cash"], initial["income"], initial["fees"])
    entry_orders = []
    if execute:
        state["q"], state["cash"], entry_orders = settle_raw(
            state["q"], state["cash"], state["income"].value, desired=current_target,
            prices=initial["opening_prices"], stamp=_at(current_day, "9h47min"), fees=state["fees"])
    between = sessions[(sessions > current_day) & (sessions <= next_day)]
    if not len(between) or between[-1] != next_day:
        raise ValueError("needs_data: local branch interval is not on the qualified session grid")
    receipts = []
    for day in between:
        state["q"], state["cash"], applied = state["income"].apply(day, state["q"], state["cash"], schedule)
        receipts.extend(applied)
    reset_orders = []
    state["q"], state["cash"], reset_orders = settle_raw(
        state["q"], state["cash"], state["income"].value, desired=next_target,
        prices=next_prices, stamp=_at(next_day, "9h47min"), fees=state["fees"])
    nav = float(state["q"] @ next_prices + state["cash"] + state["income"].value)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("blocked: branch reset NAV is invalid")
    return {"nav_after_reset": nav, "q_after_reset": state["q"].copy(),
            "cash_after_reset": state["cash"], "receivable_after_reset": state["income"].value,
            "entry_orders": entry_orders, "reset_orders": reset_orders,
            "corporate_receipts": receipts, "fees": state["fees"].monthly_shares.copy()}


def label_local_actions(
    shadow: dict[str, Any],
    *,
    point_prices: pd.DataFrame,
    schedule: dict,
    sessions: pd.DatetimeIndex,
    cutoff: pd.Timestamp = pd.Timestamp("2023-01-01 00:00", tz="America/New_York"),
) -> pd.DataFrame:
    """Create one EXECUTE/HOLD fixed-reset pair for each eligible mature row."""
    decisions = shadow["decisions"]
    contexts = shadow["_branch_contexts"]
    session_dates = pd.DatetimeIndex(sessions)
    if session_dates.tz is not None:
        session_dates = session_dates.tz_localize(None)
    session_dates = session_dates.normalize()
    records = []
    dates = list(decisions.index)
    next_by_day = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    for current_day, row in decisions.iterrows():
        context = contexts[current_day]
        record = {key: row[key] for key in ("signal_date", "decision_at", "eligible",
                    "eligibility_reasons", "prefix_mask", "path_area_signed", *_path_control_names())}
        record["trade_date"] = current_day
        record.update(context["action_features"])
        record.update(label_available_at=pd.NaT, label_status="unknown", execute_delta=np.nan,
                      execute_nav_after_reset=np.nan, hold_nav_after_reset=np.nan,
                      common_prebranch_nav=float(context["prebranch_nav"]),
                      next_common_teacher_target_present=False)
        next_day = next_by_day.get(current_day)
        if next_day is None:
            record["label_reason"] = "missing_next_common_teacher_target"
            records.append(record)
            continue
        next_context = contexts[next_day]
        next_target = np.asarray(next_context["target"], dtype=np.float64)
        next_decision_at = next_context["decision_at"]
        record["next_common_teacher_target_present"] = True
        record.update(common_reset_target_soxl=float(next_target[0]),
            common_reset_target_soxx=float(next_target[1]),
            common_reset_target_cash=float(next_target[2]))
        # A right-edge 09:48 open (the 09:47 fill) is complete at 09:48 and
        # receives the pinned +60s research availability lag.
        bar_mature_at = next_day.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=49)
        maturity = max(pd.Timestamp(next_decision_at), bar_mature_at)
        record["label_available_at"] = maturity
        if not context["eligible"]:
            record["label_reason"] = "original_open_guard_or_nonfinite_action_features"
            records.append(record)
            continue
        if maturity >= cutoff:
            record["label_reason"] = "matures_at_or_after_frozen_training_cutoff"
            records.append(record)
            continue
        current_target = np.asarray(context["target"], dtype=np.float64)
        next_prices = _price(point_prices, next_day, "9h47min")
        state = {**context["state"], "opening_prices": context["opening_prices"].copy()}
        execute = _branch_after_reset(state, execute=True, current_target=current_target,
            current_day=current_day, next_day=next_day, next_target=next_target,
            next_prices=next_prices, schedule=schedule, sessions=session_dates)
        hold = _branch_after_reset(state, execute=False, current_target=current_target,
            current_day=current_day, next_day=next_day, next_target=next_target,
            next_prices=next_prices, schedule=schedule, sessions=session_dates)
        delta = (execute["nav_after_reset"] - hold["nav_after_reset"]) / float(context["prebranch_nav"])
        if not np.isfinite(delta):
            raise ValueError("blocked: local fixed-reset label is nonfinite")
        record.update(label_status="mature", label_reason="eligible_same_state_same_reset",
            execute_delta=float(delta), execute_nav_after_reset=execute["nav_after_reset"],
            hold_nav_after_reset=hold["nav_after_reset"], execute_orders=execute["entry_orders"],
            execute_reset_orders=execute["reset_orders"], hold_orders=hold["reset_orders"],
            execute_receipts=execute["corporate_receipts"], hold_receipts=hold["corporate_receipts"],
            execute_fee_month_shares=execute["fees"], hold_fee_month_shares=hold["fees"])
        records.append(record)
    result = pd.DataFrame(records).set_index("trade_date").sort_index()
    return result

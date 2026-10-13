"""One-clock recovery value, with paired next-open reset labels and own NAV."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event
from src.workflow.intraday_recovery_direct_allocation import project_legal_weights
from src.workflow.stage1_open_action_value import prepare_on_policy_open, prepare_on_policy_recovery


def recovery_value_context(row, quantities, cash, opened, *, fees, recovery_spec, selection, partial_fraction):
    known = row[["soxl_cut", "soxx_cut"]].to_numpy(float)
    raw = row[["raw_soxl_cut", "raw_soxx_cut"]].to_numpy(float)
    nav = float(quantities @ known + cash)
    held = np.r_[quantities * known, cash] / nav
    target_beta = min(opened["raw_beta"], float(selection["action_beta"]))
    eligible = bool(opened["drawdown"] >= recovery_spec.recovery_eligibility_drawdown
        and target_beta > opened["current_beta_cap"] + 1e-12
        and row.alpha_probability >= selection["minimum_source_alpha_probability"]
        and row.get("minute_ready", False))
    full = project_legal_weights(float(row.soxl_weight), float(row.soxx_weight), target_beta,
        minimum_cash_weight=recovery_spec.minimum_cash_weight, maximum_soxl_weight=recovery_spec.maximum_soxl_weight)
    desired = (1 - partial_fraction) * held + partial_fraction * full
    desired = project_legal_weights(desired[0], desired[1], target_beta,
        minimum_cash_weight=recovery_spec.minimum_cash_weight, maximum_soxl_weight=recovery_spec.maximum_soxl_weight)
    preview_fees = deepcopy(fees)
    preview = settle_tiered_event(quantities, cash, desired=desired, adjusted_prices=known, raw_prices=raw,
        stamp=row.trade_date + pd.Timedelta(hours=10, minutes=31), fees=preview_fees)
    preview_cost = sum(order["total"] for order in preview["orders"]) / nav
    turnover = sum(order["notional"] for order in preview["orders"]) / nav
    delta = desired - held
    features = {"portfolio_drawdown": opened["drawdown"], "beta_cap": opened["current_beta_cap"],
        "raw_beta": opened["raw_beta"], "risk_state": float(opened["risk_state"]), "risk_clock": float(opened["risk_clock"]),
        "held_soxl": held[0], "held_soxx": held[1], "held_cash": held[2],
        "target_soxl": desired[0], "target_soxx": desired[1], "target_cash": desired[2],
        "beta_delta": float(3 * delta[0] + delta[1]), "preview_turnover": turnover,
        "preview_primary_entry_cost": preview_cost, "tail_probability": row.tail_probability,
        "alpha_probability": row.alpha_probability,
        "portfolio_ret_5d": float(np.prod(1 + np.asarray(opened["history"][-5:])) - 1) if opened["history"] else 0.}
    features.update({name: float(row[name]) for name in row.index if name.startswith("minute_") and name != "minute_ready"})
    return {"eligible": eligible, "desired": desired, "features": features, "known_nav": nav,
            "quantities": quantities.copy(), "cash": cash, "fee_state": deepcopy(fees), "preview_turnover": turnover}


def next_reset_advantage(context, row, reset_weights):
    """Hold/action branches share holdings, fees, future marks and reset target.

    All future quantities here are labels only. The realized holding horizon is
    action10:32 through next09:30; both branches then pay their reset costs.
    """
    outcomes = []
    for desired in (None, context["desired"]):
        fees = deepcopy(context["fee_state"])
        branch = settle_tiered_event(context["quantities"], context["cash"], desired=desired,
            adjusted_prices=row[["soxl_action", "soxx_action"]], raw_prices=row[["raw_soxl_action", "raw_soxx_action"]],
            stamp=row.trade_date + pd.Timedelta(hours=10, minutes=32), fees=fees)
        reset = settle_tiered_event(branch["quantities"], branch["cash"], desired=reset_weights,
            adjusted_prices=row[["soxl_exit_open", "soxx_exit_open"]], raw_prices=row[["raw_soxl_exit_open", "raw_soxx_exit_open"]],
            stamp=row.exit_date + pd.Timedelta(hours=9, minutes=30), fees=fees)
        outcomes.append(float(reset["quantities"] @ row[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float) + reset["cash"]))
    return (outcomes[1] - outcomes[0]) / context["known_nav"]


def replay_minute_value(frame, *, controller, tariff, slippage_bps, recovery_spec, action_spec,
                        selection, initial_cash, policy, value_models=None, plan=None):
    q, cash, peak = np.zeros(2), float(initial_cash), float(initial_cash)
    state = clock = 0
    history, rows, orders, contexts, predictions = [], [], [], [], []
    fees = TieredFees(tariff, slippage_bps=slippage_bps)
    for number, (_, row) in enumerate(frame.iterrows()):
        opening = row[["soxl_open", "soxx_open"]].to_numpy(float)
        action_price = row[["soxl_action", "soxx_action"]].to_numpy(float)
        exiting = row[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
        nav = float(q @ opening + cash)
        pretrade = np.r_[q * opening, cash] / nav
        if policy == "soxx":
            opened = {"drawdown": max(0., 1 - nav / peak), "current_beta_cap": 1., "execute_open_target": number == 0}
            desired = np.array([0., 1., 0.]) if number == 0 else None
        else:
            opened = prepare_on_policy_open(row, pretrade_weights=pretrade, equity=nav, peak=peak, risk_state=state,
                risk_clock=clock, history=history, recovery_spec=recovery_spec,
                model=controller["frozen_open_action_value_model"], action_spec=action_spec)
            state, clock = opened["risk_state"], opened["risk_clock"]
            desired = opened["desired_open_weights"] if opened["execute_open_target"] else None
        result = settle_tiered_event(q, cash, desired=desired, adjusted_prices=opening,
            raw_prices=row[["raw_soxl_open", "raw_soxx_open"]], stamp=row.trade_date + pd.Timedelta(hours=9, minutes=30), fees=fees)
        q, cash = result["quantities"], result["cash"]
        session_orders = [{**o, "slot": "open", "trade_date": row.trade_date} for o in result["orders"]]
        action, target, predicted, error = False, None, np.nan, np.nan
        if policy != "soxx":
            # Features and target are decided from the last AVAILABLE 10:30
            # close, never the future 10:32 execution price.
            primary_fees = deepcopy(fees)
            primary_fees.slippage_bps = plan["primary_slippage_bps"]
            context = recovery_value_context(row, q, cash, opened, fees=primary_fees, recovery_spec=recovery_spec,
                selection=selection, partial_fraction=plan["partial_fraction"])
            context["shared_reset_weights"] = opened["desired_open_weights"]
            context["trade_date"] = row.trade_date
            contexts.append(context)
            if policy == "current":
                current = prepare_on_policy_recovery(row, open_state=opened, recovery_spec=recovery_spec,
                    model=controller["frozen_recovery_model"], selected_model=selection)
                action, target = current["recovery_action"], current["target_weights"]
            elif context["eligible"] and row.trade_date.year in value_models:
                feature = pd.DataFrame([context["features"]])
                estimate = value_models[int(row.trade_date.year)].predict_frame(feature).iloc[0]
                # Preview approximation uses only known holdings/prices. Exact
                # realized entry/reset fees belong to labels/accounting.
                cost_delta = 2 * context["preview_turnover"] * (slippage_bps - plan["primary_slippage_bps"]) / 10_000
                predicted, error = float(estimate["mean"] - cost_delta), float(estimate["mean_error"])
                action = predicted - plan["confidence_z"] * error > 0
                target = context["desired"] if action else None
            predictions.append({"trade_date": row.trade_date, "eligible": context["eligible"],
                "predicted_net_advantage": predicted, "mean_error": error, "action": action})
        if action:
            result = settle_tiered_event(q, cash, desired=target, adjusted_prices=action_price,
                raw_prices=row[["raw_soxl_action", "raw_soxx_action"]], stamp=row.trade_date + pd.Timedelta(hours=10, minutes=32), fees=fees)
            q, cash = result["quantities"], result["cash"]
            session_orders += [{**o, "slot": "recovery", "trade_date": row.trade_date} for o in result["orders"]]
        ending_nav = float(q @ exiting + cash)
        value = ending_nav / nav - 1
        history.append(value)
        peak = max(peak, ending_nav)
        total = sum(o["total"] for o in session_orders)
        slip = sum(o["slippage"] for o in session_orders)
        weights = np.r_[q * exiting, cash] / ending_nav
        orders.extend(session_orders)
        rows.append({"signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date,
            "strategy_return": value, "equity": ending_nav / initial_cash, "native_expected_nav": ending_nav,
            "turnover": sum(o["notional"] for o in session_orders) / nav, "fees": total / nav,
            "broker_and_external_fees_usd": total - slip, "slippage_reserve_usd": slip,
            "portfolio_drawdown": opened["drawdown"], "current_beta_cap": opened["current_beta_cap"],
            "risk_state": state, "risk_clock": clock, "execute_open_target": opened["execute_open_target"],
            "recovery_action": action, "order_count": len(session_orders),
            "ending_soxl_weight": weights[0], "ending_soxx_weight": weights[1], "ending_cash_weight": weights[2],
            "ending_soxl_quantity": q[0], "ending_soxx_quantity": q[1], "ending_cash": cash})
    return {"replay": pd.DataFrame(rows), "orders": pd.DataFrame(orders), "contexts": contexts,
            "predictions": pd.DataFrame(predictions), "monthly_fee_shares": fees.monthly_shares}

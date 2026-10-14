"""Own-ledger inference at a known-price decision, followed by later fills."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event
from src.workflow.stage1_open_action_value import prepare_on_policy_open, prepare_on_policy_recovery


def replay_full_path(frame, *, models, plan, slippage_bps, controller=None, recovery_spec=None,
                     action_spec=None, selection=None, benchmark=False, state_transition=None, controller_by_date=None):
    if controller_by_date is not None and controller is not None:
        raise ValueError('provide static or dated controller, not both')
    q, cash, peak = np.zeros(2), float(plan["initial_cash"]), float(plan["initial_cash"])
    fees = TieredFees(plan["tariff"], slippage_bps=slippage_bps)
    records, orders, targets, history = [], [], [], []
    state = clock = 0
    previous_known = None
    for number, (_, row) in enumerate(frame.iterrows()):
        active_controller = controller_by_date(pd.Timestamp(row.trade_date)) if controller_by_date is not None else controller
        if controller_by_date is not None and not isinstance(active_controller, dict):
            raise ValueError('dated controller must return explicit controller models')
        known = row[["known_soxl", "known_soxx"]].to_numpy(float)
        executable = row[["soxl_open", "soxx_open"]].to_numpy(float)
        exiting = row[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
        closing = row[["close_soxl", "close_soxx"]].to_numpy(float)
        known_nav = float(q @ known + cash)
        peak = max(peak, known_nav)
        if previous_known is not None:
            history.append(known_nav / previous_known - 1)
        previous_known = known_nav
        drawdown = max(0., 1 - known_nav / peak)
        pretrade = np.r_[q * known, cash] / known_nav
        nav = float(q @ executable + cash)
        recovery, opened = False, None
        if benchmark:
            desired = np.array([0., 1., 0.]) if number == 0 else None
        elif active_controller is not None:
            opened = prepare_on_policy_open(row, pretrade_weights=pretrade, equity=known_nav, peak=peak,
                risk_state=state, risk_clock=clock, history=history, recovery_spec=recovery_spec,
                model=active_controller["frozen_open_action_value_model"], action_spec=action_spec, state_transition=state_transition)
            state, clock = opened["risk_state"], opened["risk_clock"]
            desired = opened["desired_open_weights"] if opened["execute_open_target"] else None
        else:
            requests = [model.target(row, q, cash, known, peak) for model in models]
            if not requests or (any(value is None for value in requests) and not all(value is None for value in requests)):
                raise ValueError("blocked: empty or mixed HOLD/target policy ensemble")
            desired = None if all(value is None for value in requests) else np.mean(requests, axis=0)
            if desired is not None and (not np.isfinite(desired).all() or desired[0] > plan["maximum_soxl_weight"] + 1e-10
                    or desired[2] < plan["minimum_cash_weight"] - 1e-10):
                raise ValueError("blocked: path policy target violates legal target constraints")
        targets.append({"signal_date": row.signal_date, "trade_date": row.trade_date, "decision_at": row.trade_date + pd.Timedelta(hours=9, minutes=46),
            "known_equity": known_nav, "known_drawdown": drawdown, "soxl_weight": desired[0] if desired is not None else pretrade[0],
            "soxx_weight": desired[1] if desired is not None else pretrade[1], "cash_weight": desired[2] if desired is not None else pretrade[2]})
        traded = settle_tiered_event(q, cash, desired=desired, adjusted_prices=executable,
            raw_prices=row[["raw_soxl_open", "raw_soxx_open"]], stamp=row.trade_date + pd.Timedelta(hours=9, minutes=47), fees=fees)
        q, cash = traded["quantities"], traded["cash"]
        session = [{**order, "slot": "open", "trade_date": row.trade_date} for order in traded["orders"]]
        if active_controller is not None:
            recovered = prepare_on_policy_recovery(row, open_state=opened, recovery_spec=recovery_spec,
                model=active_controller["frozen_recovery_model"], selected_model=selection)
            recovery = recovered["recovery_action"]
            if recovery:
                traded = settle_tiered_event(q, cash, desired=recovered["target_weights"],
                    adjusted_prices=row[["soxl_recovery", "soxx_recovery"]], raw_prices=row[["raw_soxl_recovery", "raw_soxx_recovery"]],
                    stamp=row.trade_date + pd.Timedelta(hours=10, minutes=32), fees=fees)
                q, cash = traded["quantities"], traded["cash"]
                session += [{**order, "slot": "recovery", "trade_date": row.trade_date} for order in traded["orders"]]
        close_nav = float(q @ closing + cash)
        peak = max(peak, close_nav)
        monitored = max(drawdown, max(0., 1 - close_nav / peak))
        ending_nav = float(q @ exiting + cash)
        ending = np.r_[q * exiting, cash] / ending_nav
        costs, slipped = sum(order["total"] for order in session), sum(order["slippage"] for order in session)
        orders.extend(session)
        records.append({"signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date,
            "strategy_return": ending_nav / nav - 1, "equity": ending_nav / plan["initial_cash"], "native_expected_nav": ending_nav,
            "turnover": sum(order["notional"] for order in session) / nav, "fees": costs / nav,
            "broker_and_external_fees_usd": costs - slipped, "slippage_reserve_usd": slipped,
            "portfolio_drawdown": drawdown, "monitored_drawdown": monitored,
            "risk_state": state, "risk_clock": clock, "current_beta_cap": opened["current_beta_cap"] if opened else np.nan,
            "execute_open_target": desired is not None, "recovery_action": recovery,
            "ending_soxl_quantity": q[0], "ending_soxx_quantity": q[1], "ending_cash": cash,
            "ending_soxl_weight": ending[0], "ending_soxx_weight": ending[1], "ending_cash_weight": ending[2]})
    return {"replay": pd.DataFrame(records), "orders": pd.DataFrame(orders), "targets": pd.DataFrame(targets),
            "monthly_fee_shares": fees.monthly_shares}

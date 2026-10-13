"""Own-cost Stage1 accounts with independent native Qlib fee reconciliation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import IBKRTieredExchange, TieredFees, settle_tiered_event
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec, prepare_on_policy_open, prepare_on_policy_recovery


SYMBOLS = ("SOXL", "SOXX")


def replay_tiered_stage1(frame, *, models, tariff, slippage_bps, recovery_spec,
                         action_spec, recovery_selection, initial_cash, benchmark=False):
    quantities, cash, peak = np.zeros(2), float(initial_cash), float(initial_cash)
    state = clock = 0
    history, records, orders = [], [], []
    fees = TieredFees(tariff, slippage_bps=slippage_bps)
    for number, (_, row) in enumerate(frame.iterrows()):
        opening = row[["soxl_open", "soxx_open"]].to_numpy(float)
        cut = row[["soxl_cut", "soxx_cut"]].to_numpy(float)
        exiting = row[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
        nav = float(quantities @ opening + cash)
        pretrade = np.r_[quantities * opening, cash] / nav
        if benchmark:
            desired = np.array([0., 1., 0.]) if number == 0 else None
            opened = {"drawdown": max(0., 1. - nav / peak), "current_beta_cap": 1., "execute_open_target": number == 0}
        else:
            opened = prepare_on_policy_open(row, pretrade_weights=pretrade, equity=nav, peak=peak,
                risk_state=state, risk_clock=clock, history=history, recovery_spec=recovery_spec,
                model=models["frozen_open_action_value_model"], action_spec=action_spec)
            state, clock = opened["risk_state"], opened["risk_clock"]
            desired = opened["desired_open_weights"] if opened["execute_open_target"] else None
        result = settle_tiered_event(quantities, cash, desired=desired, adjusted_prices=opening,
            raw_prices=row[["raw_soxl_open", "raw_soxx_open"]], stamp=row.trade_date + pd.Timedelta(hours=9, minutes=45), fees=fees)
        quantities, cash = result["quantities"], result["cash"]
        session_orders = [{**order, "slot": "open", "trade_date": row.trade_date} for order in result["orders"]]
        if benchmark:
            recovered = {"recovery_action": False, "target_weights": None}
        else:
            recovered = prepare_on_policy_recovery(row, open_state=opened, recovery_spec=recovery_spec,
                model=models["frozen_recovery_model"], selected_model=recovery_selection)
        if recovered["recovery_action"]:
            result = settle_tiered_event(quantities, cash, desired=recovered["target_weights"], adjusted_prices=cut,
                raw_prices=row[["raw_soxl_cut", "raw_soxx_cut"]], stamp=row.trade_date + pd.Timedelta(hours=10, minutes=30), fees=fees)
            quantities, cash = result["quantities"], result["cash"]
            session_orders += [{**order, "slot": "recovery", "trade_date": row.trade_date} for order in result["orders"]]
        ending_nav = float(quantities @ exiting + cash)
        value = ending_nav / nav - 1.
        history.append(value)
        peak = max(peak, ending_nav)
        costs = sum(order["total"] for order in session_orders)
        slipped = sum(order["slippage"] for order in session_orders)
        orders.extend(session_orders)
        ending = np.r_[quantities * exiting, cash] / ending_nav
        records.append({"signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date,
            "strategy_return": value, "equity": ending_nav / initial_cash,
            "turnover": sum(order["notional"] for order in session_orders) / nav,
            "fees": costs / nav, "broker_and_external_fees_usd": costs - slipped, "slippage_reserve_usd": slipped,
            "portfolio_drawdown": opened["drawdown"], "current_beta_cap": opened["current_beta_cap"],
            "risk_state": state, "risk_clock": clock, "execute_open_target": opened["execute_open_target"],
            "execute_open_trade": any(order["slot"] == "open" for order in session_orders),
            "recovery_action": recovered["recovery_action"], "order_count": len(session_orders),
            "ending_soxl_weight": ending[0], "ending_soxx_weight": ending[1], "ending_cash_weight": ending[2],
            "ending_soxl_quantity": quantities[0], "ending_soxx_quantity": quantities[1], "ending_cash": cash,
            "native_expected_nav": ending_nav})
    return {"replay": pd.DataFrame(records), "orders": pd.DataFrame(orders),
            "monthly_fee_shares": fees.monthly_shares}


def native_tiered_stage1(frame, replayed, *, provider_uri: Path, tariff: dict,
                         slippage_bps: float, initial_cash: float, tolerance=1e-7,
                         freq="15min", opening_offset=pd.Timedelta(hours=9, minutes=45)):
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order, TradeDecisionWO
    from qlib.data import D
    from src.backtest.strategy import DirectTargetWeightStrategy

    qlib.init(provider_uri=str(provider_uri), region="us", kernels=1)
    rows, orders = replayed["replay"], replayed["orders"]
    first, last = rows.trade_date.iloc[0], rows.trade_date.iloc[-1]
    calendar = pd.DatetimeIndex(D.calendar(start_time=first, end_time=last + pd.Timedelta(hours=16), freq=freq))
    day_last = pd.Series(calendar, index=calendar.normalize()).groupby(level=0).last()
    scheduled = dict(tuple(orders.groupby("stamp", sort=False))) if len(orders) else {}

    class PrescribedWholeShareOrders(DirectTargetWeightStrategy):
        def __init__(self):
            super().__init__(signal=pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])), risk_degree=1.)

        def generate_trade_decision(self, execute_result=None):
            start, end = self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step())
            entries = scheduled.get(pd.Timestamp(start))
            submitted = [] if entries is None else [Order(str(row.symbol), float(row.adjusted_amount),
                Order.SELL if row.sell else Order.BUY, start, end) for row in entries.itertuples()]
            return TradeDecisionWO(submitted, self)

    exchange = IBKRTieredExchange(tariff=tariff, slippage_bps=slippage_bps, freq=freq,
        start_time=first, end_time=last + pd.Timedelta(hours=16), codes=list(SYMBOLS),
        deal_price="execution_price", limit_threshold=None)
    outputs, indicators = backtest(start_time=first + opening_offset,
        end_time=last + pd.Timedelta(hours=16), strategy=PrescribedWholeShareOrders(),
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": freq, "generate_portfolio_metrics": True}},
        account=initial_cash, benchmark=pd.Series(0., index=calendar), exchange_kwargs={"exchange": exchange})
    report, positions = outputs[freq]
    checked, previous = [], float(initial_cash)
    for i, row in rows.iterrows():
        position = positions[pd.Timestamp(day_last.loc[row.trade_date])]
        quantities = np.array([position.get_stock_amount(symbol) if position.check_stock(symbol) else 0. for symbol in SYMBOLS])
        cash = float(position.get_cash())
        marks = frame.iloc[i][["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
        nav = float(quantities @ marks + cash)
        expected_quantities = row[["ending_soxl_quantity", "ending_soxx_quantity"]].to_numpy(float)
        weights = np.r_[quantities * marks, cash] / nav
        checked.append({"signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date,
            "native_nav": nav, "native_return": nav / previous - 1., "return_error": nav / previous - 1. - row.strategy_return,
            "relative_cash_error": abs(cash - row.ending_cash) / max(1., row.native_expected_nav),
            "relative_quantity_value_error": float(np.max(abs(quantities - expected_quantities) * marks)) / max(1., row.native_expected_nav),
            "maximum_weight_error": float(np.max(abs(weights - row[["ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"]].to_numpy(float))))})
        previous = nav
    checked = pd.DataFrame(checked)
    errors = checked[["return_error", "relative_cash_error", "relative_quantity_value_error", "maximum_weight_error"]].abs().max()
    native_costs = pd.DataFrame(exchange.executed_fees)
    if len(native_costs) != len(orders) or (len(orders) and not np.allclose(native_costs.total, orders.total, rtol=0, atol=1e-7)):
        raise ValueError("blocked: native per-order IBKR fees disagree with the independent ledger")
    if errors.max() > tolerance:
        raise ValueError("blocked: native tiered cash/quantities/NAV disagree with the independent ledger")
    authority = rows.copy()
    authority["strategy_return"] = checked.native_return
    authority["equity"] = checked.native_nav / initial_cash
    return {"authority": authority, "independent_replay": rows, "orders": orders, "native_orders": native_costs,
        "native_report": report, "native_positions": positions, "native_indicators": indicators, "reconciliation": checked,
        "audit": {"status": "passed", "sessions": len(rows), "orders": len(orders), "errors": errors.to_dict(),
            "provider_uri": str(provider_uri), "tariff_role": tariff["role"], "slippage_bps": slippage_bps,
            "maximum_monthly_fee_counted_raw_shares": max(exchange.fees.monthly_shares.values(), default=0.),
            "commission_uses_raw_whole_share_orders": True, "actual_broker_fill_or_corporate_action_parity": False,
            "native_api": ["qlib.backtest.backtest", "SimulatorExecutor", "Exchange", "Position"]}}

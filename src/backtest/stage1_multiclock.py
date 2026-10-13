"""Frozen interval values inside the current daily risk/position contract."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import IBKRTieredExchange, TieredFees, settle_tiered_event
from src.workflow.intraday_recovery_direct_allocation import project_legal_weights
from src.workflow.stage1_open_action_value import prepare_on_policy_open, prepare_on_policy_recovery


def economic_choice(q, cash, prices, raw, means, targets, *, fees, stamp, allow_hold=True):
    """Every preview starts from the same cash, quantities and monthly counter."""
    options = ([None] if allow_hold else []) + list(targets)
    best, best_fees, best_value = None, None, -np.inf
    for desired in options:
        quoted = deepcopy(fees)
        settled = settle_tiered_event(q, cash, desired=desired, adjusted_prices=prices, raw_prices=raw, stamp=stamp, fees=quoted)
        value = float(settled["quantities"] @ (prices * (1 + means)) + settled["cash"])
        if value > best_value + 1e-8:
            best, best_fees, best_value = settled, quoted, value
    return best, best_fees


def replay_multiclock(quotes, daily, forecasts, *, family, plan, controller, recovery_spec, action_spec, selection, slip):
    prices = quotes["$close"].unstack("instrument").loc[:, ["SOXL", "SOXX"]]
    raw = quotes["$raw_execution_price"].unstack("instrument").loc[:, ["SOXL", "SOXX"]]
    q, cash, previous_nav, peak = np.zeros(2), float(plan["initial_cash"]), float(plan["initial_cash"]), float(plan["initial_cash"])
    fees = TieredFees(plan["tariff"], slippage_bps=slip)
    state = clock = 0; history, records, orders = [], [], []
    previous_open, opened, source = None, None, None
    daily = daily.set_index("trade_date")
    scores = forecasts.set_index("entry_at") if forecasts is not None else None
    for number, stamp in enumerate(prices.index):
        px, rp = prices.loc[stamp].to_numpy(float), raw.loc[stamp].to_numpy(float)
        nav = float(q @ px + cash); held = np.r_[q * px, cash] / nav
        desired = None; settled = None; repaired = False; action = False
        if family == "soxx":
            desired = np.array([0., 1., 0.]) if number == 0 else None
        elif stamp.strftime("%H:%M") == "09:45":
            source = daily.loc[stamp.normalize()] if stamp.normalize() in daily.index else None
            opened = None
            if source is not None:
                if previous_open is not None: history.append(nav / previous_open - 1)
                previous_open = nav; peak = max(peak, nav)
                opened = prepare_on_policy_open(source, pretrade_weights=held, equity=nav, peak=peak,
                    risk_state=state, risk_clock=clock, history=history, recovery_spec=recovery_spec,
                    model=controller["frozen_open_action_value_model"], action_spec=action_spec)
                state, clock = opened["risk_state"], opened["risk_clock"]
                desired = opened["desired_open_weights"] if opened["execute_open_target"] else None
        elif opened is not None and stamp.strftime("%H:%M") in plan["action_clocks"]:
            if family == "current":
                if stamp.strftime("%H:%M") == "10:45":
                    recovered = prepare_on_policy_recovery(source, open_state=opened, recovery_spec=recovery_spec,
                        model=controller["frozen_recovery_model"], selected_model=selection)
                    desired = recovered["target_weights"] if recovered["recovery_action"] else None
                    action = recovered["recovery_action"]
            else:
                base_cap = opened["current_beta_cap"]
                recovery_cap = min(opened["raw_beta"], float(selection["action_beta"]))
                permitted = bool(opened["drawdown"] >= recovery_spec.recovery_eligibility_drawdown and recovery_cap > base_cap + 1e-12
                    and source.alpha_probability >= selection["minimum_source_alpha_probability"])
                cap = recovery_cap if permitted else base_cap
                project = lambda beta: project_legal_weights(float(source.soxl_weight), float(source.soxx_weight), beta,
                    minimum_cash_weight=recovery_spec.minimum_cash_weight, maximum_soxl_weight=recovery_spec.maximum_soxl_weight)
                base = project(base_cap); reduced = np.array([0., base[0] + base[1], base[2]])
                choices = [base, reduced] + ([project(recovery_cap)] if permitted else [])
                legal = bool(held[0] <= recovery_spec.maximum_soxl_weight + 1e-12 and held[2] >= recovery_spec.minimum_cash_weight - 1e-12
                    and 3 * held[0] + held[1] <= cap + 1e-12)
                if scores is not None and stamp in scores.index:
                    means = scores.loc[stamp, ["pred_soxl", "pred_soxx"]].to_numpy(float)
                    settled, fees = economic_choice(q, cash, px, rp, means, choices, fees=fees, stamp=stamp, allow_hold=legal)
                    action = bool(settled["orders"])
                    repaired = not legal
                elif not legal:
                    desired = base; repaired = True
        if settled is None:
            settled = settle_tiered_event(q, cash, desired=desired, adjusted_prices=px, raw_prices=rp, stamp=stamp, fees=fees)
        q, cash = settled["quantities"], settled["cash"]
        orders.extend(settled["orders"])
        ending = float(q @ px + cash)
        cost = sum(order["total"] for order in settled["orders"])
        slip_cost = sum(order["slippage"] for order in settled["orders"])
        records.append({"datetime": stamp, "nav": ending, "return": ending / previous_nav - 1,
            "cash": cash, "soxl_quantity": q[0], "soxx_quantity": q[1], "total_cost": cost,
            "broker_fees": cost - slip_cost, "slippage_cost": slip_cost, "trade_notional": sum(o["notional"] for o in settled["orders"]),
            "action": action, "mandatory_repair": repaired, "risk_state": state, "risk_clock": clock,
            "cash_weight": cash / ending, "soxl_weight": q[0] * px[0] / ending})
        previous_nav = ending
    return {"ledger": pd.DataFrame(records).set_index("datetime"), "orders": pd.DataFrame(orders), "monthly_shares": fees.monthly_shares}


def native_multiclock(replay, *, provider_uri: Path, plan, slip):
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order, TradeDecisionWO
    from src.backtest.strategy import DirectTargetWeightStrategy
    from src.backtest.qlib_continuous_position import QlibContinuousPosition
    qlib.init(provider_uri=str(provider_uri), region="us", kernels=1)
    ledger, orders = replay["ledger"], replay["orders"]
    schedule = dict(tuple(orders.groupby("stamp", sort=False))) if len(orders) else {}
    class Orders(DirectTargetWeightStrategy):
        def __init__(self):
            super().__init__(signal=pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])), risk_degree=1.)
        def generate_trade_decision(self, execute_result=None):
            start, end = self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step())
            rows = schedule.get(pd.Timestamp(start))
            submitted = [] if rows is None else [Order(row.symbol, row.adjusted_amount, Order.SELL if row.sell else Order.BUY, start, end) for row in rows.itertuples()]
            return TradeDecisionWO(submitted, self)
    exchange = IBKRTieredExchange(tariff=plan["tariff"], slippage_bps=slip, freq="15min", start_time=ledger.index[0], end_time=ledger.index[-1],
        codes=["SOXL", "SOXX"], deal_price="close", limit_threshold=None)
    reports, indicators = backtest(start_time=ledger.index[0], end_time=ledger.index[-1], strategy=Orders(),
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor", "kwargs": {"time_per_step": "15min", "generate_portfolio_metrics": True}},
        account=plan["initial_cash"], benchmark=pd.Series(0., index=ledger.index), pos_type=QlibContinuousPosition, exchange_kwargs={"exchange": exchange})
    report, positions = reports["15min"]
    net = report["return"] - report["cost"]
    if not report.index.equals(ledger.index): raise ValueError("blocked: native calendar changed")
    nav_error = float(np.max(abs(report.account.to_numpy() - ledger.nav.to_numpy()) / ledger.nav.to_numpy()))
    return_error = float(np.max(abs(net.to_numpy() - ledger["return"].to_numpy())))
    paid = pd.DataFrame(exchange.executed_fees)
    if nav_error > 1e-7 or return_error > 1e-7 or len(paid) != len(orders) or (len(orders) and not np.allclose(paid.total, orders.total, atol=1e-7, rtol=0)):
        first_error = int(np.argmax(abs(report.account.to_numpy() - ledger.nav.to_numpy())))
        fee_error = float(np.max(abs(paid.total.to_numpy() - orders.total.to_numpy()))) if len(paid) == len(orders) and len(orders) else None
        point = positions[ledger.index[first_error]]
        detail = {s: (point.get_stock_amount(s), point.get_stock_price(s), type(point.get_stock_amount(s)).__name__, type(point.get_stock_price(s)).__name__) for s in ["SOXL", "SOXX"] if point.check_stock(s)}
        raise ValueError(f"blocked: native reconciliation nav={nav_error} return={return_error} orders={len(paid)}/{len(orders)} fee={fee_error} at={ledger.index[first_error]} account={report.account.iloc[first_error]} independent={ledger.nav.iloc[first_error]} position={detail} expected_q={ledger.iloc[first_error][['soxl_quantity','soxx_quantity']].to_dict()} cash_error={point.get_cash()-ledger.cash.iloc[first_error]}")
    final = positions[report.index[-1]]
    q = [final.get_stock_amount(s) if final.check_stock(s) else 0. for s in ["SOXL", "SOXX"]]
    if not np.allclose(q, ledger.iloc[-1][["soxl_quantity", "soxx_quantity"]].to_numpy(float), atol=1e-7, rtol=0) or abs(final.get_cash() - ledger.cash.iloc[-1]) > 1e-6:
        raise ValueError("blocked: native terminal quantities or cash differ")
    return {"native_report": report, "native_positions": positions, "native_indicators": indicators, "ledger": ledger,
        "orders": orders, "native_orders": paid, "daily_net": (1 + net).groupby(net.index.normalize()).prod() - 1,
        "audit": {"nav_relative_error": nav_error, "return_error": return_error, "orders": len(orders), "status": "passed",
            "position_type": "QlibContinuousPosition", "positive_adjusted_share_residuals_preserved": True}}

"""Native Qlib account reconciliation of a prescribed intraday schedule.

This checks orders, fees, cash and quantities at observed historical marks.
It does not validate the controller or actual 09:31/10:31 broker fill prices.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.native_execution_marks import session_price_marks


def native_intraday_reconciliation(frame: pd.DataFrame, replay: pd.DataFrame, *,
                                   provider_uri: Path, cost_bps: float,
                                   account: float = 100_000.0,
                                   initial_weights: np.ndarray | None = None,
                                   tolerance: float = 1e-7, freq: str = "15min",
                                   target_convention: str = "post_fee_desired") -> dict:
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order, TradeDecisionWO
    from qlib.data import D
    from src.backtest.strategy import DirectTargetWeightStrategy
    from src.workflow.stage1_economic_action import fee_affordable_target

    symbols = ("SOXL", "SOXX")
    if freq not in {"day", "15min"} or target_convention not in {"post_fee_desired", "pre_fee_account"}:
        raise ValueError("blocked: unknown native frequency or target convention")
    schedule, market = replay.copy().reset_index(drop=True), frame.copy().reset_index(drop=True)
    required = {"signal_date", "trade_date", "exit_date", "strategy_return", "execute_open_target", "recovery_action",
                "open_soxl_weight", "open_soxx_weight", "open_cash_weight", "soxl_weight", "soxx_weight", "cash_weight"}
    if required - set(schedule) or schedule.empty or len(schedule) != len(market):
        raise ValueError("blocked: incomplete native intraday schedule")
    if not np.isfinite([account, cost_bps, tolerance]).all() or account <= 0 or cost_bps < 0 or tolerance <= 0:
        raise ValueError("blocked: invalid native account/cost/tolerance")
    for key in ["signal_date", "trade_date", "exit_date"]:
        schedule[key], market[key] = pd.to_datetime(schedule[key]), pd.to_datetime(market[key])
        if not schedule[key].equals(market[key]):
            raise ValueError("blocked: native market/schedule dates differ")
    if (schedule.trade_date.duplicated().any() or not schedule.trade_date.is_monotonic_increasing
            or not schedule.signal_date.lt(schedule.trade_date).all()
            or not schedule.trade_date.lt(schedule.exit_date).all()
            or not schedule.exit_date.iloc[:-1].reset_index(drop=True).equals(schedule.trade_date.iloc[1:].reset_index(drop=True))):
        raise ValueError("blocked: native schedule must be contiguous and causal")
    if not np.isfinite(schedule.strategy_return).all():
        raise ValueError("blocked: nonfinite research returns")
    for key in ["execute_open_target", "recovery_action"]:
        if not schedule[key].isin([True, False, 0, 1]).all():
            raise ValueError("blocked: unknown event decision")
    if freq == "day" and schedule.recovery_action.astype(bool).any():
        raise ValueError("blocked: day executor cannot represent recovery")
    for columns in [["open_soxl_weight", "open_soxx_weight", "open_cash_weight"], ["soxl_weight", "soxx_weight", "cash_weight"]]:
        weights = schedule[columns].to_numpy(float)
        if not np.isfinite(weights).all() or (weights < -1e-12).any() or not np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-9):
            raise ValueError("blocked: invalid prescribed target weights")
    native_marks = session_price_marks(provider_uri=provider_uri, sessions=schedule, freq=freq)
    return_columns = ["soxl_return", "soxx_return"] + (["early_ret_soxl", "early_ret_soxx"] if freq == "15min" else [])
    for key in return_columns:
        if key not in market or not np.allclose(market[key], native_marks[key], atol=tolerance, rtol=0):
            raise ValueError(f"blocked: research {key} differs from native price marks")
    qlib.init(provider_uri=str(Path(provider_uri).resolve()), region="us", kernels=1)
    first, last = schedule.trade_date.iloc[0], schedule.trade_date.iloc[-1]
    calendar = pd.DatetimeIndex(D.calendar(start_time=first, end_time=last + pd.Timedelta(hours=16), freq=freq))
    day_last = pd.Series(calendar, index=calendar.normalize()).groupby(level=0).last()
    if not set(schedule.trade_date).issubset(set(day_last.index)):
        raise ValueError("blocked: missing native session calendar")

    class PrescribedEvents(DirectTargetWeightStrategy):
        def __init__(self):
            super().__init__(signal=pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])), risk_degree=1.0)
            self.rows = schedule.set_index("trade_date")
            self.executed = []

        def generate_trade_decision(self, execute_result=None):
            start, end = self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step())
            stamp = pd.Timestamp(start)
            day = stamp.normalize()
            if day not in self.rows.index:
                return TradeDecisionWO([], self)
            row = self.rows.loc[day]
            if (freq == "day" or stamp.strftime("%H:%M") == "09:45") and bool(row.execute_open_target):
                columns, slot = ["open_soxl_weight", "open_soxx_weight", "open_cash_weight"], "open"
            elif freq == "15min" and stamp.strftime("%H:%M") == "10:30" and bool(row.recovery_action):
                columns, slot = ["soxl_weight", "soxx_weight", "cash_weight"], "recovery"
            else:
                return TradeDecisionWO([], self)
            quantities = self.trade_position.get_stock_amount_dict()
            quotes = {symbol: self.trade_exchange.get_deal_price(symbol, start, end, direction=Order.BUY) for symbol in symbols}
            values = np.array([quantities.get(symbol, 0.0) * quotes[symbol] for symbol in symbols] + [self.trade_position.get_cash()])
            nav = values.sum()
            desired = row[columns].to_numpy(float)
            if target_convention == "post_fee_desired":
                funded = fee_affordable_target(desired, pretrade_weights=values / nav, cost_bps=cost_bps)
            else:
                funded = desired
                fee_fraction = cost_bps / 10_000.0 * np.abs(desired[:2] - values[:2] / nav).sum()
                if desired[2] < fee_fraction - 1e-12:
                    raise ValueError("blocked: pre-fee target cannot fund its commissions")
            targets = {symbol: nav * funded[i] / quotes[symbol] for i, symbol in enumerate(symbols)}
            orders = self.trade_exchange.generate_order_for_target_amount_position(targets, quantities, start, end)
            self.executed.append({"calendar_at": stamp, "slot": slot, "nav_before": nav, "order_count": len(orders)})
            return TradeDecisionWO(orders, self)

    strategy = PrescribedEvents()
    initial = np.array([0., 0., 1.]) if initial_weights is None else np.asarray(initial_weights, dtype=float)
    if initial.shape != (3,) or not np.isfinite(initial).all() or (initial < -1e-12).any() or not np.isclose(initial.sum(), 1.0):
        raise ValueError("blocked: invalid inherited native position")
    native_account = {"cash": float(account * initial[2])}
    for i, symbol in enumerate(symbols):
        opening = float(native_marks.iloc[0][f"{symbol.lower()}_open"])
        if initial[i] > 0:
            native_account[symbol] = {"amount": float(account * initial[i] / opening), "price": opening}
    outputs, indicators = backtest(
        start_time=first + (pd.Timedelta(hours=9, minutes=45) if freq == "15min" else pd.Timedelta(0)), end_time=last + pd.Timedelta(hours=16),
        strategy=strategy,
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": freq, "generate_portfolio_metrics": True}},
        # An explicit Series prevents Qlib's initial day-frequency Account from
        # querying its default CSI300 benchmark in a 15min-only provider. Cash
        # is only a neutral ledger-report reference; SOXX comparisons are made
        # separately on the matched session account.
        account=native_account, benchmark=pd.Series(0.0, index=calendar),
        exchange_kwargs={"codes": list(symbols), "freq": freq, "deal_price": "execution_price" if freq == "15min" else "open",
                         "limit_threshold": None, "trade_unit": None, "open_cost": cost_bps / 10_000.,
                         "close_cost": cost_bps / 10_000., "min_cost": 0.0},
    )
    report, positions = outputs["15min" if freq == "15min" else "1day"]
    reconciliation = []
    previous_nav = float(account)
    for i, row in schedule.iterrows():
        position = positions[pd.Timestamp(day_last.loc[row.trade_date])]
        quantities, cash = position.get_stock_amount_dict(), float(position.get_cash())
        nav = cash + sum(quantities.get(symbol, 0.0) * native_marks.iloc[i][f"{symbol.lower()}_exit_open"] for symbol in symbols)
        value = nav / previous_nav - 1.
        ending = np.array([quantities.get(symbol, 0.0) * native_marks.iloc[i][f"{symbol.lower()}_exit_open"] for symbol in symbols] + [cash]) / nav
        entry = {"signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date,
                 "native_nav": nav, "native_return": value, "research_return": row.strategy_return,
                 "return_error": value - row.strategy_return}
        for j, symbol in enumerate(symbols):
            entry[f"native_{symbol.lower()}_ending_weight"] = ending[j]
        if {"ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"}.issubset(schedule):
            entry["maximum_ending_weight_error"] = float(np.abs(ending - row[["ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"]].to_numpy(float)).max())
        reconciliation.append(entry)
        previous_nav = nav
    checked = pd.DataFrame(reconciliation)
    error = float(checked.return_error.abs().max())
    weight_error = float(checked.get("maximum_ending_weight_error", pd.Series([0.])).max())
    if max(error, weight_error) > tolerance:
        raise ValueError(f"blocked: native intraday mismatch returns={error:g}, weights={weight_error:g}")
    return {"report": report, "positions": positions, "indicators": indicators,
            "reconciliation": checked, "events": pd.DataFrame(strategy.executed),
            "audit": {"status": "passed", "sessions": len(schedule), "cost_bps": float(cost_bps),
                      "maximum_session_return_error": error, "maximum_ending_weight_error": weight_error,
                      "provider_uri": str(Path(provider_uri).resolve()), "tolerance": tolerance,
                      "freq": freq, "target_convention": target_convention,
                      "initial_position_inherited": initial_weights is not None,
                      "scope": "prescribed opening/first-hour-close schedule; native orders, fees, cash and quantities",
                      "native_api": ["qlib.backtest.backtest", "SimulatorExecutor", "Exchange", "Position"],
                      "controller_validation": False, "live_fill_validation": False,
                      "native_report_benchmark": "zero-return cash; not the economic SOXX comparison",
                      "share_unit_convention": "fractional adjusted-price units; not broker integer-share or dividend-cash validation",
                      "execution_limitation": "opening and 10:30-close historical marks, not actual 09:31/10:31 fills"}}

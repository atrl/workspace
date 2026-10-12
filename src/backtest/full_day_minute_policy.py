"""Native Qlib accounts for a frozen allocator and matched intraday forecasts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def run_full_day_native_book(*, provider_uri: Path, parent_targets: pd.DataFrame,
                             predictions: pd.DataFrame | None, cost_bps: float,
                             start: str, end: str, reference: str = "model") -> dict:
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order, TradeDecisionWO
    from qlib.data import D
    from src.backtest.strategy import DirectTargetWeightStrategy
    from src.workflow.stage1_economic_action import fee_affordable_target

    qlib.init(provider_uri=str(Path(provider_uri).resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end + " 16:00:00", freq="15min"))
    if calendar.empty:
        raise ValueError("blocked: empty native execution calendar")
    source_calendar = pd.DatetimeIndex(D.calendar(freq="15min"))
    if not (source_calendar > calendar[-1]).any():
        raise ValueError("blocked: native executor requires a real next calendar boundary")
    parents = parent_targets.set_index("trade_date")
    forecast = predictions.set_index("entry_at") if predictions is not None else pd.DataFrame()
    symbols = ["SOXL", "SOXX"]

    class ForecastPolicy(DirectTargetWeightStrategy):
        def __init__(self):
            super().__init__(signal=pd.Series(dtype=float, index=pd.MultiIndex.from_arrays(
                [[], []], names=["datetime", "instrument"])), risk_degree=1.0)
            self.events = []
            self.benchmark_entered = False

        def generate_trade_decision(self, execute_result=None):
            first, last = self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step())
            stamp = pd.Timestamp(first)
            day = stamp.normalize()
            quantities = self.trade_position.get_stock_amount_dict()
            quotes = {s: self.trade_exchange.get_deal_price(s, first, last, direction=Order.BUY) for s in symbols}
            values = np.array([quantities.get(s, 0.) * quotes[s] for s in symbols] + [self.trade_position.get_cash()])
            nav = float(values.sum())
            held = values / nav
            if reference == "soxx_buy_hold":
                if self.benchmark_entered:
                    return TradeDecisionWO([], self)
                desired = np.array([0., 1., 0.])
                self.benchmark_entered = True
            elif reference == "parent_allocator":
                if stamp.strftime("%H:%M") != "09:45" or day not in parents.index:
                    return TradeDecisionWO([], self)
                desired = parents.loc[day, ["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(float)
            else:
                if stamp not in forecast.index or day not in parents.index:
                    # Unknown source/parent sessions remain in the native book.
                    # HOLD is explicit; no return/calendar row is deleted.
                    return TradeDecisionWO([], self)
                means = forecast.loc[stamp, ["pred_soxl", "pred_soxx"]].to_numpy(float)
                original = parents.loc[day, ["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(float)
                reduced = np.array([0., original[0] + original[1], original[2]])
                choices = [held, original, reduced]
                utilities = []
                for choice in choices:
                    funded = fee_affordable_target(choice, pretrade_weights=held, cost_bps=cost_bps)
                    fee = cost_bps / 10000. * np.abs(funded[:2] - held[:2]).sum()
                    utilities.append(float(funded[:2] @ means - fee))
                best = int(np.argmax(utilities))
                if best == 0 or utilities[best] <= utilities[0] + 1e-12:
                    return TradeDecisionWO([], self)
                desired = choices[best]
            funded = fee_affordable_target(desired, pretrade_weights=held, cost_bps=cost_bps)
            targets = {s: nav * funded[i] / quotes[s] for i, s in enumerate(symbols)}
            orders = self.trade_exchange.generate_order_for_target_amount_position(targets, quantities, first, last)
            self.events.append({"at": stamp, "nav_before": nav, "soxl_weight": desired[0],
                                "soxx_weight": desired[1], "order_count": len(orders)})
            return TradeDecisionWO(orders, self)

    policy = ForecastPolicy()
    output, _ = backtest(start_time=calendar[0], end_time=calendar[-1], strategy=policy,
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "15min", "generate_portfolio_metrics": True}},
        account=100000., benchmark=pd.Series(0., index=calendar),
        exchange_kwargs={"codes": symbols, "freq": "15min", "deal_price": "close",
            "limit_threshold": None, "trade_unit": None, "open_cost": cost_bps/10000.,
            "close_cost": cost_bps/10000., "min_cost": 0.})
    report, positions = output["15min"]
    net = report["return"] - report["cost"]
    daily = (1. + net).groupby(net.index.normalize()).prod() - 1.
    events = pd.DataFrame(policy.events)
    return {"report": report, "positions": positions, "daily_net": daily,
            "events": events, "cost_bps": cost_bps, "reference": reference}

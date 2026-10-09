"""Native Qlib reconciliation retaining positive continuous-share residuals.

The earlier Stage1 helper is sealed by a live paper release. This small adapter
therefore keeps a separate native backtest call instead of changing that helper
or monkeypatching Qlib globally. It changes only Position sale precision.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def native_open_reconciliation(frame, replay, *, provider_uri: str | Path,
                               cost_bps: float = 10., account: float = 100_000.) -> dict:
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order, TradeDecisionWO
    from qlib.contrib.evaluate import risk_analysis
    from qlib.data import D
    from src.analysis.integrated_allocation import DATES, _dates, validate_replay
    from src.backtest.qlib_continuous_position import QlibContinuousPosition
    from src.backtest.strategy import DirectTargetWeightStrategy
    from src.workflow.stage1_economic_action import fee_affordable_target

    market, schedule = _dates(frame), validate_replay(replay)
    if not market.loc[:, list(DATES)].equals(schedule.loc[:, list(DATES)]):
        raise ValueError("blocked: continuous Qlib schedule and market dates differ")
    if not np.isfinite([cost_bps, account]).all() or not 0 <= cost_bps < 10_000 or account <= 0:
        raise ValueError("blocked: invalid continuous Qlib accounting assumptions")
    if not np.allclose(schedule.cost_bps, cost_bps, rtol=0, atol=1e-12):
        raise ValueError("blocked: continuous Qlib cost differs from research replay")
    if "recovery_action" in schedule and schedule.recovery_action.astype(bool).any():
        raise ValueError("blocked: daily Qlib reconciliation cannot execute recovery slots")
    if "execute_open_target" not in schedule or not schedule.execute_open_target.isin([True, False, 0, 1]).all():
        raise ValueError("blocked: continuous Qlib needs explicit target/HOLD decisions")
    columns = [f"open_{asset}_weight" for asset in ("soxl", "soxx", "cash")]
    targets = schedule.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(targets).all() or (targets < 0).any() or not np.allclose(targets.sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("blocked: invalid continuous Qlib net targets")
    provider = Path(provider_uri).resolve()
    if not provider.is_dir():
        raise ValueError("blocked: continuous Qlib provider unavailable")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    symbols = ("SOXL", "SOXX")
    first, last, exit_day = schedule.trade_date.iloc[0], schedule.trade_date.iloc[-1], schedule.exit_date.iloc[-1]
    opening = D.features(list(symbols), ["$open"], start_time=first, end_time=exit_day, freq="day")["$open"].unstack("instrument").astype(float)
    if not np.isfinite(opening.reindex(pd.DatetimeIndex([*schedule.trade_date, exit_day])).to_numpy()).all():
        raise ValueError("blocked: missing native continuous open prices")
    for symbol in symbols:
        actual = opening[symbol].reindex(schedule.exit_date).to_numpy() / opening[symbol].reindex(schedule.trade_date).to_numpy() - 1
        if not np.allclose(actual, market[f"{symbol.lower()}_return"], rtol=0, atol=1e-7):
            raise ValueError("blocked: retained returns disagree with native continuous Qlib prices")

    class NativeContinuousSchedule(DirectTargetWeightStrategy):
        def __init__(self):
            super().__init__(signal=pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])), risk_degree=1.)
            self.schedule = schedule.set_index("trade_date")

        def generate_trade_decision(self, execute_result=None):
            start, end = self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step())
            row = self.schedule.loc[pd.Timestamp(start).normalize()]
            if not bool(row.execute_open_target):
                return TradeDecisionWO([], self)
            held = self.trade_position.get_stock_amount_dict()
            quote = {symbol: float(self.trade_exchange.get_deal_price(symbol, start, end, direction=Order.BUY)) for symbol in symbols}
            values = np.array([float(held.get(symbol, 0.)) * quote[symbol] for symbol in symbols]
                              + [float(self.trade_position.get_cash())], dtype=float)
            nav = float(values.sum())
            target = row.loc[columns].to_numpy(dtype=float)
            executable = fee_affordable_target(target, pretrade_weights=values / nav, cost_bps=cost_bps)
            amounts = {symbol: nav * executable[i] / quote[symbol] for i, symbol in enumerate(symbols)}
            orders = self.trade_exchange.generate_order_for_target_amount_position(amounts, held, start, end)
            return TradeDecisionWO(orders, self)

    native, indicators = backtest(
        start_time=first, end_time=last, strategy=NativeContinuousSchedule(),
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
        account=account, benchmark="SOXX", pos_type=QlibContinuousPosition,
        exchange_kwargs={"codes": list(symbols), "freq": "day", "deal_price": "open", "limit_threshold": None,
                         "trade_unit": None, "open_cost": cost_bps / 10_000, "close_cost": cost_bps / 10_000, "min_cost": 0.},
    )
    report, positions = native["1day"]
    previous = float(account)
    rows = []
    for row in schedule.itertuples():
        position = positions[row.trade_date]
        if not isinstance(position, QlibContinuousPosition):
            raise ValueError("blocked: Qlib did not instantiate the requested continuous Position")
        amounts = position.get_stock_amount_dict()
        next_nav = float(position.get_cash()) + sum(float(amounts.get(symbol, 0.)) * opening.loc[row.exit_date, symbol] for symbol in symbols)
        native_return = next_nav / previous - 1
        rows.append({"trade_date": row.trade_date, "exit_date": row.exit_date, "native_next_open_nav": next_nav,
                     "native_next_open_return": native_return, "research_return": row.strategy_return,
                     "return_error": native_return - row.strategy_return})
        previous = next_nav
    comparison = pd.DataFrame(rows)
    maximum = float(comparison.return_error.abs().max())
    return {"audit": {"status": "passed" if maximum <= 1e-7 else "failed", "sessions": len(schedule),
            "start": str(first.date()), "end": str(last.date()), "provider_uri": str(provider),
            "cost_bps": float(cost_bps), "initial_cash": float(account),
            "maximum_session_return_error": maximum, "tolerance": 1e-7,
            "native_api": "qlib.backtest.backtest + SimulatorExecutor + Exchange + Account + Position",
            "position_type": "src.backtest.qlib_continuous_position.QlibContinuousPosition",
            "position_precision_fix": "Every positive fractional-sale residual remains owned; no relative isclose liquidation.",
            "report_mark": "native same-day close; next-open reconciliation marks native positions through Qlib D.features",
            "sealed_stage1_code_changed": False,
            "qlib_api_reference": "https://qlib.readthedocs.io/en/latest/reference/api.html#qlib.backtest.backtest"},
        "report": report, "positions": positions, "indicators": indicators, "reconciliation": comparison,
        "risk_analysis": risk_analysis(report["return"] - report["cost"], N=252, freq=None, mode="product")}


__all__ = ["native_open_reconciliation"]

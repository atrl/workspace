"""Fixed cost-aware forecast allocation over a disclosed wealth-cushion polytope."""
from itertools import combinations

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event


def feasible_vertices(beta, minimum_cash, maximum_soxl):
    a = np.array([[1., 0.], [0., 1.], [1., 0.], [1., 1.], [3., 1.]])
    b = np.array([0., 0., maximum_soxl, 1-minimum_cash, beta])
    output = []
    for pair in combinations(range(len(a)), 2):
        matrix = a[list(pair)]
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        stocks = np.linalg.solve(matrix, b[list(pair)])
        if stocks.min() < -1e-10 or stocks[0] > maximum_soxl+1e-10 or stocks.sum() > 1-minimum_cash+1e-10 or stocks @ [3., 1.] > beta+1e-10:
            continue
        stocks[np.abs(stocks) < 1e-12] = 0.
        value = np.r_[stocks, 1-stocks.sum()]
        if not any(np.allclose(value, previous, rtol=0, atol=1e-12) for previous in output):
            output.append(value)
    return output


class ForecastVertices:
    def __init__(self, predictions, *, plan, slippage_bps):
        self.predictions, self.plan, self.slippage_bps = predictions, plan, slippage_bps

    def target(self, row, q, cash, known_prices, peak):
        nav = float(q @ known_prices + cash)
        if nav <= 0 or not np.isfinite(nav):
            raise ValueError("invalid forecast allocation wealth")
        beta = min(self.plan["multiplier"] * max(0., 1-self.plan["floor_fraction"]*peak/nav),
            1-self.plan["minimum_cash_weight"]+2*self.plan["maximum_soxl_weight"])
        prediction = self.predictions.loc[row.name]
        scores = np.array([prediction.leverage_relative_h5, 0., prediction.cash_relative_h5])
        if not np.isfinite(scores).all():
            raise ValueError("forecast unavailable; no invented allocation signal")
        held = np.r_[q*known_prices, cash] / nav
        legal_hold = held[0] <= self.plan["maximum_soxl_weight"]+1e-10 and held[2] >= self.plan["minimum_cash_weight"]-1e-10 and held[:2] @ [3.,1.] <= beta+1e-10
        best, best_score = None, float(held @ scores) if legal_hold else -np.inf
        for weights in feasible_vertices(beta, self.plan["minimum_cash_weight"], self.plan["maximum_soxl_weight"]):
            # Fresh counter conservatively quotes the first commission tier for selection.
            # Actual replay/native execution separately maintains true monthly share counters.
            preview = settle_tiered_event(q, cash, desired=weights, adjusted_prices=known_prices,
                raw_prices=row[["raw_known_soxl", "raw_known_soxx"]], stamp=row.trade_date+pd.Timedelta(hours=9,minutes=46),
                fees=TieredFees(self.plan["tariff"], slippage_bps=self.slippage_bps))
            cost = sum(order["total"] for order in preview["orders"]) / nav
            score = float(weights @ scores) - self.plan["decision_cost_multiplier"]*cost
            if score > best_score + 1e-12:
                best, best_score = weights, score
        if not legal_hold and best is None:
            raise ValueError("no feasible mandatory risk repair")
        return best

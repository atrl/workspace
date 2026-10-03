"""Two funded books sharing their provably cost-independent normalized state.

Only the same stateless, cost-blind sparse policy is supported.  Its target and
next normalized holdings do not depend on fee funding.  Every cost still gets
its own exact funding roots, orders, fees, NAV multipliers and compounded book.
This is an execution optimization, not a zero-cost counterfactual or a borrowed
benchmark/component state.  Tests compare with independently repeated rollouts.
"""
from __future__ import annotations

import numpy as np

from src.model.sparse_holdings_net_excess import numpy_rollout
from src.model.benchmark_anchor_utility import NumericalFitFailure


def _fund_state_path(state, returns, cost):
    held, target = state["pretrade_weights"], state["target_weights"]
    rate = float(cost) / 10000.
    signs = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    # Explicit two-term sums retain the scalar kernel's arithmetic order.
    h = held[:, None, :2]; z = target[:, None, :2]
    candidates = (1 + rate * (signs[:, 0] * h[..., 0] + signs[:, 1] * h[..., 1])) / (
        1 + rate * (signs[:, 0] * z[..., 0] + signs[:, 1] * z[..., 1]))
    trades = candidates[..., None] * z - h
    residual = np.abs(candidates + rate * (np.abs(trades[..., 0]) + np.abs(trades[..., 1])) - 1)
    n = candidates[np.arange(len(held)), np.argmin(residual, axis=1)]
    x0, x1, x2 = n * target[:, 0], n * target[:, 1], n * target[:, 2]
    turnover = np.abs(x0-held[:, 0]) + np.abs(x1-held[:, 1])
    fee = rate * turnover
    executable_cash = 1-x0-x1
    gross = target[:, 0]*(1+returns[:, 0]) + target[:, 1]*(1+returns[:, 1]) + target[:, 2]
    net = n*gross
    if (not np.isfinite(net).all() or (net <= 0).any() or (n <= 0).any()
            or (x2 < -1e-12).any() or np.max(np.abs(n+fee-1)) > 1e-12
            or np.max(np.abs(executable_cash-fee-x2)) > 1e-12):
        raise NumericalFitFailure("invalid paired-cost funding identity")
    # Arrays are owned by this book; consumers cannot mutate the other cost.
    path = {name: value.copy() for name, value in state.items()}
    path.update(net_equity=net, fees=fee, turnover=turnover,
                open_net_equity=x0+x1+x2,
                execution_target_weights=np.column_stack((x0, x1, executable_cash)),
                net_return=net-1, equity=np.cumprod(net), exact_zero_order=turnover==0)
    if not all(np.isfinite(v).all() for v in path.values()) or (path["equity"] <= 0).any():
        raise NumericalFitFailure("invalid paired-cost chronological wealth")
    return path


def rollout_costs(models, features, returns, *, costs, initial_weights=None, force_soxx_mask=None):
    """Return independently funded paths, keyed by distinct declared bp costs."""
    costs = tuple(float(c) for c in costs)
    if not costs or len(set(costs)) != len(costs) or not np.isfinite(costs).all() or any(c < 0 or c >= 10000 for c in costs):
        raise ValueError("blocked: distinct finite proportional costs are required")
    # The authoritative policy implementation validates feature/return shape,
    # masks, parameters and actual owned states before any sharing is allowed.
    first = numpy_rollout(models, features, returns, cost_bps=costs[0],
                          initial_weights=initial_weights, force_soxx_mask=force_soxx_mask)
    r = np.asarray(returns, dtype=np.float64)
    paths = {costs[0]: first}
    for cost in costs[1:]:
        paths[cost] = _fund_state_path(first, r, cost)
    return paths


__all__ = ["rollout_costs"]

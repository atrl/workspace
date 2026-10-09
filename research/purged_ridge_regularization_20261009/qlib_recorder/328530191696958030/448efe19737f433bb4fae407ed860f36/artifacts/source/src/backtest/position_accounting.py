"""Cash-fee accounting for fixed target decisions within one market session."""
from __future__ import annotations

import numpy as np


def settle_target_session(
    pretrade_weights,
    *,
    open_target_weights,
    asset_returns,
    cost_bps: float,
    recovery_target_weights=None,
    early_returns=None,
) -> dict:
    """Trade at each event's pre-fee NAV, charge cash, then mark actual values.

    ``None`` means HOLD and produces no order. Turnover and fees are expressed
    relative to the session's initial NAV; ending weights use final net NAV.
    The legacy ``gross_return`` field restores explicit paid cash fees to this
    path; it is not a separate zero-cost counterfactual. Stress must settle again.
    """
    values = np.asarray(pretrade_weights, dtype="float64").copy()
    returns = np.asarray(asset_returns, dtype="float64")
    if values.shape != (3,) or returns.shape != (3,) or not np.isclose(values.sum(), 1.0):
        raise ValueError("blocked: session accounting requires three normalized assets")
    if cost_bps < 0 or not np.isfinite(values).all() or not np.isfinite(returns).all():
        raise ValueError("blocked: invalid session accounting inputs")
    total_fee = 0.0

    def trade(target):
        nonlocal values, total_fee
        if target is None:
            return 0.0, 0.0
        weights = np.asarray(target, dtype="float64")
        if weights.shape != (3,) or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("blocked: invalid target weights")
        target_values = weights * values.sum()
        traded = float(np.abs(target_values[:2] - values[:2]).sum())
        fee = traded * cost_bps / 10_000.0
        target_values[2] -= fee
        if target_values[2] < -1e-12:
            raise ValueError("blocked: target has insufficient cash for transaction fees")
        values = target_values
        total_fee += fee
        return traded, fee

    open_turnover, open_fee = trade(open_target_weights)
    open_net_equity = float(values.sum())
    open_net_weights = values / open_net_equity
    recovery_turnover = recovery_fee = 0.0
    recovery_net_equity = None
    if recovery_target_weights is not None:
        early = np.asarray(early_returns, dtype="float64")
        if early.shape != (3,) or not np.isfinite(early).all() or np.any(early <= -1.0):
            raise ValueError("blocked: recovery accounting requires valid observed early returns")
        values *= 1.0 + early
        recovery_turnover, recovery_fee = trade(recovery_target_weights)
        recovery_net_equity = float(values.sum())
        post = (1.0 + returns) / (1.0 + early) - 1.0
        values *= 1.0 + post
    else:
        values *= 1.0 + returns
    net_equity = float(values.sum())
    if net_equity <= 0:
        raise ValueError("blocked: session ended with nonpositive net equity")
    return {
        "net_return": net_equity - 1.0,
        "gross_return": net_equity + total_fee - 1.0,
        "turnover": open_turnover + recovery_turnover,
        "open_turnover": open_turnover,
        "recovery_turnover": recovery_turnover,
        "fees": total_fee,
        "open_fee": open_fee,
        "recovery_fee": recovery_fee,
        "open_net_equity": open_net_equity,
        "open_net_weights": open_net_weights,
        "recovery_net_equity": recovery_net_equity,
        "net_equity": net_equity,
        "ending_values": values,
        "ending_weights": values / net_equity,
    }

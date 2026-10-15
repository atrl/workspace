"""Funded composition cancellation; nominal beta is not actual risk equivalence."""
import numpy as np


def cancel_leverage_cash(weights):
    before = np.asarray(weights, dtype=float)
    if before.shape != (3,) or not np.isfinite(before).all() or before.min() < 0 or abs(before.sum()-1) > 1e-10:
        raise ValueError('composition requires finite funded long-only weights')
    delta = min(before[0], before[2]/2)
    after = before + np.array([-delta, 3*delta, -2*delta])
    if after.min() < -1e-12 or abs(after.sum()-1) > 1e-10:
        raise ValueError('composition left funded simplex')
    if abs(after @ np.array([3.,1.,0.])-before @ np.array([3.,1.,0.])) > 1e-12:
        raise ValueError('composition changed nominal beta')
    return after


class AuditedComposition:
    def __init__(self):
        self.rows=[]

    def __call__(self, weights, *, slot, trade_date):
        after=cancel_leverage_cash(weights)
        self.rows.append({'trade_date':trade_date,'slot':slot,
            **{f'before_{a}':float(w) for a,w in zip(['soxl','soxx','cash'],weights)},
            **{f'after_{a}':float(w) for a,w in zip(['soxl','soxx','cash'],after)},
            'nominal_beta_before':float(3*weights[0]+weights[1]),
            'nominal_beta_after':float(3*after[0]+after[1]),
            'beta_error':float(3*after[0]+after[1]-3*weights[0]-weights[1]),
            'simplex_error':float(after.sum()-1)})
        return after

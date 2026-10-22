"""A fixed Stage1 target map driven by its predecision tail probability."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tail_probability_composition(weights, probability):
    """Move tail-weighted SOXL to SOXX, then recycle low-tail cash at fixed beta."""
    before = np.asarray(weights, dtype=float)
    p = float(probability)
    if (before.shape != (3,) or not np.isfinite(before).all()
            or before.min() < 0 or abs(before.sum() - 1) > 1e-10
            or not np.isfinite(p) or not 0 <= p <= 1):
        raise ValueError("tail composition requires funded weights and finite probability")
    tail_shift = p * before[0]
    remaining_soxl = before[0] - tail_shift
    compose = (1 - p) * min(remaining_soxl, before[2] / 2)
    after = before + np.array([-tail_shift - compose,
                               tail_shift + 3 * compose,
                               -2 * compose])
    if (after.min() < -1e-12 or abs(after.sum() - 1) > 1e-10
            or abs(after @ np.array([3., 1., 0.])
                   - (before @ np.array([3., 1., 0.]) - 2 * tail_shift)) > 1e-10):
        raise ValueError("tail composition broke the funded beta contract")
    return after, tail_shift, compose


class AuditedTailProbabilityComposition:
    def __init__(self, frame):
        dates = pd.DatetimeIndex(frame.trade_date)
        probabilities = np.asarray(frame.tail_probability, dtype=float)
        if (len(dates) != len(probabilities) or not dates.is_unique
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()):
            raise ValueError("tail composition requires one known probability per trade")
        self.probabilities = dict(zip(dates, probabilities))
        self.rows = []

    def __call__(self, weights, *, slot, trade_date):
        day = pd.Timestamp(trade_date)
        if day not in self.probabilities:
            raise ValueError("tail composition trade date is outside frozen predictions")
        before = np.asarray(weights, dtype=float)
        p = self.probabilities[day]
        after, tail_shift, compose = tail_probability_composition(before, p)
        self.rows.append({
            "trade_date": day, "slot": slot, "tail_probability": float(p),
            "tail_shift": float(tail_shift), "composition_delta": float(compose),
            **{f"before_{a}": float(w) for a, w in zip(("soxl", "soxx", "cash"), before)},
            **{f"after_{a}": float(w) for a, w in zip(("soxl", "soxx", "cash"), after)},
            "nominal_beta_before": float(3 * before[0] + before[1]),
            "nominal_beta_after": float(3 * after[0] + after[1]),
            "beta_error": float(3 * after[0] + after[1]
                                - (3 * before[0] + before[1] - 2 * tail_shift)),
            "simplex_error": float(after.sum() - 1),
        })
        return after

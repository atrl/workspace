"""A fixed wealth-cushion target, distinct from clipping a signal portfolio."""
import numpy as np


class CushionTarget:
    def __init__(self, *, floor_fraction, multiplier, minimum_cash, maximum_soxl, constant=False):
        self.floor_fraction, self.multiplier = floor_fraction, multiplier
        self.minimum_cash, self.maximum_soxl, self.constant = minimum_cash, maximum_soxl, constant

    def target(self, row, q, cash, known_prices, peak):
        nav = float(np.asarray(q) @ np.asarray(known_prices) + cash)
        if not np.isfinite([nav, peak]).all() or nav <= 0 or peak < nav - 1e-7:
            raise ValueError("blocked: invalid realized wealth/peak for cushion target")
        cushion_fraction = 1 - self.floor_fraction if self.constant else max(0., 1 - self.floor_fraction * peak / nav)
        beta = min(self.multiplier * cushion_fraction, 1 - self.minimum_cash + 2 * self.maximum_soxl)
        # Minimum SOXL needed to reach the exposure while keeping cash >=floor.
        soxl = max(0., (beta - (1 - self.minimum_cash)) / 2)
        soxx = beta - 3 * soxl
        return np.array([soxl, soxx, 1 - soxl - soxx])

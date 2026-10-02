"""Narrow Qlib Position precision correction for continuous share replays.

Qlib's default sell handler treats ``np.isclose(held, sold)`` as liquidation.
With its relative tolerance, a positive residual can disappear without being
sold or credited to cash. Preserve every representable positive residual; only
a negative subtraction error within a few machine ulps may close at zero.
All other position, account, exchange and report behavior remains native Qlib.
"""
from __future__ import annotations

import math

import numpy as np

from qlib.backtest.position import Position


class QlibContinuousPosition(Position):
    """Native Position with share-preserving fractional-sale subtraction."""

    def _sell_stock(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        if stock_id not in self.position:
            raise KeyError(f"{stock_id} not in current position")
        held = float(self.position[stock_id]["amount"])
        value, fee, price = float(trade_val), float(cost), float(trade_price)
        if not all(math.isfinite(x) for x in (held, value, fee, price)) or min(held, value, fee) < 0 or price <= 0:
            raise ValueError("blocked: invalid continuous Qlib sale")
        sold = value / price
        remaining = held - sold
        if not math.isfinite(sold) or not math.isfinite(remaining):
            raise ValueError("blocked: nonfinite continuous Qlib sale quantity")
        # This tolerance only handles a negative floating subtraction residual;
        # it never discards positive stock and is not an economic trade band.
        roundoff = 8 * max(math.ulp(held), math.ulp(sold))
        if remaining < -roundoff:
            raise ValueError(f"only have {held} {stock_id}, require {sold}")
        if self._settle_type not in {self.ST_CASH, self.ST_NO}:
            raise NotImplementedError(f"This type of input is not supported: {self._settle_type}")
        if remaining <= 0:
            self._del_stock(stock_id)
        else:
            # Keep NumPy float64 storage: native close quotes are float32 and
            # multiplying those by a Python float can downcast under NEP 50.
            self.position[stock_id]["amount"] = np.float64(remaining)
        # Keep the native cash and delayed-settlement semantics unchanged.
        cash_delta = value - fee
        if self._settle_type == self.ST_CASH:
            self.position["cash_delay"] += cash_delta
        else:
            self.position["cash"] += cash_delta


__all__ = ["QlibContinuousPosition"]

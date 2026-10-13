"""Per-order IBKR cost projection using observed unadjusted share prices.

The caller supplies a dated tariff and venue assumption. This is not a broker
statement reconciler. Historical NAV may use adjusted Qlib units; commission
share counts always use actual raw prices, never those adjusted units.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from qlib.backtest.exchange import Exchange


class TieredFees:
    def __init__(self, tariff: dict, *, slippage_bps: float):
        self.tariff = deepcopy(tariff)
        self.slippage_bps = float(slippage_bps)
        self.monthly_shares: dict[str, float] = {}
        if self.slippage_bps < 0 or not np.isfinite(self.slippage_bps):
            raise ValueError("blocked: invalid execution slippage scenario")

    def quote(self, *, shares: int, raw_price: float, sell: bool, stamp) -> dict:
        if shares < 0 or shares != int(shares) or not np.isfinite(raw_price) or raw_price <= 0:
            raise ValueError("blocked: fees require whole actual shares and a valid raw price")
        names = ("brokerage", "venue", "clearing", "sec", "taf", "cat", "pass_through", "slippage")
        if shares == 0:
            return {**{name: 0. for name in names}, "total": 0., "notional": 0., "cap_applied": False}
        value = shares * float(raw_price)
        month = pd.Timestamp(stamp).strftime("%Y-%m")
        volume = self.monthly_shares.get(month, 0.)
        remaining, broker, lower = float(shares), 0., 0.
        for tier in self.tariff["tiers"]:
            upper = float(tier["through_shares"]) if tier["through_shares"] is not None else np.inf
            count = min(remaining, max(0., upper - max(volume, lower)))
            broker += count * tier["per_share"]
            remaining -= count
            volume += count
            lower = upper
            if remaining <= 0:
                break
        if remaining > 0:
            raise ValueError("blocked: fee tiers do not cover order volume")
        before_cap = max(broker, self.tariff["minimum_per_order"])
        cap = value * self.tariff["maximum_order_fraction"]
        broker = min(before_cap, cap)
        result = {"brokerage": broker, "venue": (shares * self.tariff["venue_per_share"] if raw_price >= 1.
                    else value * self.tariff["venue_subdollar_fraction"]),
            "clearing": min(shares * self.tariff["clearing_per_share"], value * self.tariff["clearing_maximum_fraction"]),
            "sec": value * self.tariff["sec_sale_fraction"] if sell else 0.,
            "taf": min(shares * self.tariff["taf_per_sold_share"], self.tariff["taf_maximum_per_trade"]) if sell else 0.,
            "cat": shares * self.tariff["cat_per_share"],
            "pass_through": broker * self.tariff["commission_pass_through_fraction"],
            "slippage": value * self.slippage_bps / 10_000.,
            "notional": value, "cap_applied": cap < before_cap}
        result["total"] = sum(result[name] for name in names)
        return result

    def commit(self, shares: int, *, stamp, cap_applied: bool):
        # IBKR excludes commission-cap orders from the volume-tier counter.
        if not cap_applied and shares:
            month = pd.Timestamp(stamp).strftime("%Y-%m")
            self.monthly_shares[month] = self.monthly_shares.get(month, 0.) + int(shares)


def settle_tiered_event(quantities, cash, *, desired, adjusted_prices, raw_prices,
                        stamp, fees: TieredFees, symbols=("SOXL", "SOXX")) -> dict:
    """Integer raw-share changes; sell first and fund buys including every fee."""
    held = np.asarray(quantities, float).copy()
    adjusted, raw = np.asarray(adjusted_prices, float), np.asarray(raw_prices, float)
    if (held.shape != (2,) or adjusted.shape != (2,) or raw.shape != (2,)
            or not np.isfinite(np.r_[held, cash, adjusted, raw]).all()
            or (held < -1e-10).any() or min(adjusted.min(), raw.min()) <= 0 or cash < -1e-8):
        raise ValueError("blocked: invalid event quantities/cash/observed prices")
    before_nav = float(held @ adjusted + cash)
    if desired is None:
        return {"quantities": held, "cash": float(cash), "orders": [], "nav_before": before_nav}
    desired = np.asarray(desired, float)
    if desired.shape != (3,) or not np.isfinite(desired).all() or desired.min() < 0 or not np.isclose(desired.sum(), 1.):
        raise ValueError("blocked: event target is not a funded simplex")
    delta = desired[:2] * before_nav - held * adjusted
    shares = np.sign(delta) * np.floor(np.abs(delta) / raw + 1e-10)
    orders = []
    for sell in (True, False):
        for number, symbol in enumerate(symbols):
            signed = int(shares[number])
            if not signed or (signed < 0) != sell:
                continue
            amount = abs(signed)
            quote = fees.quote(shares=amount, raw_price=raw[number], sell=sell, stamp=stamp)
            if not sell:
                # The SOXX benchmark requests all stock; this funding loop also
                # handles finite cash without hiding minimum commissions.
                while amount and quote["notional"] + quote["total"] > cash + 1e-8:
                    amount -= max(1, int(np.ceil((quote["notional"] + quote["total"] - cash) / raw[number])))
                    amount = max(0, amount)
                    quote = fees.quote(shares=amount, raw_price=raw[number], sell=False, stamp=stamp)
                if not amount:
                    continue
            direction = -1 if sell else 1
            adjusted_amount = amount * raw[number] / adjusted[number]
            held[number] += direction * adjusted_amount
            cash -= direction * quote["notional"] + quote["total"]
            fees.commit(amount, stamp=stamp, cap_applied=quote["cap_applied"])
            orders.append({"symbol": symbol, "sell": sell, "raw_shares": amount,
                "adjusted_amount": adjusted_amount, "adjusted_price": adjusted[number],
                "raw_price": raw[number], "stamp": pd.Timestamp(stamp), **quote})
    if held.min() < -1e-8 or cash < -1e-7:
        raise ValueError("blocked: integer event could not fund cash/stock orders")
    return {"quantities": held, "cash": float(cash), "orders": orders, "nav_before": before_nav}


class IBKRTieredExchange(Exchange):
    """Use Qlib execution/Position with a raw-share fee calculation per fill."""
    def __init__(self, *, tariff, slippage_bps, **kwargs):
        self.fees = TieredFees(tariff, slippage_bps=slippage_bps)
        self.executed_fees = []
        kwargs.update(open_cost=0., close_cost=0., min_cost=0., impact_cost=0., trade_unit=None)
        kwargs["subscribe_fields"] = [*kwargs.get("subscribe_fields", []), "$raw_execution_price"]
        super().__init__(**kwargs)

    def _fee_quote(self, order, trade_val):
        from qlib.backtest.decision import Order

        raw = float(self.get_quote_info(order.stock_id, order.start_time, order.end_time, "$raw_execution_price"))
        shares = trade_val / raw
        integer = int(round(shares))
        if abs(shares - integer) > 1e-6:
            raise ValueError("blocked: native order is not whole shares in the raw price plane")
        return integer, self.fees.quote(shares=integer, raw_price=raw,
            sell=order.direction == Order.SELL, stamp=order.start_time)

    def _calc_trade_info_by_order(self, order, position, dealt_order_amount):
        price, value, _ = super()._calc_trade_info_by_order(order, position, dealt_order_amount)
        _, fee = self._fee_quote(order, value)
        if position is not None and order.direction == order.BUY and value + fee["total"] > position.get_cash() + 1e-7:
            raise ValueError("blocked: native tiered buy cannot fund its commission")
        return price, value, fee["total"]

    def deal_order(self, order, **kwargs):
        value, cost, price = super().deal_order(order, **kwargs)
        if value > 1e-5:
            shares, fee = self._fee_quote(order, value)
            self.fees.commit(shares, stamp=order.start_time, cap_applied=fee["cap_applied"])
            self.executed_fees.append({"symbol": order.stock_id, "stamp": order.start_time,
                                       "raw_shares": shares, **fee})
        return value, cost, price

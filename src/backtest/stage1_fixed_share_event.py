"""Causal one-asset raw-whole-share buy for Stage1 research episodes.

Quantity is frozen from the completed decision bar. The later fill can fail
cash affordability, but it cannot silently resize or rebalance another asset.
Broker rejection/partial-fill behavior remains an unqualified source boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_buy_intent(*, symbol: str, decision_nav: float, raw_close: float,
                     cash: float, fees, decision_at, nominal_beta_increment=.10):
    if symbol not in ("SOXL", "SOXX"):
        raise ValueError("blocked: fixed buy asset differs")
    if (not np.isfinite([decision_nav, raw_close, cash]).all() or
            decision_nav <= 0 or raw_close <= 0 or cash < -1e-8 or
            nominal_beta_increment <= 0):
        raise ValueError("blocked: fixed buy decision state invalid")
    nominal_weight = nominal_beta_increment / (3 if symbol == "SOXL" else 1)
    shares = int(np.floor(nominal_weight * decision_nav / raw_close + 1e-10))
    status = "ready"
    if shares <= 0:
        status = "decision_no_whole_share"
    else:
        projected = fees.quote(shares=shares, raw_price=raw_close, sell=False,
                               stamp=pd.Timestamp(decision_at))
        if projected["notional"] + projected["total"] > cash + 1e-8:
            status = "decision_cash_insufficient"
    return {"symbol": symbol, "raw_shares": shares, "status": status,
            "decision_at": pd.Timestamp(decision_at),
            "decision_raw_close": float(raw_close),
            "decision_nav": float(decision_nav),
            "nominal_beta_increment": float(nominal_beta_increment)}


def settle_fixed_buy(quantities, cash, *, intent: dict,
                     adjusted_fill_prices, raw_fill_prices, fill_at, fees):
    q = np.asarray(quantities, float).copy()
    adjusted = np.asarray(adjusted_fill_prices, float)
    raw = np.asarray(raw_fill_prices, float)
    if (q.shape != (2,) or adjusted.shape != (2,) or raw.shape != (2,) or
            not np.isfinite([*q, cash, *adjusted, *raw]).all() or
            q.min() < -1e-8 or cash < -1e-8 or adjusted.min() <= 0 or raw.min() <= 0):
        raise ValueError("blocked: fixed-buy fill source invalid")
    if not isinstance(intent, dict) or set(intent) != {
            "symbol", "raw_shares", "status", "decision_at", "decision_raw_close",
            "decision_nav", "nominal_beta_increment"}:
        raise ValueError("blocked: fixed-buy intent contract differs")
    fill = pd.Timestamp(fill_at)
    if (intent["symbol"] not in ("SOXL", "SOXX") or
            fill != pd.Timestamp(intent["decision_at"]) + pd.Timedelta(minutes=1) or
            intent["raw_shares"] != int(intent["raw_shares"]) or intent["raw_shares"] < 0):
        raise ValueError("blocked: fixed-buy asset, share count or clock differs")
    before = float(q @ adjusted + cash)
    if intent["status"] != "ready":
        return {"quantities":q,"cash":float(cash),"orders":[],
                "status":intent["status"],"nav_before":before}
    number = 0 if intent["symbol"] == "SOXL" else 1
    shares = int(intent["raw_shares"])
    quote = fees.quote(shares=shares, raw_price=float(raw[number]),
                       sell=False, stamp=fill)
    if quote["notional"] + quote["total"] > cash + 1e-8:
        return {"quantities":q,"cash":float(cash),"orders":[],
                "status":"insufficient_fill_cash","nav_before":before}
    q[number] += shares * raw[number] / adjusted[number]
    remaining = float(cash - quote["notional"] - quote["total"])
    if remaining < -1e-7:
        raise ValueError("blocked: fixed-buy cash negative after quote")
    fees.commit(shares, stamp=fill, cap_applied=quote["cap_applied"])
    order = {"symbol":intent["symbol"],"sell":False,"raw_shares":shares,
             "adjusted_amount":shares * raw[number] / adjusted[number],
             "adjusted_price":float(adjusted[number]),"raw_price":float(raw[number]),
             "stamp":fill,**quote}
    after = float(q @ adjusted + remaining)
    if abs(after - (before - quote["total"])) > 1e-7:
        raise ValueError("blocked: fixed-buy NAV does not equal fee loss")
    return {"quantities":q,"cash":remaining,"orders":[order],
            "status":"funded","nav_before":before}

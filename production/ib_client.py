#!/usr/bin/env python3
from __future__ import annotations

"""Thin Interactive Brokers client wrapper for the SOXL execution path.

This file intentionally keeps the dependency surface narrow. The rest of the
runtime talks to a BrokerAdapter, not directly to `ib_insync` or TWS/Gateway.
That makes it possible to test the execution flow with PaperBroker first and
swap in a real IB implementation only when the environment is ready.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class IBConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 37
    account: str = ""
    paper: bool = True


class IBClient:
    """Minimal wrapper around Interactive Brokers connectivity.

    The adapter only exposes the data needed by the current SOXL semi-auto
    workflow. Order type selection and richer account models can be layered on
    later without changing the runtime contract.
    """

    def __init__(self, config: IBConnectionConfig):
        self.config = config
        self._ib = None

    def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError as exc:
            raise RuntimeError("ib_insync is not installed; cannot connect to IB") from exc

        self._ib = IB()
        self._ib.connect(self.config.host, self.config.port, clientId=self.config.client_id, readonly=False)

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def healthcheck(self) -> Tuple[bool, str]:
        if self._ib is None:
            return False, "IB client not connected"
        if not self._ib.isConnected():
            return False, "IB connection lost"
        return True, "IB connection ready"

    def open_orders(self) -> List[Any]:
        if self._ib is None:
            return []
        return list(self._ib.openTrades())

    def positions(self) -> List[Any]:
        if self._ib is None:
            return []
        return list(self._ib.positions(account=self.config.account or ""))

    def account_summary(self) -> Dict[str, float]:
        if self._ib is None:
            return {}
        rows = self._ib.accountSummary(account=self.config.account or "")
        summary: Dict[str, float] = {}
        for row in rows:
            try:
                summary[row.tag] = float(row.value)
            except (TypeError, ValueError):
                continue
        return summary

    def place_target_order(
        self,
        symbol: str,
        quantity_delta: int,
        use_market_order: bool = False,
        limit_price: Optional[float] = None,
    ) -> Tuple[str, str]:
        """Submit a simple stock order and return `(order_id, status)`.

        The first production stage stays semi-automatic, so callers should print
        the order payload before invoking this method.
        """

        if self._ib is None:
            raise RuntimeError("IB client not connected")
        if quantity_delta == 0:
            return "", "skipped"

        from ib_insync import LimitOrder, MarketOrder, Stock

        contract = Stock(symbol, "SMART", "USD")
        action = "BUY" if quantity_delta > 0 else "SELL"
        quantity = abs(int(quantity_delta))
        if use_market_order:
            order = MarketOrder(action, quantity)
        else:
            if limit_price is None:
                raise RuntimeError("limit_price is required when use_market_order=False")
            order = LimitOrder(action, quantity, limit_price)

        trade = self._ib.placeOrder(contract, order)
        order_id = str(trade.order.orderId) if getattr(trade, "order", None) is not None else ""
        status = getattr(getattr(trade, "orderStatus", None), "status", "submitted")
        return order_id, status

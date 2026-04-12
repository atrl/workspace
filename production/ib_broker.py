#!/usr/bin/env python3
from __future__ import annotations

"""Interactive Brokers adapter for the SOXL runtime contract."""

from typing import List, Sequence, Tuple

from production.ib_client import IBClient, IBConnectionConfig
from src.soxl_quant.runtime import (
    AccountSnapshot,
    BrokerAdapter,
    BrokerExecutionResult,
    BrokerOrderStatus,
    ExecutionCommand,
)


class IBBroker(BrokerAdapter):
    broker_name = "interactive_brokers"

    def __init__(
        self,
        config: IBConnectionConfig,
        symbol: str = "SOXL",
        lot_size: int = 1,
        use_market_order: bool = False,
    ):
        self.client = IBClient(config)
        self.symbol = symbol
        self.lot_size = int(lot_size)
        self.use_market_order = bool(use_market_order)

    def connect(self) -> None:
        self.client.connect()

    def disconnect(self) -> None:
        self.client.disconnect()

    def healthcheck(self) -> Tuple[bool, str]:
        return self.client.healthcheck()

    def get_account_snapshot(self, date: str, price: float) -> AccountSnapshot:
        summary = self.client.account_summary()
        equity = float(summary.get("NetLiquidation", 0.0))
        cash = float(summary.get("TotalCashValue", 0.0))
        shares = 0
        market_value = 0.0
        for pos in self.client.positions():
            contract = getattr(pos, "contract", None)
            if contract is not None and getattr(contract, "symbol", "") == self.symbol:
                shares = int(getattr(pos, "position", 0))
                market_value = shares * price
                break
        account_value = equity if equity > 0 else cash + market_value
        position = market_value / account_value if account_value > 0 else 0.0
        return AccountSnapshot(date, cash, shares, market_value, account_value, position)

    def get_position(self, symbol: str, price: float) -> float:
        snapshot = self.get_account_snapshot(date="", price=price)
        del symbol
        return snapshot.position

    def get_open_orders(self, symbol: str) -> Sequence[BrokerOrderStatus]:
        orders: List[BrokerOrderStatus] = []
        for trade in self.client.open_orders():
            contract = getattr(trade, "contract", None)
            if contract is None or getattr(contract, "symbol", "") != symbol:
                continue
            order = getattr(trade, "order", None)
            status = getattr(getattr(trade, "orderStatus", None), "status", "submitted")
            orders.append(
                BrokerOrderStatus(
                    date="",
                    symbol=symbol,
                    side=getattr(order, "action", ""),
                    quantity=int(getattr(order, "totalQuantity", 0)),
                    avg_price=float(getattr(getattr(trade, "orderStatus", None), "avgFillPrice", 0.0) or 0.0),
                    status=status,
                    broker_order_id=str(getattr(order, "orderId", "")),
                    message="working order",
                )
            )
        return orders

    def place_target_position(
        self,
        date: str,
        symbol: str,
        target_position: float,
        price: float,
        reason: str,
    ):
        snapshot = self.get_account_snapshot(date, price)
        target_shares = int(snapshot.account_value * target_position / max(price, 1e-6) / self.lot_size) * self.lot_size
        delta = target_shares - snapshot.shares
        if delta == 0:
            return (
                [],
                [],
                BrokerExecutionResult(
                    submitted=False,
                    broker_name=self.broker_name,
                    status="skipped",
                    target_position=target_position,
                    resulting_position=snapshot.position,
                    error="no share delta",
                ),
            )
        order_id, status = self.client.place_target_order(
            symbol=symbol,
            quantity_delta=delta,
            use_market_order=self.use_market_order,
            limit_price=None if self.use_market_order else price,
        )
        side = "BUY" if delta > 0 else "SELL"
        command = ExecutionCommand(date, symbol, side, abs(delta), price, target_position, reason)
        broker_status = BrokerOrderStatus(
            date=date,
            symbol=symbol,
            side=side,
            quantity=abs(delta),
            avg_price=price,
            status=status,
            broker_order_id=order_id,
            message="submitted from semi-auto session",
        )
        resulting_position = target_position if status.lower() not in {"cancelled", "rejected"} else snapshot.position
        result = BrokerExecutionResult(
            submitted=True,
            broker_name=self.broker_name,
            status=status,
            target_position=target_position,
            resulting_position=resulting_position,
            broker_order_id=order_id,
            filled_qty=0,
            avg_price=0.0,
        )
        return [command], [broker_status], result

    def cancel_order(self, order_id: str) -> bool:
        del order_id
        # IB cancellation needs order lookup by id; keep this explicit until the
        # first live dry run proves the exact order lifecycle we want.
        return False

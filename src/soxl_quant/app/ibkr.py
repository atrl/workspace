from __future__ import annotations

"""Interactive Brokers adapter for the SOXL runtime contract.

This module owns the repo-side IBKR integration. Production scripts should
remain thin entrypoints and import the adapter from here.
"""

import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def check_ibkr_port(host: str = "127.0.0.1", port: int = 7497, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


@dataclass(frozen=True)
class IBConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 37
    account: str = ""
    paper: bool = True
    connect_timeout_sec: float = 10.0


class IBClient:
    """Minimal wrapper around IBKR TWS/Gateway connectivity."""

    def __init__(self, config: IBConnectionConfig):
        self.config = config
        self._ib = None

    def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError as exc:
            raise RuntimeError("ib-insync is not installed; cannot connect to IBKR") from exc

        self._ib = IB()
        self._ib.connect(
            self.config.host,
            self.config.port,
            clientId=self.config.client_id,
            timeout=self.config.connect_timeout_sec,
            readonly=False,
            account=self.config.account or "",
        )

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def healthcheck(self) -> Tuple[bool, str]:
        if self._ib is None:
            return False, "IBKR client not connected"
        if not self._ib.isConnected():
            return False, "IBKR connection lost"
        account_detail = f" account={self.config.account}" if self.config.account else ""
        mode = "paper" if self.config.paper else "live"
        return True, f"IBKR {mode} connection ready{account_detail}"

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
        if self._ib is None:
            raise RuntimeError("IBKR client not connected")
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


class IBBroker:
    broker_name = "interactive_brokers"

    def __init__(
        self,
        config: IBConnectionConfig,
        symbol: str = "SOXL",
        lot_size: int = 1,
        use_market_order: bool = False,
    ):
        self.config = config
        self.client = IBClient(config)
        self.symbol = symbol
        self.lot_size = int(lot_size)
        self.use_market_order = bool(use_market_order)

    def connect(self) -> None:
        if not check_ibkr_port(
            self.config.host,
            self.config.port,
            timeout=min(float(self.config.connect_timeout_sec), 5.0),
        ):
            raise RuntimeError(f"IBKR TWS/Gateway is not reachable at {self.config.host}:{self.config.port}")
        self.client.connect()

    def disconnect(self) -> None:
        self.client.disconnect()

    def healthcheck(self) -> Tuple[bool, str]:
        return self.client.healthcheck()

    def get_account_snapshot(self, date: str, price: float) -> AccountSnapshot:
        from src.soxl_quant.runtime import AccountSnapshot

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
        del symbol
        snapshot = self.get_account_snapshot(date="", price=price)
        return snapshot.position

    def get_open_orders(self, symbol: str) -> Sequence[BrokerOrderStatus]:
        from src.soxl_quant.runtime import BrokerOrderStatus

        orders: List[BrokerOrderStatus] = []
        for trade in self.client.open_orders():
            contract = getattr(trade, "contract", None)
            if contract is None or getattr(contract, "symbol", "") != symbol:
                continue
            order = getattr(trade, "order", None)
            order_status = getattr(trade, "orderStatus", None)
            status = getattr(order_status, "status", "submitted")
            orders.append(
                BrokerOrderStatus(
                    date="",
                    symbol=symbol,
                    side=getattr(order, "action", ""),
                    quantity=int(getattr(order, "totalQuantity", 0)),
                    avg_price=float(getattr(order_status, "avgFillPrice", 0.0) or 0.0),
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
        from src.soxl_quant.runtime import BrokerExecutionResult, BrokerOrderStatus, ExecutionCommand

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
            message="submitted from IBKR session",
        )
        normalized_status = str(status).lower()
        filled = normalized_status == "filled"
        resulting_position = target_position if filled else snapshot.position
        result = BrokerExecutionResult(
            submitted=True,
            broker_name=self.broker_name,
            status=status,
            target_position=target_position,
            resulting_position=resulting_position,
            broker_order_id=order_id,
            filled_qty=abs(delta) if filled else 0,
            avg_price=price if filled else 0.0,
        )
        return [command], [broker_status], result

    def cancel_order(self, order_id: str) -> bool:
        del order_id
        return False

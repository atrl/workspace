from __future__ import annotations

"""Futu OpenAPI adapter for the SOXL runtime contract.

This module owns the repo-side Futu simulated-trading integration. Production
scripts should stay as thin entrypoints and import the adapter from here.
"""

import socket
import signal
import time
from contextlib import contextmanager
from typing import Callable, List, Mapping, Sequence, Tuple


def check_opend_port(host: str = "127.0.0.1", port: int = 11111, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


@contextmanager
def connect_timeout(seconds: float):
    if seconds <= 0:
        yield
        return

    def handle_timeout(_signum, _frame):
        raise TimeoutError("Futu OpenD SDK connection timed out")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handle_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def call_with_connect_timeout(func: Callable[[], object], *, seconds: float):
    try:
        with connect_timeout(seconds):
            return func()
    except TimeoutError as exc:
        raise RuntimeError(str(exc)) from exc


class FutuBroker:
    """Broker adapter for Futu OpenAPI simulated trading.

    Requirements:
      - OpenD gateway running (default 127.0.0.1:11111)
      - futu-api Python SDK installed
      - Futu account logged in via OpenD QR code
    """

    broker_name = "futu"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        symbol: str = "US.SOXL",
        lot_size: int = 1,
        use_market_order: bool = False,
        unlock_password: str = "",
        connect_timeout_sec: float = 20.0,
        account_notional: float = 0.0,
    ):
        self.host = host
        self.port = port
        self.symbol = symbol
        self.lot_size = int(lot_size)
        self.use_market_order = bool(use_market_order)
        self.unlock_password = unlock_password
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.account_notional = float(account_notional or 0.0)

        self._quote_ctx = None
        self._trade_ctx = None

    def _import_futu(self):
        try:
            import futu

            return futu
        except Exception as exc:
            raise RuntimeError("futu-api not installed") from exc

    def connect(self) -> None:
        if not check_opend_port(self.host, self.port, timeout=min(self.connect_timeout_sec, 5.0)):
            raise RuntimeError(f"Futu OpenD is not reachable at {self.host}:{self.port}")
        futu = self._import_futu()
        if hasattr(futu, "SysConfig"):
            futu.SysConfig.set_all_thread_daemon(True)
        self._quote_ctx = call_with_connect_timeout(
            lambda: futu.OpenQuoteContext(host=self.host, port=self.port),
            seconds=self.connect_timeout_sec,
        )
        self._trade_ctx = call_with_connect_timeout(
            lambda: futu.OpenSecTradeContext(
                filter_trdmarket=futu.TrdMarket.US,
                host=self.host,
                port=self.port,
            ),
            seconds=self.connect_timeout_sec,
        )
        if self.unlock_password:
            self._trade_ctx.unlock_trade(self.unlock_password)

    def disconnect(self) -> None:
        if self._quote_ctx:
            self._quote_ctx.close()
            self._quote_ctx = None
        if self._trade_ctx:
            self._trade_ctx.close()
            self._trade_ctx = None

    def healthcheck(self) -> Tuple[bool, str]:
        try:
            futu = self._import_futu()
            if self._quote_ctx is None:
                return False, "quote context not connected"
            ret, data = self._quote_ctx.get_global_state()
            if ret == futu.RET_OK:
                return True, "futu quote ok"
            return False, f"quote health failed: {data}"
        except Exception as exc:
            return False, str(exc)

    def _get_account_info(self) -> Tuple[float, float]:
        """Return (cash, total_asset) from Futu simulated account."""
        futu = self._import_futu()
        if self._trade_ctx is None:
            raise RuntimeError("trade context not connected")
        ret, data = self._trade_ctx.accinfo_query(trd_env=futu.TrdEnv.SIMULATE)
        if ret != futu.RET_OK:
            raise RuntimeError(f"accinfo_query failed: {data}")

        def first_float(cols):
            for col in cols:
                if col in data.columns:
                    value = data[col].iloc[0]
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
            return 0.0

        cash = first_float(["cash", "Cash", "total_cash"])
        total_asset = first_float(["total_assets", "TotalAssets", "market_val", "net_assets"])
        if total_asset <= 0:
            total_asset = cash
        return cash, total_asset

    def _get_position_record(self, symbol: str) -> Mapping[str, object] | None:
        futu = self._import_futu()
        if self._trade_ctx is None:
            return None
        ret, data = self._trade_ctx.position_list_query(trd_env=futu.TrdEnv.SIMULATE)
        if ret != futu.RET_OK or data is None or len(data) == 0:
            return None
        if not hasattr(data, "columns") or "code" not in data.columns:
            return None
        mask = data["code"] == symbol
        if mask.sum() == 0:
            return None
        return data.loc[mask].iloc[0].to_dict()

    @staticmethod
    def _record_float(record: Mapping[str, object] | None, columns: Sequence[str]) -> float:
        if not record:
            return 0.0
        for column in columns:
            if column not in record:
                continue
            try:
                return float(record[column])
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _first_order_id(data: object) -> str:
        for column in ("order_id", "orderId"):
            if isinstance(data, dict) and column in data:
                return str(data[column])
            if hasattr(data, "columns") and column in data.columns and len(data) > 0:
                return str(data[column].iloc[0])
        return ""

    def _get_position_shares(self, symbol: str) -> int:
        return int(self._record_float(self._get_position_record(symbol), ["qty", "Qty"]))

    def get_account_snapshot(self, date: str, price: float) -> AccountSnapshot:
        from src.soxl_quant.runtime import AccountSnapshot

        cash, account_value = self._get_account_info()
        position_record = self._get_position_record(self.symbol)
        shares = int(self._record_float(position_record, ["qty", "Qty"]))
        market_value = shares * price
        if self.account_notional > 0:
            cost_price = self._record_float(
                position_record,
                ["cost_price", "cost_price_valid", "average_cost", "avg_price", "cost"],
            )
            if shares and cost_price > 0:
                cash = self.account_notional - shares * cost_price
                account_value = cash + market_value
            else:
                account_value = self.account_notional
                cash = self.account_notional - market_value
            if account_value <= 0:
                account_value = self.account_notional
        position = market_value / account_value if account_value > 0 else 0.0
        return AccountSnapshot(date, cash, shares, market_value, account_value, position)

    def get_position(self, symbol: str, price: float) -> float:
        del symbol
        snapshot = self.get_account_snapshot(date="", price=price)
        return snapshot.position

    def get_open_orders(self, symbol: str) -> Sequence[BrokerOrderStatus]:
        from src.soxl_quant.runtime import BrokerOrderStatus

        futu = self._import_futu()
        if self._trade_ctx is None:
            return []
        ret, data = self._trade_ctx.order_list_query(status_filter_list=[], trd_env=futu.TrdEnv.SIMULATE)
        if ret != futu.RET_OK or data is None or len(data) == 0:
            return []
        orders: List[BrokerOrderStatus] = []
        for _, row in data.iterrows():
            if row.get("code") != symbol:
                continue
            orders.append(
                BrokerOrderStatus(
                    date="",
                    symbol=symbol,
                    side=row.get("trd_side", ""),
                    quantity=int(row.get("qty", 0)),
                    avg_price=float(row.get("dealt_avg_price", 0.0) or 0.0),
                    status=row.get("order_status", "submitted"),
                    broker_order_id=str(row.get("order_id", "")),
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
        target_shares = int(
            snapshot.account_value * target_position / max(price, 1e-6) / self.lot_size
        ) * self.lot_size
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

        futu = self._import_futu()
        side = futu.TrdSide.BUY if delta > 0 else futu.TrdSide.SELL
        order_type = futu.OrderType.MARKET if self.use_market_order else futu.OrderType.NORMAL

        ret, data = self._trade_ctx.place_order(
            price=0.0 if self.use_market_order else price,
            qty=abs(delta),
            code=symbol,
            trd_side=side,
            order_type=order_type,
            trd_env=futu.TrdEnv.SIMULATE,
        )

        if ret != futu.RET_OK:
            return (
                [],
                [],
                BrokerExecutionResult(
                    submitted=False,
                    broker_name=self.broker_name,
                    status="rejected",
                    target_position=target_position,
                    resulting_position=snapshot.position,
                    error=str(data),
                ),
            )

        order_id = self._first_order_id(data)
        broker_side = "BUY" if delta > 0 else "SELL"
        command = ExecutionCommand(date, symbol, broker_side, abs(delta), price, target_position, reason)

        filled, filled_qty, avg_fill_price = self._poll_order_fill(order_id, timeout_sec=300)

        if filled:
            status_msg = f"filled {filled_qty} @ {avg_fill_price:.2f}"
            resulting_position = target_position
        else:
            status_msg = "timeout / partial fill"
            resulting_position = snapshot.position

        broker_status = BrokerOrderStatus(
            date=date,
            symbol=symbol,
            side=broker_side,
            quantity=abs(delta),
            avg_price=avg_fill_price if filled else price,
            status="filled" if filled else "timeout",
            broker_order_id=order_id,
            message=status_msg,
        )
        result = BrokerExecutionResult(
            submitted=True,
            broker_name=self.broker_name,
            status="filled" if filled else "timeout",
            target_position=target_position,
            resulting_position=resulting_position,
            broker_order_id=order_id,
            filled_qty=filled_qty,
            avg_price=avg_fill_price,
        )
        return [command], [broker_status], result

    def _poll_order_fill(
        self,
        order_id: str,
        timeout_sec: int = 300,
        poll_interval: int = 5,
    ) -> Tuple[bool, int, float]:
        futu = self._import_futu()
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._trade_ctx is None:
                return False, 0, 0.0
            ret, data = self._trade_ctx.order_list_query(
                status_filter_list=[futu.OrderStatus.FILLED_PART, futu.OrderStatus.SUBMITTED],
                trd_env=futu.TrdEnv.SIMULATE,
            )
            if ret == futu.RET_OK and data is not None and len(data) > 0:
                for _, row in data.iterrows():
                    if str(row.get("order_id", "")) == order_id:
                        status = row.get("order_status", "")
                        dealt_qty = int(row.get("dealt_qty", 0) or 0)
                        dealt_avg = float(row.get("dealt_avg_price", 0.0) or 0.0)
                        if status in (futu.OrderStatus.FILLED_ALL, "FILLED_ALL", "FILLED"):
                            return True, dealt_qty, dealt_avg
                        if status in (futu.OrderStatus.CANCELLED_ALL, "CANCELLED_ALL", "REJECTED"):
                            return False, dealt_qty, dealt_avg
            time.sleep(poll_interval)
        self.cancel_order(order_id)
        return False, 0, 0.0

    def cancel_order(self, order_id: str) -> bool:
        futu = self._import_futu()
        if self._trade_ctx is None:
            return False
        try:
            ret, _data = self._trade_ctx.modify_order(
                modify_order_op=futu.ModifyOrderOp.CANCEL,
                order_id=order_id,
                qty=0,
                price=0.0,
                trd_env=futu.TrdEnv.SIMULATE,
            )
            return ret == futu.RET_OK
        except Exception:
            return False

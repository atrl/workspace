from __future__ import annotations

import json
import socket
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


NEW_YORK_TZ = ZoneInfo("America/New_York")
TRACKED_CODES = {"SOXL": "US.SOXL", "SOXX": "US.SOXX"}


def opend_reachable(host: str = "127.0.0.1", port: int = 11111, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_us_regular_session(now: datetime | None = None) -> bool:
    current = now or datetime.now(tz=NEW_YORK_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=NEW_YORK_TZ)
    current = current.astimezone(NEW_YORK_TZ)
    return current.weekday() < 5 and time(9, 30) <= current.time() < time(16, 0)


class FutuSimulatedBroker:
    """Futu adapter with no REAL trading code path.

    The adapter resolves exactly one US stock paper account and passes
    ``TrdEnv.SIMULATE`` plus that account id to every account/order call.
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 11111) -> None:
        self.host = host
        self.port = int(port)
        self._futu = None
        self._quote_ctx = None
        self._trade_ctx = None
        self._account_id: int | None = None

    def connect(self) -> None:
        if not opend_reachable(self.host, self.port):
            raise RuntimeError(f"blocked: Futu OpenD is not reachable at {self.host}:{self.port}")
        import futu

        if hasattr(futu, "SysConfig"):
            futu.SysConfig.set_all_thread_daemon(True)
        self._futu = futu
        self._quote_ctx = futu.OpenQuoteContext(host=self.host, port=self.port)
        self._trade_ctx = futu.OpenSecTradeContext(
            filter_trdmarket=futu.TrdMarket.US,
            host=self.host,
            port=self.port,
        )
        ret, accounts = self._trade_ctx.get_acc_list()
        if ret != futu.RET_OK:
            self.close()
            raise RuntimeError(f"blocked: Futu simulated account query failed: {accounts}")
        simulate = accounts.loc[
            accounts["trd_env"].astype(str).str.upper().eq("SIMULATE")
        ].copy()
        if "sim_acc_type" in simulate:
            stock = simulate.loc[
                simulate["sim_acc_type"].astype(str).str.upper().str.contains("STOCK")
            ]
            if not stock.empty:
                simulate = stock
        if len(simulate) != 1:
            self.close()
            raise RuntimeError(
                f"blocked: expected exactly one Futu US stock simulated account, found {len(simulate)}"
            )
        self._account_id = int(simulate.iloc[0]["acc_id"])

    def close(self) -> None:
        if self._quote_ctx is not None:
            self._quote_ctx.close()
        if self._trade_ctx is not None:
            self._trade_ctx.close()
        self._quote_ctx = None
        self._trade_ctx = None
        self._account_id = None

    def __enter__(self) -> "FutuSimulatedBroker":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _require_connected(self) -> tuple[Any, Any, int]:
        if self._futu is None or self._trade_ctx is None or self._account_id is None:
            raise RuntimeError("Futu simulated broker is not connected")
        return self._futu, self._trade_ctx, self._account_id

    def quotes(self) -> dict[str, float]:
        if self._futu is None or self._quote_ctx is None:
            raise RuntimeError("Futu simulated broker is not connected")
        ret, data = self._quote_ctx.get_market_snapshot(list(TRACKED_CODES.values()))
        if ret != self._futu.RET_OK:
            raise RuntimeError(f"blocked: Futu quote query failed: {data}")
        values: dict[str, float] = {}
        reverse = {code: symbol for symbol, code in TRACKED_CODES.items()}
        for _, row in data.iterrows():
            code = str(row["code"])
            if code not in reverse:
                continue
            price = float(row["last_price"])
            if price <= 0:
                raise RuntimeError(f"blocked: non-positive Futu price for {code}")
            values[reverse[code]] = price
        if set(values) != set(TRACKED_CODES):
            raise RuntimeError("blocked: Futu quote response is missing SOXL or SOXX")
        return values

    def us_market_state(self) -> str:
        if self._futu is None or self._quote_ctx is None:
            raise RuntimeError("Futu simulated broker is not connected")
        ret, state = self._quote_ctx.get_global_state()
        if ret != self._futu.RET_OK:
            raise RuntimeError(f"blocked: Futu US market-state query failed: {state}")
        return str(state.get("market_us", "")).upper()

    def account_snapshot(self) -> dict[str, Any]:
        futu, trade_ctx, account_id = self._require_connected()
        ret, info = trade_ctx.accinfo_query(
            trd_env=futu.TrdEnv.SIMULATE,
            acc_id=account_id,
        )
        if ret != futu.RET_OK or info.empty:
            raise RuntimeError(f"blocked: Futu simulated account info failed: {info}")
        ret, positions = trade_ctx.position_list_query(
            trd_env=futu.TrdEnv.SIMULATE,
            acc_id=account_id,
        )
        if ret != futu.RET_OK:
            raise RuntimeError(f"blocked: Futu simulated position query failed: {positions}")
        quantities = {symbol: 0 for symbol in TRACKED_CODES}
        if not positions.empty:
            reverse = {code: symbol for symbol, code in TRACKED_CODES.items()}
            for _, row in positions.iterrows():
                code = str(row.get("code", ""))
                if code in reverse:
                    quantities[reverse[code]] = int(float(row.get("qty", 0)))
        row = info.iloc[0]
        return {
            "environment": "SIMULATE",
            "real_trading_supported": False,
            "total_assets": float(row.get("total_assets", row.get("total_assets_val", 0.0))),
            "cash": float(row.get("cash", 0.0)),
            "positions": quantities,
        }

    def order_history(self, *, start: str, end: str) -> list[dict[str, Any]]:
        futu, trade_ctx, account_id = self._require_connected()
        ret, orders = trade_ctx.history_order_list_query(
            start=start,
            end=end,
            trd_env=futu.TrdEnv.SIMULATE,
            acc_id=account_id,
        )
        if ret != futu.RET_OK:
            raise RuntimeError(f"blocked: Futu simulated order history failed: {orders}")
        return orders.to_dict("records") if hasattr(orders, "to_dict") else []

    def rebalance(
        self,
        *,
        weights: dict[str, float],
        prices: dict[str, float],
        portfolio_equity: float,
        signal_date: str,
        slot: str,
        state_path: Path,
        dry_run: bool,
        execution_mode: str = "normal",
        current_quantities: dict[str, int] | None = None,
        now: datetime | None = None,
        remark_prefix: str = "stage1",
    ) -> dict[str, Any]:
        futu, trade_ctx, account_id = self._require_connected()
        if set(weights) != {"SOXL", "SOXX", "cash"}:
            raise ValueError("target weights must contain exactly SOXL, SOXX, and cash")
        if any(float(value) < 0.0 for value in weights.values()):
            raise ValueError("target weights cannot be negative")
        if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-8:
            raise ValueError("target weights must sum to one")
        if portfolio_equity <= 0:
            raise ValueError("portfolio_equity must be positive")
        if execution_mode not in {"normal", "close_only"}:
            raise ValueError("execution_mode must be normal or close_only")
        if not remark_prefix.startswith("stage1_") and remark_prefix != "stage1":
            raise ValueError("remark_prefix must identify a Stage1 paper cycle")
        if not dry_run and not is_us_regular_session(now):
            raise RuntimeError("blocked: Futu SIMULATE orders are only submitted during US regular hours")
        if not dry_run and self.us_market_state() not in {"MORNING", "AFTERNOON"}:
            raise RuntimeError("blocked: Futu reports that the US regular session is not open")

        snapshot = self.account_snapshot()
        current = {
            symbol: int((current_quantities or snapshot["positions"]).get(symbol, 0))
            for symbol in TRACKED_CODES
        }
        raw_target = {
            symbol: int(float(portfolio_equity) * float(weights[symbol]) / float(prices[symbol]))
            for symbol in TRACKED_CODES
        }
        target = (
            raw_target
            if execution_mode == "normal"
            else {symbol: min(current[symbol], raw_target[symbol]) for symbol in TRACKED_CODES}
        )
        key = f"{signal_date}:{slot}"
        state = _read_state(state_path)
        if not dry_run and (
            key in state.get("completed_slots", {}) or key in state.get("pending_slots", {})
        ):
            return {
                "status": "duplicate_or_pending_skipped",
                "slot_key": key,
                "environment": "SIMULATE",
                "real_broker_order": False,
                "prices": prices,
                "weights": weights,
                "portfolio_equity": float(portfolio_equity),
                "execution_mode": execution_mode,
                "current_quantities": current,
                "target_quantities": target,
                "orders": [],
            }

        deltas = {symbol: target[symbol] - int(current.get(symbol, 0)) for symbol in TRACKED_CODES}
        planned = [
            {
                "symbol": symbol,
                "code": TRACKED_CODES[symbol],
                "side": "BUY" if delta > 0 else "SELL",
                "quantity": abs(delta),
            }
            for symbol, delta in deltas.items()
            if delta
        ]
        planned.sort(key=lambda item: 0 if item["side"] == "SELL" else 1)
        orders: list[dict[str, Any]] = []
        for order in planned:
            if dry_run:
                orders.append(order | {"status": "dry_run", "order_id": ""})
                continue
            side = futu.TrdSide.BUY if order["side"] == "BUY" else futu.TrdSide.SELL
            ret, data = trade_ctx.place_order(
                price=0.0,
                qty=order["quantity"],
                code=order["code"],
                trd_side=side,
                order_type=futu.OrderType.MARKET,
                trd_env=futu.TrdEnv.SIMULATE,
                acc_id=account_id,
                remark=f"{remark_prefix}_{slot}_{signal_date}"[:64],
            )
            if ret != futu.RET_OK:
                raise RuntimeError(
                    f"blocked: Futu SIMULATE order rejected for {order['code']}: {data}"
                )
            order_id = ""
            if hasattr(data, "columns") and "order_id" in data.columns and len(data):
                order_id = str(data.iloc[0]["order_id"])
            orders.append(order | {"status": "submitted", "order_id": order_id})

        result = {
            "status": (
                "dry_run"
                if dry_run
                else ("submitted" if orders else "no_change_completed")
            ),
            "slot_key": key,
            "environment": "SIMULATE",
            "real_broker_order": False,
            "portfolio_equity": float(portfolio_equity),
            "execution_mode": execution_mode,
            "prices": prices,
            "weights": weights,
            "current_quantities": current,
            "target_quantities": target,
            "orders": orders,
        }
        if not dry_run:
            execution_session_date = (now or datetime.now(tz=NEW_YORK_TZ)).astimezone(
                NEW_YORK_TZ
            ).date().isoformat()
            slot_state = {
                "signal_date": signal_date,
                "slot": slot,
                "execution_session_date": execution_session_date,
                "prices": {symbol: float(prices[symbol]) for symbol in TRACKED_CODES},
                "weights": {asset: float(weight) for asset, weight in weights.items()},
                "target_quantities": target,
                "order_ids": [order["order_id"] for order in orders],
                "status": "submitted_awaiting_fill_confirmation",
                "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            if orders:
                state.setdefault("pending_slots", {})[key] = slot_state
            else:
                state.setdefault("completed_slots", {})[key] = slot_state | {
                    "status": "no_change_completed",
                    "confirmed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            _write_state(state_path, state)
        return result


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "environment": "SIMULATE",
            "real_trading_supported": False,
            "completed_slots": {},
            "pending_slots": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("environment") != "SIMULATE" or payload.get("real_trading_supported") is not False:
        raise ValueError("blocked: invalid Futu simulated execution state")
    return payload


def reconcile_execution_state(path: Path, orders: list[dict[str, Any]]) -> dict[str, Any]:
    state = _read_state(path)
    by_id = {str(order.get("order_id", "")): order for order in orders}
    completed = state.setdefault("completed_slots", {})
    pending = state.setdefault("pending_slots", {})
    changed = False
    for key, slot in list(pending.items()):
        order_ids = [str(value) for value in slot.get("order_ids", []) if str(value)]
        statuses = [str(by_id.get(order_id, {}).get("order_status", "")).upper() for order_id in order_ids]
        if not order_ids or (statuses and all(status == "FILLED_ALL" for status in statuses)):
            completed[key] = dict(slot) | {
                "status": "filled_confirmed" if order_ids else "no_change_completed",
                "confirmed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            del pending[key]
            changed = True
    if changed:
        _write_state(path, state)
    return state


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

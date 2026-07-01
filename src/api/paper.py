from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters import OrderIntent, TargetPosition


class PaperBroker:
    def __init__(self, *, cash: float) -> None:
        self.cash = float(cash)
        self.positions: dict[str, int] = {}

    def create_order_intent(self, target: TargetPosition) -> OrderIntent:
        if not 0.0 <= target.target_percent <= 1.0:
            raise ValueError("target_percent must be between 0.0 and 1.0 for paper/sim")
        if target.price <= 0:
            raise ValueError("price must be positive for paper/sim")
        if not target.recorder_id:
            raise ValueError("paper/sim order intent requires qlib recorder_id")
        quantity = int((self.cash * target.target_percent) / target.price)
        self.positions[target.symbol] = quantity
        return OrderIntent(
            symbol=target.symbol,
            target_percent=target.target_percent,
            quantity=quantity,
            price=target.price,
            as_of=target.as_of,
            recorder_id=target.recorder_id,
        )


class PaperLedger:
    """Project-owned paper/sim account ledger.

    This deliberately does not talk to a broker. Each account_id is an isolated
    virtual account, so multiple strategies can hold the same symbol separately.
    """

    def __init__(self, *, path: Path, initial_cash: float) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive for project-owned paper/sim")
        self.path = Path(path)
        self.initial_cash = float(initial_cash)

    def apply_target(self, target: TargetPosition, *, account_id: str = "default") -> dict[str, Any]:
        if not account_id:
            raise ValueError("account_id is required for project-owned paper/sim")
        if not 0.0 <= target.target_percent <= 1.0:
            raise ValueError("target_percent must be between 0.0 and 1.0 for project-owned paper/sim")
        if target.price <= 0:
            raise ValueError("price must be positive for project-owned paper/sim")
        if not target.recorder_id:
            raise ValueError("project-owned paper/sim target requires qlib recorder_id")

        state = self._load_state()
        account = self._account(state, account_id)
        previous_position = dict(account["positions"].get(target.symbol, {}))
        previous_quantity = int(previous_position.get("quantity", 0))
        previous_target_percent = float(previous_position.get("target_percent", 0.0))
        equity = _account_equity(account, target_symbol=target.symbol, target_price=float(target.price))
        target_quantity = int((equity * float(target.target_percent)) / float(target.price))
        quantity_delta = target_quantity - previous_quantity
        trade_value = float(quantity_delta) * float(target.price)
        account["cash"] = float(account["cash"]) - trade_value
        market_value = float(target_quantity) * float(target.price)
        account["positions"][target.symbol] = {
            "symbol": target.symbol,
            "quantity": target_quantity,
            "target_percent": float(target.target_percent),
            "last_price": float(target.price),
            "market_value": market_value,
            "as_of": target.as_of,
            "recorder_id": target.recorder_id,
        }
        account["equity"] = _account_equity(account, target_symbol=target.symbol, target_price=float(target.price))
        account["updated_at"] = target.as_of
        state["updated_at"] = target.as_of

        previous_intent = _paper_intent_payload(
            target,
            account_id=account_id,
            target_percent=previous_target_percent,
            quantity=previous_quantity,
        )
        current_intent = _paper_intent_payload(
            target,
            account_id=account_id,
            target_percent=float(target.target_percent),
            quantity=target_quantity,
        )
        fill = {
            "account_id": account_id,
            "symbol": target.symbol,
            "as_of": target.as_of,
            "price": float(target.price),
            "previous_quantity": previous_quantity,
            "quantity": target_quantity,
            "quantity_delta": quantity_delta,
            "trade_value": trade_value,
            "target_percent": float(target.target_percent),
            "previous_target_percent": previous_target_percent,
            "recorder_id": target.recorder_id,
            "execution_mode": "paper_sim",
            "real_broker_order": False,
        }
        if quantity_delta != 0:
            state["fills"].append(fill)
        self._write_state(state)
        return {
            "execution_mode": "project_owned_paper_sim",
            "real_broker_orders": False,
            "account_id": account_id,
            "previous_intent": previous_intent,
            "current_intent": current_intent,
            "fill": fill,
            "account": dict(account),
        }

    def _load_state(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "version": 1,
            "execution_mode": "project_owned_paper_sim",
            "real_broker_orders": False,
            "initial_cash": self.initial_cash,
            "accounts": {},
            "fills": [],
        }

    def _write_state(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _account(self, state: dict[str, Any], account_id: str) -> dict[str, Any]:
        accounts = state.setdefault("accounts", {})
        if account_id not in accounts:
            accounts[account_id] = {
                "account_id": account_id,
                "initial_cash": self.initial_cash,
                "cash": self.initial_cash,
                "equity": self.initial_cash,
                "positions": {},
            }
        return accounts[account_id]


def _account_equity(account: dict[str, Any], *, target_symbol: str, target_price: float) -> float:
    market_value = 0.0
    for symbol, position in dict(account.get("positions", {})).items():
        price = target_price if symbol == target_symbol else float(position.get("last_price", 0.0))
        market_value += float(position.get("quantity", 0)) * price
    return float(account.get("cash", 0.0)) + market_value


def _paper_intent_payload(
    target: TargetPosition,
    *,
    account_id: str,
    target_percent: float,
    quantity: int,
) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "symbol": target.symbol,
        "target_percent": float(target_percent),
        "quantity": int(quantity),
        "price": float(target.price),
        "as_of": target.as_of,
        "recorder_id": target.recorder_id,
        "execution_mode": "paper_sim",
        "real_broker_order": False,
    }

from __future__ import annotations

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
        quantity = int((self.cash * target.target_percent) / target.price)
        self.positions[target.symbol] = quantity
        return OrderIntent(
            symbol=target.symbol,
            target_percent=target.target_percent,
            quantity=quantity,
            price=target.price,
            as_of=target.as_of,
        )

    def set_target_position(self, symbol: str, *, target_percent: float, price: float) -> dict:
        intent = self.create_order_intent(
            TargetPosition(symbol=symbol, target_percent=target_percent, price=price, as_of="paper_sim")
        )
        quantity = intent.quantity
        return {"symbol": symbol, "target_percent": target_percent, "quantity": quantity}

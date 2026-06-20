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

from __future__ import annotations


class PaperBroker:
    def __init__(self, *, cash: float) -> None:
        self.cash = float(cash)
        self.positions: dict[str, int] = {}

    def set_target_position(self, symbol: str, *, target_percent: float, price: float) -> dict:
        quantity = int((self.cash * target_percent) / price)
        self.positions[symbol] = quantity
        return {"symbol": symbol, "target_percent": target_percent, "quantity": quantity}

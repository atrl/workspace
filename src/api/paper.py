from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    shares: int
    price: float

    @property
    def account_value(self) -> float:
        return self.cash + self.shares * self.price

    @property
    def position(self) -> float:
        value = self.account_value
        if value <= 0.0:
            return 0.0
        return self.shares * self.price / value


class PaperBrokerAdapter:
    broker_name = "paper"

    def __init__(self, *, cash: float = 10000.0, shares: int = 0):
        self.cash = float(cash)
        self.shares = int(shares)

    def healthcheck(self) -> tuple[bool, str]:
        return True, "paper adapter ready"

    def snapshot(self, *, price: float) -> AccountSnapshot:
        return AccountSnapshot(cash=self.cash, shares=self.shares, price=float(price))

    def place_target_position(self, *, target_position: float, price: float) -> AccountSnapshot:
        current = self.snapshot(price=price)
        target_value = current.account_value * max(0.0, min(float(target_position), 1.0))
        target_shares = int(target_value // float(price))
        delta = target_shares - self.shares
        self.cash -= delta * float(price)
        self.shares = target_shares
        return self.snapshot(price=price)

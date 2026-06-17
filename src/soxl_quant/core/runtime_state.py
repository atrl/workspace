from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrancheLedger:
    """Position cost ledger shared by backtest, runtime, and observations."""

    tranches: List[dict[str, float]] = None
    high_water_price: float = 0.0
    days_in_position: int = 0

    def __post_init__(self) -> None:
        if self.tranches is None:
            self.tranches = []

    def reset(self) -> None:
        self.tranches = []
        self.high_water_price = 0.0
        self.days_in_position = 0

    def update(self, prev_pos: float, new_pos: float, price: float) -> None:
        price = float(price)
        if new_pos <= 1e-6:
            self.reset()
            return
        if prev_pos <= 1e-6:
            self.tranches = [{"price": price, "shares": new_pos}]
            self.high_water_price = price
            self.days_in_position = 1
            return

        self.days_in_position += 1
        self.high_water_price = max(self.high_water_price, price)
        if new_pos > prev_pos + 1e-6:
            add_size = max(new_pos - prev_pos, 0.0)
            self.tranches.append({"price": price, "shares": add_size})
        elif new_pos < prev_pos - 1e-6:
            sell_size = prev_pos - new_pos
            remaining = sell_size
            while remaining > 1e-9 and self.tranches:
                last = self.tranches[-1]
                shares_sold = min(remaining, last["shares"])
                last["shares"] -= shares_sold
                if last["shares"] <= 1e-9:
                    self.tranches.pop()
                remaining -= shares_sold

    def snapshot(self, price: float) -> Tuple[float, float, float, float, float, float]:
        if self.days_in_position <= 0 or price <= 0 or not self.tranches:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        total_shares = sum(t["shares"] for t in self.tranches)
        total_cost = sum(t["price"] * t["shares"] for t in self.tranches)
        avg_entry_price = total_cost / total_shares if total_shares > 0 else 0.0
        avg_pnl = price / avg_entry_price - 1.0 if avg_entry_price > 0 else 0.0
        last_add_price = self.tranches[-1]["price"] if self.tranches else 0.0
        add_pnl = price / last_add_price - 1.0 if last_add_price > 0 else 0.0
        high_gap = price / self.high_water_price - 1.0 if self.high_water_price > 0 else 0.0
        age = min(self.days_in_position / 63.0, 1.0)
        last_tranche_pnl = add_pnl
        n_tranches_ratio = min(len(self.tranches) / 10.0, 1.0)
        return (
            float(age),
            float(avg_pnl),
            float(add_pnl),
            float(high_gap),
            float(last_tranche_pnl),
            float(n_tranches_ratio),
        )


def build_price_path(df: pd.DataFrame, base_price: float = 100.0) -> np.ndarray:
    returns = df["return"].fillna(0.0).to_numpy(dtype=np.float32)
    return np.cumprod(1.0 + returns, dtype=np.float32) * np.float32(base_price)


def rolling_kelly_from_returns(returns: pd.Series, min_observations: int = 5) -> tuple[float, float, float, float]:
    clean_returns = pd.Series(returns, dtype=float).dropna()
    if len(clean_returns) < int(min_observations):
        return 0.0, 0.5, 1.0, 0.0
    win_mask = clean_returns > 0
    win_rate = float(win_mask.mean())
    avg_win = float(clean_returns.where(win_mask, 0.0).mean())
    avg_loss = float((-clean_returns).where(~win_mask, 0.0).mean())
    avg_loss = max(avg_loss, 1e-8)
    odds = avg_win / avg_loss
    if odds <= 1e-8:
        kelly_raw = -1.0
    else:
        kelly_raw = win_rate - (1.0 - win_rate) / odds
    kelly_pos = float(np.clip(kelly_raw, -1.0, 1.5))
    sharpe = float(clean_returns.mean() / (clean_returns.std() + 1e-8) * np.sqrt(252))
    return kelly_pos, win_rate, odds, sharpe


__all__ = ["TrancheLedger", "build_price_path", "rolling_kelly_from_returns"]

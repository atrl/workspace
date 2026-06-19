from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core import SCHEMA_VERSION, stable_hash


class InMemoryDataAdapter:
    def __init__(self, frame: pd.DataFrame, *, source_uri: str, date_col: str = "date") -> None:
        if date_col not in frame.columns:
            raise ValueError(f"missing date column: {date_col}")
        self.source_uri = source_uri
        self.date_col = date_col
        self._frame = frame.copy()
        self._frame[date_col] = pd.to_datetime(self._frame[date_col])

    def load_frame(self, *, as_of: str) -> pd.DataFrame:
        cutoff = pd.Timestamp(as_of)
        return self._frame.loc[self._frame[self.date_col] <= cutoff].copy()

    def to_manifest(self, *, as_of: str) -> dict[str, Any]:
        frame = self.load_frame(as_of=as_of)
        return {
            "schema_version": SCHEMA_VERSION,
            "source_uri": self.source_uri,
            "as_of": as_of,
            "row_count": len(frame),
            "data_hash": stable_hash(frame.to_dict("records")),
        }


@dataclass(frozen=True)
class TargetPosition:
    symbol: str
    target_percent: float
    price: float
    as_of: str


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    target_percent: float
    quantity: int
    price: float
    as_of: str
    execution_mode: str = "paper_sim"
    real_broker_order: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_percent": self.target_percent,
            "quantity": self.quantity,
            "price": self.price,
            "as_of": self.as_of,
            "execution_mode": self.execution_mode,
            "real_broker_order": self.real_broker_order,
        }


class RealBrokerAdapter:
    def submit_order(self, intent: OrderIntent) -> None:
        raise RuntimeError("real broker execution is disabled in v1")

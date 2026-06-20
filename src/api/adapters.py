from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data import QlibProviderConfig, load_qlib_feature_frame


@dataclass(frozen=True)
class TargetPosition:
    symbol: str
    target_percent: float
    price: float
    as_of: str
    recorder_id: str


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    target_percent: float
    quantity: int
    price: float
    as_of: str
    recorder_id: str
    execution_mode: str = "paper_sim"
    real_broker_order: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_percent": self.target_percent,
            "quantity": self.quantity,
            "price": self.price,
            "as_of": self.as_of,
            "recorder_id": self.recorder_id,
            "execution_mode": self.execution_mode,
            "real_broker_order": self.real_broker_order,
        }


@dataclass(frozen=True)
class QlibDataAdapter:
    config: QlibProviderConfig
    instruments: list[str]
    fields: list[str]

    def load_frame(self, *, start: str, end: str):
        return load_qlib_feature_frame(
            self.config,
            instruments=self.instruments,
            fields=self.fields,
            start=start,
            end=end,
        )

    def to_manifest(self, *, start: str, end: str) -> dict[str, Any]:
        frame = self.load_frame(start=start, end=end)
        return {
            "source": "qlib.data.D.features",
            "provider_uri": self.config.resolved_provider_uri.as_posix(),
            "market": self.config.market,
            "region": self.config.region,
            "instruments": self.instruments,
            "fields": self.fields,
            "date_range": [start, end],
            "row_count": int(len(frame)),
        }


class RealBrokerAdapter:
    def submit_order(self, intent: OrderIntent) -> None:
        raise RuntimeError("real broker execution is disabled in v1")

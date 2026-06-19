from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MarketQuote:
    instrument: str
    date: str
    price: float
    field: str = "close"
    data_mode: str = "paper"
    source_path: str = ""
    schema_version: str = "api.market_quote.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instrument": self.instrument,
            "date": self.date,
            "price": self.price,
            "field": self.field,
            "data_mode": self.data_mode,
            "source_path": self.source_path,
        }


class PaperMarketDataAdapter:
    adapter_name = "paper_market_data"
    data_mode = "paper"

    def __init__(self, frame: pd.DataFrame, *, as_of: str, source_path: str = "", price_field: str = "close") -> None:
        self.as_of = pd.Timestamp(as_of).normalize()
        self.source_path = str(source_path)
        self.price_field = str(price_field)
        self.frame = self._prepare_frame(frame)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        as_of: str,
        source_path: str | Path = "",
        price_field: str = "close",
    ) -> "PaperMarketDataAdapter":
        return cls(frame, as_of=as_of, source_path=str(source_path), price_field=price_field)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        as_of: str,
        price_field: str = "close",
    ) -> "PaperMarketDataAdapter":
        csv_path = Path(path)
        frame = pd.read_csv(csv_path)
        return cls(frame, as_of=as_of, source_path=str(csv_path), price_field=price_field)

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "instrument", self.price_field}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"market data frame missing columns: {missing}")
        result = frame.copy()
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        future = result.loc[result["date"] > self.as_of]
        if not future.empty:
            first = pd.Timestamp(future["date"].min()).date().isoformat()
            raise ValueError(f"future market data row after as_of={self.as_of.date().isoformat()}: first={first}")
        result[self.price_field] = result[self.price_field].astype(float)
        return result.sort_values(["instrument", "date"]).reset_index(drop=True)

    def latest_quote(self, instrument: str) -> MarketQuote:
        symbol = str(instrument)
        rows = self.frame.loc[self.frame["instrument"].astype(str).eq(symbol)]
        if rows.empty:
            raise ValueError(f"no market data for instrument: {symbol}")
        latest = rows.sort_values("date").iloc[-1]
        return MarketQuote(
            instrument=symbol,
            date=pd.Timestamp(latest["date"]).date().isoformat(),
            price=float(latest[self.price_field]),
            field=self.price_field,
            data_mode=self.data_mode,
            source_path=self.source_path,
        )

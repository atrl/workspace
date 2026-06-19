from __future__ import annotations

import json
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

    @classmethod
    def from_registry(
        cls,
        registry_path: str | Path,
        *,
        dataset_key: str,
        as_of: str | None = None,
        price_field: str = "close",
    ) -> "PaperMarketDataAdapter":
        path = Path(registry_path)
        registry = json.loads(path.read_text(encoding="utf-8"))
        datasets = registry.get("datasets", {})
        if dataset_key not in datasets:
            raise ValueError(f"data registry missing dataset_key: {dataset_key}")
        current = datasets[dataset_key].get("current") or {}
        snapshot_path = _existing_path(current.get("snapshot_path", ""), base_dir=path.parent)
        manifest_path = _existing_path(current.get("manifest_path", ""), base_dir=path.parent)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_name = manifest.get("paths", {}).get("frame", "canonical_frame.csv")
        frame_path = snapshot_path / frame_name
        if not frame_path.exists():
            raise FileNotFoundError(str(frame_path))
        snapshot_as_of = as_of or str(current.get("as_of") or manifest.get("dataset", {}).get("metadata", {}).get("as_of", ""))
        if not snapshot_as_of:
            raise ValueError(f"data registry current snapshot missing as_of: {dataset_key}")
        return cls.from_csv(frame_path, as_of=snapshot_as_of, price_field=price_field)

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


def _existing_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.api import (
    PaperBrokerAdapter,
    PaperMarketDataAdapter,
    build_target_position_intent_from_dashboard,
    execute_paper_intent,
)
import src.data as data_api


def _price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-01-02", "instrument": "SOXL", "open": 10.0, "close": 10.5, "volume": 1000},
            {"date": "2026-01-05", "instrument": "SOXL", "open": 11.0, "close": 11.5, "volume": 1200},
            {"date": "2026-01-06", "instrument": "SOXL", "open": 12.0, "close": 12.5, "volume": 1300},
        ]
    )


def test_paper_market_data_adapter_returns_latest_as_of_quote_without_future_leakage():
    adapter = PaperMarketDataAdapter.from_frame(
        _price_frame().iloc[:2],
        as_of="2026-01-05",
        source_path="data/snapshots/unit/canonical_frame.csv",
    )

    quote = adapter.latest_quote("SOXL")

    assert quote.schema_version == "api.market_quote.v1"
    assert quote.instrument == "SOXL"
    assert quote.date == "2026-01-05"
    assert quote.price == 11.5
    assert quote.field == "close"
    assert quote.data_mode == "paper"
    assert quote.source_path == "data/snapshots/unit/canonical_frame.csv"
    assert adapter.adapter_name == "paper_market_data"


def test_paper_market_data_adapter_rejects_rows_after_as_of_before_filtering():
    with pytest.raises(ValueError, match="future market data row"):
        PaperMarketDataAdapter.from_frame(_price_frame(), as_of="2026-01-05")


def test_paper_market_data_adapter_can_price_paper_intent_execution():
    adapter = PaperMarketDataAdapter.from_frame(_price_frame().iloc[:2], as_of="2026-01-05")
    intent = build_target_position_intent_from_dashboard(
        {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "unit_dashboard",
            "top_models": [{"model": "ridge_mse", "avg_position": 0.5}],
        },
        instrument="SOXL",
    )

    quote = adapter.latest_quote(intent.instrument)
    result = execute_paper_intent(intent, broker=PaperBrokerAdapter(cash=1000.0), price=quote.price)

    assert result.snapshot.price == 11.5
    assert result.snapshot.shares == 43
    assert abs(result.snapshot.position - 0.4945) < 1e-4


def test_paper_market_data_adapter_loads_current_snapshot_from_data_registry(tmp_path):
    raw = pd.DataFrame(
        [
            {"date": "2026-01-02", "instrument": "SOXL", "close": 10.0, "volume": 100},
            {"date": "2026-01-05", "instrument": "SOXL", "close": 12.0, "volume": 120},
        ]
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05"]), name="date")
    snapshot = data_api.build_canonical_snapshot(
        raw,
        name="unit_soxl_daily",
        universe=("SOXL",),
        calendar=calendar,
        calendar_path="data/qlib/soxl/calendars/day.txt",
        source_paths=("provider:unit",),
        start="2026-01-01",
        end="2026-01-05",
        as_of="2026-01-05",
    )
    data_api.publish_canonical_snapshot(
        snapshot,
        output_dir=tmp_path / "data" / "snapshots" / "unit_soxl_daily" / "2026-01-05",
        dataset_key="daily.unit_soxl",
        registry_path=tmp_path / "data" / "data_registry.json",
        published_at="2026-01-05T21:00:00Z",
    )

    adapter = PaperMarketDataAdapter.from_registry(
        tmp_path / "data" / "data_registry.json",
        dataset_key="daily.unit_soxl",
    )
    quote = adapter.latest_quote("SOXL")
    registry = json.loads((tmp_path / "data" / "data_registry.json").read_text(encoding="utf-8"))
    current = registry["datasets"]["daily.unit_soxl"]["current"]

    assert quote.date == "2026-01-05"
    assert quote.price == 12.0
    assert quote.source_path == str(tmp_path / "data" / "snapshots" / "unit_soxl_daily" / "2026-01-05" / "canonical_frame.csv")
    assert adapter.as_of.date().isoformat() == current["as_of"]

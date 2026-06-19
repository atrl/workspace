from __future__ import annotations

import pandas as pd
import pytest

from src.api import (
    PaperBrokerAdapter,
    PaperMarketDataAdapter,
    build_target_position_intent_from_dashboard,
    execute_paper_intent,
)


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

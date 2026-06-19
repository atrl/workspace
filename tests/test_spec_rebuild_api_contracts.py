from __future__ import annotations

import pytest
import pandas as pd

import src.api as api
from src.api import PaperBroker


def test_data_adapter_filters_to_as_of_and_exposes_source_manifest():
    assert hasattr(api, "InMemoryDataAdapter")
    adapter = api.InMemoryDataAdapter(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-07"]),
                "close": [10.0, 11.0, 12.0],
            }
        ),
        source_uri="qlib://soxl/day",
    )

    frame = adapter.load_frame(as_of="2026-01-05")
    manifest = adapter.to_manifest(as_of="2026-01-05")

    assert frame["date"].dt.date.astype(str).tolist() == ["2026-01-02", "2026-01-05"]
    assert manifest["source_uri"] == "qlib://soxl/day"
    assert manifest["as_of"] == "2026-01-05"
    assert manifest["row_count"] == 2


def test_paper_broker_creates_order_intents_and_live_broker_fails_closed():
    assert hasattr(api, "TargetPosition")
    assert hasattr(api, "RealBrokerAdapter")
    paper = PaperBroker(cash=10_000)
    target = api.TargetPosition(symbol="SOXL", target_percent=0.25, price=50.0, as_of="2026-01-05")

    intent = paper.create_order_intent(target)

    assert intent.to_dict() == {
        "symbol": "SOXL",
        "target_percent": 0.25,
        "quantity": 50,
        "price": 50.0,
        "as_of": "2026-01-05",
        "execution_mode": "paper_sim",
        "real_broker_order": False,
    }
    assert paper.positions["SOXL"] == 50

    with pytest.raises(ValueError, match="target_percent"):
        paper.create_order_intent(api.TargetPosition(symbol="SOXL", target_percent=1.5, price=50.0, as_of="2026-01-05"))

    with pytest.raises(RuntimeError, match="disabled in v1"):
        api.RealBrokerAdapter().submit_order(intent)

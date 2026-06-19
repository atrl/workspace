from __future__ import annotations

import pandas as pd
import pytest

from src.api import (
    BrokerAdapter,
    MarketDataAdapter,
    PaperBrokerAdapter,
    PaperMarketDataAdapter,
    assert_paper_sim_mode,
)


def _price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-01-02", "instrument": "SOXL", "close": 10.0},
            {"date": "2026-01-05", "instrument": "SOXL", "close": 12.0},
        ]
    )


def test_paper_adapters_satisfy_api_adapter_contracts():
    broker = PaperBrokerAdapter(cash=1000.0, shares=0)
    market_data = PaperMarketDataAdapter.from_frame(_price_frame(), as_of="2026-01-05")

    assert isinstance(broker, BrokerAdapter)
    assert isinstance(market_data, MarketDataAdapter)
    assert broker.broker_name == "paper"
    assert market_data.adapter_name == "paper_market_data"
    assert market_data.latest_quote("SOXL").price == 12.0


def test_api_mode_guard_allows_only_paper_or_sim():
    assert assert_paper_sim_mode("paper", surface="unit") == "paper"
    assert assert_paper_sim_mode("sim", surface="unit") == "sim"

    with pytest.raises(ValueError, match="paper/sim"):
        assert_paper_sim_mode("live", surface="unit")

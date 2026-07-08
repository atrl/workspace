from __future__ import annotations

import pandas as pd
import pytest


def test_normalize_polygon_option_contracts_keeps_contract_fields():
    from src.data.options import normalize_polygon_option_contracts

    contracts = normalize_polygon_option_contracts(
        [
            {
                "ticker": "O:SOXL260131C00028000",
                "contract_type": "call",
                "expiration_date": "2026-01-31",
                "strike_price": 28,
                "shares_per_contract": 100,
                "exercise_style": "american",
                "primary_exchange": "BATO",
            }
        ],
        underlying="SOXL",
        as_of="2026-01-02",
        fetched_at="2026-07-08T00:00:00+00:00",
    )

    assert contracts.iloc[0]["provider"] == "polygon"
    assert contracts.iloc[0]["option_ticker"] == "O:SOXL260131C00028000"
    assert contracts.iloc[0]["contract_type"] == "call"
    assert contracts.iloc[0]["expiration_date"] == "2026-01-31"
    assert contracts.iloc[0]["strike_price"] == pytest.approx(28.0)
    assert contracts.iloc[0]["as_of"] == "2026-01-02"


def test_select_daily_option_targets_uses_lag_free_spot_open_and_dte_window():
    from src.data.options import select_daily_option_targets

    contracts = pd.DataFrame(
        [
            {
                "option_ticker": "O:SOXL260131C00028000",
                "contract_type": "call",
                "expiration_date": "2026-01-31",
                "strike_price": 28.0,
            },
            {
                "option_ticker": "O:SOXL260131P00028000",
                "contract_type": "put",
                "expiration_date": "2026-01-31",
                "strike_price": 28.0,
            },
            {
                "option_ticker": "O:SOXL260131C00040000",
                "contract_type": "call",
                "expiration_date": "2026-01-31",
                "strike_price": 40.0,
            },
        ]
    )
    spot = pd.DataFrame([{"date": "2026-01-02", "underlying": "SOXL", "underlying_open": 28.13}])

    selected = select_daily_option_targets(contracts, spot, target_specs=["atm_call", "atm_put"])

    assert selected["option_ticker"].tolist() == ["O:SOXL260131C00028000", "O:SOXL260131P00028000"]
    assert selected["dte"].tolist() == [29, 29]
    assert selected.iloc[0]["strike_to_open"] == pytest.approx(28.0 / 28.13)


def test_normalize_polygon_option_daily_aggregates_and_merge_replace_overlap():
    from src.data.options import merge_option_daily_bars, normalize_polygon_option_daily_aggregates

    raw = [
        {"t": 1735828200000, "o": 3.1, "h": 3.4, "l": 2.58, "c": 2.68, "v": 73, "vw": 3.0448, "n": 10}
    ]
    bars = normalize_polygon_option_daily_aggregates(
        raw,
        option_ticker="O:SOXL250131C00028000",
        underlying="SOXL",
        fetched_at="2026-07-08T00:00:00+00:00",
    )
    replacement = bars.copy()
    replacement.loc[0, "close"] = 2.99

    merged = merge_option_daily_bars(bars, replacement)

    assert len(merged) == 1
    assert merged.iloc[0]["date"] == "2025-01-02"
    assert merged.iloc[0]["close"] == pytest.approx(2.99)

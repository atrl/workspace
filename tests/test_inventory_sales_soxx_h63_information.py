import numpy as np
import pandas as pd

from src.workflow.inventory_sales_soxx_h63_information import (
    label_and_controls,
    monthly_signal,
    stationary_bootstrap_lower,
)


def test_partial_basket_signal_uses_raw_weights_and_excludes_missing_member():
    date = pd.Timestamp("2024-05-01")
    pairs = pd.DataFrame([
        {"ticker": "TSM", "accession": "tsm", "inventory": 120, "prior_inventory": 100,
         "revenue": 1100, "prior_revenue": 1000, "conditional_available_session": pd.Timestamp("2024-04-30")},
        {"ticker": "NVDA", "accession": "nvda", "inventory": 220, "prior_inventory": 200,
         "revenue": 1000, "prior_revenue": 1000, "conditional_available_session": pd.Timestamp("2024-04-30")},
    ])
    support = pd.DataFrame([
        {"datetime": date, "ticker": "TSM", "weight_pct": 5.0,
         "source_supported_weight_pct": 5.0, "original_pair_accession": "tsm"},
        {"datetime": date, "ticker": "NVDA", "weight_pct": 10.0,
         "source_supported_weight_pct": 10.0, "original_pair_accession": "nvda"},
    ])
    signal, excluded = monthly_signal(support, pairs)
    expected = .05 * (np.log(1.2) - np.log(1.1)) + .10 * np.log(1.1)
    assert len(excluded) == 0
    assert np.isclose(signal.iloc[0].raw_four_name_weight, .15)
    assert np.isclose(signal.iloc[0].inventory_sales_mismatch, expected)
    support.loc[1, "source_supported_weight_pct"] = 0.0
    signal, excluded = monthly_signal(support, pairs)
    assert signal.empty and excluded.iloc[0].tickers == ["NVDA"]


def test_label_starts_at_next_open_and_matures_after_63_intervals():
    calendar = pd.bdate_range("2023-01-02", periods=200)
    opens = pd.Series(np.exp(np.arange(200) * .001), index=calendar)
    monthly = pd.DataFrame([{"datetime": calendar[100], "instrument": "SOXX_PARTIAL_BASKET",
                             "inventory_sales_mismatch": .1, "raw_four_name_weight": .2,
                             "active_issuer_rows": 2}])
    labelled, excluded = label_and_controls(monthly, calendar, opens, 63)
    assert excluded.empty and len(labelled) == 1
    assert labelled.iloc[0].entry_session == calendar[101]
    assert labelled.iloc[0].label_maturity_session == calendar[164]
    assert np.isclose(labelled.iloc[0].SOXX_future_h63_open_return, .063)


def test_stationary_bootstrap_is_fixed_and_positive_for_positive_pairs():
    x = np.array([.1, .2, .3, .4])
    a = stationary_bootstrap_lower(x, mean_block=3, draws=1000, seed=20261014, quantile=.025 / 15)
    b = stationary_bootstrap_lower(x, mean_block=3, draws=1000, seed=20261014, quantile=.025 / 15)
    assert a == b and a > 0

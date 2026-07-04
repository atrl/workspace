from __future__ import annotations

import pandas as pd

from src.workflow.intraday_execution_label_model import (
    build_external_context_feature_frame,
    merge_external_context_features,
)


def test_external_context_features_capture_lead_lag_and_semis_breadth_without_future_rows():
    datetimes = pd.date_range("2026-01-02 09:45:00", periods=4, freq="15min", tz="America/New_York")
    base = pd.DataFrame(
        {
            "datetime": datetimes,
            "date": [item.date() for item in datetimes],
            "soxl_mom_3": [0.0, 0.03, 0.06, 0.09],
        }
    )
    rows = []
    prices = {
        "QQQ": [100.0, 101.0, 102.0, 103.0],
        "SMH": [200.0, 202.0, 204.0, 206.0],
        "SOXX": [300.0, 303.0, 306.0, 309.0],
        "NVDA": [10.0, 11.0, 12.0, 13.0],
        "AVGO": [20.0, 21.0, 22.0, 23.0],
        "AMD": [30.0, 31.0, 32.0, 33.0],
        "TSM": [40.0, 39.0, 38.0, 37.0],
        "ASML": [50.0, 49.0, 48.0, 47.0],
    }
    for symbol, closes in prices.items():
        for idx, close in enumerate(closes):
            rows.append(
                {
                    "instrument": symbol,
                    "datetime": datetimes[idx],
                    "$open": close - 0.5,
                    "$high": close + 0.25,
                    "$low": close - 0.75,
                    "$close": close,
                    "$volume": 1000 + idx,
                    "$vwap": close - 0.1,
                    "$bar_index": idx,
                    "$bar_count": 4,
                }
            )
    external_raw = pd.DataFrame(rows).set_index(["instrument", "datetime"])

    external_context, external_cols, audit = build_external_context_feature_frame(external_raw)
    merged, merged_cols, merge_audit = merge_external_context_features(base, external_context)

    assert audit["available_symbols"] == ["AMD", "ASML", "AVGO", "NVDA", "QQQ", "SMH", "SOXX", "TSM"]
    assert merge_audit["external_context_rows"] == 4
    assert "qqq_smh_soxx_lead_lag_3" in merged_cols
    assert "semis_breadth_mom3_pos" in merged_cols
    assert "semis_equal_weight_mom3" in merged_cols
    assert merged["external_context_available"].tolist() == [True, True, True, True]
    assert merged.loc[merged.index[0], "semis_breadth_mom3_pos"] == 0.0
    assert merged.loc[merged.index[-1], "semis_breadth_mom3_pos"] == 0.6
    assert merged.loc[merged.index[-1], "qqq_smh_soxx_lead_lag_3"] != 0.0

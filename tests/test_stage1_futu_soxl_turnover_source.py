import pandas as pd

from src.data.stage1_futu_soxl_turnover_source import turnover_field_quality


def test_zero_reported_turnover_rate_with_real_trading_is_unavailable():
    source = pd.DataFrame({
        "turnover_rate": [0.0, 0.15, 0.0],
        "volume": [100, 200, 0],
        "turnover": [1000.0, 3000.0, 0.0],
    })
    quality = turnover_field_quality(source)
    assert quality["zero_rate_with_positive_trading"] == 1
    assert quality["usable_rate"].notna().tolist() == [False, True, False]

import numpy as np
import pandas as pd

from src.factor.turnover_surprise import prior_median_log_turnover_surprise


def test_current_turnover_excluded_from_normalizer():
    x = pd.Series([1.0] * 20 + [np.e**2], index=pd.date_range("2025-01-01", periods=21))
    value = prior_median_log_turnover_surprise(x).iloc[-1]
    assert np.isclose(value, 2.0)


def test_missing_or_zero_turnover_remains_unknown():
    x = pd.Series([1.0] * 20 + [0.0, 2.0])
    y = prior_median_log_turnover_surprise(x)
    assert pd.isna(y.iloc[20])
    assert pd.isna(y.iloc[21])

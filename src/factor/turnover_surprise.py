"""Decision-time turnover surprise against a strictly prior session baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prior_median_log_turnover_surprise(turnover_rate: pd.Series, *, lookback: int = 20) -> pd.Series:
    if lookback < 2:
        raise ValueError("lookback must include at least two prior sessions")
    values = pd.to_numeric(turnover_rate, errors="coerce").astype(float)
    log_values = np.log(values.where(values > 0))
    baseline = log_values.shift(1).rolling(lookback, min_periods=lookback).median()
    result = log_values - baseline
    result.name = "SOXX_turnover_surprise20"
    return result

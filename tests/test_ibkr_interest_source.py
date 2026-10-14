import numpy as np
import pytest
from src.data.ibkr_interest_source import parse_benchmark_month


def test_official_rate_units_unknown_flags_and_calendar_guards():
    kwargs={'month':'202610','as_of':'2026-10-14'}
    f=parse_benchmark_month('Date USD EUR\n20261001 3.880* 2.0\n20261002 N/A 2.0\n20261005 (0.100) 2.0',**kwargs)
    assert f.benchmark_rate.iloc[0]==.0388 and f.prior_day_flag.iloc[0]==1
    assert np.isnan(f.benchmark_rate.iloc[1]) and f.benchmark_rate.iloc[2]==-.001
    assert parse_benchmark_month('Date USD EUR',**kwargs).empty
    with pytest.raises(ValueError,match='month differs'):parse_benchmark_month('Date USD\n20260901 3.8',**kwargs)
    with pytest.raises(ValueError,match='exceeds as_of'):parse_benchmark_month('Date USD\n20261015 3.8',**kwargs)

import numpy as np
import pandas as pd
import pytest
from src.analysis.forecast_information_increment import paired_forecast_information


def _forecasts():
    dates=pd.DatetimeIndex(['2017-01-03','2017-01-04','2017-01-05','2023-01-03','2023-01-04','2023-01-05'])
    actual=np.array([.2,.3,.1,.15,.25,.35])
    control=pd.DataFrame(dict(actual_soxl=actual,actual_soxx=.05,predicted_soxl=0.,predicted_soxx=0.,
        prior_soxl=0.,prior_soxx=0.,model_used=True,
        label_available_at=dates.tz_localize('America/New_York')+pd.Timedelta(days=6)),index=dates)
    candidate=control.copy();candidate.predicted_soxl=actual+.01
    return candidate,control


def _analyze(candidate,control):
    return paired_forecast_information(candidate,control,cutoff=pd.Timestamp('2024-01-01',tz='America/New_York'),
        windows={'development':[2017],'reused_diagnostic':[2023]})


def test_both_fixed_windows_must_improve_excess_mse():
    candidate,control=_forecasts();table,passed=_analyze(candidate,control)
    assert passed and len(table)==6
    assert table.query("response=='excess'").incremental_mse_pass.all()
    candidate.loc[candidate.index.year==2023,'predicted_soxl']=0.
    assert not _analyze(candidate,control)[1]


def test_equal_forecast_is_not_increment_and_masks_must_match():
    candidate,control=_forecasts()
    assert not _analyze(control,control)[1]
    candidate.loc[candidate.index[0],'model_used']=False
    with pytest.raises(ValueError,match='model-use mask'):
        _analyze(candidate,control)


def test_zero_reference_error_and_naive_cutoff_block():
    candidate,control=_forecasts();control.predicted_soxl=control.actual_soxl
    with pytest.raises(ValueError,match='nonpositive'):
        _analyze(candidate,control)
    with pytest.raises(ValueError,match='timezone'):
        paired_forecast_information(candidate,control,cutoff='2024-01-01',windows={'all':[2017,2023]})

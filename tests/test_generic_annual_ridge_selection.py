"""The generalized selected model must reproduce frozen H5 mathematics on synthetic data."""
import numpy as np
import pandas as pd
import pytest
from src.data.convex_allocation import FEATURE_COLUMNS, LABEL_COLUMNS, build_horizon_dataset, predict_annual
from src.model.annual_ridge_selection import select_annual_ridge
from src.model import generic_annual_ridge_selection as selection
from src.model.generic_horizon_estimator import predict_horizon


def _source():
    rng = np.random.default_rng(710)
    calendar = pd.bdate_range('2010-08-02', periods=1500)
    dates = pd.Series(calendar, index=calendar)
    frame = pd.DataFrame(rng.normal(size=(len(calendar), len(FEATURE_COLUMNS))), index=calendar, columns=FEATURE_COLUMNS)
    frame.index.name = 'datetime'
    def clock(values, hour, minute=0):
        return pd.Series(pd.DatetimeIndex(values).tz_localize('America/New_York')
                         + pd.Timedelta(hours=hour, minutes=minute), index=calendar)
    frame['signal_date'] = dates
    frame['trade_date'], frame['exit_date'], frame['h5_exit_date'] = dates.shift(-1), dates.shift(-2), dates.shift(-6)
    frame['feature_available_at'], frame['decision_at'] = clock(dates, 16), clock(dates, 16, 15)
    frame['return_available_at'], frame['h5_available_at'] = clock(dates.shift(-2), 9, 31), clock(dates.shift(-6), 9, 31)
    frame['forecast_available_at'] = frame.h5_available_at
    frame['auxiliary_available_at'] = clock(dates.shift(-5), 16)
    y = rng.normal(0, .02, len(frame))
    frame['soxl_return_h5'], frame['soxx_return_h5'] = 2.5*y, y
    frame.loc[frame.index[-6:], list(LABEL_COLUMNS)] = np.nan
    frame = frame.loc['2011-01-03':].copy()
    dataset = build_horizon_dataset(frame, as_of=str(calendar[-1].date()))
    dataset.calendar = calendar
    return dataset, frame


def _select(dataset, frame):
    return selection.select_annual_horizon(dataset, frame, feature_columns=FEATURE_COLUMNS,
        label_columns=LABEL_COLUMNS, availability_column='h5_available_at', outer_year=2014)


def test_h5_selected_model_and_uncertainty_equal_existing_native_protocol():
    dataset, frame = _source()
    old = select_annual_ridge(dataset, frame, outer_year=2014)
    new = _select(dataset, frame)
    assert old['selected_name'] == new['selected_name']
    np.testing.assert_allclose(old['candidate_scores'].normalized_mse,
                               new['candidate_scores'].normalized_mse, rtol=0, atol=1e-12)
    query = frame.loc[frame.index.year == 2014]
    reference = predict_annual(old['selected_fit'], query)
    result = predict_horizon(new['selected_fit'], query, feature_columns=FEATURE_COLUMNS)
    for name in ('ridge', 'prior', 'Vmean'):
        np.testing.assert_allclose(reference[name], result[name], rtol=0, atol=1e-12)
    assert new['candidate_scores'].inner_training_frame_sha256.nunique() == 1
    assert new['audit']['model_fit_calls'] == 5


def test_mismatched_native_frame_blocks_before_any_candidate_fit(monkeypatch):
    dataset, frame = _source()
    changed = frame.copy()
    changed.loc[changed.index[0], FEATURE_COLUMNS[0]] += 1
    def unexpected_fit(*args, **kwargs):
        pytest.fail('Mismatched native inputs must block before fitting')
    monkeypatch.setattr(selection, 'fit_horizon_prior', unexpected_fit)
    monkeypatch.setattr(selection, 'fit_horizon_estimator', unexpected_fit)
    with pytest.raises(ValueError, match='inference frame values differ'):
        _select(dataset, changed)

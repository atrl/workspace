"""Temporal regressions and registered gate semantics on synthetic data only."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import causal_cash_calibration as c


@pytest.fixture
def plan():
    result = json.loads(Path('research/causal_cash_calibration_20261003/plan.json').read_text())
    result['deadline_utc'] = '2099-01-01T00:00:00Z'
    return result


@pytest.fixture
def rows():
    dates = pd.bdate_range('2013-01-01', '2026-09-25', name='datetime')
    n = len(dates)
    p = .005 * np.sin(np.arange(n) / 9.)
    y = .0003 + .2 * p
    days = pd.Series(dates, index=dates)
    frame = pd.DataFrame({'signal_date': days, 'trade_date': days.shift(-1),
        'exit_date': days.shift(-2), 'model_year': dates.year,
        'cash_advantage_prediction': p, 'prior_mean_prediction': -.0005,
        'cash_advantage_1d': y, 'soxx_return': np.expm1(-y),
        'forecast_available': True, 'h1_outcome_available': True, 'score_available': True}, index=dates)
    frame['feature_available_at'] = pd.DatetimeIndex(dates).tz_localize('America/New_York') + pd.Timedelta(hours=16)
    frame['forecast_model_available_at'] = pd.to_datetime(dates.year.astype(str) + '-01-01').tz_localize('America/New_York')
    frame['return_available_at'] = pd.DatetimeIndex(frame.exit_date).tz_localize('America/New_York') + pd.Timedelta(hours=9, minutes=30)
    frame.loc[frame.exit_date.isna(), ['cash_advantage_1d', 'soxx_return']] = np.nan
    frame.loc[frame.exit_date.isna(), ['score_available', 'h1_outcome_available']] = False
    return frame


def test_affine_fit_is_prior_only_and_matches_known_coefficients(rows, plan):
    predicted, fits, audit = c.fit_calibrations(rows, plan)
    assert len(fits) == 11
    for year, fit in fits.items():
        assert fit['intercept'] == pytest.approx(.0003, abs=1e-16)
        assert fit['slope'] == pytest.approx(.2, abs=1e-13)
        train = rows.loc[fit['training_dates']]
        cutoff = pd.Timestamp(f'{year}-01-01', tz='America/New_York')
        assert (train.index.year < year).all()
        for field in ('feature_available_at', 'forecast_model_available_at', 'return_available_at'):
            assert train[field].lt(cutoff).all()
    mature = rows.score_available & (rows.index.year >= 2016)
    np.testing.assert_allclose(predicted.loc[mature, 'calibrated_prediction'], rows.loc[mature, 'cash_advantage_1d'], atol=1e-16)
    assert len(predicted) == len(rows) and not predicted.loc[rows.index.year < 2016, 'calibrated_prediction'].notna().any()
    assert audit.training_rows.is_monotonic_increasing


def test_current_and_future_outcomes_cannot_change_current_year_prediction(rows, plan):
    before, fits, _ = c.fit_calibrations(rows, plan)
    changed = rows.copy()
    mask = (changed.index.year >= 2020) & changed.score_available
    changed.loc[mask, 'cash_advantage_1d'] += .05
    changed.loc[mask, 'soxx_return'] = np.expm1(-changed.loc[mask, 'cash_advantage_1d'])
    after, other, _ = c.fit_calibrations(changed, plan)
    np.testing.assert_array_equal(before.loc[rows.index.year <= 2020, 'calibrated_prediction'], after.loc[rows.index.year <= 2020, 'calibrated_prediction'])
    assert fits[2020]['intercept'] == other[2020]['intercept']
    assert fits[2021]['intercept'] != other[2021]['intercept']


@pytest.mark.parametrize('field', ['feature_available_at', 'forecast_model_available_at', 'return_available_at'])
def test_cutoff_equality_is_excluded_and_prior_nan_is_rejected(rows, plan, field):
    date = pd.Timestamp('2014-06-02')
    changed = rows.copy()
    changed.loc[date, field] = pd.Timestamp('2016-01-01', tz='America/New_York')
    _, fits, _ = c.fit_calibrations(changed, plan)
    assert date not in fits[2016]['training_dates']
    assert date in fits[2017]['training_dates']


def test_rank_and_minimum_and_scored_integrity_fail_closed(rows, plan):
    constant = rows.copy()
    constant['cash_advantage_prediction'] = .001
    with pytest.raises(ValueError, match='rank-deficient'): c.fit_calibrations(constant, plan)
    bad = deepcopy(plan); bad['fit']['minimum_rows'] = 999999
    with pytest.raises(ValueError, match='insufficient'): c.fit_calibrations(rows, bad)
    badrows = rows.copy(); badrows.loc['2014-06-02', 'cash_advantage_prediction'] = np.nan
    with pytest.raises(ValueError, match='finite'): c.fit_calibrations(badrows, plan)


def test_deadline_fails_at_exact_boundary(plan):
    plan['deadline_utc'] = '2026-10-03T02:04:34Z'
    c.deadline_check(plan, now=datetime(2026, 10, 3, 2, 4, 33, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match='deadline'): c.deadline_check(plan, now=datetime(2026, 10, 3, 2, 4, 34, tzinfo=timezone.utc))


def test_bootstrap_window_comparators_share_draws_and_undefined_events_fail(plan, monkeypatch):
    # Isolate the orchestration on exactly 38 dates to keep this synthetic test small.
    dates = pd.bdate_range('2026-08-03', periods=38)
    frame = pd.DataFrame({'trade_date': dates, 'exit_date': dates, 'score_available': True,
        'cash_advantage_1d': np.linspace(-.01, .01, 38), 'calibrated_prediction': 0.,
        'cash_advantage_prediction': .003, 'prior_mean_prediction': -.001}, index=dates)
    monkeypatch.setattr(c, 'window_definitions', lambda r, p: iter([
        *[(f'signal_{y}', 'annual', y, np.ones(len(r), bool)) for y in range(2016, 2023)],
        *[(name, 'pooled', None, np.ones(len(r), bool)) for name in c.POOLED],
        ('user_trade_2026_08_03', 'user_window', None, np.ones(len(r), bool))]))
    calls = []
    original = c.bootstrap_window
    def short(*args, **kwargs):
        calls.append(kwargs.copy()); kwargs['resamples'] = 8
        return original(*args, **kwargs)
    monkeypatch.setattr(c, 'bootstrap_window', short)
    result = c.analyze(frame, plan)
    assert len(calls) == 8 and all(call['resamples'] == 2000 for call in calls)
    assert [call['mean_block'] for call in calls] == [20, 20, 60, 60, 20, 20, 60, 60]
    for name in c.POOLED:
        for block in (20, 60):
            raw = result['bootstrap_draws'][(name, block, 'raw')]
            prior = result['bootstrap_draws'][(name, block, 'prior_mean')]
            assert raw['indices_sha256'] == prior['indices_sha256']
    gates = result['tables']['screening_gates']
    assert len(gates) == 26
    assert not gates.loc[gates.gate.str.startswith('event_'), 'passed'].any()


def test_partial_run_blocks_repeat(tmp_path):
    (tmp_path / 'recorder_reference.json').write_text('{"complete":false}')
    with pytest.raises(ValueError, match='partial'): c._cached(tmp_path, {}, {})

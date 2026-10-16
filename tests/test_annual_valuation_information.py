import numpy as np
import pandas as pd

from src.data.annual_valuation_information import MARKET, VALUATION, TARGETS, add_optional_rank, count_selection, monthly_intervals
from src.model.monthly_valuation_information import MonthlyValuationRidge
from src.workflow.annual_valuation_information import make_dataset


def test_monthly_cutoff_uses_december_and_counts_no_return_values():
    # Synthetic calendar tests chronology only, not exchange-session validity.
    frame = monthly_intervals(pd.bdate_range('2019-01-01','2026-08-04'))
    frame['source_qualified'] = True
    frame['market_complete'] = True
    plan = {'first_execution_year_candidates':[2023,2024], 'minimum_train_rows':24}
    receipt = count_selection(frame, plan)
    assert receipt['selected_first_execution_year'] == 2023
    cutoff = pd.Timestamp(receipt['counts'][0]['cutoff'])
    assert cutoff == pd.Timestamp('2022-12-30 16:15', tz='America/New_York')
    # The November signal's label exits in January and is not yet trainable.
    assert frame.loc['2022-11-30','exit_at'] > cutoff
    assert frame.loc['2022-10-31','exit_at'] < cutoff
    frame.loc[:,'source_qualified'] = False
    assert count_selection(frame, plan)['selected_first_execution_year'] is None


def test_optional_rank_does_not_require_growth_or_future_observations():
    dates = pd.date_range('2020-01-31',periods=4,freq='ME')
    frame = pd.DataFrame({'source_qualified':True,'growth_source_qualified':False,'method_id':'annual_parent_v1',
                          'signed_annual_ep':[-.01,.02,.03,.01], 'annual_earnings_change':np.nan}, index=dates)
    prefix = add_optional_rank(frame.iloc[:3], minimum_prior_snapshots=2)
    full = add_optional_rank(frame, minimum_prior_snapshots=2)
    pd.testing.assert_frame_equal(prefix, full.iloc[:3])
    assert prefix.iloc[2].annual_ep_rank == 1.
    assert prefix.source_qualified.all() and prefix.growth_missing.eq(1).all()


def test_qlib_model_ignores_test_labels_and_all_missing_training_coordinate():
    dates = pd.date_range('2019-01-31',periods=30,freq='ME')
    frame = pd.DataFrame(index=dates)
    t = np.arange(len(frame),dtype=float)
    for i, name in enumerate(MARKET):
        frame[name] = np.sin(t/(i+2))
    frame['signed_annual_ep'] = .04+.001*t
    frame['annual_earnings_change'] = np.nan
    frame['loss_weight'] = .1
    frame['annual_ep_rank'] = np.nan
    frame['growth_missing'] = 1.
    frame['rank_missing'] = 1.
    frame[TARGETS[0]] = np.cos(t/3)*.1
    frame[TARGETS[1]] = np.sin(t/4)*.2
    train = pd.Series(t<24,index=dates); test = ~train
    dataset = make_dataset(frame, train, test)
    model = MonthlyValuationRidge(MARKET+VALUATION,TARGETS)
    model.fit(dataset)
    baseline = model.predict(dataset)
    changed = frame.copy()
    changed.loc[test,TARGETS] = 1e6
    changed.loc[test,'annual_ep_rank'] = .99
    changed_dataset = make_dataset(changed,train,test)
    other = MonthlyValuationRidge(MARKET+VALUATION,TARGETS)
    other.fit(changed_dataset)
    pd.testing.assert_frame_equal(baseline,other.predict(changed_dataset))
    assert 'annual_ep_rank' in model.audit['inactive_features']

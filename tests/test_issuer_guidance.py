import copy
import numpy as np
import pandas as pd
import pytest
from src.data.issuer_guidance import normalize_record,original_guidance_events,guidance_information_state


def fact(*,date='2020-02-20',actual=100.,reported='FY2020Q4',target='FY2021Q1',mid=110.,unit='USD_million',role='initial'):
    return dict(accession='0001045810-20-000001',release_date=date,filing_date=date,
        api_accepted_original=date+'T22:00:00Z',index_accepted_original=None,
        reported_fiscal_quarter=reported,target_fiscal_quarter=target,actual_revenue_million=actual,
        guidance_lower=None if mid is None else mid-2,guidance_mid=mid,guidance_upper=None if mid is None else mid+2,
        guidance_unit=unit,event_role=role,source_url='https://www.sec.gov/Archives/edgar/data/1045810/example.htm')


def test_original_value_not_replaced_by_interim_estimate_and_supplement_not_double_quarter():
    a=fact();early=fact(date='2020-05-05',actual=None,reported='FY2021Q1',target=None,mid=None)
    supplement=fact(date='2020-05-07',reported='FY2021Q1',target='FY2021Q2',mid=90.,role='update')
    revision=fact(date='2020-04-01',mid=80.,reported='FY2020Q4',target='FY2021Q1',role='update')
    records=[a,revision,early,supplement]
    for i,r in enumerate(records):r['accession']=str(i)
    frame=pd.DataFrame([normalize_record(r,as_of='2020-06-01') for r in records]).sort_values('date_bound')
    events=original_guidance_events(frame,expected_target_quarters=['FY2021Q1','FY2021Q2'])
    assert events.accession.tolist()==['0','3']
    np.testing.assert_allclose(events.guidance_growth,[.1,-.1])
    calendar=pd.bdate_range('2020-02-19','2020-05-11')
    state,_=guidance_information_state(events,calendar)
    assert state.iloc[0].known==0 and np.isnan(state.iloc[0].guidance_qoq)
    assert state.loc[state.datetime.eq('2020-05-06'),'guidance_qoq'].iloc[0]==pytest.approx(.1)
    assert state.loc[state.datetime.eq('2020-05-08'),'guidance_qoq'].iloc[0]==pytest.approx(-.1)


def test_zero_flat_outlook_is_known_not_unknown():
    normalized=normalize_record(fact(mid=0.,unit='pct_sequential_revenue_growth'),as_of='2020-06-01')
    assert normalized['next_quarter_numeric'] and normalized['guidance_growth']==0


def test_raw_clock_conflict_preserved_and_date_bound_conservative():
    r=fact(date='2011-02-16',reported='FY2011Q4',target='FY2012Q1')
    r['api_accepted_original']='2011-02-17T02:22:17Z';r['index_accepted_original']='2011-02-16 16:22:17'
    n=normalize_record(r,as_of='2011-03-01')
    assert n['api_accepted_raw']==r['api_accepted_original']
    assert n['index_accepted_raw']==r['index_accepted_original']
    assert n['date_bound']==pd.Timestamp('2011-02-16') and not n['first_public_timestamp_verified']


@pytest.mark.parametrize('change',[{'release_date':'2021-01-01'},{'guidance_unit':'invented'},{'actual_revenue_million':0},{'guidance_lower':200},{'api_accepted_original':'2020-02-20 16:00'}])
def test_unknown_units_unfunded_values_future_disclosures_and_naive_clocks_block(change):
    r=fact();r.update(change)
    with pytest.raises(ValueError):normalize_record(r,as_of='2020-06-01')


def test_missing_quarter_cannot_be_silently_carried_or_calendar_shortened():
    frame=pd.DataFrame([normalize_record(fact(),as_of='2020-06-01')])
    with pytest.raises(ValueError,match='incomplete'):
        original_guidance_events(frame,expected_target_quarters=['FY2021Q1','FY2021Q2'])


def test_nested_actual_units_cannot_silently_change_ratio():
    r=fact()
    r['actual_revenue']={'value':.1,'currency':'USD','unit':'billion'}
    r['reported_fiscal_period']='FY2020Q4'
    r['guidance']={'target_fiscal_period':'FY2021Q1','currency':'USD','unit':'million','lower':108,'midpoint':110,'upper':112}
    r['api_accepted_at_raw']=r['api_accepted_original']
    with pytest.raises(ValueError,match='USD million'):
        normalize_record(r,as_of='2020-06-01')

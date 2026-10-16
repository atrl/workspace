import numpy as np
import pandas as pd
from src.data.etf_quarterly_profit_diffusion import bounded_change,prepare_facts,domestic_quarters,latest_quarter_pair,aggregate_issuer_rows


def test_signed_change_and_unknown_coverage_do_not_renormalize():
    assert bounded_change(0.,0.)==0.
    assert bounded_change(10.,-10.)==1.
    assert bounded_change(-10.,10.)==-1.
    assert np.isnan(bounded_change(np.nan,0.))
    rows=[{'weight':.9,'pair_valid':True,'current_valid':True,'profit_change':1.,'bounded_growth':.5,'current_profit':1.,'quarter_age_days':90,'reason':'paired'}, {'weight':.1,'pair_valid':False,'current_valid':False,'profit_change':np.nan,'bounded_growth':np.nan,'current_profit':np.nan,'quarter_age_days':np.nan,'reason':'no_reported_quarter'}]
    out=aggregate_issuer_rows(rows)
    assert out['pair_coverage_weight']==.9 and out['raw_known_improve_weight']==.9
    assert out['raw_known_bounded_growth']==.45
    assert not out['eligible'] and np.isnan(out['profit_bounded_growth'])


def test_ytd_quarter_and_yoy_use_same_currency_tag_and_known_vintage():
    base={'taxonomy':'us-gaap','tag':'ProfitLoss','unit':'USD','form':'10-Q'}
    rows=[{**base,'start':'2022-01-01','end':'2022-03-31','filed':'2022-05-01','accn':'oldq1','val':10},
          {**base,'start':'2022-01-01','end':'2022-06-30','filed':'2022-08-01','accn':'oldytd','val':30},
          {**base,'start':'2023-01-01','end':'2023-03-31','filed':'2023-05-01','accn':'q1','val':20},
          {**base,'start':'2023-01-01','end':'2023-06-30','filed':'2023-08-01','accn':'ytd','val':60},
          {**base,'start':'2023-01-01','end':'2023-06-30','filed':'2024-08-01','accn':'future','val':1000}]
    calendar=pd.bdate_range('2022-01-01','2025-01-01')
    prepared=prepare_facts(pd.DataFrame(rows),calendar)
    q,p,status=latest_quarter_pair(domestic_quarters(prepared,'2023-08-31','USD'),'2023-08-31')
    assert status=='paired' and q['profit']==40 and p['profit']==20
    assert q['period_start']==pd.Timestamp('2023-04-01')
    assert 'future' not in q['lineage']
    rows[0]['unit']='EUR'
    modified=prepare_facts(pd.DataFrame(rows),calendar)
    q,p,status=latest_quarter_pair(domestic_quarters(modified,'2023-08-31','USD'),'2023-08-31')
    assert p is None and status=='same_channel_previous_year_quarter_missing'


def test_latest_quarter_not_replaced_by_older_comparable_quarter():
    common={'channel':'same','tag_priority':0,'scope':'consolidated','currency':'USD','basis':'us-gaap','origin':'direct_quarter','available_at':pd.Timestamp('2023-11-01'),'lineage':'[]','duration_days':90}
    q=[{**common,'period_start':pd.Timestamp('2023-07-01'),'period_end':pd.Timestamp('2023-09-30'),'profit':10}, {**common,'period_start':pd.Timestamp('2023-04-01'),'period_end':pd.Timestamp('2023-06-30'),'profit':20}, {**common,'period_start':pd.Timestamp('2022-04-01'),'period_end':pd.Timestamp('2022-06-30'),'profit':10}]
    current,prior,reason=latest_quarter_pair(q,'2023-11-30')
    assert current['period_end']==pd.Timestamp('2023-09-30') and prior is None
    assert reason=='same_channel_previous_year_quarter_missing'


def test_conflicting_latest_quarter_is_unknown_barrier_not_older_fallback():
    from src.data.etf_quarterly_profit_diffusion import foreign_quarters
    base={'taxonomy':'us-gaap','tag':'ProfitLoss','unit':'USD','form':'10-Q'}
    rows=[{**base,'start':'2023-07-01','end':'2023-09-30','filed':'2023-11-01','accn':'conflict_a','val':100},
          {**base,'start':'2023-07-01','end':'2023-09-30','filed':'2023-11-01','accn':'conflict_b','val':200},
          {**base,'start':'2023-04-01','end':'2023-06-30','filed':'2023-08-01','accn':'q2','val':30},
          {**base,'start':'2022-04-01','end':'2022-06-30','filed':'2022-08-01','accn':'oldq2','val':20}]
    calendar=pd.bdate_range('2022-01-01','2024-01-01')
    q,prior,reason=latest_quarter_pair(domestic_quarters(prepare_facts(pd.DataFrame(rows),calendar),'2023-11-30','USD'),'2023-11-30')
    assert q['period_end']==pd.Timestamp('2023-09-30') and np.isnan(q['profit']) and prior is None
    assert reason=='conflicting_latest_quarter_vintage'
    foreign=pd.DataFrame([{'ticker':'TSM','period_kind':'quarter','accounting_basis':'Taiwan_IFRS','currency':'TWD','profit_concept':'attributable_to_parent','period_start':pd.Timestamp(x['start']),'period_end':pd.Timestamp(x['end']),'available_at':pd.Timestamp(x['filed'],tz='UTC'),'eligible_session':pd.Timestamp(x['filed'])+pd.offsets.BDay(1),'value':x['val'],'extraction':'reported','component_lineage':'[]'} for x in rows])
    q,prior,reason=latest_quarter_pair(foreign_quarters(foreign,'TSM','2023-11-30'),'2023-11-30')
    assert q['period_end']==pd.Timestamp('2023-09-30') and np.isnan(q['profit']) and prior is None
    assert reason=='conflicting_latest_quarter_vintage'
    # A conflicting newest cumulative period also blocks retreat to an older Q1.
    cumulative=[{**base,'start':'2023-01-01','end':'2023-03-31','filed':'2023-05-01','accn':'q1','val':20},
                {**base,'start':'2022-01-01','end':'2022-03-31','filed':'2022-05-01','accn':'oldq1','val':10},
                {**base,'start':'2023-01-01','end':'2023-06-30','filed':'2023-08-01','accn':'ytd_a','val':60},
                {**base,'start':'2023-01-01','end':'2023-06-30','filed':'2023-08-01','accn':'ytd_b','val':70}]
    q,prior,reason=latest_quarter_pair(domestic_quarters(prepare_facts(pd.DataFrame(cumulative),calendar),'2023-08-31','USD'),'2023-08-31')
    assert q['period_end']==pd.Timestamp('2023-06-30') and np.isnan(q['profit']) and prior is None
    assert reason=='latest_quarter_from_conflicting_YTD'
    # Known annual profit without its nine-month comparator is not a known Q4.
    underivable=[{**base,'start':'2023-07-01','end':'2023-09-30','filed':'2023-11-01','accn':'q3','val':30},
                 {**base,'start':'2022-07-01','end':'2022-09-30','filed':'2022-11-01','accn':'oldq3','val':20},
                 {**base,'start':'2023-01-01','end':'2023-12-31','filed':'2024-02-01','accn':'annual','val':100}]
    longer_calendar=pd.bdate_range('2022-01-01','2024-04-01')
    q,prior,reason=latest_quarter_pair(domestic_quarters(prepare_facts(pd.DataFrame(underivable),longer_calendar),'2024-02-29','USD'),'2024-02-29')
    assert q['period_end']==pd.Timestamp('2023-12-31') and np.isnan(q['profit']) and prior is None
    assert reason=='latest_reported_period_quarter_underivable'

import pandas as pd
from src.data.etf_foreign_quarterly_intake import parse_document,canonicalize,information_state


def test_statement_currency_periods_and_signed_parent_income():
    html=b'''<html><p>Prepared in accordance with Regulations Governing financial reports of the Republic of China.</p><table><tr><td>For the three-month periods ended March 31, 2022 and 2021 (Expressed in Thousands of New Taiwan Dollars, Except for Earnings per Share) For the three-month periods ended March 31, Notes 2022 2021 Operating revenues</td></tr><tr><td>NET INCOME (LOSS) ATTRIBUTABLE TO:</td></tr><tr><td>Stockholders of the parent</td><td>$(1,234,567)</td><td>$2,345,678</td></tr></table></html>'''
    target={'ticker':'UMC','cik':1033767,'accession':'a','filed':'2022-05-01','url':'https://www.sec.gov/example','accepted_at':'2022-05-01T15:00:00Z'}
    events,diagnostics=parse_document(html,target,{'sha256':'fixture'})
    assert not diagnostics
    assert [(e['period_start'],e['period_end'],e['value']) for e in events]==[('2022-01-01','2022-03-31',-1234567000.),('2021-01-01','2021-03-31',2345678000.)]
    assert events[0]['profit_concept']=='attributable_to_parent'


def test_q4_uses_same_basis_known_components_and_state_never_rewinds():
    base={'ticker':'TSM','cik':1046179,'accounting_basis':'Taiwan_IFRS','profit_concept':'attributable_to_parent','currency':'TWD','source_sha256':'fixture','source_url':'https://www.sec.gov/example','extraction':'explicit_statement_parent_row','comparative':False,'eps':None,'shares':None}
    rows=[{**base,'accession':'nine','period_start':'2023-01-01','period_end':'2023-09-30','period_kind':'ytd','value':80.,'filed':'2023-11-01','accepted_at':'2023-11-01T15:00:00Z'},
          {**base,'accession':'annual','period_start':'2023-01-01','period_end':'2023-12-31','period_kind':'annual','value':100.,'filed':'2024-02-01','accepted_at':'2024-02-01T15:00:00Z'},
          {**base,'accession':'future_revision','period_start':'2023-01-01','period_end':'2023-09-30','period_kind':'ytd','value':90.,'filed':'2024-03-01','accepted_at':'2024-03-01T15:00:00Z'}]
    calendar=pd.bdate_range('2023-01-01','2024-04-01')
    events,_=canonicalize(pd.DataFrame(rows),calendar)
    q4=events[events.period_kind.eq('quarter')]
    assert q4.value.tolist()==[20.]
    assert q4.iloc[0].available_at==pd.Timestamp('2024-02-01T15:00:00Z')
    assert 'future_revision' not in q4.iloc[0].component_lineage
    state=information_state(events,calendar)
    known=state[state.instrument.eq('TSM_TAIWAN_IFRS')&state.datetime.eq('2024-03-04')].iloc[0]
    assert known.quarter_parent_income_twd==20.

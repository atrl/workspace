import pandas as pd
from src.data.asml_quarterly_profit import parse_statement,canonical_events


def test_direct_quarter_ni_is_not_ytd_or_eps_and_scale_is_signed():
    html=b'<html>ASML Summary US GAAP Consolidated Statements of Operations Three months ended Six months ended Jun 29, Jun 28, Jun 29, Jun 28, (in millions EUR, except per share data) 2025 2026 2025 2026 Net system sales 1.0 2.0 3.0 4.0 Net income 2,290.3 (2,917.6) 4,645.3 5,674.3 Basic net income per ordinary share 1.1 2.2</html>'
    target={'filed':'2026-07-15','accepted_at':'2026-07-15T14:00:00Z','accession':'a','url':'https://www.sec.gov/example'}
    rows=parse_statement(html,target,{'sha256':'fixture'})
    assert len(rows)==2 and rows[1]['value']==-2917600000.
    assert rows[1]['period_end']==pd.Timestamp('2026-06-28') and rows[1]['nominal_quarter']=='2026Q2'
    assert rows[1]['source_precision_eur']==100000.


def test_fiscal_start_anchor_cannot_use_future_disclosure():
    base={'ticker':'ASML','cik':937966,'currency':'EUR','profit_concept':'consolidated_reported_net_income','accounting_basis':'US_GAAP','source_sha256':'fixture','source_url':'https://www.sec.gov/example','source_row':'fixture','source_precision_eur':100000.}
    raw=pd.DataFrame([{**base,'period_end':pd.Timestamp('2017-12-31'),'nominal_quarter':'2017Q4','value':10.,'filed':'2018-01-17','accepted_at':'2018-01-17T14:00:00Z','accession':'known_anchor'}, {**base,'period_end':pd.Timestamp('2018-04-01'),'nominal_quarter':'2018Q1','value':20.,'filed':'2018-04-18','accepted_at':'2018-04-18T14:00:00Z','accession':'current'}, {**base,'period_end':pd.Timestamp('2017-12-30'),'nominal_quarter':'2017Q4','value':10.,'filed':'2019-01-17','accepted_at':'2019-01-17T14:00:00Z','accession':'future_anchor'}])
    events,_,_=canonical_events(raw,pd.bdate_range('2018-01-01','2019-02-01'))
    assert len(events)==1 and events.iloc[0].period_start==pd.Timestamp('2018-01-01')
    assert events.iloc[0].period_start_source_accession=='known_anchor'

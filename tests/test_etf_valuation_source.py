import pandas as pd
from src.data.etf_valuation_source import inspect_download, original_period_candidates


def test_historical_download_must_not_accept_html_success():
    assert inspect_download(b'<!DOCTYPE html><html>current page</html>')['format'] == 'html_not_historical_workbook'
    payload=b'<ss:Workbook xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"><ss:Worksheet ss:Name="Holdings"><ss:Data>Fund Holdings as of</ss:Data><ss:Data>Oct 14, 2026</ss:Data></ss:Worksheet></ss:Workbook>'
    result=inspect_download(payload)
    assert result['reported_asof'] == ['Oct 14, 2026']
    assert result['valuation_labels'] == []


def test_first_filing_preserves_original_and_detects_conflict():
    base={'ticker':'X','tag':'NetIncomeLoss','unit':'USD','start':'2020-01-01','end':'2020-03-31','form':'10-Q'}
    values=pd.DataFrame([{**base,'filed':'2020-05-01','accn':'original','val':-10}, {**base,'filed':'2021-05-01','accn':'restated','val':99}])
    result=original_period_candidates(values)
    assert result.val.tolist() == [-10]
    conflict=pd.concat([values,pd.DataFrame([{**base,'filed':'2020-05-01','accn':'other','val':-12}])],ignore_index=True)
    assert original_period_candidates(conflict).first_filing_conflict.all()


def test_ttm_inventory_rejects_ytd_double_counting_and_future_filing():
    from src.data.etf_valuation_fundamentals import accession_facts, candidate_income_at
    import json
    facts=[{'start':'2020-01-01','end':'2020-03-31','filed':'2020-05-01','accn':'q1','form':'10-Q','val':1},
           {'start':'2020-01-01','end':'2020-06-30','filed':'2020-08-01','accn':'q2','form':'10-Q','val':3},
           {'start':'2020-01-01','end':'2020-09-30','filed':'2020-11-01','accn':'q3','form':'10-Q','val':6},
           {'start':'2020-01-01','end':'2020-12-31','filed':'2021-02-01','accn':'annual','form':'10-K','val':-2}]
    doc={'cik':1,'entityName':'test','facts':{'us-gaap':{'NetIncomeLoss':{'units':{'USD':facts}}}}}
    _,first=accession_facts(json.dumps(doc))
    assert candidate_income_at(first,'2021-01-31') is None
    result=candidate_income_at(first,'2021-02-02')
    assert result['income']==-2 and result['method']=='annual_report'


def test_monetary_ttm_uses_ytd_alignment_and_no_future_restatement():
    from src.data.etf_valuation_ttm import prepare_income, monetary_ttm_at
    base={'taxonomy':'us-gaap','tag':'NetIncomeLoss','unit':'USD','form':'10-Q'}
    rows=[{**base,'start':'2022-01-01','end':'2022-12-31','val':100,'filed':'2023-02-01','accn':'annual'},
          {**base,'start':'2022-01-01','end':'2022-09-30','val':60,'filed':'2022-11-01','accn':'old_ytd'},
          {**base,'start':'2023-01-01','end':'2023-09-30','val':90,'filed':'2023-11-01','accn':'new_ytd'},
          {**base,'start':'2022-01-01','end':'2022-12-31','val':500,'filed':'2024-02-01','accn':'later_revision'}]
    prepared=prepare_income(pd.DataFrame(rows),pd.bdate_range('2022-01-01','2024-04-01'))
    result=monetary_ttm_at(prepared,'2023-11-02')
    assert result['income']==130
    assert 'later_revision' not in result['component_accessions']
    assert monetary_ttm_at(prepared,'2023-11-01') is None
    # A prior YTD with a different fiscal start cannot be used as a denominator.
    rows[1]['start']='2022-02-01'
    assert monetary_ttm_at(prepare_income(pd.DataFrame(rows),pd.bdate_range('2022-01-01','2024-04-01')),'2023-11-02') is None
    # ProfitLoss is admitted only under an explicit scope and cannot bridge tags.
    rows[1]['start']='2022-01-01'
    for row in rows:row['tag']='ProfitLoss'
    calendar=pd.bdate_range('2022-01-01','2024-04-01')
    assert monetary_ttm_at(prepare_income(pd.DataFrame(rows),calendar),'2023-11-02') is None
    assert monetary_ttm_at(prepare_income(pd.DataFrame(rows),calendar,['us-gaap:ProfitLoss']),'2023-11-02')['income']==130
    rows[2]['tag']='NetIncomeLoss'
    assert monetary_ttm_at(prepare_income(pd.DataFrame(rows),calendar,['us-gaap:ProfitLoss','us-gaap:NetIncomeLoss']),'2023-11-02') is None

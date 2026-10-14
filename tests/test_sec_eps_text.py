import pytest
from src.data.sec_eps_text import quarterly_filing_identity,extract_sec_eps_text


def test_schema_aliases_and_gaap_quarter_isolated_from_non_gaap_and_annual():
    a={'reported_fiscal_period':'FY2020Q4','accession':'0001045810-20-000007','event_role':'quarterly_earnings_release'}
    b={'reported_fiscal_quarter':'FY2020Q4','accession':a['accession'],'event_role':'initial'}
    assert quarterly_filing_identity(a)==quarterly_filing_identity(b)
    text='https://www.sec.gov/Archives/edgar/data/1045810/000104581020000007/q4fy20pr.htm\nL1: GAAP\nL2: Q4 FY20 | Q3 FY20 | Q4 FY19\nL3: Diluted earnings per share | $1.53 | $1.45 | $0.92\nL4: Non-GAAP\nL5: Diluted earnings per share | $1.89 | $1.78 | $0.80\nL6: Diluted earnings per share | $4.52 | $6.63'
    assert extract_sec_eps_text(text,quarter='FY2020Q4',accession=a['accession'])['current_eps']=='1.53'
    with pytest.raises(ValueError,match='columns'):extract_sec_eps_text(text,quarter='FY2020Q3',accession=a['accession'])
    with pytest.raises(ValueError,match='not an original'):quarterly_filing_identity({**a,'event_role':'update'})

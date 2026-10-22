import pandas as pd

from src.data.inventory_sales_original_value import _comparable_pairs, _original_fact


def test_original_xml_requires_own_undimensioned_period_and_currency():
    xml = b'''<xbrl xmlns="http://www.xbrl.org/2003/instance"
        xmlns:ifrs-full="http://xbrl.ifrs.org/taxonomy/2017-03-09/ifrs-full"
        xmlns:xbrldi="http://xbrl.org/2006/xbrldi">
      <context id="own"><entity><identifier>0001046179</identifier></entity>
        <period><startDate>2017-01-01</startDate><endDate>2017-12-31</endDate></period></context>
      <context id="other"><entity><identifier>0001046179</identifier>
        <segment><xbrldi:explicitMember dimension="ifrs-full:SegmentsAxis">Other</xbrldi:explicitMember></segment></entity>
        <period><startDate>2017-01-01</startDate><endDate>2017-12-31</endDate></period></context>
      <unit id="TWD"><measure>iso4217:TWD</measure></unit>
      <ifrs-full:Revenue contextRef="own" unitRef="TWD">100000000</ifrs-full:Revenue>
      <ifrs-full:Revenue contextRef="other" unitRef="TWD">999000000</ifrs-full:Revenue>
    </xbrl>'''
    spec = {"cik": 1046179, "unit": "TWD", "period_start": "2017-01-01",
            "period_end": "2017-12-31", "fact_format": "original_XML"}
    assert _original_fact(xml, spec, "ifrs-full:Revenue", "revenue") == ({100000000}, {"own"})


def test_original_inline_scale_and_exact_context():
    html = b'''<html><body><xbrli:context id="own"><xbrli:entity>
      <xbrli:identifier>0001046179</xbrli:identifier></xbrli:entity>
      <xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period>
      </xbrli:context><xbrli:unit id="TWD"><xbrli:measure>iso4217:TWD</xbrli:measure></xbrli:unit>
      <ix:nonFraction name="ifrs-full:Inventories" contextRef="own" unitRef="TWD" scale="6">2,500</ix:nonFraction>
    </body></html>'''
    spec = {"cik": 1046179, "unit": "TWD", "period_start": "2024-01-01",
            "period_end": "2024-12-31", "fact_format": "original_inline"}
    assert _original_fact(html, spec, "ifrs-full:Inventories", "inventory") == ({2500000000}, {"own"})


def test_merger_and_accounting_break_pairs_stay_excluded():
    rows = []
    for ticker, years in {"ASX": [2017, 2018, 2019, 2020], "TSM": [2017, 2018, 2019]}.items():
        for year in years:
            rows.append({"ticker": ticker, "cik": 1122411 if ticker == "ASX" else 1046179,
                         "unit": "TWD", "fiscal_year": year,
                         "period_start": pd.Timestamp(f"{year}-01-01"),
                         "period_end": pd.Timestamp(f"{year}-12-31"),
                         "accession": f"{ticker}-{year}", "inventory": 100 + year,
                         "revenue": 1000 + year, "available_session": pd.Timestamp(f"{year+1}-05-01"),
                         "status": "original_own_period_qualified"})
    plan = {"source_rules": {"excluded_growth_pairs": [["ASX", 2018], ["ASX", 2019], ["NVDA", 2019]]}}
    pairs = _comparable_pairs(pd.DataFrame(rows), plan)
    assert set(zip(pairs.ticker, pairs.fiscal_end.dt.year)) == {("ASX", 2020), ("TSM", 2018), ("TSM", 2019)}

from src.data.delayed_gross_margin import extract_gross_margin


def test_percentage_layouts_keep_gaap_and_original_quarter():
    prefix='https://www.sec.gov/Archives/edgar/data/1045810/000104581020000007/q4fy20pr.htm\nL1: GAAP\nL2: Q4 FY20 | Q3 FY20 | Q4 FY19\n'
    suffix='\nL4: Diluted earnings per share | $1.53 | $1.45 | $0.92\nL5: Non-GAAP\nL6: Gross margin | 99.0% | 98.0% | 97.0%'
    for row in ['Gross margin | 60.5% | 59.4% | 55.1%', 'Gross margin | 60.5 | % | 59.4 | % | 55.1 | %']:
        x=extract_gross_margin(prefix+'L3: '+row+suffix,quarter='FY2020Q4',accession='0001045810-20-000007')
        assert (x['gross_margin_pct'],x['prior_year_margin_pct'])==('60.5','55.1')

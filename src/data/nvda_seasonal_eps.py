"""Narrow issuer-release EPS extraction; no consensus or latest restatement feed."""
from decimal import Decimal
import re
from bs4 import BeautifulSoup
import pandas as pd


def extract_q2_eps(html,*,fiscal_year):
    soup=BeautifulSoup(html,'html.parser');title=soup.find('h1')
    if title is None or f'Second Quarter Fiscal {fiscal_year}' not in title.get_text(' ',strip=True):raise ValueError('unexpected release title')
    dates=soup.select('.article-date')
    if len(dates)!=1:raise ValueError('ambiguous publication date')
    released=pd.to_datetime(dates[0].get_text(' ',strip=True),format='%B %d, %Y')
    tables=[t for t in soup.find_all('table') if t.get_text(' ',strip=True).startswith('GAAP ') and f'Q2 FY{str(fiscal_year)[-2:]}' in t.get_text(' ',strip=True)]
    if len(tables)!=1:raise ValueError('ambiguous GAAP summary table')
    table=tables[0];text=table.get_text(' ',strip=True)
    if f'Q2 FY{str(fiscal_year-1)[-2:]}' not in text:raise ValueError('missing comparable prior-year quarter')
    rows=[tr.get_text(' ',strip=True) for tr in table.find_all('tr') if tr.get_text(' ',strip=True).lower().startswith('diluted earnings per share')]
    if len(rows)!=1:raise ValueError('ambiguous diluted EPS row')
    values=re.findall(r'\$\s*(\(?\d+(?:\.\d+)?\)?)',rows[0])
    if len(values)!=3:raise ValueError('expected current, prior-quarter and prior-year EPS')
    values=[str(-Decimal(x[1:-1]) if x.startswith('(') else Decimal(x)) for x in values]
    body=soup.get_text(' ',strip=True)
    headline=re.findall(r'(?<![-A-Za-z])GAAP earnings per diluted share (?:for the quarter were|was) \$(\d+(?:\.\d+)?)',body)
    if len(headline)!=1 or Decimal(headline[0])!=Decimal(values[0]):raise ValueError('GAAP headline/table EPS mismatch')
    return {'fiscal_quarter':f'FY{fiscal_year}Q2','release_date':str(released.date()),'current_eps':values[0],'prior_quarter_comparative_eps':values[1],'prior_year_comparative_eps':values[2],'accounting_basis':'GAAP','share_basis':'diluted','unit':'USD_per_share_as_displayed_in_this_release','precise_publication_clock_verified':False,'first_vintage_bytes_verified':False}


def split_bridge(previous,current,*,split_ratio):
    quarters=[re.fullmatch(r'FY(\d{4})Q2',x['fiscal_quarter']) for x in [previous,current]]
    if any(x is None for x in quarters) or int(quarters[1].group(1))!=int(quarters[0].group(1))+1:raise ValueError('adjacent fiscal-year same-quarter pair required')
    if any(x.get('accounting_basis')!='GAAP' or x.get('share_basis')!='diluted' for x in [previous,current]):raise ValueError('consistent GAAP diluted EPS required')
    ratio=Decimal(str(split_ratio))
    if ratio<=0:raise ValueError('positive split ratio required')
    old=Decimal(previous['current_eps']);now=Decimal(current['current_eps']);comparison=Decimal(current['prior_year_comparative_eps'])
    def half_unit(value):return Decimal(10)**Decimal(value).as_tuple().exponent/2
    old_half=half_unit(previous['current_eps']);comparison_half=half_unit(current['prior_year_comparative_eps']);adjusted=old/ratio
    overlap=max((old-old_half)/ratio,comparison-comparison_half)<=min((old+old_half)/ratio,comparison+comparison_half)
    return {'previous_quarter':previous['fiscal_quarter'],'current_quarter':current['fiscal_quarter'],'split_ratio':str(ratio),'naive_eps_change':str(now-old),'prior_eps_on_current_split_basis':str(adjusted),'split_normalized_change':str(now-adjusted),'current_release_comparative_change':str(now-comparison),'comparative_rounding_intervals_overlap':bool(overlap),'scope':'input semantic diagnostic, not market surprise or SUE score'}

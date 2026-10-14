"""Narrow issuer-release EPS extraction; no consensus or latest restatement feed."""
from decimal import Decimal
import re
from bs4 import BeautifulSoup
import pandas as pd


def extract_quarter_eps(html,*,fiscal_year,quarter):
    words={1:'First',2:'Second',3:'Third',4:'Fourth'}
    if quarter not in words:raise ValueError('invalid fiscal quarter')
    soup=BeautifulSoup(html,'html.parser');title=soup.find('h1')
    expected=rf"{words[quarter]} Quarter (?:and )?Fiscal (?:Year )?{fiscal_year}\b"
    if title is None or not re.search(expected,title.get_text(' ',strip=True),re.I):raise ValueError('unexpected release title')
    dates=soup.select('.article-date')
    if len(dates)!=1:raise ValueError('ambiguous publication date')
    released=pd.to_datetime(dates[0].get_text(' ',strip=True),format='%B %d, %Y')
    marker=f'Q{quarter}FY{str(fiscal_year)[-2:]}';compact=lambda value:re.sub(r'\s+','',value)
    tables=[t for t in soup.find_all('table') if t.get_text(' ',strip=True).startswith('GAAP ') and marker in compact(t.get_text(' ',strip=True))]
    if len(tables)!=1:raise ValueError('ambiguous GAAP quarterly summary table')
    table=tables[0];text=table.get_text(' ',strip=True)
    if f'Q{quarter}FY{str(fiscal_year-1)[-2:]}' not in compact(text):raise ValueError('missing comparable prior-year quarter')
    previous_quarter=quarter-1 if quarter>1 else 4
    previous_year=fiscal_year if quarter>1 else fiscal_year-1
    headers=[marker,f'Q{previous_quarter}FY{str(previous_year)[-2:]}',f'Q{quarter}FY{str(fiscal_year-1)[-2:]}']
    positions=[compact(text).find(header) for header in headers]
    if min(positions)<0 or positions!=sorted(set(positions)):raise ValueError('unexpected quarter column order')
    rows=[]
    for tr in table.find_all('tr'):
        line=tr.get_text(' ',strip=True)
        if line.lower().startswith('non-gaap'):break
        if line.lower().startswith(('diluted earnings per share','earnings per share')):rows.append(line)
    if len(rows)!=1:raise ValueError('ambiguous diluted EPS row')
    values=re.findall(r'\$\s*(\(?\d+(?:\.\d+)?\)?)',rows[0])
    if len(values)!=3:raise ValueError('expected current, prior-quarter and prior-year EPS')
    values=[str(-Decimal(x[1:-1]) if x.startswith('(') else Decimal(x)) for x in values]
    body=soup.get_text(' ',strip=True)
    if quarter==4:
        patterns=[r'For the quarter, GAAP earnings per diluted share (?:was|were) \$\s*(\d+(?:\.\d+)?)',r'(?<![-A-Za-z])GAAP earnings per diluted share for the quarter (?:was|were) \$\s*(\d+(?:\.\d+)?)']
    else:patterns=[r'(?<![-A-Za-z])GAAP earnings per diluted share (?:for the quarter )?(?:was|were) \$\s*(\d+(?:\.\d+)?)']
    headline=[x for pattern in patterns for x in re.findall(pattern,body,re.I)]
    if rows[0].lower().startswith('earnings per share') and not headline:raise ValueError('generic EPS row needs explicit diluted headline')
    if headline and any(Decimal(x)!=Decimal(values[0]) for x in headline):raise ValueError('GAAP headline/table EPS mismatch')
    return {'fiscal_quarter':f'FY{fiscal_year}Q{quarter}','release_date':str(released.date()),'current_eps':values[0],'prior_quarter_comparative_eps':values[1],'prior_year_comparative_eps':values[2],'accounting_basis':'GAAP','share_basis':'diluted','unit':'USD_per_share_as_displayed_in_this_release','headline_crosschecked':bool(headline),'early_release_disclosure':all(term in body.lower() for term in ['preliminary draft','inadvertently','in advance of trading']),'precise_publication_clock_verified':False,'first_vintage_bytes_verified':False}


def extract_q2_eps(html,*,fiscal_year):
    result=extract_quarter_eps(html,fiscal_year=fiscal_year,quarter=2)
    if not result['headline_crosschecked']:raise ValueError('missing Q2 headline crosscheck')
    return result


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

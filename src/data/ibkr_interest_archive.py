"""Observed dates from archived official tables, including rolling current views."""
from io import StringIO
from urllib.parse import urlparse,parse_qs
import pandas as pd
from src.data.ibkr_interest_source import parse_benchmark_month


def parse_archived_rates(payload,*,original_url,as_of):
    if payload.lstrip().startswith('Date'):
        table=pd.read_csv(StringIO(payload),sep=r'\s+',dtype=str)
    else:
        tables=pd.read_html(StringIO(payload),keep_default_na=False)
        matches=[t for t in tables if 'Date' in t.columns and 'USD' in t.columns]
        if len(matches)!=1:raise ValueError('ambiguous archived USD table')
        table=matches[0]
    if table.empty:return pd.DataFrame(columns=['datetime','benchmark_rate','prior_day_flag'])
    dates=pd.to_datetime(table.Date.astype(str),format='%Y%m%d',errors='raise')
    requested=parse_qs(urlparse(original_url).query).get('date',[None])[0]
    if requested and not dates.dt.strftime('%Y%m').eq(requested).all():raise ValueError('archived requested month differs')
    parts=[]
    for month,indices in dates.groupby(dates.dt.strftime('%Y%m')).groups.items():
        # Default current views can span month boundaries; dates, not headings, govern.
        text='Date USD\n'+'\n'.join(str(table.loc[i,'Date'])+' '+str(table.loc[i,'USD']) for i in indices)
        parts.append(parse_benchmark_month(text,month=month,as_of=as_of))
    result=pd.concat(parts,ignore_index=True).sort_values('datetime')
    if result.datetime.duplicated().any():raise ValueError('duplicate archived date')
    return result

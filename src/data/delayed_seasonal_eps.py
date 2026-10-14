"""Date-buffered seasonal EPS reconstruction, not an announcement-time feed."""
from decimal import Decimal
import re
import numpy as np
import pandas as pd
from src.data.nvda_seasonal_eps import split_ratio_between_reports

FEATURE='nvda_seasonal_eps_score'
FLAG='nvda_seasonal_eps_ready'


def delayed_filing_availability(reports, calendar):
    calendar=pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:raise ValueError('ordered unique Qlib calendar required')
    rows=reports.sort_values('quarter').to_dict('records')
    sequence=[4*int(x['quarter'][2:6])+int(x['quarter'][-1]) for x in rows]
    if any(b-a!=1 for a,b in zip(sequence,sequence[1:])):raise ValueError('continuous fiscal quarters required')
    result=[]
    for row in rows:
        dates=[]
        for key in ['release_date','filing_date','api_accepted_raw','index_accepted_raw']:
            value=row.get(key)
            if value is None or pd.isna(value):
                if key in ['release_date','filing_date','api_accepted_raw']:raise ValueError('required date missing')
                continue
            if not re.match(r'^\d{4}-\d{2}-\d{2}(?:$|[T ])',str(value)):raise ValueError('unrecognized date')
            pd.Timestamp(value)
            dates.append(pd.Timestamp(str(value)[:10]))
        last_date=max(dates);pos=calendar.searchsorted(last_date,side='right')
        available=calendar[pos] if pos<len(calendar) else pd.NaT
        result.append({'quarter':row['quarter'],'release_date':row['release_date'],'latest_literal_date':str(last_date.date()),'available_date':available})
    return pd.DataFrame(result)


def delayed_seasonal_events(reports, splits, calendar):
    available=delayed_filing_availability(reports,calendar)
    rows=reports.sort_values('quarter').to_dict('records');result=[]
    for t,row in enumerate(rows):
        score=np.nan;change=np.nan;scale=np.nan
        if t>=12:
            levels=[]
            for old in rows[t-12:t+1]:
                ratio=Decimal('1') if old is row else split_ratio_between_reports(old,row,splits)
                levels.append(float(Decimal(str(old['current_eps']))/ratio))
            differences=np.asarray(levels[4:])-np.asarray(levels[:-4])
            change=float(differences[-1]);scale=float(differences[:-1].std(ddof=1))
            if scale>0 and np.isfinite(scale):score=change/scale
        result.append({**available.iloc[t].to_dict(),'seasonal_change_current_share_basis':change,'past_eight_change_std':scale,FEATURE:score,'prior_innovations_used':8 if t>=12 else 0})
    return pd.DataFrame(result)


def align_delayed_events(events, signal_index, trade_dates):
    dates=pd.DatetimeIndex(trade_dates);known=events[events.available_date.notna()].sort_values('available_date')
    if known.available_date.duplicated().any():raise ValueError('ambiguous same-session events')
    positions=pd.DatetimeIndex(known.available_date).searchsorted(dates,side='right')-1
    values=np.full(len(dates),np.nan)
    for i,pos in enumerate(positions):
        if pos>=0 and not pd.isna(dates[i]):values[i]=known.iloc[pos][FEATURE]
    return pd.DataFrame({FEATURE:values,FLAG:np.isfinite(values).astype(float)},index=signal_index)

"""Validate observed five-minute context without extending missing sessions."""
import numpy as np
import pandas as pd


def audit_context(frame,schedule,*,symbols):
    if frame.duplicated(['symbol','timestamp']).any():raise ValueError('duplicate context bars')
    expected_dates={str(pd.Timestamp(day).date()) for day in schedule.index}
    if not frame.date.isin(expected_dates).all() or not frame.symbol.isin(symbols).all():raise ValueError('unexpected context date or symbol')
    groups=dict(tuple(frame.groupby(['symbol','date'],sort=False)));rows=[]
    for day,hours in schedule.iterrows():
        date=str(day.date());expected=pd.date_range(pd.Timestamp(hours.open)+pd.Timedelta(minutes=5),pd.Timestamp(hours.close),freq='5min').tz_convert('America/New_York')
        for symbol in symbols:
            part=groups.get((symbol,date))
            stamps=pd.DatetimeIndex([]) if part is None else pd.DatetimeIndex(part.timestamp)
            rows.append({'symbol':symbol,'date':date,'expected_bars':len(expected),'observed_bars':len(stamps),'missing_bars':len(expected.difference(stamps)),'unexpected_bars':len(stamps.difference(expected)),'invalid_rows':0 if part is None else int((~part.price_valid).sum())})
    return pd.DataFrame(rows)

"""Read only completed observed bars; never repair a missing prefix by interpolation."""
import numpy as np
import pandas as pd


def closed_window_return(part, *, day, cutoff, frequency_minutes):
    day=pd.Timestamp(day).normalize()
    end=day+pd.Timedelta(hours=int(cutoff[:2]),minutes=int(cutoff[3:]))
    start=day+pd.Timedelta(hours=9,minutes=30)
    expected=pd.date_range(start+pd.Timedelta(minutes=frequency_minutes),end,freq=f'{frequency_minutes}min')
    observed=part.loc[part.datetime.gt(start)&part.datetime.le(end)].sort_values('datetime')
    if not len(expected) or not pd.DatetimeIndex(observed.datetime).equals(expected):return np.nan
    if not observed.price_valid.eq(1).all() or not np.isfinite(observed[['open','close','volume']]).all().all():return np.nan
    if not observed[['open','close']].gt(0).all().all() or observed.volume.iloc[0]<=0 or observed.volume.iloc[-1]<=0:return np.nan
    return float(observed.close.iloc[-1]/observed.open.iloc[0]-1)

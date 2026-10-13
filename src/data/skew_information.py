"""Lag on full verified market calendar, retaining unknown SKEW observations."""
import numpy as np
import pandas as pd


def lagged_skew(native,calendar,decision_dates,*,lag,eligible_after):
    calendar=pd.DatetimeIndex(calendar)
    if lag<2 or calendar.has_duplicates or not calendar.is_monotonic_increasing:raise ValueError('invalid source lag/calendar')
    value=native['$skew'].reindex(calendar)
    observed=native['$observed'].reindex(calendar).eq(1)
    value=value.where(observed & (calendar>=pd.Timestamp(eligible_after)))
    source_dates=pd.Series(calendar,index=calendar).shift(lag)
    available=pd.Series(calendar,index=calendar).shift(lag-1)+pd.Timedelta(hours=17)
    result=pd.DataFrame({'skew_scaled':(value.shift(lag)-100)/10,'source_date':source_dates,'assumed_available_at':available},index=calendar).reindex(pd.DatetimeIndex(decision_dates))
    result['skew_known']=result.skew_scaled.notna().astype(float)
    valid=result.source_date.notna()
    if not (result.loc[valid,'assumed_available_at']<result.index[valid]+pd.Timedelta(hours=9,minutes=46)).all():raise ValueError('source unavailable at decision')
    return result

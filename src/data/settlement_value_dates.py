"""Standard value dates from an explicit settlement-service calendar.

This module never substitutes an exchange trading calendar or infers holidays.
Schedules describe standard settlement, not broker-confirmed value dates.
"""
import numpy as np
import pandas as pd


def standard_value_dates(trade_dates, *, settlement_open, cycle_by_effective_date):
    days=pd.DatetimeIndex(settlement_open.index)
    trades=pd.DatetimeIndex(pd.to_datetime(trade_dates))
    effective=pd.DatetimeIndex(cycle_by_effective_date.index)
    for dates in [days,trades,effective]:
        if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()):
            raise ValueError('explicit naive local dates required')
    if not len(days) or not days.is_unique or not days.equals(pd.date_range(days[0],days[-1],freq='D')):
        raise ValueError('complete settlement calendar span required')
    if str(settlement_open.dtype) not in ['bool','boolean']:
        raise ValueError('settlement status must be boolean or explicitly unknown')
    if not len(effective) or not effective.is_unique or not effective.is_monotonic_increasing:
        raise ValueError('ordered settlement-cycle effective dates required')
    lags=cycle_by_effective_date.to_numpy(float)
    if not np.isfinite(lags).all() or (lags<1).any() or not np.equal(lags,np.floor(lags)).all():
        raise ValueError('settlement-cycle lags must be positive integers')
    result=[]
    for trade in trades:
        if trade<days[0] or trade>days[-1]:raise ValueError('trade outside settlement calendar coverage')
        policy=effective.searchsorted(trade,side='right')-1
        if policy<0:raise ValueError('needs_data: no cycle applicable on trade date')
        remaining=int(lags[policy]);settled=None
        for position in range(days.searchsorted(trade,side='right'),len(days)):
            status=settlement_open.iloc[position]
            if pd.isna(status):raise ValueError('needs_data: unknown settlement service date')
            if status:remaining-=1
            if remaining==0:
                settled=days[position];break
        if settled is None:raise ValueError('needs_data: calendar ends before settlement')
        result.append(settled)
    return pd.DatetimeIndex(result,name='standard_value_date')

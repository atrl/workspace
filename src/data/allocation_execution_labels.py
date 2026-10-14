"""Paired economic-return labels; preserve original classification semantics."""
import numpy as np
import pandas as pd
from src.model.minute_pit_allocator import TARGETS


def paired_execution_returns(primary,timing,execution):
    prices=execution.set_index('signal_date').reindex(primary.index)
    if prices.index.has_duplicates:raise ValueError('duplicate execution signal dates')
    covered=prices.trade_date.notna()&prices.exit_date.notna()
    for name in ['trade_date','exit_date']:
        if not pd.to_datetime(prices.loc[covered,name]).equals(pd.to_datetime(primary.loc[covered,name])):
            raise ValueError('execution calendar disagrees: '+name)
    old=primary.copy();new=primary.copy();valid=covered.copy()
    for asset in ['soxl','soxx']:
        start=prices[asset+'_open'];end=prices[asset+'_exit_open'];ok=np.isfinite(start)&np.isfinite(end)&start.gt(0)&end.gt(0)
        valid &= ok
        new[asset+'_forward_open_return']=(end/start-1).where(ok)
    available=pd.to_datetime(prices.exit_date).dt.tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=49)
    if 'label_available_at' in prices:
        declared=pd.to_datetime(prices.label_available_at)
        declared=declared.dt.tz_localize('America/New_York') if declared.dt.tz is None else declared.dt.tz_convert('America/New_York')
        if not declared.loc[covered].equals(available.loc[covered]):raise ValueError('execution label availability differs')
    h5=pd.to_datetime(timing.label_available_at)
    valid &= h5.notna()&available.le(h5)
    valid &= np.isfinite(old[list(TARGETS)].to_numpy(float)).all(axis=1)&np.isfinite(new[list(TARGETS)].to_numpy(float)).all(axis=1)
    old.loc[~valid,list(TARGETS)]=np.nan;new.loc[~valid,list(TARGETS)]=np.nan
    pd.testing.assert_frame_equal(old[list(TARGETS[2:])],new[list(TARGETS[2:])])
    return old,new,pd.DataFrame({'paired_label_known':valid,'execution_return_available_at':available,'retained_h5_available_at':h5},index=primary.index)

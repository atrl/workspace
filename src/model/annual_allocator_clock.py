"""Assign a refit by decision year, never by the feature's earlier signal date."""
import pandas as pd


def annual_decision_groups(timing, signal_dates):
    dates=pd.DatetimeIndex(signal_dates)
    if not dates.is_unique:raise ValueError('duplicate required signal dates')
    decision=timing.decision_at.reindex(dates)
    if decision.isna().any():raise ValueError('unknown decision clock')
    local=pd.to_datetime(decision).dt.tz_convert('America/New_York')
    groups={}
    for year in sorted(local.dt.year.unique()):
        cutoff=pd.Timestamp(year=int(year),month=1,day=1,tz='America/New_York')
        selected=dates[local.dt.year.eq(year).to_numpy()]
        if not local.loc[selected].gt(cutoff).all():raise ValueError('refit must precede decisions')
        groups[int(year)]={'cutoff':cutoff,'signal_dates':selected}
    return groups

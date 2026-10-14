"""Observed opening-state inputs aligned to the actual 09:46 decision."""
import numpy as np
import pandas as pd

FEATURES=['opening_soxl_gap','opening_soxl_return15','opening_soxx_gap','opening_soxx_return15']
FLAG='opening_state_ready'


def opening_state_features(primary,execution,quotes):
    frame=execution.set_index('signal_date').reindex(primary.index)
    result=pd.DataFrame(index=primary.index)
    for symbol in ['SOXL','SOXX']:
        asset=symbol.lower()
        daily=quotes.xs(symbol,level='instrument')
        opening=daily['$open930'].reindex(pd.DatetimeIndex(frame.trade_date)).set_axis(primary.index)
        valid=daily['$valid930'].reindex(pd.DatetimeIndex(frame.trade_date)).set_axis(primary.index).eq(1)
        # Explicit previous signal-date lookup: current session close never enters its own decision.
        closing=execution.set_index('trade_date')['close_'+asset]
        previous=closing.reindex(primary.index).set_axis(primary.index)
        known=frame['known_'+asset]
        ok=valid&opening.gt(0)&previous.gt(0)&known.gt(0)
        result['opening_'+asset+'_gap']=np.log(opening/previous).where(ok)
        result['opening_'+asset+'_return15']=np.log(known/opening).where(ok)
    result[FLAG]=np.isfinite(result[FEATURES]).all(axis=1).astype(float)
    result.loc[result[FLAG].eq(0),FEATURES]=np.nan
    timing=pd.DataFrame(index=primary.index)
    timing['decision_at']=pd.to_datetime(frame.trade_date).dt.tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=46)
    timing['latest_bar_end']=timing.decision_at-pd.Timedelta(minutes=1)
    timing['feature_available_at']=timing.decision_at
    return result,timing

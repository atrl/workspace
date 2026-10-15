import numpy as np
import pandas as pd
from src.analysis.fullminute_frozen_position_risk import value_path


def test_fill_boundary_terminal_exclusion_and_nonsynchronous_extrema_bound():
    index=pd.to_datetime(['2023-01-04 09:48','2023-01-04 09:49','2023-01-04 09:50'])
    quotes=pd.DataFrame(index=index)
    for symbol in ['SOXL','SOXX']:
        quotes[symbol+'_open']=[100,100,1]
        quotes[symbol+'_close']=[100,100,1]
        quotes[symbol+'_high']=[120,120,99999]
        quotes[symbol+'_low']=[80,80,.0001]
    events=pd.DataFrame({'stamp':[pd.Timestamp('2023-01-04 09:47')],'soxl':[.5],'soxx':[.4],'cash':[9.]}).set_index('stamp')
    known=pd.DataFrame(columns=['SOXL','SOXX'],index=pd.DatetimeIndex([]))
    path,summary,episode=value_path(quotes,events,known,start='2023-01-04 09:47',end='2023-01-04 09:49',initial_cash=100.)
    assert path.loc[path.kind.eq('minute_bar'),'soxl'].eq(.5).all()
    assert summary['minute_bar_count']==2 and summary['terminal_nav']==9.9
    # Terminal uses nextbar OPEN only; laterhigh99999/low.0001 must not enter envelope.
    assert path.upper.max()==117. and path.lower.min()==9.9
    assert path.loc[path.kind.eq('open_after_fill')].iloc[0]['nav']==99.
    assert summary['upper_bound_drawdown']>=summary['observed_drawdown']
    assert 'not simultaneous' in episode.loc[episode.method.eq('upper_bound'),'extrema_interpretation'].iloc[0]

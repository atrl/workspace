import pandas as pd
import pytest
from src.backtest.frozen_open_trace import FrozenOpenTrace


def test_trace_preserves_hold_and_rejects_unrepresented_recovery():
    t=pd.DataFrame({'signal_date':pd.to_datetime(['2024-01-02','2024-01-03']),'trade_date':pd.to_datetime(['2024-01-03','2024-01-04']),'soxl_weight':[.2,.2],'soxx_weight':[.5,.5],'cash_weight':[.3,.3]})
    l=t[['signal_date','trade_date']].assign(execute_open_target=[True,False],recovery_action=False)
    model=FrozenOpenTrace(t,l)
    assert model.target(t.iloc[0],None,0,None,0).tolist()==[.2,.5,.3]
    assert model.target(t.iloc[1],None,1e9,None,1e9) is None
    l.loc[1,'recovery_action']=True
    with pytest.raises(ValueError,match='recovery'):FrozenOpenTrace(t,l)
    row=t.iloc[0].copy();row.trade_date=pd.Timestamp('2024-01-05')
    with pytest.raises(ValueError,match='date changed'):model.target(row,None,0,None,0)

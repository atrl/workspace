"""Material accounting contract: both execution slots and native residual."""
import numpy as np
import pandas as pd
from src.analysis.decision_accounting import attribute_account


def test_recovery_cost_and_relative_wealth_reconcile():
    day=pd.Timestamp('2023-01-04')
    frame=pd.DataFrame([dict(trade_date=day,signal_date=day-pd.Timedelta(days=1),soxl_open=100.,soxx_open=100.,soxl_exit_open=110.,soxx_exit_open=105.,soxl_recovery=105.,soxx_recovery=102.)])
    orders=pd.DataFrame([dict(trade_date=day,slot=slot,symbol=symbol,sell=sell,adjusted_amount=amount,adjusted_price=price,total=fee) for slot,symbol,sell,amount,price,fee in [('open','SOXL',False,20,100,2),('open','SOXX',False,30,100,3),('recovery','SOXL',True,10,105,1),('recovery','SOXX',False,10,102,1)]])
    native_residual=1e-9
    path=pd.DataFrame([dict(trade_date=day,strategy_return=.0323+native_residual,equity=1.0323+native_residual,ending_soxl_quantity=10.,ending_soxx_quantity=40.,ending_cash=5023.,risk_state=2,execute_open_target=True,recovery_action=True)])
    independent=path.copy();independent['strategy_return']=.0323
    benchmark={'authority':pd.DataFrame([dict(trade_date=day,strategy_return=.05,equity=1.05)])}
    result=attribute_account(frame,{'authority':path,'independent_replay':independent,'orders':orders},benchmark,initial_cash=10000,cash_reference=.15).iloc[0]
    factor=np.log((1.0323+native_residual)/1.05)/(.0323+native_residual-.05)
    np.testing.assert_allclose(result.recovery_gross,-20/10000*factor)
    np.testing.assert_allclose(result.execution_cost,-(5*1.05+2)/10000*factor)
    np.testing.assert_allclose(result.native_reconciliation_residual,native_residual*factor,atol=1e-16)

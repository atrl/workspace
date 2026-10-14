import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.analysis.tiered_action_labels import reset_advantage, Flat25Projection
from src.backtest.ibkr_tiered import TieredFees


def test_both_branch_fees_and_common_reset_are_isolated():
    tariff=json.loads((Path(__file__).resolve().parents[1]/'research/ibkr_tiered_minute_diagnosis_20261013/fee_schedule.json').read_text())['tariff']
    row=pd.Series({'trade_date':pd.Timestamp('2023-01-04'),'soxl_open':100.,'soxx_open':100.,'raw_soxl_open':100.,'raw_soxx_open':100.})
    nxt=row.copy();nxt['trade_date']=pd.Timestamp('2023-01-05')
    args=dict(row=row,next_row=nxt,target=np.array([.2,0.,.8]),reset_target=np.array([0.,0.,1.]))
    fees=TieredFees(tariff,slippage_bps=2)
    value=reset_advantage(np.zeros(2),10000.,fees=fees,**args)
    expected=-sum(fees.quote(shares=20,raw_price=100.,sell=sell,stamp=row.trade_date)['total'] for sell in [False,True])/10000
    np.testing.assert_allclose(value,expected,atol=1e-15)
    assert fees.monthly_shares=={}
    legacy=reset_advantage(np.zeros(2),10000.,fees=Flat25Projection(tariff,slippage_bps=0),**args)
    np.testing.assert_allclose(legacy,-.001,atol=1e-15)
    assert value>legacy
    recovery=row.copy()
    for symbol in ['soxl','soxx']:
        recovery[symbol+'_recovery']=50.
        recovery['raw_'+symbol+'_recovery']=50.
        nxt[symbol+'_open']=50.
        nxt['raw_'+symbol+'_open']=50.
    recovery_args={**args,'row':recovery,'next_row':nxt}
    actual=reset_advantage(np.zeros(2),10000.,fees=fees,entry_slot='recovery',**recovery_args)
    expected=-sum(fees.quote(shares=40,raw_price=50.,sell=sell,stamp=row.trade_date)['total'] for sell in [False,True])/10000
    np.testing.assert_allclose(actual,expected,atol=1e-15)
    assert fees.monthly_shares=={}

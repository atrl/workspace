from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd

from src.backtest.close_rod_raw import prepare_close_permission, fill_increment, close_increment, replay_close_permission_raw
from src.backtest.ibkr_tiered import TieredFees
from src.backtest.native_raw_corporate import native_raw_validate
from src.data import build_qlib_store_from_frame
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec


class ConstantProbability:
    def __init__(self, probability): self.probability=probability
    def predict_proba(self, features):
        return np.tile([1-self.probability,self.probability],(len(features),1))


def test_temporary_whole_share_lot_and_native_own_cash_contract(tmp_path):
    spec=IntradayRecoveryDirectAllocationSpec(root=Path.cwd())
    selection={"action_beta":.6,"minimum_source_alpha_probability":.25,"probability_threshold":.825}
    tariff={"tiers":[{"through_shares":None,"per_share":0.}],"minimum_per_order":0.,"maximum_order_fraction":1.,
        "venue_per_share":0.,"venue_subdollar_fraction":0.,"clearing_per_share":0.,"clearing_maximum_fraction":0.,
        "sec_sale_fraction":0.,"taf_per_sold_share":0.,"taf_maximum_per_trade":0.,"cat_per_share":0.,"commission_pass_through_fraction":0.}
    source=pd.Series({"source_ready":True,"rod_known":True,"rod_positive":True})
    request=prepare_close_permission(pd.Series({"soxl_weight":.006,"soxx_weight":.5,"alpha_probability":.8}),
        opened={"drawdown":.1,"current_beta_cap":.2},q=np.array([1.,20.]),cash=790.,receivable=0.,
        known_prices=np.array([10.,10.]),peak=1200.,history=[],recovery_spec=spec,selection=selection,source=source,require_ROD=True)
    # -0.4 share uses no baseline sale, but beta/raw-risk envelope still binds.
    assert request['requested_buy'].tolist()==[0,30]
    kwargs=dict(requested=request['requested_buy'],prices=np.array([10.,10.]),reserve_prices=np.array([10.,10.]),
        stamp=pd.Timestamp('2023-01-05 15:32'),exit_stamp=pd.Timestamp('2023-01-05 15:59'),cap=request['late_cap'],raw_risk=request['raw_risk'],spec=spec)
    q,cash,orders,lot=fill_increment(np.array([1.,20.]),790.,0.,fees=TieredFees(tariff,slippage_bps=0),**kwargs)
    assert lot.tolist()==[0,28] and (q*np.array([10.,10.])).sum()/(q@np.array([10.,10.])+cash)<=.506
    assert fill_increment(np.array([1.,20.]),0.,790.,fees=TieredFees(tariff,slippage_bps=0),**kwargs)[3].sum()==0
    exited,_,_=close_increment(q,cash,lot=lot,prices=np.array([10.,11.]),fees=TieredFees(tariff,slippage_bps=0),stamp=pd.Timestamp('2023-01-05 15:59'))
    np.testing.assert_array_equal(exited,[1.,20.])

    days=pd.bdate_range('2023-01-04',periods=3)
    frame=pd.DataFrame({'signal_date':pd.bdate_range('2023-01-03',periods=2),'trade_date':days[:-1],'exit_date':days[1:],
        'soxl_weight':.2,'soxx_weight':.6,'cash_weight':.2,'tail_probability':.1,'alpha_probability':.8})
    for asset in ['soxl','soxx']:
        for field in ['ret','range','from_low','volume']:frame[f'early_{field}_{asset}']=0.
    stamps=[day+pd.Timedelta(minutes=minute) for day in days[:-1] for minute in [585,587,632,930,932,959,960]]
    stamps += [days[-1]+pd.Timedelta(minutes=minute) for minute in [585,587]]
    prices=pd.DataFrame(100.,index=pd.DatetimeIndex(stamps),columns=['SOXL','SOXX'])
    prices.loc[prices.index>=days[0]+pd.Timedelta(hours=16)]=80.
    prices.loc[prices.index>=days[1]+pd.Timedelta(minutes=959)]=82.
    late=pd.DataFrame({'actual_close_at':days[:-1]+pd.Timedelta(hours=16),'regular_session':True,
        'source_ready':True,'rod_known':True,'rod_positive':True},index=days[:-1])
    tariff=deepcopy(tariff);tariff.update(minimum_per_order=.35,maximum_order_fraction=.01)
    tariff['tiers']=[{'through_shares':None,'per_share':.0035}]
    plan={'initial_cash':100000.,'tariff':tariff,'slippage_bps':[2]}
    args=dict(point_prices=prices,schedule={},plan=plan,controller={'frozen_open_action_value_model':ConstantProbability(1.),
        'frozen_recovery_model':ConstantProbability(0.)},recovery_spec=spec,action_spec=Stage1OpenActionValueSpec(root=Path.cwd()),
        selection=selection,slippage_bps=2)
    scheduled=replay_close_permission_raw(frame,late_inputs=late,require_ROD=False,**args)
    positive=replay_close_permission_raw(frame,late_inputs=late,require_ROD=True,**args)
    pd.testing.assert_frame_equal(scheduled['orders'],positive['orders'])
    off=late.copy();off.rod_positive=False
    gated=replay_close_permission_raw(frame,late_inputs=off,require_ROD=True,**args)
    from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
    original_args=dict(args)
    original_args['point_prices']=prices.loc[~prices.index.strftime('%H:%M').isin(['15:30','15:32','15:59'])]
    original=replay_stage1_raw(frame,**original_args)
    pd.testing.assert_frame_equal(gated['orders'],original['orders'])
    pd.testing.assert_frame_equal(gated['replay'][original['replay'].columns],original['replay'])
    assert scheduled['late_events'].late_action.tolist()==[False,True]
    assert not gated['late_events'].late_action.any()
    event=scheduled['late_events'].iloc[-1]
    assert event.entry_shares_soxx>0 and event.after_exit_soxx==event.baseline_soxx and event.after_exit_soxl==event.baseline_soxl
    assert event.cash_after>event.cash_before and scheduled['replay'].iloc[-1].equity>gated['replay'].iloc[-1].equity
    guarded=pd.concat([prices,pd.DataFrame([prices.iloc[-1]],index=[prices.index[-1]+pd.Timedelta(minutes=1)])])
    fields=['open','high','low','close','execution_price','raw_execution_price']
    rows=[{'datetime':stamp,'instrument':symbol,**{field:value for field in fields}} for symbol in prices for stamp,value in guarded[symbol].items()]
    provider=tmp_path/'provider'
    build_qlib_store_from_frame(pd.DataFrame(rows),provider_uri=provider,freq='1min',region='us',as_of='2023-01-06',field_map={x:x for x in fields},source='synthetic temporary lot contract')
    native=native_raw_validate(scheduled,provider_uri=provider,point_prices=prices,schedule={},tariff=tariff,cost=2,initial_cash=100000.)
    assert native['audit']['status']=='passed' and native['audit']['max_relative_error']<1e-10

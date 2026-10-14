"""Causal local recovery labels on a reconciled deterministic reference path."""
import numpy as np
import pandas as pd
from src.backtest.ibkr_tiered import TieredFees,settle_tiered_event
from src.analysis.tiered_action_labels import reset_advantage
from src.workflow.stage1_open_action_value import prepare_on_policy_open
from src.workflow.intraday_recovery_direct_allocation import _feature_row,project_legal_weights,FEATURE_COLUMNS


def recovery_training_labels(frame,book,*,plan,recovery_spec,action_spec,selection,slippage_bps):
    frame=frame.reset_index(drop=True);path=book['independent_replay'].reset_index(drop=True)
    if not frame.trade_date.equals(path.trade_date):raise ValueError('reference dates differ')
    if path.recovery_action.any() or not path.execute_open_target.all():raise ValueError('deterministic always-open reference without recovery required')
    targets=book['targets'].reset_index(drop=True)
    q=np.zeros(2);cash=peak=float(plan['initial_cash']);previous=None;history=[];state=clock=0
    fees=TieredFees(plan['tariff'],slippage_bps=slippage_bps);rows=[]
    for i,row in frame.iterrows():
        known=row[['known_soxl','known_soxx']].to_numpy(float);nav=float(q@known+cash);peak=max(peak,nav)
        if previous is not None:history.append(nav/previous-1)
        previous=nav
        opened=prepare_on_policy_open(row,pretrade_weights=np.r_[q*known,cash]/nav,equity=nav,peak=peak,risk_state=state,risk_clock=clock,history=history,recovery_spec=recovery_spec,model=None,action_spec=action_spec)
        state,clock=opened['risk_state'],opened['risk_clock'];expected=path.iloc[i]
        if (state,clock)!=(int(expected.risk_state),int(expected.risk_clock)):raise ValueError('reference state identity differs')
        traded=settle_tiered_event(q,cash,desired=opened['desired_open_weights'],adjusted_prices=row[['soxl_open','soxx_open']],raw_prices=row[['raw_soxl_open','raw_soxx_open']],stamp=row.trade_date+pd.Timedelta(hours=9,minutes=47),fees=fees)
        q,cash=traded['quantities'],traded['cash'];np.testing.assert_allclose(np.r_[q,cash],expected[['ending_soxl_quantity','ending_soxx_quantity','ending_cash']].to_numpy(float),rtol=1e-10,atol=1e-7)
        features=_feature_row(row,drawdown=opened['drawdown'],cap=opened['current_beta_cap'],raw_beta=opened['raw_beta'],state=state,history=history)
        target_beta=min(opened['raw_beta'],float(selection['action_beta']))
        eligible=bool(opened['drawdown']>=recovery_spec.recovery_eligibility_drawdown and target_beta>opened['current_beta_cap']+1e-12 and row.alpha_probability>=selection['minimum_source_alpha_probability'] and np.isfinite([features[k] for k in FEATURE_COLUMNS]).all())
        item={'signal_date':row.signal_date,'trade_date':row.trade_date,'decision_at':row.trade_date+pd.Timedelta(hours=10,minutes=31),'label_available_at':pd.NaT,'eligible':eligible,'net_advantage':np.nan,**features}
        if eligible and i+1<len(frame):
            target=project_legal_weights(row.soxl_weight,row.soxx_weight,target_beta,minimum_cash_weight=recovery_spec.minimum_cash_weight,maximum_soxl_weight=recovery_spec.maximum_soxl_weight)
            reset=targets.iloc[i+1][['soxl_weight','soxx_weight','cash_weight']].to_numpy(float)
            item['net_advantage']=reset_advantage(q,cash,row=row,next_row=frame.iloc[i+1],target=target,reset_target=reset,fees=fees,entry_slot='recovery')
            item['label_available_at']=frame.iloc[i+1].trade_date+pd.Timedelta(hours=9,minutes=49)
        peak=max(peak,float(q@row[['close_soxl','close_soxx']].to_numpy(float)+cash));rows.append(item)
    result=pd.DataFrame(rows)
    for field in ['decision_at','label_available_at']:result[field]=pd.to_datetime(result[field]).dt.tz_localize('America/New_York')
    valid=result.net_advantage.notna()
    if not (result.loc[valid,'label_available_at']>result.loc[valid,'decision_at']).all():raise ValueError('recovery label maturity is noncausal')
    return result

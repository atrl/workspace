"""Same-state, same-reset execution labels; not a complete policy rollout."""
from copy import deepcopy
import numpy as np
import pandas as pd
from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event
from src.workflow.stage1_open_action_value import prepare_on_policy_open, prepare_on_policy_recovery


class Flat25Projection(TieredFees):
    """Legacy fee-only projection, not reproduction of legacy training labels."""
    def quote(self, *, shares, raw_price, sell, stamp):
        quote=super().quote(shares=shares,raw_price=raw_price,sell=sell,stamp=stamp)
        for key in ['brokerage','venue','clearing','sec','taf','cat','pass_through','slippage']:
            quote[key]=0.
        quote['slippage']=quote['notional']*.0025
        quote['total']=quote['slippage'];quote['cap_applied']=False
        return quote


def reset_advantage(q,cash,*,row,next_row,target,reset_target,fees,entry_slot="open"):
    if entry_slot not in {"open", "recovery"}:
        raise ValueError("unsupported entry slot")
    entry_prices = [f"{symbol}_{entry_slot}" for symbol in ("soxl", "soxx")]
    entry_raw = [f"raw_{symbol}_{entry_slot}" for symbol in ("soxl", "soxx")]
    entry_offset = pd.Timedelta(hours=9, minutes=47) if entry_slot == "open" else pd.Timedelta(hours=10, minutes=32)
    values=[]
    for desired in [target,None]:
        branch=deepcopy(fees)
        opened=settle_tiered_event(q,cash,desired=desired,adjusted_prices=row[entry_prices],raw_prices=row[entry_raw],stamp=row.trade_date+entry_offset,fees=branch)
        reset=settle_tiered_event(opened['quantities'],opened['cash'],desired=reset_target,adjusted_prices=next_row[['soxl_open','soxx_open']],raw_prices=next_row[['raw_soxl_open','raw_soxx_open']],stamp=next_row.trade_date+pd.Timedelta(hours=9,minutes=47),fees=branch)
        values.append(float(reset['quantities']@next_row[['soxl_open','soxx_open']].to_numpy(float)+reset['cash']))
    nav=float(q@row[entry_prices].to_numpy(float)+cash)
    return (values[0]-values[1])/nav


def own_path_labels(frame,book,*,controller,recovery_spec,action_spec,selection,plan,slippage_bps,training_reference=False):
    if training_reference and any(controller[k] is not None for k in ["frozen_open_action_value_model","frozen_recovery_model"]):
        raise ValueError("training reference must not use fitted controllers")
    path=book['independent_replay'].reset_index(drop=True);frame=frame.reset_index(drop=True)
    if not frame.trade_date.equals(path.trade_date): raise ValueError('label dates differ')
    if path.recovery_action.any(): raise ValueError('label helper requires reference without recovery orders')
    targets=book['targets'].reset_index(drop=True)
    q=np.zeros(2);cash=peak=float(plan['initial_cash']);history=[];previous=None;state=clock=0
    fees=TieredFees(plan['tariff'],slippage_bps=slippage_bps);rows=[]
    for i,row in frame.iterrows():
        known=row[['known_soxl','known_soxx']].to_numpy(float);nav=float(q@known+cash);peak=max(peak,nav)
        if previous is not None:history.append(nav/previous-1)
        previous=nav
        opened=prepare_on_policy_open(row,pretrade_weights=np.r_[q*known,cash]/nav,equity=nav,peak=peak,risk_state=state,risk_clock=clock,history=history,recovery_spec=recovery_spec,model=controller['frozen_open_action_value_model'],action_spec=action_spec)
        state,clock=opened['risk_state'],opened['risk_clock']
        expected=path.iloc[i]
        if (state,clock,bool(opened['execute_open_target']))!=(int(expected.risk_state),int(expected.risk_clock),bool(expected.execute_open_target)):raise ValueError('frozen decision identity differs')
        recovered=prepare_on_policy_recovery(row,open_state=opened,recovery_spec=recovery_spec,model=controller['frozen_recovery_model'],selected_model=selection)
        if recovered['recovery_action']: raise ValueError('recovery requires a different branch contract')
        eligible=not any(opened[k] for k in ['mandatory_derisk','large_transition','mandatory_position_repair']) and (training_reference or np.isfinite(opened['execute_probability']))
        result={'trade_date':row.trade_date,'decision_at':row.trade_date+pd.Timedelta(hours=9,minutes=46),'label_available_at':pd.NaT,'slippage_bps':slippage_bps,'eligible':eligible,'execute':opened['execute_open_target'],'probability':opened['execute_probability'],'tiered_advantage':np.nan,'flat25_advantage':np.nan}
        if training_reference:result.update(opened['features'])
        if eligible and i+1<len(frame):
            next_row=frame.iloc[i+1];reset=targets.iloc[i+1][['soxl_weight','soxx_weight','cash_weight']].to_numpy(float)
            kwargs=dict(row=row,next_row=next_row,target=opened['desired_open_weights'],reset_target=reset)
            result['tiered_advantage']=reset_advantage(q,cash,fees=fees,**kwargs)
            result['flat25_advantage']=reset_advantage(q,cash,fees=Flat25Projection(plan['tariff'],slippage_bps=0),**kwargs)
            result['label_available_at']=next_row.trade_date+pd.Timedelta(hours=9,minutes=49)
        traded=settle_tiered_event(q,cash,desired=opened['desired_open_weights'] if opened['execute_open_target'] else None,adjusted_prices=row[['soxl_open','soxx_open']],raw_prices=row[['raw_soxl_open','raw_soxx_open']],stamp=row.trade_date+pd.Timedelta(hours=9,minutes=47),fees=fees)
        q,cash=traded['quantities'],traded['cash']
        np.testing.assert_allclose(np.r_[q,cash],expected[['ending_soxl_quantity','ending_soxx_quantity','ending_cash']].to_numpy(float),rtol=1e-10,atol=1e-7)
        peak=max(peak,float(q@row[['close_soxl','close_soxx']].to_numpy(float)+cash));rows.append(result)
    return pd.DataFrame(rows)

"""Ordered cash decomposition on an unchanged reference ledger."""
import numpy as np
import pandas as pd
from src.workflow.intraday_recovery_direct_allocation import _advance_state, project_legal_weights
from src.workflow.stage1_open_action_value import prepare_on_policy_open


COMPONENTS=['raw_cash','legality_increment','cycle_increment','trailing_increment','emergency_increment','hold_increment','known_to_fill_increment','fees_rounding_increment']


def projected_layers(row,*,state,clock,drawdown,history,spec):
    state,clock,q=_advance_state(state,clock,drawdown,spec)
    beta=3*row.soxl_weight+row.soxx_weight
    cycle=max(spec.defensive_beta,q*beta)
    recent=float(np.prod(1+np.asarray(history[-spec.trailing_loss_days:]))-1) if len(history)>=spec.trailing_loss_days else np.nan
    trailing=min(cycle,spec.trailing_loss_beta) if recent<=-spec.trailing_loss_trigger else cycle
    emergency=min(trailing,spec.emergency_beta) if drawdown>=spec.emergency_drawdown else trailing
    caps=[beta,cycle,trailing,emergency]
    weights=[project_legal_weights(row.soxl_weight,row.soxx_weight,cap,minimum_cash_weight=spec.minimum_cash_weight,maximum_soxl_weight=spec.maximum_soxl_weight) for cap in caps]
    cash=[float(row.cash_weight),*[float(w[2]) for w in weights]]
    parts=dict(zip(COMPONENTS[:5],[cash[0],*np.diff(cash)]))
    return parts,weights[-1],emergency,{'cycle_beta_binding':min(beta,cycle)<beta-1e-10,'trailing_beta_binding':min(beta,trailing)<min(beta,cycle)-1e-10,'emergency_beta_binding':min(beta,emergency)<min(beta,trailing)-1e-10,'recent_return':recent}


def trace_cash_layers(frame,book,*,controller,recovery_spec,action_spec,initial_cash,slippage_bps):
    frame=frame.reset_index(drop=True);path=book['independent_replay'].reset_index(drop=True)
    if not frame.trade_date.equals(path.trade_date) or path.recovery_action.any():raise ValueError('incompatible reference calendar or recovery orders')
    q=np.zeros(2);cash=peak=float(initial_cash);state=clock=0;history=[];previous=None;rows=[]
    for i,row in frame.iterrows():
        known=row[['known_soxl','known_soxx']].to_numpy(float);fill=row[['soxl_open','soxx_open']].to_numpy(float)
        nav=float(q@known+cash);peak=max(peak,nav);drawdown=1-nav/peak
        if previous is not None:history.append(nav/previous-1)
        previous=nav
        parts,target,cap,flags=projected_layers(row,state=state,clock=clock,drawdown=drawdown,history=history,spec=recovery_spec)
        opened=prepare_on_policy_open(row,pretrade_weights=np.r_[q*known,cash]/nav,equity=nav,peak=peak,risk_state=state,risk_clock=clock,history=history,recovery_spec=recovery_spec,model=controller['frozen_open_action_value_model'],action_spec=action_spec)
        np.testing.assert_allclose(target,opened['desired_open_weights'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(cap,opened['current_beta_cap'],rtol=0,atol=1e-12)
        state,clock=opened['risk_state'],opened['risk_clock'];expected=path.iloc[i]
        if (state,clock,bool(opened['execute_open_target']))!=(int(expected.risk_state),int(expected.risk_clock),bool(expected.execute_open_target)):raise ValueError('reference decision differs')
        chosen=float(target[2]) if opened['execute_open_target'] else cash/nav
        intended_fill=float(target[2]) if opened['execute_open_target'] else cash/float(q@fill+cash)
        q=expected[['ending_soxl_quantity','ending_soxx_quantity']].to_numpy(float);cash=float(expected.ending_cash)
        actual=cash/float(q@fill+cash)
        parts.update(hold_increment=chosen-target[2],known_to_fill_increment=intended_fill-chosen,fees_rounding_increment=actual-intended_fill)
        np.testing.assert_allclose(sum(parts.values()),actual,rtol=0,atol=1e-12)
        rows.append({'trade_date':row.trade_date,'slippage_bps':slippage_bps,'risk_state':state,**parts,**flags,'actual_cash':actual,'accounting_error':sum(parts.values())-actual})
        peak=max(peak,float(q@row[['close_soxl','close_soxx']].to_numpy(float)+cash))
    return pd.DataFrame(rows)

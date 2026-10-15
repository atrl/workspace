"""One fixed local STE probe; exact forwards do not imply useful gradients."""
import numpy as np
import pandas as pd
import torch
from src.model.full_path_tiered import tiered_tensor_settle
from src.model.minute_pit_allocator import _scaled,COMMON_FEATURES
from src.workflow.pit_constituent_auxiliary_training_allocation import _project_legal_weights,_smooth_weights
from src.workflow.intraday_recovery_direct_allocation import _advance_state


def frozen_logits(models,features,common,dates):
    result=[]
    for model in models:
        if any(p.dtype!=torch.float32 for p in model.network.parameters()):raise ValueError('frozen inference dtype differs')
        with torch.no_grad():
            x=_scaled(features.reindex(dates),model.base_features,model.primary_scaler)
            c=_scaled(common.reindex(dates),list(COMMON_FEATURES),model.common_scaler)
            logits,tail,alpha=model.network(x,c)
        result.append((logits.detach(),torch.sigmoid(tail).numpy(),torch.sigmoid(alpha).numpy()))
    return result


def raw_weights(logits,theta,spec):
    offset=theta[0]*theta.new_tensor([1.,-1.,0.])+theta[1]*theta.new_tensor([0.,-1.,1.])
    values=[]
    for value,_,_ in logits:
        w=_project_legal_weights(value+offset,maximum_soxl_weight=spec.maximum_soxl_weight,minimum_cash_weight=spec.minimum_cash_weight).double()
        values.append(w/w.sum(dim=1,keepdim=True))
    total=sum(values)/len(values)
    return total/total.sum(dim=1,keepdim=True)


def reference_targets(logits,theta,spec,candidate,dates,returns):
    with torch.no_grad():weights=raw_weights(logits,torch.tensor(theta,dtype=torch.float32),spec).numpy()
    frame=pd.DataFrame(weights,index=dates,columns=['soxl_weight','soxx_weight','cash_weight'])
    # Original float32 auxiliary-output averaging, not a changed classifier.
    frame['tail_probability']=sum(x[1] for x in logits)/len(logits)
    frame['alpha_probability']=sum(x[2] for x in logits)/len(logits)
    result=_smooth_weights(frame,spec=spec,candidate=candidate,asset_returns=returns)
    return frame,result


def smooth_probe(raw,raw_exact,returns,spec,candidate):
    span=candidate.smoothing_span if candidate.smoothing_span is not None else spec.smoothing_span
    threshold=candidate.rebalance_threshold if candidate.rebalance_threshold is not None else spec.rebalance_threshold
    alpha=2/(span+1);exact_ema=raw_exact[['soxl_weight','soxx_weight','cash_weight']].ewm(span=span,adjust=False).mean().to_numpy()
    exact_current=np.array([0.,0.,1.]);current=raw.new_tensor(exact_current);ema=None;rows=[];accepted=[]
    for i in range(len(raw)):
        ema=raw[i] if ema is None else (1-alpha)*ema+alpha*raw[i]
        take=bool(abs(exact_ema[i,:2]-exact_current[:2]).sum()>=threshold);accepted.append(take)
        chosen=ema if take else current;exact_chosen=exact_ema[i] if take else exact_current
        marked=chosen*raw.new_tensor([1+returns[i,0],1+returns[i,1],1.]);current=marked/marked.sum()
        emarked=exact_chosen*np.array([1+returns[i,0],1+returns[i,1],1.]);exact_current=emarked/max(emarked.sum(),1e-12)
        chosen=torch.clamp(chosen,0.,1.);rows.append(chosen/chosen.sum())
    return torch.stack(rows),accepted


def legal(weights,cap,spec):
    zero=cap.new_tensor(0.);limit=cap.new_tensor(1-spec.minimum_cash_weight)
    beta=torch.minimum(3*weights[0]+weights[1],torch.maximum(zero,cap))
    risk=torch.minimum(torch.minimum(weights[0]+weights[1],beta),limit)
    l=torch.maximum(zero,torch.minimum(cap.new_tensor(spec.maximum_soxl_weight),(beta-risk)/2))
    x=torch.maximum(zero,torch.minimum(limit-l,risk-l))
    return torch.stack([l,x,1-l-x])


def composition(weights):
    delta=torch.minimum(weights[0],weights[2]/2)
    return weights+torch.stack([-delta,3*delta,-2*delta])


def exact_slots(path,initial_cash):
    q=np.zeros(2);cash=float(initial_cash);monthly={};out={}
    for row in path['replay'].itertuples():
        for slot in ['open','recovery']:
            orders=path['orders'].loc[path['orders'].trade_date.eq(row.trade_date)&path['orders'].slot.eq(slot)]
            month=pd.Timestamp(row.trade_date).strftime('%Y-%m')
            for order in orders.itertuples():
                j={'SOXL':0,'SOXX':1}[order.symbol];amount=(-1 if order.sell else 1)*order.adjusted_amount
                q[j]+=amount;cash-=(-1 if order.sell else 1)*order.notional+order.total
                if not order.cap_applied:monthly[month]=monthly.get(month,0)+order.raw_shares
            out[(row.trade_date,slot)]=(q.copy(),cash,float(monthly.get(month,0)))
    return out


def objective(close_nav,drawdowns,B,plan):
    relative=torch.log(close_nav[-1]/close_nav[62])-close_nav.new_tensor(B)
    return -relative+torch.sqrt(torch.relu(-relative).square()+1e-12)+5*torch.relu(drawdowns.max()-.30)


def reference_objective(paths,frame,B):
    path=paths[2]['replay'];prices=frame[['close_soxl','close_soxx']].to_numpy(float)
    close=(path[['ending_soxl_quantity','ending_soxx_quantity']].to_numpy(float)*prices).sum(axis=1)+path.ending_cash.to_numpy()
    relative=float(np.log(close[-1]/close[62])-B);dd=float(path.monitored_drawdown.max())
    return -relative+float(np.sqrt(max(-relative,0)**2+1e-12))+5*max(dd-.30,0),close


def tensor_probe(frame,weights,paths,trace,recovery_spec,selection,tariff,B):
    costs=[0,2,5];state={c:{'q':weights.new_zeros(2),'cash':weights.new_tensor(100000.),'peak':weights.new_tensor(100000.),'monthly':weights.new_tensor(0.),'state':0,'clock':0,'month':None,'closes':[],'dds':[]} for c in costs}
    slots={c:exact_slots(paths[c],100000.) for c in costs};traces={c:trace.loc[trace.slippage_bps.eq(c)].reset_index(drop=True) for c in costs};audits=[];funding_count=0
    for i,(_,row) in enumerate(frame.iterrows()):
        known=weights.new_tensor(row[['known_soxl','known_soxx']].to_numpy(float));closing=weights.new_tensor(row[['close_soxl','close_soxx']].to_numpy(float));caps={};opened={}
        for c in costs:
            s=state[c];t=traces[c].iloc[i];nav=s['q']@known+s['cash'];s['peak']=torch.maximum(s['peak'],nav);dd=torch.clamp(1-nav/s['peak'],min=0);s['dds'].append(dd)
            st,clock,qfactor=_advance_state(s['state'],s['clock'],float(t.known_drawdown),recovery_spec)
            if (st,clock)!=(int(t.risk_state),int(t.risk_clock)):raise ValueError('detached state branch differs')
            s['state'],s['clock']=st,clock
            raw_beta=3*weights[i,0]+weights[i,1];cap=torch.maximum(weights.new_tensor(recovery_spec.defensive_beta),qfactor*raw_beta)
            if pd.notna(t.trailing4_known_return) and t.trailing4_known_return<=-recovery_spec.trailing_loss_trigger:cap=torch.minimum(cap,weights.new_tensor(recovery_spec.trailing_loss_beta))
            if t.known_drawdown>=recovery_spec.emergency_drawdown:cap=torch.minimum(cap,weights.new_tensor(recovery_spec.emergency_beta))
            caps[c]=cap;opened[c]=(nav,dd)
        common=torch.amin(torch.stack(list(caps.values())));desired=composition(legal(weights[i],common,recovery_spec))
        for slot in ['open','recovery']:
            recover=bool(traces[0].recovery_consensus.iloc[i])
            if slot=='recovery' and not recover:continue
            target=desired if slot=='open' else composition(legal(weights[i],torch.minimum(3*weights[i,0]+weights[i,1],weights.new_tensor(selection['action_beta'])),recovery_spec))
            pricecols=['soxl_open','soxx_open'] if slot=='open' else ['soxl_recovery','soxx_recovery'];rawcols=['raw_'+x for x in pricecols]
            price=weights.new_tensor(row[pricecols].to_numpy(float));raw=weights.new_tensor(row[rawcols].to_numpy(float))
            for c in costs:
                s=state[c];month=pd.Timestamp(row.trade_date).strftime('%Y-%m')
                if s['month']!=month:s['monthly']=weights.new_tensor(0.);s['month']=month
                q,cash,monthly,_=tiered_tensor_settle(s['q'][None,:],s['cash'][None],target[None,:],price[None,:],raw[None,:],s['monthly'][None],tariff=tariff,slippage_bps=c)
                eq,ec,em=slots[c][(row.trade_date,slot)];funding_count+=int(float(cash[0].detach()) < -1e-7)
                # Native cash-funding/fees/whole shares determine forward; STE is only backward approximation.
                s['q']=q[0]+(weights.new_tensor(eq)-q[0]).detach();s['cash']=cash[0]+(weights.new_tensor(ec)-cash[0]).detach();s['monthly']=monthly[0]+(weights.new_tensor(em)-monthly[0]).detach()
                prefix='open_after_T_' if slot=='open' else 'recovery_after_T_';expected=traces[c].iloc[i][[prefix+a for a in ['soxl','soxx','cash']]].to_numpy(float)
                audits.append({'trade_date':row.trade_date,'cost':c,'slot':slot,'target_max_error':float(np.max(abs(target.detach().numpy()-expected))),'q_max_error':float(np.max(abs(s['q'].detach().numpy()-eq))),'cash_error':abs(float(s['cash'].detach())-ec),'monthly_error':abs(float(s['monthly'].detach())-em),'local_cap_error':abs(float(caps[c].detach())-float(traces[c].local_cap.iloc[i])),'common_cap_error':abs(float(common.detach())-float(traces[c].common_cap.iloc[i])),'known_nav_relative_error':abs(float(opened[c][0].detach())-float(traces[c].known_nav.iloc[i]))/float(traces[c].known_nav.iloc[i])})
        for c in costs:
            s=state[c];close=s['q']@closing+s['cash'];s['peak']=torch.maximum(s['peak'],close);s['dds'].append(torch.clamp(1-close/s['peak'],min=0));s['closes'].append(close)
    primary=state[2];J=objective(torch.stack(primary['closes']),torch.stack(primary['dds']),B,{})
    return J,pd.DataFrame(audits),{'recovery_executions':int(traces[0].recovery_consensus.sum()),'surrogate_unfunded_precorrection_slots':funding_count,'recovery_gradient_qualified':False,'scope':'Only exercised baseline branches; detachedbranch exact-forward/STE-backward proposal'}

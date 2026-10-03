"""Independent audit of an already-finished registered synthetic Recorder.

This script has no fitting, scipy.optimize, project model inference or paired
cost-path imports. It reconstructs saved coordinates using plain NumPy. The
optional batched replay advances an independent owned book separately at each
cost and is checked against a scalar four-root reference on fixed samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


COSTS = (10., 25.)
SEEDS = (17, 41, 71)
ROWS = (0, 2, 3)
VALUE_TOL = 1e-10
RISK_TOL_PP = 1e-10
MARGIN_F = 1e-8
EXPECTED_PLAN_SHA256 = 'c7818045478368cd472b4f37c54a19ffcad014478e36d558f4af31af6e5f40b7'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def array_sha(value):
    return digest(np.asarray(value, dtype='<f8').tobytes())


def tensor_sha(state):
    h=hashlib.sha256()
    for name in sorted(state):
        a=np.asarray(state[name])
        assert a.dtype==np.float64 and np.isfinite(a).all()
        h.update(name.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def check_model(payload):
    assert payload['dtype']=='float64' and payload['feature_count']==27 and payload['hidden_units']==8
    state={k:np.asarray(v).copy() for k,v in payload['state_dict'].items()}
    assert tensor_sha(state)==payload['tensor_sha256']
    return state


def encoder_sha(state):
    h=hashlib.sha256()
    for name in ('bias','weight'):
        h.update(name.encode());h.update(np.asarray(state['hidden.'+name],dtype='<f8').tobytes())
    return h.hexdigest()


def decode(coordinates, encoder):
    z=np.asarray(coordinates,dtype=np.float64)
    assert z.shape==(33,) and np.isfinite(z).all()
    rows=z.reshape(3,11)
    state={'A':np.zeros((4,8)), 'a':np.zeros(4), 'K':np.zeros((4,3)),
           'hidden.weight':encoder['hidden.weight'].copy(),
           'hidden.bias':encoder['hidden.bias'].copy()}
    state['A'][list(ROWS)]=rows[:,:8];state['a'][list(ROWS)]=rows[:,8]
    state['K'][list(ROWS),0]=rows[:,9];state['K'][list(ROWS),2]=rows[:,10]
    return state


def embed_parent(state):
    t=state['theta'];assert t.shape==(3,9) and t.min()>=0 and t.sum()<=1+1e-12
    embedded={'A':np.zeros((4,8)), 'a':np.zeros(4), 'K':np.zeros((4,3)),
              'hidden.weight':state['hidden.weight'].copy(),
              'hidden.bias':state['hidden.bias'].copy()}
    embedded['A'][list(ROWS)]=t[:,1:];embedded['a'][list(ROWS)]=t[:,0]
    embedded['A'][1]=-t[:,1:].sum(0);embedded['a'][1]=1-t[:,0].sum()
    return embedded


def sigmoid(z):
    result=np.empty_like(z);positive=z>=0
    result[positive]=1/(1+np.exp(-z[positive]));e=np.exp(z[~positive]);result[~positive]=e/(1+e)
    return result


def project(scores):
    shifted=scores-scores.max(axis=-1,keepdims=True)
    ordered=np.sort(shifted,axis=-1)[...,::-1];sums=np.cumsum(ordered,axis=-1)
    eligible=ordered-(sums-1)/np.arange(1,5)>0
    count=eligible.sum(-1)
    tau=(np.take_along_axis(sums,(count-1)[...,None],axis=-1)[...,0]-1)/count
    q=np.maximum(shifted-tau[...,None],0.)
    assert q.min()>=0 and np.max(np.abs(q.sum(-1)-1))<1e-12
    return q


def scalar_funding(held,target,rate):
    """Independent simple four-region reference, no project kernel call."""
    if np.array_equal(held,target):return 1.,0.,0.
    candidates=[]
    for a,b in ((-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)):
        nav=(1+rate*(a*held[0]+b*held[1]))/(1+rate*(a*target[0]+b*target[1]))
        resid=abs(nav+rate*(abs(nav*target[0]-held[0])+abs(nav*target[1]-held[1]))-1)
        candidates.append((resid,nav))
    nav=min(enumerate(candidates),key=lambda v:(v[1][0],v[0]))[1][1]
    turn=abs(nav*target[0]-held[0])+abs(nav*target[1]-held[1]);fee=rate*turn
    assert nav>0 and abs(nav+fee-1)<1e-12
    return nav,fee,turn


def scalar_rollout(states,x,returns,cost):
    """Each component sees this book's actual holdings, never private books."""
    bases=[]
    for d in states:
        gate=sigmoid(x@d['hidden.weight'].T+d['hidden.bias'])
        bases.append(gate@d['A'].T+d['a'])
    held=np.array([0.,0.,1.]);rate=cost/10000.;n=len(x)
    out={k:np.empty((n,3)) for k in ('pretrade_weights','target_weights','ending_weights','open_net_weights','execution_target_weights')}
    out.update({k:np.empty(n) for k in ('net_equity','fees','turnover','open_net_equity','logit_scale')})
    out['expert_weights']=np.empty((n,4));out['exact_hold']=np.empty(n,dtype=bool)
    for t in range(n):
        qs=[];targets=[];holds=[];scale=0.
        for d,base in zip(states,bases):
            scores=base[t]+d['K']@held;q=project(scores)
            hold=bool(q[3]==1 and np.count_nonzero(q)==1)
            qs.append(q);holds.append(hold);targets.append(held.copy() if hold else q[:3]+q[3]*held)
            scale=max(scale,float(np.abs(scores).max()))
        target=held.copy() if all(holds) else np.mean(targets,axis=0)
        nav,fee,turn=scalar_funding(held,target,rate)
        marked=np.array([target[0]*(1+returns[t,0]),target[1]*(1+returns[t,1]),target[2]])
        gross=marked[0]+marked[1]+marked[2];ending=marked/gross
        out['pretrade_weights'][t]=held;out['target_weights'][t]=target;out['ending_weights'][t]=ending
        out['open_net_weights'][t]=target;out['execution_target_weights'][t]=[nav*target[0],nav*target[1],1-nav*target[0]-nav*target[1]]
        out['net_equity'][t]=nav*gross;out['fees'][t]=fee;out['turnover'][t]=turn
        out['open_net_equity'][t]=float((nav*target).sum());out['logit_scale'][t]=scale
        out['expert_weights'][t]=np.mean(qs,axis=0);out['exact_hold'][t]=all(holds);held=ending
    out['net_return']=out['net_equity']-1;out['equity']=out['net_equity'].cumprod();out['exact_zero_order']=out['turnover']==0
    assert all(np.isfinite(v).all() for v in out.values()) and out['net_equity'].min()>0
    return out


def batch_rollout(states,x,returns,cost):
    """Independent book per saved trial, advanced anew for EACH cost.

    Funding uses16 fixed-point iterations, independently of the scalar
    four-root reference. Its contraction factor is <=rate<=.0025; each actual
    funding/cash residual is still checked. No model/paired helper is used.
    """
    m,n=len(states),len(x);rate=cost/10000.
    market=np.stack([sigmoid(x@d['hidden.weight'].T+d['hidden.bias'])@d['A'].T+d['a'] for d in states])
    K=np.stack([d['K'] for d in states]);held=np.tile([0.,0.,1.],(m,1));nets=np.empty((m,n))
    support_counts=np.zeros((m,4),int);sole_counts=np.zeros((m,4),int)
    exact_hold=np.zeros(m,int);zero_orders=np.zeros(m,int);reported_orders=np.zeros(m,int)
    max_scale=np.zeros(m);max_residual=0.;max_cash_error=0.
    for t in range(n):
        scores=market[:,t]+np.einsum('mki,mi->mk',K,held)
        q=project(scores);support=(q>0).sum(1);hold=(q[:,3]==1)&(support==1)
        target=q[:,:3]+q[:,3,None]*held;target[hold]=held[hold]
        nav=np.ones(m)
        for _ in range(16):
            nav=1-rate*np.abs(nav[:,None]*target[:,:2]-held[:,:2]).sum(1)
        turn=np.abs(nav[:,None]*target[:,:2]-held[:,:2]).sum(1);fees=rate*turn
        max_residual=max(max_residual,float(np.abs(nav+fees-1).max()))
        cash_error=np.abs((1-nav*target[:,0]-nav*target[:,1])-fees-nav*target[:,2])
        max_cash_error=max(max_cash_error,float(cash_error.max()))
        marked=target*np.r_[1+returns[t],1.]
        gross=marked[:,0]+marked[:,1]+marked[:,2];nets[:,t]=nav*gross;held=marked/gross[:,None]
        assert np.isfinite(nets[:,t]).all() and nets[:,t].min()>0
        for j in range(4):
            support_counts[:,j]+=(support==j+1);sole_counts[:,j]+=(support==1)&(q[:,j]==1)
        exact_hold+=hold;zero_orders+=(turn==0);reported_orders+=(turn>1e-12)
        max_scale=np.maximum(max_scale,np.abs(scores).max(1))
    assert max_residual<1e-12 and max_cash_error<1e-12
    return {'net_equity':nets,'support_counts':support_counts,'sole_counts':sole_counts,
            'exact_hold_count':exact_hold,'exact_zero_order_count':zero_orders,
            'reported_trade_count':reported_orders,'maximum_absolute_logit':max_scale,
            'maximum_funding_residual':max_residual,'maximum_cash_error':max_cash_error}


def risk_losses(nets):
    a=np.atleast_2d(nets);logs=np.log(a);levels=np.c_[np.zeros(len(a)),logs.cumsum(1)]
    dd=-np.expm1(np.min(levels-np.maximum.accumulate(levels,axis=1),axis=1))
    daily=np.sort(np.expm1(logs),axis=1);mass=.05*a.shape[1];whole=int(np.floor(mass))
    total=daily[:,:whole].sum(1)
    if whole<daily.shape[1]:total=total+(mass-whole)*daily[:,whole]
    cvar=-total/mass
    return np.c_[dd,cvar]


def metrics(nets,reference,reference_losses):
    a=np.atleast_2d(nets);relative=np.log(a)-np.log(reference)
    A=63.*relative.mean(1);gaps=100*(risk_losses(a)-reference_losses)
    return A,gaps


def path_error(reconstructed,saved):
    errors={}
    for name,value in reconstructed.items():
        if name in saved:
            a,b=np.asarray(value),np.asarray(saved[name])
            assert a.shape==b.shape
            if name=='equity':
                errors['relative_equity']=float(np.max(np.abs(a/b-1)))
            else:
                errors[name]=float(np.max(np.abs(a.astype(float)-b.astype(float))))
    return errors


def load_recorder(reference_path):
    """Qlib is the evidence source; byte identities checked before loading."""
    from qlib.workflow import R
    reference_path=Path(reference_path);raw=reference_path.read_bytes();reference=json.loads(raw)
    assert reference['complete'] is True
    assert reference['plan_sha256']==EXPECTED_PLAN_SHA256
    objects={};identities=[]
    with R.uri_context(reference['recorder_uri']):
        recorder=R.get_recorder(experiment_name='feasible_parent_sparse_search_20261003',recorder_id=reference['recorder_id'])
        assert recorder.client.get_run(recorder.id).info.status=='FINISHED'
        base=(Path(recorder.get_local_dir())/'artifacts').resolve()
        for key,identity in reference['artifacts'].items():
            path=(base/identity['artifact']).resolve();assert path.is_relative_to(base)
            assert path.stat().st_size==identity['bytes'] and file_sha(path)==identity['sha256']
            if identity.get('deserialize',True):objects[key]=recorder.load_object(identity['artifact'])
            assert file_sha(path)==identity['sha256']
            identities.append({'artifact':identity['artifact'],'sha256':identity['sha256'],'bytes':identity['bytes']})
        plan_path=base/'inputs/selection_plan.json'
        assert file_sha(plan_path)==reference['plan_sha256']
        plan=json.loads(plan_path.read_text())
    assert reference_path.read_bytes()==raw
    return reference,objects,identities,plan


def audit(reference_path,output_prefix,batch_size=128):
    ref,objects,identities,plan=load_recorder(reference_path)
    inputs=objects['preflight/inputs.pkl'];x=np.asarray(inputs['features']);r=np.asarray(inputs['returns']);n=504
    assert x.shape==(n,27) and r.shape==(n,2) and inputs['chronology']=='ordinal'
    sign=np.where(np.arange(n)%2==0,1.,-1.)
    np.testing.assert_array_equal(x[:,0],sign);assert not np.any(x[:,1:])
    np.testing.assert_array_equal(r[:,1],.002+.008*sign);np.testing.assert_array_equal(r[:,0],3*r[:,1])
    assert inputs['source_sha256']==ref['source_sha256']
    assert plan['architecture']['trainable_search_coordinates']==33
    assert plan['search']['options']=={'maxiter':2000,'rhobeg':1.,'tol':1e-6,'catol':0.,'disp':0}
    assert plan['search']['unique_whole_two_cost_path_evaluation_cap']==2000
    assert inputs['search_spec']['max_evaluations']==2000 and inputs['search_spec']['scipy_maxiter']==2000
    for key,value in {'updates':120,'learning_rate':.003,'gradient_clip':5.,'dual_learning_rate':.003,
                      'context_sessions':21,'gradient_chunk_length':63,'risk_group_length':63,
                      'minimum_rows':252,'torch_threads':1,'downside_coefficient':0.,
                      'rms_epsilon':1e-12,'risk_tolerance_percentage_points':1e-10}.items():
        assert inputs['parent_spec'][key]==value
    assert tuple(inputs['parent_spec']['cost_bps'])==COSTS
    references={c:(1+r[:,1]).copy() for c in COSTS}
    for c in COSTS:references[c][0]/=1+c/10000.
    B={c:risk_losses(v)[0] for c,v in references.items()}
    maximum={'F':0.,'A':0.,'raw_gap_pp':0.,'normalized_g':0.,'selected_path':0.,'selected_relative_NAV':0.,
             'batched_vs_scalar_net':0.,'parent_F':0.,'funding_residual':0.,'cash_error':0.}
    failures=[];diagnostic_differences=[];trial_rows=[];seed_rows=[];sample_rows=[]
    parent_states=[];selected_states=[];parent_checkpoint_count=0
    for seed in SEEDS:
        parent_payload=objects[f'preflight/parents/{seed}/model.pkl'];assert parent_payload['seed']==seed
        parent=check_model(parent_payload)
        parent_full=objects[f'preflight/parents/{seed}/audit.pkl']
        assert parent_full['seed']==seed
        if parent_full['failure'] is None:assert len(parent_full['epochs'])==121
        assert [e['epoch'] for e in parent_full['epochs']]==list(range(len(parent_full['epochs'])))
        parent_checkpoint_count+=len(parent_full['epochs'])
        for e in parent_full['epochs']:check_model(e['checkpoint'])
        assert parent_full['spec']==inputs['parent_spec']
        parent_state=embed_parent(parent);parent_states.append(parent_state)
        parent_A=[];parent_gaps=[]
        for c in COSTS:
            path=scalar_rollout([parent_state],x,r,c);A,gap=metrics(path['net_equity'],references[c],B[c])
            parent_A.append(float(A[0]));parent_gaps.extend(gap[0])
        parent_F=min(parent_A);status=objects[f'preflight/parents/{seed}/revalidation.pkl']
        maximum['parent_F']=max(maximum['parent_F'],abs(parent_F-status['F']))
        parent_healthy=parent_full['failure'] is None and status['healthy'] and bool(np.max(parent_gaps)<=RISK_TOL_PP) and parent_F>0
        search_key=f'preflight/search/{seed}/audit.pkl'
        if search_key not in objects:
            seed_rows.append({'seed':seed,'search_ran':False,'parent_stage_passed':bool(parent_healthy),'incremental_passed':False})
            continue
        meta=objects[search_key];trials=meta['trials'];count=len(trials)
        assert meta['seed']==seed
        assert 1<=count<=2000 and meta['unique_evaluations']==count
        assert meta['spec']==inputs['search_spec']
        assert meta['optimizer_diagnostics']['unique_full_path_evaluations']==count
        attempted=meta.get('attempted_full_path_evaluations',count)
        assert count<=attempted<=2000
        assert [t['trial_id'] for t in trials]==list(range(count))
        keys=[np.asarray(t['coordinates'],dtype='<f8').tobytes() for t in trials]
        assert len(set(keys))==count
        eh=encoder_sha(parent)
        assert meta['frozen_encoder_sha256_before']==eh==meta['frozen_encoder_sha256_after']
        embedded=check_model(meta['parent_model'])
        assert encoder_sha(embedded)==eh
        states=[]
        for t in trials:
            assert array_sha(t['coordinates'])==t['coordinate_sha256']
            state=decode(t['coordinates'],parent)
            assert tensor_sha(state)==t['tensor_sha256']
            states.append(state)
        assert tensor_sha(states[0])==tensor_sha(embedded) and trials[0]['initial_parent']
        assert all(not t['initial_parent'] for t in trials[1:])
        # Two independent cost books for every saved head; no reused other-cost state.
        A_all={c:np.empty(count) for c in COSTS};gaps_all={c:np.empty((count,2)) for c in COSTS}
        fixed_samples=sorted(set([0,count-1]+([meta['selected_trial']] if isinstance(meta['selected_trial'],int) else [])))
        for start in range(0,count,batch_size):
            end=min(count,start+batch_size);subset=states[start:end]
            for c in COSTS:
                batch=batch_rollout(subset,x,r,c)
                A,gaps=metrics(batch['net_equity'],references[c],B[c]);A_all[c][start:end]=A;gaps_all[c][start:end]=gaps
                maximum['funding_residual']=max(maximum['funding_residual'],batch['maximum_funding_residual'])
                maximum['cash_error']=max(maximum['cash_error'],batch['maximum_cash_error'])
                for i in fixed_samples:
                    if start<=i<end:
                        scalar=scalar_rollout([states[i]],x,r,c)
                        err=float(np.abs(scalar['net_equity']-batch['net_equity'][i-start]).max())
                        maximum['batched_vs_scalar_net']=max(maximum['batched_vs_scalar_net'],err)
                        sample_rows.append({'seed':seed,'trial':i,'cost_bps':c,'batch_scalar_net_error':err})
                for j,t in enumerate(trials[start:end]):
                    saved=t['action_diagnostics'][c]
                    if batch['exact_hold_count'][j]!=saved['exact_hold_count']:
                        diagnostic_differences.append({'seed':seed,'trial':t['trial_id'],'cost':c,'kind':'exact_hold_count','saved':saved['exact_hold_count'],'independent':int(batch['exact_hold_count'][j])})
                    for k in (1,2,3,4):
                        if batch['support_counts'][j,k-1]/n!=saved['support_cardinality_fraction'][k]:
                            diagnostic_differences.append({'seed':seed,'trial':t['trial_id'],'cost':c,'kind':'support_cardinality','support':k})
        best_F=0.;best_origin='null';best_trial='anchor'
        if meta['parent_F']>0:best_F=meta['parent_F'];best_origin='parent';best_trial='parent'
        status_mismatch_count=0
        for i,t in enumerate(trials):
            g=np.concatenate([gaps_all[c][i] for c in COSTS]);f=min(A_all[c][i] for c in COSTS)
            feasible=bool(np.all(g<=RISK_TOL_PP));stored_g=np.concatenate([t['risks_by_cost'][c]['raw_percentage_point_gaps'] for c in COSTS])
            maximum['F']=max(maximum['F'],abs(f-t['F']));maximum['raw_gap_pp']=max(maximum['raw_gap_pp'],float(np.abs(g-stored_g).max()))
            for c in COSTS:
                maximum['A']=max(maximum['A'],abs(A_all[c][i]-t['A_by_cost'][c]))
                independent_norm=gaps_all[c][i]/100/B[c]
                offset=0 if c==10 else 2
                maximum['normalized_g']=max(maximum['normalized_g'],float(np.abs(independent_norm-t['normalized_g'][offset:offset+2]).max()))
            if feasible!=t['feasible']:status_mismatch_count+=1
            # Replay EXACT stored selection scalars after independently checking
            # their numerical reproduction, so no new approximate tie rule is introduced.
            if i>0 and t['feasible'] and t['F']>best_F:
                best_F=t['F'];best_origin='search';best_trial=i
            assert t['selected_origin_so_far']==best_origin and t['selected_trial_so_far']==best_trial
            trial_rows.append({'seed':seed,'trial':i,'stored_F':t['F'],'independent_F':float(f),
                'stored_feasible':t['feasible'],'independent_feasible':feasible,
                'A10':float(A_all[10.][i]),'A25':float(A_all[25.][i]),
                'DD_gap10_pp':float(g[0]),'CVaR_gap10_pp':float(g[1]),
                'DD_gap25_pp':float(g[2]),'CVaR_gap25_pp':float(g[3]),
                'selected_origin_so_far':best_origin,'selected_trial_so_far':best_trial})
        assert meta['F']==best_F and meta['selected_origin']==best_origin and meta['selected_trial']==best_trial
        selected_payload=objects[f'preflight/search/{seed}/model.pkl'];assert selected_payload['seed']==seed
        selected=check_model(selected_payload);selected_states.append(selected)
        assert encoder_sha(selected)==eh
        if best_origin=='search':assert tensor_sha(selected)==trials[best_trial]['tensor_sha256']
        elif best_origin=='parent':assert tensor_sha(selected)==tensor_sha(embedded)
        else:
            np.testing.assert_array_equal(selected['A'],0);np.testing.assert_array_equal(selected['K'],0)
            np.testing.assert_array_equal(selected['a'],[0,1,0,0])
        chosen_A=[];chosen_gaps=[]
        for c in COSTS:
            path=scalar_rollout([selected],x,r,c)
            errors=path_error(path,meta['training_paths'][c]);maximum['selected_relative_NAV']=max(maximum['selected_relative_NAV'],errors.pop('relative_equity',0.))
            maximum['selected_path']=max(maximum['selected_path'],max(errors.values()))
            A,g=metrics(path['net_equity'],references[c],B[c]);chosen_A.append(float(A[0]));chosen_gaps.extend(g[0])
        measured_F=min(chosen_A);measured_feasible=bool(np.max(chosen_gaps)<=RISK_TOL_PP)
        assert meta['training_feasible']==meta['training_feasibility_passed']==measured_feasible
        increment=measured_F-parent_F
        seed_pass=parent_healthy and meta['failure'] is None and best_origin=='search' and measured_feasible and increment>MARGIN_F
        callbacks=meta['requested_callbacks'];cache_hits=meta['cache_hits']
        if meta['failure'] is None and meta['termination']=='solver_returned':
            assert sum(callbacks.values())==count+cache_hits
            assert meta['solver_nfev']<=2000
        elif meta['termination']=='unique_evaluation_budget':
            assert count==2000 and sum(callbacks.values())==count+cache_hits+1
        seed_rows.append({'seed':seed,'search_ran':True,'parent_F':parent_F,'selected_F':measured_F,
            'increment_F':increment,'parent_stage_passed':bool(parent_healthy),'incremental_passed':bool(seed_pass),
            'selected_origin':best_origin,'selected_trial':best_trial,'unique_evaluations':count,
            'attempted_evaluations':attempted,'callbacks':callbacks,'cache_hits':cache_hits,
            'solver_nfev':meta['solver_nfev'],'termination':meta['termination'],
            'raw_feasibility_status_mismatches':status_mismatch_count,'failure':meta['failure']})
        if status_mismatch_count:failures.append(f'seed{seed}:independent raw feasibility status differs')
        print(f'audited seed{seed}: {count} unique heads, incremental_gate={seed_pass}',flush=True)
    ensemble_rows={};ensemble_pass=False
    saved_ensemble=objects.get('preflight/ensemble_audit.pkl')
    if len(selected_states)==3 and saved_ensemble is not None:
        for kind,states,saved_key in [('parent',parent_states,'parent_ensemble'),('candidate',selected_states,'candidate_ensemble')]:
            As=[];gaps=[]
            for c in COSTS:
                path=scalar_rollout(states,x,r,c);A,g=metrics(path['net_equity'],references[c],B[c]);As.append(float(A[0]));gaps.extend(g[0])
                errors=path_error(path,saved_ensemble[kind+'_training_paths'][c]);maximum['selected_relative_NAV']=max(maximum['selected_relative_NAV'],errors.pop('relative_equity',0.))
                maximum['selected_path']=max(maximum['selected_path'],max(errors.values()))
            F=min(As);feasible=bool(np.max(gaps)<=RISK_TOL_PP);saved=saved_ensemble[saved_key]
            maximum['F']=max(maximum['F'],abs(F-saved['F']))
            if feasible!=saved['feasible']:failures.append(kind+' ensemble feasibility differs')
            ensemble_rows[kind]={'F':F,'A_by_cost':dict(zip(COSTS,As)),'raw_gaps_pp':list(map(float,gaps)),'feasible':feasible}
        ensemble_pass=(saved_ensemble['failure'] is None and ensemble_rows['parent']['feasible'] and ensemble_rows['candidate']['feasible']
            and ensemble_rows['parent']['F']>=0 and ensemble_rows['candidate']['F']>=0
            and ensemble_rows['candidate']['F']>ensemble_rows['parent']['F']+MARGIN_F
            and saved_ensemble['selected_origin']=='search')
        chosen_origin='null';best=0.
        for name,origin in [('parent','parent'),('candidate','search')]:
            saved=saved_ensemble[name+'_ensemble']
            if saved['feasible'] and saved['F']>best:chosen_origin=origin;best=saved['F']
        assert saved_ensemble['selected_origin']==chosen_origin and saved_ensemble['F']==best
        ensemble_rows['retained_origin']=chosen_origin
    gate=objects['gate'];independent_pass=all(row['incremental_passed'] for row in seed_rows) and ensemble_pass
    assert gate['all_three_seed_increments_passed']==all(row['incremental_passed'] for row in seed_rows)
    assert gate['actual_candidate_ensemble_increment_passed']==bool(ensemble_pass)
    if independent_pass!=gate['passed']:failures.append('registered whole preflight gate differs')
    for key in ('F','A','raw_gap_pp','parent_F'):
        if maximum[key]>VALUE_TOL:failures.append(key+' reconstruction tolerance exceeded')
    if maximum['normalized_g']>1e-9:failures.append('normalized_g reconstruction tolerance exceeded')
    if maximum['selected_path']>1e-12 or maximum['batched_vs_scalar_net']>1e-12:failures.append('physical path reconstruction tolerance exceeded')
    if maximum['selected_relative_NAV']>1e-10:failures.append('relative compounded NAV reconstruction tolerance exceeded')
    if diagnostic_differences:failures.append('sparse support/exactH diagnostics require review')
    report={'scope':'Read-only complete saved-Cobyla-head reconstruction at two independently advanced costs; no fit or optimizer call, no paired helper, no market data or new candidate selection.',
        'recorder_id':ref['recorder_id'],'recorder_uri':ref['recorder_uri'],
        'plan_sha256':ref['plan_sha256'],'reference_sha256':file_sha(reference_path),
        'source_sha256':ref['source_sha256'],'verified_artifact_count':len(identities),
        'method_sha256':file_sha(__file__),'batch_size':batch_size,
        'funding_method':'Batch16-iteration contracting funding root, actualresidualverified; fixed initial/selected/last trials checkedagainst independent scalarfour-root books atbothcosts.',
        'selection_method':'Stored exactF/rawfeasibility tie order replayed after independent numeric reconstruction; no epsilon rescue or new selected model.',
        'reconstructed_unique_heads':len(trial_rows),'reconstructed_cost_paths':2*len(trial_rows),
        'reconstructed_decisions':2*n*len(trial_rows),'parent_checkpoint_hashes_verified':parent_checkpoint_count,
        'fixed_sample_cost_paths':len(sample_rows),'maximum_errors':maximum,
        'seed_results':seed_rows,'ensemble':ensemble_rows,'ensemble_incremental_gate':bool(ensemble_pass),
        'stored_preflight_passed':gate['passed'],'independent_preflight_passed':bool(independent_pass),
        'mathematical_audit_passed':not failures,'failures':failures,
        'support_diagnostic_differences':diagnostic_differences,
        'financial_qualification':False,
        'limitations':['Numerical reconstruction is not an optimizer convergence certificate.',
                       'Synthetic incremental success cannot establish market alpha or deployment eligibility.',
                       'Searchscalars are independently recalculated; original parent120state trajectories are hash/inventory checked, not reoptimized.']}
    prefix=Path(output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else str(a))+'\n')
    pd.DataFrame(trial_rows).to_csv(str(prefix)+'-trials.csv',index=False)
    pd.DataFrame(sample_rows).to_csv(str(prefix)+'-samples.csv',index=False)
    print(json.dumps({'report':str(prefix.with_suffix('.json')),'mathematical_audit_passed':not failures,
                      'preflight_passed':bool(independent_pass),'failures':failures,'maximum_errors':maximum},indent=2))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True,help='Finished authoritative preflight_reference.json')
    p.add_argument('--output-prefix',default='/tmp/feasible-parent-preflight-independent-audit')
    p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args()
    if args.batch_size<1:raise ValueError('positive audit batch size required')
    report=audit(args.reference,args.output_prefix,args.batch_size)
    if not report['mathematical_audit_passed']:raise SystemExit(1)


if __name__=='__main__':main()

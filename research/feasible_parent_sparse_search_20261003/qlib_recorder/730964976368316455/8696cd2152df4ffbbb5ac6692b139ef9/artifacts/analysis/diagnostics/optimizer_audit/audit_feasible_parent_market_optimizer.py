"""Read-only audit of finished Phase9 search evidence, without model fitting.

All saved trial hashes/scalars/incumbency are checked. Independent full books
are rebuilt only for each retained policy, embedded parent, final searchpoint
and the complete parent/candidate ensembles. No alternative OOS policy, new
optimizer trial or full replay of every market searchpoint is constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


PREFLIGHT_METHOD=Path('/tmp/audit_feasible_parent_preflight.py')
module=importlib.util.spec_from_file_location('independent_values',PREFLIGHT_METHOD)
a=importlib.util.module_from_spec(module);module.loader.exec_module(a)
PLAN_SHA='c7818045478368cd472b4f37c54a19ffcad014478e36d558f4af31af6e5f40b7'
MODEL_SHA='47db72c75283ce2d6ad05f60234a731f2994c0900ad3f6346413e2bfa0b6d323'
COSTS=(10.,25.)
SEEDS=(17,41,71)
RAW_TOL_PP=1e-10


def project(scores):
    """Declared finite sparsemax arithmetic, without zero snapping.

    The earlier preflight supplement proved the algebraically equivalent
    divide-first test is not strict-support roundoff invariant. This explicit
    source convention avoids counting its tiny-zero difference as a new bug.
    """
    shifted=scores-scores.max();ordered=np.sort(shifted)[::-1];cumulative=np.cumsum(ordered)
    count=int(np.sum(1+np.arange(1,5)*ordered>cumulative))
    q=np.maximum(shifted-(cumulative[count-1]-1)/count,0.)
    assert q.min()>=0 and abs(q.sum()-1)<1e-12
    return q


def rollout(states,x,r,cost):
    """Standalone per-cost book; no project model/paired-path helper calls."""
    bases=[a.sigmoid(x@d['hidden.weight'].T+d['hidden.bias'])@d['A'].T+d['a'] for d in states]
    h=np.array([0.,0.,1.]);n=len(r);rate=cost/10000.
    path={name:np.empty((n,3)) for name in ['pretrade_weights','target_weights','ending_weights','open_net_weights','execution_target_weights']}
    path.update({name:np.empty(n) for name in ['net_equity','fees','turnover','open_net_equity','logit_scale']})
    path['expert_weights']=np.empty((n,4));path['exact_hold']=np.empty(n,dtype=bool)
    for t in range(n):
        qs=[];targets=[];holds=[];scale=0.
        for d,base in zip(states,bases):
            scores=base[t]+d['K']@h;q=project(scores);hold=bool(q[3]==1 and np.count_nonzero(q)==1)
            qs.append(q);holds.append(hold);targets.append(h.copy() if hold else q[:3]+q[3]*h)
            scale=max(scale,float(np.abs(scores).max()))
        z=h.copy() if all(holds) else np.mean(targets,axis=0)
        nav,fee,turn=a.scalar_funding(h,z,rate)
        marked=np.array([z[0]*(1+r[t,0]),z[1]*(1+r[t,1]),z[2]])
        gross=marked[0]+marked[1]+marked[2];end=marked/gross
        path['pretrade_weights'][t]=h;path['target_weights'][t]=z;path['ending_weights'][t]=end
        path['open_net_weights'][t]=z;path['execution_target_weights'][t]=[nav*z[0],nav*z[1],1-nav*z[0]-nav*z[1]]
        path['net_equity'][t]=nav*gross;path['fees'][t]=fee;path['turnover'][t]=turn
        path['open_net_equity'][t]=float((nav*z).sum());path['logit_scale'][t]=scale
        path['expert_weights'][t]=np.mean(qs,axis=0);path['exact_hold'][t]=all(holds);h=end
    path['equity']=path['net_equity'].cumprod();path['net_return']=path['net_equity']-1;path['exact_zero_order']=path['turnover']==0
    assert all(np.isfinite(v).all() for v in path.values()) and path['net_equity'].min()>0
    return path


def legacy_J(relative):
    n=len(relative);starts=list(range(0,n-63+1,63))
    if starts[-1]!=n-63:starts.append(n-63)
    groups=np.array([relative[s:s+63].sum() for s in starts])
    return float(63/n*relative.sum()-np.sqrt(np.mean(np.minimum(groups,0.)**2)+1e-12)+1e-6)


class Reader:
    def __init__(self,reference_path,expected_id=None):
        from qlib.workflow import R
        self.path=Path(reference_path);self.raw=self.path.read_bytes();self.ref=json.loads(self.raw)
        assert self.ref['complete'] is True and self.ref['plan_sha256']==PLAN_SHA
        if expected_id:assert self.ref['recorder_id']==expected_id
        with R.uri_context(self.ref['recorder_uri']):
            self.recorder=R.get_recorder(experiment_name='feasible_parent_sparse_search_20261003',recorder_id=self.ref['recorder_id'])
        assert self.recorder.client.get_run(self.recorder.id).info.status=='FINISHED'
        self.base=(Path(self.recorder.get_local_dir())/'artifacts').resolve()
        self.by_path={v['artifact']:v for v in self.ref['artifacts'].values()}
        self.verified=set()

    def load(self,artifact):
        from qlib.workflow import R
        identity=self.by_path[artifact];path=(self.base/artifact).resolve();assert path.is_relative_to(self.base)
        assert path.stat().st_size==identity['bytes'] and a.file_sha(path)==identity['sha256']
        with R.uri_context(self.ref['recorder_uri']):result=self.recorder.load_object(artifact)
        assert a.file_sha(path)==identity['sha256'];self.verified.add(artifact)
        return result

    def stable(self):
        assert self.path.read_bytes()==self.raw
        for name in self.verified:
            assert a.file_sha(self.base/name)==self.by_path[name]['sha256']


class Audit:
    def __init__(self):
        self.maximum={key:0. for key in ['trial_scalar_A','trial_scalar_risk_pp','trial_scalar_normalized_g',
            'reference_normalizers','path_daily','path_relative_NAV','path_F_A','path_raw_gap_pp','legacy_J']}
        self.rows=[];self.sample_rows=[];self.fit_rows=[];self.ensemble_rows=[];self.failures=[]
        self.physical_cost_paths=0;self.physical_decisions=0;self.policy_cache={}

    def max(self,name,value):self.maximum[name]=max(self.maximum[name],float(value))

    def scalar(self,row,n,B):
        costs=row.get('A_by_cost',row.get('centered_utility_by_cost'))
        assert set(costs)==set(COSTS) and all(np.isfinite(list(costs.values())))
        F=row.get('F',row.get('centered_worst_utility'));assert F==min(costs.values())
        risks=row.get('risks_by_cost',row.get('selected_risks_by_cost'));assert set(risks)==set(COSTS)
        flags=[];norms=[]
        for c in COSTS:
            risk=risks[c];metric=np.array([risk['max_drawdown_pct'],risk['cvar5_pct']])
            bench=np.array([risk['benchmark_max_drawdown_pct'],risk['benchmark_cvar5_pct']])
            raw=np.asarray(risk['raw_percentage_point_gaps']);loss=np.asarray(risk['raw_loss_gaps']);normalizers=np.asarray(risk['normalizers'])
            assert np.isfinite(np.r_[metric,bench,raw,loss,normalizers]).all() and (normalizers>0).all()
            self.max('trial_scalar_risk_pp',np.abs(raw-(bench-metric)).max())
            self.max('trial_scalar_risk_pp',np.abs(raw-100*loss).max())
            self.max('reference_normalizers',np.abs(normalizers-B[c]).max())
            flag=bool((raw<=RAW_TOL_PP).all());flags.append(flag)
            assert np.array_equal(np.asarray(risk['risk_pass']),raw<=RAW_TOL_PP)
            assert risk['feasible']==flag
            norms.extend(loss/normalizers)
            if 'aggregate_log_excess_by_cost' in row:
                total=row['aggregate_log_excess_by_cost'][c];mean=row['daily_mean_log_excess_by_cost'][c]
                self.max('trial_scalar_A',abs(costs[c]-63/n*total));self.max('trial_scalar_A',abs(mean-total/n))
        normalized=row.get('normalized_g',row.get('selected_normalized_g'))
        if normalized is not None:self.max('trial_scalar_normalized_g',np.abs(np.asarray(norms)-normalized).max())
        feasible=all(flags)
        if 'feasible' in row:assert row['feasible']==feasible
        if 'training_feasibility_passed' in row:assert row['training_feasibility_passed']==feasible
        return feasible

    def profiles(self,states,x,r,costs,saved,B,benchmark,identity):
        As={};oldJs={};gaps={}
        for c in COSTS:
            signature=(identity['variant'],identity['year'],tuple(a.tensor_sha(d) for d in states),c)
            if signature not in self.policy_cache:
                path=rollout(states,x,r,c);self.policy_cache[signature]=path
                self.physical_cost_paths+=1;self.physical_decisions+=len(x)
            else:path=self.policy_cache[signature]
            relative=np.log(path['net_equity'])-np.log(benchmark[c])
            As[c]=float(63/len(x)*relative.sum());oldJs[c]=legacy_J(relative)
            gaps[c]=100*(a.risk_losses(path['net_equity'])[0]-B[c])
            if saved is not None:
                errors=a.path_error(path,saved[c]);self.max('path_relative_NAV',errors.pop('relative_equity',0.))
                self.max('path_daily',max(errors.values(),default=0.))
            if costs is not None:
                self.max('path_F_A',abs(As[c]-costs[c]))
        return {'F':min(As.values()),'A':As,'legacy_J':min(oldJs.values()),'raw_gaps':gaps,
                'feasible':all((gap<=RAW_TOL_PP).all() for gap in gaps.values())}

    def compare_risks(self,result,records):
        for c in COSTS:self.max('path_raw_gap_pp',np.abs(result['raw_gaps'][c]-records[c]['raw_percentage_point_gaps']).max())


def run(root,market_reference,preflight_reference,output_prefix,expected_id):
    import qlib
    root=Path(root).resolve();qlib.init(provider_uri=str(root/'data/qlib/published_futu_affine_20260925_v1'),region='us')
    market=Reader(market_reference,expected_id);preflight=Reader(preflight_reference)
    assert market.ref['preflight_reference_sha256']==a.file_sha(preflight_reference)
    assert market.ref['source_sha256']==preflight.ref['source_sha256']
    assert market.ref['source_sha256']['src/model/feasible_parent_sparse_search.py']==MODEL_SHA
    gate=preflight.load('preflight/gate.pkl');assert gate['passed'] is True
    assert gate['source_sha256']==market.ref['source_sha256']
    summary=market.load('analysis/final_summary.pkl');sources=market.load('inputs/source_hashes.pkl')
    assert sources==market.ref['source_sha256']==summary['source_sha256']
    folds=market.load('inputs/development_fold_inputs.pkl')
    folds.update(market.load('inputs/diagnostic_fold_inputs.pkl'))
    fit_index=market.load('analysis/training_diagnostics.pkl');ensemble_index=market.load('analysis/ensemble_audits.pkl')
    assert len(fit_index)==75 and len(ensemble_index)==25 and len(folds)==25
    auditor=Audit();all_ids=set();total_trials=0;origin_counts={};failure_counts={};budget_terminations={}
    model_states={};parent_states={};fit_origins={};fit_stats={}
    for entry in sorted(fit_index,key=lambda d:(d['variant'],d['year'],d['seed'])):
        name,year,seed=entry['variant'],entry['year'],entry['seed'];key=(name,year,seed)
        assert key not in all_ids and seed in SEEDS;all_ids.add(key)
        fold=folds[(name,year)];x=fold['features'];r=fold['returns'];n=len(x)
        assert x.shape==(n,27) and r.shape==(n,2) and np.isfinite(x).all() and np.isfinite(r).all()
        where=entry['recorder_artifact'];meta=market.load(where);payload=market.load(where.rsplit('/',1)[0]+'/model.pkl')
        assert (meta['variant'],meta['year'],meta['seed'])==key
        assert meta['training_feature_sha256']==a.array_sha(x) and meta['training_return_sha256']==a.array_sha(r)
        selected=a.check_model(payload);model_states[key]=selected;fit_origins[key]=meta['selected_origin']
        assert payload['seed']==seed
        benchmark={c:(1+r[:,1]).copy() for c in COSTS}
        for c in COSTS:benchmark[c][0]/=1+c/10000.
        B={c:a.risk_losses(benchmark[c])[0] for c in COSTS}
        trials=meta.get('trials',[]);count=len(trials);total_trials+=count
        selected_normalizers_valid=all((B[c]>0).all() for c in COSTS)
        assert selected_normalizers_valid
        chosen_result=auditor.profiles([selected],x,r,meta['centered_utility_by_cost'],meta['training_paths'],B,benchmark,{'variant':name,'year':year})
        auditor.compare_risks(chosen_result,meta['selected_risks_by_cost'])
        auditor.max('legacy_J',abs(chosen_result['legacy_J']-meta['legacy_J']))
        assert chosen_result['feasible'] and meta['F']>=0
        if 'parent_model' not in meta:
            assert count==0 and meta['failure'] is not None and meta['selected_origin']=='null'
            parent_states[key]=None
            failure_counts[meta['failure']['kind']]=failure_counts.get(meta['failure']['kind'],0)+1
            auditor.fit_rows.append({'variant':name,'year':year,'seed':seed,'search_ran':False,'failure':meta['failure'],'selected_origin':'null'})
            continue
        parent=a.check_model(meta['parent_model']);parent_states[key]=parent
        eh=a.encoder_sha(parent);assert a.encoder_sha(selected)==eh==meta['frozen_encoder_sha256_before']==meta['frozen_encoder_sha256_after']
        assert meta['parent_revalidation']['frozen_encoder_sha256']==eh and meta['parent_revalidation']['passed'] is True
        if 'retained_tensor_sha256' in (meta.get('parent_identity') or {}):
            assert meta['parent_revalidation']['parent_tensor_sha256']==meta['parent_identity']['retained_tensor_sha256']
        else:
            identity=meta.get('parent_identity') or {}
            if identity.get('origin')=='new_reproduction_of_never_materialized_outer_parent':
                original=market.load(identity['artifact']+'/model.pkl')
                assert original['tensor_sha256']==meta['parent_revalidation']['parent_tensor_sha256']
        parent_result=auditor.profiles([parent],x,r,meta['parent_evaluation']['A_by_cost'],meta['parent_training_paths'],B,benchmark,{'variant':name,'year':year})
        auditor.compare_risks(parent_result,meta['parent_evaluation']['risks_by_cost'])
        assert parent_result['feasible'] and meta['parent_F']>=0
        auditor.max('path_F_A',abs(parent_result['F']-meta['parent_F']))
        stats=meta['optimizer_diagnostics'];assert stats['options']=={'maxiter':2000,'rhobeg':1.,'tol':1e-6,'catol':0.,'disp':0}
        assert stats['model_gradient_calls']==0 and stats['optimizer_restarts']==0
        assert 1<=count<=2000 and count==stats['unique_full_path_evaluations']==meta['unique_evaluations']
        attempted=meta.get('attempted_full_path_evaluations',count);assert count<=attempted<=2000
        assert [t['trial_id'] for t in trials]==list(range(count))
        assert [t['initial_parent'] for t in trials]==[True]+[False]*(count-1)
        coordinate_bytes=set();bestF=0.;best_origin='null';best_trial='anchor'
        if meta['parent_F']>0:bestF=meta['parent_F'];best_origin='parent';best_trial='parent'
        feasible_count=positive_feasible_count=legacy_negative_feasible_count=0
        for i,t in enumerate(trials):
            z=np.asarray(t['coordinates'],dtype='<f8');assert z.shape==(33,) and np.isfinite(z).all()
            assert z.tobytes() not in coordinate_bytes;coordinate_bytes.add(z.tobytes())
            assert a.array_sha(z)==t['coordinate_sha256']
            state=a.decode(z,parent);assert a.tensor_sha(state)==t['tensor_sha256']
            if i==0:assert t['tensor_sha256']==meta['parent_model']['tensor_sha256']
            feasible=auditor.scalar(t,n,B)
            if feasible:feasible_count+=1
            if feasible and t['F']>0:positive_feasible_count+=1
            if feasible and t['legacy_J']<0:legacy_negative_feasible_count+=1
            if i>0 and feasible and t['F']>bestF:bestF=t['F'];best_origin='search';best_trial=i
            assert t['selected_origin_so_far']==best_origin and t['selected_trial_so_far']==best_trial
            auditor.rows.append({'variant':name,'year':year,'seed':seed,'trial':i,'F':t['F'],
                'A10':t['A_by_cost'][10.],'A25':t['A_by_cost'][25.],'feasible':feasible,
                'legacy_J':t['legacy_J'],'coordinate_sha256':t['coordinate_sha256'],
                'tensor_sha256':t['tensor_sha256'],'selected_origin_so_far':best_origin,
                'selected_trial_so_far':best_trial})
        assert meta['F']==bestF and meta['selected_origin']==best_origin and meta['selected_trial']==best_trial
        assert meta['F_delta_from_parent']==meta['F']-meta['parent_F']
        if best_origin=='search':assert payload['tensor_sha256']==trials[best_trial]['tensor_sha256']
        elif best_origin=='parent':assert payload['tensor_sha256']==meta['parent_model']['tensor_sha256']
        else:
            np.testing.assert_array_equal(selected['A'],0);np.testing.assert_array_equal(selected['K'],0);np.testing.assert_array_equal(selected['a'],[0,1,0,0])
        auditor.scalar(meta,n,B)
        callbacks=stats['requested_callbacks'];hits=stats['cache_hits'];term=stats['termination']
        budget_terminations[term]=budget_terminations.get(term,0)+1
        if meta['failure'] is None and term=='solver_returned':
            assert sum(callbacks.values())==count+hits and stats['solver_result']['nfev']<=2000
        elif term=='unique_evaluation_budget':assert count==2000 and sum(callbacks.values())==count+hits+1
        if meta['failure'] is not None:
            kind=meta['failure']['kind'];failure_counts[kind]=failure_counts.get(kind,0)+1
            assert meta['qualification_eligible'] is False and meta['accounting_only'] is True
        else:assert meta['qualification_eligible'] is True
        # Fixed initial and selected are already rebuilt above. Rebuild the
        # final trial only if distinct, without examining it for new selection.
        sample_ids=sorted(set([0,count-1]+([best_trial] if isinstance(best_trial,int) else [])))
        for i in sample_ids:
            t=trials[i];sample_state=a.decode(t['coordinates'],parent)
            result=auditor.profiles([sample_state],x,r,t['A_by_cost'],None,B,benchmark,{'variant':name,'year':year})
            auditor.compare_risks(result,t['risks_by_cost']);auditor.max('legacy_J',abs(result['legacy_J']-t['legacy_J']))
            # A value within a numerical tie is recorded, not used to rerank.
            if result['feasible']!=t['feasible']:auditor.failures.append(f'{key}/trial{i}: independent rawrisk status differs')
            auditor.sample_rows.append({'variant':name,'year':year,'seed':seed,'trial':i,
                'stored_F':t['F'],'independent_F':result['F'],'stored_feasible':t['feasible'],
                'independent_feasible':result['feasible']})
        origin_counts[best_origin]=origin_counts.get(best_origin,0)+1
        row={'variant':name,'year':year,'seed':seed,'search_ran':True,'selected_origin':best_origin,
            'selected_trial':best_trial,'parent_F':meta['parent_F'],'F':meta['F'],
            'F_delta':meta['F_delta_from_parent'],'legacy_J':meta['legacy_J'],
            'unique_evaluations':count,'attempted_evaluations':attempted,'feasible_trials':feasible_count,
            'positive_feasible_trials':positive_feasible_count,'negative_legacy_J_feasible_trials':legacy_negative_feasible_count,
            'termination':term,'failure':meta['failure']}
        auditor.fit_rows.append(row);fit_stats[key]=row
        print(f'audited fit {name}/{year}/{seed}: {count} trial identities/scalars and fixed selected/parent/last books',flush=True)
        # Bound scratch memory: no path is needed by a later seed/fold audit.
        auditor.policy_cache.clear()
    assert len(all_ids)==75 and total_trials==summary['parent_work']['new_unique_two_cost_evaluations']
    for entry in sorted(ensemble_index,key=lambda d:(d['variant'],d['year'])):
        name,year=entry['variant'],entry['year'];x=folds[(name,year)]['features'];r=folds[(name,year)]['returns'];n=len(x)
        meta=market.load(entry['recorder_artifact']);bundle=market.load(entry['recorder_artifact'].rsplit('/',1)[0]+'/bundle.pkl')
        benchmark={c:(1+r[:,1]).copy() for c in COSTS}
        for c in COSTS:benchmark[c][0]/=1+c/10000.
        B={c:a.risk_losses(v)[0] for c,v in benchmark.items()}
        candidates=[a.check_model(bundle['candidate_ensemble_models'][i]) for i in range(3)]
        parents=[a.check_model(bundle['parent_ensemble_models'][i]) for i in range(3)]
        for i,seed in enumerate(SEEDS):
            assert a.tensor_sha(candidates[i])==a.tensor_sha(model_states[(name,year,seed)])
            if parent_states[(name,year,seed)] is not None:
                assert a.tensor_sha(parents[i])==a.tensor_sha(parent_states[(name,year,seed)])
        assert meta['candidate_component_origins']==[fit_origins[(name,year,s)] for s in SEEDS]
        outcomes={};bestF=0.;origin='null'
        if meta.get('ensemble_comparison_skipped_reason'):
            assert meta['failure'] is not None and meta['selected_origin']=='null'
        else:
            for role,states in [('parent',parents),('candidate',candidates)]:
                saved=meta[role+'_ensemble'];auditor.scalar(saved,n,B)
                result=auditor.profiles(states,x,r,saved['A_by_cost'],meta[role+'_training_paths'],B,benchmark,{'variant':name,'year':year})
                auditor.compare_risks(result,saved['risks_by_cost']);auditor.max('legacy_J',abs(result['legacy_J']-saved['legacy_J']))
                if result['feasible']!=saved['feasible']:auditor.failures.append(f'{name}/{year}/{role}: ensemble rawrisk status differs')
                outcomes[role]=result
                if saved['feasible'] and saved['F']>bestF:bestF=saved['F'];origin='parent' if role=='parent' else 'search'
            assert meta['selected_origin']==origin and meta['F']==bestF
        selected=[a.check_model(p) for p in bundle['ensemble_models']]
        if origin=='parent':assert [a.tensor_sha(s) for s in selected]==[a.tensor_sha(s) for s in parents]
        elif origin=='search':assert [a.tensor_sha(s) for s in selected]==[a.tensor_sha(s) for s in candidates]
        else:
            assert len(selected)==1
            np.testing.assert_array_equal(selected[0]['a'],[0,1,0,0]);np.testing.assert_array_equal(selected[0]['A'],0);np.testing.assert_array_equal(selected[0]['K'],0)
        result=auditor.profiles(selected,x,r,meta['centered_utility_by_cost'],meta['training_paths'],B,benchmark,{'variant':name,'year':year})
        auditor.compare_risks(result,meta['selected_risks_by_cost'])
        assert result['feasible'] and meta['F']>=0
        auditor.ensemble_rows.append({'variant':name,'year':year,'selected_origin':meta['selected_origin'],
            'parent_F':meta.get('parent_ensemble_F'),'candidate_F':meta.get('candidate_ensemble_F'),
            'selected_F':meta['F'],'parent_feasible':meta.get('parent_ensemble_feasible'),
            'candidate_feasible':meta.get('candidate_feasible'),'component_origins':meta['candidate_component_origins'],
            'legacy_J':meta['legacy_J'],'failure':meta['failure']})
        auditor.policy_cache.clear()
        print(f'audited whole ensemble {name}/{year}: parent/candidate/selected owned books',flush=True)
    for key,value in auditor.maximum.items():
        limit=1e-10 if key in ('path_F_A','path_raw_gap_pp','path_relative_NAV','legacy_J') else 1e-12
        if value>limit:auditor.failures.append(f'{key}: {value} exceeds independent numerical tolerance {limit}')
    market.stable();preflight.stable()
    report={'scope':'All saved trial parameter/hash/scalar/first-best checks, plus complete parent/retained and fixedinitial/last training paths and parent/candidate/selected wholeensembles. No optimizer, refit, newpolicy/OOSselection, or fullall-trialpath replay.',
        'market_recorder_id':market.ref['recorder_id'],'preflight_recorder_id':preflight.ref['recorder_id'],
        'market_reference_sha256':a.file_sha(market_reference),'preflight_reference_sha256':a.file_sha(preflight_reference),
        'plan_sha256':PLAN_SHA,'source_sha256':sources,'method_sha256':a.file_sha(__file__),
        'independent_value_helper_sha256':a.file_sha(PREFLIGHT_METHOD),
        'fit_count':len(fit_index),'ensemble_count':len(ensemble_index),'unique_trial_count':total_trials,
        'fit_origin_counts':origin_counts,'failure_kind_counts':failure_counts,'termination_counts':budget_terminations,
        'fixed_sample_model_count':len(auditor.sample_rows),'physical_cost_paths_reconstructed':auditor.physical_cost_paths,
        'physical_decisions_reconstructed':auditor.physical_decisions,'verified_market_artifacts':len(market.verified),
        'maximum_errors':auditor.maximum,'passed':not auditor.failures,'failures':auditor.failures,
        'fit_summary':auditor.fit_rows,'ensemble_summary':auditor.ensemble_rows,
        'financial_qualification':summary['historical_qualification_pass'],
        'interpretation_scope':'Training arithmetic and feasibility are verified separately from historical financialgates; neither is a forward-profit guarantee. NegativelegacyJ does not veto legitimatepositiveF.',
        'sparse_support_convention':'Source-faithful1+k*ordered>cumsum, no near-zero threshold; priorfloating-roundoffresolution remainsseparateevidence.'}
    prefix=Path(output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v))+'\n')
    pd.DataFrame(auditor.rows).to_csv(str(prefix)+'-trials.csv',index=False)
    pd.DataFrame(auditor.sample_rows).to_csv(str(prefix)+'-samples.csv',index=False)
    print(json.dumps({'report':str(prefix.with_suffix('.json')),'passed':report['passed'],
        'fit_count':len(fit_index),'trial_count':total_trials,'maximum_errors':auditor.maximum,'failures':auditor.failures},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--market-reference',required=True)
    p.add_argument('--preflight-reference',required=True);p.add_argument('--expected-id',required=True)
    p.add_argument('--output-prefix',default='/tmp/feasible-parent-market-optimizer-audit')
    args=p.parse_args();result=run(args.root,args.market_reference,args.preflight_reference,args.output_prefix,args.expected_id)
    if not result['passed']:raise SystemExit(1)

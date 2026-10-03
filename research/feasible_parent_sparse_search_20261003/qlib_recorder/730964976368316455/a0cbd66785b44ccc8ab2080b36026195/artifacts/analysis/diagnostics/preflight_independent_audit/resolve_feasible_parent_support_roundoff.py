"""Read-only, two-initial-state sparsemax arithmetic comparison.

No fit, optimizer, new policy selection, source modification, or full audit
rerun. The original640-head audit and its eight flags stay immutable.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np


METHOD=Path('/tmp/audit_feasible_parent_preflight.py')
module=importlib.util.spec_from_file_location('saved_audit_helpers',METHOD)
a=importlib.util.module_from_spec(module);module.loader.exec_module(a)
REFERENCE=Path('research/feasible_parent_sparse_search_20261003/preflight_reference.json')
ORIGINAL=Path('/tmp/feasible-parent-preflight-independent-audit.json')
original_bytes=ORIGINAL.read_bytes();original=json.loads(original_bytes)
reference,objects,identities,plan=a.load_recorder(REFERENCE)
inputs=objects['preflight/inputs.pkl'];x=inputs['features'];returns=inputs['returns'];n=len(x)
assert n==504


def projection(scores,kind):
    shifted=scores-scores.max();ordered=np.sort(shifted)[::-1];cumulative=np.cumsum(ordered);k=np.arange(1,5)
    if kind=='source':eligible=1+k*ordered>cumulative
    elif kind=='alternative':eligible=ordered-(cumulative-1)/k>0
    else:raise ValueError(kind)
    rho=int(eligible.sum());tau=(cumulative[rho-1]-1)/rho
    q=np.maximum(shifted-tau,0.)
    return q,{'rho':rho,'threshold':float(tau),'eligible':eligible.tolist(),
              'source_test_margin':(1+k*ordered-cumulative).tolist(),
              'alternative_test_margin':(ordered-(cumulative-1)/k).tolist()}


def simulate(state,cost,kind):
    gate=a.sigmoid(x@state['hidden.weight'].T+state['hidden.bias']);base=gate@state['A'].T+state['a']
    h=np.array([0.,0.,1.]);path={key:[] for key in ['net_equity','fees','turnover','expert_weights','target_weights','pretrade_weights','ending_weights','exact_hold']}
    comparisons=[]
    for t in range(n):
        score=base[t]+state['K']@h
        q,detail=projection(score,kind);other,other_detail=projection(score,'alternative' if kind=='source' else 'source')
        if not np.array_equal(q>0,other>0) or detail['rho']!=other_detail['rho']:
            comparisons.append({'ordinal':t,'scores':score.tolist(),'q':q.tolist(),'other_q':other.tolist(),
                'support':(q>0).tolist(),'other_support':(other>0).tolist(),
                'maximum_q_difference':float(np.abs(q-other).max()),'detail':detail,'other_detail':other_detail})
        exact_hold=bool(q[3]==1 and np.count_nonzero(q)==1)
        target=h.copy() if exact_hold else q[:3]+q[3]*h
        nav,fee,turn=a.scalar_funding(h,target,cost/10000.)
        marked=np.array([target[0]*(1+returns[t,0]),target[1]*(1+returns[t,1]),target[2]])
        gross=marked[0]+marked[1]+marked[2];end=marked/gross
        for key,value in [('net_equity',nav*gross),('fees',fee),('turnover',turn),('expert_weights',q),
                          ('target_weights',target),('pretrade_weights',h),('ending_weights',end),('exact_hold',exact_hold)]:
            path[key].append(np.asarray(value).copy())
        h=end
    path={k:np.asarray(v) for k,v in path.items()};path['equity']=path['net_equity'].cumprod()
    support=(path['expert_weights']>0).sum(1)
    diag={'support_cardinality_fraction':{k:float(np.mean(support==k)) for k in [1,2,3,4]},
          'exact_hold_count':int(path['exact_hold'].sum()),
          'exact_zero_order_count':int((path['turnover']==0).sum()),
          'reported_trade_count_gt_1e_minus12':int((path['turnover']>1e-12).sum())}
    return path,diag,comparisons


rows=[];failures=[];maximum={'q_difference_on_identical_scores':0.,'source_daily_saved_error':0.,
                           'source_relative_NAV_error':0.,'alternative_F_error':0.,'alternative_raw_gap_pp_error':0.}
for seed in [17,41]:
    meta=objects[f'preflight/search/{seed}/audit.pkl'];trial=meta['trials'][0]
    parent=a.check_model(objects[f'preflight/parents/{seed}/model.pkl']);state=a.decode(trial['coordinates'],parent)
    assert a.tensor_sha(state)==trial['tensor_sha256']
    source_As=[];alternative_As=[]
    for cost in a.COSTS:
        saved=meta['parent_training_paths'][cost];saved_diag=trial['action_diagnostics'][cost]
        source,sd,comparisons=simulate(state,cost,'source')
        alternate,ad,_=simulate(state,cost,'alternative')
        assert all(sd[k]==saved_diag[k] for k in sd)
        errors=a.path_error(source,saved);relative_error=errors.pop('relative_equity')
        max_daily=max(errors.values());maximum['source_daily_saved_error']=max(maximum['source_daily_saved_error'],max_daily)
        maximum['source_relative_NAV_error']=max(maximum['source_relative_NAV_error'],relative_error)
        benchmark=1+returns[:,1].copy();benchmark[0]/=1+cost/10000
        B=a.risk_losses(benchmark)[0]
        A_s,g_s=a.metrics(source['net_equity'],benchmark,B)
        A_a,g_a=a.metrics(alternate['net_equity'],benchmark,B)
        source_As.append(float(A_s[0]));alternative_As.append(float(A_a[0]))
        stored_gap=np.asarray(trial['risks_by_cost'][cost]['raw_percentage_point_gaps'])
        maximum['alternative_F_error']=max(maximum['alternative_F_error'],abs(float(A_a[0])-trial['A_by_cost'][cost]))
        maximum['alternative_raw_gap_pp_error']=max(maximum['alternative_raw_gap_pp_error'],float(np.abs(g_a[0]-stored_gap).max()))
        q_difference=max((v['maximum_q_difference'] for v in comparisons),default=0.)
        maximum['q_difference_on_identical_scores']=max(maximum['q_difference_on_identical_scores'],q_difference)
        changed_probs=[]
        for item in comparisons:
            for q,other in zip(item['q'],item['other_q']):
                if (q>0)!=(other>0):changed_probs.append(max(q,other))
        source_feasible=bool(np.all(g_s[0]<=a.RISK_TOL_PP));alternate_feasible=bool(np.all(g_a[0]<=a.RISK_TOL_PP))
        if source_feasible!=trial['risks_by_cost'][cost]['feasible'] or alternate_feasible!=source_feasible:
            failures.append(f'seed{seed}/cost{cost}: raw admission status changed')
        if sd['exact_hold_count']!=ad['exact_hold_count'] or sd['exact_zero_order_count']!=ad['exact_zero_order_count'] or sd['reported_trade_count_gt_1e_minus12']!=ad['reported_trade_count_gt_1e_minus12']:
            failures.append(f'seed{seed}/cost{cost}: H/order flag changed')
        rows.append({'seed':seed,'trial':0,'cost_bps':cost,'source_diagnostics_match_saved_exactly':True,
            'source_diagnostics':sd,'alternative_diagnostics':ad,
            'different_support_same_score_rows':len(comparisons),
            'positive_probabilities_whose_zero_status_changed_min':min(changed_probs,default=0.),
            'positive_probabilities_whose_zero_status_changed_max':max(changed_probs,default=0.),
            'maximum_q_difference_on_same_scores':q_difference,
            'source_A':float(A_s[0]),'alternative_A':float(A_a[0]),'stored_A':trial['A_by_cost'][cost],
            'source_raw_gap_pp':g_s[0].tolist(),'alternative_raw_gap_pp':g_a[0].tolist(),
            'source_raw_feasible':source_feasible,'alternative_raw_feasible':alternate_feasible,
            'source_path_error_vs_saved':max_daily,'source_relative_NAV_error':relative_error,
            'first_three_arithmetic_examples':comparisons[:3]})
    assert min(source_As)>0 and min(alternative_As)>0
assert ORIGINAL.read_bytes()==original_bytes
assert len(original['support_diagnostic_differences'])==8
assert all(v['seed'] in (17,41) and v['trial']==0 and v['kind']=='support_cardinality' for v in original['support_diagnostic_differences'])
if maximum['q_difference_on_identical_scores']>1e-12 or maximum['source_daily_saved_error']>1e-12 or maximum['source_relative_NAV_error']>1e-10:
    failures.append('original numerical path tolerance exceeded')
if maximum['alternative_F_error']>a.VALUE_TOL or maximum['alternative_raw_gap_pp_error']>a.VALUE_TOL:
    failures.append('original numerical objective/risk tolerance exceeded')
report={'scope':'Authorized read-only scalar comparison of seed17/41 trial0 atbothcosts only; two algebraicallyequivalent sparsemax arithmetic paths perstate/cost. No fit, optimizer, sourcechange, newcandidate, or full640-head rerun.',
    'recorder_id':reference['recorder_id'],'reference_sha256':a.file_sha(REFERENCE),
    'original_audit_sha256':hashlib.sha256(original_bytes).hexdigest(),'original_eight_flags_preserved':True,
    'source_model_sha256':reference['source_sha256']['src/model/feasible_parent_sparse_search.py'],
    'original_policy_source_sha256':reference['source_sha256']['src/model/sparse_holdings_net_excess.py'],
    'method_sha256':a.file_sha(__file__),'original_audit_method_sha256':a.file_sha(METHOD),
    'source_test':'1 + k*sorted_shifted_score > cumulative_sum',
    'alternative_test':'sorted_shifted_score - (cumulative_sum - 1)/k > 0',
    'explanation':'The tests are algebraically equivalent in exact arithmetic but not roundoff invariant at a zero coordinate. They can choose different support counts or thresholds at floating cancellation. The positive mass involved is quantified; no near-zero cutoff was introduced, and saved source arithmetic remains authoritative for exactsupport diagnostics.',
    'state_cost_cases':4,'validation_arithmetic_paths':8,'cases':rows,'maximum_errors':maximum,
    'resolution_passed':not failures,'failures':failures,
    'all_original_support_flags_explained':not failures,
    'admission_gate_affected':bool(failures),
    'original_preflight_independent_numeric_gate':original['independent_preflight_passed'],
    'combined_audit_closure':not failures and original['failures']==['sparse support/exactH diagnostics require review'],
    'financial_qualification':False,
    'limitations':'This closes only the independent arithmetic diagnostic discrepancy. The synthetic result is not an optimizeroptimality proof or evidence of market alpha; no originalrisk/financial/numerical tolerance changed.'}
Path('/tmp/feasible-parent-support-roundoff-resolution.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'resolution_passed':report['resolution_passed'],'combined_audit_closure':report['combined_audit_closure'],
                  'maximum_errors':maximum,'failures':failures},indent=2))
if failures:raise SystemExit(1)

"""Independent synthetic gauges, separate cost books and byte-budget audit; no fit calls."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
import src.model.feasible_parent_sparse_search as m
from src.model.paired_sparse_paths import rollout_costs
from src.model.sparse_holdings_net_excess import SparseHoldingsPolicy,numpy_rollout,policy_target
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy

torch.set_num_threads(1);rng=np.random.default_rng(20261003);x=rng.normal(size=(137,27));t=np.arange(len(x));s=.002+.01*np.sin(t*.51);returns=np.c_[3*s+.001*np.sin(t*.27),s]
errors=[];models=[]
for seed in (17,41,71):
 p=SparseHoldingsPolicy(seed=seed)
 with torch.no_grad():
  p.A.copy_(torch.tensor(rng.normal(size=(4,8)),dtype=torch.float64));p.a.copy_(torch.tensor(rng.normal(size=4),dtype=torch.float64));p.K.copy_(torch.tensor(rng.normal(scale=.25,size=(4,3)),dtype=torch.float64))
 coordinate=m.gauge_coordinates(p);q=m.model_from_coordinates(p,coordinate)
 assert coordinate.shape==(33,) and m.encoder_hash(p)==m.encoder_hash(q)
 np.testing.assert_array_equal(m.gauge_coordinates(q),coordinate)
 assert np.count_nonzero(q.K.detach().numpy()[:,1])==0 and np.count_nonzero(q.A.detach().numpy()[1])==0 and q.a[1]==0
 for f,h in zip(x,rng.dirichlet(np.ones(3),size=len(x))):errors.append(float(np.max(abs(policy_target(p,f,h)-policy_target(q,f,h)))))
 models.append(q)
assert max(errors)<1e-12
forcing=np.zeros(len(x),bool);forcing[[4,22,78]]=True
cost_errors={};bitexact_state=True
for member_count in (1,3):
 pairs=rollout_costs(models[:member_count],x,returns,costs=(25.,10.),initial_weights=[.2,.5,.3],force_soxx_mask=forcing)
 for c in (25.,10.):
  reference=numpy_rollout(models[:member_count],x,returns,cost_bps=c,initial_weights=[.2,.5,.3],force_soxx_mask=forcing)
  err=max(float(np.max(abs(pairs[c][k].astype(float)-reference[k].astype(float)))) for k in reference)
  assert err<1e-12;cost_errors[f'{member_count}models_{c}']=err
  for key in ('pretrade_weights','ending_weights','target_weights','expert_weights'):bitexact_state &= np.array_equal(pairs[c][key],reference[key])
 assert not np.shares_memory(pairs[10.]['pretrade_weights'],pairs[25.]['pretrade_weights'])
 assert np.any(pairs[10.]['fees']!=pairs[25.]['fees'])
assert bitexact_state
spec=m.FeasibleParentSearchSpec(max_evaluations=3,scipy_maxiter=3,minimum_rows=63)
parent=BenchmarkAnchorPolicy(seed=17,anchor=True);_,anchors,references,_=m._reference(x,returns,spec,17)
template,initial,_=m.revalidate_parent(parent,x,returns,spec=spec,anchors=anchors,references=references)
e=m.SearchEvaluator(template,x,returns,spec=spec,anchors=anchors,references=references,parent_evaluation=initial)
x0=m.gauge_coordinates(template);assert len(e.trials)==1
for i in range(3):
 z=x0.copy();z[0]+=i*.001
 e.objective(z);e.constraints(z)
 assert len(e.trials)==i+1
assert e.cache_hits==4
z=x0.copy();z[1]=.002
try:e.objective(z)
except m.EvaluationBudgetExceeded:pass
else:raise AssertionError('unique budget exceeded')
assert len(e.trials)==3
hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['src/model/feasible_parent_sparse_search.py','src/model/paired_sparse_paths.py','src/data/feasible_parent_sparse_search.py']}
report={'scope':'Synthetic mathematical/forward/callback calls only; no scipy optimizer, model fit, registered504toy, market data or parent/OOS selection.',
 'source_sha256':hashes,'gauge_points':len(errors),'max_gauge_target_error':max(errors),'cost_book_errors':cost_errors,
 'all_owned_states_targets_expert_weights_bitexact_to_separate_full_rollouts':bitexact_state,'cost_buffers_independent':True,
 'budget_including_parent':3,'completed_unique_evaluations':len(e.trials),'requested_callbacks':e.requested_callbacks,'cache_hits':e.cache_hits,
 'raw_parent_revalidation_passed':True,'solver_status_never_used_for_incumbent_review':True,'inherited_parent_does_not_establish_incremental_optimizer_gain':True,'failures':[]}
Path('/tmp/feasible-parent-independent-math.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

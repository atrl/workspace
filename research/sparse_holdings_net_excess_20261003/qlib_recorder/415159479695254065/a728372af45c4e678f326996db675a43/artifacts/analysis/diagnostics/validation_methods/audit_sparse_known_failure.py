"""Independent cached synthetic-checkpoint audit; no fit or optimizer call."""
from pathlib import Path
import hashlib,json,pickle
import numpy as np

payload_path=Path('/tmp/sparse-holdings-known-information-full.pkl');raw=payload_path.read_bytes();bundle=pickle.loads(raw)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=lambda b:hashlib.sha256(b).hexdigest()
x=np.asarray(bundle['features'],float);r=np.asarray(bundle['returns'],float);N=504;candidates=[]
assert x.shape==(504,27) and r.shape==(504,2)
sign=np.where(np.arange(N)%2==0,1.,-1.)
np.testing.assert_array_equal(x[:,0],sign);assert not np.any(x[:,1:]);np.testing.assert_array_equal(r[:,1],.002+.008*sign);np.testing.assert_array_equal(r[:,0],3*r[:,1])
spec=bundle['spec'];assert spec.updates==120 and spec.learning_rate==.003 and spec.dual_learning_rate==.003 and spec.gradient_clip==5 and tuple(spec.cost_bps)==(10.,25.)
assert spec.context_sessions==21 and spec.gradient_chunk_length==63 and spec.risk_group_length==63

def projection(v):
 shifted=v-max(v);ordered=np.sort(shifted)[::-1];sums=np.cumsum(ordered)
 k=np.where(ordered-(sums-1)/np.arange(1,5)>0)[0][-1]+1
 q=np.maximum(shifted-(sums[k-1]-1)/k,0.)
 assert min(q)>=0 and abs(sum(q)-1)<1e-12
 return q

signs=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
def funding(h,z,c):
 if np.array_equal(h,z):return 1.,0.,0.
 roots=(1+c*(signs*h[:2]).sum(1))/(1+c*(signs*z[:2]).sum(1))
 residual=np.abs(roots+c*np.abs(roots[:,None]*z[:2]-h[:2]).sum(1)-1)
 n=roots[np.argmin(residual)];turn=np.abs(n*z[:2]-h[:2]).sum();fee=c*turn
 assert abs(n+fee-1)<1e-12
 return float(n),float(fee),float(turn)

def model_check(p):
 assert p['schema_version']=='sparse_holdings_v1' and p['parameter_count']==272 and p['dtype']=='float64'
 d=p['state_dict'];hashobj=hashlib.sha256()
 for name in sorted(d):
  a=np.asarray(d[name]);assert a.dtype==np.float64 and np.isfinite(a).all()
  hashobj.update(name.encode());hashobj.update(str(a.shape).encode());hashobj.update(a.tobytes())
 assert sum(v.size for v in d.values())==272 and hashobj.hexdigest()==p['tensor_sha256']
 return d

def simulate(p,cost):
 d=model_check(p);pre=x@d['hidden.weight'].T+d['hidden.bias'];gate=1/(1+np.exp(-pre));base=gate@d['A'].T+d['a']
 held=np.array([0.,0.,1.]);nets=[];fees=[];turns=[];qs=[];targets=[];states=[]
 for i in range(N):
  q=projection(base[i]+d['K']@held)
  target=held.copy() if q[3]==1 and np.count_nonzero(q)==1 else q[:3]+q[3]*held
  n,fee,turn=funding(held,target,cost/10000)
  marked=target*np.r_[1+r[i],1.];gross=marked.sum();net=n*gross
  assert net>0 and np.isfinite(net);nets.append(net);fees.append(fee);turns.append(turn);qs.append(q);targets.append(target);states.append(held.copy());held=marked/gross
 return {k:np.array(v) for k,v in [('net',nets),('fees',fees),('turnover',turns),('q',qs),('targets',targets),('pretrade',states)]}

def risks(net):
 log=np.log(net);lv=np.r_[0.,np.cumsum(log)];D=float(-np.expm1(np.min(lv-np.maximum.accumulate(lv))))
 daily=np.sort(np.expm1(log));mass=.05*len(daily);whole=int(mass);C=float(-(daily[:whole].sum()+(mass-whole)*daily[whole])/mass)
 return np.array([D,C])

anchors={};Bs={}
for cost in (10.,25.):
 net=1+r[:,1].copy();net[0]/=1+cost/10000;anchors[cost]=net;Bs[cost]=risks(net)

maximum={'F':0.,'raw_gap_pp':0.,'normalized_g':0.,'dual_update':0.,'risk_percent':0.,'selected_saved_path':0.};all_rows=[];seed_rows=[];failures=[]
for fit in bundle['fits']:
 seed=fit['seed'];metadata=fit['metadata'];epochs=metadata['epochs'];assert len(epochs)==121 and [e['epoch'] for e in epochs]==list(range(121))
 assert metadata['failure'] is None and metadata['optimizer_diagnostics']['updates_completed']==120
 assert metadata['selected_epoch']=='anchor' and metadata['F']==0 and metadata['null_selected']
 previous_dual=np.zeros(4);rows=[]
 for e in epochs:
  A={};gaps=[];norms=[];costrows={}
  for cost in (10.,25.):
   p=simulate(e['checkpoint'],cost);logs=np.log(p['net']);relative=logs-np.log(anchors[cost]);A[cost]=float(63/N*relative.sum())
   risk=risks(p['net']);raw_gap=100*(risk-Bs[cost]);g=(risk-Bs[cost])/Bs[cost]
   gaps.extend(raw_gap.tolist());norms.extend(g.tolist());saved=e['risks_by_cost'][cost]
   maximum['raw_gap_pp']=max(maximum['raw_gap_pp'],float(np.max(abs(raw_gap-np.array(saved['raw_percentage_point_gaps'])))))
   maximum['risk_percent']=max(maximum['risk_percent'],abs(-100*risk[0]-saved['max_drawdown_pct']),abs(-100*risk[1]-saved['cvar5_pct']))
   assert abs(A[cost]-e['A_by_cost'][cost])<1e-10
   diag=e['action_diagnostics'][cost];q=p['q'];support=(q>0).sum(1)
   for k in (1,2,3,4):assert abs(np.mean(support==k)-diag['support_cardinality_fraction'][k])<1e-12
   for j,name in enumerate(('soxl','soxx','cash','hold')):assert abs(np.mean((q[:,j]==1)&(support==1))-diag['sole_expert_fraction'][name])<1e-12
   costrows[str(int(cost))]={'raw_gap_pp':raw_gap.tolist(),'positive_signal_mean_q':q[sign>0].mean(0).tolist(),'negative_signal_mean_q':q[sign<0].mean(0).tolist(),'active_experts':np.flatnonzero(np.any(q>0,axis=0)).tolist(),'all_SOXX_support':bool(np.all(q[:,1]==1))}
  F=min(A.values());feasible=bool(np.all(np.array(gaps)<=1e-10))
  maximum['F']=max(maximum['F'],abs(F-e['F']));maximum['normalized_g']=max(maximum['normalized_g'],float(np.max(abs(np.array(norms)-e['postg']))))
  assert feasible==e['feasible'] and not (feasible and F>1e-10)
  used=np.asarray(e['lambda_used']);next_dual=np.asarray(e['lambda_next']);assert np.array_equal(used,previous_dual)
  expected=used if e['epoch']==0 else np.maximum(0,used+.003*np.asarray(e['postg']))
  maximum['dual_update']=max(maximum['dual_update'],float(np.max(abs(expected-next_dual))));assert np.array_equal(expected,next_dual)
  previous_dual=next_dual
  assert e['selected_epoch_so_far']=='anchor'
  row={'seed':seed,'epoch':e['epoch'],'F':e['F'],'independent_F':F,'feasible':feasible,'raw_gap_pp':gaps,'lambda_next':next_dual.tolist(),'costs':costrows,
   'zero_gradient_update':bool(e['gradient'] and e['gradient']['all_parameter_gradients_exact_zero']),
   'max_forward_parity_error':e['gradient']['forward_parity_max_error'] if e['gradient'] else None,'support_mismatches':e['gradient']['support_mismatch_count'] if e['gradient'] else None}
  rows.append(row);all_rows.append(row)
 # Independently reconstruct the retained exactnull's saved training path.
 for cost in (10.,25.):
  p=simulate(fit['selected_model'],cost);saved=metadata['training_paths'][cost]
  for key,stored in [('net','net_equity'),('fees','fees'),('turnover','turnover'),('targets','target_weights'),('pretrade','pretrade_weights')]:maximum['selected_saved_path']=max(maximum['selected_saved_path'],float(np.max(abs(p[key]-saved[stored]))))
  assert np.max(abs(p['net']-anchors[cost]))<1e-12
 positive=[e for e in epochs if e['F']>0];feasible=[e for e in epochs if e['feasible']];zero=[e['epoch'] for e in epochs if e['gradient'] and e['gradient']['all_parameter_gradients_exact_zero']]
 best=max(rows,key=lambda e:e['F']);last=rows[-1]
 singleton=[e['epoch'] for e in rows if all(c['all_SOXX_support'] for c in e['costs'].values())]
 seed_rows.append({'seed':seed,'positive_F_states':len(positive),'feasible_states':len(feasible),'feasible_positive_F_states':sum(e['feasible'] and e['F']>0 for e in epochs),
 'first_feasible_epoch':feasible[0]['epoch'] if feasible else None,'first_all_SOXX_singleton_epoch':singleton[0] if singleton else None,
 'zero_gradient_updates':len(zero),'first_zero_gradient_update':zero[0] if zero else None,'maximum_F_state':best,'final_state':last})
 print('audited seed',seed,flush=True)
# Registered finite witness, directly specified as expertweights, no construction search/fitting.
gate=1/(1+np.exp(-2*sign));witness_q=np.c_[.5*gate,np.full(N,.5),.5*(1-gate),np.zeros(N)];witness={}
for cost in (10.,25.):
 held=np.array([0.,0.,1.]);nets=[]
 for i,z in enumerate(witness_q[:,:3]):
  n,_,_=funding(held,z,cost/10000);marked=z*np.r_[1+r[i],1.];nets.append(n*sum(marked));held=marked/sum(marked)
 net=np.array(nets);witness[cost]={'A':float(63/N*np.sum(np.log(net)-np.log(anchors[cost]))),'raw_gap_pp':(100*(risks(net)-Bs[cost])).tolist()}
assert min(v['A'] for v in witness.values())>0 and all(max(v['raw_gap_pp'])<=1e-10 for v in witness.values())
assert abs(min(v['A'] for v in witness.values())-bundle['prior_probe']['witness']['F'])<1e-10
assert maximum['F']<1e-10 and maximum['raw_gap_pp']<1e-10 and maximum['normalized_g']<1e-9 and maximum['dual_update']==0 and maximum['selected_saved_path']<1e-12
report={'scope':'Independent NumPy reconstruction of all363 saved synthetic training checkpoints at both costs, registered finite witness and retained anchor paths; no fitting, optimizer calls, new model choice, market data or OOS. Fullinput payload immutable.',
 'payload_path':str(payload_path),'payload_sha256':digest(raw),'model_source_sha256_in_payload':bundle['source_sha256'],
 'math_data_contracts_passed':True,'known_information_feasible_training_gate_passed':False,'market_fits_permitted':False,
 'checkpoint_count':363,'checkpoint_cost_paths':726,'training_decisions_reconstructed':363*2*N,'maximum_reconstruction_errors':maximum,
 'registered_witness':witness,'seeds':seed_rows,'failures':failures,
 'interpretation':'All positive-F learned states violate at least one original raw tail gate. Seeds17/41 enter all-SOXX singleton support at epochs52/53, after which every projected-score Jacobian is0 and updates53/54 onward have zero parameter gradients; later Adam momentum can move parameters without informative gradients. Seed71 remains a non-singleton SOXX/HOLD policy with nonzero gradients, but its first feasible state is epoch85 and every feasible state has nonpositiveF. This is finite-budget constrained optimization failure with observed sparse-support saturation in two starts, not evidence of class infeasibility, no signal, or a demonstrated arithmetic bug. Prior off-kink derivative audits passed; those do not establish optimizer convergence or certify every nonsmooth trajectory gradient.',
 'selection_status':'The stored exact SOXX incumbent and ensemble F0 are the correct registered rejection, not a passing information-use control. Risk and positive-F gates are unchanged.'}
assert payload_path.read_bytes()==raw
Path('/tmp/sparse-holdings-validation/known-information-independent-audit.json').write_text(json.dumps(report,indent=2)+'\n')
Path('/tmp/sparse-holdings-validation/known-information-independent-epochs.json').write_text(json.dumps(all_rows,indent=2)+'\n')
print(json.dumps(report,indent=2))

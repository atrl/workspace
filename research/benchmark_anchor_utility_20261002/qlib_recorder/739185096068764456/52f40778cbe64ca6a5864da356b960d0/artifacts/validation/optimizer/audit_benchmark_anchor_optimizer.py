import os
os.environ['MLFLOW_ALLOW_FILE_STORE']='true';os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np,pandas as pd,qlib
from qlib.workflow import R
ROOT=Path('/Users/rziiiii/dev/workspace-worktrees/benchmark-anchor-utility-20261002')
SUMMARY=ROOT/'research/benchmark_anchor_utility_20261002/summary.json'
s=json.loads(SUMMARY.read_text());assert s['recorder_id']!='57c3446d79b74e87b9ec7be9a1f7a859', 'Initial lifecycle-failed run is not authority'
qlib.init(provider_uri=str(ROOT/s['provider_uri']),region='us');R.set_uri(s['recorder_uri'])
r=R.get_recorder(experiment_id=Path(s['recorder_local_dir']).parent.name,recorder_id=s['recorder_id'])
load=r.load_object
fits=load('analysis/training_diagnostics.pkl');ensemble_audits=load('analysis/ensemble_audits.pkl')
folds={**load('inputs/development_fold_inputs.pkl'),**load('inputs/diagnostic_fold_inputs.pkl')}
models=load('models/development.pkl')
for variant,years in load('models/selected_annual_refits.pkl').items():models.setdefault(variant,{}).update(years)
failures=[];maxima={};counts=Counter();fit_rows=[];ensemble_rows=[];seed_initials={};constant_models={}
def ck(value,label):
 if not bool(value):failures.append(label)
def maxdiff(a,b,label,tol=1e-10):
 value=float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0));maxima[label]=max(maxima.get(label,0),value);ck(np.isfinite(value) and value<=tol,label);return value
def groups(n,length):
 starts=list(range(0,n-length+1,length))
 if starts[-1]!=n-length:starts.append(n-length)
 return [(i,i+length) for i in starts]
def utility(net,benchmark,spec):
 x=np.log(net)-np.log(benchmark);g=groups(len(x),spec['risk_group_length']);q=np.array([x[a:b].sum() for a,b in g]);D=np.sqrt(np.minimum(q,0).dot(np.minimum(q,0))/len(q)+spec['rms_epsilon'])
 return float(spec['risk_group_length']/len(x)*x.sum()-spec['downside_coefficient']*D+spec['downside_coefficient']*np.sqrt(spec['rms_epsilon']))
def tensor_audit(p):
 state=p['state_dict'];theta=state['theta'];ck(p['dtype']=='float64' and theta.shape==(3,9),'tensor schema');ck(all(np.isfinite(v).all() for v in state.values()),'finite checkpoint');ck(theta.min()>=0 and theta.sum()<=1+1e-12,'theta simplex')
 digest=hashlib.sha256()
 for name in sorted(state):digest.update(name.encode());digest.update(str(state[name].shape).encode());digest.update(state[name].tobytes())
 ck(digest.hexdigest()==p['tensor_sha256'],'checkpoint hash');counts['checkpoint_tensors']+=1
 return state
def alphas(payloads,x):
 result=np.zeros((len(x),3))
 for p in payloads:
  st=p['state_dict'];z=x@st['hidden.weight'].T+st['hidden.bias'];gate=np.empty_like(z);mask=z>=0;gate[mask]=1/(1+np.exp(-z[mask]));ex=np.exp(z[~mask]);gate[~mask]=ex/(1+ex);result+=np.c_[np.ones(len(x)),gate]@st['theta'].T/len(payloads)
 return result
def path_audit(path,ret,alpha,cost,label):
 h=path['pretrade_weights'];z=path['target_weights'];w=path['open_net_weights'];fees=path['fees'];net=path['net_equity'];a=alpha
 expected=np.c_[a[:,0]+a[:,2]*h[:,0],1-a.sum(1)+a[:,2]*h[:,1],a[:,1]+a[:,2]*h[:,2]]
 maxdiff(expected,z,'checkpoint_action_identity');maxdiff(h[0],[0,0,1],'cash_origin');maxdiff(h[1:],path['ending_weights'][:-1],'carried_training_holdings')
 maxdiff(z,w,'net_weight_identity');maxdiff(1-fees,path['open_net_equity'],'cash_funding_identity');post=(1-fees)[:,None]*w
 maxdiff(np.abs(post[:,:2]-h[:,:2]).sum(1),path['turnover'],'true_risky_L1');maxdiff(path['turnover']*cost/10000,fees,'fee_identity')
 maxdiff((1-fees)*(1+np.sum(w[:,:2]*ret,axis=1)),net,'return_identity');maxdiff(path['equity'],np.cumprod(net),'cumulative_NAV',tol=1e-8)
 ck(np.isfinite(net).all() and (net>0).all(),'positive net multipliers');ck(w.min()>=-1e-12,'long only training')
 counts['audited_training_decisions']+=len(net)
ck(len(fits)==75,'expected75fits');ck(len(ensemble_audits)==25,'expected25ensembles')
for info in fits:
 key=(info['variant'],info['year']);seed=info['seed'];fold=folds[key];bundle=models[key[0]][key[1]];selected=bundle['seed_models'][seed];spec=info['spec'];history=info['epochs'];label=str((*key,seed));counts['fits']+=1
 ck(info['failure'] is None and info['qualification_eligible'],'fit numerical qualification '+label);ck(len(history)==121 and [h['epoch'] for h in history]==list(range(121)),'all epochs '+label);ck(info['optimizer_diagnostics']['updates_completed']==120,'updates count')
 x=np.asarray(fold['features'],float);ret=np.asarray(fold['returns'],float);n=len(x)
 ck(info['training_rows']==n and hashlib.sha256(x.tobytes()).hexdigest()==info['training_feature_sha256'],'training feature binding');ck(hashlib.sha256(ret.tobytes()).hexdigest()==info['training_return_sha256'],'training return binding')
 expected_groups=groups(n,spec['risk_group_length']);ck([tuple(a) for a in info['risk_groups']]==expected_groups,'groups all observations')
 best=0.;best_epoch='anchor';below_incumbent=0
 for epoch,hist in enumerate(history):
  st=tensor_audit(hist['checkpoint']);j=float(hist['J']);cu=hist['centered_utility'];ck(j==min(cu.values()),'worst aggregate cost branch')
  active=[c for c,v in cu.items() if v==j];ck(hist['cost_weights']=={c:1/len(active) if c in active else 0. for c in cu},'symmetric exact tie or worst cost')
  if j>best:best=j;best_epoch=epoch
  elif j<best:below_incumbent+=1
  ck(hist['selected_epoch_so_far']==(-1 if best_epoch=='anchor' else best_epoch),'strict sequential incumbent '+label)
  if epoch==0:
   maxdiff(st['theta'],np.full((3,9),1/54),'interior theta initialization',tol=0)
   maxdiff(st['hidden.bias'],np.zeros(8),'zero hidden bias',tol=0)
   initial_hash=hashlib.sha256(st['hidden.weight'].tobytes()).hexdigest();ck(seed_initials.setdefault(seed,initial_hash)==initial_hash,'fresh seeded hidden initialization')
  else:
   grad=hist['gradient'];ck(grad['parameter_epoch']==epoch-1,'trajectory gradient parameter epoch');ck(grad['scored_rows']==n,'gradient every row once')
   for field in ['surrogate_loss','theta_gradient_norm','hidden_gradient_norm','unclipped_global_norm']:ck(np.isfinite(grad[field]),'finite gradient '+field)
   val=grad['forward_parity_max_error'];maxima['current_trajectory_numpy_torch_parity']=max(maxima.get('current_trajectory_numpy_torch_parity',0),val);ck(val<=1e-10,'current cache parity')
   expected_boundaries=[(max(0,a-spec['context_sessions']),a,min(n,a+spec['gradient_chunk_length'])) for a in range(0,n,spec['gradient_chunk_length'])]
   ck([tuple(v) for v in grad['chunk_boundaries']]==expected_boundaries,'true chunk coverage contract');counts['gradient_updates']+=1
  counts['epoch_records']+=1
 ck(info['selected_epoch']==best_epoch,'selected first strict best '+label);maxdiff(info['centered_worst_utility'],best,'selected utility best',tol=0);tensor_audit(selected)
 if best_epoch=='anchor':ck(np.count_nonzero(selected['state_dict']['theta'])==0,'selected anchor is theta0');counts['null_seed_fits']+=1
 else:ck(selected['tensor_sha256']==history[best_epoch]['checkpoint']['tensor_sha256'],'selected complete checkpoint')
 alpha=alphas([selected],x);calculated={}
 for cost,path in info['training_paths'].items():
  benchmark=info['benchmark_net_equity'][cost];analytic=1+ret[:,1].copy();analytic[0]/=1+float(cost)/10000;maxdiff(analytic,benchmark,'independent analytic anchor',tol=1e-12)
  for field in ['maximum_multiplier_error','maximum_relative_wealth_error','initial_entry_fee_error','maximum_later_fee']:ck(info['anchor_audit'][cost][field]<=1e-10,'recorded anchor '+field)
  path_audit(path,ret,alpha,float(cost),label);u=utility(path['net_equity'],benchmark,spec);calculated[cost]=u;maxdiff(u,info['centered_utility_by_cost'][cost],'selected utility independent');ck(u>=-1e-12,'training incumbent no worse anchor both costs')
 if key[0]=='constant':
  q=(key[1],seed);constant_models[q]={'theta':selected['state_dict']['theta'],'bias':selected['state_dict']['hidden.bias'],'paths':info['training_paths'],'selected_epoch':best_epoch}
 fit_rows.append({'variant':key[0],'year':key[1],'seed':seed,'rows':n,'selected_epoch':best_epoch,'J':best,'centered_by_cost':calculated,'null_selected':info['null_selected'],'trajectory_states_below_incumbent':below_incumbent,'theta_mass':float(selected['state_dict']['theta'].sum()),'max_hidden_gradient_norm':max(h['gradient']['hidden_gradient_norm'] for h in history[1:]),'max_theta_gradient_norm':max(h['gradient']['theta_gradient_norm'] for h in history[1:])})
for e in ensemble_audits:
 key=(e['variant'],e['year']);fold=folds[key];bundle=models[key[0]][key[1]];x=np.asarray(fold['features'],float);ret=np.asarray(fold['returns'],float);spec=next(f['spec'] for f in fits if (f['variant'],f['year'])==key)
 ck(e['component_count']==3 and e['all_three_seed_models_used'],'all ensemble seeds used');rejected=e['ensemble_rejected_for_training_utility'];ck(rejected==(e['ensemble_J']<0),'ensemble acceptance threshold')
 accepted=bundle['ensemble_models'];alpha=alphas(accepted,x);accepted_u={};raw_u={}
 for cost,path in e['training_paths'].items():
  baseline=1+ret[:,1].copy();baseline[0]/=1+float(cost)/10000;path_audit(path,ret,alpha,float(cost),'ensemble '+str(key));u=utility(path['net_equity'],baseline,spec);accepted_u[cost]=u;maxdiff(u,e['centered_utility_by_cost'][cost],'ensemble selected utility');ck(u>=-1e-10,'ensemble incumbent no worse anchor')
 raw_paths=e['rejected_ensemble_training_paths'] if rejected else e['training_paths'];raw_alpha=alphas([bundle['seed_models'][seed] for seed in [17,41,71]],x)
 for cost,path in raw_paths.items():
  if rejected:path_audit(path,ret,raw_alpha,float(cost),'rejected ensemble '+str(key))
  baseline=1+ret[:,1].copy();baseline[0]/=1+float(cost)/10000;raw_u[cost]=utility(path['net_equity'],baseline,spec)
  maxdiff(raw_u[cost],e['ensemble_centered_utility'][cost],'raw ensemble utility')
 maxdiff(min(raw_u.values()),e['ensemble_J'],'ensemble worst utility');counts['ensemble_anchor_selected']+=e['anchor_selected'];counts['ensemble_rejected']+=rejected
 ensemble_rows.append({'variant':key[0],'year':key[1],'J':e['ensemble_J'],'selected_J':e['centered_worst_utility'],'rejected':rejected,'anchor_selected':e['anchor_selected'],'raw_centered_utility':raw_u,'selected_centered_utility':accepted_u})
for year in range(2016,2023):
 a=constant_models[(year,17)]
 for seed in [41,71]:
  b=constant_models[(year,seed)];maxdiff(a['theta'],b['theta'],'constant same effective theta',tol=0);maxdiff(a['bias'],b['bias'],'constant same effective bias',tol=0);ck(a['selected_epoch']==b['selected_epoch'],'constant selected epoch same')
  for c in [10,25]:maxdiff(a['paths'][c]['net_equity'],b['paths'][c]['net_equity'],'constant seed paths identical',tol=0)
# Lifecycle-only correction must reproduce every fitted decision object.
initial_r=R.get_recorder(experiment_id=Path(s['recorder_local_dir']).parent.name,recorder_id='57c3446d79b74e87b9ec7be9a1f7a859')
lifecycle={'initial_failed_recorder':'57c3446d79b74e87b9ec7be9a1f7a859','model_comparisons':0,'epoch_objective_comparisons':0,'epoch_tensor_comparisons':0,'ensemble_comparisons':0}
for info in fits:
 variant,year,seed=info['variant'],info['year'],info['seed'];prefix=f'fit_checkpoints/{variant}/{year}/{seed}'
 old_model=initial_r.load_object(prefix+'/model.pkl');old=initial_r.load_object(prefix+'/training_audit.pkl');new_model=models[variant][year]['seed_models'][seed]
 ck(old_model['tensor_sha256']==new_model['tensor_sha256'],'lifecycle selected tensor identity');ck(old['selected_epoch']==info['selected_epoch'],'lifecycle selected epoch');ck(old['failure']==info['failure'],'lifecycle failure')
 for field in ['centered_worst_utility','centered_utility_by_cost','training_feature_sha256','training_return_sha256']:ck(old[field]==info[field],'lifecycle '+field)
 for a,b in zip(old['epochs'],info['epochs']):
  ck(a['J']==b['J'] and a['centered_utility']==b['centered_utility'],'lifecycle all epoch utilities');ck(a['checkpoint']['tensor_sha256']==b['checkpoint']['tensor_sha256'],'lifecycle all checkpoint tensors');lifecycle['epoch_objective_comparisons']+=1;lifecycle['epoch_tensor_comparisons']+=1
 lifecycle['model_comparisons']+=1
for info in ensemble_audits:
 variant,year=info['variant'],info['year'];old=initial_r.load_object(f'fit_checkpoints/{variant}/{year}/ensemble/training_audit.pkl')
 for field in ['ensemble_J','ensemble_centered_utility','ensemble_rejected_for_training_utility','centered_worst_utility','centered_utility_by_cost','anchor_selected','component_count']:ck(old[field]==info[field],'lifecycle ensemble '+field)
 lifecycle['ensemble_comparisons']+=1

comparison=load('analysis/comparison.pkl');controls=load('analysis/controls.pkl');attribution=load('analysis/attribution.pkl')
result={'recorder_id':s['recorder_id'],'lifecycle_reproduction':lifecycle,'failure_count':len(failures),'failures':failures,'counts':dict(counts),'maxima':maxima,'fits':fit_rows,'ensembles':ensemble_rows,'comparison':comparison.to_dict('records'),'controls':controls.to_dict('records'),'attribution':attribution.to_dict('records'),'limitations':['No market refitting, new return path or candidate selection in this audit. Saved target actions, fee funding and utility are independently reconstructed on saved paths.','Every checkpoint recorded objective and strict incumbent ordering are checked; non-selected checkpoint full paths are not stored, so their objective scalars are not independently rerun.','Gradient-cache correctness is supported by frozen source control flow, parameter_epoch linkage and recorded per-update exact forward parity. No global optimum or full-horizon gradient claim.','Three constant-input seeds are structurally identical effective policies, not independent market evidence.']}
for count_name in ['null_seed_fits','ensemble_anchor_selected','ensemble_rejected']:result['counts'].setdefault(count_name,0)
result['lifecycle_reproduction']['all_compared_economics_identical']=not any('lifecycle' in f for f in failures)
portfolio=load('analysis/portfolio_replays.pkl');exposures=[]
for (name,cost),a in portfolio[(portfolio.segment=='development') & (portfolio.seed=='ensemble')].groupby(['variant','cost_bps']):
 exposures.append({'variant':name,'cost_bps':cost,'mean_soxl':float(a.soxl_weight.mean()),'mean_soxx':float(a.soxx_weight.mean()),'mean_cash':float(a.cash_weight.mean()),'mean_nominal_beta':float((3*a.soxl_weight+a.soxx_weight).mean()),'max_soxl':float(a.soxl_weight.max()),'minimum_cash':float(a.cash_weight.min()),'maximum_cash':float(a.cash_weight.max()),'turnover':float(a.turnover.sum())})
result['formal_development_exposures']=exposures
fit_table=pd.DataFrame(fit_rows)
result['training_selected_epoch_ranges']=fit_table.groupby('variant').selected_epoch.agg(['min','median','max']).reset_index().to_dict('records')
result['development_fit_summary']=fit_table[fit_table.year<=2022].groupby('variant').agg(count=('J','count'),minimum_J=('J','min'),maximum_J=('J','max'),mean_J=('J','mean'),minimum_theta_mass=('theta_mass','min'),maximum_theta_mass=('theta_mass','max')).reset_index().to_dict('records')

Path('/tmp/benchmark-anchor-optimizer-audit.json').write_text(json.dumps(result,indent=2,default=str));print(json.dumps({k:result[k] for k in ['recorder_id','failure_count','counts','maxima']},indent=2));print(pd.DataFrame(fit_rows).to_string(index=False));print(pd.DataFrame(ensemble_rows).to_string(index=False))

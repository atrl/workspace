import os
os.environ['MLFLOW_ALLOW_FILE_STORE']='true';os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np,pandas as pd,qlib,torch
from qlib.workflow import R
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy,policy_to_dict,policy_from_dict,numpy_rollout
ROOT=Path('/Users/rziiiii/dev/workspace-worktrees/constrained-net-excess-20261003');CARD=ROOT/'research/constrained_net_excess_20261003';s=json.load(open(CARD/'summary.json'));plan=json.load(open(CARD/'selection_plan.json'));assert s['recorder_id']=='300530137db54b6c807c39ebb9e8c331'
qlib.init(provider_uri=str(ROOT/s['provider_uri']),region='us');R.set_uri(s['recorder_uri']);r=R.get_recorder(experiment_id=Path(s['recorder_local_dir']).parent.name,recorder_id=s['recorder_id'])
fi=r.load_object('analysis/training_diagnostics.pkl');ei=r.load_object('analysis/ensemble_audits.pkl');folds={**r.load_object('inputs/development_fold_inputs.pkl'),**r.load_object('inputs/diagnostic_fold_inputs.pkl')};models=r.load_object('models/development.pkl')
for v,years in r.load_object('models/selected_annual_refits.pkl').items():models.setdefault(v,{}).update(years)
with R.uri_context(plan['parent_input_source']['recorder_uri']):
 parent=R.get_recorder(experiment_name=plan['parent_input_source']['experiment'],recorder_id=plan['parent_input_source']['recorder_id'])
 parent_init={seed:parent.load_object(f'fit_checkpoints/constant/2016/{seed}/training_audit.pkl')['epochs'][0]['checkpoint']['tensor_sha256'] for seed in [17,41,71]}
failures=[];checks={};maxima={};counts=Counter();fit_rows=[];ensemble_rows=[];sample_rows=[];torch.set_num_threads(1)
def ck(test,label):
 checks[label]=checks.get(label,True) and bool(test)
 if not bool(test):failures.append(label)
def diff(a,b,label,tol=1e-10):
 d=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float)),initial=0));maxima[label]=max(maxima.get(label,0),d);ck(np.isfinite(d) and d<=tol,label);return d
def digest(array):return hashlib.sha256(np.asarray(array,dtype=float).tobytes()).hexdigest()
def tensor_check(payload):
 state=payload['state_dict'];ck(payload['dtype']=='float64','model dtype');ck(all(np.isfinite(v).all() for v in state.values()),'finite primal model');theta=state['theta'];ck(theta.shape==(3,9) and theta.min()>=0 and theta.sum()<=1+1e-12,'theta simplex')
 h=hashlib.sha256()
 for name in sorted(state):h.update(name.encode());h.update(str(state[name].shape).encode());h.update(state[name].tobytes())
 ck(h.hexdigest()==payload['tensor_sha256'],'primal tensor hash');counts['primal_tensor_checks']+=1
 return state
def alphas(payloads,x):
 result=np.zeros((len(x),3))
 for payload in payloads:
  st=payload['state_dict'];z=x@st['hidden.weight'].T+st['hidden.bias'];gate=np.empty_like(z);positive=z>=0;gate[positive]=1/(1+np.exp(-z[positive]));ez=np.exp(z[~positive]);gate[~positive]=ez/(1+ez);result+=np.c_[np.ones(len(x)),gate]@st['theta'].T/len(payloads)
 return result
def independent_risk(logs):
 logs=np.asarray(logs,dtype=float);net=np.exp(logs);simple=np.expm1(logs);navlog=np.r_[0.,np.cumsum(logs)];running=np.maximum.accumulate(navlog);under=navlog-running;trough=int(np.argmin(under));peak=int(np.flatnonzero(navlog[:trough+1]==running[trough])[0]);D=float(-np.expm1(under[trough]));dadj=np.zeros(len(logs))
 if D>0:dadj[peak:trough]=-np.exp(under[trough])
 else:D=0.
 sorted_r=np.sort(simple);mass=len(simple)*.05;whole=int(np.floor(mass));fraction=mass-whole;C=float(-(sorted_r[:whole].sum()+fraction*sorted_r[whole])/mass);boundary=sorted_r[int(np.ceil(mass))-1];below=simple<boundary;tied=simple==boundary;w=np.zeros(len(logs));w[below]=1/mass;w[tied]=(mass-int(below.sum()))/(int(tied.sum())*mass);cadj=-w*net
 return np.array([D,C]),np.stack((dadj,cadj))
def scores(path,benchmark,spec):
 logs=np.log(path['net_equity'])-np.log(benchmark);N=len(logs);L=spec['risk_group_length'];A=float(L/N*logs.sum());starts=list(range(0,N-L+1,L))
 if starts[-1]!=N-L:starts.append(N-L)
 q=np.array([logs[a:a+L].sum() for a in starts]);legacy=A-float(np.sqrt(np.mean(np.minimum(q,0)**2)+spec['rms_epsilon']))+float(np.sqrt(spec['rms_epsilon']))
 return A,legacy,float(logs.sum())
def risks_record_audit(record,normalized=None):
 B=np.asarray(record['normalizers'],float);loss=np.array([-record['max_drawdown_pct']/100,-record['cvar5_pct']/100]);raw=np.asarray(record['raw_loss_gaps']);pp=np.asarray(record['raw_percentage_point_gaps']);diff(raw,loss-B,'raw loss gap identity');diff(pp,100*raw,'percentage point units');diff([-record['benchmark_max_drawdown_pct']/100,-record['benchmark_cvar5_pct']/100],B,'samecost benchmark normalizers')
 flags=pp<=1e-10;ck(np.array_equal(flags,record['risk_pass']),'raw four risk flags');ck(record['feasible']==bool(flags.all()),'raw risk feasibility');ck(record['normalizers_valid_for_optimization']==bool(np.isfinite(B).all() and (B>0).all()),'normalizer integrity')
 if normalized is not None:diff(raw/B,normalized,'normalized constraint identity')
 return bool(flags.all())
def audit_path(path,payloads,x,ret,benchmark,riskrec,cost,spec):
 h=path['pretrade_weights'];a=alphas(payloads,x);z=np.c_[a[:,0]+a[:,2]*h[:,0],1-a.sum(1)+a[:,2]*h[:,1],a[:,1]+a[:,2]*h[:,2]]
 diff(z,path['target_weights'],'saved model action identity');diff(h[0],[0,0,1],'real cash origin');diff(h[1:],path['ending_weights'][:-1],'true own holdings');diff(path['open_net_weights'],z,'fee funded target identity');fees=path['fees'];post=(1-fees)[:,None]*path['open_net_weights'];diff(np.abs(post[:,:2]-h[:,:2]).sum(1),path['turnover'],'true risky L1');diff(fees,cost/10000*path['turnover'],'cash fee identity');diff((1-fees)*(1+(path['open_net_weights'][:,:2]*ret).sum(1)),path['net_equity'],'net return identity');ck(np.min(path['net_equity'])>0,'positive NAV multipliers')
 loss,_=independent_risk(np.log(path['net_equity']));B,_=independent_risk(np.log(benchmark));diff(loss,[-riskrec['max_drawdown_pct']/100,-riskrec['cvar5_pct']/100],'selected fullpath independent risks',tol=1e-12);diff(B,riskrec['normalizers'],'selected independent risk reference',tol=1e-12)
 counts['saved_training_decisions_audited']+=len(ret)
 return scores(path,benchmark,spec)
ck(len(fi)==75 and len(ei)==25,'75fits25ensembles')
for index in fi:
 info=r.load_object(index['recorder_artifact']);v,y,seed=index['variant'],index['year'],index['seed'];fold=folds[(v,y)];x=np.asarray(fold['features'],float);ret=np.asarray(fold['returns'],float);spec=info['spec'];costs=tuple(spec['cost_bps']);model=models[v][y]['seed_models'][seed];history=info['epochs'];counts['fits']+=1
 ck(info['failure']==index['failure'],'index failure matches');counts['numerical_failed_fits']+=info['failure'] is not None
 if info['failure'] is not None:
  ck(info['qualification_eligible'] is False and info['null_selected'] and np.count_nonzero(model['state_dict']['theta'])==0,'failed fit accounting null disqualified');continue
 ck(info['qualification_eligible'] is True and info['training_feasibility_passed'] is True,'healthy retained fit proof');ck(len(history)==121 and [h['epoch'] for h in history]==list(range(121)),'all121states');ck(info['history']==history,'epochs history alias');ck(spec['downside_coefficient']==0 and spec['dual_learning_rate']==.003,'registered F and dualstep');ck(digest(x)==info['training_feature_sha256'] and digest(ret)==info['training_return_sha256'],'fixed input hashes')
 ck(info['initial_tensor_sha256']==parent_init[seed]==history[0]['checkpoint']['tensor_sha256'],'exact actual parent initial seed tensors');ck(policy_to_dict(BenchmarkAnchorPolicy(seed=seed))['tensor_sha256']==parent_init[seed],'unchanged architecture initializer');diff(history[0]['checkpoint']['state_dict']['theta'],np.full((3,9),1/54),'interior theta init',tol=0)
 best=0.;best_epoch='anchor';prev=None;feasible_count=0;positive_feasible_count=0;negativelegacyaccepted=0;lambda_peak=0.;increases=decreases=0
 for i,h in enumerate(history):
  state=tensor_check(h['checkpoint']);counts['epoch_records']+=1;used=np.asarray(h['lambda_used']);nxt=np.asarray(h['lambda_next']);g=np.asarray(h['postg']);ck(used.shape==nxt.shape==g.shape==(4,) and np.isfinite(np.r_[used,nxt,g]).all() and min(used.min(),nxt.min())>=0,'finite nonnegative four duals');lambda_peak=max(lambda_peak,float(np.max(nxt)))
  ck(h['F']==min(h['A_by_cost'].values()) and h['F']==h['centered_worst_utility'],'fixed worst aggregate F');ck(h['legacy_J']==min(h['legacy_centered_utility_by_cost'].values()),'legacy report minimum');active=[c for c,A in h['A_by_cost'].items() if A==h['F']];ck(h['cost_weights']=={c:1/len(active) if c in active else 0. for c in costs},'exact symmetric F tie')
  feasible=True
  for k,c in enumerate(costs):
   feasible &= risks_record_audit(h['risks_by_cost'][c],g[2*k:2*k+2]);diff(h['A_by_cost'][c],63*h['daily_mean_log_excess_by_cost'][c],'F63 scale');diff(h['aggregate_log_excess_by_cost'][c],len(x)*h['daily_mean_log_excess_by_cost'][c],'aggregate distinct from F')
  ck(feasible==h['feasible'],'all four raw gates jointly');feasible_count+=feasible;positive_feasible_count+=feasible and h['F']>0
  if feasible and h['F']>best:best=float(h['F']);best_epoch=i;negativelegacyaccepted+=h['legacy_J']<0
  ck(h['selected_epoch_so_far']==best_epoch,'first strict feasible F incumbent independent of legacy and Lagrangian')
  diff(h['post_primal_lagrangian_at_lambda_used'],-h['F']+used@g,'postprimal diagnostic Lagrangian')
  if i==0:
   diff(used,np.zeros(4),'zero lambda initialization',tol=0);diff(nxt,np.zeros(4),'no pre-first-proposal dual update',tol=0);ck(h['dual_update_applied'] is False,'epoch0 no dual update')
  else:
   counts['primal_updates']+=1;diff(used,prev['lambda_next'],'dual continuity',tol=0);expected=np.maximum(0,used+.003*g);diff(nxt,expected,'signed postprimal uncapped dual update',tol=0);increases+=int((nxt>used).sum());decreases+=int((nxt<used).sum());ck(h['dual_update_applied'] is True,'postproposal dual update flag')
   diff(h['pre_primal_lagrangian'],-prev['F']+used@np.asarray(prev['postg']),'preprimal actual current trajectory Lagrangian');delta=sum(np.square(state[k]-prev['checkpoint']['state_dict'][k]).sum() for k in state);diff(h['primal_step_norm'],np.sqrt(delta),'actual projected primal step norm',tol=1e-12)
   grad=h['gradient'];ck(grad['parameter_epoch']==i-1 and grad['scored_rows']==len(x),'current trajectory gradient coverage');ck(all(np.isfinite(grad[k]) for k in ['theta_gradient_norm','hidden_gradient_norm','unclipped_global_norm','gradient_proxy_value']),'finite scalar gradients');ck('chunk_boundaries' not in grad and 'daily_weights' not in grad,'compact epoch gradient storage');maxima['recorded_current_path_parity']=max(maxima.get('recorded_current_path_parity',0),grad['forward_parity_max_error']);ck(grad['forward_parity_max_error']<=1e-10,'current path TorchNumPy parity')
  prev=h
 ck(info['selected_epoch']==best_epoch,'saved selected epoch firstbest');diff(info['F'],best,'saved incumbent fixedF',tol=0);tensor_check(model);counts['valid_null_fits']+=info['null_selected']
 if best_epoch=='anchor':ck(model['tensor_sha256']==policy_to_dict(BenchmarkAnchorPolicy(seed=seed,anchor=True))['tensor_sha256'],'complete exact null state')
 else:ck(model['tensor_sha256']==history[best_epoch]['checkpoint']['tensor_sha256'],'complete selected checkpoint state')
 ck(info['optimizer_state_parameter_epoch']==120 and info['optimizer_diagnostics']['updates_completed']==120,'actual final optimizer epoch');ck(info['optimizer_initial_settings']['lr']==.003 and info['optimizer_initial_settings']['weight_decay']==0,'registered Adam settings')
 for state in info['final_optimizer_state']['state'].values():
  ck(float(state['step'])==120 and all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)),'finite final Adam moment state')
 expected_boundaries=[(max(0,a-21),a,min(len(x),a+63)) for a in range(0,len(x),63)];ck(info['gradient_chunk_boundaries']==expected_boundaries,'one whole-coverage chunk map')
 score_by_cost={};legacy_by_cost={}
 for cost,path in info['training_paths'].items():
  benchmark=info['benchmark_net_equity'][cost];analytic=1+ret[:,1].copy();analytic[0]/=1+cost/10000;diff(benchmark,analytic,'independent funded SOXX multiplier',tol=1e-12);risks_record_audit(info['selected_risks_by_cost'][cost]);A,J,aggregate=audit_path(path,[model],x,ret,benchmark,info['selected_risks_by_cost'][cost],cost,spec);score_by_cost[cost]=A;legacy_by_cost[cost]=J;diff(A,info['centered_utility_by_cost'][cost],'chosen path recomputed F component');diff(J,info['legacy_centered_utility_by_cost'][cost],'chosen path recomputed oldJ component');ck(A>=-1e-14 and info['selected_risks_by_cost'][cost]['feasible'],'chosen bothcost mean and rawrisks')
 diff(min(score_by_cost.values()),info['F'],'chosen fullpath F');diff(min(legacy_by_cost.values()),info['legacy_J'],'chosen fullpath oldJ')
 counts['selected_positiveF_negativeLegacyJ']+=(info['F']>0 and info['legacy_J']<0);counts['negativeLegacyJ_incumbent_acceptances']+=negativelegacyaccepted
 # Declared checkpoint-gradient sample: update1/60/120 uses states0/59/119.
 for update in [1,60,120]:
  source=history[update-1];grad=history[update]['gradient'];dual=np.asarray(history[update]['lambda_used']);sample_model=policy_from_dict(source['checkpoint']);paths={};A={};adjoints={};risk_hash=hashlib.sha256()
  for k,cost in enumerate(costs):
   path=numpy_rollout(sample_model,x,ret,cost_bps=cost);paths[cost]=path;benchmark=info['benchmark_net_equity'][cost];loss,rawadj=independent_risk(np.log(path['net_equity']));B,_=independent_risk(np.log(benchmark));adj=rawadj/B[:,None];adjoints[cost]=adj;risk_hash.update(np.asarray(adj,dtype=float).tobytes());A[cost]=scores(path,benchmark,spec)[0];rec=source['risks_by_cost'][cost]
   diff(100*loss,[-rec['max_drawdown_pct'],-rec['cvar5_pct']],'sample independent risk values');diff((loss-B)/B,source['postg'][2*k:2*k+2],'sample independent normalized g');diff(A[cost],source['A_by_cost'][cost],'sample current trajectory F')
  ck(risk_hash.hexdigest()==grad['normalized_risk_adjoints_sha256'],'sample independent normalized adjoint hash');worst=min(A.values());active=[c for c in costs if A[c]==worst];proxy_value=0.;inactive_nonzero=0
  for k,cost in enumerate(costs):
   weight=1/len(active) if cost in active else 0.;weights=np.full(len(x),63/len(x))*weight-dual[2*k:2*k+2]@adjoints[cost];ck(digest(weights)==grad['combined_daily_weights_sha256_by_cost'][cost],'sample bothcost combined adjoint hash');proxy_value-=float(weights@np.log(paths[cost]['net_equity']));inactive_nonzero+=bool(weight==0 and np.any(weights!=0))
  diff(proxy_value,grad['gradient_proxy_value'],'sample global weighted gradient proxy');sample_rows.append({'variant':v,'year':y,'seed':seed,'update':update,'parameter_epoch':update-1,'F':worst,'active_objective_costs':active,'inactive_cost_nonzero_risk_weights':inactive_nonzero,'normalized_adjoint_hash_match':risk_hash.hexdigest()==grad['normalized_risk_adjoints_sha256']});counts['sampled_checkpoint_gradient_states']+=1;counts['sampled_training_cost_paths']+=2;counts['sampled_inactive_cost_risk_contributions']+=inactive_nonzero
 fit_rows.append({'variant':v,'year':y,'seed':seed,'selected_epoch':best_epoch,'F':info['F'],'legacy_J':info['legacy_J'],'null_selected':info['null_selected'],'feasible_states':feasible_count,'feasible_positive_F_states':positive_feasible_count,'maximum_lambda':lambda_peak,'final_lambda':history[-1]['lambda_next'],'lambda_coordinate_increases':increases,'lambda_coordinate_decreases':decreases,'negative_legacy_incumbent_acceptances':negativelegacyaccepted})
for index in ei:
 e=r.load_object(index['recorder_artifact']);v,y=index['variant'],index['year'];fold=folds[(v,y)];x=np.asarray(fold['features'],float);ret=np.asarray(fold['returns'],float);bundle=models[v][y];spec=next(row for row in fi if row['variant']==v and row['year']==y);seed_info=r.load_object(spec['recorder_artifact']);base_spec=seed_info['spec'];costs=tuple(base_spec['cost_bps']);counts['ensembles']+=1;counts['numerical_failed_ensembles']+=e['failure'] is not None
 ck(e['component_count']==3 and e['all_three_seed_models_used'],'all three ensemble members');ck(e['failure'] is None and e['qualification_eligible'] and e['training_feasibility_passed'],'healthy feasible ensemble');rawfeasible=all(risks_record_audit(vv) for vv in e['ensemble_risks_by_cost'].values());ck(rawfeasible==e['ensemble_feasible'],'raw ensemble feasibility');rejected=e['ensemble_F']<0 or not rawfeasible;ck(rejected==e['ensemble_rejected_for_training_utility_or_risk'],'ensemble rawF/risk acceptance only');counts['ensemble_rejections']+=rejected;counts['ensemble_valid_nulls']+=e['anchor_selected'];selected=bundle['ensemble_models'];rawmodels=[bundle['seed_models'][seed] for seed in [17,41,71]]
 rawpaths=e['rejected_ensemble_training_paths'] if rejected else e['training_paths'];rawA={};selectedA={};oldJ={}
 for cost,path in rawpaths.items():
  benchmark=seed_info['benchmark_net_equity'][cost];A,J,_=audit_path(path,rawmodels,x,ret,benchmark,e['ensemble_risks_by_cost'][cost],cost,base_spec);rawA[cost]=A;diff(A,e['ensemble_centered_utility'][cost],'own ensemble path rawF')
 for cost,path in e['training_paths'].items():
  benchmark=seed_info['benchmark_net_equity'][cost];A,J,_=audit_path(path,selected,x,ret,benchmark,e['selected_risks_by_cost'][cost],cost,base_spec);selectedA[cost]=A;oldJ[cost]=J;ck(A>=-1e-14 and e['selected_risks_by_cost'][cost]['feasible'],'retained ensemble rawconstraints and F')
 diff(min(rawA.values()),e['ensemble_F'],'raw ensemble F minimum');diff(min(selectedA.values()),e['F'],'retained ensemble F minimum');diff(min(oldJ.values()),e['legacy_J'],'retained ensemble legacyJ report');counts['ensemble_positiveF_negativeLegacyJ']+=e['F']>0 and e['legacy_J']<0
 ensemble_rows.append({'variant':v,'year':y,'raw_ensemble_F':e['ensemble_F'],'selected_F':e['F'],'legacy_J':e['legacy_J'],'raw_feasible':e['ensemble_feasible'],'rejected':rejected,'anchor_selected':e['anchor_selected']})
fitframe=pd.DataFrame(fit_rows);grouped=fitframe.groupby('variant').agg(fits=('seed','size'),valid_nulls=('null_selected','sum'),minimum_F=('F','min'),maximum_F=('F','max'),mean_F=('F','mean'),feasible_states=('feasible_states','sum'),feasible_positive_states=('feasible_positive_F_states','sum'),maximum_lambda=('maximum_lambda','max'),dual_increases=('lambda_coordinate_increases','sum'),dual_decreases=('lambda_coordinate_decreases','sum')).reset_index()
for key in ['valid_null_fits','numerical_failed_fits','numerical_failed_ensembles','ensemble_rejections','ensemble_valid_nulls']:counts.setdefault(key,0)
counts['saved_training_decision_checks']=counts.pop('saved_training_decisions_audited')
counts['unique_selected_training_decisions_audited']=sum(2*len(folds[(row['variant'],row['year'])]['features']) for row in fit_rows)+sum(2*len(folds[(row['variant'],row['year'])]['features']) for row in ensemble_rows)
counts['selected_training_cost_paths_audited']=2*(len(fit_rows)+len(ensemble_rows))
result={'recorder_id':s['recorder_id'],'failures':failures,'checks':checks,'check_count':len(checks),'counts':dict(counts),'maxima':maxima,'fit_summary':grouped.to_dict('records'),'fits':fit_rows,'ensembles':ensemble_rows,'fixed_gradient_samples':sample_rows,'sample_rule':'Declared before audit: update1/60/120, prior parameter states0/59/119 for every healthy fit. No optimizer steps or OOS replays.','scope':'All recorded epoch scalar/gate/dual/incumbent arithmetic and all retained paths checked; actual fullpath risk/globaladjoint reconstruction limited to the fixed225state sample plus chosen paths. No global convergence or forward-risk claim.','model_fits_performed':False,'source_changed':False}
Path('/tmp/constrained-optimizer-audit.json').write_text(json.dumps(result,indent=2,default=str));pd.DataFrame(sample_rows).to_csv('/tmp/constrained-optimizer-gradient-samples.csv',index=False);print(json.dumps({k:v for k,v in result.items() if k not in ['checks','fits','ensembles','fixed_gradient_samples']},indent=2))

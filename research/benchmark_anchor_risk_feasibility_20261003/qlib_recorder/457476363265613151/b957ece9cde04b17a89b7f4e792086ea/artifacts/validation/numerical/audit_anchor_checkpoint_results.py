import os
os.environ['MLFLOW_ALLOW_FILE_STORE']='true';os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
from pathlib import Path
from itertools import product
import json,hashlib
import numpy as np,pandas as pd,qlib,torch
from qlib.workflow import R
from src.model.benchmark_anchor_utility import policy_from_dict,BenchmarkAnchorPolicy,numpy_rollout
ROOT=Path('/Users/rziiiii/dev/workspace-worktrees/anchor-risk-feasibility-20261003');CARD=ROOT/'research/benchmark_anchor_risk_feasibility_20261003'
s=json.loads((CARD/'summary.json').read_text());plan=json.loads((CARD/'selection_plan.json').read_text());assert s['parent_recorder_id']=='52f40778cbe64ca6a5864da356b960d0'
qlib.init(provider_uri=str(ROOT/s['provider_uri']),region='us');R.set_uri(s['recorder_uri']);r=R.get_recorder(experiment_id=Path(s['recorder_local_dir']).parent.name,recorder_id=s['recorder_id'])
metrics=r.load_object('analysis/checkpoint_cost_metrics.pkl');status=r.load_object('analysis/checkpoint_joint_feasibility.pkl');null_metrics=r.load_object('analysis/null_metrics.pkl');null_status=r.load_object('analysis/null_joint_feasibility.pkl');summaries=r.load_object('analysis/run_summaries.pkl');reproduction=r.load_object('analysis/utility_reproduction.pkl');audits=r.load_object('analysis/integrity_audits.pkl')
failures=[];maxima={};checks={}
def ck(test,label):
 checks[label]=checks.get(label,True) and bool(test)
 if not bool(test):failures.append(label)
def difference(left,right,label,tol=1e-10):
 d=float(np.max(np.abs(np.asarray(left,dtype=float)-np.asarray(right,dtype=float)),initial=0));maxima[label]=max(maxima.get(label,0),d);ck(np.isfinite(d) and d<=tol,label);return d
def optional_equal(left,right):
 if left is None:return pd.isna(right)
 return left==right
runs=set(product(plan['inventory']['variants'],plan['inventory']['training_cutoff_years'],plan['inventory']['seeds']));epochs=plan['inventory']['saved_epochs'];costs=plan['recompute']['cost_bps'];idcols=['variant','model_year','seed'];all_status=pd.concat([status,null_status],ignore_index=True);all_metrics=pd.concat([metrics,null_metrics],ignore_index=True)
ck(len(status)==7623 and len(null_status)==63,'joint counts');ck(len(metrics)==15246 and len(null_metrics)==126,'cost counts');ck(len(summaries)==63 and len(audits)==63,'run counts')
expected={(v,y,z,e) for v,y,z in runs for e in epochs};actual=set(map(tuple,status[idcols+['epoch']].itertuples(index=False,name=None)));ck(actual==expected and len(actual)==len(status),'complete unique learner identities');ck(set(map(tuple,null_status[idcols].itertuples(index=False,name=None)))==runs and null_status.epoch.eq('exact_SOXX_null').all(),'complete null identities')
expectedcost={(*key,c) for key in expected for c in costs};actualcost=set(map(tuple,metrics[idcols+['epoch','cost_bps']].itertuples(index=False,name=None)));ck(expectedcost==actualcost and len(actualcost)==len(metrics),'complete unique learner cost identities');ck(not status.explicit_null.any() and null_status.explicit_null.all() and not metrics.explicit_null.any() and null_metrics.explicit_null.all(),'explicit null separation')
metric_index=all_metrics.set_index(idcols+['epoch','cost_bps']);tol=plan['feasibility']['risk_tolerance'];j_tol=plan['recompute']['recompute_vs_stored_utility_absolute_tolerance'];margin=plan['feasibility']['positive_utility_margin'];sample_index=[];new_counts=[]
for key,rows in all_status.groupby(idcols):
 learner=rows[~rows.explicit_null].sort_values('epoch');null=rows[rows.explicit_null].iloc[0];summary=summaries[(summaries.variant==key[0])&(summaries.model_year==key[1])&(summaries.seed==key[2])].iloc[0]
 for st in rows.itertuples(index=False):
  four=[];raw_excess=[];recomputed=[];stored=[]
  for cost in costs:
   met=metric_index.loc[(*key,st.epoch,cost)];dd=met.max_drawdown_pct-met.benchmark_max_drawdown_pct;cv=met.cvar5_pct-met.benchmark_cvar5_pct
   ck(getattr(st,f'drawdown_{cost}_pass')==(dd>=-tol),'DD flag');ck(getattr(st,f'cvar_{cost}_pass')==(cv>=-tol),'CVaR flag');difference(getattr(st,f'drawdown_gap_{cost}_pp'),dd,'DD gap identity');difference(getattr(st,f'cvar_gap_{cost}_pp'),cv,'CVaR gap identity');difference(getattr(st,f'net_log_excess_{cost}'),met.log_excess_sum,'aggregate excess separate from utility')
   four.extend([dd>=-tol,cv>=-tol]);raw_excess.append(met.log_excess_sum);recomputed.append(met.centered_utility);stored.append(met.stored_centered_utility)
   difference(met.centered_utility-met.stored_centered_utility,met.recomputed_minus_stored_U,'signed component difference');ck(abs(met.recomputed_minus_stored_U)<=j_tol,'component parity');ck(met.U_sign_disagreement==(np.sign(met.centered_utility)!=np.sign(met.stored_centered_utility)),'component sign flag');difference(st.J,met.J,'cost/J binding');difference(st.stored_J,met.stored_J,'cost/storedJ binding')
  risk=all(four);ck(st.risk_feasible==risk,'joint four risk gates');ck(st.risk_and_nonnegative_J==(risk and st.J>=0),'no negative-J tolerance rescue');ck(st.risk_and_strict_positive_J==(risk and st.J>0),'raw strict positive gate');ck(st.risk_and_positive_J_margin==(risk and st.J>margin),'J-only numerical margin gate');ck(st.both_costs_raw_positive_log_excess==all(v>0 for v in raw_excess),'independent raw net excess flag')
  difference(st.J,min(recomputed),'worst aggregate cost utility');difference(st.stored_J,min(stored),'stored worst utility');difference(st.J-st.stored_J,st.recomputed_minus_stored_J,'signed J difference');ck(abs(st.recomputed_minus_stored_J)<=j_tol,'J parity');ck(st.J_sign_disagreement==(np.sign(st.J)!=np.sign(st.stored_J)),'J sign disagreement');
  if st.explicit_null:ck(st.J==0 and st.risk_feasible,'null Jzero and feasible')
 for name in ['risk_feasible','risk_and_nonnegative_J','risk_and_strict_positive_J','risk_and_positive_J_margin']:
  sub=learner[learner[name]];ck(summary[name+'_count']==len(sub),'summary count '+name);first=int(sub.epoch.iloc[0]) if len(sub) else None;ck(optional_equal(first,summary['first_'+name+'_epoch']),'first diagnostic '+name)
  if len(sub):
   best=sub.sort_values(['J','epoch'],ascending=[False,True]).iloc[0];ck(summary['best_'+name+'_epoch']==best.epoch,'best diagnostic '+name);difference(summary['best_'+name+'_J'],best.J,'best raw J '+name)
   details=summary['best_'+name+'_details']
   for field,value in details.items():difference(value,best[field],'best diagnostic raw detail')
  else:ck(pd.isna(summary['best_'+name+'_epoch']) and pd.isna(summary['best_'+name+'_J']),'empty best diagnostic '+name)
 best_epoch='exact_SOXX_null';best_J=0.
 for row in learner[learner.risk_feasible].itertuples():
  if row.J>best_J:best_J=float(row.J);best_epoch=int(row.epoch)
 ck(str(summary.null_inclusive_best_risk_feasible_epoch)==str(best_epoch),'null-inclusive best tie');difference(summary.null_inclusive_best_risk_feasible_J,best_J,'null-inclusive best J')
 ck(summary.learner_checkpoint_count==121 and summary.explicit_null_count==1,'null excluded from count');ck(summary.checkpoint_identifiers_are_diagnostic_only,'not model designation')
 for cost in costs:
  for risk in ['drawdown','cvar']:ck(summary[f'{risk}_{cost}_pass_count']==int(learner[f'{risk}_{cost}_pass'].sum()),'single gate count')
 # Fixed BEFORE outcomes: null, global best raw utility learner, best risk-feasible learner. No OOS/model search.
 chosen=[('exact_SOXX_null','null')];highest=learner.sort_values(['J','epoch'],ascending=[False,True]).iloc[0];chosen.append((int(highest.epoch),'highest_raw_J'))
 feasible=learner[learner.risk_feasible]
 if len(feasible):chosen.append((int(feasible.sort_values(['J','epoch'],ascending=[False,True]).iloc[0].epoch),'highest_risk_feasible_J'))
 seen=set()
 for epoch,reason in chosen:
  if epoch not in seen:sample_index.append({'variant':key[0],'model_year':int(key[1]),'seed':int(key[2]),'epoch':epoch,'reason':reason});seen.add(epoch)
columns=['risk_feasible','risk_and_nonnegative_J','risk_and_strict_positive_J','risk_and_positive_J_margin'];counts=status.groupby('variant')[columns].sum();counts['trajectory_checkpoints']=status.groupby('variant').size();counts=counts.reset_index()
for row in s['counts']:
 rec=counts[counts.variant==row['variant']].iloc[0]
 for field in columns+['trajectory_checkpoints']:ck(int(rec[field])==row[field],'root counts '+field)
pd.testing.assert_frame_equal(counts,pd.read_csv(CARD/'feasibility_counts.csv'),check_dtype=False)
ck(len(reproduction)==len(all_status),'reproduction all states');ck(all(a['model_fitting_performed']==False and a['out_of_sample_rows_used']==False and a['new_model_selection_performed']==False for a in audits),'analysis scope nofit/noOOS')
# Reconstruct only the predeclared verification sample; no alternate policy export or test-year path.
with R.uri_context(plan['source']['recorder_uri']):
 parent=R.get_recorder(experiment_name=plan['source']['experiment'],recorder_id=plan['source']['recorder_id']);folds=parent.load_object(plan['source']['input_fold_artifact'])
parent_before={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in s['parent_source_sha256']};sample_rows=[];parent_audits={};torch.set_num_threads(1)
def independent_risk(net):
 value=np.asarray(net)-1;nav=np.r_[1.,np.cumprod(net)];dd=100*float(np.min(nav/np.maximum.accumulate(nav)-1));ordered=np.sort(value);mass=len(value)*.05;k=int(np.floor(mass));cv=100*float((ordered[:k].sum()+(mass-k)*ordered[k])/mass);return dd,cv
for sample in sample_index:
 v,y,z,epoch=sample['variant'],sample['model_year'],sample['seed'],sample['epoch'];fold=folds[(v,y)];x=np.asarray(fold['features'],float);ret=np.asarray(fold['returns'],float);key=(v,y,z)
 if key not in parent_audits:parent_audits[key]=parent.load_object(f'fit_checkpoints/{v}/{y}/{z}/training_audit.pkl')
 cached=parent_audits[key];original=cached['spec'];model=BenchmarkAnchorPolicy(seed=z,anchor=True) if epoch=='exact_SOXX_null' else policy_from_dict(cached['epochs'][epoch]['checkpoint']);components=[]
 for cost in costs:
  result=numpy_rollout(model,x,ret,cost_bps=cost);baseline=numpy_rollout(BenchmarkAnchorPolicy(seed=z,anchor=True),x,ret,cost_bps=cost);analytic=1+ret[:,1].copy();analytic[0]/=1+cost/10000;difference(baseline['net_equity'],analytic,'sample independent SOXX entry-hold',tol=1e-10)
  dd,cv=independent_risk(result['net_equity']);bdd,bcv=independent_risk(analytic);met=metric_index.loc[(v,y,z,epoch,cost)];difference(dd,met.max_drawdown_pct,'sample independently anchored DD',tol=1e-10);difference(cv,met.cvar5_pct,'sample independently fractional CVaR',tol=1e-10);difference(bdd,met.benchmark_max_drawdown_pct,'sample benchmark DD',tol=1e-10);difference(bcv,met.benchmark_cvar5_pct,'sample benchmark CVaR',tol=1e-10)
  logs=np.log(result['net_equity'])-np.log(baseline['net_equity']);length=original['risk_group_length'];starts=list(range(0,len(logs)-length+1,length));
  if starts[-1]!=len(logs)-length:starts.append(len(logs)-length)
  q=np.array([logs[i:i+length].sum() for i in starts]);utility=length/len(logs)*logs.sum()-original['downside_coefficient']*np.sqrt(np.mean(np.minimum(q,0)**2)+original['rms_epsilon'])+original['downside_coefficient']*np.sqrt(original['rms_epsilon']);components.append(float(utility));difference(utility,met.centered_utility,'sample independently reconstructed utility',tol=1e-10)
  # All transition costs and entire path retained; no warmup return mask.
  weights=result['open_net_weights'];h=result['pretrade_weights'];post=(1-result['fees'])[:,None]*weights;difference(np.abs(post[:,:2]-h[:,:2]).sum(1),result['turnover'],'sample true risky-L1 fees');difference(result['fees'],result['turnover']*cost/10000,'sample fee rate');difference(result['pretrade_weights'][1:],result['ending_weights'][:-1],'sample actual own path')
  sample_rows.append({**sample,'cost_bps':cost,'rows':len(x),'independent_max_drawdown_pct':dd,'independent_cvar5_pct':cv,'independent_centered_utility':float(utility),'recorded_centered_utility':float(met.centered_utility)})
 st=all_status[(all_status.variant==v)&(all_status.model_year==y)&(all_status.seed==z)&(all_status.epoch==epoch)].iloc[0];difference(min(components),st.J,'sample worst-cost J')
ck(all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in parent_before.items()),'parent financial source untouched by sample audit')
result={'recorder_id':s['recorder_id'],'parent_recorder_id':s['parent_recorder_id'],'failures':failures,'check_count':len(checks),'checks':checks,'maxima':maxima,'learner_models':len(status),'nulls':len(null_status),'learner_cost_rows':len(metrics),'null_cost_rows':len(null_metrics),'counts':counts.to_dict('records'),'fixed_sample_rule':'Per run exact null, highest rawJ learner and best risk-feasible learner; earliest ties; dedup only verification sample, not source inventory. Declared to root before authoritative results.','sampled_models':len(sample_index),'sampled_cost_paths':len(sample_rows),'samples':sample_rows,'diagnostic_model_selection_performed':False,'new_market_fits':False,'out_of_sample_rows_used':False,'scope_limit':'All aggregate gap/status/count arithmetic checked; independently reconstructed daily risk only for fixed sampled saved models, not all7623states. Parent kernel unchanged; risk/utility formulas independently evaluated with NumPy.'}
Path('/tmp/anchor-checkpoint-results-independent-audit.json').write_text(json.dumps(result,indent=2,default=str));pd.DataFrame(sample_rows).to_csv('/tmp/anchor-checkpoint-results-independent-samples.csv',index=False);print(json.dumps({k:v for k,v in result.items() if k not in ['checks','samples']},indent=2))

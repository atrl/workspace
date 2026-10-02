from pathlib import Path
from time import perf_counter
import json,pickle
import numpy as np,torch
from src.model.sparse_holdings_net_excess import *
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy
n=504;sign=np.where(np.arange(n)%2==0,1.,-1.);x=np.zeros((n,27));x[:,0]=sign;rS=.002+.008*sign;returns=np.c_[3*rS,rS];spec=ConstrainedNetExcessSpec()
parent=BenchmarkAnchorPolicy(seed=17,anchor=True)
with torch.no_grad():
 parent.hidden.weight.zero_();parent.hidden.bias.zero_();parent.hidden.weight[:4,0]=2.;parent.hidden.weight[4:,0]=-2.;parent.theta[0,1:5]=.125;parent.theta[1,5:9]=.125
witness=embed_parent_policy(parent);value=evaluate_policy(witness,x,returns,spec=spec)
report={'synthetic_only':True,'market_data_used':False,'fixture':{'rows':504,'feature':'alternating+1/-1 column0,other26zero','SOXX_return':'.002+.008x','SOXL_return':'3*SOXX'},'unchanged_schedule':True,'witness':{'F':value['F'],'feasible':value['feasible']},'fits':[]};models=[]
for seed in [17,41,71]:
 started=perf_counter();model,meta=fit_policy(x,returns,seed=seed,spec=spec);models.append(model)
 row={'seed':seed,'elapsed_seconds':perf_counter()-started,'failure':meta['failure'],'epochs':len(meta['epochs']),'selected_epoch':meta['selected_epoch'],'F':meta['F'],'legacy_J':meta['legacy_J'],'feasible':meta['training_feasibility_passed'],'null_selected':meta['null_selected'],'feasible_positive_F_states':sum(h['feasible'] and h['F']>0 for h in meta['epochs']),'maximum_trajectory_F':max(h['F'] for h in meta['epochs']),'selected_action_diagnostics':meta['selected_action_diagnostics'],'initial_path_parity':meta['initial_path_parity'],'metadata_pickle_bytes':len(pickle.dumps(meta))};report['fits'].append(row);print(row,flush=True)
selected,audit=evaluate_ensemble(models,x,returns,spec=spec);report['ensemble']={k:audit[k] for k in ['F','legacy_J','training_feasibility_passed','anchor_selected','failure']};report['all3seeds_and_ensemble_positive']=all(f['failure'] is None and f['F']>0 and f['feasible'] for f in report['fits']) and audit['F']>0 and audit['training_feasibility_passed']
Path('/tmp/sparse-holdings-known-information-probe.json').write_text(json.dumps(report,indent=2));print('RESULT',report['all3seeds_and_ensemble_positive'],report['ensemble'])

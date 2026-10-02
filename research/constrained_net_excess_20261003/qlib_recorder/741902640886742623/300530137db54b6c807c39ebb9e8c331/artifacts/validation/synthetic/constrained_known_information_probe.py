from pathlib import Path
from time import perf_counter
import json,pickle
import numpy as np,torch
from src.model.constrained_net_excess import *
n=504;sign=np.where(np.arange(n)%2==0,1.,-1.);features=np.zeros((n,27));features[:,0]=sign;underlying=.002+.008*sign;returns=np.c_[3*underlying,underlying];spec=ConstrainedNetExcessSpec()
witness=BenchmarkAnchorPolicy(seed=17,anchor=True)
with torch.no_grad():
 witness.hidden.weight.zero_();witness.hidden.bias.zero_();witness.hidden.weight[:4,0]=2.;witness.hidden.weight[4:,0]=-2.;witness.theta[0,1:5]=.125;witness.theta[1,5:9]=.125
value=evaluate_policy(witness,features,returns,spec=spec)
report={'synthetic_only':True,'market_data_used':False,'fixture':{'rows':n,'feature':'alternating+1/-1 incolumn0,other26zero','soxx_return':'.002+.008*x','soxl_return':'3*SOXX','witness_gate_weight':2,'witness_theta_L_mass':.5,'witness_theta_C_mass':.5},'schedule_changed':False,'known_class_witness':{'F':value['F'],'A_by_cost':value['A_by_cost'],'legacy_J':value['legacy_J'],'feasible':value['feasible'],'normalized_g':value['normalized_g'].tolist()},'fits':[]}
for seed in [17,41,71]:
 start=perf_counter();model,meta=fit_policy(features,returns,seed=seed,spec=spec)
 row={'seed':seed,'elapsed_seconds':perf_counter()-start,'failure':meta['failure'],'selected_epoch':meta['selected_epoch'],'F':meta['F'],'A_by_cost':meta['centered_utility_by_cost'],'legacy_J':meta['legacy_J'],'training_feasibility_passed':meta['training_feasibility_passed'],'null_selected':meta['null_selected'],'epochs':len(meta['epochs']),'feasible_trajectory_states':sum(h['feasible'] for h in meta['epochs']),'feasible_positive_F_states':sum(h['feasible'] and h['F']>0 for h in meta['epochs']),'maximum_trajectory_F':max(h['F'] for h in meta['epochs']),'final_lambda':meta['optimizer_diagnostics']['lambda_final'],'metadata_pickle_bytes':len(pickle.dumps(meta))}
 report['fits'].append(row);print(row,flush=True)
report['all_seeds_find_feasible_positive_F']=all(row['failure'] is None and row['F']>0 and row['training_feasibility_passed'] for row in report['fits'])
Path('/tmp/constrained-known-information-probe.json').write_text(json.dumps(report,indent=2));print('witness',report['known_class_witness']);print('allfixedseedspositive',report['all_seeds_find_feasible_positive_F'])

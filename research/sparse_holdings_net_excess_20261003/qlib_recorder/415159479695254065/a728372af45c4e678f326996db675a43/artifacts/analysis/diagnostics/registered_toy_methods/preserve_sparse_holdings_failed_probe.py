from pathlib import Path
from time import perf_counter
import json,pickle,hashlib
import numpy as np
from src.model.sparse_holdings_net_excess import *
prior=json.load(open('/tmp/sparse-holdings-known-information-probe.json'));n=504;sign=np.where(np.arange(n)%2==0,1.,-1.);x=np.zeros((n,27));x[:,0]=sign;rS=.002+.008*sign;returns=np.c_[3*rS,rS];spec=ConstrainedNetExcessSpec()
results={'purpose':'Exact deterministic reproduction to retain full evidence for the already observed FAILED pre-fit positive-control; no hyperparameter/fixture search.','features':x,'returns':returns,'spec':spec,'prior_probe':prior,'source_sha256':hashlib.sha256(Path('src/model/sparse_holdings_net_excess.py').read_bytes()).hexdigest(),'fits':[]};rows=[]
for seed in [17,41,71]:
 model,meta=fit_policy(x,returns,seed=seed,spec=spec);latest=policy_from_dict(meta['epochs'][-1]['checkpoint']);evaluation=evaluate_policy(latest,x,returns,spec=spec);before=next(f for f in prior['fits'] if f['seed']==seed)
 assert meta['failure']==before['failure'] and meta['selected_epoch']==before['selected_epoch'] and meta['F']==before['F'] and max(h['F'] for h in meta['epochs'])==before['maximum_trajectory_F']
 results['fits'].append({'seed':seed,'selected_model':policy_to_dict(model),'metadata':meta,'latest_model':policy_to_dict(latest),'latest_evaluation':evaluation})
 for h in meta['epochs']:
  row={'seed':seed,'epoch':h['epoch'],'F':h['F'],'feasible':h['feasible'],'legacy_J':h['legacy_J'],'selected_epoch_so_far':h['selected_epoch_so_far'],'lambda_max':max(h['lambda_next']),'logit_max':max(v['maximum_absolute_logit'] for v in h['action_diagnostics'].values())}
  for c in [10,25]:
   risk=h['risks_by_cost'][c];d=h['action_diagnostics'][c];row.update({f'DD_gap_{c}':risk['raw_percentage_point_gaps'][0],f'CVaR_gap_{c}':risk['raw_percentage_point_gaps'][1],f'soleS_fraction_{c}':d['sole_expert_fraction']['soxx'],f'soleL_fraction_{c}':d['sole_expert_fraction']['soxl'],f'soleC_fraction_{c}':d['sole_expert_fraction']['cash'],f'soleH_fraction_{c}':d['sole_expert_fraction']['hold'],f'support1_fraction_{c}':d['support_cardinality_fraction'][1]})
  if h['gradient']:row.update({k:h['gradient'][k] for k in ['encoder_gradient_norm','head_gradient_norm','K_gradient_norm','all_parameter_gradients_exact_zero','unclipped_global_norm']})
  rows.append(row)
 print('retained seed',seed,'lastF',evaluation['F'],'last feasible',evaluation['feasible'],flush=True)
import pandas as pd
Path('/tmp/sparse-holdings-known-information-full.pkl').write_bytes(pickle.dumps(results));pd.DataFrame(rows).to_csv('/tmp/sparse-holdings-known-information-trajectories.csv',index=False)
print('All summaries match initial failed probe exactly; retainedbytes',Path('/tmp/sparse-holdings-known-information-full.pkl').stat().st_size)

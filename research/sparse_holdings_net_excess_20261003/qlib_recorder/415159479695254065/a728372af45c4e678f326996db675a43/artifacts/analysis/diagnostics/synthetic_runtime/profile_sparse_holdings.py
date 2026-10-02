from pathlib import Path
import hashlib,json,pickle,time
import numpy as np
from src.model import sparse_holdings_net_excess as model
source=Path(model.__file__);before=hashlib.sha256(source.read_bytes()).hexdigest()
n=3000;rng=np.random.default_rng(203);x=rng.normal(size=(n,27));u=.0003+.008*np.sin(np.arange(n)*.73);r=np.c_[2.9*u+rng.normal(0,.0004,n),u]
start=time.perf_counter();chosen,info=model.fit_policy(x,r,seed=17,spec=model.ConstrainedNetExcessSpec());elapsed=time.perf_counter()-start
after=hashlib.sha256(source.read_bytes()).hexdigest();assert before==after
report={'synthetic_only':True,'market_data_used':False,'rows':n,'seed':17,'updates':120,'elapsed_seconds':elapsed,'seconds_per_update':elapsed/120,'model_source_sha256':before,'source_unchanged':True,'failure':info['failure'],'epochs':len(info['epochs']),'optimizer_steps':info['optimizer_state_parameter_epoch'],'metadata_pickle_bytes':len(pickle.dumps(info)),'epoch_records_pickle_bytes':len(pickle.dumps(info['epochs'])),'selected_paths_pickle_bytes':len(pickle.dumps(info['training_paths'])),'maximum_forward_parity_error':info['optimizer_diagnostics']['maximum_forward_parity_error'],'all_support_mismatches':sum(e['gradient']['support_mismatch_count'] for e in info['epochs'] if e['gradient']),'initial_path_parity':info['initial_path_parity'],'interpretation':'Execution/memory profile only; this artificial chronology is not economic evidence. No schedule, architecture or financial thresholds were tuned.'}
Path('/tmp/sparse-holdings-runtime.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='initial_path_parity'},indent=2),flush=True)
assert info['failure'] is None and len(info['epochs'])==121 and info['optimizer_state_parameter_epoch']==120

from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
from scipy.stats import spearmanr
import qlib
from qlib.workflow import R
from src.objective.relative_wealth import stationary_block_bootstrap_indices
root=Path.cwd();c=root/'research/soxs_cash5_revalidation_20261003';s=json.loads((c/'summary.json').read_text());ref=json.loads((c/'recorder_reference.json').read_text());plan=json.loads((c/'plan.json').read_text());qlib.init(provider_uri=str(root/plan['source']['provider_uri']),region='us')
with R.uri_context(s['recorder_uri']):
 rec=R.get_recorder(experiment_name=s['experiment'],recorder_id=s['recorder_id']);assert rec.client.get_run(rec.id).info.status=='FINISHED'
 for n,h in ref['artifact_sha256'].items():assert hashlib.sha256((Path(rec.get_local_dir())/'artifacts'/n).read_bytes()).hexdigest()==h
 p=rec.load_object('analysis/predictions.pkl');f=rec.load_object('inputs/features_labels.pkl');m=rec.load_object('models/fitted_models.pkl');draws=rec.load_object('analysis/bootstrap_draws.pkl');errors=[]
 for year in [2016,2023]:
  d=p.loc[p.fit_year==year].iloc[[0,-1]];pred=m[str(year)+'_base_plus_soxs'].predict(f.loc[d.index,plan['features']['base_plus_soxs']]);errors.append(float(np.max(np.abs(pred-d.base_plus_soxs.to_numpy()))));assert errors[-1]<1e-14
 for date in [pd.Timestamp('2016-01-04'),pd.Timestamp('2023-01-03'),pd.Timestamp('2026-09-17')]:
  i=f.index.get_loc(date);assert f.loc[date,'trade_date']==f.index[i+1] and f.loc[date,'label_mature_date']==f.index[i+6]
 d=p[p.segment=='development'];ix=stationary_block_bootstrap_indices(observation_count=len(d),path_length=len(d),path_count=2000,expected_block_length=20,seed=20261003)[0];z=d.iloc[ix];y=z.cash_advantage_5d.to_numpy(float);base=z.base.to_numpy(float);full=z.base_plus_soxs.to_numpy(float);h=np.log((1+.0025)/(1-.0025));mask=full>h
 values={'base_plus_soxs_ic':float(spearmanr(y,full).statistic),'delta_ic_full_minus_base':float(spearmanr(y,full).statistic-spearmanr(y,base).statistic),'mse_reduction_base_minus_full':float(np.mean((base-y)**2-(full-y)**2)),'full_conditional_net_event_mean_bp25':float(10000*np.mean(y[mask]-h))}
 saved=draws['development_block20'].iloc[0];deltas={k:float(abs(v-saved[k])) for k,v in values.items()};assert max(deltas.values())<1e-10
 out={'passed':True,'scope':'One bounded independent review: all20 scientificartifact hashes;2frozenmodels on2dates each;3H5calendaralignments;devblock20draw0rank,MSE,fee stats. No newfits/paths.','recorder_id':rec.id,'model_prediction_max_errors':errors,'bootstrap_draw0_absolute_errors':deltas,'legacy_signal_diagnostic_passed':True,'information_passed':False,'incremental_passed':False,'economic_passed':False,'decision':'end_route_no_further_search','objective_achieved':False}
 (c/'independent_review.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

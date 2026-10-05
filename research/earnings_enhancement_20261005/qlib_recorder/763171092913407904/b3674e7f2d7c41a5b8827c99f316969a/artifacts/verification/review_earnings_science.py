from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,qlib
from scipy.stats import spearmanr
from qlib.data import D
from qlib.workflow import R
from src.objective.relative_wealth import stationary_block_bootstrap_indices
from src.workflow.stage1_economic_action import fee_affordable_target
from src.backtest.position_accounting import settle_target_session
r=Path.cwd();c=r/'research/earnings_enhancement_20261005';h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((c/'summary.json').read_text());ref=json.loads((c/'recorder_reference.json').read_text());auth=json.loads((c/'run_authorization.json').read_text())
assert all(h(r/k)==v for k,v in auth['source_files'].items())
qlib.init(provider_uri=str(r/'data/qlib/published_futu_affine_20260925_v1'),region='us',kernels=1)
with R.uri_context(Path(s['recorder_uri']).as_uri()):rec=R.get_recorder(experiment_name=s['experiment'],recorder_id=s['recorder_id'])
assert rec.status=='FINISHED'
a=Path(rec.get_local_dir())/'artifacts'
assert all(h(a/k)==v for k,v in ref['artifact_sha256'].items())
f=rec.load_object('inputs/features_labels.pkl');p=rec.load_object('analysis/predictions.pkl');models=rec.load_object('models/fitted_models.pkl');draws=rec.load_object('analysis/bootstrap_draws.pkl');specs=rec.load_object('analysis/bootstrap_indices_spec.pkl')
assert len(models)==14 and len(p)==1762 and set(p.index.year)==set(range(2016,2023))
cal=pd.DatetimeIndex(D.calendar(freq='day'));q=D.features(['SOXL','SOXX'],['$open'],start_time=cal[0],end_time=cal[-1],freq='day')
prices={z:q.xs(z,level='instrument')['$open'].reindex(cal) for z in ('SOXL','SOXX')}
gxx=prices['SOXX'].shift(-6)/prices['SOXX'].shift(-1);gxl=prices['SOXL'].shift(-6)/prices['SOXL'].shift(-1)
for head,y in [('cash5',-np.log(gxx)),('leverage5',np.log(gxl)-np.log(gxx))]:np.testing.assert_allclose(f[head],y.reindex(f.index),rtol=0,atol=1e-15)
base=['soxx_trend20','soxx_trend60','soxx_vol20','soxx_drawdown60','relative_momentum20'];full=base+['earnings_surprise_nvda','earnings_surprise_amd','earnings_surprise_intc'];maxerr=0.
for year in range(2016,2023):
 train=f.loc[f.index<pd.Timestamp(f'{year}-01-01')].iloc[:-20]
 cutoff=pd.Timestamp(f'{year}-01-01',tz='America/New_York')
 assert train.label_available_at.max()<cutoff and train.feature_available_at.max()<cutoff
 test=f.loc[f.index.year==year]
 for family,cols in [('base',base),('base_plus_earnings',full)]:
  m=models[f'{year}_{family}'];np.testing.assert_allclose(m['scale'].mean_,train[cols].to_numpy(dtype=np.float64).mean(axis=0),rtol=0,atol=1e-14)
  expected=m.predict(test[cols]);actual=p.loc[test.index,[f'{z}_{family}' for z in ['cash5','leverage5']]].to_numpy()
  maxerr=max(maxerr,float(np.abs(expected-actual).max()));np.testing.assert_array_equal(expected,actual)
bs=pd.read_csv(c/'bootstrap_summary.csv');bootstrap_checks=[]
for key,spec in specs.items():
 assert spec['sampler_source_sha256']==h(r/spec['sampler_source'])
 indices=stationary_block_bootstrap_indices(observation_count=len(p),path_length=len(p),path_count=spec['shape'][0],expected_block_length=spec['block_length'],seed=spec['seed'])
 assert hashlib.sha256(indices.tobytes()).hexdigest()==spec['indices_sha256']
 for head in ('cash5','leverage5'):
  distribution=draws[key][head];idx=indices[0];y=p[head].to_numpy()[idx];v=p[f'{head}_base'].to_numpy()[idx];w=p[f'{head}_base_plus_earnings'].to_numpy()[idx]
  di=float(spearmanr(y,w).statistic-spearmanr(y,v).statistic)
  np.testing.assert_allclose(di,distribution.iloc[0]['delta_ic_full_minus_base'],atol=1e-13,rtol=0)
  row=bs.loc[(bs['head']==head)&(bs.block_length==spec['block_length'])].iloc[0]
  for col in distribution:
   values=distribution[col].to_numpy();valid=np.isfinite(values).all()
   for quantile in [.0125,.025,.5,.975,.9875]:
    actual=row[f'{col}_q{quantile:g}']
    if valid:np.testing.assert_allclose(actual,np.quantile(values,quantile),rtol=1e-12,atol=1e-12)
    else:assert pd.isna(actual)
  bootstrap_checks.append({'head':head,'block':spec['block_length'],'indices_hash_verified':True,'first_delta_ic_direct_scipy':di,'all_quantiles_verified':True})
cost_checks=[]
for bps in (10,25):
 cost=bps/10000;A=(1-cost)/(1+cost)
 for head,target,gross in [('cash5',np.array([0.,0.,1.]),1.),('leverage5',np.array([1.,0.,0.]),1.12)]:
  initial=np.array([0.,1.,0.]);entry=fee_affordable_target(target,pretrade_weights=initial,cost_bps=bps)
  first=settle_target_session(initial,open_target_weights=entry,asset_returns=np.array([.12,.04,0.]),cost_bps=bps)
  exit=fee_affordable_target(initial,pretrade_weights=first['ending_weights'],cost_bps=bps)
  last=settle_target_session(first['ending_weights'],open_target_weights=exit,asset_returns=np.zeros(3),cost_bps=bps)
  observed=first['net_equity']*last['net_equity']/1.04
  expected=A**(1 if head=='cash5' else 2)*gross/1.04
  np.testing.assert_allclose(observed,expected,rtol=0,atol=1e-12)
  cost_checks.append({'head':head,'cost_bps':bps,'absolute_error':abs(observed-expected)})
result={'passed':True,'scope':'Read-only recorded-model/label/bootstrap/economic-arithmetic verification; no new model fit or portfolio path.','original_scientific_artifacts_verified':len(ref['artifact_sha256']),'source_files_verified':len(auth['source_files']),'model_count':len(models),'prediction_rows':len(p),'cached_prediction_max_error':maxerr,'all_dual_labels_native_readback_equal':True,'all_scaler_means_prior_training_only':True,'bootstrap_checks':bootstrap_checks,'roundtrip_accounting_checks':cost_checks,'source_deployment_gate':False}
(c/'independent_science_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
import qlib
from qlib.workflow import R
root=Path.cwd();c=root/'research/causal_cash_calibration_20261003';s=json.loads((c/'summary.json').read_text());ref=json.loads((c/'recorder_reference.json').read_text());qlib.init(provider_uri=str(root/s['provider_uri']),region='us')
with R.uri_context(s['recorder_uri']):
 rec=R.get_recorder(experiment_name=s['experiment'],recorder_id=s['recorder_id']);assert rec.client.get_run(rec.id).info.status=='FINISHED'
 for n,b in ref['artifacts'].items():assert hashlib.sha256((Path(rec.get_local_dir())/'artifacts'/n).read_bytes()).hexdigest()==b['sha256']
 original=rec.load_object('inputs/diagnostic_rows.pkl');rows=rec.load_object('analysis/calibrated_predictions.pkl');fits=rec.load_object('models/calibrations.pkl')
 checks=[]
 for yr in [2016,2026]:
  cutoff=pd.Timestamp(str(yr)+'-01-01',tz='America/New_York');mask=(original.index>=pd.Timestamp('2013-01-01')) & (original.index.year<yr) & original.score_available & original.forecast_available
  for field in ['feature_available_at','forecast_model_available_at','return_available_at']:mask &= pd.to_datetime(original[field],utc=True).lt(cutoff)
  d=original.loc[mask];x=d.cash_advantage_prediction.to_numpy(float);y=d.cash_advantage_1d.to_numpy(float);b=np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean());a=y.mean()-b*x.mean();err=max(abs(a-fits[yr]['intercept']),abs(b-fits[yr]['slope']));assert err<1e-12;assert d.index.equals(fits[yr]['training_dates']);checks.append({'year':yr,'rows':len(d),'coefficient_error':float(err)})
 counts={'raw':0,'prior_mean':0}
 for yr in range(2016,2023):
  d=rows.loc[(rows.index.year==yr)&rows.score_available];y=d.cash_advantage_1d.to_numpy(float);mse=((d.calibrated_prediction.to_numpy(float)-y)**2).mean()
  for role,col in [('raw','cash_advantage_prediction'),('prior_mean','prior_mean_prediction')]:counts[role]+=int(mse<((d[col].to_numpy(float)-y)**2).mean())
 assert counts=={'raw':3,'prior_mean':5}
 d=rows.loc[rows.index.year.isin(range(2016,2023))&rows.score_available];p=d.calibrated_prediction.to_numpy(float);y=d.cash_advantage_1d.to_numpy(float);h=np.log((1+.001)/(1-.001));hit=p>h;mean=float(1e4*(y[hit]-h).mean());assert hit.sum()==9 and abs(mean-(-65.129654))<1e-5
 report={'passed':True,'recorder':rec.id,'scope':'One independent bounded review: all saved artifact hashes; centered-covariance OLS for2016/2026; annual MSE counts; dev10bp event arithmetic. No alternative model or path.','fits':checks,'annual_MSE_better_years':counts,'dev10bp_event_count':int(hit.sum()),'dev10bp_event_mean_net_bp':mean,'decision':'abandon_cash_calibration_route','objective_achieved':False}
 (c/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

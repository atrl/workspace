from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
from scipy.stats import spearmanr
import qlib
from qlib.data import D
from qlib.workflow import R
from src.objective.relative_wealth import stationary_block_bootstrap_indices
r=Path.cwd();c=r/'research/semiconductor_breadth_20261005';s=json.loads((c/'summary.json').read_text());p=json.loads((c/'plan.json').read_text());ref=json.loads((c/'recorder_reference.json').read_text());pit=r/p['sector_source']['provider_uri'];supp=r/p['sector_source']['supplemental_provider_uri'];day=pd.Timestamp('2024-06-28')
qlib.init(provider_uri=str(pit),region='us',kernels=1);cal=pd.DatetimeIndex(D.calendar());pos=cal.get_loc(day);days=cal[pos-60:pos+1];prev=cal[pos-1];hold=pd.read_csv(pit/'sec_nport_holdings_history.csv',dtype={'accession':str});hold.filing_date=pd.to_datetime(hold.filing_date);fdate=hold.loc[hold.filing_date<=prev,'filing_date'].max();snapshot=hold.loc[hold.filing_date==fdate];snapshot=snapshot.loc[snapshot.accession==snapshot.accession.max()];symbols=snapshot.ticker.tolist();assert len(symbols)==30 and 'WOLF' in symbols
x=D.features([n for n in symbols if n!='WOLF'],['$close'],start_time=days[0],end_time=day)['$close'].unstack('instrument').reindex(days)
qlib.init(provider_uri=str(supp),region='us',kernels=1);x['WOLF']=D.features(['WOLF'],['$close'],start_time=days[0],end_time=day)['$close'].xs('WOLF',level='instrument');x=x.loc[:,symbols].astype(float);assert x.notna().all().all() and len(x)==61
ret=x.pct_change(fill_method=None).iloc[1:];last=ret.iloc[-1];w=snapshot.weight_pct.to_numpy(float);w/=w.sum();corr=ret.corr().to_numpy();values={'breadth_up_fraction':float((last>0).mean()),'equal_minus_disclosed_weight_return':float(last.mean()-np.dot(last.to_numpy(),w)),'cross_section_return_std':float(last.std(ddof=0)),'mean_abs_correlation_60':float(np.abs(corr[np.triu_indices(30,1)]).mean())}
with R.uri_context(s['recorder_uri']):
 rec=R.get_recorder(experiment_name=s['experiment'],recorder_id=s['recorder_id']);assert rec.client.get_run(rec.id).info.status=='FINISHED';ar=Path(rec.get_local_dir())/'artifacts'
 for n,h in ref['artifact_sha256'].items():assert hashlib.sha256((ar/n).read_bytes()).hexdigest()==h
 frame=rec.load_object('inputs/features_labels.pkl');pred=rec.load_object('analysis/predictions.pkl');draws=rec.load_object('analysis/bootstrap_draws.pkl');models=rec.load_object('models/fitted_models.pkl');errors={k:abs(v-float(frame.loc[day,k])) for k,v in values.items()};assert max(errors.values())<1e-13
 forecast=models['2024_base_plus_breadth'].predict(frame.loc[[day],p['features']['base_plus_breadth']])[0];forecast_error=float(abs(forecast-pred.loc[day,'base_plus_breadth']));assert forecast_error<1e-14
 ix=stationary_block_bootstrap_indices(observation_count=len(pred),path_length=len(pred),path_count=2000,expected_block_length=20,seed=20261005)[0];z=pred.iloc[ix];y=z.cash_advantage_5d.to_numpy(float);full=z.base_plus_breadth.to_numpy(float);base=z.base.to_numpy(float);delta=float(spearmanr(y,full).statistic-spearmanr(y,base).statistic);saved=draws['pooled_2024_2026_partial_block20'].iloc[0];err=float(abs(delta-saved.delta_ic_full_minus_base));assert err<1e-13
 out={'passed':True,'scope':'One bounded independent review:21frozenartifact hashes;full30memberJune28features fromdirectnativeDprices/SEC;onecachedprediction;pairedbootstrapdraw0ICdifference. No newfits/strategies.','recorder_id':rec.id,'feature_check_date':str(day.date()),'membership_known_from_filing':str(fdate.date()),'feature_max_errors':errors,'cached_forecast_error':forecast_error,'paired_IC_draw0_error':err,'new_fit_count':0,'registered_information_screen_pass':False,'objective_achieved':False};(c/'independent_review.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

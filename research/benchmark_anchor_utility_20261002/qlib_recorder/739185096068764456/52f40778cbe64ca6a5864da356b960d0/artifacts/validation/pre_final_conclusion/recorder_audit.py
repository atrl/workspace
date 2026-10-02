"""Read-only final source/date/portfolio audit; no fitting or parameter search."""
from pathlib import Path
import os,json,hashlib,re
os.environ['MLFLOW_ALLOW_FILE_STORE']='true';os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
import numpy as np
import pandas as pd
from qlib.workflow import R
from src.data.conditional_variance import load_qualified_primary
from src.data.benchmark_anchor_utility import build_anchor_dataset,prepare_training_rows,transform_features,PRICE_FEATURES
from src.data.integrated_allocation import frame_sha256
from src.workflow.benchmark_anchor_utility import _source_hashes
from src.analysis.integrated_allocation import validate_replay,paired_metrics
root=Path.cwd();out=Path('/tmp/benchmark-anchor-validation');card=root/'research/benchmark_anchor_utility_20261002'
summary=json.loads((card/'summary.json').read_text());plan=json.loads((card/'selection_plan.json').read_text());seal=json.loads((card/'selection_seal.json').read_text());initial=json.loads((card/'initial_seal.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={};details={};failures=[]
def check(name,value,detail=None):
 checks[name]=bool(value)
 if not value:failures.append(name)
 if detail is not None:details[name]=detail
check('source_hashes_match_current',summary['source_sha256']==_source_hashes(root))
check('plan_unchanged_from_registration',sha(card/'selection_plan.json')==summary['plan_sha256']==seal['plan_sha256']==initial['plan_sha256'])
check('model_and_data_unchanged_from_initial_seal',all(summary['source_sha256'][p]==h for p,h in initial['source_sha256'].items() if p!='src/workflow/benchmark_anchor_utility.py'))
check('no_promotion_or_execution',summary['objective_achieved'] is False and summary['new_forward_qualified'] is False and summary['production_change'] is False and summary['real_broker_orders'] is False and summary['campaign_status']=='active_research')
provider=load_qualified_primary(root,provider_uri=plan['source']['provider_uri'],parent_summary=f"research/{plan['source']['experiment']}/summary.json",expected_recorder_id=plan['source']['recorder_id'],as_of=plan['source']['as_of'],train_start=plan['source']['model_signal_start'])
check('fresh_native_provider_matches_source_audit',provider['audit']==summary['source_audit'])
with R.uri_context(summary['recorder_uri']):
 recorder=R.get_recorder(experiment_name=summary['experiment'],recorder_id=summary['recorder_id']);load=recorder.load_object
 primary=load('inputs/primary_frame.pkl');prices=load('inputs/execution_prices.pkl');source_audit=load('inputs/source_audit.pkl');source_hashes=load('inputs/source_hashes.pkl')
 forecasts=load('inputs/forecasts.pkl');forecast_fit=load('inputs/forecast_fit_audit.pkl');forecast_models=load('models/forecast_models.pkl')
 development_folds=load('inputs/development_fold_inputs.pkl');outer_folds=load('inputs/diagnostic_fold_inputs.pkl');outer_frames=load('inputs/diagnostic_frames.pkl')
 development_models=load('models/development.pkl');outer_models=load('models/selected_annual_refits.pkl')
 paths=load('analysis/portfolio_replays.pkl');seed_metrics=load('analysis/seed_metrics.pkl');gates=load('analysis/gates.pkl');bootstrap=load('analysis/bootstrap.pkl')
 comparison=load('analysis/comparison.pkl');annual=load('analysis/annual.pkl');native=load('portfolio_analysis/native_reconciliation.pkl');path_audits=load('analysis/independent_path_audits.pkl')
check('source_objects_match_native',frame_sha256(primary)==frame_sha256(provider['primary_frame']) and frame_sha256(prices)==frame_sha256(provider['execution_prices']) and source_audit==summary['source_audit'] and source_hashes==summary['source_sha256'])
check('primary_source_asof',str(primary.index.max().date())==plan['source']['as_of'] and str(prices.index.max().date())==plan['source']['as_of'])
if summary['selected']['variant']=='price_vol':
 from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES
 frozen=outer_frames['frozen2022_parameters']
 predicted=forecast_models[2023].predict(frozen.loc[:,list(COMMON_PRIMARY_FEATURES)])
 check('frozen_forecast_uses_only_2023_head',np.array_equal(predicted[:,1],frozen.future_vol_prediction.to_numpy(float)) and frozen.forecast_available.eq(1).all() and pd.to_datetime(frozen.forecast_model_available_at,utc=True).eq(pd.Timestamp('2023-01-01',tz='America/New_York').tz_convert('UTC')).all())
ready=forecast_fit[forecast_fit.available]
check('all_forecast_fits_strictly_mature',bool((pd.to_datetime(ready.label_available_at_max,utc=True)<pd.to_datetime(ready.decision_at,utc=True)).all()))
dataset=build_anchor_dataset(forecasts,as_of=plan['source']['as_of']);fold_report=[]
folds=development_folds|outer_folds;records={**{(name,y):r for name,ys in development_models.items() for y,r in ys.items()},**{(name,y):r for name,ys in outer_models.items() for y,r in ys.items()}}
check('all_25_annual_variant_folds_present',len(folds)==25 and set(folds)==set(records))
for (name,year),fold in sorted(folds.items()):
 rows,ma=prepare_training_rows(dataset,year,min_rows=plan['training']['minimum_rows']);scaler=fold['scaler'];record=records[(name,year)]
 prefix=f'fold:{name}:{year}'
 check(prefix+':native_rows',frame_sha256(rows)==frame_sha256(fold['rows']) and ma==fold['audit'])
 check(prefix+':binding',record['variant']==name and record['model_year']==year and record['scaler_sha256']==scaler['scaler_sha256'] and record['scaler']==scaler and scaler['training_rows_sha256']==frame_sha256(rows))
 x=rows[list(PRICE_FEATURES)].to_numpy(float);constant=np.ptp(x,axis=0)==0;means=x.mean(0);std=x.std(0,ddof=0);means[constant]=x[0,constant];std[constant]=1.
 check(prefix+':price_scaler',np.array_equal(means,np.asarray(scaler['price_mean'])) and np.array_equal(std,np.asarray(scaler['price_scale'])))
 if name=='price_vol' and scaler['eligible']:
  available=rows.forecast_available.eq(1).to_numpy();v=rows.future_vol_prediction.to_numpy(float)[available]
  mean=float(v.mean());scale=1. if np.ptp(v)==0 else float(v.std(ddof=0))
  check(prefix+':vol_scaler',scaler['vol_fitted'] and scaler['available_training_forecast_rows']==len(v) and np.isclose(mean,scaler['vol_mean'],rtol=0,atol=1e-15) and np.isclose(scale,scaler['vol_scale'],rtol=0,atol=1e-15))
 transformed=transform_features(rows,scaler,name)
 check(prefix+':features_and_returns',np.array_equal(transformed['features'],fold['features']) and np.array_equal(rows[['soxl_return','soxx_return']].to_numpy(float),fold['returns']))
 check(prefix+':worker_input_hash',record['training_feature_sha256']==hashlib.sha256(fold['features'].tobytes()).hexdigest() and record['training_return_sha256']==hashlib.sha256(fold['returns'].tobytes()).hexdigest())
 check(prefix+':all_seeds',set(record['seed_models'])=={17,41,71} and all(record['seed_models'][seed]['seed']==seed for seed in (17,41,71)))
 fold_report.append({'variant':name,'year':year,'rows':len(rows),'last_signal':str(rows.index.max().date()),'max_return_available':ma['return_available_at_max'],'scaler_sha256':scaler['scaler_sha256'],'invalid_training':record['invalid_training'],'numerical_failure':record['numerical_failure']})
# Reproduce saved portfolio targets from serialized weights and scalers, using
# each saved path's own pretrade holdings; verify accounting with direct algebra.
def alphas(payloads,x):
 values=[]
 for payload in payloads:
  p=payload['state_dict'];z=x@np.asarray(p['hidden.weight']).T+np.asarray(p['hidden.bias']);sigmoid=np.empty_like(z);positive=z>=0;sigmoid[positive]=1/(1+np.exp(-z[positive]));v=np.exp(z[~positive]);sigmoid[~positive]=v/(1+v)
  values.append(np.c_[np.ones(len(x)),sigmoid]@np.asarray(p['theta']).T)
 return np.mean(values,axis=0)
assets=['soxl','soxx','cash'];path_report=[];maximum_errors={};saved_groups=paths.groupby(['segment','variant','seed','cost_bps'],sort=True)
check('exactly_32_saved_paths',len(saved_groups)==32)
for (segment,name,seed,cost),saved in saved_groups:
 p=validate_replay(saved.reset_index(drop=True));dates=pd.DatetimeIndex(p.signal_date);prefix=f'path:{segment}:{name}:{seed}:{cost}'
 is_benchmark=name=='SOXX_buy_hold';chosen=summary['selected']['variant']
 if segment=='development':frame=forecasts.loc[dates];annual_models=development_models.get(name);mode=None
 else:
  mode='frozen2022_parameters' if name.endswith('frozen2022_parameters') else 'annual_refit';frame=outer_frames[mode].loc[dates];annual_models=outer_models[chosen]
 h=p[[f'pretrade_{a}_weight' for a in assets]].to_numpy(float) if not is_benchmark else None
 z=p[[f'{a}_weight' for a in assets]].to_numpy(float)
 r=np.c_[frame[['soxl_return','soxx_return']].to_numpy(float),np.zeros(len(frame))];g=1+p.strategy_return.to_numpy(float);fee=p.fees.to_numpy(float);turn=p.turnover.to_numpy(float)
 if is_benchmark:
  analytic=1+r[:,1];analytic[0]/=1+cost/1e4
  error=float(np.max(abs(g-analytic)));check(prefix+':anchor',error<=1e-10);path_report.append({'segment':segment,'variant':name,'seed':str(seed),'cost_bps':cost,'sessions':len(p),'anchor_multiplier_error':error});continue
 expected_h=np.vstack(([0.,0.,1.],p[[f'ending_{a}_weight' for a in assets]].to_numpy(float)[:-1]))
 errors={'pretrade':float(np.max(abs(h-expected_h))),'simplex':float(np.max(abs(z.sum(1)-1))),
  'fee_L1':float(np.max(abs(fee-cost/1e4*np.abs((1-fee[:,None])*z[:,:2]-h[:,:2]).sum(1)))),
  'turnover':float(np.max(abs(turn-np.abs((1-fee[:,None])*z[:,:2]-h[:,:2]).sum(1)))),
  'net_multiplier':float(np.max(abs(g-(1-fee)*(1+(z*r).sum(1))))),
  'ending':float(np.max(abs(p[[f'ending_{a}_weight' for a in assets]].to_numpy(float)-(1-fee[:,None])*z*(1+r)/g[:,None]))),
  'equity':float(np.max(abs(p.equity.to_numpy(float)/np.cumprod(g)-1))),'target_from_model':0.}
 for year in sorted(dates.year.unique()):
  mask=dates.year==year;use=2023 if mode=='frozen2022_parameters' else int(year);record=annual_models[use];rows=frame.loc[mask];transformed=transform_features(rows,record['scaler'],chosen if segment!='development' else name)
  check(prefix+f':year:{year}:causal_bundle',record['model_year']==use and p.loc[mask,'model_year'].eq(use).all() and pd.Timestamp(record['scaler']['cutoff'])<=pd.Timestamp(rows.index.min()).tz_localize('America/New_York'))
  payloads=record['ensemble_models'] if str(seed)=='ensemble' else [record['seed_models'][int(seed)]]
  a=alphas(payloads,transformed['features']);bad=transformed['invalid_forecast_mask'];a[bad]=0.
  predicted=np.c_[a[:,0]+a[:,2]*h[mask,0],1-a.sum(1)+a[:,2]*h[mask,1],a[:,1]+a[:,2]*h[mask,2]]
  errors['target_from_model']=max(errors['target_from_model'],float(np.max(abs(z[mask]-predicted))))
  check(prefix+f':year:{year}:failure_flags',np.array_equal(p.loc[mask,'observed_forecast_fallback'].to_numpy(bool),bad) and p.loc[mask,'invalid_training_fallback'].eq(record['invalid_training']).all() and p.loc[mask,'numerical_failure'].eq(record['numerical_failure']).all())
 check(prefix+':accounting_and_target',max(errors.values())<=1e-10 and z.min()>=-1e-12,errors)
 check(prefix+':source_bounds',pd.Timestamp(p.exit_date.max())<=pd.Timestamp(plan['source']['as_of']))
 for k,v in errors.items():maximum_errors[k]=max(maximum_errors.get(k,0.),v)
 path_report.append({'segment':segment,'variant':name,'seed':str(seed),'cost_bps':cost,'sessions':len(p),'max_error':max(errors.values()),'failure_events':int(p[['observed_forecast_fallback','invalid_training_fallback','numerical_failure']].any(axis=1).sum())})
check('same_two_seed_gate',all(int(group.pivot(index='seed',columns='cost_bps',values='log_excess_sum').gt(0).all(1).sum())==int(gates[gates.variant.eq(name)].seeds_positive_at_both_costs.iloc[0]) for name,group in seed_metrics.groupby('variant')))
for name,group in seed_metrics.groupby('variant'):
 expected={(s,c) for s in (17,41,71) for c in (10,25)}
 check('seed_completeness:'+name,set(zip(group.seed,group.cost_bps))==expected)
 for row in group.itertuples():
  p=paths[paths.segment.eq('development')&paths.variant.eq(name)&paths.seed.astype(str).eq(str(row.seed))&paths.cost_bps.eq(row.cost_bps)].reset_index(drop=True)
  b=paths[paths.segment.eq('development')&paths.variant.eq('SOXX_buy_hold')&paths.cost_bps.eq(row.cost_bps)].reset_index(drop=True)
  check(f'seed_metric:{name}:{row.seed}:{row.cost_bps}',abs(paired_metrics(p,b)['log_excess_sum']-row.log_excess_sum)<=1e-12)
check('bootstrap_family_and_cost_scope',bootstrap.declared_variant_family_size.eq(17).all() and set(bootstrap.cost_bps)=={10,25} and set(bootstrap.mean_block)=={20,60})
check('selected_seal_matches_summary',seal['selected']==summary['selected'] and seal['selection_uses_diagnostic_signal_years'] is False and summary['selection_uses_diagnostic_signal_years'] is False)
check('native_both_costs_pass',set(native)=={10,25} and all(v['audit']==summary['native_qlib_audit'][str(c)] and v['audit']['status']=='passed' and np.isfinite(v['audit']['maximum_session_return_error']) and v['audit']['maximum_session_return_error']<=1e-7 and v['audit']['tolerance']==1e-7 for c,v in native.items()))
for file,value in [('backtest_benchmark_comparison.csv',comparison),('annual_metrics.csv',annual),('development_gates.csv',gates),('seed_metrics.csv',seed_metrics)]:
 pd.testing.assert_frame_equal(pd.read_csv(card/file),value,check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
check('small_CSV_outputs_equal_Recorder',True)
nb=json.loads((card/'benchmark_anchor_utility_20261002.ipynb').read_text());code=[c for c in nb['cells'] if c['cell_type']=='code'];outputs=[o for c in code for o in c.get('outputs',[])];text='\n'.join(''.join(o.get('text',o.get('data',{}).get('text/plain',[]))) for o in outputs)
check('notebook_executed_no_errors',all(c.get('execution_count') is not None for c in code) and not any(o['output_type']=='error' for o in outputs))
check('notebook_final_Recorder_visible',summary['recorder_id'] in text)
sealpos=text.find('Anchor utility development sealed:');outerpos=re.search(r'Anchor utility \S+ year=2023 seed=',text)
check('selection_sealed_before_diagnostic_fit_output',sealpos>=0 and outerpos is not None and sealpos<outerpos.start())
markdown=''.join(nb['cells'][-1].get('source',[]));conclusion=(card/'conclusion.md').read_text()
notebook_pending='待执行' in markdown or summary['recorder_id'] not in markdown or markdown.strip()!=conclusion.strip()
report={'recorder_id':summary['recorder_id'],'scope':'Read-only source/date/scaler/saved-portfolio/gate/native audit; optimizer epochs audited separately by model agent; no market refitting or new policy search',
 'checks':checks,'failures':failures,'details':details,'source_sha256':summary['source_sha256'],'plan_sha256':summary['plan_sha256'],'source_asof':plan['source']['as_of'],
 'annual_variant_folds':fold_report,'saved_path_count':len(path_report),'path_audits':path_report,'maximum_independent_portfolio_errors':maximum_errors,
 'native_maximum_return_errors':{str(c):v['audit']['maximum_session_return_error'] for c,v in native.items()},'qualification':summary['historical_qualification_pass'],'selected':summary['selected'],
 'notebook_final_markdown_pending':notebook_pending,'notebook_sha256':sha(card/'benchmark_anchor_utility_20261002.ipynb'),'optimizer_audit':'separate report from regime_model pending'}
(out/'recorder-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda value:value.item() if isinstance(value,np.generic) else str(value))+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('checks','details','source_sha256','annual_variant_folds','path_audits')},indent=2,default=lambda value:value.item() if isinstance(value,np.generic) else str(value)))

"""Read-only source/date/bundle/saved-path/gate/native audit, no NN fitting."""
from pathlib import Path
import json,hashlib,re
import numpy as np
import pandas as pd
from qlib.workflow import R
from src.data.constrained_net_excess import load_constrained_inputs
from src.data.benchmark_anchor_utility import prepare_training_rows,transform_features,PRICE_FEATURES
from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES,frame_sha256
from src.analysis.integrated_allocation import validate_replay,paired_metrics
from src.workflow.constrained_net_excess import _source_hashes
root=Path.cwd();out=Path('/tmp/constrained-net-excess-validation');card=root/'research/constrained_net_excess_20261003'
summary=json.loads((card/'summary.json').read_text());plan=json.loads((card/'selection_plan.json').read_text());seal=json.loads((card/'selection_seal.json').read_text());initial=json.loads((card/'initial_seal.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();checks={};details={}
def check(name,value,detail=None):
 checks[name]=bool(value)
 if detail is not None:details[name]=detail
check('plan_bytes_match_initial_seal_and_summary',sha(card/'selection_plan.json')==summary['plan_sha256']==initial['plan_sha256']==seal['plan_sha256'])
check('all23_source_hashes_match',len(summary['source_sha256'])==23 and _source_hashes(root)==summary['source_sha256']==initial['source_sha256'])
check('all4_test_hashes_match_initial_seal',len(initial['test_sha256'])==4 and all(sha(root/p)==h for p,h in initial['test_sha256'].items()))
# Reuse only native input reconstruction and frozen forecast/scaler validation.
# No fit_policy, optimizer step or parameter choice is called by this audit.
bundle=load_constrained_inputs(root,plan)
check('native_frozen_parent_audit_matches',bundle['source_audit']==summary['source_audit'])
check('exact_parent_input_scope',bundle['source_audit']['verified_parent_artifacts']==8 and bundle['source_audit']['verified_development_folds']==21 and bundle['source_audit']['forecast_refit_performed'] is False and bundle['source_audit']['allocation_models_loaded'] is False and bundle['source_audit']['data_refreshed'] is False)
with R.uri_context(summary['recorder_uri']):
 recorder=R.get_recorder(experiment_name=summary['experiment'],recorder_id=summary['recorder_id']);load=recorder.load_object
 source_audit=load('inputs/source_audit.pkl');source_details=load('inputs/source_details.pkl');source_hashes=load('inputs/source_hashes.pkl');raw=load('inputs/primary_frame.pkl');prices=load('inputs/execution_prices.pkl')
 forecasts=load('inputs/forecasts.pkl');forecast_audit=load('inputs/forecast_fit_audit.pkl');forecast_models=load('models/forecast_models.pkl')
 dev_folds=load('inputs/development_fold_inputs.pkl');outer_folds=load('inputs/diagnostic_fold_inputs.pkl');outer_frames=load('inputs/diagnostic_frames.pkl')
 dev_models=load('models/development.pkl');outer_models=load('models/selected_annual_refits.pkl')
 paths=load('analysis/portfolio_replays.pkl');seed_metrics=load('analysis/seed_metrics.pkl');gates=load('analysis/gates.pkl');bootstrap=load('analysis/bootstrap.pkl');comparison=load('analysis/comparison.pkl');annual=load('analysis/annual.pkl');windows=load('analysis/window_metrics.pkl');native=load('portfolio_analysis/native_reconciliation.pkl')
 training_index=load('analysis/training_diagnostics.pkl');ensemble_index=load('analysis/ensemble_audits.pkl');artifact_root=Path(recorder.get_local_dir())/'artifacts'
 check('final_Recorder_FINISHED',recorder.client.get_run(recorder.id).info.status=='FINISHED')
 check('recorded_plan_and_seal_match',sha(artifact_root/'inputs/selection_plan.json')==summary['plan_sha256'] and json.loads((artifact_root/'inputs/selection_seal.json').read_text())==seal)
check('recorded_input_audit_objects_match',source_audit==summary['source_audit'] and source_details==bundle['source_details'] and source_hashes==summary['source_sha256'])
check('all_primary_price_forecast_frames_match_parent',frame_sha256(raw)==frame_sha256(bundle['raw']) and frame_sha256(prices)==frame_sha256(bundle['prices']) and frame_sha256(forecasts)==frame_sha256(bundle['forecasts']))
check('source_asof_fixed',str(raw.index.max().date())==plan['source']['as_of']==str(prices.index.max().date()))
check('all75fit_and25ensemble_audit_pointers_present',len(training_index)==75 and len(ensemble_index)==25 and all((artifact_root/item['recorder_artifact']).is_file() for item in training_index+ensemble_index))
check('aggregate_indexes_do_not_duplicate_epoch_or_daily_paths',all(not any(key in item for key in ('epochs','history','training_paths','rejected_ensemble_training_paths')) for item in training_index+ensemble_index))
check('F_and_legacy_J_semantics_explicit',summary['training_objective']=='F=min_c63mean_net_log_excess; legacy_J_report_only' and all('legacy_J' in item and 'centered_worst_utility' in item for item in training_index))
folds=dev_folds|outer_folds;records={**{(name,y):r for name,years in dev_models.items() for y,r in years.items()},**{(name,y):r for name,years in outer_models.items() for y,r in years.items()}};fold_report=[]
check('all25_annual_variant_folds_bound',len(folds)==25 and set(folds)==set(records))
for (name,year),fold in sorted(folds.items()):
 rows,maturity=prepare_training_rows(bundle['dataset'],year,min_rows=plan['training']['minimum_rows']);scaler=fold['scaler'];record=records[(name,year)];prefix=f'fold:{name}:{year}'
 check(prefix+':native_rows',frame_sha256(rows)==frame_sha256(fold['rows']) and maturity==fold['audit'])
 check(prefix+':model_scaler_binding',record['variant']==name and record['model_year']==year and record['scaler']==scaler and record['scaler_sha256']==scaler['scaler_sha256'] and scaler['training_rows_sha256']==frame_sha256(rows))
 x=rows[list(PRICE_FEATURES)].to_numpy(float);constant=np.ptp(x,axis=0)==0;mean=x.mean(0);std=x.std(0,ddof=0);mean[constant]=x[0,constant];std[constant]=1.
 check(prefix+':prior_price_scaler',np.array_equal(mean,np.asarray(scaler['price_mean'])) and np.array_equal(std,np.asarray(scaler['price_scale'])))
 transformed=transform_features(rows,scaler,name)
 check(prefix+':exact_matrix_binding',np.array_equal(transformed['features'],fold['features']) and np.array_equal(rows[['soxl_return','soxx_return']].to_numpy(float),fold['returns']) and record['training_feature_sha256']==hashlib.sha256(fold['features'].tobytes()).hexdigest() and record['training_return_sha256']==hashlib.sha256(fold['returns'].tobytes()).hexdigest())
 if (name,year) in bundle['parent_dev_folds']:
  parent=bundle['parent_dev_folds'][(name,year)]
  check(prefix+':exact_phase5_development_inputs',scaler==parent['scaler'] and np.array_equal(fold['features'],parent['features']) and np.array_equal(fold['returns'],parent['returns']))
 if name=='price_vol' and scaler['eligible']:
  used=rows.forecast_available.eq(1).to_numpy();vol=rows.future_vol_prediction.to_numpy(float)[used];volscale=1. if np.ptp(vol)==0 else float(vol.std(ddof=0))
  check(prefix+':prior_available_vol_scaler',scaler['available_training_forecast_rows']==len(vol) and np.isclose(vol.mean(),scaler['vol_mean'],atol=1e-15,rtol=0) and np.isclose(volscale,scaler['vol_scale'],atol=1e-15,rtol=0))
 check(prefix+':same_three_seeds',set(record['seed_models'])=={17,41,71} and all(record['seed_models'][seed]['seed']==seed for seed in (17,41,71)))
 fold_report.append({'variant':name,'year':year,'rows':len(rows),'max_return_available':maturity['return_available_at_max'],'cutoff':maturity['cutoff'],'invalid_training':record['invalid_training'],'numerical_failure':record['numerical_failure']})
ready=forecast_audit[forecast_audit.available]
check('forecast_maturity_strict',bool((pd.to_datetime(ready.label_available_at_max,utc=True)<pd.to_datetime(ready.decision_at,utc=True)).all()))
chosen=summary['selected']['variant']
if chosen=='price_vol':
 frozen=outer_frames['frozen2022_parameters'];prediction=forecast_models[2023].predict(frozen.loc[:,list(COMMON_PRIMARY_FEATURES)])
 check('frozen_forecast_reproduces_2023_head_only',np.array_equal(prediction[:,1],frozen.future_vol_prediction.to_numpy(float)) and frozen.forecast_available.eq(1).all() and pd.to_datetime(frozen.forecast_model_available_at,utc=True).eq(pd.Timestamp('2023-01-01',tz='America/New_York').tz_convert('UTC')).all())
def alphas(payloads,x):
 result=[]
 for payload in payloads:
  p=payload['state_dict'];z=x@np.asarray(p['hidden.weight']).T+np.asarray(p['hidden.bias']);sig=np.empty_like(z);pos=z>=0;sig[pos]=1/(1+np.exp(-z[pos]));e=np.exp(z[~pos]);sig[~pos]=e/(1+e);result.append(np.c_[np.ones(len(x)),sig]@np.asarray(p['theta']).T)
 return np.mean(result,axis=0)
assets=['soxl','soxx','cash'];path_report=[];max_errors={};groups=paths.groupby(['segment','variant','seed','cost_bps'],sort=True)
check('exactly32_saved_paths',len(groups)==32)
for (segment,name,seed,cost),saved in groups:
 p=validate_replay(saved.reset_index(drop=True));dates=pd.DatetimeIndex(p.signal_date);prefix=f'path:{segment}:{name}:{seed}:{cost}';benchmark=name=='SOXX_buy_hold'
 if segment=='development':frame=forecasts.loc[dates];annual_models=dev_models.get(name);mode=None
 else:
  mode='frozen2022_parameters' if name.endswith('frozen2022_parameters') else 'annual_refit';frame=outer_frames[mode].loc[dates];annual_models=outer_models[chosen]
 r=np.c_[frame[['soxl_return','soxx_return']].to_numpy(float),np.zeros(len(frame))];g=1+p.strategy_return.to_numpy(float);fee=p.fees.to_numpy(float);turn=p.turnover.to_numpy(float)
 if benchmark:
  analytic=1+r[:,1];analytic[0]/=1+cost/1e4;error=float(np.max(abs(g-analytic)));check(prefix+':independent_anchor',error<=1e-10);path_report.append({'segment':segment,'variant':name,'seed':str(seed),'cost_bps':float(cost),'sessions':len(p),'anchor_multiplier_error':error});continue
 h=p[[f'pretrade_{a}_weight' for a in assets]].to_numpy(float);z=p[[f'{a}_weight' for a in assets]].to_numpy(float);ending=p[[f'ending_{a}_weight' for a in assets]].to_numpy(float)
 expected_h=np.vstack(([0.,0.,1.],ending[:-1]));errors={'pretrade':float(np.max(abs(h-expected_h))),'simplex':float(np.max(abs(z.sum(1)-1))),
  'fee_L1':float(np.max(abs(fee-cost/1e4*np.abs((1-fee[:,None])*z[:,:2]-h[:,:2]).sum(1)))),
  'turnover':float(np.max(abs(turn-np.abs((1-fee[:,None])*z[:,:2]-h[:,:2]).sum(1)))),
  'net_multiplier':float(np.max(abs(g-(1-fee)*(1+(z*r).sum(1))))),
  'ending':float(np.max(abs(ending-(1-fee[:,None])*z*(1+r)/g[:,None]))),
  'equity':float(np.max(abs(p.equity.to_numpy(float)/np.cumprod(g)-1))),'target_from_model':0.}
 flags=['observed_forecast_fallback','invalid_training_fallback','numerical_failure']
 check(prefix+':complete_boolean_flags',p[flags].notna().all().all() and all(p[c].map(lambda v:isinstance(v,(bool,np.bool_))).all() for c in flags))
 for year in sorted(dates.year.unique()):
  mask=dates.year==year;use=2023 if mode=='frozen2022_parameters' else int(year);record=annual_models[use];rows=frame.loc[mask];transformed=transform_features(rows,record['scaler'],chosen if segment!='development' else name)
  check(prefix+f':{year}:causal_bundle',record['model_year']==use and p.loc[mask,'model_year'].eq(use).all() and pd.Timestamp(record['scaler']['cutoff'])<=pd.Timestamp(rows.index.min()).tz_localize('America/New_York'))
  payloads=record['ensemble_models'] if str(seed)=='ensemble' else [record['seed_models'][int(seed)]];a=alphas(payloads,transformed['features']);bad=transformed['invalid_forecast_mask'];a[bad]=0.
  target=np.c_[a[:,0]+a[:,2]*h[mask,0],1-a.sum(1)+a[:,2]*h[mask,1],a[:,1]+a[:,2]*h[mask,2]];errors['target_from_model']=max(errors['target_from_model'],float(np.max(abs(z[mask]-target))))
  check(prefix+f':{year}:flags_match_bundle',np.array_equal(p.loc[mask,'observed_forecast_fallback'].to_numpy(bool),bad) and p.loc[mask,'invalid_training_fallback'].eq(record['invalid_training']).all() and p.loc[mask,'numerical_failure'].eq(record['numerical_failure']).all())
  if str(seed)=='ensemble':check(prefix+f':{year}:frozen_ensemble_choice',p.loc[mask,'ensemble_anchor_selected'].eq(record['ensemble_audit']['anchor_selected']).all())
 check(prefix+':accounting_and_serialized_target',max(errors.values())<=1e-10 and z.min()>=-1e-12,errors)
 check(prefix+':source_date_bounds',p.exit_date.max()<=pd.Timestamp(plan['source']['as_of']))
 for key,value in errors.items():max_errors[key]=max(max_errors.get(key,0.),value)
 path_report.append({'segment':segment,'variant':name,'seed':str(seed),'cost_bps':float(cost),'sessions':len(p),'max_error':max(errors.values()),'failure_events':int(p[flags].any(axis=1).sum())})
check('exact18_seed_cost_metrics',len(seed_metrics)==18)
for name,table in seed_metrics.groupby('variant'):
 check('seed_complete:'+name,set(zip(table.seed,table.cost_bps))=={(s,c) for s in (17,41,71) for c in (10,25)})
 both=int(table.pivot(index='seed',columns='cost_bps',values='log_excess_sum').gt(0).all(1).sum())
 check('same_two_seeds_both_costs:'+name,gates[gates.variant.eq(name)].seeds_positive_at_both_costs.eq(both).all() and gates[gates.variant.eq(name)].minimum_positive_seeds_at_both_costs.eq(both>=2).all())
 for row in table.itertuples():
  p=paths[paths.segment.eq('development')&paths.variant.eq(name)&paths.seed.astype(str).eq(str(row.seed))&paths.cost_bps.eq(row.cost_bps)].reset_index(drop=True);b=paths[paths.segment.eq('development')&paths.variant.eq('SOXX_buy_hold')&paths.cost_bps.eq(row.cost_bps)].reset_index(drop=True)
  check(f'seed_metric:{name}:{row.seed}:{row.cost_bps}',abs(paired_metrics(p,b)['log_excess_sum']-row.log_excess_sum)<=1e-12)
check('bootstrap_family20_blocks_and_costs',bootstrap.declared_variant_family_size.eq(20).all() and set(bootstrap.mean_block)=={20,60} and set(bootstrap.cost_bps)=={10,25})
for row in gates.itertuples():
 p=paths[paths.segment.eq('development')&paths.variant.eq(row.variant)&paths.seed.eq('ensemble')&paths.cost_bps.eq(row.cost_bps)]
 check(f'gate_flags:{row.variant}:{row.cost_bps}',bool(row.zero_invalid_source_or_numerical_events)==(not bool(p[['observed_forecast_fallback','invalid_training_fallback','numerical_failure']].any().any())))
check('gate_all_pass_matches_all_registered_checks',gates[[c for c in gates if gates[c].dtype==bool and c not in ('all_pass','seed_evidence_complete')]].all(axis=1).equals(gates.all_pass))
check('native_both_costs_same_R_and_tolerance',set(native)=={10,25} and all(v['audit']==summary['native_qlib_audit'][str(c)] and v['audit']['status']=='passed' and v['audit']['tolerance']==1e-7 and np.isfinite(v['audit']['maximum_session_return_error']) and v['audit']['maximum_session_return_error']<=1e-7 for c,v in native.items()))
check('development_seal_matches_final_selection',seal['selected']==summary['selected'] and seal['selection_uses_diagnostic_signal_years'] is False)
for file,value in [('backtest_benchmark_comparison.csv',comparison),('annual_metrics.csv',annual),('development_gates.csv',gates),('seed_metrics.csv',seed_metrics),('window_metrics.csv',windows)]:
 pd.testing.assert_frame_equal(pd.read_csv(card/file),value,check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
check('key_sidecars_match_Recorder',True)
check('Aug3_is_actual_trade_origin_for8cases',len(windows)==8 and windows.sessions.eq(38).all() and windows.start_signal.eq('2026-07-31').all() and windows.last_exit.eq('2026-09-25').all() and pd.Timestamp(forecasts.loc['2026-07-31','trade_date'])==pd.Timestamp('2026-08-03'))
check('research_only_no_promotion_or_orders',summary['objective_achieved'] is False and summary['new_forward_qualified'] is False and summary['production_change'] is False and summary['real_broker_orders'] is False and summary['campaign_status']=='active_research')
nbpath=card/'constrained_net_excess_20261003.ipynb';nb=json.loads(nbpath.read_text());code=[c for c in nb['cells'] if c['cell_type']=='code'];outputs=[o for c in code for o in c.get('outputs',[])];text='\n'.join(''.join(o.get('text',o.get('data',{}).get('text/plain',[]))) for o in outputs)
check('notebook_executed_without_errors',all(c.get('execution_count') is not None for c in code) and not any(o['output_type']=='error' for o in outputs))
check('notebook_final_Recorder_present',summary['recorder_id'] in text)
sealpos=text.find('Constrained excess development sealed:');outerpos=re.search(r'Constrained excess \S+ year=2023 seed=',text);check('seal_precedes_any_diagnostic_fit_output',sealpos>=0 and outerpos is not None and sealpos<outerpos.start())
markdown=''.join(nb['cells'][-1].get('source',[]));conclusion=(card/'conclusion.md').read_text();pending=markdown.strip()!=conclusion.strip() or summary['recorder_id'] not in markdown
report={'recorder_id':summary['recorder_id'],'scope':'Read-only source/native input/date/scaler/model-bundle/saved-target/accounting/gate/native audit. No neural fitting or optimizer updates. Full75fit/25ensemble/dual audit is separate.',
 'checks':checks,'failures':[k for k,value in checks.items() if not value],'details':details,'source_sha256':summary['source_sha256'],'plan_sha256':summary['plan_sha256'],'source_asof':plan['source']['as_of'],
 'parent_input_artifacts':8,'annual_variant_folds':fold_report,'saved_path_count':len(path_report),'path_audits':path_report,'maximum_independent_portfolio_errors':max_errors,
 'native_maximum_return_errors':{str(c):v['audit']['maximum_session_return_error'] for c,v in native.items()},'qualification':summary['historical_qualification_pass'],'selected':summary['selected'],'notebook_final_markdown_pending':pending,'notebook_sha256':sha(nbpath),'optimizer_audit':'separate model-agent report pending'}
(out/'source-recorder-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda value:value.item() if isinstance(value,np.generic) else str(value))+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('checks','details','source_sha256','annual_variant_folds','path_audits')},indent=2))

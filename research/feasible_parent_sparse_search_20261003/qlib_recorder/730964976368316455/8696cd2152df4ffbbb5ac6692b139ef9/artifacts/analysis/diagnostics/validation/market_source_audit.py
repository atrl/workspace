"""Bounded final market audit of stored books/metadata, never a fit or alternate policy replay.
Run only after the final market reference is complete. Optimizer trial/training
path evidence is audited separately by the model reviewer.
"""
from pathlib import Path
from dataclasses import asdict
import json,hashlib
import numpy as np,pandas as pd
import qlib
from qlib.workflow import R
from qlib.data import D
from src.data.qlib_native import QlibProviderConfig,load_qlib_feature_frame
from src.data.price_basis import price_frame_sha256
from src.data.integrated_allocation import frame_sha256,COMMON_PRIMARY_FEATURES
from src.data.benchmark_anchor_utility import build_anchor_dataset,prepare_training_rows,PRICE_FEATURES
from src.data.sparse_holdings_net_excess import transform_for_variant,variant_mapping
from src.model.sparse_holdings_net_excess import policy_from_dict,_numpy_market,_numpy_decision
from src.analysis.integrated_allocation import validate_replay,paired_metrics
from src.workflow.stable_excess_integration import choose_family
from src.workflow.feasible_parent_sparse_search import _source_hashes,parent_spec
root=Path.cwd();out=Path('/tmp/feasible-parent-validation');card=root/'research/feasible_parent_sparse_search_20261003'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();ah=lambda a:hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
ref=json.loads((card/'market_reference.json').read_text())
if ref.get('complete') is not True:raise SystemExit('NOT READY: immutable completed market reference required')
s=json.loads((card/'summary.json').read_text());plan=json.loads((card/'selection_plan.json').read_text());seal=json.loads((card/'selection_seal.json').read_text());impl=json.loads((card/'implementation_seal.json').read_text());pre=json.loads((card/'preflight_reference.json').read_text())
checks={};details={}
def check(name,value,detail=None):
 checks[name]=bool(value)
 if detail is not None:details[name]=detail
check('market_and_summary_identity',s['recorder_id']==ref['recorder_id']=='8696cd2152df4ffbbb5ac6692b139ef9')
check('source_plan_test_seals',_source_hashes(root)==s['source_sha256']==ref['source_sha256']==impl['source_sha256'] and len(s['source_sha256'])==36 and sha(card/'selection_plan.json')==s['plan_sha256']==ref['plan_sha256']==seal['plan_sha256']==impl['plan_sha256'] and all(sha(root/p)==h for p,h in impl['test_sha256'].items()))
check('passing_distinct_prefit_binding',pre['complete'] is True and pre['passed'] is True and pre['recorder_id']!=ref['recorder_id'] and ref['preflight_reference_sha256']==sha(card/'preflight_reference.json'))
qlib.init(provider_uri=str(root/plan['source']['provider_uri']),region='us',kernels=1)
verified=[]
with R.uri_context(ref['recorder_uri']):
 recorder=R.get_recorder(experiment_name=plan['experiment'],recorder_id=ref['recorder_id']);base=Path(recorder.get_local_dir())/'artifacts'
 check('market_Recorder_FINISHED',recorder.client.get_run(recorder.id).info.status=='FINISHED')
 def load(path):
  matches=[v for v in ref['artifacts'].values() if v['artifact']==path]
  if len(matches)!=1:raise ValueError('missing/duplicate bound artifact '+path)
  item=matches[0];p=(base/path).resolve()
  if not p.is_relative_to(base.resolve()) or sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:raise ValueError('bad bound artifact '+path)
  obj=recorder.load_object(path);verified.append(item)
  if sha(p)!=item['sha256']:raise ValueError('changed artifact '+path)
  return obj
 raw=load('inputs/primary_frame.pkl');prices=load('inputs/execution_prices.pkl');forecasts=load('inputs/forecasts.pkl')
 source=load('inputs/source_audit.pkl');source_details=load('inputs/source_details.pkl');source_hashes=load('inputs/source_hashes.pkl')
 dev_folds=load('inputs/development_fold_inputs.pkl');outer_folds=load('inputs/diagnostic_fold_inputs.pkl');outer_frames=load('inputs/diagnostic_frames.pkl')
 dev=load('models/development.pkl');outer=load('models/selected_annual_refits.pkl');forecast_models=load('models/forecast_models.pkl');forecast_audit=load('inputs/forecast_fit_audit.pkl')
 paths=load('analysis/portfolio_replays.pkl');gates=load('analysis/gates.pkl');boot=load('analysis/bootstrap.pkl');comparison=load('analysis/comparison.pkl');annual=load('analysis/annual.pkl');seed_metrics=load('analysis/seed_metrics.pkl');windows=load('analysis/window_metrics.pkl')
 training=load('analysis/training_diagnostics.pkl');ensembles=load('analysis/ensemble_audits.pkl');native=load('portfolio_analysis/native_reconciliation.pkl')
 dev_bindings=load('inputs/development_parent_bindings.pkl');dev_parent=load('inputs/development_parent_source_audit.pkl')
 outer_bindings=load('inputs/outer_parent_bindings.pkl');outer_parent=load('inputs/outer_parent_source_audit.pkl');reproductions=load('inputs/parent_reproduction_summary.pkl')
 parent_comparison=load('analysis/paired_parent_comparison.pkl');sparsity=load('analysis/execution_sparsity.pkl');fee_decomposition=load('analysis/saved_path_fee_decomposition.pkl')
 check('recorded_seal_plan',sha(base/'inputs/selection_plan.json')==s['plan_sha256'] and json.loads((base/'inputs/selection_seal.json').read_text())==seal)
check('source_objects_match',source==s['source_audit'] and source_hashes==s['source_sha256'])
for name,frame,key in [('raw',raw,'raw_frame_sha256'),('prices',prices,'prices_frame_sha256'),('forecasts',forecasts,'forecasts_frame_sha256')]:check(name+'_frozen_frame_hash',frame_sha256(frame)==source[key])
check('source_asof',str(raw.index.max().date())==str(prices.index.max().date())==plan['source']['as_of'])
# A bounded native data readback, not a forecast/model reconstruction or fit.
observed=load_qlib_feature_frame(QlibProviderConfig(provider_uri=root/plan['source']['provider_uri']),instruments=['SOXL','SOXX'],fields=['$open','$high','$low','$close','$volume'],start=plan['source']['history_start'],end=plan['source']['as_of']).sort_index()
check('native_data_bytes_unchanged',price_frame_sha256(observed)==source['native_frame_sha256'])
calendar=pd.DatetimeIndex(D.calendar(freq='day'));check('native_calendar_count_and_bounds',len(calendar)==4162 and str(calendar[0].date())==plan['source']['history_start'] and str(calendar[-1].date())==plan['source']['as_of'])
check('native_universe',sorted(D.list_instruments(D.instruments('all'),as_list=True))==['SOXL','SOXX'])
dataset=build_anchor_dataset(forecasts,as_of=plan['source']['as_of']);mapping=variant_mapping(plan);chosen=s['selected']['variant'];folds=dev_folds|outer_folds
records={(name,year):rec for name,years in dev.items() for year,rec in years.items()}|{(name,year):rec for name,years in outer.items() for year,rec in years.items()}
check('annual_fold_inventory',len(folds)==25 and set(folds)==set(records))
check('all_fit_metadata_index_inventory',len(training)==75 and len(ensembles)==25 and all((base/x['recorder_artifact']).is_file() for x in training+ensembles))
check('indexes_compact_and_new_optimizer_units',all(not set(x)&{'epochs','history','trials','training_paths','candidate_training_paths','parent_training_paths'} for x in training+ensembles) and all(x['optimizer_diagnostics']['method'] in ('COBYLA','not_started') for x in training))
prepared={};fold_report=[]
for (name,year),fold in sorted(folds.items()):
 if year not in prepared:prepared[year]=prepare_training_rows(dataset,year,min_rows=plan['training']['minimum_rows'])
 rows,maturity=prepared[year];record=records[(name,year)];scaler=record['scaler'];prefix=f'fold:{name}:{year}'
 check(prefix+':native_maturity_rows',rows.equals(fold['rows']) and maturity==fold['audit'] and (rows.return_available_at<pd.Timestamp(scaler['cutoff'])).all() and (rows.feature_available_at<pd.Timestamp(scaler['cutoff'])).all())
 check(prefix+':bundle_scaler_identity',record['variant']==name and record['parent_market_variant']==mapping[name] and record['model_year']==year and record['scaler']==fold['scaler'] and record['scaler_sha256']==scaler['scaler_sha256'] and scaler['variant']==mapping[name] and scaler['model_year']==year)
 transformed=transform_for_variant(rows,scaler,variant=name,plan=plan)
 check(prefix+':exact_arrays',np.array_equal(transformed['features'],fold['features']) and np.array_equal(rows[['soxl_return','soxx_return']].to_numpy(float),fold['returns']) and record['training_feature_sha256']==ah(fold['features']) and record['training_return_sha256']==ah(fold['returns']))
 x=rows[list(PRICE_FEATURES)].to_numpy(float);constant=np.ptp(x,axis=0)==0;mean=x.mean(0);scale=x.std(0);mean[constant]=x[0,constant];scale[constant]=1
 check(prefix+':prior_scaler_statistics',np.array_equal(mean,scaler['price_mean']) and np.array_equal(scale,scaler['price_scale']))
 check(prefix+':three_seed_and_parent_inventory',set(record['seed_models'])==set(record['parent_seed_models'])=={17,41,71})
 fold_report.append({'variant':name,'year':year,'rows':len(rows),'cutoff':maturity['cutoff'],'return_available_at_max':maturity['return_available_at_max'],'invalid_training':record['invalid_training'],'numerical_failure':record['numerical_failure']})
ready=forecast_audit[forecast_audit.available]
check('frozen_forecast_maturity',len(forecast_models)==14 and (pd.to_datetime(ready.label_available_at_max,utc=True)<pd.to_datetime(ready.decision_at,utc=True)).all())
if chosen=='price_vol_holdings':
 f=outer_frames['frozen2022_parameters'];p=forecast_models[2023].predict(f[list(COMMON_PRIMARY_FEATURES)])
 check('frozen_2023_head_only',np.array_equal(p[:,1],f.future_vol_prediction.to_numpy()) and f.forecast_available.eq(1).all() and pd.to_datetime(f.forecast_model_available_at,utc=True).eq(pd.Timestamp('2023-01-01',tz='America/New_York').tz_convert('UTC')).all())
# Parent source/reproduction authorizations, without replaying parent fit histories.
check('development_cached_parent_inventory',len(dev_bindings)==dev_parent['development_parent_count']==63 and all(x['parent_recorder_id']==plan['parent_model_source']['recorder_id'] for x in dev_bindings.values()))
check('outer_certificate_matches_sealed_selection',outer_parent['scheduled_variant']==chosen and outer_parent['scheduled_years']==plan['training']['diagnostic_signal_years'] and seal['selection_uses_diagnostic_signal_years'] is False)
check('outer_binding_inventory',set(outer_bindings)=={(chosen,y,z) for y in plan['training']['diagnostic_signal_years'] for z in (17,41,71)})
expected_reproductions=0 if mapping[chosen]=='price_vol' else 12
check('only_never_materialized_parent_families_reproduced',len(reproductions)==expected_reproductions and outer_parent['genuinely_missing_count']==expected_reproductions and outer_parent['cached_parent_count']==12-expected_reproductions)
for (name,year,seed),binding in outer_bindings.items():
 check(f'outerparent:{year}:{seed}:cutoff',pd.Timestamp(binding['training_cutoff'])==pd.Timestamp(f'{year}-01-01',tz='America/New_York') and binding['scaler_sha256']==folds[(name,year)]['scaler']['scaler_sha256'])
 if expected_reproductions:
  check(f'outerparent:{year}:{seed}:new_provenance',binding['origin']=='new_reproduction_of_never_materialized_outer_parent' and binding['recorder_id']==ref['recorder_id'] and binding['spec']==asdict(parent_spec(plan)))
 else:check(f'outerparent:{year}:{seed}:cached_provenance',binding['parent_recorder_id']==plan['parent_model_source']['recorder_id'])
seal_mtime=(base/'inputs/selection_seal.json').stat().st_mtime_ns
check('outer_artifact_completion_after_seal',all((base/item['recorder_artifact']).stat().st_mtime_ns>=seal_mtime for item in training if item['year']>=2023))
# Direct identities on saved actions/current holdings. No alternative OOS book.
assets=['soxl','soxx','cash'];groups=paths.groupby(['segment','variant','seed','cost_bps'],sort=True);check('all38_saved_books',len(groups)==38)
path_reports=[];max_errors={};books={}
for (segment,name,seed,cost),saved in groups:
 p=validate_replay(saved.reset_index(drop=True));dates=pd.DatetimeIndex(p.signal_date);key=(segment,name,str(seed),float(cost));books[key]=p
 benchmark=name=='SOXX_buy_hold';parent_control=name.endswith('_embedded_parent_control');mode=None
 if segment=='development':
  frame=forecasts.loc[dates];family=name.removesuffix('_embedded_parent_control');annual_models=dev.get(family)
 else:
  mode='frozen2022_parameters' if name.endswith('frozen2022_parameters') else 'annual_refit';frame=outer_frames[mode].loc[dates];family=chosen;annual_models=outer[chosen]
 r=np.c_[frame[['soxl_return','soxx_return']].to_numpy(float),np.zeros(len(frame))];g=1+p.strategy_return.to_numpy(float);fee=p.fees.to_numpy(float);turn=p.turnover.to_numpy(float)
 prefix='book:'+':'.join(map(str,key));check(prefix+':date_pairing',p[['signal_date','trade_date','exit_date']].reset_index(drop=True).equals(frame[['signal_date','trade_date','exit_date']].reset_index(drop=True)) and p.exit_date.max()<=pd.Timestamp(plan['source']['as_of']))
 if benchmark:
  expected=1+r[:,1];expected[0]/=1+cost/10000;check(prefix+':analytic_SOXX_entry',np.max(abs(g-expected))<1e-10 and np.max(abs(fee[1:]))<1e-10);continue
 h=p[[f'pretrade_{a}_weight' for a in assets]].to_numpy(float);z=p[[f'{a}_weight' for a in assets]].to_numpy(float);ending=p[[f'ending_{a}_weight' for a in assets]].to_numpy(float)
 gross=(z*(1+r)).sum(1);actual_turn=np.abs((1-fee[:,None])*z[:,:2]-h[:,:2]).sum(1)
 errors={'owned_state':float(np.max(abs(h-np.vstack(([0.,0.,1.],ending[:-1]))))),
  'simplex':float(np.max(abs(z.sum(1)-1))),'fee_L1':float(np.max(abs(fee-cost/10000*actual_turn))),
  'turnover':float(np.max(abs(turn-actual_turn))),'net_multiplier':float(np.max(abs(g-(1-fee)*gross))),
  'normalized_ending':float(np.max(abs(ending-z*(1+r)/gross[:,None]))),
  'compounded_equity':float(np.max(abs(p.equity.to_numpy()/np.cumprod(g)-1))),
  'source_faithful_logged_decision':0.}
 flags=['observed_forecast_fallback','invalid_training_fallback','numerical_failure','execute_open_target','exact_hold_support']
 check(prefix+':complete_boolean_flags',p[flags].notna().all().all() and all(p[c].map(lambda v:isinstance(v,(bool,np.bool_))).all() for c in flags))
 for year in sorted(dates.year.unique()):
  mask=dates.year==year;positions=np.flatnonzero(mask);use=2023 if mode=='frozen2022_parameters' else int(year);record=annual_models[use];rows=frame.loc[mask]
  transformed=transform_for_variant(rows,record['scaler'],variant=family,plan=plan);invalid=transformed['invalid_forecast_mask']
  payloads=record['parent_ensemble_models'] if parent_control else record['ensemble_models'] if str(seed)=='ensemble' else [record['seed_models'][int(seed)]]
  prepared_models=[_numpy_market(policy_from_dict(payload),transformed['features']) for payload in payloads]
  expected_targets=[];expected_q=[];expected_hold=[]
  for local,pos in enumerate(positions):
   q,target,hold,_=_numpy_decision([x[0] for x in prepared_models],[x[1] for x in prepared_models],local,h[pos].copy())
   if invalid[local]:q,target,hold=np.array([0.,1.,0.,0.]),np.array([0.,1.,0.]),False
   expected_targets.append(target);expected_q.append(q);expected_hold.append(hold)
  expected_targets=np.array(expected_targets);expected_q=np.array(expected_q);expected_hold=np.array(expected_hold)
  errors['source_faithful_logged_decision']=max(errors['source_faithful_logged_decision'],float(np.max(abs(z[mask]-expected_targets))))
  qcols=[f'expert_{a}_weight' for a in ('soxl','soxx','cash','hold')]
  check(prefix+f':{year}:q_and_HOLD',np.max(abs(p.loc[mask,qcols].to_numpy(float)-expected_q))<1e-12 and np.array_equal(p.loc[mask,'exact_hold_support'].to_numpy(),expected_hold) and np.array_equal(p.loc[mask,'execute_open_target'].to_numpy(),~expected_hold))
  check(prefix+f':{year}:causal_model_flags',p.loc[mask,'model_year'].eq(use).all() and pd.Timestamp(record['scaler']['cutoff'])<=pd.Timestamp(rows.index.min()).tz_localize('America/New_York') and np.array_equal(p.loc[mask,'observed_forecast_fallback'].to_numpy(bool),invalid) and p.loc[mask,'invalid_training_fallback'].eq(record['invalid_training']).all() and p.loc[mask,'numerical_failure'].eq(False if parent_control else record['numerical_failure']).all())
 hold=p.exact_hold_support.to_numpy(bool);check(prefix+':no_HOLD_order',not hold.any() or (np.array_equal(z[hold],h[hold]) and np.all(fee[hold]==0) and np.all(turn[hold]==0)))
 check(prefix+':accounting',max(errors.values())<=1e-10 and np.isfinite(z).all() and z.min()>=-1e-12,errors)
 for k,v in errors.items():max_errors[k]=max(max_errors.get(k,0),v)
 path_reports.append({'segment':segment,'variant':name,'seed':str(seed),'cost_bps':cost,'sessions':len(p),'exact_HOLD_days':int(hold.sum()),'maximum_error':max(errors.values())})
# Stored metrics, gate composition and development-only family ranking.
for row in comparison.itertuples():
 book=books[(row.segment,row.variant,'ensemble',float(row.cost_bps))];benchmark=books[(row.segment,'SOXX_buy_hold','benchmark',float(row.cost_bps))]
 actual=paired_metrics(book,benchmark)
 check(f'metric:{row.variant}:{row.cost_bps}',all(abs(float(actual[k])-float(getattr(row,k)))<1e-9 for k in ('return_pct','relative_wealth','log_excess_sum','max_drawdown_pct','cvar5_pct')))
check('18seed_cost_records',len(seed_metrics)==18)
for row in seed_metrics.itertuples():
 actual=paired_metrics(books[('development',row.variant,str(int(row.seed)),float(row.cost_bps))],books[('development','SOXX_buy_hold','benchmark',float(row.cost_bps))])
 check(f'seedmetrics:{row.variant}:{row.seed}:{row.cost_bps}',all(abs(float(actual[k])-float(getattr(row,k)))<1e-9 for k in ('return_pct','relative_wealth','log_excess_sum','max_drawdown_pct','cvar5_pct')))
for row in annual.itertuples():
 book=books[(row.segment,row.variant,'ensemble',float(row.cost_bps))];benchmark=books[(row.segment,'SOXX_buy_hold','benchmark',float(row.cost_bps))];mask=book.signal_date.dt.year.eq(int(row.year))
 actual=paired_metrics(book.loc[mask],benchmark.loc[mask]);check(f'annualmetrics:{row.variant}:{row.year}:{row.cost_bps}',abs(float(actual['log_excess_sum'])-float(row.log_excess_sum))<1e-10)

for name,part in seed_metrics.groupby('variant'):
 check('three_seeds_both_costs:'+name,set(zip(part.seed,part.cost_bps))=={(z,c) for z in (17,41,71) for c in (10,25)})
 both=int(part.pivot(index='seed',columns='cost_bps',values='log_excess_sum').gt(0).all(1).sum());g=gates[gates.variant.eq(name)]
 check('same_seed_gate:'+name,g.seeds_positive_at_both_costs.eq(both).all() and g.minimum_positive_seeds_at_both_costs.eq(both>=2).all())
 boolcols=[c for c in g if g[c].dtype==bool and c not in ('all_pass','seed_evidence_complete')];check('gate_all_composition:'+name,g[boolcols].all(1).equals(g.all_pass))
 for cost in (10,25):
  p=books[('development',name,'ensemble',float(cost))];bad=p[['observed_forecast_fallback','invalid_training_fallback','numerical_failure']].any().any()
  check(f'failure_disqualification:{name}:{cost}',g.loc[g.cost_bps.eq(cost),'zero_invalid_source_or_numerical_events'].eq(not bad).all())
check('family26_bootstrap',len(boot)==12 and boot.declared_variant_family_size.eq(26).all() and boot.resamples.eq(2000).all() and boot.random_seed.eq(20261002).all() and np.allclose(boot.alpha,.05/26,rtol=0,atol=1e-15))
for row in gates.itertuples():
 a=annual[(annual.segment=='development')&(annual.variant==row.variant)&(annual.cost_bps==row.cost_bps)];m=comparison[(comparison.segment=='development')&(comparison.variant==row.variant)&(comparison.cost_bps==row.cost_bps)].iloc[0];excess=a.log_excess_sum.to_numpy()
 expected={'aggregate_positive_net_log_excess':m.log_excess_sum>0,'minimum_positive_signal_years':int((excess>0).sum())>=5,'median_annual_log_excess_positive':np.median(excess)>0,'leave_any_one_year_out_log_excess_positive':np.min(excess.sum()-excess)>0,'anchored_max_drawdown_not_worse_than_soxx':m.max_drawdown_pct>=m.benchmark_max_drawdown_pct-1e-10,'cvar5_not_worse_than_soxx':m.cvar5_pct>=m.benchmark_cvar5_pct-1e-10,'above_numerical_null':m.log_excess_sum>1e-8}
 check(f'raw_gate_arithmetic:{row.variant}:{row.cost_bps}',len(excess)==7 and all(bool(getattr(row,k))==bool(v) for k,v in expected.items()))
for row in boot.itertuples():
 g=gates[(gates.variant==row.variant)&(gates.cost_bps==row.cost_bps)].iloc[0]
 check(f'bootstrap_metadata:{row.variant}:{row.cost_bps}:{row.mean_block}',row.mean_block in (20,60) and abs(row.annualized_lower_bound-252*row.daily_lower_bound)<1e-12 and bool(row.lower_bound_positive)==bool(row.daily_lower_bound>0)==bool(g[f'stationary_block_{row.mean_block}_lower_bound_positive']))

evidence={name:{'gates':gates[gates.variant.eq(name)],'bootstrap':boot[boot.variant.eq(name)],'metrics':comparison[comparison.variant.eq(name)&comparison.segment.eq('development')],'historical_qualification_pass':bool(gates[gates.variant.eq(name)].all_pass.all())} for name in mapping}
ranked,_=choose_family(evidence);check('sealed_development_ranking_reproduces',ranked==seal['selected']==s['selected'])
check('8registered_window_cases',len(windows)==8 and windows.sessions.eq(38).all() and pd.to_datetime(windows.start_signal).eq(pd.Timestamp('2026-07-31')).all() and pd.to_datetime(windows.last_exit).eq(pd.Timestamp('2026-09-25')).all())
# Fresh-window daily paths are not retained. No new OOS path is generated here.
for cost,result in native.items():
 a=result['audit'];recon=result['reconciliation'];p=books[('development',chosen,'ensemble',float(cost))]
 check('native:'+str(cost),a['status']=='passed' and a['sessions']==len(p) and a['tolerance']==1e-7 and np.isfinite(a['maximum_session_return_error']) and a['maximum_session_return_error']<=1e-7 and abs(float(recon.return_error.abs().max())-a['maximum_session_return_error'])<1e-15 and np.array_equal(recon.research_return.to_numpy(),p.strategy_return.to_numpy()) and a['risk_annualization_sessions']==252)
nb=json.loads((card/'feasible_parent_sparse_search_20261003.ipynb').read_text());texts=[''.join(c.get('source',[])) for c in nb['cells'] if c['cell_type']=='markdown'];pending=not texts or texts[-1].strip()!=(card/'conclusion.md').read_text().strip()
report={'scope':'Bounded source/fold/parentauthorization/storedportfolio/gates/native audit. No model fitting, alternate policy/OOS replay, or all-trial market replay. Serialized decisions reconstructed only against their logged current holdings. Optimizer trial/selectedtrainingbooks handled by separate model audit.',
 'recorder_id':ref['recorder_id'],'preflight_recorder_id':pre['recorder_id'],'checks':checks,'failures':[k for k,v in checks.items() if not v],'details':details,'verified_loaded_artifacts':verified,'source_sha256':s['source_sha256'],'plan_sha256':s['plan_sha256'],
 'folds':fold_report,'portfolio_books':len(groups),'portfolio_book_audits':path_reports,'maximum_logged_accounting_errors':max_errors,'qualification':s['historical_qualification_pass'],'selected':s['selected'],'parent_reproduction_count':len(reproductions),
 'native_return_errors':{str(c):v['audit']['maximum_session_return_error'] for c,v in native.items()},'window_scope':'All8reportedcase dates/counts checked; no fresh-CASH window paths independently regenerated because only their metrics were retained.',
 'notebook_final_markdown_pending':pending,'notebook_sha256':sha(card/'feasible_parent_sparse_search_20261003.ipynb'),'conclusion_sha256':sha(card/'conclusion.md')}
(out/'market-source-recorder-audit.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps({k:report[k] for k in ['recorder_id','preflight_recorder_id','portfolio_books','maximum_logged_accounting_errors','qualification','selected','parent_reproduction_count','native_return_errors','notebook_final_markdown_pending','failures']},indent=2))
if report['failures']:raise SystemExit(1)

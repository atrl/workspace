"""Finished-Recorder economic interpretation only; no fit, target or path generation."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import qlib
from qlib.workflow import R
ROOT=Path('/Users/rziiiii/dev/workspace-worktrees/feasible-parent-sparse-search-20261003')
CARD=ROOT/'research/feasible_parent_sparse_search_20261003'
reference=json.loads((CARD/'market_reference.json').read_text())
if reference.get('complete') is not True:raise SystemExit('Market reference incomplete; no economic conclusions computed.')
plan=json.loads((CARD/'selection_plan.json').read_text())
qlib.init(provider_uri=str(ROOT/plan['source']['provider_uri']),region='us',kernels=1)
identities={}
with R.uri_context(reference['recorder_uri']):
 rec=R.get_recorder(experiment_name=plan['experiment'],recorder_id=reference['recorder_id'])
 assert rec.client.get_run(rec.id).info.status=='FINISHED'
 def get(key):
  entry=reference['artifacts'][key];path=Path(rec.get_local_dir())/'artifacts'/entry['artifact']
  assert path.stat().st_size==entry['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
  value=rec.load_object(entry['artifact']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
  identities[key]=entry;return value
 summary=get('summary')
 names=['comparison','annual','gates','bootstrap','seed_metrics','attribution','window_metrics','paired_parent_comparison',
        'execution_sparsity','saved_path_fee_decomposition','seed_path_equality','training_diagnostics','ensemble_audits','portfolio_replays']
 objects={name:get('analysis/'+name+'.pkl') for name in names}
def rows(frame):return json.loads(frame.to_json(orient='records',date_format='iso',double_precision=15))
paths=objects['portfolio_replays'];events=[];annual_magnitudes=[]
for (segment,variant,cost),part in paths.loc[paths.seed.eq('ensemble')].groupby(['segment','variant','cost_bps']):
 part=part.sort_values('signal_date').reset_index(drop=True)
 ref=paths.loc[paths.segment.eq(segment)&paths.seed.eq('benchmark')&paths.cost_bps.eq(cost)].sort_values('signal_date').reset_index(drop=True)
 assert part[['signal_date','trade_date','exit_date']].equals(ref[['signal_date','trade_date','exit_date']])
 excess=np.log1p(part.strategy_return.to_numpy())-np.log1p(ref.strategy_return.to_numpy())
 totals=float(excess.sum());first=float(excess[0]);best=int(np.argmax(excess));worst=int(np.argmin(excess))
 identity={'segment':segment,'variant':variant,'cost_bps':int(cost)}
 events.append({**identity,'sessions':len(part),'first_trade':str(part.trade_date.iloc[0].date()),'last_exit':str(part.exit_date.iloc[-1].date()),
  'total_net_log_excess':totals,'first_observation_log_excess':first,'remaining_saved_observations_log_excess':float(excess[1:].sum()),
  'largest_positive_observation_log_excess':float(excess[best]),'largest_positive_observation_trade_date':str(part.trade_date.iloc[best].date()),
  'largest_negative_observation_log_excess':float(excess[worst]),'largest_negative_observation_trade_date':str(part.trade_date.iloc[worst].date()),
  'exactly_equal_benchmark_return_days':int((part.strategy_return.to_numpy()==ref.strategy_return.to_numpy()).sum()),
  'max_absolute_daily_net_log_difference':float(abs(excess).max()),
  'mean_soxl':float(part.soxl_weight.mean()),'mean_soxx':float(part.soxx_weight.mean()),'mean_cash':float(part.cash_weight.mean()),
  'exact_soxx_target_days':int(((part.soxl_weight==0)&(part.soxx_weight==1)&(part.cash_weight==0)).sum()),
  'scope':'Additive attribution on the saved book, not a restarted or counterfactual strategy.'})
 for year,index in part.groupby(part.signal_date.dt.year).groups.items():
  index=np.array(list(index));a=excess[index]
  annual_magnitudes.append({**identity,'signal_year':int(year),'net_log_excess':float(a.sum()),
   'first_model_year_observation_log_excess':float(a[0]),'remaining_saved_year_log_excess':float(a[1:].sum()),
   'largest_absolute_daily_log_excess':float(abs(a).max()),
   'absolute_total_below_existing_aggregate_numerical_guard_1e_minus8':bool(abs(a.sum())<1e-8),
   'annotation_scope':'Magnitude-only use of existing aggregate1e-8 numerical guard; no annual gate was changed.'})
training=pd.DataFrame(objects['training_diagnostics']);ensembles=pd.DataFrame(objects['ensemble_audits'])
training_summary=[]
for variant,part in training.groupby('variant'):
 development=part[part.year.le(2022)]
 training_summary.append({'variant':variant,'all_fits':len(part),'development_fits':len(development),
  'development_selected_origins':development.selected_origin.value_counts().to_dict(),
  'development_F_delta_min':float(development.F_delta_from_parent.min()),'development_F_delta_max':float(development.F_delta_from_parent.max()),
  'development_F_delta_median':float(development.F_delta_from_parent.median()),
  'development_failures':int(development.failure.notna().sum()),
  'development_unique_two_cost_evaluations':sum(a['unique_full_path_evaluations'] for a in development.optimizer_diagnostics)})
report={'scope':'Read-only interpretation of completed market evidence; previously reviewed history, not fresh forward.',
 'recorder_id':reference['recorder_id'],'market_reference_sha256':hashlib.sha256((CARD/'market_reference.json').read_bytes()).hexdigest(),
 'source_artifacts':identities,'official_selected':summary['selected'],'official_historical_qualification':summary['historical_qualification_pass'],
 'new_forward_qualified':summary['new_forward_qualified'],'objective_achieved':summary['objective_achieved'],'parent_work':summary['parent_work'],
 'comparison':rows(objects['comparison']),'annual':rows(objects['annual']),'official_gates':rows(objects['gates']),
 'bootstrap':rows(objects['bootstrap']),'seed_metrics':rows(objects['seed_metrics']),
 'attribution':rows(objects['attribution']),'window_metrics':rows(objects['window_metrics']),
 'paired_parent_comparison':rows(objects['paired_parent_comparison']),'execution_sparsity':rows(objects['execution_sparsity']),
 'gross_fee_decomposition':rows(objects['saved_path_fee_decomposition']),
 'seed_path_equality':rows(objects['seed_path_equality']),'training_summary':training_summary,
 'ensemble_summary':rows(ensembles[[c for c in ['variant','year','selected_origin','candidate_ensemble_F','parent_ensemble_F','candidate_minus_parent_F','failure','candidate_component_origins'] if c in ensembles]]),
 'saved_book_contribution_diagnostics':events,'annual_magnitude_annotations':annual_magnitudes}
Path('/tmp/phase9_economic_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=str))
for key in ['comparison','official_gates','paired_parent_comparison','training_summary','window_metrics','saved_book_contribution_diagnostics']:
 print(key);print(pd.DataFrame(report[key]).to_string(index=False))

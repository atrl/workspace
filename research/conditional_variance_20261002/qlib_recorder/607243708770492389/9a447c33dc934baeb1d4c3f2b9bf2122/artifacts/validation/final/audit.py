from pathlib import Path
import hashlib,json,re,os
import numpy as np
import pandas as pd
os.environ['MLFLOW_ALLOW_FILE_STORE']='true'
os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
import qlib
from qlib.workflow import R
from src.data.conditional_variance import load_qualified_primary
from src.data.integrated_allocation import frame_sha256
from src.workflow.conditional_variance import verify_model_repetitions,verify_forecast_repetitions,_prediction_view,_calibration_identity
from src.workflow.convex_excess import verify_repetitions
root=Path.cwd();out=Path('/tmp/conditional-variance-validation');card=root/'research/conditional_variance_20261002'
summary=json.loads((card/'summary.json').read_text());plan=json.loads((card/'selection_plan.json').read_text())
seal=json.loads((card/'selection_seal.json').read_text());correction=json.loads((card/'implementation_correction.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_hashes={p:sha(root/p) for p in summary['source_sha256']}
checks={
 'current_source_hashes_match_recorded':source_hashes==summary['source_sha256'],
 'plan_hash_matches':sha(card/'selection_plan.json')==summary['plan_sha256']==seal['plan_sha256'],
 'pre_HAC_seal_unchanged':correction['development_seal']==seal,
 'no_financial_or_plan_change_in_HAC_correction':correction['market_gate_change'] is False and correction['plan_changed'] is False,
 'objective_active':summary['campaign_status']=='active_research' and summary['objective_achieved'] is False,
 'no_promotion_or_orders':summary['historical_qualification_pass'] is False and summary['new_forward_qualified'] is False and summary['production_change'] is False and summary['real_broker_orders'] is False,
}
bundle=load_qualified_primary(root,provider_uri=plan['source']['provider_uri'],parent_summary='research/'+plan['source']['experiment']+'/summary.json',expected_recorder_id=plan['source']['recorder_id'],as_of=plan['source']['as_of'],train_start=plan['source']['model_signal_start'])
checks['fresh_native_provider_and_parent_audit_match']=bundle['audit']==summary['source_audit']
with R.uri_context(summary['recorder_uri']):
 recorder=R.get_recorder(experiment_name=summary['experiment'],recorder_id=summary['recorder_id'])
 load=recorder.load_object
 profile=load('analysis/profile_audits.pkl');repro=load('analysis/reproduction.pkl');daily=load('analysis/daily_solver_audits.pkl')
 forecasts=load('inputs/forecast_frames.pkl');models=load('models/annual_forecasters.pkl');cals=load('inputs/annual_calibrations.pkl')
 paths=load('analysis/portfolio_replays.pkl');inputs=load('policy/development_inputs.pkl');outer_inputs=load('policy/diagnostic_inputs.pkl')
 outer_forecasts=load('policy/diagnostic_forecasts.pkl');native=load('portfolio_analysis/native_reconciliation.pkl');gates=load('analysis/gates.pkl')
 primary=load('inputs/primary_frame.pkl');prices=load('inputs/execution_prices.pkl');audit=load('inputs/source_audit.pkl')
 comparisons=load('analysis/comparison.pkl');controls=load('analysis/controls.pkl')
checks['recorder_raw_inputs_match_native_provider']=frame_sha256(primary)==frame_sha256(bundle['primary_frame']) and frame_sha256(prices)==frame_sha256(bundle['execution_prices'])
checks['recorder_source_audit_matches_summary']=audit==summary['source_audit']
checks['all_36_profiles_verified_zero_fallback']=len(profile)==36 and bool(profile.all_optimality_verified.all()) and bool(profile[['solver_fallback_sessions','estimator_fallback_sessions']].eq(0).all().all())
checks['all_predicted_risk_ratios_within_budget']=bool(profile.maximum_variance_budget_ratio.le(1+1e-8).all())
checks['all_32_recorded_reproduction_checks_match']=len(repro)==32 and bool(repro.all_values_match.all()) and bool(repro.maximum_absolute_difference.le(1e-8).all())
checks['no_repeated_runs_counted_as_independent']=not bool(repro.independent_statistical_evidence.any())
independent_repro=[]
for repeat in (2,3):
 independent_repro.append(verify_model_repetitions(models[1],models[repeat]))
 independent_repro.append(verify_forecast_repetitions(_prediction_view(forecasts[1]),_prediction_view(forecasts[repeat])))
 for mode in ('annual_refit','frozen2022_calibration'):
  independent_repro.append(verify_forecast_repetitions(_prediction_view(outer_forecasts[(1,mode)]),_prediction_view(outer_forecasts[(repeat,mode)])))
calibration_equal=[]
for segment in ('development','reused_diagnostic'):
 reference={r['year']:r['calibration'] for r in cals if r['repeat_id']==1 and r['segment']==segment}
 for repeat in (2,3):
  current={r['year']:r['calibration'] for r in cals if r['repeat_id']==repeat and r['segment']==segment}
  calibration_equal.append(_calibration_identity(reference)==_calibration_identity(current))
checks['all_annual_calibrations_reproduced']=all(calibration_equal)
for (segment,variant,cost),group in paths.loc[paths.repeat_id.gt(0)].groupby(['segment','variant','cost_bps']):
 reference=group.loc[group.repeat_id.eq(1)].drop(columns='repeat_id').reset_index(drop=True)
 columns=reference.columns[~reference.isna().all()]
 for repeat in (2,3):
  current=group.loc[group.repeat_id.eq(repeat)].drop(columns='repeat_id').reset_index(drop=True)
  independent_repro.append(verify_repetitions(reference[columns],current[columns]))
checks['independent_model_forecast_and_path_recheck']=all(r['all_values_match'] for r in independent_repro)
checks['three_distinct_model_objects_per_year']=all(len({id(models[r][year]) for r in (1,2,3)})==3 for year in models[1])
checks['strict_calibration_cutoffs']=all(pd.Timestamp(row['calibration']['audit']['label_available_at_max'])<pd.Timestamp(row['calibration']['cutoff']) and pd.Timestamp(row['calibration']['audit']['h5_label_available_at_max'])<pd.Timestamp(row['calibration']['cutoff']) for row in cals)
fit=pd.read_csv(card/'forecast_training_audit.csv');ready=fit[fit.available]
checks['strict_OOF_training_cutoffs']=bool((pd.to_datetime(ready.label_available_at_max,utc=True)<pd.to_datetime(ready.decision_at,utc=True)).all())
checks['all_frozen_inputs_use_2023_parameters']=all(np.array_equal(outer_inputs[(r,'frozen2022_calibration')]['calibration_year'],np.full(len(outer_inputs[(r,'frozen2022_calibration')]['calibration_year']),2023)) for r in (1,2,3))
all_daily=[row for group in daily.values() for row in group]
checks['all_daily_solver_decisions_verified']=all(row['optimality_verified'] is True and row['status']=='solved' and not row['solver']['verification_failure'] for row in all_daily)
checks['all_daily_solver_external_positive_budget']=all(row['solver']['budget_mode']=='external_positive' and row['solver']['risk_budget_variance']>0 and row['solver']['robust_radius']==0 for row in all_daily)
checks['all_eight_candidate_cost_gates_failed']=len(gates)==8 and not bool(gates.all_pass.any())
checks['native_reports_match_summary']=all(value['audit']==summary['native_qlib_audit'][str(cost)] for cost,value in native.items())
checks['native_both_costs_pass_unchanged_tolerance']=set(native)=={10,25} and all(value['audit']['status']=='passed' and value['audit']['tolerance']==1e-7 and np.isfinite(value['audit']['maximum_session_return_error']) and value['audit']['maximum_session_return_error']<=1e-7 for value in native.values())
for file,recorded in [('backtest_benchmark_comparison.csv',comparisons),('control_comparison.csv',controls),('development_gates.csv',gates),('risk_solver_audit.csv',profile),('reproduction_audit.csv',repro)]:
 pd.testing.assert_frame_equal(pd.read_csv(card/file),recorded,check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
checks['key_CSV_sidecars_equal_recorder']=True
nb=json.loads((card/'conditional_variance_20261002.ipynb').read_text());code=[c for c in nb['cells'] if c['cell_type']=='code'];outputs=[o for c in code for o in c.get('outputs',[])]
text='\n'.join(''.join(o.get('text',o.get('data',{}).get('text/plain',[]))) for o in outputs)
checks['notebook_code_all_executed_without_errors']=all(c.get('execution_count') is not None for c in code) and not any(o['output_type']=='error' for o in outputs)
checks['notebook_output_identifies_final_recorder']=summary['recorder_id'] in text
checks['notebook_selection_before_diagnostic_outputs']=text.index('Variance development sealed:')<text.index('Variance diagnostic repeat=')
final_markdown=''.join(nb['cells'][-1].get('source',[]))
checks['final_notebook_conclusion_current']='待执行' not in final_markdown and summary['recorder_id'] in final_markdown
report={'recorder_id':summary['recorder_id'],'checks':checks,'failed_checks':[key for key,value in checks.items() if not value],
 'source_sha256':source_hashes,'test_file_sha256':{p:sha(root/p) for p in ['tests/test_conditional_variance.py','tests/test_conditional_variance_workflow.py','tests/test_convex_allocation.py']},
 'source_scope':list(source_hashes),'plan_sha256':summary['plan_sha256'],'notebook_sha256_at_audit':sha(card/'conditional_variance_20261002.ipynb'),
 'profile_count':len(profile),'daily_solver_decisions':len(all_daily),'reproduction_recorded_checks':len(repro),'independent_reproduction_checks':len(independent_repro),
 'reproduction_maximum_absolute_difference':max(r['maximum_absolute_difference'] for r in independent_repro),
 'native_maximum_return_errors':{str(c):v['audit']['maximum_session_return_error'] for c,v in native.items()},
 'selected':summary['selected'],'objective_achieved':False,'campaign_status':'active_research','no_market_refitting_in_audit':True}
(out/'artifact-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['checks','source_sha256','source_scope','test_file_sha256']},indent=2))

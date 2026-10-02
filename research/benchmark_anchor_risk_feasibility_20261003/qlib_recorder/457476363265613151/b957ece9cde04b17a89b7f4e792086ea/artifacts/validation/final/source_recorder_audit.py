"""Read-only provenance/inventory audit; no policy inference or checkpoint risk evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from qlib.workflow import R
from src.workflow.anchor_checkpoint_feasibility import load_registered_inputs,expected_identities
root=Path.cwd();out=Path('/tmp/anchor-feasibility-validation');card=root/'research/benchmark_anchor_risk_feasibility_20261003'
summary=json.loads((card/'summary.json').read_text());plan=json.loads((card/'selection_plan.json').read_text());seal=json.loads((card/'initial_seal.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();checks={}
def check(name,value):checks[name]=bool(value)
check('plan_bytes_match_seal_and_summary',sha(card/'selection_plan.json')==summary['plan_sha256']==seal['registered_plan_sha256'])
check('new_sources_match_final_summary',all(sha(root/p)==value for p,value in summary['source_sha256'].items()))
check('parent_sources_unchanged',all(sha(root/p)==value for p,value in summary['parent_source_sha256'].items()))
# Source deserialization and metadata/native Dataset checks only; this function
# has no evaluate_policy or optimizer path and does not read OOS artifacts.
bundle=load_registered_inputs(root,plan)
check('exact_registered_parent_and_hashes',bundle['parent_summary']['recorder_id']==plan['source']['recorder_id']==summary['parent_recorder_id'] and bundle['parent_source_hashes']==summary['parent_source_sha256'])
with R.uri_context(summary['recorder_uri']):
 recorder=R.get_recorder(experiment_name=summary['experiment'],recorder_id=summary['recorder_id']);load=recorder.load_object
 index=load('inputs/checkpoint_index.pkl');maturity=load('inputs/maturity.pkl');hashes=load('inputs/source_hashes.pkl');parent_hashes=load('inputs/parent_source_hashes.pkl')
 metrics=load('analysis/checkpoint_cost_metrics.pkl');joint=load('analysis/checkpoint_joint_feasibility.pkl');null_metrics=load('analysis/null_metrics.pkl');null_joint=load('analysis/null_joint_feasibility.pkl')
 runs=load('analysis/run_summaries.pkl');reproduction=load('analysis/utility_reproduction.pkl')
 artifact_root=Path(recorder.get_local_dir())/'artifacts';source_attestations=json.loads((artifact_root/'inputs/source_artifact_manifest.json').read_text())
 recorded_plan=json.loads((artifact_root/'inputs/selection_plan.json').read_text())
 check('new_Recorder_FINISHED',recorder.client.get_run(recorder.id).info.status=='FINISHED')
 check('recorded_plan_bytes_match',sha(artifact_root/'inputs/selection_plan.json')==summary['plan_sha256'] and recorded_plan==plan)
check('recorded_source_hashes_match',hashes==summary['source_sha256'] and parent_hashes==summary['parent_source_sha256'])
check('all66_source_attestations_reverified',len(source_attestations)==66 and source_attestations==bundle['source_artifacts'])
allowed={plan['source']['parent_registration_artifact'],plan['source']['source_hash_artifact'],plan['source']['input_fold_artifact']}
identities=expected_identities(plan)
allowed|={plan['source']['checkpoint_artifact_template'].format(variant=v,year=y,seed=s) for v,y,s in identities}
check('only_registered_development_source_objects',len(allowed)==66 and {r['artifact'] for r in source_attestations}==allowed)
pd.testing.assert_frame_equal(maturity,bundle['maturity']);check('all21_frozen_maturity_rows_rebuilt',len(maturity)==21)
check('maturity_strictly_before_cutoff',bool((pd.to_datetime(maturity.return_available_at_max,utc=True)<pd.to_datetime(maturity.cutoff,utc=True)).all()))
check('index_contains_only_diagnostic_identifiers_and_hashes',set(index)=={'variant','model_year','seed','epoch','tensor_sha256','training_feature_sha256','training_return_sha256'})
wanted={(v,y,s,e) for v,y,s in identities for e in plan['inventory']['saved_epochs']}
actual=set(index[['variant','model_year','seed','epoch']].itertuples(index=False,name=None));check('all7623_index_identities_exact',len(index)==len(wanted)==7623 and actual==wanted and not index.duplicated(['variant','model_year','seed','epoch']).any())
index_matches=True
for row in index.itertuples():
 parent=bundle['training_audits'][(row.variant,row.model_year,row.seed)];state=parent['epochs'][row.epoch]
 index_matches &= row.tensor_sha256==state['checkpoint']['tensor_sha256'] and row.training_feature_sha256==parent['training_feature_sha256'] and row.training_return_sha256==parent['training_return_sha256']
check('index_hashes_match_immutable_parent_snapshots',index_matches)
check('trajectory_counts_exclude_null',len(metrics)==15246 and len(joint)==7623 and not metrics.explicit_null.any() and not joint.explicit_null.any())
check('null_records_separate_and_complete',len(null_metrics)==126 and len(null_joint)==63 and null_metrics.explicit_null.all() and null_joint.explicit_null.all() and null_joint.epoch.eq('exact_SOXX_null').all())
check('joint_index_hashes_match',joint.set_index(['variant','model_year','seed','epoch']).checkpoint_tensor_sha256.sort_index().equals(index.set_index(['variant','model_year','seed','epoch']).tensor_sha256.sort_index().rename('checkpoint_tensor_sha256')))
check('exact_per_cost_trajectory_inventory',len(metrics[['variant','model_year','seed','epoch','cost_bps']].drop_duplicates())==15246 and set(metrics.cost_bps)=={10.,25.} and metrics.groupby(['variant','model_year','seed','epoch']).size().eq(2).all())
check('run_summary_and_nulls_exact63identities',len(runs)==63 and set(runs[['variant','model_year','seed']].itertuples(index=False,name=None))==identities and set(null_joint[['variant','model_year','seed']].itertuples(index=False,name=None))==identities)
check('only_development_models_in_outputs',set(joint.model_year)==set(range(2016,2023)) and set(metrics.model_year)==set(range(2016,2023)))
check('all63_immutable_chunks_written',all((artifact_root/'completed_chunks'/str(v)/str(y)/str(s)/'result.pkl').is_file() for v,y,s in identities))
pd.testing.assert_frame_equal(pd.read_csv(card/'maturity_summary.csv'),maturity,check_dtype=False)
csv_runs=pd.read_csv(card/'run_summary.csv');check('run_summary_sidecar_rows',len(csv_runs)==len(runs) and set(csv_runs.columns)==set(runs.columns))
counts=pd.read_csv(card/'feasibility_counts.csv');check('summary_counts_match_small_sidecar',counts.to_dict('records')==summary['counts'])
check('no_fit_OOS_or_strategy_change_claim',summary['market_model_fits'] is False and summary['out_of_sample_evaluation'] is False and summary['new_strategy_selected'] is False and summary['original_phase5_selection_unchanged'] is True and summary['objective_achieved'] is False and summary['campaign_status']=='active_research' and summary['real_broker_orders'] is False and summary['existing_sim_change'] is False)
nbpath=card/'benchmark_anchor_risk_feasibility_20261003.ipynb';nb=json.loads(nbpath.read_text());code=[c for c in nb['cells'] if c['cell_type']=='code'];outputs=[o for c in code for o in c.get('outputs',[])];text='\n'.join(''.join(o.get('text',o.get('data',{}).get('text/plain',[]))) for o in outputs)
check('notebook_executed_without_error',all(c.get('execution_count') is not None for c in code) and not any(o['output_type']=='error' for o in outputs))
check('final_recorder_visible_in_notebook',summary['recorder_id'] in text)
markdown=''.join(nb['cells'][-1].get('source',[]));conclusion=(card/'conclusion.md').read_text();pending=markdown.strip()!=conclusion.strip() or summary['recorder_id'] not in markdown
report={'recorder_id':summary['recorder_id'],'parent_recorder_id':summary['parent_recorder_id'],'scope':'Source/manifest/native-row/scaler/index/output-provenance audit only; no model fitting, policy inference, checkpoint feasibility recomputation or OOS comparison.',
 'checks':checks,'failures':[k for k,v in checks.items() if not v],'source_artifact_count':len(source_attestations),'folds':len(maturity),'runs':len(runs),'trajectory_index_rows':len(index),'trajectory_cost_rows':len(metrics),'trajectory_joint_rows':len(joint),'null_cost_rows':len(null_metrics),'null_joint_rows':len(null_joint),'utility_reproduction_rows':len(reproduction),
 'counts':summary['counts'],'plan_sha256':summary['plan_sha256'],'source_sha256':summary['source_sha256'],'parent_source_sha256':summary['parent_source_sha256'],'parent_artifact_manifest_sha256':summary['parent_artifact_manifest_sha256'],'source_asof':summary['source_as_of'],'notebook_final_markdown_pending':pending,'notebook_sha256':sha(nbpath),'metric_audit':'Separate model-agent report pending; first/best/count/risk/utility arithmetic is outside this source audit.'}
(out/'source-recorder-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('checks','source_sha256','parent_source_sha256')},indent=2))

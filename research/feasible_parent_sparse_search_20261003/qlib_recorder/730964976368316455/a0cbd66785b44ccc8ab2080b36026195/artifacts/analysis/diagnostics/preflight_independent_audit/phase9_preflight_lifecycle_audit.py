"""Read-only identity/lifecycle audit; no fitting, rollout or optimizer call."""
from pathlib import Path
from dataclasses import asdict
import json,hashlib
import qlib
from qlib.workflow import R
from src.workflow import feasible_parent_sparse_search as wf
ROOT=Path('/Users/rziiiii/dev/workspace-worktrees/feasible-parent-sparse-search-20261003')
CARD=ROOT/'research/feasible_parent_sparse_search_20261003'
plan=json.loads((CARD/'selection_plan.json').read_text());hashes=wf._source_hashes(ROOT)
seal=wf._implementation_seal(ROOT,CARD,plan,hashes)
reference=json.loads((CARD/'preflight_reference.json').read_text())
qlib.init(provider_uri=str(ROOT/plan['source']['provider_uri']),region='us',kernels=1)
objects=wf._load_reference(reference,wf.sha(CARD/'selection_plan.json'),hashes)
with R.uri_context(reference['recorder_uri']):
 recorder=R.get_recorder(experiment_name=wf.EXPERIMENT,recorder_id=reference['recorder_id'])
 info=recorder.client.get_run(recorder.id)
 assert info.info.status=='FINISHED'
 base=Path(recorder.get_local_dir())/'artifacts'
parents={s:objects[f'preflight/parents/{s}/revalidation.pkl'] for s in wf.SEEDS}
searches={s:objects[f'preflight/search/{s}/audit.pkl'] for s in wf.SEEDS}
ensemble=objects['preflight/ensemble_audit.pkl']
table,gate=wf._incremental_gate(parents,searches,ensemble,plan)
assert all(objects['gate'][k]==v for k,v in gate.items())
assert gate['passed'] and reference['passed']
assert objects['gate']['parent_spec']==asdict(wf.parent_spec(plan))
assert objects['gate']['search_spec']==asdict(wf.search_spec(plan))
counts=[]
for seed in wf.SEEDS:
 old=objects[f'preflight/parents/{seed}/audit.pkl'];new=searches[seed]
 assert [e['epoch'] for e in old['epochs']]==list(range(121))
 assert old['failure'] is None and new['failure'] is None
 assert new['selected_origin']=='search'
 stats=new['optimizer_diagnostics']
 assert len(new['trials'])==stats['unique_full_path_evaluations']<=2000
 assert new['frozen_encoder_sha256_before']==new['frozen_encoder_sha256_after']
 assert old['training_feature_sha256']==new['training_feature_sha256']
 assert old['training_return_sha256']==new['training_return_sha256']
 counts.append({'seed':seed,'parent_states':len(old['epochs']),'parent_spec_updates':old['spec']['updates'],
  'parent_retained_epoch':old['selected_epoch'],'parent_F':new['parent_F'],'selected_origin':new['selected_origin'],
  'search_F':new['F'],'F_delta':new['F_delta_from_parent'],'unique_evaluations':stats['unique_full_path_evaluations'],
  'requested_callbacks':stats['requested_callbacks'],'cache_hits':stats['cache_hits'],
  'solver_nfev':stats['solver_result']['nfev'],'termination':stats['termination']})
summary=json.loads((CARD/'summary.json').read_text())
assert summary['decision']=='preflight_passed_market_not_started'
assert summary['market_model_fits_started'] is False and summary['market_model_fits']==0 and summary['market_backtests']==0
assert not (CARD/'market_reference.json').exists()
assert not (CARD/'selection_seal.json').exists()
assert summary['objective_achieved'] is False and summary['selected'] is None
assert hashes==wf._source_hashes(ROOT)
report={'scope':'Read-only preflight Recorder/source/reference/lifecycle audit; no independent financial-path recomputation',
 'recorder_id':reference['recorder_id'],'recorder_status':info.info.status,'plan_sha256':wf.sha(CARD/'selection_plan.json'),
 'implementation_seal_sha256':seal['file_sha256'],'reference_sha256':wf.sha(CARD/'preflight_reference.json'),
 'source_count':len(hashes),'bound_artifact_count':len(reference['artifacts']),
 'parent_and_search_options_bound':True,'all_parent_121_states_retained':True,'no_hidden_encoder_updates':True,
 'gate_recomputed_from_saved_seed_and_actual_shared_ensemble_fields':gate,'seed_counts':counts,
 'market_reference_absent':True,'market_fits':0,'market_backtests':0,'selected':None,'objective_achieved':False,
 'source_hashes_unchanged':True,'financial_trial_reconstruction':'Delegated independent audit still required before market flag.',
 'source_sha256':hashes}
Path('/tmp/phase9_preflight_lifecycle_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

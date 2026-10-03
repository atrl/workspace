"""Read-only final static/registration/test audit, without registered campaign execution."""
from pathlib import Path
import json,hashlib,re,subprocess
import pandas as pd
from src.workflow.feasible_parent_sparse_search import _source_hashes,_incremental_gate
root=Path.cwd();out=Path('/tmp/feasible-parent-validation');card=root/'research/feasible_parent_sparse_search_20261003';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((card/'selection_plan.json').read_text());receipt=json.loads((card/'registration_clarification.json').read_text());stage=receipt['followup_clarifications'][0]
original=json.loads(receipt['prior_plan_utf8']);middle=json.loads(stage['prior_plan_utf8'])
assert hashlib.sha256(receipt['prior_plan_utf8'].encode()).hexdigest()==receipt['prior_plan_sha256']=='0db1a5d0fcde0b6cfc3801e1ecd5ea718ecbe139fe2f34d50046227238cfd093'
assert hashlib.sha256(stage['prior_plan_utf8'].encode()).hexdigest()==stage['prior_plan_sha256']=='713b85da7aac9517907676fc75d15bd468a08f4379d181be7aeedbcccb189dfe'
assert sha(card/'selection_plan.json')==stage['current_plan_sha256']==receipt['current_plan_sha256']=='c7818045478368cd472b4f37c54a19ffcad014478e36d558f4af31af6e5f40b7'
def diff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):return sum([diff(a.get(k),b.get(k),path+'/'+k) for k in sorted(a.keys()|b.keys())],[])
 return [] if a==b else [path]
change1=diff(original,middle);change2=diff(middle,plan)
assert set(change1)=={'/artifact_policy/recorder_only_artifacts','/artifact_policy/root_policy','/pre_probe_clarifications','/synthetic_known_information/ensemble_gate'}
assert set(change2)=={'/pre_probe_clarifications','/registered_before_market_model_fits_or_portfolio_results'}
assert receipt['registered_experiment_fits_started'] is False and stage['registered_experiment_fits_started'] is False
assert original['registered_at_utc']==middle['registered_at_utc']==plan['registered_at_utc']
assert pd.Timestamp(plan['registered_at_utc'])<pd.Timestamp(receipt['at_utc'])<pd.Timestamp(stage['at_utc'])
assert plan['registered_before_market_model_fits_or_portfolio_results'] is True
assert not (card/'preflight_reference.json').exists() and not (card/'market_reference.json').exists()
parents={s:{'healthy':True,'feasible':True,'F':.1} for s in (17,41,71)}
search={s:{'failure':None,'qualification_eligible':True,'training_feasibility_passed':True,'F':.2,'selected_origin':'search','optimizer_diagnostics':{'unique_full_path_evaluations':5}} for s in parents}
ensemble={'failure':None,'parent_ensemble':{'feasible':True,'F':-.2},'candidate_ensemble':{'feasible':True,'F':-.1},'selected_origin':'null'}
assert _incremental_gate(parents,search,ensemble,plan)[1]['passed'] is False
parents[17]['healthy']='false'
try:_incremental_gate(parents,search,ensemble,plan)
except ValueError:pass
else:raise AssertionError('string Boolean allowed through incremental gate')
workflow=(root/'src/workflow/feasible_parent_sparse_search.py').read_text()
assert "'artifact_policy':plan['artifact_policy']" in workflow and '本轮是有限预算的COBYLA' in workflow
before=json.loads((out/'pretest-source-state.json').read_text());source=_source_hashes(root);tests={n:sha(root/n) for n in before['test_sha256']}
assert source==before['source_sha256'] and tests==before['test_sha256'] and sha(card/'selection_plan.json')==before['plan_sha256']
assert sha(card/'registration_clarification.json')==before['registration_receipt_sha256']
static={'scope':'Final source/registration/gate review and existing regression logs only; no Phase9 registered parent reproduction, incremental search or market campaign.',
 'static_go_for_registered_preflight_only':True,'registered_incremental_gate_not_yet_run':True,'market_fit_authorization_from_this_audit':False,
 'registration_chain':[receipt['prior_plan_sha256'],stage['prior_plan_sha256'],stage['current_plan_sha256']],
 'registration_at_utc':plan['registered_at_utc'],'clarification_times':[receipt['at_utc'],stage['at_utc']],
 'exact_preserved_prior_bytes_verified':True,'first_change_paths':change1,'second_change_paths':change2,
 'financial_solver_data_seed_fixture_and_budget_fields_unchanged':True,
 'negative_ensemble_SOXX_fallback_cannot_pass':True,'non_boolean_health_rejected':True,
 'reviewed_workflow_contracts':['partial references block automatic restart','fullparent evidence saved before later failure','scheduled never-materialized outer parents require persisted developmentseal before workers','outer/native work outside activeR; errors stamp same marketRFAILED','whole searched ensemble increment checked before fallback','newCOBYLA trial budgets and originalparentAdam work reported separately','registered Recorder artifact namespaces used'],
 'source_sha256':source,'test_sha256':tests,'plan_sha256':before['plan_sha256'],'registration_receipt_sha256':before['registration_receipt_sha256'],
 'failures':[]}
(out/'static-review.json').write_text(json.dumps(static,indent=2)+'\n')
def parse(p):
 text=p.read_text();lines=[l.strip().strip('= ') for l in text.splitlines() if re.search(r'\d+ (?:failed|passed).* in [\d.]+s',l)];summary=lines[-1]
 counts={k:int(v) for v,k in re.findall(r'(\d+) (failed|passed|skipped|errors?|warnings)',summary)}
 ids=sorted(set(re.findall(r'^FAILED\s+(\S+)',text,re.M)));assert len(ids)==counts.get('failed',0)
 return {'log':str(p),'sha256':sha(p),'summary':summary,**counts,'failed_test_ids':ids}
baseline=json.loads((out/'baseline-phase8.json').read_text());current_status=subprocess.check_output(['git','status','--short'],text=True).splitlines()
report={'worktree':str(root),'baseline':{'path':str(out/'baseline-phase8.json'),'full_passed':1182,'full_failed':139,'required_passed':8,'required_failed':61},
 'source_sha256':source,'test_sha256':tests,'source_count':len(source),'plan_sha256':before['plan_sha256'],
 'source_hashes_unchanged_during_validation':True,'test_hashes_unchanged_during_validation':True,'plan_hash_unchanged_during_validation':True,'registration_receipt_hash_unchanged':True,
 'environment':{'MLFLOW_ALLOW_FILE_STORE':'true','MLFLOW_DISABLE_AGENT_HINT':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':'.'},
 'deltas':{},'formal_recorder_audit':None,'registered_phase9_campaign_executed_by_validation_agent':False,'market_fits_by_validation_agent':False,
 'legacy_regression_scope':'The mandated fullsuite includes the pre-existing Phase7 test_registered_optimizer_recognizes_fixed_synthetic_known_information (504rows,3original120-update unit fits). Those disposable unit results were not retained/reused as Phase9 parents, not compared to choose outcomes, and no Phase9 registered preflight/search ran. Both Phase9 campaign-reference files were absent at this audit.',
 'git_status_short_before':before['git_status_short'],'git_status_short_after':current_status,
 'git_status_entries_added_during_tests':sorted(set(current_status)-set(before['git_status_short'])),
 'git_status_entries_removed_during_tests':sorted(set(before['git_status_short'])-set(current_status)),
 'git_diff_check':subprocess.run(['git','diff','--check'],capture_output=True).returncode==0,
 'static_review':{'path':str(out/'static-review.json'),'sha256':sha(out/'static-review.json'),'failures':[]},
 'known_information_gate_status':'not_yet_run; passing unit/math/data checks do not confer incremental optimizer or financial qualification'}
for kind in ('full','required'):
 value=parse(out/(kind+'.log'));report[kind]=value;old=set(baseline[kind]['failed_test_ids']);now=set(value['failed_test_ids'])
 report['deltas'][kind]={'added_failed_test_ids':sorted(now-old),'removed_failed_test_ids':sorted(old-now),'same_failed_test_ids':old==now,'passed_delta':value.get('passed',0)-baseline[kind].get('passed',0)}
 assert old==now,(kind,report['deltas'][kind])
focused=Path('/tmp/feasible-parent-combined-focused.log');report['focused']=parse(focused);assert report['focused']['passed']==97
report['old_phase8_sources_unchanged']=all(source[name]==digest for name,digest in baseline['source_sha256'].items());assert report['old_phase8_sources_unchanged']
for name,path in {'data_readback':Path('/tmp/feasible-parent-cache-readback-final.json'),'independent_math':Path('/tmp/feasible-parent-independent-math.json')}.items():report[name]={'path':str(path),'sha256':sha(path)}
(out/'final-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'static_go':True,'full':report['full']['summary'],'required':report['required']['summary'],'focused':report['focused']['summary'],'deltas':report['deltas'],'source_count':len(source),'final_verification_sha256':sha(out/'final-verification.json'),'static_review_sha256':sha(out/'static-review.json')},indent=2))

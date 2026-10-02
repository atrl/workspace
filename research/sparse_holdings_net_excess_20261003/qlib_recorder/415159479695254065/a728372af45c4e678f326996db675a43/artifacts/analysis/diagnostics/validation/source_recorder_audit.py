"""Final Phase8 Recorder/card identity audit; no fit, replay, test, or data reconstruction."""
from pathlib import Path
import hashlib,json,re
import pandas as pd
from qlib.workflow import R
from src.workflow.sparse_holdings_net_excess import _require_initial_seal
root=Path.cwd();card=root/'research/sparse_holdings_net_excess_20261003';out=Path('/tmp/sparse-holdings-validation')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=lambda data:hashlib.sha256(data).hexdigest()
s=json.loads((card/'summary.json').read_text());reference=json.loads((card/'preflight_probe_reference.json').read_text());verification=json.loads((out/'final-verification.json').read_text());seal=json.loads((card/'initial_seal.json').read_text())
assert s['recorder_id']==reference['recorder_id']=='a728372af45c4e678f326996db675a43'
assert s['plan_sha256']==reference['plan_sha256']==seal['plan_sha256']==verification['plan_sha256']==sha(card/'selection_plan.json')
assert s['source_sha256']==seal['source_sha256']==verification['source_sha256'] and seal['test_sha256']==verification['test_sha256']
assert all(sha(root/n)==d for n,d in s['source_sha256'].items())
assert all(sha(root/n)==d for n,d in seal['test_sha256'].items())
assert s['market_model_fits']==s['market_backtests']==0 and s['market_model_fits_started'] is False
assert s['selected'] is None and s['historical_qualification_pass'] is False and s['new_forward_qualified'] is False and s['objective_achieved'] is False
assert s['mathematical_and_data_contracts_passed'] is True and s['registered_known_information_gate_passed'] is False
assert seal['pre_fit_contracts_passed'] is False and seal['synthetic_known_information_all_seeds_and_ensemble_passed'] is False
try:_require_initial_seal(root,card,s['plan_sha256'],s['source_sha256'])
except ValueError as error:blocked_message=str(error)
else:raise AssertionError('failed actual seal permitted market fitting')
import qlib
qlib.init(provider_uri=str(root/s['source']['provider_uri']),region='us',kernels=1)
with R.uri_context(s['recorder_uri']):
 recorder=R.get_recorder(experiment_name=s['experiment'],recorder_id=s['recorder_id'])
 assert recorder.client.get_run(recorder.id).info.status=='FINISHED'
 base=Path(recorder.get_local_dir())/'artifacts';assert sha(base/'inputs/selection_plan.json')==s['plan_sha256']
 original={}
 for name,item in reference['artifacts'].items():
  p=base/item['artifact'];assert sha(p)==item['sha256'];original[name]=p.read_bytes()
 assert digest(original['original_payload.pkl'])==s['probe_sha256']=='3d589805851fc3c4a959fd00e53f9f5fd8409adef0d9c8da111c095b9c3f36d5'
 receipt=json.loads(original['identity_receipt.json'])
 assert digest(original['initial_model.py'])==s['preflight']['initial_failed_probe_source_sha256']==receipt['initial_failed_probe_source_sha256']
 assert digest(original['retention_model.py'])==s['preflight']['retention_reproduction_source_sha256']==receipt['retention_reproduction_source_sha256']==s['source_sha256']['src/model/sparse_holdings_net_excess.py']
 assert original['initial_model.py'].replace(b'reported_trade_count_gt_1e12',b'reported_trade_count_gt_1e_minus12')==original['retention_model.py']
 assert recorder.load_object('inputs/observer_source_hashes.pkl')==s['source_sha256']
 native=recorder.load_object('inputs/native_frozen_source_audit.pkl');assert native==s['source_audit']
 details=recorder.load_object('inputs/native_frozen_source_details.pkl');canonical=json.dumps(details,sort_keys=True,separators=(',',':'),allow_nan=False).encode();assert digest(canonical)==native['details_sha256']
 recorded=recorder.load_object('analysis/preflight_audit.pkl');assert recorded==s['preflight']
 table=recorder.load_object('analysis/synthetic_seed_summary.pkl')
 pd.testing.assert_frame_equal(table,pd.read_csv(card/'preflight_seed_summary.csv'),check_dtype=False)
 assert table.seed.tolist()==[17,41,71] and table.saved_states.eq(121).all() and table.healthy.all()
 assert table.retained_F.eq(0).all() and table.feasible_positive_F_states.eq(0).all() and not table.registered_positive_control_pass.any()
 assert not (base/'models/development.pkl').exists() and not (base/'analysis/portfolio_replays.pkl').exists()
 assert not (base/'portfolio_analysis/native_reconciliation.pkl').exists()
assert (native['rows'],native['calendar_sessions'],native['forecast_model_count'],native['mapped_development_folds'])==(3956,4162,14,21)
assert native['forecast_refit_performed'] is False and native['allocation_models_loaded'] is False and details['holdings_columns_stored_in_dataset'] is False
nbpath=card/'sparse_holdings_net_excess_20261003.ipynb';nb=json.loads(nbpath.read_text());con=(card/'conclusion.md').read_text();assert ''.join(nb['cells'][-1]['source']).strip()==con.strip()
code=[c for c in nb['cells'] if c['cell_type']=='code'];assert all(c['execution_count'] is not None for c in code)
assert not any(o.get('output_type')=='error' for c in code for o in c.get('outputs',[]))
notebook_code='\n'.join(''.join(c['source']) for c in code);assert 'record_preflight_diagnostic' in notebook_code and 'run_experiment(' not in notebook_code
assert s['recorder_id'] in con and '没有进行市场拟合或市场回测' in con and '139' in con and '1,182' in con and '77项' in con
assert '26ed0c13' in con and '3c690d11' in con and '365,904' in con and '尚未实现或运行' in con
links=[]
for link in re.findall(r'\]\(([^)]+)\)',con):
 assert (card/link).exists(),link;links.append(link)
selected=json.loads((card/'selected_model.json').read_text());assert selected['selected'] is None
independent=json.loads((out/'known-information-independent-audit.json').read_text());assert independent['known_information_feasible_training_gate_passed'] is False and not independent['failures']
files=[p for p in card.iterdir() if p.is_file() and p.name not in ('verification.json','artifact_manifest.json')]
report={'scope':'Final diagnostic Recorder/card identity and conclusion consistency only; no market fits, replays, repeated data reconstruction or retests. Earlier363-state independent synthetic audit is linked separately.',
 'recorder_id':s['recorder_id'],'recorder_status':'FINISHED','completion_scope':'Evidence archival complete; registered positive-control gate FAILED.',
 'cache_reference_recorder_id':reference['recorder_id'],'one_recorder_suffices_for_cached_rerun':True,
 'plan_sha256':s['plan_sha256'],'observer_source_count':len(s['source_sha256']),'observer_sources_match_frozen_verification':True,'sealed_test_count':len(seal['test_sha256']),
 'native_source_counts':{'rows':3956,'calendar_sessions':4162,'forecast_heads':14,'mapped_folds':21},
 'original_preflight_artifacts':reference['artifacts'],'original_source_change':'Diagnostic-key spelling only; no original full-trajectory equality claim.',
 'toy_seed_count':3,'toy_saved_states':363,'registered_positive_control_pass':False,'mathematical_and_data_contracts_passed':True,
 'market_model_fits':0,'market_backtests':0,'market_gate_rejected_actual_failed_seal':blocked_message,
 'selected':None,'objective_achieved':False,'historical_qualification_assessed':False,
 'notebook_final_markdown_pending':False,'notebook_conclusion_equal':True,'executed_code_cells':len(code),'notebook_error_outputs':0,
 'notebook_sha256':sha(nbpath),'conclusion_sha256':sha(card/'conclusion.md'),'card_sha256':{p.name:sha(p) for p in sorted(files)},
 'card_hash_scope':'Before archivist adds evidence pointers/manifest and verification copy; notebook/conclusion are final frozen bytes.',
 'independent_known_information_audit':{'path':str(out/'known-information-independent-audit.json'),'sha256':sha(out/'known-information-independent-audit.json')},'failures':[]}
(out/'source-recorder-audit.json').write_text(json.dumps(report,indent=2)+'\n')
verification['formal_recorder_audit']={'recorder_id':s['recorder_id'],'report':str(out/'source-recorder-audit.json'),'report_sha256':sha(out/'source-recorder-audit.json'),'failures':[],
 'notebook_final_markdown_pending':False,'notebook_sha256':report['notebook_sha256'],'conclusion_sha256':report['conclusion_sha256'],
 'market_model_fits':0,'market_backtests':0,'known_information_gate_passed':False,'single_recorder_cache_identity':True,'source_count':32}
verification['final_card_refresh']={'scope':report['scope'],'notebook_conclusion_equal':True,'notebook_error_outputs':0,'card_sha256':report['card_sha256'],'notebook_final_markdown_pending':False,'historical_qualification_assessed':False}
(out/'final-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('recorder_id','notebook_sha256','conclusion_sha256','toy_saved_states','market_model_fits','market_backtests','registered_positive_control_pass','failures')},indent=2))
print('source_report_sha256',sha(out/'source-recorder-audit.json'));print('final_verification_sha256',sha(out/'final-verification.json'))

"""Archive already-produced Phase8 diagnostic evidence; never run a model."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
import qlib
from qlib.workflow import R
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts

ROOT=Path('/Users/rziiiii/dev/workspace-worktrees/sparse-holdings-net-excess-20261003')
CARD=ROOT/'research/sparse_holdings_net_excess_20261003'
EXPERIMENT='sparse_holdings_net_excess_20261003'
RECORDER='a728372af45c4e678f326996db675a43'
EXPERIMENT_ID='415159479695254065'
URI=Path('/Users/rziiiii/dev/workspace/research')/EXPERIMENT/'qlib_recorder'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((CARD/'summary.json').read_text())
assert summary['recorder_id']==RECORDER and summary['market_model_fits']==0 and summary['market_backtests']==0
assert summary['registered_known_information_gate_passed'] is False
assert (CARD/'verification.json').is_file(),'wait for independent final verification/card ready'
verification=json.loads((CARD/'verification.json').read_text())
assert verification['known_information_gate']['market_fits_permitted'] is False
assert verification['new_code_regression_failures']==0
assert all(sha(ROOT/name)==value for name,value in summary['source_sha256'].items())

items=[]
def add(path,group,purpose):
 p=Path(path);assert p.is_file(),p
 items.append({'path':p,'artifact':f'analysis/diagnostics/{group}/{p.name}','purpose':purpose,
               'source_path':str(p),'source_mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),
               'sha256':sha(p),'bytes':p.stat().st_size})
for p in sorted(Path('/tmp/sparse-holdings-validation').iterdir()):
 if p.is_file():add(p,'validation','Independent technical/data/gradient/registered-toy and regression verification; no market fit')
for p in ['/tmp/sparse_holdings_core_integration.py','/tmp/sparse_holdings_gradient_audit.py','/tmp/audit_sparse_known_failure.py']:
 add(p,'validation_methods','Executable method for the archived independent synthetic verification')
for stem in ['implicit_settlement_gradient_audit','post_fee_state_gradient_audit']:
 for suffix in ['py','json','log','md']:add(f'/tmp/{stem}.{suffix}',stem,'Synthetic fee/state derivative method, full numerical audit and original log; no financial strategy result')
for p in ['/tmp/sparse-holdings-runtime.json','/tmp/sparse-holdings-runtime.log','/tmp/profile_sparse_holdings.py']:
 add(p,'synthetic_runtime','Original synthetic implementation timing and method, retaining its original model source identity')
for p in ['/tmp/sparse_holdings_known_information_probe.py','/tmp/preserve_sparse_holdings_failed_probe.py',
          '/tmp/sparse-holdings-known-information-probe.log','/tmp/sparse-holdings-known-information-retention.log',
          '/tmp/sparse-holdings-known-information-trajectories.csv']:
 add(p,'registered_toy_methods','Original positive-control and retention methods/logs/summary; original full payload remains in preflight/original_payload.pkl')
for p in ['/tmp/sparse-holdings-failed-optimization-diagnosis.json','/tmp/diagnose_sparse_holdings_failed_probe.py',
          '/tmp/sparse-holdings-failed-optimization-diagnosis.log']:
 add(p,'failed_optimizer_diagnosis','Analysis of already-retained synthetic states; no new target, fit or optimizer trajectory')
add('/tmp/sparse-holdings-notebook-execution.log','execution','Execution log of diagnostic-only notebook; zero market fits/backtests')
add('/tmp/sparse_holdings_data_readback.json','source_readback','Frozen native Qlib source and21mapped development-fold identity readback; no market fitting')
add(__file__,'archival','Method used to archive existing evidence and build an exact card/Recorder manifest')

qlib.init(provider_uri=str(ROOT/summary['provider_uri']),region='us',kernels=1)
with R.uri_context(str(URI)):
 record=R.get_recorder(experiment_name=EXPERIMENT,recorder_id=RECORDER)
 assert record.client.get_run(record.id).info.status=='FINISHED'
 recdir=Path(record.get_local_dir());assert recdir==URI/EXPERIMENT_ID/RECORDER
 auto={'artifacts/code_diff.txt','artifacts/code_cached.txt','artifacts/code_status.txt'}
 science_before={str(p.relative_to(recdir)):sha(p) for p in recdir.rglob('*') if p.is_file() and str(p.relative_to(recdir)) not in auto}
 for item in items:
  record.log_artifact(str(item['path']),artifact_path=str(Path(item['artifact']).parent))
  assert sha(recdir/'artifacts'/item['artifact'])==item['sha256']
 # Card snapshot is only taken after the independent validator declares it final.
 root_files=sorted(p for p in CARD.iterdir() if p.is_file() and p.name!='artifact_manifest.json')
 for p in root_files:
  record.log_artifact(str(p),artifact_path='card')
  assert (recdir/'artifacts/card'/p.name).read_bytes()==p.read_bytes()
 _remove_qlib_code_snapshot_artifacts(recdir)
 assert all((recdir/name).is_file() and sha(recdir/name)==digest for name,digest in science_before.items())
 assert record.client.get_run(record.id).info.status=='FINISHED'

# Copy exactly the single diagnostic run and its experiment metadata; no parent trees.
local=CARD/'qlib_recorder';(local/EXPERIMENT_ID).mkdir(parents=True,exist_ok=True)
shutil.copy2(URI/EXPERIMENT_ID/'meta.yaml',local/EXPERIMENT_ID/'meta.yaml')
shutil.copytree(recdir,local/EXPERIMENT_ID/RECORDER,dirs_exist_ok=True)
local_rec=local/EXPERIMENT_ID/RECORDER
canonical={str(p.relative_to(recdir)):(p.stat().st_size,sha(p)) for p in recdir.rglob('*') if p.is_file()}
copied={str(p.relative_to(local_rec)):(p.stat().st_size,sha(p)) for p in local_rec.rglob('*') if p.is_file()}
assert canonical==copied
assert sorted(p.name for p in (local/EXPERIMENT_ID).iterdir() if p.is_dir())==[RECORDER]

lookup={item['artifact']:item for item in items}
entries=[]
for p in sorted(CARD.rglob('*')):
 if not p.is_file() or p.name=='artifact_manifest.json':continue
 rel=p.relative_to(CARD).as_posix()
 if '__pycache__' in p.parts or p.name=='.DS_Store':raise AssertionError(f'transient in card:{rel}')
 artifact=rel.split('/artifacts/',1)[1] if '/artifacts/' in rel else None
 if artifact in lookup:
  item=lookup[artifact];purpose=item['purpose'];source={'archived_from':item['source_path'],'original_sha256':item['sha256'],'original_mtime_utc':item['source_mtime_utc']}
 elif artifact and artifact.startswith('card/'):
  purpose='Final independently reviewed slim-card snapshot';source={'card_file':artifact[5:],'recorder_id':RECORDER}
 elif artifact and artifact.startswith('preflight/'):
  purpose='Byte-preserved registered synthetic failed probe, original26ed summary/source and retention3c690 full payload/source; never market results';source={'recorder_id':RECORDER,'artifact':artifact,'identity_reference':'preflight_probe_reference.json'}
 elif artifact:
  purpose='Native frozen source identity or saved synthetic preflight analysis';source={'recorder_id':RECORDER,'artifact':artifact,'parent_input_recorder':'52f40778cbe64ca6a5864da356b960d0'}
 elif rel.startswith('qlib_recorder/'):
  purpose='Native Qlib/MLflow experiment/run identity, parameters and diagnostic-only scope tags';source={'recorder_id':RECORDER,'canonical_recorder_uri':str(URI)}
 else:
  purpose='Slim card entry, registration, executed diagnostic notebook, gate/verification summary or explicitly unrun next-design note';source={'card_root':True,'experiment':EXPERIMENT}
 entries.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p),'recorder_id':RECORDER,
                 'role':'failed_registered_synthetic_preflight_diagnostic','purpose':purpose,'source':source,
                 'date_range':{'native_source_as_of':'2026-09-25','registered_toy_ordinal_rows':('0..503' if artifact and (artifact.startswith('preflight/') or 'known' in artifact or 'optimizer_diagnosis' in artifact or 'preflight_audit' in artifact or 'synthetic_seed' in artifact) else None),'other_synthetic_fixtures':'Artificial units/labels specified by each audit; not observed market performance','market_fit_or_backtest_dates':None}})
manifest={'schema_version':2,'experiment':EXPERIMENT,'artifact_policy':'slim_card_recorder_first',
 'authoritative_recorder':RECORDER,'record_type':'failed_registered_synthetic_preflight_diagnostic',
 'recorder_finished_means':'Diagnostic evidence archived; registered positive-control gate FAILED; no market fit/backtest',
 'retained_recorder_ids':[RECORDER],'canonical_recorder_uri':str(URI),
 'parent_input_recorder':'52f40778cbe64ca6a5864da356b960d0','qualified_price_recorder':'bd754e6362b241f194351a1e7aebff6f',
 'prior_phase7_recorder':'300530137db54b6c807c39ebb9e8c331',
 'parent_retention':'References only; no full parent Recorder trees copied.',
 'source_as_of':'2026-09-25','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'path_base':'card directory','self_reference_policy':'Root manifest excluded from itself and not copied into Recorder.',
 'entry_count':len(entries),'total_bytes':sum(e['bytes'] for e in entries),'maximum_file_bytes':max(e['bytes'] for e in entries),
 'entries':entries}
(CARD/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
assert all((CARD/e['path']).stat().st_size==e['bytes'] and sha(CARD/e['path'])==e['sha256'] for e in entries)
print(json.dumps({'recorder':RECORDER,'record_status':'FINISHED_diagnostic_only','entry_count':len(entries),
 'total_bytes':manifest['total_bytes'],'max_file_bytes':manifest['maximum_file_bytes'],
 'manifest_sha256':sha(CARD/'artifact_manifest.json'),'root_files':sorted(p.name for p in CARD.iterdir() if p.is_file()),
 'science_before_preserved':len(science_before),'copied_single_recorder':True,'market_fits':0,'market_backtests':0},indent=2))

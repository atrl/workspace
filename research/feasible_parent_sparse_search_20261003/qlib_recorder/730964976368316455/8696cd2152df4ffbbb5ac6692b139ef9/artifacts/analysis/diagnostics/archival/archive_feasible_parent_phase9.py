"""Archive existing Phase9 evidence only; never fit, optimize, or replay a policy."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE","true")
import qlib
from qlib.workflow import R
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts

ROOT=Path('/Users/rziiiii/dev/workspace-worktrees/feasible-parent-sparse-search-20261003')
EXPERIMENT='feasible_parent_sparse_search_20261003'
CARD=ROOT/'research'/EXPERIMENT
URI=Path('/Users/rziiiii/dev/workspace/research')/EXPERIMENT/'qlib_recorder'
EID='730964976368316455'
IDS={'market':'8696cd2152df4ffbbb5ac6692b139ef9','preflight':'a0cbd66785b44ccc8ab2080b36026195'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((CARD/'summary.json').read_text())
assert summary['recorder_id']==IDS['market'] and summary['market_model_fits']==75
assert summary['historical_qualification_pass'] is False and summary['objective_achieved'] is False
assert all(sha(ROOT/n)==h for n,h in summary['source_sha256'].items())
verification_path=Path('/tmp/feasible-parent-validation/final-verification.json')
v=json.loads(verification_path.read_text())
assert v['formal_recorder_audit']['failures']==[]
assert v['formal_recorder_audit']['notebook_final_markdown_pending'] is False,'Wait for final independent card hash refresh.'
assert (CARD/'verification.json').read_bytes()==verification_path.read_bytes(),'Final validator sidecar must be synchronized before archival.'
assert ''.join(json.loads((CARD/f'{EXPERIMENT}.ipynb').read_text())['cells'][-1]['source'])==(CARD/'conclusion.md').read_text()
refs={role:json.loads((CARD/f'{role}_reference.json').read_text()) for role in IDS}
refbytes={role:sha(CARD/f'{role}_reference.json') for role in IDS}
for role,ref in refs.items():
 assert ref['complete'] is True and ref['recorder_id']==IDS[role]
 assert ref['plan_sha256']==summary['plan_sha256']
 for artifact,binding in ref['artifacts'].items():
  p=URI/EID/IDS[role]/'artifacts'/binding['artifact']
  assert p.stat().st_size==binding['bytes'] and sha(p)==binding['sha256'],artifact
assert sha(CARD/'preflight_reference.json')==refs['market']['preflight_reference_sha256']

items=[]
def add(path,role,group,purpose):
 p=Path(path);assert p.is_file(),p
 artifact=f'analysis/diagnostics/{group}/{p.name}'
 assert not any(x['role']==role and x['artifact']==artifact for x in items)
 items.append({'path':p,'role':role,'artifact':artifact,'purpose':purpose,'source_path':str(p),
               'mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),
               'sha256':sha(p),'bytes':p.stat().st_size})
# Frozen verification, raw logs and methods. No transient caches or drafts.
for p in sorted(Path('/tmp/feasible-parent-validation').iterdir()):
 if p.is_file():add(p,'market','validation','Independent static/native/data/Recorder and baseline-comparison verification; unchanged raw output and method')
for p in ['/tmp/feasible-parent-combined-focused.log','/tmp/feasible-parent-cache-readback-final.json',
          '/tmp/feasible-parent-cache-readback-final.log','/tmp/feasible_parent_data_readback.py',
          '/tmp/feasible-parent-independent-math.json','/tmp/feasible-parent-independent-math.log',
          '/tmp/feasible_parent_independent_math.py']:
 add(p,'market','source_math_verification','Pre-fit source/cache/native or standalone mathematical verification with original methods/logs')
for p in ['/tmp/feasible-parent-market-optimizer-audit.json','/tmp/feasible-parent-market-optimizer-audit.log',
          '/tmp/feasible-parent-market-optimizer-audit-trials.csv','/tmp/feasible-parent-market-optimizer-audit-samples.csv',
          '/tmp/audit_feasible_parent_market_optimizer.py']:
 add(p,'market','optimizer_audit','Read-only independent verification of retained search trials, first-best selection, frozen encoder and exact accounting; not a new optimization')
for p in ['/tmp/phase9_economic_audit.ipynb','/tmp/phase9_economic_audit.json','/tmp/phase9_economic_audit.py',
          '/tmp/phase9_startup_attribution_code.txt','/tmp/phase9_economic_review.md']:
 add(p,'market','economic_review','Executed read-only interpretation and additive attribution of saved market/training books, no alternative policy or fit')
for p in ['/tmp/feasible-parent-preflight-independent-audit.json','/tmp/feasible-parent-preflight-independent-audit.log',
          '/tmp/feasible-parent-preflight-independent-audit-trials.csv','/tmp/feasible-parent-preflight-independent-audit-samples.csv',
          '/tmp/audit_feasible_parent_preflight.py','/tmp/feasible-parent-support-roundoff-resolution.json',
          '/tmp/feasible-parent-support-roundoff-resolution.log','/tmp/resolve_feasible_parent_support_roundoff.py',
          '/tmp/phase9_preflight_lifecycle_audit.json','/tmp/phase9_preflight_lifecycle_audit.py']:
 add(p,'preflight','preflight_independent_audit','Original synthetic audit and separate source-faithful support-roundoff closure; original flagged audit is retained without rewriting')
add('/tmp/feasible-parent-preflight-execution.log','preflight','execution','Original executable notebook preflight log; synthetic ordinal units, not market returns')
add('/tmp/feasible-parent-market-execution.log','market','execution','Original executable notebook market campaign log, including selection seal/native lifecycle')
for p in ['/tmp/paired-sparse-paths-profile.json','/tmp/paired-sparse-paths-profile.log']:
 add(p,'market','paired_kernel_profile','Fixed synthetic two-cost shared-state kernel equivalence/runtime evidence; not a model-return claim')
add('/tmp/feasible-parent-seal.log','market','execution','Original implementation-seal/static-contract creation output, before registered preflight')
add('/tmp/finalize_phase9_economic_card.py','market','archival','Method deriving final small economic sidecars from the executed read-only audit')
add(__file__,'market','archival','Archive method, exact immutable-reference verification and two-Recorder manifest construction')
# Preserve final complete and any retained partial references exactly as present; invent none.
reference_files=sorted(p for p in CARD.iterdir() if p.is_file() and 'reference' in p.name and p.suffix=='.json')
for p in reference_files:
 role='preflight' if p.name.startswith('preflight') else 'market'
 add(p,role,'reference_bindings','Exact pre-existing recorder reference bytes; complete/partial status retained without rewriting')

qlib.init(provider_uri=str(ROOT/summary['provider_uri']),region='us',kernels=1)
science_counts={};local=CARD/'qlib_recorder';(local/EID).mkdir(parents=True,exist_ok=True)
with R.uri_context(str(URI)):
 for role,rid in IDS.items():
  record=R.get_recorder(experiment_name=EXPERIMENT,recorder_id=rid)
  assert record.client.get_run(rid).info.status=='FINISHED'
  recdir=Path(record.get_local_dir());assert recdir==URI/EID/rid
  automatic={'artifacts/code_diff.txt','artifacts/code_cached.txt','artifacts/code_status.txt'}
  before={str(p.relative_to(recdir)):sha(p) for p in recdir.rglob('*') if p.is_file() and str(p.relative_to(recdir)) not in automatic}
  for item in items:
   if item['role']!=role:continue
   existing=recdir/'artifacts'/item['artifact']
   if existing.exists():assert sha(existing)==item['sha256'],'Never overwrite a different diagnostic artifact.'
   record.log_artifact(str(item['path']),artifact_path=str(Path(item['artifact']).parent))
   assert sha(existing)==item['sha256']
  # Final financial card belongs to market R. Preflight retains only its own original/reference evidence.
  if role=='market':
   for p in sorted(CARD.iterdir()):
    if not p.is_file() or p.name=='artifact_manifest.json':continue
    destination=recdir/'artifacts/card'/p.name
    if destination.exists():assert destination.read_bytes()==p.read_bytes(),'Never overwrite a different final card snapshot.'
    record.log_artifact(str(p),artifact_path='card')
    assert destination.read_bytes()==p.read_bytes()
  _remove_qlib_code_snapshot_artifacts(recdir)
  assert all((recdir/n).is_file() and sha(recdir/n)==h for n,h in before.items()),role
  assert record.client.get_run(rid).info.status=='FINISHED'
  science_counts[role]=len(before)
  shutil.copytree(recdir,local/EID/rid,dirs_exist_ok=True)
  original={str(p.relative_to(recdir)):(p.stat().st_size,sha(p)) for p in recdir.rglob('*') if p.is_file()}
  destination=local/EID/rid
  copied={str(p.relative_to(destination)):(p.stat().st_size,sha(p)) for p in destination.rglob('*') if p.is_file()}
  assert original==copied,role
shutil.copy2(URI/EID/'meta.yaml',local/EID/'meta.yaml')
assert sorted(p.name for p in (local/EID).iterdir() if p.is_dir())==sorted(IDS.values())
assert refbytes=={role:sha(CARD/f'{role}_reference.json') for role in IDS}
for role,ref in refs.items():
 for artifact,binding in ref['artifacts'].items():assert sha(local/EID/IDS[role]/'artifacts'/binding['artifact'])==binding['sha256']

lookup={(i['role'],i['artifact']):i for i in items};entries=[]
for p in sorted(CARD.rglob('*')):
 if not p.is_file() or p.name=='artifact_manifest.json':continue
 rel=p.relative_to(CARD).as_posix()
 assert '__pycache__' not in p.parts and p.name not in {'.DS_Store','.lock'},rel
 rid=next((rid for rid in IDS.values() if rid in p.parts),None)
 role=next((role for role,value in IDS.items() if value==rid),'card')
 artifact=rel.split('/artifacts/',1)[1] if '/artifacts/' in rel else None
 item=lookup.get((role,artifact))
 if item:
  purpose=item['purpose'];source={'archived_from':item['source_path'],'original_sha256':item['sha256'],'original_mtime_utc':item['mtime_utc']}
 elif artifact and artifact.startswith('card/'):
  purpose='Final independently reviewed slim-card snapshot, enriched economic interpretation distinct from bound final_summary.pkl';source={'card_file':artifact[5:],'recorder_id':rid}
 elif artifact:
  if artifact.startswith('search_trials/'):purpose='All unique 33-coordinate heads, scalar whole-path risks/utilities and optimizer request/cache counts; selected model and retained books'
  elif artifact.startswith('parent_reproductions/'):purpose='Only genuinely absent selected outer parents reproduced under the frozen original protocol; parent work distinct from new search'
  elif artifact.startswith('preflight/'):purpose='Registered ordinal synthetic parent reproduction/search/ensemble gate; not market performance'
  elif artifact.startswith('inputs/'):purpose='Frozen source identity, native Dataset fold/scaler evidence or exact source/plan/reference snapshot; parent trees referenced, not copied'
  elif artifact.startswith('portfolio_analysis/'):purpose='Registered own-book portfolio and native Qlib accounting/risk evidence, without fresh-forward or real-fill claim'
  else:purpose='Authoritative saved model/selection/financial analysis evidence, preserved byte for byte'
  source={'recorder_id':rid,'artifact':artifact,'bound_reference':f'{role}_reference.json' if artifact in {b['artifact'] for b in refs[role]['artifacts'].values()} else None}
 elif rel.startswith('qlib_recorder/'):
  purpose='Native Qlib/MLflow experiment/run identity, parameters and scope tags';source={'canonical_recorder_uri':str(URI),'recorder_id':rid}
 else:
  purpose='Slim card entry, immutable registration/references, executed notebook, final conclusion or small diagnostic/verification sidecar';source={'card_root':True,'experiment':EXPERIMENT}
 dates={'source_as_of':'2026-09-25','development_signal_years':'2016-2022','diagnostic_signal_years':'2023-2026; already seen history','user_window':'2026-08-03 trade through2026-09-25open;38intervals','synthetic_ordinal_rows':None}
 if role=='preflight':dates={'synthetic_ordinal_rows':'0..503','market_performance_dates':None,'native_source_check_as_of':'2026-09-25'}
 entries.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p),'recorder_id':rid,'role':role,'purpose':purpose,'source':source,'date_range':dates})
manifest={'schema_version':2,'experiment':EXPERIMENT,'artifact_policy':'slim_card_recorder_first','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'canonical_recorder_uri':str(URI),'authoritative_market_recorder':IDS['market'],'registered_preflight_recorder':IDS['preflight'],
 'retained_recorder_ids':list(IDS.values()),'reference_files':{p.name:sha(p) for p in reference_files},
 'partial_reference_policy':'Preserve every pre-existing reference snapshot; no partial reference files were found at card root and none is fabricated. Complete references and all currently persisted bound artifacts are retained.',
 'original_financial_artifacts_unchanged':True,'recorder_finished_scope':{'market':'Completed registered historical research; all three families failed historical qualification.','preflight':'Passed registered synthetic incremental optimization diagnostic; not a market gain.'},
 'parent_recorders_referenced_only':{'data':'52f40778cbe64ca6a5864da356b960d0','qualified_source':'bd754e6362b241f194351a1e7aebff6f','allocation_parent':'300530137db54b6c807c39ebb9e8c331','prior_failed_preflight':'a728372af45c4e678f326996db675a43'},
 'parent_retention':'No full parent Recorder trees copied. Required frozen input/fold evidence and actually reproduced missing parents belong to this registered run.',
 'path_base':'card directory','self_reference_policy':'Root manifest excludes itself and is not recursively copied into a Recorder.',
 'entry_count':len(entries),'total_bytes':sum(e['bytes'] for e in entries),'maximum_file_bytes':max(e['bytes'] for e in entries),'entries':entries}
(CARD/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
assert all((CARD/e['path']).stat().st_size==e['bytes'] and sha(CARD/e['path'])==e['sha256'] for e in entries)
report={'recorders':IDS,'science_before_preserved':science_counts,'entry_count':len(entries),'total_bytes':manifest['total_bytes'],'maximum_file_bytes':manifest['maximum_file_bytes'],
 'manifest_sha256':sha(CARD/'artifact_manifest.json'),'root_files':sorted(p.name for p in CARD.iterdir() if p.is_file()),'bound_references_unchanged':True,'original_financial_artifacts_unchanged':True,
 'source_files_unchanged':all(sha(ROOT/n)==h for n,h in summary['source_sha256'].items()),'historical_qualification':False}
print(json.dumps(report,ensure_ascii=False,indent=2))

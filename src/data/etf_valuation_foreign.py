"""Record bounded foreign issuer quarterly source transport qualification."""
import hashlib
import json
import os
from pathlib import Path
import pandas as pd


def run(root,staging='/tmp/etf_foreign_quarterly'):
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.nas_market_data import archive_nas_recorder
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root).resolve();staging=Path(staging);card=root/'research/etf_foreign_quarterly_source_20261016'
    provider=root/'data/qlib/soxx_pit_leveraged_training_v1';qlib.init(provider_uri=str(provider),region='us',kernels=1)
    calendar=D.calendar(start_time='2019-01-01',end_time='2026-07-02')
    if not len(calendar):raise ValueError('Qlib calendar missing')
    if (card/'summary.json').exists():
        s=json.loads((card/'summary.json').read_text());R.set_uri(s['recorder_uri']);rec=R.get_recorder(recorder_id=s['recorder_id'],experiment_id=s['experiment_id'])
        payload=rec.load_object('inputs/raw_payloads')
        for e in rec.load_object('inputs/acquisition_manifest'):assert hashlib.sha256(payload[e['name']]).hexdigest()==e['sha256']
        return s
    plan=json.loads((card/'plan.json').read_text());review=json.loads((card/'design_review.json').read_text());ph=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if review.get('plan_sha256')!=ph or review.get('decision')!='approved_bounded_source_feasibility_only':raise ValueError('unapproved source plan')
    entries=json.loads((staging/'entries.json').read_text())
    if len(entries)>plan['max_new_http_requests'] or {e['url'] for e in entries}!={e['url'] for e in plan['sources']}:raise ValueError('source scope changed')
    payload={e['name']:(staging/e['name']).read_bytes() if (staging/e['name']).exists() else b'' for e in entries}
    for e in entries:
        if hashlib.sha256(payload[e['name']]).hexdigest()!=e['sha256']:raise ValueError('payload changed')
    matrix=pd.DataFrame(entries);matrix['qualified_quarterly_input']=False
    matrix['finding']=['native_HTTP403_not_financial_data_missing' if e['status']==403 else 'native_transport_error_not_financial_data_missing' if e['curl_returncode'] else 'requires_document_parser' for e in entries]
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*']};tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,acquisition_manifest=entries,raw_payloads=payload,indexed_discovery=json.loads((card/'indexed_discovery.json').read_text()))
        rec.save_objects(artifact_path='analysis',qualification=matrix);rec.save_objects(artifact_path='sources',module=Path(__file__).read_text())
        summary={'experiment':plan['experiment'],'decision':'historical_quarter_reports_discoverable_native_transport_not_qualified','new_requests':len(entries),'http403':sum(e['status']==403 for e in entries),'curl_errors':sum(e['curl_returncode']!=0 for e in entries),'qualified_quarterly_rows':0,'new_fits':0,'new_backtests':0,'objective_achieved':False,'provider_uri':str(provider),'region':'us','market':'foreign SOXX issuer source qualification','as_of':'2026-10-16','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'plan_sha256':ph,'artifact_policy':policy,'availability_boundary':'Search-indexed issuer excerpts establish discovery, not original historical bytes/acceptance/vintage; no financial input synthesized','next_action':'Use an authorized working transport or SEC original 6-K exhibits for identified issuer quarters; extract complete continuous histories with publication, attribution, TWD/FX and ADR basis before EP; this six-URL probe stops here.'}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),requests=len(entries),qualified_quarterly_rows=0)
    matrix.to_csv(card/'source_qualification.csv',index=False)
    binding=json.loads((root/'research/etf_pit_fundamentals_repair_20261016/nas_binding.json').read_text());binding.pop('evidence',None);(card/'nas_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    (card/'verification_manifest.json').write_text(json.dumps({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},indent=2)+'\n')
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));(card/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary

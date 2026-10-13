"""Record a limited manually reviewed source gate; original bytes unavailable."""
import json,hashlib
from pathlib import Path
import pandas as pd
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed source gate changed')
        return old
    raw=(card/'reviewed_facts.json').read_bytes();rows=json.loads(raw)
    if {r['quarter'] for r in rows}!={'Q4 2025','Q1 2026','Q2 2026'}:raise ValueError('review scope differs')
    for row in rows:
        if row['url'] not in plan['urls'].values() or row['quarter'] not in row['reviewed_table_header']:raise ValueError('reviewed source/header differs')
        if pd.Timestamp(row['release_date'])>pd.Timestamp(plan['as_of']):raise ValueError('future release')
        if row['bookings_row_present']!=(row['current_net_bookings_eur_million'] is not None):raise ValueError('unknown bookings cannot be encoded as a number')
        if row['original_http_bytes_verified']:raise ValueError('original source bytes not available')
    tracking=card/'qlib_recorder';current=[r for r in rows if r['quarter'] in ['Q1 2026','Q2 2026']];passed=all(r['bookings_row_present'] for r in current)
    with R.start(experiment_name=plan['experiment'],recorder_name='reviewed_bookings_continuity',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='sources',reviewed_facts=rows,review_sha256=hashlib.sha256(raw).hexdigest(),access={'direct_html_status':403,'linked_pdf_status':403,'review_channel':'web tool rendered official releases; no raw HTML/PDF bytes retained'})
        rec.save_objects(artifact_path='code',workflow=Path(__file__).read_text())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'needs_current_numeric_bookings_source_no_fits','current_table_continuity_pass':passed,'objective_achieved':False,'new_fits':0,'new_backtests':0,'scope':plan['scope'],'source_access':plan['source_access'],'original_http_bytes_verified':False,'provider_uri':str(root/plan['provider_uri']),'provider_role':'reference environment only; no market data used','region':'us','as_of':plan['as_of'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/recorder.html','artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['sources/*','code/*']}}
        rec.save_objects(artifact_path='analysis',summary=summary,rows=rows)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    pd.DataFrame([{k:row[k] for k in ['quarter','release_date','bookings_row_present','current_net_bookings_eur_million','current_net_sales_eur_million']} for row in rows]).to_csv(card/'source_gate.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'recorder_uri':str(tracking),'source_type':'manually reviewed official tables via web provider; original byte download blocked','artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

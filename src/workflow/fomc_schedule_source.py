"""Qlib-recorded prospective schedule chain and explicit cancellation audit."""
import hashlib,json,io
from pathlib import Path
import pandas as pd
from pypdf import PdfReader
from src.data.fomc_schedule_source import collect_sources
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.data import D
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed schedule plan changed')
        return old
    calendar=pd.DatetimeIndex(D.calendar(freq='day'));calendar=calendar[calendar<=pd.Timestamp(plan['market_as_of'])]
    rows,manifest,blobs=collect_sources(root/plan['raw_snapshot'],plan)
    cancellation=' '.join(PdfReader(io.BytesIO(blobs[plan['cancellation_url']])).pages[4].extract_text().split())
    if 'in lieu of the meeting' not in cancellation or 'scheduled for next Tuesday and Wednesday' not in cancellation:raise ValueError('cancellation source not confirmed')
    actual={r['minutes_meeting_end'] for r in rows};records=[]
    for row in rows:
        r=dict(row);r['date_evidence_status']=r['parse_status'];r['entry_date']=None;r['known_before_entry']=False
        if r['parse_status']=='parsed':
            event=pd.Timestamp(r['next_meeting_end']);position=calendar.searchsorted(event)
            if position>0 and position<len(calendar) and calendar[position]==event:
                entry=calendar[position-1];r['entry_date']=str(entry.date());r['known_before_entry']=pd.Timestamp(r['release_date'])+pd.Timedelta(hours=23,minutes=59,seconds=59)<entry+pd.Timedelta(hours=9,minutes=46)
            if r['next_meeting_end']==plan['cancellation']['scheduled_end']:
                r['date_evidence_status']='cancelled_publicly_before_entry';r['cancellation_available_date']=plan['cancellation']['assumed_available_date']
                if r['entry_date'] is None or pd.Timestamp(r['cancellation_available_date'])>=pd.Timestamp(r['entry_date']):raise ValueError('cancellation not before entry')
            elif event>pd.Timestamp(plan['market_as_of']):r['date_evidence_status']='future_schedule_no_market_outcome'
            elif r['next_meeting_end'] not in actual:r['date_evidence_status']='planned_actual_mismatch_needs_review'
            elif not r['known_before_entry']:r['date_evidence_status']='not_known_before_entry'
            else:r['date_evidence_status']='prior_public_date_evidence'
        records.append(r)
    table=pd.DataFrame(records);usable=table.loc[table.date_evidence_status.eq('prior_public_date_evidence') & pd.to_datetime(table.next_meeting_end).ge(plan['research_start'])].copy()
    if usable.next_meeting_end.duplicated().any():raise ValueError('duplicate prospective schedule event')
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='prior_minutes_schedule_evidence',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='sources',manifest=manifest,documents=blobs,cancellation_page5_text=cancellation,reference_calendar=calendar)
        rec.save_objects(artifact_path='code',data=(root/'src/data/fomc_schedule_source.py').read_text(),workflow=Path(__file__).read_text())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'date_chain_collected_needs_event_timing_and_information_test','objective_achieved':False,'new_fits':0,'new_backtests':0,'minutes_documents':len(rows),'qualified_past_date_events':len(usable),'status_counts':table.date_evidence_status.value_counts().to_dict(),'original_first_receipt_certified':False,'current_calendar_used_as_advance_signal':False,'announcement_clock_certified_per_event':False,'provider_uri':str(root/plan['reference_provider']),'region':'us','as_of':plan['as_of'],'market_as_of':plan['market_as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/recorder.html','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['sources/*','code/*','analysis/full_schedule']}}
        rec.save_objects(artifact_path='analysis',full_schedule=table,usable=usable,summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    table.drop(columns=['sentence'],errors='ignore').to_csv(card/'schedule_evidence.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

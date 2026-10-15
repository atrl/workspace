"""New information run explicitly bound to qualified official rendered text."""
import hashlib,json
from pathlib import Path
import pandas as pd
from src.workflow.minute_action_value import _source
from src.workflow.cpi_opening_context_information import fit_information
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('new information plan changed')
        return old
    calendar,calendar_audit=_source(root,plan['calendar_source'],plan['calendar_source']['artifacts'])
    events=calendar['sources/event_manifest'];gate=calendar['sources/gate']
    if not gate['passed'] or len(events)!=48 or events.release_date.nunique()!=48:raise ValueError('qualified48-event source required')
    if int(events.is_trading_session.sum())!=47 or events.loc[~events.is_trading_session,'release_date'].tolist()!=['2020-04-10']:raise ValueError('nontrading source mapping changed')
    source,_=_source(root,plan['sources']['frame'],['inputs/full_reference_frame'])
    opening,_=_source(root,plan['sources']['opening'],['inputs/opening_timing'])
    frame=source['inputs/full_reference_frame'];frame=frame.loc[frame.trade_date.between('2019-01-01','2022-12-31')].copy()
    timing=opening['inputs/opening_timing'].reindex(pd.DatetimeIndex(frame.signal_date))
    released=set(pd.to_datetime(events.release_date));flags=frame.trade_date.isin(released)
    counts=flags.groupby(frame.trade_date.dt.year).sum().to_dict()
    if len(frame)!=1008 or int(flags.sum())!=47 or counts!={2019:12,2020:11,2021:12,2022:12}:raise ValueError('same-date native event counts differ')
    if flags.loc[frame.trade_date.eq(pd.Timestamp('2020-04-13'))].any() or frame.trade_date.eq(pd.Timestamp('2020-04-10')).any():raise ValueError('nontrading event shifted/synthetic row')
    decisions=dict(zip(frame.trade_date,timing.decision_at));join=[]
    for event in events.itertuples():
        date=pd.Timestamp(event.release_date)
        if event.is_trading_session:
            if date not in decisions or pd.Timestamp(event.release_at)>decisions[date]:raise ValueError('release not known before actual decision')
        elif date in decisions:raise ValueError('nontrading release entered native rows')
        join.append({'release_date':event.release_date,'is_trading_session':bool(event.is_trading_session),'release_at':event.release_at,'decision_at':str(decisions[date]) if date in decisions else None})
    join=pd.DataFrame(join)
    path=card/'run_state.json';progress=json.loads(path.read_text()) if path.exists() else {'plan_sha256':digest,'fit_attempts':[],'fitted':[]}
    if progress['plan_sha256']!=digest:raise ValueError('fit progress identity changed')
    def save():atomic_json(progress,path)
    tracking=card/'qlib_recorder';kwargs={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kwargs.update(recorder_id=progress['recorder_id'],resume=True)
    with R.start(**kwargs):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));save()
        if not progress.get('source_saved'):
            rec.save_objects(artifact_path='inputs',plan=plan,calendar_source_audit=calendar_audit,calendar_gate=gate,event_join=join)
            rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),fixed_information_workflow=(root/'src/workflow/cpi_opening_context_information.py').read_text(),saved_input_preparation=(root/'src/workflow/seed_disagreement_information.py').read_text(),analysis=(root/'src/analysis/seed_disagreement_information.py').read_text())
            progress['source_saved']=True;save()
    result=fit_information(root,card,plan,rec,events,progress,save)
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/data','models/*','analysis/daily','analysis/scalers','analysis/bootstrap_indices','analysis/bootstrap_replicates']}
    summary={'experiment':plan['experiment'],'plan_sha256':digest,**result,'source_gate':gate,'source_format':'official rendered text capture, not publisherHTML bytes','objective_achieved':False,'new_statistical_fits':len(progress['fitted']),'new_allocator_fits':0,'new_native_cases':0,'source_events':48,'same_day_trading_events':47,'new_retrieval_operations':0,'evaluation_as_of':plan['evaluation_as_of'],'provider_uri':str(root/plan['provider_uri']),'region':'us','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'source_amendment':plan['source_amendment']}
    rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')),completed_statistical_fits=len(progress['fitted']))
    join.to_csv(card/'event_join.csv',index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['calendar_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

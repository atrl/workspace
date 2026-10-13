"""Close-session repair with immutable before/after NAS lineage and Qlib audit."""
import hashlib,json
from pathlib import Path
import pandas as pd
import exchange_calendars as xc
from src.data.intraday import read_intraday_cache
from src.data.intraday_quality import audit_minute_coverage
from src.data.minute_refresh import refresh_published_core
from src.data.nas_market_data import load_current_nas_binding,archive_nas_recorder
from src.data.minute_intake import atomic_json


def _audit(raw,session_date):
    return audit_minute_coverage(raw,xc.get_calendar('XNYS',start=raw.date.min(),end=session_date).schedule,symbols=['SOXL','SOXX'],timestamp_convention='right_edge',observed_at=pd.Timestamp.now(tz='UTC'))


def run(*,root,card,plan):
    import qlib,yaml
    from qlib.data import D
    from qlib.workflow import R
    from futu import SysConfig
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('repair plan changed')
        return old
    release=root/'production/stage1_futu_sim_v1/runtime_release.json'
    if hashlib.sha256(release.read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release identity changed')
    before_binding=load_current_nas_binding(workspace_root=root)
    if before_binding['source_id']!=plan['expected_before_source_id']:raise ValueError('NAS source advanced; re-audit before any write')
    cache=root/'data/processed/intraday/soxl_soxx_none_1m.parquet';entry=next(x for x in before_binding['objects'] if x['local_path']==str(cache.relative_to(root)))
    if hashlib.sha256(cache.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('local raw differs from NAS; do not overwrite unrelated state')
    before=read_intraday_cache(cache);coverage_before=_audit(before,plan['session_date']);target=coverage_before.loc[coverage_before.date.eq(plan['session_date'])]
    if len(target)!=2 or target.status.eq('in_progress').any():raise ValueError('target session not closed')
    if target.missing_minutes.sum()==0:raise ValueError('no closed-minute gap remains; do not refetch')
    before_tail=before.loc[before.date.eq(plan['session_date'])].copy();columns=['open','high','low','close','volume','turnover']
    prefix=before.loc[before.date.lt(plan['session_date'])].set_index(['symbol','timestamp'])[columns].sort_index()
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='closed_tail_repair',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='inputs',plan=plan,before_binding=before_binding,before_coverage=coverage_before,before_tail=before_tail)
    atomic_json({'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'plan_sha256':digest,'phase':'refreshing'},card/'run_state.json')
    SysConfig.set_all_thread_daemon(True)
    result=refresh_published_core(root=root,session_date=plan['session_date'],start_session_date=plan['session_date'],host=plan['host'],port=plan['port'])
    atomic_json(result,card/'refresh_receipt.json')
    after_binding=load_current_nas_binding(workspace_root=root);after=read_intraday_cache(cache);coverage_after=_audit(after,plan['session_date']);latest=coverage_after.loc[coverage_after.date.eq(plan['session_date'])]
    if not latest.training_eligible.all() or coverage_after.missing_minutes.sum()!=0:raise ValueError('closed minute gaps remain after refresh')
    pd.testing.assert_frame_equal(prefix,after.loc[after.date.lt(plan['session_date'])].set_index(['symbol','timestamp'])[columns].sort_index(),check_dtype=False,check_exact=True)
    authority=result['nas_source']['observation']['source_id']
    if after_binding['source_id']!=authority:raise ValueError('NAS current advanced after publication; inspect receipts')
    entry=next(x for x in after_binding['objects'] if x['local_path']==str(cache.relative_to(root)))
    if hashlib.sha256(cache.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('new local raw differs from NAS receipt')
    left=before_tail.set_index(['symbol','timestamp'])[columns];right=after.loc[after.date.eq(plan['session_date'])].set_index(['symbol','timestamp'])[columns];overlap=left.index.intersection(right.index)
    differences=(left.loc[overlap]-right.loc[overlap]).abs()
    native_audits=[]
    for freq,manifest in result['manifests'].items():
        qlib.init(provider_uri=manifest['provider_uri'],region='us',kernels=1)
        fields=['$close']+(['$price_valid'] if freq=='1min' else ['$is_complete','$observed_1m_count','$expected_1m_count'])
        values=D.features(plan['symbols'],fields,start_time=plan['session_date'],end_time=plan['session_date']+' 16:00:00',freq=freq)
        if values['$close'].isna().any():raise ValueError('native latest close missing')
        if freq!='1min' and not values['$is_complete'].eq(1).all():raise ValueError('native latest aggregate incomplete')
        if freq=='1min' and not values['$price_valid'].eq(1).all():raise ValueError('native latest source invalid')
        counts=values.groupby(level='instrument').size().to_dict();expected=390/int(freq.replace('min',''))
        if set(counts)!=set(plan['symbols']) or any(n!=expected for n in counts.values()):raise ValueError('native latest row count differs')
        native_audits.append({'freq':freq,'rows_per_symbol':counts,'status':'passed'});rec.save_objects(artifact_path='analysis/native',**{freq:values})
    if hashlib.sha256(release.read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed during data-only task')
    summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'closed_core_tail_repaired_context_not_extended','objective_achieved':False,'new_fits':0,'new_backtests':0,'before_source_id':before_binding['source_id'],'after_source_id':after_binding['source_id'],'core_as_of':after_binding['as_of'],'context_as_of':after_binding.get('metadata',{}).get('context_as_of'),'raw_rows_before':len(before),'raw_rows_after':len(after),'new_minutes':len(after)-len(before),'missing_before':int(coverage_before.missing_minutes.sum()),'missing_after':int(coverage_after.missing_minutes.sum()),'invalid_rows_after':int(coverage_after.invalid_rows.sum()),'prefix_raw_prices_unchanged':True,'overlap_tail_rows':len(overlap),'overlap_tail_rows_changed':int(differences.gt(1e-10).any(axis=1).sum()),'native_audits':native_audits,'provider_uri':result['manifests']['1min']['provider_uri'],'region':'us','as_of':plan['session_date'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/data.html','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','analysis/native/*','analysis/full_coverage','sources/*']}}
    rec.save_objects(artifact_path='analysis',summary=summary,refresh_receipt=result,after_binding=after_binding,full_coverage=coverage_after,tail=after.loc[after.date.eq(plan['session_date'])],overlap_differences=differences)
    rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),refresh=(root/'src/data/minute_refresh.py').read_text());artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),repaired_minutes=summary['new_minutes'])
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    coverage_before.tail(6).to_csv(card/'before_coverage.csv',index=False);coverage_after.tail(6).to_csv(card/'after_coverage.csv',index=False);coverage_after.loc[coverage_after.repair_required].to_csv(card/'remaining_anomalies.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');atomic_json({'phase':'complete','recorder_id':rec.id},card/'run_state.json');return summary

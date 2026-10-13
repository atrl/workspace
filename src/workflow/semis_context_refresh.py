"""Merge a bounded peer-context tail and publish a source-linked NAS version."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xc
from src.data.minute_intake import collect_futu_bundle,atomic_json,atomic_parquet
from src.data.minute_price_basis import normalize_none_history,published_intraday_basis
from src.data.semis_context_refresh import audit_context
from src.data.qlib_native import build_qlib_store_from_frame,QlibProviderConfig,load_qlib_feature_frame
from src.data.nas_market_data import load_current_nas_binding,publish_nas_snapshot,register_nas_source,archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed context plan changed')
        return old
    before=load_current_nas_binding(workspace_root=root)
    if before['source_id']!=plan['expected_before_source_id'] or before['as_of']!=plan['as_of']:raise ValueError('NAS source advanced; re-audit before publishing')
    release=root/'production/stage1_futu_sim_v1/runtime_release.json'
    if hashlib.sha256(release.read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    original_manifest=root/plan['old_manifest']
    if hashlib.sha256(original_manifest.read_bytes()).hexdigest()!=plan['old_manifest_sha256']:raise ValueError('old context manifest differs')
    original=json.loads(original_manifest.read_text());core_path=root/plan['core_manifest'];core_bytes=core_path.read_bytes();core=json.loads(core_bytes)
    raw_core=root/core['raw_cache']
    if hashlib.sha256(raw_core.read_bytes()).hexdigest()!=core['raw_sha256']:raise ValueError('core cache differs from NAS')
    output=root/plan['raw_output'];tail=collect_futu_bundle(requests={s:'K_5M' for s in plan['symbols']},start=plan['tail_start'],end=plan['as_of'],output_dir=output/'tail',checkpoint_dir=Path('/tmp/peer-refresh-20261014/checkpoints'),host=plan['host'],port=plan['port'],workers=3)
    pieces=[];events={};counts=[];files={}
    for symbol in plan['symbols']:
        old_path=root/plan['old_sources'][symbol]['path']
        if hashlib.sha256(old_path.read_bytes()).hexdigest()!=plan['old_sources'][symbol]['sha256']:raise ValueError('old raw context bytes differ')
        old=pd.read_parquet(old_path);prior=next(e for e in original['entries'] if e['symbol']==symbol)
        entry=next(e for e in tail['entries'] if e['symbol']==symbol);new_path=output/'tail'/entry['path'];new=pd.read_parquet(new_path)
        if not old.time_key.str[:10].lt(plan['tail_start']).all() or not new.time_key.str[:10].between(plan['tail_start'],plan['as_of']).all():raise ValueError('tail overlaps old context or exceeds cutoff')
        merged=pd.concat([old,new],ignore_index=True).sort_values('time_key').reset_index(drop=True)
        pd.testing.assert_frame_equal(old.sort_values('time_key').reset_index(drop=True),merged.iloc[:len(old)].reset_index(drop=True),check_dtype=False,check_exact=True)
        raw_path=output/'full'/(symbol+'_none.parquet');atomic_parquet(merged,raw_path);files[str(raw_path.relative_to(root))]='source'
        pieces += [normalize_none_history(old,symbol=symbol,interval='5m',fetched_at=prior['fetched_at']),normalize_none_history(new,symbol=symbol,interval='5m',fetched_at=entry['fetched_at'])]
        rehab=next(e for e in tail['rehab'] if e['symbol']==symbol);events[symbol]=pd.read_parquet(output/'tail'/rehab['path'])
        counts.append({'symbol':symbol,'old_rows':len(old),'new_rows':len(new),'merged_rows':len(merged)})
    raw=pd.concat(pieces,ignore_index=True).sort_values(['symbol','timestamp']).reset_index(drop=True)
    coverage=audit_context(raw,xc.get_calendar('XNYS',start=raw.date.min(),end=plan['as_of']).schedule,symbols=plan['symbols'])
    if coverage.missing_bars.sum() or coverage.unexpected_bars.sum():raise ValueError('context native calendar incomplete; do not publish')
    if coverage.loc[coverage.date.ge(plan['tail_start']),'invalid_rows'].sum() or not raw.loc[raw.date.ge(plan['tail_start']),'is_final'].all():raise ValueError('new context tail invalid or unfinished')
    raw_cache=output/'context_none_5m.parquet';atomic_parquet(raw,raw_cache);files[str(raw_cache.relative_to(root))]='source'
    published,basis=published_intraday_basis(raw,events,as_of=plan['as_of']);provider=root/plan['provider'];fields=['open','high','low','close','volume','turnover','vwap','price_valid']
    store=build_qlib_store_from_frame(published.rename(columns={'symbol':'instrument','timestamp':'datetime'}),provider_uri=provider,freq='5min',as_of=plan['as_of'],field_map={k:k for k in fields},source='frozen old peer5m plus observed NONE tail, published corporate-action basis')
    native=load_qlib_feature_frame(QlibProviderConfig(provider,freq='5min'),instruments=plan['symbols'],fields=['$close','$price_valid'],start=plan['tail_start'],end=plan['as_of']+' 16:00:00')
    if not native['$price_valid'].eq(1).all() or native['$close'].isna().any():raise ValueError('native peer tail readback failed')
    if len(native)!=sum(row['new_rows'] for row in counts):raise ValueError('native peer tail counts differ')
    for path in (output/'tail').iterdir():
        if path.is_file():files[str(path.relative_to(root))]='source'
    for path in provider.rglob('*'):
        if path.is_file():files[str(path.relative_to(root))]='derived'
    # Keep the canonical core schema usable from this new manifest location.
    for symbol in ['SOXL','SOXX']:
        source=core_path.parent/(symbol+'_rehab.parquet')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=core['rehab_sha256'][symbol]:raise ValueError('core corporate-action source changed')
        destination=output/source.name;shutil.copyfile(source,destination);files[str(destination.relative_to(root))]='source'
    context_meta={'context_as_of':plan['as_of'],'context_raw_cache':str(raw_cache.relative_to(root)),'context_rehab_paths':{s:str((output/'tail'/(s+'_rehab.parquet')).relative_to(root)) for s in plan['symbols']},'context_provider_uri':str(provider.relative_to(root)),'context_manifest_path':str((output/'source_manifest.json').relative_to(root))}
    merged_manifest={**core,'parent_source_id':before['source_id'],'parent_snapshot':before['snapshot_root'],'context':context_meta,'context_tail_manifest_sha256':hashlib.sha256((output/'tail/source_manifest.json').read_bytes()).hexdigest(),'context_raw_sha256':hashlib.sha256(raw_cache.read_bytes()).hexdigest()}
    source_manifest=output/'source_manifest.json';atomic_json(merged_manifest,source_manifest);files[str(source_manifest.relative_to(root))]='source'
    if load_current_nas_binding(workspace_root=root)['source_id']!=before['source_id']:raise ValueError('concurrent source update; keep new files unregistered')
    inherited=[{**e,'snapshot_root':e.get('snapshot_root',before['snapshot_root'])} for e in before['objects'] if e['local_path'] not in files]
    binding=publish_nas_snapshot(workspace_root=root,source_manifest=source_manifest,files=files,as_of=plan['as_of'],inherited_objects=inherited,source_metadata={**before.get('metadata',{}),**context_meta,'context_rows':len(raw),'context_invalid_rows':int(coverage.invalid_rows.sum()),'quality_status':'partial' if before.get('metadata',{}).get('malformed_rows',0)+coverage.invalid_rows.sum() else 'ok','collector':'mac_futu_opend_core_plus_context'})
    source_status=register_nas_source(binding,workspace_root=root,error_count=int(before.get('metadata',{}).get('malformed_rows',0)+coverage.invalid_rows.sum()),observation={**binding['metadata'],'missing_closed_context_bars':0})
    current=load_current_nas_binding(workspace_root=root)
    if current['source_id']!=binding['source_id'] or hashlib.sha256(raw_core.read_bytes()).hexdigest()!=core['raw_sha256'] or core_path.read_bytes()!=core_bytes:raise ValueError('publication identity or original core changed')
    if hashlib.sha256(release.read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed during intake')
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='context_tail_refresh',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='inputs',plan=plan,before_binding=before,after_binding=binding,source_registration=source_status,tail_manifest=tail,provider_manifest=store,basis_audit=basis)
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),quality=(root/'src/data/semis_context_refresh.py').read_text(),core_refresh=(root/'src/data/minute_refresh.py').read_text())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'core_and_context_source_dates_aligned_feature_pipeline_not_refit','objective_achieved':False,'new_fits':0,'new_backtests':0,'before_source_id':before['source_id'],'after_source_id':binding['source_id'],'core_as_of':binding['as_of'],'context_as_of':plan['as_of'],'new_context_bars':sum(r['new_rows'] for r in counts),'context_rows':len(raw),'source_counts':counts,'missing_context_bars':int(coverage.missing_bars.sum()),'context_invalid_rows':int(coverage.invalid_rows.sum()),'old_context_raw_prefix_unchanged':True,'core_source_unchanged':True,'historical_first_receipt_certified':False,'all_model_inputs_rebuilt':False,'provider_uri':str(provider),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/data.html','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/coverage','analysis/native_tail']}}
        rec.save_objects(artifact_path='analysis',summary=summary,coverage=coverage,native_tail=native);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),new_context_bars=summary['new_context_bars'])
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    pd.DataFrame(counts).to_csv(card/'source_counts.csv',index=False);coverage.loc[coverage.date.ge(plan['tail_start'])].to_csv(card/'latest_coverage.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

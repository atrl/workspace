"""Notebook entry for a source repair, without strategy or vintage claims."""
import json,hashlib
from pathlib import Path
import pandas as pd
from src.data.cboe_skew_source import collect,read_skew,build_store
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    import qlib,yaml
    from qlib.data import D
    from qlib.workflow import R
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed intake plan changed')
        return old
    qlib.init(provider_uri=str(root/plan['reference_provider']),region='us',kernels=1)
    reference=pd.DatetimeIndex(D.calendar(freq='day'));reference=reference[reference<=pd.Timestamp(plan['market_as_of'])]
    snapshot=root/plan['raw_snapshot'];meta=collect(snapshot,urls=plan['urls']);values=read_skew((snapshot/'history.csv').read_bytes())
    if values.index.max()>pd.Timestamp(plan['intake_as_of']):raise ValueError('source observation exceeds intake as_of')
    legacy_path=root/plan['legacy_file'];legacy_raw=legacy_path.read_bytes();legacy=read_skew(legacy_raw)
    common=values.index.intersection(legacy.index);delta=(values.loc[common]-legacy.loc[common]).abs()
    relevant=reference[reference>=pd.Timestamp(plan['audit_start'])];missing=relevant.difference(values.index)
    provider=root/plan['provider'];native,store=build_store(values,reference_calendar=reference,provider=provider,as_of=plan['intake_as_of'],source_sha256=meta['history']['sha256'])
    gaps=pd.DataFrame({'source_date':missing,'status':'official_observation_missing_no_fill'})
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='official_skew_source',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='sources',manifest=meta,history=(snapshot/'history.csv').read_bytes(),methodology=(snapshot/'methodology.pdf').read_bytes(),legacy_sha256=hashlib.sha256(legacy_raw).hexdigest(),legacy=legacy_raw,reference_calendar=reference)
        rec.save_objects(artifact_path='inputs',native=native,provider_manifest=store,provider_files={str(p.relative_to(provider)):p.read_bytes() for p in provider.rglob('*') if p.is_file()})
        rec.save_objects(artifact_path='code',data=(root/'src/data/cboe_skew_source.py').read_text(),workflow=Path(__file__).read_text())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'retrospective_source_updated_vintage_unverified','objective_achieved':False,'new_fits':0,'new_backtests':0,'source_rows':len(values),'source_first':str(values.index.min().date()),'source_last':str(values.index.max().date()),'legacy_last':str(legacy.index.max().date()),'additional_source_dates':len(values.index.difference(legacy.index)),'overlap_rows':len(common),'maximum_overlap_absolute_difference':float(delta.max()),'overlap_differences_above_0_0001':int(delta.gt(1e-4).sum()),'reference_missing_dates':missing.strftime('%Y-%m-%d').tolist(),'observations_filled':0,'historical_vintage_certified':False,'exact_first_publication_certified':False,'prelaunch_backfill_as_feature':False,'provider_uri':str(provider),'region':'us','as_of':plan['intake_as_of'],'market_as_of':plan['market_as_of'],'source_manifest':meta,'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/data.html','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['sources/*','inputs/*','code/*']}}
        rec.save_objects(artifact_path='analysis',summary=summary,gaps=gaps)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    gaps.to_csv(card/'source_gaps.csv',index=False);atomic_json(meta,card/'source_manifest.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

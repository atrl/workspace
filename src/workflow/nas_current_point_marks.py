"""Read pinned current NAS core data through the dual-schema point loader."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.full_path_prices import build_full_path_prices
from src.data.qlib_native import QlibProviderConfig,load_qlib_feature_frame
from src.data.nas_market_data import load_current_nas_binding,archive_nas_recorder
from src.data.minute_intake import atomic_json


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('adapter audit plan changed')
        return old
    binding=load_current_nas_binding(workspace_root=root)
    if binding['source_id']!=plan['source_id']:raise ValueError('NAS current changed; preserve pinned plan')
    ref_path=root/plan['reference_provider'];ref_manifest_before=hashlib.sha256((ref_path/'qlib_store_manifest.json').read_bytes()).hexdigest()
    reference=load_qlib_feature_frame(QlibProviderConfig(ref_path,freq='1min'),instruments=['SOXL','SOXX'],fields=['$execution_price','$raw_execution_price'],start=plan['start'],end=plan['reference_as_of']+' 16:00:00')
    provider=root/plan['provider'];quotes,manifest=build_full_path_prices(source_manifest=root/plan['source_manifest'],source_sha256=plan['source_sha256'],provider_uri=provider,start=plan['start'],as_of=plan['as_of'],workspace_root=root)
    common=quotes.index.intersection(reference.index)
    if not common.equals(reference.index):raise ValueError('new source omitted a frozen reference mark')
    delta=(quotes.loc[common]-reference).abs();added=quotes.loc[~quotes.index.isin(common)]
    if hashlib.sha256((ref_path/'qlib_store_manifest.json').read_bytes()).hexdigest()!=ref_manifest_before:raise ValueError('frozen provider changed')
    session_counts=quotes.reset_index().assign(date=lambda x:pd.to_datetime(x.datetime).dt.strftime('%Y-%m-%d')).groupby(['instrument','date']).size()
    if not session_counts.eq(4).all():raise ValueError('required point count differs')
    if not np.isfinite(quotes.to_numpy()).all():raise ValueError('nonfinite required price')
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='canonical_current_source_readback',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_binding=binding,source_manifest=(root/plan['source_manifest']).read_bytes(),provider_manifest=manifest,provider_files={str(p.relative_to(provider)):p.read_bytes() for p in provider.rglob('*') if p.is_file()})
        rec.save_objects(artifact_path='sources',loader=(root/'src/data/full_path_prices.py').read_text(),caller=(root/'src/workflow/full_path_tiered.py').read_text(),workflow=Path(__file__).read_text())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'canonical_core_price_layer_ready_context_lagged','objective_achieved':False,'new_fits':0,'new_backtests':0,'source_id':binding['source_id'],'source_schema':manifest['source_schema'],'core_as_of':plan['as_of'],'context_as_of':plan['context_as_of'],'all_model_inputs_updated':False,'native_price_rows':len(quotes),'overlap_price_rows':len(common),'new_price_rows':len(added),'maximum_overlap_price_difference':{k:float(v) for k,v in delta.max().items()},'overlap_prices_identical':bool(delta.eq(0).all().all()),'required_points_per_symbol_session':4,'observed_session_close_required':True,'unfinished_required_marks_rejected':True,'frozen_reference_provider_unchanged':True,'provider_uri':str(provider),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/data.html','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/quotes','analysis/overlap_differences']}}
        rec.save_objects(artifact_path='analysis',summary=summary,quotes=quotes,overlap_differences=delta,new_marks=added,session_counts=session_counts)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),new_price_rows=len(added))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    added.to_csv(card/'new_marks.csv');pd.DataFrame([{'field':k,'max_absolute_difference':v} for k,v in summary['maximum_overlap_price_difference'].items()]).to_csv(card/'overlap_audit.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

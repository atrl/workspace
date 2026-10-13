"""Qlib-Recorder cash signal audit; no new fit or strategy selection."""
import json,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from src.analysis.allocator_cash_information import cash_labels, diagnostic_tables
from src.workflow.minute_action_value import _source
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed signal plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    source,audit=_source(root,plan['source'],['inputs/frame'])
    predictions,pred_audit=_source(root,plan['action_source'],['inputs/raw_predictions'])
    frame=source['inputs/frame'];frame=frame.loc[frame.signal_date.between(plan['start'],plan['end'])].copy()
    daily=cash_labels(frame,horizons=plan['horizons'],tail_threshold=plan['tail_threshold']);names=[]
    for family in ['frozen_current','base']:
        pred=predictions['inputs/raw_predictions'][family].reindex(pd.DatetimeIndex(frame.signal_date))
        if not np.isfinite(pred.to_numpy(float)).all():raise ValueError('missing frozen score')
        for column in ['cash_weight','tail_probability']:
            name=family+'_'+column;daily[name]=pred[column].to_numpy(float);names.append(name)
    stats,intervals=diagnostic_tables(daily,score_names=names,plan=plan);tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='frozen_cash_information',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audit=audit,prediction_audit=pred_audit)
        rec.save_objects(artifact_path='sources',analysis=(root/'src/analysis/allocator_cash_information.py').read_text(),workflow=Path(__file__).read_text())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'diagnostic_only_no_promotion','objective_achieved':False,'new_fits':0,'new_backtests':0,'rows':len(daily),'mature_h1':int(daily.cash_advantage_h1.notna().sum()),'mature_h5':int(daily.cash_advantage_h5.notna().sum()),'tail_h5_count':int(daily.tail_h5.sum()),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/'data/qlib/full_path_tiered_allocation_20261013'),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/recorder.html','limits':plan['limits'],'tail_definition':'SOXX close or exit09:47 marks <=95% of entry09:47 price in next5 holding intervals; not peak-to-trough drawdown','artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/daily']}}
        rec.save_objects(artifact_path='analysis',daily=daily,statistics=stats,intervals=intervals,summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    stats.to_csv(card/'signal_statistics.csv',index=False);intervals.to_csv(card/'block_intervals.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

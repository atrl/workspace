"""Frozen model inference, projection accounting, and objective provenance."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.workflow.minute_action_value import _source
from src.model.minute_pit_allocator import _scaled
from src.workflow.pit_constituent_auxiliary_training_allocation import PitAuxiliaryAllocationNet, _project_legal_weights
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed geometry plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    source,audit=_source(root,plan['source'],['inputs/features','inputs/common','inputs/frozen_parent_models','inputs/plan'])
    predictions,pred_audit=_source(root,plan['prediction_source'],['inputs/raw_predictions','inputs/market'])
    source_plan=source['inputs/plan'];state=source['inputs/frozen_parent_models']['outer_refit']
    dates=pd.DatetimeIndex(predictions['inputs/market'].signal_date)
    dates=dates[(dates>=pd.Timestamp(plan['source_signal_start'])) & (dates<=pd.Timestamp(plan['source_signal_end']))]
    features=source['inputs/features'].reindex(dates);common=source['inputs/common'].reindex(dates)
    if not dates.is_unique or features[state['primary_features']].isna().any().any():raise ValueError('missing primary source input')
    net=PitAuxiliaryAllocationNet(primary_feature_count=len(state['primary_features']),common_feature_count=len(state['common_features']),hidden_units=source_plan['hidden_units'],tail_hidden_units=source_plan['tail_hidden_units'])
    net.load_state_dict({k:torch.tensor(v) for k,v in state['model_state'].items()});net.eval();torch.set_num_threads(1)
    x=_scaled(features,state['primary_features'],state['primary_scaler']);z=_scaled(common,state['common_features'],state['common_scaler'])
    floor=source_plan['minimum_cash_weight'];cap=source_plan['maximum_soxl_weight']
    with torch.no_grad():
        logits,tail,alpha=net(x,z)
        probability=torch.softmax(torch.clamp(logits,-8.,8.),dim=1)
        before=probability*(1-floor);before[:,2]+=floor
        after=_project_legal_weights(logits,maximum_soxl_weight=cap,minimum_cash_weight=floor)
        neutral_logits,_,_=net(torch.zeros((1,x.shape[1])),torch.zeros((1,z.shape[1])))
        neutral=_project_legal_weights(neutral_logits,maximum_soxl_weight=cap,minimum_cash_weight=floor)
    expected=predictions['inputs/raw_predictions']['frozen_current'].reindex(dates)
    observed=np.c_[after.numpy(),torch.sigmoid(tail).numpy(),torch.sigmoid(alpha).numpy()]
    np.testing.assert_allclose(observed,expected[['soxl_weight','soxx_weight','cash_weight','tail_probability','alpha_probability']].to_numpy(float),rtol=1e-6,atol=1e-7)
    daily=pd.DataFrame({'signal_date':dates,'cash_floor':floor,'softmax_cash':probability[:,2].numpy()*(1-floor),'cap_cash_increment':after[:,2].numpy()-before[:,2].numpy(),'final_cash':after[:,2].numpy(),'uncapped_soxl':before[:,0].numpy(),'cap_binding':before[:,0].numpy()>cap+1e-7,'soxl_excess':np.maximum(before[:,0].numpy()-cap,0),'final_soxl':after[:,0].numpy(),'final_soxx':after[:,1].numpy()})
    np.testing.assert_allclose(daily.cash_floor+daily.softmax_cash+daily.cap_cash_increment,daily.final_cash,atol=1e-7,rtol=0)
    rows=[]
    for year in ['all',*sorted(daily.signal_date.dt.year.unique())]:
        part=daily if year=='all' else daily.loc[daily.signal_date.dt.year.eq(year)]
        rows.append({'year':year,'sessions':len(part),**{k:100*part[k].mean() for k in ['cash_floor','softmax_cash','cap_cash_increment','final_cash','uncapped_soxl','soxl_excess','final_soxl','final_soxx']},'cap_binding_sessions':int(part.cap_binding.sum())})
    table=pd.DataFrame(rows);tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='frozen_cash_geometry',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audit=audit,prediction_audit=pred_audit,source_plan=source_plan,candidate=state['candidate'])
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),allocator=(root/'src/workflow/pit_constituent_auxiliary_training_allocation.py').read_text())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'diagnostic_only_no_promotion','objective_achieved':False,'new_fits':0,'new_backtests':0,'sessions':len(dates),'first_signal':str(dates.min()),'last_signal':str(dates.max()),'prediction_max_absolute_error':float(np.max(np.abs(observed-expected[['soxl_weight','soxx_weight','cash_weight','tail_probability','alpha_probability']].to_numpy(float)))),'standardized_zero_input_weights':neutral.numpy()[0].tolist(),'zero_logits_legal_weights':_project_legal_weights(torch.zeros((1,3)),maximum_soxl_weight=cap,minimum_cash_weight=floor).numpy()[0].tolist(),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/'data/qlib/full_path_tiered_allocation_20261013'),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/model.html','scope':'projection identity not causal value, zero-input is not cash optimality or a tradable control','artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/daily','analysis/logits']}}
        rec.save_objects(artifact_path='analysis',daily=daily,logits=logits.numpy(),comparison=table,summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    table.to_csv(card/'cash_geometry.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

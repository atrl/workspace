"""Recorder-only daily cash accounting with compact aggregate sidecars."""
import json,hashlib
from pathlib import Path
import pandas as pd
from src.analysis.cash_layers import trace_cash_layers, COMPONENTS
from src.workflow.minute_action_value import _source
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    source,audit=_source(root,plan['source'],['inputs/frame',*[f'analysis/native/cash_start_2023_current_anchor_{s}' for s in plan['slippage_bps']]])
    controller,control_audit=_source(root,plan['price_source'],['inputs/controller','inputs/plan'])
    action,action_audit=_source(root,plan['action_source'],['inputs/plan','inputs/raw_predictions','inputs/source_targets'])
    if controller['inputs/plan']['tariff']!=plan['tariff']:raise ValueError('fee source differs')
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);action_spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    frame=source['inputs/frame'];frame=frame.loc[frame.signal_date.between(plan['start'],plan['end'])]
    daily=pd.concat([trace_cash_layers(frame,source[f'analysis/native/cash_start_2023_current_anchor_{slip}'],controller=controller['inputs/controller'],recovery_spec=recovery,action_spec=action_spec,initial_cash=plan['initial_cash'],slippage_bps=slip) for slip in plan['slippage_bps']],ignore_index=True)
    raw=action['inputs/raw_predictions']['frozen_current'].reindex(pd.DatetimeIndex(frame.signal_date))
    smoothed=action['inputs/source_targets']['frozen_current'].reindex(pd.DatetimeIndex(frame.signal_date))
    import numpy as np
    np.testing.assert_allclose(smoothed.cash_weight.to_numpy(float),frame.cash_weight.to_numpy(float),rtol=0,atol=1e-12)
    if raw.cash_weight.isna().any():raise ValueError('missing raw model predictions')
    raw_by_trade=pd.Series(raw.cash_weight.to_numpy(float),index=pd.DatetimeIndex(frame.trade_date))
    daily['model_cash']=daily.trade_date.map(raw_by_trade)
    daily['source_smoothing_increment']=daily.raw_cash-daily.model_cash
    report_components=['model_cash','source_smoothing_increment',*COMPONENTS[1:]]
    np.testing.assert_allclose(daily[report_components].sum(axis=1),daily.actual_cash,rtol=0,atol=1e-12)
    rows=[]
    for slip,group in daily.groupby('slippage_bps'):
        for year in ['all',*sorted(group.trade_date.dt.year.unique())]:
            part=group if year=='all' else group.loc[group.trade_date.dt.year.eq(year)]
            rows.append({'slippage_bps':slip,'year':year,'sessions':len(part),**{k:100*part[k].mean() for k in [*report_components,'raw_cash','actual_cash']},**{k:int(part[k].sum()) for k in ['cycle_beta_binding','trailing_beta_binding','emergency_beta_binding']}})
    table=pd.DataFrame(rows);tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='cash_layer_accounting',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_native_cases=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audit=audit,controller_audit=control_audit,action_audit=action_audit)
        rec.save_objects(artifact_path='sources',analysis=(root/'src/analysis/cash_layers.py').read_text(),workflow=Path(__file__).read_text())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'diagnostic_only_no_promotion','objective_achieved':False,'new_fits':0,'new_native_cases':0,'reused_native_cases':3,'sessions_per_cost':len(frame),'maximum_cash_identity_error':float(daily.accounting_error.abs().max()),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/'data/qlib/full_path_tiered_allocation_20261013'),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/recorder.html','units':'mean cash weight percentage points; binding columns are session counts','scope':'ordered actual-state accounting; raw_cash means already-smoothed source target; model_cash is frozen pre-EMA output; not causal attribution or counterfactual performance','artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/daily']}}
        rec.save_objects(artifact_path='analysis',daily=daily,comparison=table,summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),maximum_cash_identity_error=summary['maximum_cash_identity_error'])
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    table.to_csv(card/'cash_layer_summary.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

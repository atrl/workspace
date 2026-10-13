"""Recorder-backed diagnostic; no candidate fit or account replay."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.analysis.decision_accounting import attribute_account
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.workflow.minute_action_value import _source


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest: raise ValueError('fixed diagnostic plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:
        raise ValueError('release changed')
    keys=[f'analysis/native/cash_start_2023_{f}_{s}' for f in [*plan['families'],'soxx'] for s in plan['slippage_bps']]
    source,audit=_source(root,plan['source'],['inputs/frame',*keys])
    tracking=card/'qlib_recorder'; aggregates=[]; years=[]; states=[]; max_error=0.
    with R.start(experiment_name=plan['experiment'],recorder_name='exact_decision_accounting',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audit=audit)
        rec.save_objects(artifact_path='sources',analysis=(root/'src/analysis/decision_accounting.py').read_text(),workflow=Path(__file__).read_text())
        for family in plan['families']:
            for slip in plan['slippage_bps']:
                result=attribute_account(source['inputs/frame'],source[f'analysis/native/cash_start_2023_{family}_{slip}'],source[f'analysis/native/cash_start_2023_soxx_{slip}'],initial_cash=plan['initial_cash'],cash_reference=plan['minimum_cash_reference'])
                components=['reference_cash','additional_cash','soxl_relative','recovery_gross','execution_cost','benchmark_residual','native_reconciliation_residual','relative_log']
                meta={'family':family,'slippage_bps':slip}
                aggregates.append({**meta,'sessions':len(result),'mean_cash':result.cash_weight_after_open.mean(),**result[components].sum().to_dict()})
                for year,part in result.groupby(pd.to_datetime(result.signal_date).dt.year):
                    years.append({**meta,'year':int(year),'sessions':len(part),'mean_cash':part.cash_weight_after_open.mean(),**part[components].sum().to_dict()})
                for state,part in result.groupby('risk_state'):
                    states.append({**meta,'risk_state':int(state),'sessions':len(part),'mean_cash':part.cash_weight_after_open.mean(),**part[components].sum().to_dict()})
                max_error=max(max_error,float(result.reconciliation_error.abs().max()))
                rec.save_objects(artifact_path='analysis/daily',**{f'{family}_{slip}':result})
        table,annual,state_table=map(pd.DataFrame,[aggregates,years,states])
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'diagnostic_only_no_promotion','objective_achieved':False,'new_fits':0,'new_backtests':0,'reused_native_accounts':9,'maximum_daily_return_reconciliation_error':max_error,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/'data/qlib/full_path_tiered_allocation_20261013'),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/recorder.html','artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/daily/*']},'interpretation':'Log-linked accounting components; not causal removable losses or feasible standalone returns. 15% reference cash is an attribution split, not a deployable counterfactual. Reused historical anchor is not paper fills.'}
        rec.save_objects(artifact_path='analysis',totals=table,annual=annual,states=state_table,summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),maximum_daily_return_reconciliation_error=max_error)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    for name,value in [('accounting_attribution',table),('annual_attribution',annual),('state_attribution',state_table)]:value.to_csv(card/(name+'.csv'),index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id))
    atomic_json(summary,card/'summary.json')
    return summary

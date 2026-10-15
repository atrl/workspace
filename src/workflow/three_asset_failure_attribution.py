"""Frozen Recorder-only inputs; no fit, account replay, or promotion authority."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.analysis.three_asset_failure_attribution import analyze
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.workflow.minute_action_value import _source


def run(*, root, card, plan):
    from qlib.data import D
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256'] != digest: raise ValueError('frozen diagnostic plan changed')
        return old
    costs=plan['slippage_bps']; objects={};audits={}
    requests={'tangent':['inputs/plan',*[f'analysis/native/{f}_{s}' for f in ['candidate','control','current_anchor','soxx'] for s in costs]],
              'router':['inputs/plan',*[f'analysis/native/{f}_{s}' for f in ['strong_always','strong_router'] for s in costs]],
              'risk':['inputs/plan','inputs/frame',*[f'analysis/native/{f}_{s}' for f in ['bypass','full_stack','soxx'] for s in costs]]}
    for key, source in plan['sources'].items():objects[key],audits[key]=_source(root,source,requests[key])
    baseline=objects['tangent']['inputs/plan']
    for key,data in objects.items():
        for name in ['initial_cash','tariff','start','end','as_of','slippage_bps']:
            if data['inputs/plan'][name]!=baseline[name]:raise ValueError('account context mismatch '+key+':'+name)
    for cost in costs:
        for key, name, base in [('router','strong_router','candidate'),('risk','full_stack','candidate'),('risk','soxx','soxx')]:
            for part in ['authority','orders','independent_replay']:
                pd.testing.assert_frame_equal(objects[key][f'analysis/native/{name}_{cost}'][part],objects['tangent'][f'analysis/native/{base}_{cost}'][part])
    frame=objects['risk']['inputs/frame'].reset_index(drop=True)
    # Live provider validates calendar only; all prices and features remain frozen.
    calendar=pd.DatetimeIndex(D.calendar(start_time=frame.trade_date.min(),end_time=frame.exit_date.max(),freq='day'))
    frozen_calendar=pd.DatetimeIndex([*frame.trade_date,frame.exit_date.iloc[-1]])
    calendar_audit={'live_only_dates':[str(x.date()) for x in calendar.difference(frozen_calendar)],
                    'frozen_only_dates':[str(x.date()) for x in frozen_calendar.difference(calendar)],
                    'live_provider_is_price_authority':False,'source_rows':len(frame)}
    if calendar_audit['live_only_dates'] or calendar_audit['frozen_only_dates']:
        raise ValueError('Qlib calendar differs from frozen intervals: '+str(calendar_audit))
    books={(family,cost):objects[source][f'analysis/native/{key}_{cost}'] for family,(source,key) in plan['families'].items() for cost in costs}
    benchmarks={cost:objects['tangent'][f'analysis/native/soxx_{cost}'] for cost in costs}
    results=analyze(frame,books,benchmarks,initial_cash=plan['initial_cash'],cash_reference=plan['cash_reference'],rolling_sessions=plan['rolling_sessions'],pairs=plan['pairs'])
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='frozen_cross_account_attribution',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audits=audits,calendar_audit=calendar_audit)
        rec.save_objects(artifact_path='sources',analysis=(root/'src/analysis/three_asset_failure_attribution.py').read_text(),accounting=(root/'src/analysis/decision_accounting.py').read_text(),workflow=Path(__file__).read_text())
        for name, table in results.items():
            if isinstance(table,dict):
                rec.save_objects(artifact_path='analysis/'+name,**{f'{f}_{c}':v for (f,c),v in table.items()})
            else:
                rec.save_objects(artifact_path='analysis',**{name:table});table.to_csv(card/(name+'.csv'),index=False)
        maximum_error=max(float(x.reconciliation_error.abs().max()) for x in results['daily'].values())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'diagnostic_only_no_promotion','objective_achieved':False,'new_fits':0,'new_backtests':0,'reused_native_accounts':18,'sessions':len(frame),'first_trade':str(frame.trade_date.min().date()),'last_exit':str(frame.exit_date.max().date()),'maximum_daily_reconciliation_error':maximum_error,'calendar_audit':calendar_audit,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/plan['provider_uri']),'region':'us','as_of':baseline['as_of'],'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/daily/*','analysis/rolling/*']},'source_identities':plan['sources'],'interpretation':plan['interpretation'],'period_basis':plan['period_basis']}
        rec.save_objects(artifact_path='analysis',summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts'
        rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')),maximum_daily_reconciliation_error=maximum_error)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id))
    atomic_json(summary,card/'summary.json')
    return summary

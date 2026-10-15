"""Fixed coupled-scenario policy, followed by three native ledger validations."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.workflow.beta_preserving_composition import verify_marks
from src.workflow.composition_objective_training import evaluate
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.backtest.causal_cost_scenario_controller import replay_cost_scenarios
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.analysis.three_asset_failure_attribution import analyze
from src.analysis.integrated_allocation import stationary_bootstrap_excess
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed coupled policy plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    costs=plan['slippage_bps']
    paired,pa=_source(root,plan['paired_control_source'],['inputs/frame','inputs/plan',*[f'analysis/native/candidate_{c}' for c in costs]])
    reference,ra=_source(root,plan['reference_source'],['inputs/plan',*[f'analysis/native/strong_always_{c}' for c in costs]])
    strong,sa=_source(root,plan['frozen_source'],['analysis/targets','inputs/plan',*[f'analysis/native/{f}_{c}' for f in ['candidate','current_anchor','soxx'] for c in costs]])
    controller,ca=_source(root,plan['price_source'],['inputs/controller','inputs/plan']);action,aa=_source(root,plan['action_source'],['inputs/plan','controller/source_recovery_selection'])
    for data in [paired,reference,strong]:
        for key in ['initial_cash','tariff','start','end','as_of','slippage_bps']:
            if data['inputs/plan'][key]!=plan[key]:raise ValueError('source context differs '+key)
    if controller['inputs/plan']['tariff']!=plan['tariff']:raise ValueError('controller tariff differs')
    frame=paired['inputs/frame'].reset_index(drop=True)
    targets=strong['analysis/targets']['candidate'].reindex(pd.DatetimeIndex(frame.signal_date))
    for col in targets:np.testing.assert_allclose(targets[col],frame[col],rtol=0,atol=1e-12)
    for cost in costs:pd.testing.assert_frame_equal(frame[['signal_date','trade_date','exit_date']],paired[f'analysis/native/candidate_{cost}']['authority'][['signal_date','trade_date','exit_date']].reset_index(drop=True))
    marks=verify_marks(frame,reference['analysis/native/strong_always_0'])
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':digest,'shadow_attempted':False,'shadow_completed':False,'native_attempts':[],'native':[]}
    if progress['plan_sha256']!=digest:raise ValueError('progress plan mismatch')
    tracking=card/'qlib_recorder';kw={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kw.update(recorder_id=progress['recorder_id'],resume=True)
    with R.start(**kw):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        if not progress.get('source_saved'):
            rec.save_objects(artifact_path='inputs',plan=plan,frame=frame,mark_verification=marks,source_audits={'paired':pa,'reference':ra,'tangent':sa,'controller':ca,'action':aa})
            rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),policy=(root/'src/backtest/causal_cost_scenario_controller.py').read_text(),original_controller=(root/'src/workflow/stage1_open_action_value.py').read_text(),risk=(root/'src/workflow/intraday_recovery_direct_allocation.py').read_text())
            progress['source_saved']=True;atomic_json(progress,progress_path)
        if progress['shadow_completed']:paths=rec.load_object('analysis/shadow_paths');trace=rec.load_object('analysis/scenario_trace')
        else:
            if progress['shadow_attempted']:raise ValueError('shadow attempt already used; no silent rerun')
            progress['shadow_attempted']=True;atomic_json(progress,progress_path)
            paths,trace=replay_cost_scenarios(frame,plan=plan,controller=controller['inputs/controller'],recovery_spec=recovery,action_spec=spec,selection=action['controller/source_recovery_selection'])
            rec.save_objects(artifact_path='analysis',shadow_paths=paths,scenario_trace=trace)
            progress['shadow_completed']=True;atomic_json(progress,progress_path)
    for c in costs[1:]:
        np.testing.assert_allclose(paths[c]['targets'][['soxl_weight','soxx_weight','cash_weight']],paths[costs[0]]['targets'][['soxl_weight','soxx_weight','cash_weight']],rtol=0,atol=0)
        np.testing.assert_array_equal(paths[c]['replay'].recovery_action,paths[costs[0]]['replay'].recovery_action)
    books={}
    for cost in costs:
        if cost in progress['native']:book=rec.load_object(f'analysis/native/candidate_{cost}')
        else:
            if cost in progress['native_attempts'] or len(progress['native_attempts'])>=plan['maximum_new_native_cases']:raise ValueError('native attempt budget exhausted')
            progress['native_attempts'].append(cost);atomic_json(progress,progress_path)
            book=native_tiered_stage1(frame,paths[cost],provider_uri=root/plan['provider_uri'],tariff=plan['tariff'],slippage_bps=cost,initial_cash=plan['initial_cash'],freq='1min',opening_offset=pd.Timedelta(hours=9,minutes=47),position_type=QlibContinuousPosition)
            book['targets']=paths[cost]['targets'];pd.testing.assert_index_equal(book['native_report'].index,reference[f'analysis/native/strong_always_{cost}']['native_report'].index)
            rec.save_objects(artifact_path='analysis/native',**{f'candidate_{cost}':book});progress['native'].append(cost);atomic_json(progress,progress_path)
        books[('candidate',cost)]=book
        for f,value in [('posthoc_composition',paired[f'analysis/native/candidate_{cost}']),('strong_always',reference[f'analysis/native/strong_always_{cost}']),('tangent',strong[f'analysis/native/candidate_{cost}']),('original',strong[f'analysis/native/current_anchor_{cost}'])]:books[(f,cost)]=value
        rec.log_metrics(completed_native_validations=len(progress['native']),completed_shadow_paths=len(costs));print('validated coupled scenario',cost,flush=True)
    benchmarks={c:strong[f'analysis/native/soxx_{c}'] for c in costs}
    results=analyze(frame,books,benchmarks,initial_cash=plan['initial_cash'],cash_reference=.15,rolling_sessions=126,pairs=[['candidate','posthoc_composition'],['candidate','strong_always']])
    table=results['accounting_attribution'];annual=results['annual_attribution'];rows=[]
    for cost in costs:
        for comparator in plan['bootstrap']['comparators']:
            other=benchmarks[cost] if comparator=='soxx' else books[(comparator,cost)]
            rows.append({'slippage_bps':cost,'comparator':comparator,**stationary_bootstrap_excess(books[('candidate',cost)]['authority'],other['authority'],**{k:v for k,v in plan['bootstrap'].items() if k!='comparators'})})
    uncertainty=pd.DataFrame(rows);gates=evaluate(table,annual,uncertainty,results['rolling']);historical=all(v for k,v in gates.items() if k not in ['independent_development','qualified_forward'])
    capdays=trace.groupby('trade_date').local_cap.agg(lambda v:v.max()-v.min()>1e-10)
    stats=trace.groupby('slippage_bps').agg(binding_days=('binding_scenario','sum'),recovery_eligible=('recovery_eligible','sum'),local_recovery_approvals=('recovery_local_approval','sum'),consensus_recoveries=('recovery_consensus','sum'),average_common_cap=('common_cap','mean')).reset_index()
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/shadow_paths','analysis/scenario_trace','analysis/native/*','analysis/daily/*','analysis/rolling/*']}
    summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'historical_candidate_requires_independent_forward' if historical else 'not_promoted_fixed_causal_cost_scenario_controller','objective_achieved':False,'new_fits':0,'new_shadow_policy_paths':3,'new_native_cases':3,'shared_target_path':True,'own_cost_nav_and_quantities':True,'cap_disagreement_days':int(capdays.sum()),'gates':gates,'provider_uri':str(root/plan['provider_uri']),'region':'us','as_of':plan['as_of'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'interpretation':plan['package']}
    for name,value in results.items():
        if isinstance(value,dict):rec.save_objects(artifact_path='analysis/'+name,**{f'{f}_{c}':v for (f,c),v in value.items()})
        else:value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
    for name,value in [('backtest_benchmark_comparison',table),('annual_metrics',annual),('uncertainty',uncertainty),('scenario_summary',stats),('mark_verification',marks)]:value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
    rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['paired_control_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

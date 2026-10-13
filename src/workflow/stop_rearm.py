"""Single fixed stop-rearm mechanism with native matched-cost accounts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.backtest.stop_rearm import advance_rearmed_state
from src.backtest.full_path_tiered import replay_full_path
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.analysis.integrated_allocation import paired_metrics, annual_paired_metrics, stationary_bootstrap_excess
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        previous=json.loads((card/'summary.json').read_text())
        if previous['plan_sha256']!=digest: raise ValueError('frozen plan changed')
        return previous
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']: raise ValueError('release changed')
    keys=[f'analysis/native/cash_start_2023_{f}_{s}' for f in ['current_anchor','soxx'] for s in plan['slippage_bps']]
    source,audit=_source(root,plan['source'],['inputs/frame',*keys])
    controller,control_audit=_source(root,plan['price_source'],['inputs/controller','inputs/plan'])
    action,action_audit=_source(root,plan['action_source'],['inputs/plan','controller/source_recovery_selection'])
    if controller['inputs/plan']['tariff']!=plan['tariff']: raise ValueError('costs differ')
    frame=source['inputs/frame'];frame=frame.loc[frame.signal_date.between(plan['start'],plan['end'])].copy()
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec'])
    action_spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    provider=root/'data/qlib/full_path_tiered_allocation_20261013';tracking=card/'qlib_recorder'
    diagnosis=[];left=source['analysis/native/cash_start_2023_current_anchor_2']['authority'];right=source['analysis/native/cash_start_2023_current_anchor_5']['authority']
    for field in ['execute_open_target','risk_state','risk_clock']:
        mask=left[field].ne(right[field]);first=int(np.flatnonzero(mask.to_numpy())[0])
        for slip,path in [(2,left),(5,right)]:
            for _,row in path.iloc[max(0,first-2):first+3].iterrows():
                diagnosis.append({'first_divergent_field':field,'first_divergence_date':str(left.iloc[first].trade_date),'different_sessions':int(mask.sum()),'slippage_bps':slip,**row[['trade_date','portfolio_drawdown','risk_state','risk_clock','current_beta_cap','execute_open_target','ending_cash_weight']].to_dict(),'stop_margin':row.portfolio_drawdown-recovery.stop_drawdown})
    with R.start(experiment_name=plan['experiment'],recorder_name='fixed_rearm_counterfactual',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,maximum_new_native_cases=3)
        rec.save_objects(artifact_path='inputs',plan=plan,frame=frame,source_audit=audit,controller_audit=control_audit,action_audit=action_audit,recovery_spec=action['inputs/plan']['recovery_spec'])
        rec.save_objects(artifact_path='sources',transition=(root/'src/backtest/stop_rearm.py').read_text(),replay=(root/'src/backtest/full_path_tiered.py').read_text(),open_controller=(root/'src/workflow/stage1_open_action_value.py').read_text(),workflow=Path(__file__).read_text())
    books={};rows=[];annuals=[];uncertainty=[];prefix=[]
    for slip in plan['slippage_bps']:
        original=source[f'analysis/native/cash_start_2023_current_anchor_{slip}'];benchmark=source[f'analysis/native/cash_start_2023_soxx_{slip}']
        replay=replay_full_path(frame,models=None,plan=plan,slippage_bps=slip,controller=controller['inputs/controller'],recovery_spec=recovery,action_spec=action_spec,selection=action['controller/source_recovery_selection'],state_transition=advance_rearmed_state)
        base=original['independent_replay'].reset_index(drop=True);new=replay['replay'].reset_index(drop=True)
        if not new.trade_date.equals(base.trade_date): raise ValueError('calendar differs')
        first=int(np.flatnonzero(new.risk_state.ne(base.risk_state).to_numpy())[0])
        np.testing.assert_allclose(new.strategy_return.iloc[:first],base.strategy_return.iloc[:first],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(new.risk_clock.iloc[:first],base.risk_clock.iloc[:first])
        prefix.append({'slippage_bps':slip,'identical_prefix_sessions':first,'first_candidate_state_change':str(new.iloc[first].trade_date)})
        book=native_tiered_stage1(frame,replay,provider_uri=provider,tariff=plan['tariff'],slippage_bps=slip,initial_cash=plan['initial_cash'],freq='1min',opening_offset=pd.Timedelta(hours=9,minutes=47),position_type=QlibContinuousPosition)
        book['targets']=replay['targets'];books[slip]=book
        for family,b in [('candidate',book),('original',original),('soxx',benchmark)]:
            path=b['authority'];relative=np.log1p(path.strategy_return)-np.log1p(benchmark['authority'].strategy_return)
            rows.append({'family':family,'slippage_bps':slip,**paired_metrics(path,benchmark['authority']),'monitored_drawdown_pct':-100*float(path.monitored_drawdown.max()),'average_cash_pct':100*path.ending_cash_weight.mean(),'positive_126d_fraction':relative.rolling(126).sum().dropna().gt(0).mean(),'original_relative_wealth':float(path.equity.iloc[-1]/original['authority'].equity.iloc[-1]),'stop_starts':int((path.risk_state.eq(1)&path.risk_clock.eq(1)).sum())})
            annuals.append(annual_paired_metrics(path,benchmark['authority']).assign(family=family,slippage_bps=slip))
            rec.save_objects(artifact_path='analysis/native',**{f'{family}_{slip}':b})
        uncertainty.append({'slippage_bps':slip,**stationary_bootstrap_excess(book['authority'],benchmark['authority'],mean_block=63,resamples=2000,alpha=.025,seed=20261014)})
        rec.log_metrics(completed_new_native_cases=len(books));print('completed',slip,flush=True)
    table=pd.DataFrame(rows);annual=pd.concat(annuals,ignore_index=True);uncertain=pd.DataFrame(uncertainty);candidate=table.loc[table.family.eq('candidate')]
    gates={'net_soxx_excess_all_costs':bool(candidate.relative_wealth.gt(1+1e-8).all()),'drawdown_30pct':bool(candidate.max_drawdown_pct.ge(-30).all() and candidate.monitored_drawdown_pct.ge(-30).all()),'cvar_no_worse':bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),'rolling_70pct':bool(candidate.positive_126d_fraction.ge(.7).all()),'annual_70pct':bool(annual.loc[annual.family.eq('candidate')].groupby('slippage_bps').relative_wealth.apply(lambda x:x.gt(1).mean()).ge(.7).all()),'bootstrap_positive':bool(uncertain.lower_bound_positive.all()),'beat_original_all_costs':bool(candidate.original_relative_wealth.gt(1+1e-8).all()),'independent_development':False,'qualified_forward':False}
    historical=all(v for k,v in gates.items() if k not in ['independent_development','qualified_forward'])
    summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'needs_independent_validation' if historical else 'not_promoted_fixed_rearm_failed','gates':gates,'objective_achieved':False,'new_fits':0,'new_native_cases':len(books),'reused_native_cases':6,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(provider),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/strategy.html','source_stop_drawdown':recovery.stop_drawdown,'candidate_prefix_checks':prefix,'risk_scope':'daily and known/close marks only, not full-minute maximum; no future guarantee','artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/native/*']}}
    rec.save_objects(artifact_path='analysis',summary=summary,comparison=table,annual=annual,uncertainty=uncertain,first_divergence=pd.DataFrame(diagnosis))
    artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    for name,value in [('backtest_benchmark_comparison',table),('annual_metrics',annual),('uncertainty',uncertain),('first_divergence',pd.DataFrame(diagnosis))]:value.to_csv(card/(name+'.csv'),index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json')
    return summary

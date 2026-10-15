"""One frozen composition mechanism with own-account state and native Qlib books."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.backtest.full_path_tiered import replay_full_path
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.model.nominal_beta_composition import AuditedComposition
from src.analysis.integrated_allocation import paired_metrics, stationary_bootstrap_excess
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


class CountRecoveryCalls:
    """Transparent prediction proxy: source calls only occur when eligible."""
    def __init__(self, model):self.model=model;self.rows=0
    def predict_proba(self, frame):
        self.rows+=len(frame)
        return self.model.predict_proba(frame)


def verify_marks(frame, reference):
    from qlib.data import D
    quotes=D.features(['SOXL','SOXX'],['$execution_price','$raw_execution_price'],
        start_time=frame.trade_date.min(),end_time=frame.exit_date.max()+pd.Timedelta(hours=16),freq='1min')
    checks=[]
    for symbol in ['SOXL','SOXX']:
        asset=quotes.xs(symbol,level='instrument');low=symbol.lower()
        for slot, dates in [('known',frame.trade_date+pd.Timedelta(hours=9,minutes=45)),('open',frame.trade_date+pd.Timedelta(hours=9,minutes=47)),('recovery',frame.trade_date+pd.Timedelta(hours=10,minutes=32)),('exit_open',frame.exit_date+pd.Timedelta(hours=9,minutes=47))]:
            for field,prefix in [('$execution_price',''),('$raw_execution_price','raw_')]:
                col=f'{prefix}known_{low}' if slot=='known' else f'{prefix}{low}_{slot}'
                actual=asset[field].reindex(pd.DatetimeIndex(dates)).to_numpy(float)
                np.testing.assert_array_equal(actual,frame[col].to_numpy(float))
                checks.append({'symbol':symbol,'slot':slot,'field':field,'rows':len(actual),'maximum_absolute_error':0.})
        close=asset.groupby(asset.index.normalize()).last()['$execution_price'].reindex(pd.DatetimeIndex(frame.trade_date)).to_numpy(float)
        np.testing.assert_array_equal(close,frame[f'close_{low}'].to_numpy(float))
        checks.append({'symbol':symbol,'slot':'close','field':'$execution_price','rows':len(close),'maximum_absolute_error':0.})
    expected=pd.DatetimeIndex(D.calendar(start_time=frame.trade_date.iloc[0]+pd.Timedelta(hours=9,minutes=47),end_time=frame.trade_date.iloc[-1]+pd.Timedelta(hours=16),freq='1min'))
    pd.testing.assert_index_equal(pd.DatetimeIndex(reference['native_report'].index),expected,check_names=False)
    return pd.DataFrame(checks)


def evaluate_gates(table,annual,uncertainty):
    candidate=table[table.family.eq('candidate')].set_index('slippage_bps');control=table[table.family.eq('strong_always')].set_index('slippage_bps')
    complete=annual[annual.family.eq('candidate') & annual.complete_period]
    if set(complete.slippage_bps)!=set(candidate.index) or any(set(g.period.astype(str))!={'2024','2025'} for _,g in complete.groupby('slippage_bps')):raise ValueError('missing complete trade-year evaluation')
    gates={'beat_paired_all_costs':bool(candidate.terminal_wealth.gt(control.terminal_wealth*(1+1e-8)).all()),'net_soxx_excess_all_costs':bool(candidate.relative_wealth.gt(1+1e-8).all()),'drawdown_30pct':bool(candidate.max_drawdown_pct.ge(-30).all() and candidate.monitored_drawdown_pct.ge(-30).all()),'cvar_no_worse':bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),'rolling_70pct':bool(candidate.positive_126d_fraction.ge(.7).all()),'complete_annual_70pct':bool(complete.groupby('slippage_bps').relative_wealth.apply(lambda v:v.gt(1+1e-8).mean()).ge(.7).all()),'bootstrap_positive':bool(uncertainty.lower_bound_positive.all()),'independent_development':False,'qualified_forward':False}
    return gates


def run(*,root,card,plan):
    import qlib,yaml
    from qlib.workflow import R
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('frozen composition plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    costs=plan['slippage_bps']
    strong,sa=_source(root,plan['frozen_source'],['inputs/plan','analysis/targets',*[f'analysis/native/{f}_{c}' for f in ['candidate','current_anchor','soxx'] for c in costs]])
    reference,ra=_source(root,plan['reference_source'],['inputs/plan',*[f'analysis/native/strong_always_{c}' for c in costs]])
    source,ma=_source(root,plan['source'],['inputs/frame'])
    controller,ca=_source(root,plan['price_source'],['inputs/controller','inputs/plan'])
    action,aa=_source(root,plan['action_source'],['inputs/plan','controller/source_recovery_selection'])
    for data in [strong,reference]:
        for key in ['initial_cash','tariff','start','end','as_of','slippage_bps']:
            if data['inputs/plan'][key]!=plan[key]:raise ValueError('frozen context differs '+key)
    if controller['inputs/plan']['tariff']!=plan['tariff']:raise ValueError('controller tariff differs')
    frame=source['inputs/frame'];frame=frame.loc[frame.signal_date.between(plan['start'],plan['end'])].copy().reset_index(drop=True)
    targets=strong['analysis/targets']['candidate'].reindex(pd.DatetimeIndex(frame.signal_date))
    if targets.isna().any().any():raise ValueError('missing frozen targets')
    for name in targets:frame[name]=targets[name].to_numpy()
    for cost in costs:
        pd.testing.assert_frame_equal(frame[['signal_date','trade_date','exit_date']],reference[f'analysis/native/strong_always_{cost}']['authority'][['signal_date','trade_date','exit_date']].reset_index(drop=True))
    marks=verify_marks(frame,reference['analysis/native/strong_always_0'])
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec'])
    if recovery.minimum_cash_weight!=plan['controller_projection_minimum_cash_weight'] or recovery.maximum_soxl_weight!=plan['controller_projection_maximum_soxl_weight']:raise ValueError('upstream projection changed')
    action_spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    books={};transforms={};eligible={};tracking=card/'qlib_recorder';provider=root/plan['provider_uri']
    # Native wrapper reinitializes Qlib: finish source reads before opening Recorder.
    with R.start(experiment_name=plan['experiment'],recorder_name='fixed_nominal_beta_composition',uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,frame=frame,source_audits={'strong':sa,'reference':ra,'market':ma,'controller':ca,'action':aa},mark_verification=marks)
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),composition=(root/'src/model/nominal_beta_composition.py').read_text(),replay=(root/'src/backtest/full_path_tiered.py').read_text())
        rec.log_params(plan_sha256=digest,maximum_new_native_cases=3,new_fits=0)
    for cost in costs:
        transform=AuditedComposition();modified=dict(controller['inputs/controller']);modified['frozen_open_action_value_model']=None
        model=modified['frozen_recovery_model'];counter=CountRecoveryCalls(model) if model is not None else None
        modified['frozen_recovery_model']=counter
        replay=replay_full_path(frame,models=None,plan=plan,slippage_bps=cost,controller=modified,recovery_spec=recovery,action_spec=action_spec,selection=action['controller/source_recovery_selection'],target_transform=transform)
        book=native_tiered_stage1(frame,replay,provider_uri=provider,tariff=plan['tariff'],slippage_bps=cost,initial_cash=plan['initial_cash'],freq='1min',opening_offset=pd.Timedelta(hours=9,minutes=47),position_type=QlibContinuousPosition)
        book['targets']=replay['targets'];book['target_transform']=pd.DataFrame(transform.rows)
        pd.testing.assert_index_equal(book['native_report'].index,reference[f'analysis/native/strong_always_{cost}']['native_report'].index)
        books[('candidate',cost)]=book;transforms[str(cost)]=pd.DataFrame(transform.rows);eligible[cost]=counter.rows if counter else 0
        rec.save_objects(artifact_path='analysis/native',**{f'candidate_{cost}':book});rec.log_metrics(completed_new_native_cases=len(transforms))
        print('completed composition native account',cost,flush=True)
    for cost in costs:
        books[('strong_always',cost)]=reference[f'analysis/native/strong_always_{cost}']
        for family,key in [('tangent','candidate'),('original','current_anchor'),('soxx','soxx')]:books[(family,cost)]=strong[f'analysis/native/{key}_{cost}']
    rows=[];years=[];quarters=[];intervals=[];composition=[];rolling={}
    for (family,cost),book in books.items():
        path=book['authority'].reset_index(drop=True);bench=books[('soxx',cost)]['authority'].reset_index(drop=True)
        metrics=paired_metrics(path,bench);relative=np.log1p(path.strategy_return)-np.log1p(bench.strategy_return)
        windows=relative.rolling(126).sum().dropna();rolling[f'{family}_{cost}']=pd.DataFrame({'first_trade':path.trade_date.iloc[:len(windows)].to_numpy(),'last_exit':path.exit_date.iloc[125:].to_numpy(),'relative_log':windows.to_numpy()})
        rows.append({'family':family,'slippage_bps':cost,**metrics,'monitored_drawdown_pct':-100*path.monitored_drawdown.max(),'positive_126d_fraction':windows.gt(0).mean(),'mean_ending_nominal_beta':float((3*path.ending_soxl_weight+path.ending_soxx_weight).mean()),'realized_regression_beta':float(np.cov(path.strategy_return,bench.strategy_return,ddof=1)[0,1]/np.var(bench.strategy_return,ddof=1)),'mean_ending_cash_pct':100*path.ending_cash_weight.mean(),'mean_ending_soxl_pct':100*path.ending_soxl_weight.mean(),'broker_fees_usd':path.broker_and_external_fees_usd.sum(),'slippage_usd':path.slippage_reserve_usd.sum(),'recovery_triggers':int(path.recovery_action.sum())})
        for freq,destination in [('Y',years),('Q',quarters)]:
            for period,part in path.groupby(path.trade_date.dt.to_period(freq)):
                complete=period not in [path.trade_date.iloc[0].to_period(freq),path.trade_date.iloc[-1].to_period(freq)]
                destination.append({'family':family,'slippage_bps':cost,'period':str(period),'complete_period':complete,**paired_metrics(part,bench.loc[part.index])})
        if family=='candidate':
            audit=transforms[str(cost)];composition.append({'slippage_bps':cost,'transformed_requests':len(audit),'changed_requests':int(audit.before_soxl.ne(audit.after_soxl).sum()),'maximum_beta_error':audit.beta_error.abs().max(),'maximum_simplex_error':audit.simplex_error.abs().max(),'recovery_eligible_calls':eligible[cost],'recovery_triggers':int(path.recovery_action.sum())})
            for comparator in plan['bootstrap']['comparators']:
                args={k:v for k,v in plan['bootstrap'].items() if k!='comparators'}
                intervals.append({'slippage_bps':cost,'comparator':comparator,**stationary_bootstrap_excess(path,books[(comparator,cost)]['authority'],**args)})
    table=pd.DataFrame(rows);annual=pd.DataFrame(years);quarterly=pd.DataFrame(quarters);uncertainty=pd.DataFrame(intervals);comp=pd.DataFrame(composition)
    gates=evaluate_gates(table,annual,uncertainty)
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/native/*','analysis/rolling/*','analysis/transform/*']}
    summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':('historical_candidate_requires_independent_forward' if all(v for k,v in gates.items() if k not in ['independent_development','qualified_forward']) else 'not_promoted_fixed_nominal_beta_composition'),'objective_achieved':False,'new_fits':0,'new_native_cases':3,'gates':gates,'provider_uri':str(provider),'region':'us','as_of':plan['as_of'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'scope':plan['mechanism'],'limitations':'Reused history; only2024/2025 complete trade years. Nominal beta is not actual risk equivalence. Recovery classifier can encounter changed state distribution; no retuning.'}
    for name,value in [('backtest_benchmark_comparison',table),('annual_metrics',annual),('quarterly_metrics',quarterly),('uncertainty',uncertainty),('composition_audit',comp),('mark_verification',marks)]:
        value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
    rec.save_objects(artifact_path='analysis/rolling',**rolling);rec.save_objects(artifact_path='analysis/transform',**transforms);rec.save_objects(artifact_path='analysis',summary=summary)
    artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json')
    return summary

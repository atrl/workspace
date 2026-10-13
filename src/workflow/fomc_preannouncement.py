"""A fixed event-component timing screen, never a stable-excess promotion claim."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.data.fomc_event_prices import collect_clocks,event_price_store
from src.backtest.fomc_event_book import replay_events
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.analysis.integrated_allocation import paired_metrics,annual_paired_metrics
from src.workflow.fed_capacity_information import _bootstrap
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    import yaml
    from qlib.workflow import R
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed event plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    source,schedule_audit=_source(root,plan['schedule_source'],['analysis/usable','sources/reference_calendar'])
    prices,price_audit=_source(root,plan['price_source'],['inputs/frame','inputs/plan'])
    if prices['inputs/plan']['tariff']!=plan['tariff']:raise ValueError('tariff differs')
    schedule=source['analysis/usable'].sort_values('next_meeting_end');calendar=pd.DatetimeIndex(source['sources/reference_calendar'])
    clocks,documents=collect_clocks(root/'data/raw/external'/plan['experiment'],schedule.next_meeting_end)
    rows=[]
    for row in schedule.itertuples():
        day=pd.Timestamp(row.next_meeting_end);position=calendar.get_loc(day);release=pd.Timestamp(clocks[str(day.date())]['release_at'])
        exit_stamp=(day+pd.Timedelta(hours=13,minutes=59)).tz_localize('America/New_York')
        if exit_stamp>=release:raise ValueError('fixed exit is not before official release; no event may be dropped')
        for timing,shift in [('event',0),('placebo',plan['placebo_shift_sessions'])]:
            entry,exit_day=calendar[position-1-shift],calendar[position-shift]
            if pd.Timestamp(row.release_date)+pd.Timedelta(hours=23,minutes=59,seconds=59)>=entry+pd.Timedelta(hours=9,minutes=46):raise ValueError('schedule not known before matched entry')
            rows.append({'event_id':str(day.date()),'timing':timing,'entry_date':entry,'exit_date':exit_day,'prior_minutes_released':row.release_date,'official_release_at':release.isoformat()})
    windows=pd.DataFrame(rows)
    for _,w in windows.groupby('timing'):
        if w.entry_date.duplicated().any() or w.exit_date.duplicated().any() or w.entry_date.isin(w.exit_date).any():raise ValueError('overlapping fixed windows')
    frame=prices['inputs/frame'];frame=frame.loc[frame.trade_date.ge(plan['start'])].copy()
    if not windows.entry_date.isin(frame.trade_date).all() or not windows.exit_date.isin(frame.trade_date).all():raise ValueError('event outside complete price frame')
    provider=root/'data/qlib'/plan['experiment']
    native,store=event_price_store(root=root,plan=plan,exit_dates=windows.exit_date.unique(),provider=provider)
    for symbol in ['SOXL','SOXX']:
        asset=native.xs(symbol,level='instrument');lower=symbol.lower()
        np.testing.assert_allclose(asset['$execution_price'].reindex(pd.DatetimeIndex(frame.trade_date)+pd.Timedelta(hours=9,minutes=47)),frame[lower+'_open'],rtol=0,atol=1e-7)
        for field,prefix in [('$execution_price',''),('$raw_execution_price','raw_')]:frame[prefix+lower+'_event_exit']=asset[field].reindex(pd.DatetimeIndex(frame.trade_date)+pd.Timedelta(hours=13,minutes=59)).to_numpy(float)
    tracking=card/'qlib_recorder';progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':digest,'completed':[]}
    if progress['plan_sha256']!=digest:raise ValueError('event progress identity differs')
    kwargs={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kwargs.update(recorder_id=progress['recorder_id'],resume=True)
    with R.start(**kwargs):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        rec.log_params(plan_sha256=digest,maximum_new_native_cases=plan['maximum_new_native_cases'],new_fits=0)
        rec.save_objects(artifact_path='inputs',plan=plan,frame=frame,windows=windows,schedule_audit=schedule_audit,price_audit=price_audit,provider_manifest=store,provider_files={str(p.relative_to(provider)):p.read_bytes() for p in provider.rglob('*') if p.is_file()})
        rec.save_objects(artifact_path='sources',statement_manifest=clocks,statement_html=documents,workflow=Path(__file__).read_text(),prices=(root/'src/data/fomc_event_prices.py').read_text(),replay=(root/'src/backtest/fomc_event_book.py').read_text())
    books={};cases=[(timing,family,slip) for timing in ['event','placebo'] for family in ['reference','tilt'] for slip in plan['slippage_bps']]+[('all','soxx',slip) for slip in plan['slippage_bps']]
    for timing,family,slip in cases:
        key=f'{timing}_{family}_{slip}'
        if key in progress['completed']:book=rec.load_object('analysis/native/'+key)
        else:
            if len(progress['completed'])>=plan['maximum_new_native_cases']:raise ValueError('native account budget exhausted')
            w=windows.loc[windows.timing.eq(timing)] if timing!='all' else windows.loc[windows.timing.eq('event')]
            replay=replay_events(frame,w,weights=np.array(plan['weights'].get(family,[0,1,0])),plan=plan,slippage_bps=slip,benchmark=family=='soxx')
            book=native_tiered_stage1(frame,replay,provider_uri=provider,tariff=plan['tariff'],slippage_bps=slip,initial_cash=plan['initial_cash'],freq='1min',opening_offset=pd.Timedelta(hours=9,minutes=47),position_type=QlibContinuousPosition)
            book['events']=replay['events'];rec.save_objects(artifact_path='analysis/native',**{key:book});progress['completed'].append(key);atomic_json(progress,progress_path)
        books[(timing,family,slip)]=book;rec.log_metrics(completed_new_native_cases=len(progress['completed']));print('account',key,flush=True)
    comparison=[];annual=[];advantages=[];gates=[]
    for (timing,family,slip),book in books.items():
        path=book['authority'];benchmark=books[('all','soxx',slip)]['authority']
        comparison.append({'timing':timing,'family':family,'slippage_bps':slip,**paired_metrics(path,benchmark),'monitored_drawdown_pct':-100*path.monitored_drawdown.max(),'orders':len(book['orders']),'broker_fees_usd':path.broker_and_external_fees_usd.sum(),'slippage_usd':path.slippage_reserve_usd.sum(),'average_cash_pct':100*path.ending_cash_weight.mean()})
        annual.append(annual_paired_metrics(path,benchmark).assign(timing=timing,family=family,slippage_bps=slip))
    for slip in plan['slippage_bps']:
        values={}
        for timing in ['event','placebo']:
            left=books[(timing,'tilt',slip)]['events'].set_index('event_id').event_net_log_return
            right=books[(timing,'reference',slip)]['events'].set_index('event_id').event_net_log_return
            if not left.index.equals(right.index):raise ValueError('paired episodes differ')
            values[timing]=left-right
        difference=values['event']-values['placebo']
        for window,mask in [('development',pd.to_datetime(difference.index)<=pd.Timestamp(plan['development_end'])),('reused_outer',pd.to_datetime(difference.index)>pd.Timestamp(plan['development_end']))]:
            d=difference.loc[mask];boot=_bootstrap(d.to_numpy(),plan)
            boot['mean_net_timing_log_advantage']=boot.pop('mean_joint_normalized_loss_improvement');boot['method']='paired stationary bootstrap of event-minus-placebo net log advantage'
            year=d.groupby(pd.to_datetime(d.index).year).mean();row={'window':window,'slippage_bps':slip,'event_tilt_minus_reference_mean':float(values['event'].loc[mask].mean()),'placebo_tilt_minus_reference_mean':float(values['placebo'].loc[mask].mean()),'positive_year_fraction':float(year.gt(0).mean()),**boot};advantages.append(row)
            if window=='development':gates.append(row['event_tilt_minus_reference_mean']>0 and boot['lower_bound']>0 and row['positive_year_fraction']>=.75)
        rec.save_objects(artifact_path='analysis/events',**{f'paired_{slip}':pd.DataFrame({'event_advantage':values['event'],'placebo_advantage':values['placebo'],'timing_advantage':difference})})
    summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'component_pass_needs_full_strategy_test' if all(gates) else 'not_promoted_fixed_fomc_event_screen_failed','component_gate_pass':bool(all(gates)),'objective_achieved':False,'new_fits':0,'new_native_cases':len(progress['completed']),'scheduled_events':len(schedule),'all_official_statement_clocks_after_exit':True,'full_strategy_integrated':False,'independent_forward_qualified':False,'risk_monitoring':'daily native and known/exit/close marks, not complete-minute peak-to-trough','provider_uri':str(provider),'region':'us','as_of':plan['as_of'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/strategy.html','artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/native/*','analysis/events/*']}}
    table=pd.DataFrame(comparison);annual=pd.concat(annual,ignore_index=True);adv=pd.DataFrame(advantages)
    rec.save_objects(artifact_path='analysis',summary=summary,comparison=table,annual=annual,advantages=adv)
    artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    table.to_csv(card/'backtest_benchmark_comparison.csv',index=False);annual.to_csv(card/'annual_metrics.csv',index=False);adv.to_csv(card/'timing_increment.csv',index=False);windows.to_csv(card/'event_windows.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

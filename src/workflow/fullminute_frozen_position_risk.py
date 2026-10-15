"""Native frozen-position risk evidence on pinned full-minute source marks."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.data.full_path_prices import _source_frames
from src.data.minute_price_basis import published_intraday_basis
from src.data.qlib_native import build_qlib_store_from_frame,load_qlib_feature_frame,QlibProviderConfig
from src.analysis.fullminute_frozen_position_risk import holding_events,value_path
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def quotes_from_source(root,plan,frame):
    import exchange_calendars as xc
    source=root/plan['minute_source_manifest']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=plan['minute_source_id']:raise ValueError('frozen minute source changed')
    receipt=json.loads(source.read_text());raw,events=_source_frames(source,receipt,as_of=plan['as_of'],workspace_root=root)
    start,end=frame.trade_date.min(),frame.exit_date.max()
    schedule=xc.get_calendar('XNYS',start=start,end=end).schedule
    expected=pd.DatetimeIndex(np.concatenate([pd.date_range(row.open+pd.Timedelta(minutes=1),row.close,freq='1min').to_numpy() for row in schedule.itertuples()])).tz_convert('America/New_York')
    parts=[];coverage=[]
    for symbol,data in raw.items():
        data=data.loc[data.date.between(str(start.date()),str(end.date()))].copy();index=pd.DatetimeIndex(data.timestamp)
        if index.duplicated().any() or len(expected.difference(index)) or len(index.difference(expected)) or not data.price_valid.all() or not data.is_final.all():raise ValueError('needs_data: full source calendar/quality incomplete')
        if len(data)!=367110:raise ValueError('source calendar count changed')
        coverage.append({'symbol':symbol,'rows':len(data),'missing':0,'extra':0,'invalid':0,'unfinished':0});parts.append(data)
    adjusted,basis=published_intraday_basis(pd.concat(parts,ignore_index=True),events,as_of=plan['as_of'])
    adjusted['instrument']=adjusted.symbol;adjusted['datetime']=adjusted.timestamp.dt.tz_convert('America/New_York').dt.tz_localize(None)
    fields=['open','high','low','close','raw_open','raw_close']
    provider=root/plan['provider_uri']
    manifest=build_qlib_store_from_frame(adjusted,provider_uri=provider,freq='1min',as_of=plan['as_of'],field_map={x:x for x in fields},source='Frozen NAS full-minute published-affine risk revaluation; '+plan['minute_source_id'])
    quotes=load_qlib_feature_frame(QlibProviderConfig(provider,freq='1min'),instruments=['SOXL','SOXX'],fields=['$'+x for x in fields],start=str(start.date()),end=str(end.date())+' 16:00:00')
    wide=[]
    for symbol in ['SOXL','SOXX']:
        expected_values=adjusted.loc[adjusted.symbol.eq(symbol)].set_index('datetime')[fields].sort_index().astype(np.float32)
        observed=quotes.xs(symbol,level='instrument').sort_index();np.testing.assert_array_equal(observed.to_numpy(),expected_values.to_numpy())
        observed.columns=[symbol+'_'+x for x in fields];wide.append(observed)
    quotes=pd.concat(wide,axis=1)
    anchor_checks=[]
    for symbol in ['SOXL','SOXX']:
        low=symbol.lower()
        for slot,when,field in [('known',frame.trade_date+pd.Timedelta(hours=9,minutes=45),'close'),('open',frame.trade_date+pd.Timedelta(hours=9,minutes=48),'open'),('recovery',frame.trade_date+pd.Timedelta(hours=10,minutes=33),'open'),('exit_open',frame.exit_date+pd.Timedelta(hours=9,minutes=48),'open')]:
            for prefix in ['', 'raw_']:
                column=f'{prefix}known_{low}' if slot=='known' else f'{prefix}{low}_{slot}'
                actual=quotes[f'{symbol}_{prefix}{field}'].reindex(pd.DatetimeIndex(when)).to_numpy(float)
                np.testing.assert_array_equal(actual,frame[column].to_numpy(float));anchor_checks.append({'symbol':symbol,'slot':slot,'plane':prefix or 'affine','rows':len(actual),'exact':True})
        close_times=pd.DatetimeIndex([pd.Timestamp(x).tz_convert('America/New_York').tz_localize(None) for x in schedule.close])
        byday=pd.Series(close_times,index=close_times.normalize())
        actual=quotes[f'{symbol}_close'].reindex(pd.DatetimeIndex(frame.trade_date.map(byday))).to_numpy(float)
        np.testing.assert_array_equal(actual,frame['close_'+low].to_numpy(float));anchor_checks.append({'symbol':symbol,'slot':'close','plane':'affine','rows':len(actual),'exact':True})
    manifest.update(source_manifest_sha256=plan['minute_source_id'],price_basis=basis,clock='RIGHT-edge sourceOHLC; not sparse execution marks',large_bins_git_delivery=False)
    atomic_json(manifest,provider/'risk_source_manifest.json')
    return quotes,pd.DataFrame(coverage),pd.DataFrame(anchor_checks),manifest


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        prior=json.loads((card/'summary.json').read_text())
        if prior['plan_sha256']!=digest:raise ValueError('frozen risk plan changed')
        return prior
    candidate,ca=_source(root,plan['sources']['candidate'],plan['source_artifacts']['candidate'])
    benchmark,ba=_source(root,plan['sources']['benchmark'],plan['source_artifacts']['benchmark'])
    frame=candidate['inputs/frame'].reset_index(drop=True)
    books={(family,cost):(candidate[f'analysis/native/candidate_{cost}'] if family=='candidate' else benchmark[f'analysis/native/soxx_{cost}']) for family in plan['families'] for cost in plan['slippage_bps']}
    quotes,coverage,anchors,provider_manifest=quotes_from_source(root,plan,frame)
    # Finish all source/funding validations for all six books before risk metrics.
    events={};funding=[]
    for key,book in books.items():
        pd.testing.assert_frame_equal(frame[['signal_date','trade_date','exit_date']],book['authority'][['signal_date','trade_date','exit_date']].reset_index(drop=True))
        events[key],audit=holding_events(book,frame,quotes,initial_cash=plan['initial_cash'],tolerance=plan['reconciliation_tolerance']['native_relative'],fee_tolerance=plan['reconciliation_tolerance']['fees_absolute_usd'])
        funding.append(audit.assign(family=key[0],slippage_bps=key[1]))
    funding=pd.concat(funding,ignore_index=True)
    known=frame[['known_soxl','known_soxx']].rename(columns={'known_soxl':'SOXL','known_soxx':'SOXX'});known.index=pd.DatetimeIndex(frame.trade_date+pd.Timedelta(hours=9,minutes=45))
    tracking=card/'qlib_recorder';rows=[];episodes=[]
    with R.start(experiment_name=plan['experiment'],recorder_name='frozen_native_fullminute_revaluation',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_policy_paths=0,new_native_accounts=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audits={'candidate':ca,'benchmark':ba},provider_manifest=provider_manifest,fullminute_quotes=quotes,funding_audit=funding)
        rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),valuation=(root/'src/analysis/fullminute_frozen_position_risk.py').read_text(),source_basis=(root/'src/data/minute_price_basis.py').read_text())
        for (family,cost),book in books.items():
            path,metrics,episode=value_path(quotes,events[(family,cost)],known,start=plan['valuation_start'],end=plan['valuation_end'],initial_cash=plan['initial_cash'])
            original=float(book['authority'].monitored_drawdown.max())
            if metrics['observed_drawdown']+plan['reconciliation_tolerance']['native_relative']<original:raise ValueError('augmented observed marks fail to enclose parent monitored drawdown')
            terminal=float(book['authority'].equity.iloc[-1]*plan['initial_cash'])
            if abs(metrics['terminal_nav']-terminal)/terminal>plan['reconciliation_tolerance']['native_relative']:raise ValueError('terminal frozen NAV differs')
            rows.append({'family':family,'slippage_bps':cost,'parent_sampled_drawdown':original,**metrics});episodes.append(episode.assign(family=family,slippage_bps=cost))
            rec.save_objects(artifact_path='analysis/minute',**{f'{family}_{cost}':path});rec.save_objects(artifact_path='analysis/holdings',**{f'{family}_{cost}':events[(family,cost)]})
            rec.log_metrics(completed_valuations=len(rows));print('completed frozen valuation',family,cost,flush=True)
        table=pd.DataFrame(rows);episode_table=pd.concat(episodes,ignore_index=True)
        prior={'method':'parent_daily/known09:45_and_close_sampled_drawdown','source_recorder_id':plan['sources']['candidate']['recorder_id'],'original_gates':candidate['analysis/summary']['gates'],'unchanged_parent_decision':candidate['analysis/summary']['decision']}
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/minute/*','analysis/holdings/*']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'risk_revaluation_only_no_promotion','objective_achieved':False,'new_fits':0,'new_policy_paths':0,'new_native_accounts':0,'frozen_valuations':6,'provider_uri':str(root/plan['provider_uri']),'region':'us','as_of':plan['as_of'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'risk_methods':plan['risk_methods'],'qualification_scope':plan['risk_qualification']}
        rec.save_objects(artifact_path='analysis',risk_summary=table,worst_episodes=episode_table,coverage=coverage,anchors=anchors,prior_gate_snapshot=prior,summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    for name,value in [('risk_summary',table),('worst_episodes',episode_table),('source_coverage',coverage),('anchor_readback',anchors),('funding_summary',funding.groupby(['family','slippage_bps','kind']).relative_error.agg(['max','count']).reset_index())]:value.to_csv(card/(name+'.csv'),index=False)
    atomic_json(prior,card/'prior_gate_snapshot.json');atomic_json(provider_manifest,card/'provider_manifest.json')
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['sources']['candidate']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

"""Nine fixed own-raw-NAV policy paths and native Qlib validations; no fitting."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.core.recorder_sources import load_verified_recorder_objects
from src.data import QlibProviderConfig,build_qlib_store_from_frame
from src.backtest.raw_corporate_policy import make_schedule,replay_raw_package
from src.backtest.native_raw_corporate import native_raw_validate
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.analysis.raw_cashbook_assessment import analyze_raw
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    import qlib,yaml
    from qlib.data import D
    from qlib.workflow import R
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest();assert digest=='5a1c72285db384e01646c27f9bca491212dcf0095d5ec347fa49068fafddf68c'
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    assert hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()==plan['release_sha256']
    costs=plan['slippage_bps'];candidate,ca=_source(root,plan['candidate_source'],plan['source_artifacts']['candidate']);strong,sa=_source(root,plan['strong_source'],plan['source_artifacts']['strong'])
    src=plan['corporate_source'];corporate,cca=load_verified_recorder_objects(root/src['card'],expected_recorder_id=src['recorder_id'],expected_manifest_sha256=src['inventory_sha256'],artifacts=plan['source_artifacts']['corporate'])
    unit,ua=_source(root,plan['unit_source'],plan['source_artifacts']['unit']);assert unit['analysis/summary']['all_unit_invariants_passed']
    controller,cta=_source(root,plan['price_source'],['inputs/controller','inputs/plan']);action,aa=_source(root,plan['action_source'],['inputs/plan','controller/source_recovery_selection'])
    benchmark,bma=_source(root,plan['tangent_source'],[f'analysis/native/soxx_{c}' for c in costs])
    frame=candidate['inputs/frame'].reset_index(drop=True).copy();pd.testing.assert_frame_equal(frame[['signal_date','trade_date','exit_date']],strong['inputs/frame'][['signal_date','trade_date','exit_date']].reset_index(drop=True))
    assert len(frame)==944 and plan['tariff']==candidate['inputs/plan']['tariff']==strong['inputs/plan']['tariff']==controller['inputs/plan']['tariff']
    original_provider=root/plan['market_source']['provider_uri'];assert hashlib.sha256((original_provider/'qlib_store_manifest.json').read_bytes()).hexdigest()==plan['market_source']['manifest_sha256']
    qlib.init(provider_uri=str(original_provider),region='us',kernels=1)
    first=pd.Timestamp(frame.trade_date.iloc[0])+pd.Timedelta('9h45min');terminal=pd.Timestamp(frame.exit_date.iloc[-1])+pd.Timedelta('9h47min')
    cached_progress=json.loads((card/'run_state.json').read_text()) if (card/'run_state.json').exists() else {}
    if cached_progress.get('sources_saved'):
        with R.uri_context(str(card/'qlib_recorder')):
            oldrec=R.get_recorder(experiment_name=plan['experiment'],recorder_id=cached_progress['recorder_id'])
            points=oldrec.load_object('inputs/raw_points')
    else:
        quotes=D.features(['SOXL','SOXX'],['$execution_price','$raw_execution_price'],start_time=first,end_time=terminal,freq='1min')
        points=quotes['$raw_execution_price'].unstack('instrument')[['SOXL','SOXX']].astype(float).sort_index()
        assert len(points)==4*len(frame)+2 and np.isfinite(points).all().all() and points.gt(0).all().all()
        for symbol in ['SOXL','SOXX']:
            a=quotes.xs(symbol,level='instrument');low=symbol.lower()
            for slot,dates in [('known',frame.trade_date+pd.Timedelta('9h45min')),('open',frame.trade_date+pd.Timedelta('9h47min')),('recovery',frame.trade_date+pd.Timedelta('10h32min')),('exit_open',frame.exit_date+pd.Timedelta('9h47min'))]:
                for field,prefix in [('$execution_price',''),('$raw_execution_price','raw_')]:
                    column=f'{prefix}known_{low}' if slot=='known' else f'{prefix}{low}_{slot}'
                    np.testing.assert_array_equal(a[field].reindex(pd.DatetimeIndex(dates)).to_numpy(float),frame[column].to_numpy(float))
            closing=a.groupby(a.index.normalize()).last()['$execution_price'].reindex(pd.DatetimeIndex(frame.trade_date))
            np.testing.assert_array_equal(closing.to_numpy(float),frame['close_'+low].to_numpy(float))
    sessions=pd.DatetimeIndex(points.index.normalize().unique());schedule=make_schedule(corporate['data/cash_events.pkl'],corporate['data/split_events.pkl'],sessions,frame.trade_date.iloc[0],frame.exit_date.iloc[-1])
    raw_rows=[]
    for symbol in ['SOXL','SOXX']:
        for stamp,value in points[symbol].items():raw_rows.append({'datetime':stamp,'instrument':symbol,**{name:value for name in ['open','high','low','close','execution_price','raw_execution_price']}})
    # Native TradeCalendar requires a following bound. This existing source point
    # is provider padding only; point_prices/backtest/analysis still end09:47.
    guard_time=terminal+pd.Timedelta(minutes=45)
    guard=D.features(['SOXL','SOXX'],['$raw_execution_price'],start_time=guard_time,end_time=guard_time,freq='1min')
    assert len(guard)==2 and guard['$raw_execution_price'].gt(0).all()
    for symbol in ['SOXL','SOXX']:
        value=float(guard.xs(symbol,level='instrument').iloc[0,0])
        raw_rows.append({'datetime':guard_time,'instrument':symbol,**{name:value for name in ['open','high','low','close','execution_price','raw_execution_price']}})
    provider=root/plan['market_source']['new_provider'];manifest=build_qlib_store_from_frame(pd.DataFrame(raw_rows),provider_uri=provider,freq='1min',region='us',as_of=plan['as_of'],field_map={x:x for x in ['open','high','low','close','execution_price','raw_execution_price']},source='Fixed original raw execution/valuation point marks; corporate cash separate, USD gross paper assumption; no new market source')
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':digest,'path_attempts':[],'paths_done':[],'native_attempts':[],'native_done':[]}
    assert progress['plan_sha256']==digest
    tracking=card/'qlib_recorder';kw={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kw.update(recorder_id=progress['recorder_id'],resume=True)
    with R.start(**kw):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        if not progress.get('sources_saved'):
            rec.save_objects(artifact_path='inputs',plan=plan,frame=frame,raw_points=points,schedule=schedule,raw_provider_manifest=manifest,source_audits={'candidate':ca,'strong':sa,'corporate':cca,'unit':ua,'controller':cta,'action':aa,'benchmark':bma})
            files=['src/workflow/frozen_policy_raw_cashbook.py','src/backtest/raw_corporate_policy.py','src/backtest/native_raw_corporate.py','src/analysis/raw_cashbook_assessment.py','src/backtest/ibkr_tiered.py','src/workflow/stage1_open_action_value.py','src/workflow/intraday_recovery_direct_allocation.py','src/model/nominal_beta_composition.py']
            rec.save_objects(artifact_path='sources',**{f.replace('/','__'):(root/f).read_text() for f in files});progress['sources_saved']=True;atomic_json(progress,progress_path)
    if not progress.get('repair_1_source_saved'):
        repair={'reason':'Qlib terminal calendar needs following existing source bound','failed_native':'raw_candidate27baa_0','failure':'IndexError terminal calendar index3778','guard_at':str(guard_time),'valuation_terminal':str(terminal),'guard_used_for_policy_or_NAV':False,'policy_paths_rerun':0,'unique_native_budget':9,'native_attempt_budget_after_owner_amendment':10,'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['src/workflow/frozen_policy_raw_cashbook.py','src/backtest/native_raw_corporate.py']}}
        rec.save_objects(artifact_path='sources/repair_1',workflow=Path(__file__).read_text(),native=(root/'src/backtest/native_raw_corporate.py').read_text())
        rec.save_objects(artifact_path='inputs',repair_1=repair,repair_1_provider_manifest=manifest,repair_1_calendar_guard=guard)
        atomic_json(repair,card/'implementation_repair.json');progress['repair_1_source_saved']=True;atomic_json(progress,progress_path)
    books={};proxy={};traces={}
    for family in ['raw_candidate27baa','raw_strong9097','raw_SOXX']:
        data=frame.copy()
        if family=='raw_candidate27baa':
            targets=candidate['analysis/targets']['candidate'].reindex(pd.DatetimeIndex(data.signal_date))
            assert targets.notna().all().all()
            for col in targets:data[col]=targets[col].to_numpy()
        elif family=='raw_strong9097':
            for col in ['soxl_weight','soxx_weight','cash_weight','tail_probability','alpha_probability']:data[col]=strong['inputs/frame'][col].to_numpy()
        if family in progress['paths_done']:paths=rec.load_object('paths/'+family);trace=rec.load_object('traces/'+family)
        else:
            if family in progress['path_attempts'] or len(progress['path_attempts'])>=3:raise ValueError('fixed path attempt already consumed')
            progress['path_attempts'].append(family);atomic_json(progress,progress_path)
            paths,trace=replay_raw_package(data,point_prices=points,schedule=schedule,plan=plan,controller=controller['inputs/controller'],recovery_spec=recovery,action_spec=spec,selection=action['controller/source_recovery_selection'],benchmark=family=='raw_SOXX')
            rec.save_objects(artifact_path='paths',**{family:paths});rec.save_objects(artifact_path='traces',**{family:trace});progress['paths_done'].append(family);atomic_json(progress,progress_path);print('raw policy package complete',family,flush=True)
        traces[family]=trace
        for cost in costs:
            name=f'{family}_{cost}'
            if name in progress['native_done']:book=rec.load_object('native/'+name)
            else:
                if name in progress['native_attempts']:
                    if name!='raw_candidate27baa_0' or progress.get('native_repair_attempts'):raise ValueError('native attempt already consumed')
                    progress['native_repair_attempts']=[name]
                else:
                    if len(progress['native_attempts'])>=9:raise ValueError('native unique budget exhausted')
                    progress['native_attempts'].append(name)
                atomic_json(progress,progress_path)
                book=native_raw_validate(paths[cost],provider_uri=provider,point_prices=points,schedule=schedule,tariff=plan['tariff'],cost=cost,initial_cash=plan['initial_cash'])
                rec.save_objects(artifact_path='native',**{name:book});progress['native_done'].append(name);atomic_json(progress,progress_path);print('native raw validated',name,flush=True)
            books[(family,cost)]=book
            proxy[(family,cost)]=candidate[f'analysis/native/candidate_{cost}'] if family=='raw_candidate27baa' else strong[f'analysis/native/candidate_{cost}'] if family=='raw_strong9097' else benchmark[f'analysis/native/soxx_{cost}']
            rec.log_metrics(completed_raw_paths=3*len(progress['paths_done']),completed_native=len(progress['native_done']))
    results=analyze_raw(books,proxy,frame,plan);gates=results['gates'];historic=all(v for k,v in gates.items() if k not in ['fullminute_risk_qualified','independent_development','qualified_forward'])
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','paths/*','traces/*','native/*','analysis/rolling/*']}
    summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'historical_raw_candidate_requires_risk_and_independent_forward' if historic else 'not_promoted_frozen_policy_raw_accounting_assessment','objective_achieved':False,'new_model_fits':0,'policy_paths':9,'native_Qlib_validations':9,'native_backtest_attempts':len(progress['native_attempts'])+len(progress.get('native_repair_attempts',[])),'gates':gates,'currency_assumption':'USD gross pretax paper, notsourceorbrokerproof','SOXX_contract':plan['SOXX_contract'],'source_currency_certified':False,'historical_PIT_qualified':False,'new_fetches':0,'observer_launches':0,'production_changes':0,'corporate_actions_in_schedule':sum(map(len,schedule.values())),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(provider),'artifact_policy':policy}
    for name,value in results.items():
        if isinstance(value,pd.DataFrame):value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
        elif name=='rolling':rec.save_objects(artifact_path='analysis/rolling',**value)
    rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['candidate_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

"""Three fixed composition-aware fits; own-ledger frozen policy evaluation."""
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.workflow.minute_action_value import _source
from src.workflow.minute_pit_allocator import _primary_dataset,_auxiliary_dataset,_prediction_dates,_ensemble
from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.beta_preserving_composition import verify_marks,CountRecoveryCalls
from src.model.composition_objective_allocator import CompositionObjectiveAllocator
from src.model.nominal_beta_composition import AuditedComposition
from src.backtest.full_path_tiered import replay_full_path
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.analysis.three_asset_failure_attribution import analyze
from src.analysis.integrated_allocation import stationary_bootstrap_excess
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def evaluate(table,annual,uncertainty,rolling):
    candidate=table[table.family.eq('candidate')].set_index('slippage_bps')
    def beats(name):
        control=table[table.family.eq(name)].set_index('slippage_bps')
        return bool(candidate.terminal_wealth.gt(control.terminal_wealth*(1+1e-8)).all())
    complete=annual[annual.family.eq('candidate') & annual.complete_period]
    if set(complete.slippage_bps)!=set(candidate.index) or any(set(g.period.astype(str))!={'2024','2025'} for _,g in complete.groupby('slippage_bps')):raise ValueError('incomplete annual gate input')
    return {'beat_paired_all_costs':beats('posthoc_composition'),'beat_strong_always_all_costs':beats('strong_always'),
        'net_soxx_excess_all_costs':bool(candidate.relative_wealth.gt(1+1e-8).all()),
        'drawdown_30pct':bool(candidate.max_drawdown_pct.ge(-30).all() and candidate.monitored_drawdown_pct.ge(-30).all()),
        'cvar_no_worse':bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),
        'rolling_70pct':all(v.relative_log.gt(np.log1p(1e-8)).mean()>=.7 for (f,c),v in rolling.items() if f=='candidate'),
        'complete_annual_70pct':bool(complete.groupby('slippage_bps').relative_wealth.apply(lambda v:v.gt(1+1e-8).mean()).ge(.7).all()),
        'bootstrap_positive':bool(uncertainty.lower_bound_positive.all()),'independent_development':False,'qualified_forward':False}


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('frozen training plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    costs,seeds=plan['slippage_bps'],plan['seeds']
    train,ta=_source(root,plan['training_source'],['inputs/daily_frame','inputs/features','inputs/common','inputs/timing','inputs/auxiliary','inputs/plan'])
    strong,sa=_source(root,plan['frozen_source'],['inputs/plan','analysis/predictions','analysis/targets',*[f'models/candidate_{s}' for s in seeds],*[f'analysis/native/{f}_{c}' for f in ['candidate','current_anchor','soxx'] for c in costs]])
    paired,pa=_source(root,plan['paired_control_source'],['inputs/frame',*[f'analysis/native/candidate_{c}' for c in costs]])
    reference,ra=_source(root,plan['reference_source'],[f'analysis/native/strong_always_{c}' for c in costs])
    controller,ca=_source(root,plan['price_source'],['inputs/controller','inputs/plan'])
    action,aa=_source(root,plan['action_source'],['inputs/plan','inputs/market','controller/source_recovery_selection'])
    original=strong[f'models/candidate_{seeds[0]}'];frame=paired['inputs/frame'].copy().reset_index(drop=True)
    inputs={name:train['inputs/'+key] for name,key in [('frame','daily_frame'),('features','features'),('common','common'),('timing','timing')]};inputs['spec']=original.spec
    dataset=_primary_dataset(inputs,train_end=str(original.train_end.date()),predict_bounds=train['inputs/plan']['outer_window'])
    auxiliary=_auxiliary_dataset(train['inputs/auxiliary']['outer'],str(original.train_end.date()))
    marks=verify_marks(frame,reference['analysis/native/strong_always_0'])
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);action_spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    market=action['inputs/market'];dates=pd.DatetimeIndex(market.signal_date);returns=market.set_index('signal_date')[['soxl_return','soxx_return']]
    progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':digest,'fit_attempts':[],'fitted':[],'native_attempts':[],'native':[]}
    if progress['plan_sha256']!=digest:raise ValueError('progress plan mismatch')
    tracking=card/'qlib_recorder';kw={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kw.update(recorder_id=progress['recorder_id'],resume=True)
    predictions={};audits=[]
    with R.start(**kw):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        if not progress.get('source_saved'):
            rec.save_objects(artifact_path='inputs',plan=plan,source_audits={'training':ta,'tangent':sa,'paired':pa,'reference':ra,'controller':ca,'action':aa},mark_verification=marks)
            rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),model=(root/'src/model/composition_objective_allocator.py').read_text(),base_model=(root/'src/model/minute_pit_allocator.py').read_text(),objective=(root/'src/workflow/pit_constituent_auxiliary_training_allocation.py').read_text())
            progress['source_saved']=True;atomic_json(progress,progress_path)
        for seed in seeds:
            old=strong[f'models/candidate_{seed}'];settings={**asdict(old.spec),'root':str(root)}
            for key in ['absolute_drawdown_penalty','conditional_drawdown_penalty','rolling_5d_penalty','cash_target_penalty','trade_event_penalty']:
                if getattr(old.candidate,key)!=plan['allocation_loss_terms'][key]:raise ValueError('active source penalty differs '+key)
            if old.spec.epochs!=300 or old.spec.turnover_loss_weight!=.006 or old.spec.training_cost_bps!=10:raise ValueError('source training recipe differs')
            model=CompositionObjectiveAllocator(base_features=old.base_features,extra_features=old.extra_features,feature_masks=old.feature_masks,spec=settings,seed=seed,cutoff=str(old.cutoff),train_end=str(old.train_end.date()),minimum_observations=old.minimum_observations)
            if asdict(model.candidate)!=asdict(old.candidate):raise ValueError('candidate loss coefficients differ')
            a,b=old._initialize(),model._initialize()
            for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
            init_hash=hashlib.sha256(b''.join(k.encode()+v.numpy().tobytes() for k,v in sorted(b.state_dict().items()))).hexdigest()
            frozen=_prediction_dates(old.predict(dataset));saved=strong['analysis/predictions'][('candidate',seed)]
            pd.testing.assert_frame_equal(frozen,saved)
            predictions[('control',seed)]=frozen
            if seed in progress['fitted']:model=rec.load_object(f'models/candidate_{seed}')
            else:
                if seed in progress['fit_attempts'] or len(progress['fit_attempts'])>=plan['maximum_model_fits']:raise ValueError('fit attempt budget exhausted; do not retry partial fits')
                progress['fit_attempts'].append(seed);atomic_json(progress,progress_path)
                model.fit(dataset,auxiliary_dataset=auxiliary)
                rec.save_objects(artifact_path='models',**{f'candidate_{seed}':model});progress['fitted'].append(seed);atomic_json(progress,progress_path)
            if model.primary_scaler!=old.primary_scaler or model.common_scaler!=old.common_scaler:raise ValueError('training scaler changed')
            for key in ['training_rows','auxiliary_rows','first_training_signal','last_training_signal','last_mature_label_at','epochs','checkpoint_policy']:
                if model.audit[key]!=old.audit[key]:raise ValueError('mature domain mismatch '+key)
            if model.audit['training_rows']!=plan['expected_primary_rows'] or model.audit['auxiliary_rows']!=plan['expected_auxiliary_rows']:raise ValueError('sample count changed')
            predictions[('candidate',seed)]=_prediction_dates(model.predict(dataset));audits.append({'seed':seed,'initialization_sha256':init_hash,'source_candidate':asdict(old.candidate),'spec':settings,**model.audit})
            rec.log_metrics(completed_fits=len(progress['fitted']));print('fitted fixed composition objective seed',seed,flush=True)
        targets={f:_smooth_weights(_ensemble(predictions,f,seeds).reindex(dates),spec=original.spec,candidate=original.candidate,asset_returns=returns) for f in ['control','candidate']}
        np.testing.assert_allclose(targets['control'],strong['analysis/targets']['candidate'],rtol=1e-10,atol=1e-10)
        rec.save_objects(artifact_path='analysis',predictions=predictions,targets=targets,training_audits=audits)
    data=frame.copy();aligned=targets['candidate'].reindex(pd.DatetimeIndex(frame.signal_date))
    if aligned.isna().any().any():raise ValueError('missing candidate target')
    for col in aligned:data[col]=aligned[col].to_numpy()
    books={};transform_rows=[]
    for cost in costs:
        if cost in progress['native']:book=rec.load_object(f'analysis/native/candidate_{cost}')
        else:
            if cost in progress['native_attempts'] or len(progress['native_attempts'])>=plan['maximum_new_native_cases']:raise ValueError('native attempt budget exhausted')
            progress['native_attempts'].append(cost);atomic_json(progress,progress_path)
            transform=AuditedComposition();modified=dict(controller['inputs/controller']);modified['frozen_open_action_value_model']=None
            model=modified['frozen_recovery_model'];counter=CountRecoveryCalls(model) if model is not None else None;modified['frozen_recovery_model']=counter
            replay=replay_full_path(data,models=None,plan=plan,slippage_bps=cost,controller=modified,recovery_spec=recovery,action_spec=action_spec,selection=action['controller/source_recovery_selection'],target_transform=transform)
            book=native_tiered_stage1(data,replay,provider_uri=root/plan['provider_uri'],tariff=plan['tariff'],slippage_bps=cost,initial_cash=plan['initial_cash'],freq='1min',opening_offset=pd.Timedelta(hours=9,minutes=47),position_type=QlibContinuousPosition)
            book.update(targets=replay['targets'],target_transform=pd.DataFrame(transform.rows),recovery_eligible_calls=counter.rows if counter else 0)
            pd.testing.assert_index_equal(book['native_report'].index,reference[f'analysis/native/strong_always_{cost}']['native_report'].index)
            rec.save_objects(artifact_path='analysis/native',**{f'candidate_{cost}':book});progress['native'].append(cost);atomic_json(progress,progress_path)
        books[('candidate',cost)]=book;transform_rows.append({'slippage_bps':cost,'recovery_eligible_calls':book['recovery_eligible_calls'],'recovery_triggers':int(book['authority'].recovery_action.sum()),'maximum_beta_error':book['target_transform'].beta_error.abs().max()})
        rec.log_metrics(completed_native_accounts=len(progress['native']));print('completed trained composition account',cost,flush=True)
        for f,value in [('posthoc_composition',paired[f'analysis/native/candidate_{cost}']),('strong_always',reference[f'analysis/native/strong_always_{cost}']),('tangent',strong[f'analysis/native/candidate_{cost}']),('original',strong[f'analysis/native/current_anchor_{cost}'])]:books[(f,cost)]=value
    benchmarks={c:strong[f'analysis/native/soxx_{c}'] for c in costs}
    results=analyze(frame,books,benchmarks,initial_cash=plan['initial_cash'],cash_reference=.15,rolling_sessions=126,pairs=[['candidate','posthoc_composition'],['candidate','strong_always']])
    table=results['accounting_attribution'];annual=results['annual_attribution'];intervals=[]
    for cost in costs:
        for comparator in plan['bootstrap']['comparators']:
            other=benchmarks[cost] if comparator=='soxx' else books[(comparator,cost)]
            intervals.append({'slippage_bps':cost,'comparator':comparator,**stationary_bootstrap_excess(books[('candidate',cost)]['authority'],other['authority'],**{k:v for k,v in plan['bootstrap'].items() if k!='comparators'})})
    uncertainty=pd.DataFrame(intervals);gates=evaluate(table,annual,uncertainty,results['rolling'])
    historical=all(v for k,v in gates.items() if k not in ['independent_development','qualified_forward'])
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','models/*','analysis/predictions','analysis/targets','analysis/native/*','analysis/daily/*','analysis/rolling/*']}
    summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'historical_candidate_requires_independent_forward' if historical else 'not_promoted_fixed_composition_objective','objective_achieved':False,'new_fits':len(progress['fitted']),'new_native_cases':len(progress['native']),'gates':gates,'provider_uri':str(root/plan['provider_uri']),'region':'us','as_of':plan['as_of'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'proxy_limitations':plan['proxy_limitations']}
    for name,value in results.items():
        if isinstance(value,dict):rec.save_objects(artifact_path='analysis/'+name,**{f'{f}_{c}':v for (f,c),v in value.items()})
        else:value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
    for name,value in [('backtest_benchmark_comparison',table),('annual_metrics',annual),('uncertainty',uncertainty),('composition_audit',pd.DataFrame(transform_rows)),('training_audit',pd.DataFrame([{k:v for k,v in row.items() if k not in ['source_candidate','spec','extra_scaler','observed_minute_training_rows']} for row in audits]))]:value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
    rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['paired_control_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

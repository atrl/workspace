"""Fixed AMD-only linear residual on immutable learned NVDA allocators."""
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.workflow.minute_pit_allocator import _dataset,_auxiliary_dataset,_prediction_dates,_ensemble
from src.model.minute_pit_allocator import TARGETS
from src.model.frozen_allocation_residual import FrozenAllocationResidual,base_fingerprint
from src.data.amd_own_guidance_realization import align_realizations,FEATURE,FLAGS
from src.data import QlibProviderConfig,build_qlib_store_from_frame,load_qlib_feature_frame
from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.beta_preserving_composition import verify_marks
from src.backtest.causal_cost_scenario_controller import replay_cost_scenarios
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.analysis.three_asset_failure_attribution import analyze
from src.analysis.integrated_allocation import stationary_bootstrap_excess
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def evaluate(results,uncertainty,costs):
    table=results['accounting_attribution'];annual=results['annual_attribution'];c=table.loc[table.family.eq('candidate')].set_index('slippage_bps')
    if set(c.index)!=set(costs):raise ValueError('candidate cost coverage differs')
    complete=annual.loc[annual.family.eq('candidate')&annual.complete_period]
    if set(complete.slippage_bps)!=set(costs) or any(set(g.period.astype(str))!={'2024','2025'} for _,g in complete.groupby('slippage_bps')):raise ValueError('complete annual groups invalid')
    def beats(f):
        b=table.loc[table.family.eq(f)].set_index('slippage_bps');return bool(c.terminal_wealth.gt(b.terminal_wealth*(1+1e-8)).all())
    rolls={cost:results['rolling'][('candidate',cost)] for cost in costs}
    if any(v.empty for v in rolls.values()):raise ValueError('empty rolling gate')
    if len(uncertainty)!=9:raise ValueError('missing bootstrap comparison')
    return {'beat_matched_control_all_costs':beats('matched_AMD_state_residual'),'beat_frozen_NVDA27baa_all_costs':beats('frozen_NVDA27baa'),
        'net_soxx_excess_all_costs':bool(c.relative_wealth.gt(1+1e-8).all()),'drawdown_30pct':bool(c.max_drawdown_pct.ge(-30).all() and c.monitored_drawdown_pct.ge(-30).all()),
        'cvar_no_worse':bool(c.cvar5_pct.ge(c.benchmark_cvar5_pct).all()),'rolling_70pct':all(v.relative_log.gt(np.log1p(1e-8)).mean()>=.7 for v in rolls.values()),
        'complete_annual_70pct':bool(complete.groupby('slippage_bps').relative_wealth.apply(lambda v:v.gt(1+1e-8).mean()).ge(.7).all()),
        'bootstrap_positive':bool(uncertainty.lower_bound_positive.all()),'independent_development':False,'qualified_forward':False}


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('plan changed')
        return old
    if digest!='9e9ab6c1cd142a2e348476cd77664aae71f5207c941c2d2e724bdb1186a7b508':raise ValueError('unapproved plan')
    for f,sha in plan['unchanged_code_sha256'].items():
        if hashlib.sha256((root/f).read_bytes()).hexdigest()!=sha:raise ValueError('frozen code changed '+f)
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    costs=plan['slippage_bps'];seeds=plan['training']['seeds']
    train,ta=_source(root,plan['training_source'],['inputs/daily_frame','inputs/features','inputs/common','inputs/timing','inputs/auxiliary'])
    source,sa=_source(root,plan['amd_source'],['analysis/pairs'])
    strongest,sta=_source(root,plan['base_source'],['inputs/frame','inputs/plan','inputs/features','analysis/targets',*[f'models/candidate_{s}' for s in seeds],*[f'analysis/predictions/candidate_{s}' for s in seeds],*[f'analysis/native/candidate_{c}' for c in costs]])
    tangent,tga=_source(root,plan['tangent_source'],[f'analysis/native/{f}_{c}' for f in ['candidate','current_anchor','soxx'] for c in costs])
    older,oa=_source(root,plan['older_coupled_source'],[f'analysis/native/candidate_{c}' for c in costs])
    controller,ca=_source(root,plan['price_source'],['inputs/controller','inputs/plan'])
    action,aca=_source(root,plan['action_source'],['inputs/plan','inputs/market','controller/source_recovery_selection'])
    for k in ['tariff','initial_cash','start','end','as_of','slippage_bps']:
        if strongest['inputs/plan'][k]!=plan[k]:raise ValueError('source account contract differs '+k)
    if controller['inputs/plan']['tariff']!=plan['tariff']:raise ValueError('controller tariff differs')
    frame=strongest['inputs/frame'].reset_index(drop=True).copy();marks=verify_marks(frame,strongest['analysis/native/candidate_0'])
    base=train['inputs/daily_frame'];timing=train['inputs/timing'];original=strongest[f'models/candidate_{seeds[0]}']
    features,trace,events=align_realizations(source['analysis/pairs'],timing.decision_at)
    if len(original.base_features)!=42 or len(original.extra_features)!=1:raise ValueError('frozen NVDA feature dimension differs')
    for day in frame.signal_date:
        if pd.isna(timing.loc[day,'decision_at']):raise ValueError('executed signal lacks decision timestamp')
    maturity=pd.to_datetime(timing.label_available_at)
    eligible=maturity.notna()&maturity.lt(original.cutoff)&(base.index<=original.train_end)&(base.index>=pd.Timestamp(original.spec.train_start))
    eligible &= np.isfinite(base.loc[:,list(TARGETS)].to_numpy(float)).all(axis=1)
    if eligible.sum()!=3014:raise ValueError('primary training cohort differs')
    used=trace.loc[eligible];comparable=features.loc[eligible,FLAGS[1]].eq(1)
    coverage={'eligible_training_rows':int(eligible.sum()),'realized_quarter_states_in_eligible_rows':int(used.target_quarter.nunique()),'comparable_quarter_states_in_eligible_rows':int(used.loc[comparable,'target_quarter'].nunique()),'comparable_daily_training_rows':int(comparable.sum()),'known_preannounced_daily_rows':int(features.loc[eligible,FLAGS[2]].sum()),'first_training_decision':str(timing.loc[eligible,'decision_at'].min()),'last_training_decision':str(timing.loc[eligible,'decision_at'].max()),'quarterly_events_are_not_independent_daily_observations':True}
    feature_provider=root/'data/qlib'/plan['experiment'];payload=features.copy();payload.insert(0,'datetime',features.index);payload['instrument']='SOXL_SOXX_CASH';payload=payload.reset_index(drop=True)
    feature_manifest=build_qlib_store_from_frame(payload,provider_uri=feature_provider,freq='day',region='us',as_of=plan['as_of'],field_map={'$'+k:k for k in features},source='AMD-only original guidance realization; exact09:29 join; frozen NVDA inputs preserved separately')
    read=load_qlib_feature_frame(QlibProviderConfig(feature_provider),instruments=['SOXL_SOXX_CASH'],fields=['$'+k for k in features],start=str(base.index.min().date()),end=plan['as_of']).xs('SOXL_SOXX_CASH',level='instrument').rename(columns=lambda x:x.removeprefix('$')).astype(float)
    pd.testing.assert_index_equal(read.index,features.index,check_names=False);np.testing.assert_allclose(read,features,rtol=1e-6,atol=1e-7,equal_nan=True)
    dataset=_dataset(train['inputs/features'].join(strongest['inputs/features']).join(read),base[list(TARGETS)],train['inputs/common'],timing,{'train':(original.spec.train_start,str(original.train_end.date())),'predict':(plan['start'],plan['end'])})
    auxiliary=_auxiliary_dataset(train['inputs/auxiliary']['outer'],str(original.train_end.date()))
    if len(auxiliary.prepare('train'))!=20741:raise ValueError('auxiliary cohort differs')
    models={};initials={};initial_predictions={}
    for seed in seeds:
        base_model=strongest[f'models/candidate_{seed}']
        if base_model.spec.epochs!=300 or base_model.spec.turnover_loss_weight!=.006 or base_model.spec.training_cost_bps!=10:raise ValueError('source recipe differs')
        frozen=_prediction_dates(base_model.predict(dataset));saved=strongest[f'analysis/predictions/candidate_{seed}']
        pd.testing.assert_index_equal(frozen.index,saved.index);np.testing.assert_allclose(frozen,saved,rtol=0,atol=1e-12)
        for family in ['matched_AMD_state_residual','candidate']:
            model=FrozenAllocationResidual(base_model,numeric=family=='candidate');model.prepare_training(dataset)
            if base_fingerprint(model.base)!=base_fingerprint(base_model):raise ValueError('base copy differs')
            prediction=_prediction_dates(model.predict(dataset));np.testing.assert_allclose(prediction,saved,rtol=0,atol=1e-12)
            models[(family,seed)]=model;initial_predictions[(family,seed)]=prediction
        left,right=models[('matched_AMD_state_residual',seed)],models[('candidate',seed)]
        if left.state_scaler!=right.state_scaler or left.numeric_scaler!=right.numeric_scaler:raise ValueError('paired AMD scalers differ')
        initials[seed]={'base_fingerprint':base_fingerprint(base_model),'zero_residual_prediction_parity_atol':1e-12,'control_parameters':9,'candidate_parameters':12}
    market=action['inputs/market'];dates=pd.DatetimeIndex(market.signal_date);returns=market.set_index('signal_date')[['soxl_return','soxx_return']]
    for family in ['matched_AMD_state_residual','candidate']:
        initial_targets=_smooth_weights(_ensemble(initial_predictions,family,seeds).reindex(dates),spec=original.spec,candidate=original.candidate,asset_returns=returns)
        pd.testing.assert_index_equal(initial_targets.index,strongest['analysis/targets']['candidate'].index)
        np.testing.assert_allclose(initial_targets,strongest['analysis/targets']['candidate'],rtol=0,atol=1e-12)
    progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':digest,'fit_attempts':[],'fitted':[],'shadow_attempts':[],'shadow_done':[],'native_attempts':[],'native':[]}
    if progress['plan_sha256']!=digest:raise ValueError('progress plan changed')
    tracking=card/'qlib_recorder';kw={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kw.update(recorder_id=progress['recorder_id'],resume=True)
    predictions={};audits=[]
    with R.start(**kw):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        if not progress.get('source_saved'):
            rec.save_objects(artifact_path='inputs',plan=plan,frame=frame,source_audits={'training':ta,'qualification':sa,'strongest':sta,'tangent':tga,'older_coupled':oa,'controller':ca,'action':aca},features=read,feature_manifest=feature_manifest,feature_trace=trace,events=events,training_event_coverage=coverage,mark_verification=marks,initialization=initials)
            code=['src/workflow/amd_frozen_nvda_residual_training.py','src/data/amd_own_guidance_realization.py','src/model/frozen_allocation_residual.py',*plan['unchanged_code_sha256']]
            rec.save_objects(artifact_path='sources',**{f.replace('/','__'):(root/f).read_text() for f in code});progress['source_saved']=True;atomic_json(progress,progress_path)
        for seed in seeds:
            for family in ['matched_AMD_state_residual','candidate']:
                name=f'{family}_{seed}';model=models[(family,seed)]
                if name in progress['fitted']:model=rec.load_object('models/'+name)
                else:
                    if name in progress['fit_attempts'] or len(progress['fit_attempts'])>=6:raise ValueError('fit attempt budget exhausted')
                    progress['fit_attempts'].append(name);atomic_json(progress,progress_path)
                    model.fit(dataset)
                    rec.save_objects(artifact_path='models',**{name:model});progress['fitted'].append(name);atomic_json(progress,progress_path)
                models[(family,seed)]=model
                for key in ['training_rows','first_training_signal','last_training_signal','last_mature_label_at','epochs','checkpoint_policy']:
                    if model.audit[key]!=strongest[f'models/candidate_{seed}'].audit[key]:raise ValueError('training audit differs '+key)
                predictions[(family,seed)]=_prediction_dates(model.predict(dataset));np.testing.assert_allclose(predictions[(family,seed)][['tail_probability','alpha_probability']],strongest[f'analysis/predictions/candidate_{seed}'][['tail_probability','alpha_probability']],rtol=0,atol=0);rec.save_objects(artifact_path='analysis/predictions',**{name:predictions[(family,seed)]});audits.append({'family':family,'seed':seed,**model.audit});rec.log_metrics(completed_fits=len(progress['fitted']));print('completed fit',name,flush=True)
            left,right=models[('matched_AMD_state_residual',seed)],models[('candidate',seed)]
            if left.state_scaler!=right.state_scaler or base_fingerprint(left.base)!=base_fingerprint(right.base):raise ValueError('shared scaler or frozen base mismatch')
        rec.save_objects(artifact_path='analysis',training=audits)
    market=action['inputs/market'];dates=pd.DatetimeIndex(market.signal_date);returns=market.set_index('signal_date')[['soxl_return','soxx_return']]
    targets={f:_smooth_weights(_ensemble(predictions,f,seeds).reindex(dates),spec=original.spec,candidate=original.candidate,asset_returns=returns) for f in ['matched_AMD_state_residual','candidate']}
    rec.save_objects(artifact_path='analysis',targets=targets)
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    books={};traces={}
    for family in ['matched_AMD_state_residual','candidate']:
        local=frame.copy();aligned=targets[family].reindex(pd.DatetimeIndex(local.signal_date))
        if aligned.isna().any().any():raise ValueError('missing target')
        for col in aligned:local[col]=aligned[col].to_numpy()
        if family in progress['shadow_done']:paths=rec.load_object('analysis/shadow/'+family);scenario=rec.load_object('analysis/scenario/'+family)
        else:
            if family in progress['shadow_attempts'] or len(progress['shadow_attempts'])>=2:raise ValueError('shadow attempt budget exhausted')
            progress['shadow_attempts'].append(family);atomic_json(progress,progress_path)
            paths,scenario=replay_cost_scenarios(local,plan=plan,scenarios=costs,controller=controller['inputs/controller'],recovery_spec=recovery,action_spec=spec,selection=action['controller/source_recovery_selection'])
            rec.save_objects(artifact_path='analysis/shadow',**{family:paths});rec.save_objects(artifact_path='analysis/scenario',**{family:scenario});progress['shadow_done'].append(family);atomic_json(progress,progress_path)
        traces[family]=scenario
        for cost in costs:
            name=f'{family}_{cost}'
            if name in progress['native']:book=rec.load_object('analysis/native/'+name)
            else:
                if name in progress['native_attempts'] or len(progress['native_attempts'])>=6:raise ValueError('native attempt budget exhausted')
                progress['native_attempts'].append(name);atomic_json(progress,progress_path)
                book=native_tiered_stage1(local,paths[cost],provider_uri=root/plan['provider_uri'],tariff=plan['tariff'],slippage_bps=cost,initial_cash=plan['initial_cash'],freq='1min',opening_offset=pd.Timedelta(hours=9,minutes=47),position_type=QlibContinuousPosition)
                book['targets']=paths[cost]['targets'];pd.testing.assert_index_equal(book['native_report'].index,strongest[f'analysis/native/candidate_{cost}']['native_report'].index)
                rec.save_objects(artifact_path='analysis/native',**{name:book});progress['native'].append(name);atomic_json(progress,progress_path)
            books[(family,cost)]=book;rec.log_metrics(completed_native_cases=len(progress['native']));print('completed native',name,flush=True)
    for cost in costs:
        books[('frozen_NVDA27baa',cost)]=strongest[f'analysis/native/candidate_{cost}']
        books[('older_coupled9097',cost)]=older[f'analysis/native/candidate_{cost}']
    benchmarks={c:tangent[f'analysis/native/soxx_{c}'] for c in costs}
    results=analyze(frame,books,benchmarks,initial_cash=plan['initial_cash'],cash_reference=.15,rolling_sessions=126,pairs=[['candidate','matched_AMD_state_residual'],['candidate','frozen_NVDA27baa']])
    rows=[]
    for cost in costs:
        for comparator in plan['bootstrap']['comparators']:
            other=benchmarks[cost] if comparator=='soxx' else books[(comparator,cost)]
            rows.append({'slippage_bps':cost,'comparator':comparator,**stationary_bootstrap_excess(books[('candidate',cost)]['authority'],other['authority'],**{k:v for k,v in plan['bootstrap'].items() if k!='comparators'})})
    uncertainty=pd.DataFrame(rows);gates=evaluate(results,uncertainty,costs);historical=all(v for k,v in gates.items() if k not in ['independent_development','qualified_forward'])
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','models/*','analysis/predictions/*','analysis/targets','analysis/shadow/*','analysis/scenario/*','analysis/native/*','analysis/daily/*','analysis/rolling/*']}
    summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'historical_candidate_requires_independent_forward' if historical else 'not_promoted_fixed_AMD_frozen_NVDA_residual','objective_achieved':False,'new_residual_fits':len(progress['fitted']),'base_model_fits':0,'new_auxiliary_gradient_rows':0,'new_shadow_policy_paths':3*len(progress['shadow_done']),'new_native_accounts':len(progress['native']),'gates':gates,'training_event_coverage':coverage,'conditional_source_timing':True,'risk_evidence':'known/close sampled only; fullminute not run after negative','first_public_timestamp_verified':False,'all_updates_coverage_verified':False,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/plan['provider_uri']),'artifact_policy':policy}
    for name,value in results.items():
        if isinstance(value,dict):rec.save_objects(artifact_path='analysis/'+name,**{f'{f}_{c}':v for (f,c),v in value.items()})
        else:value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
    scenario_summary=pd.concat([v.groupby('slippage_bps').agg(recovery_eligible=('recovery_eligible','sum'),local_approvals=('recovery_local_approval','sum'),consensus_recoveries=('recovery_consensus','sum'),mean_common_cap=('common_cap','mean')).reset_index().assign(family=f) for f,v in traces.items()])
    for name,value in [('uncertainty',uncertainty),('training_audit',pd.DataFrame(audits)),('scenario_summary',scenario_summary)]:value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
    rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['base_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

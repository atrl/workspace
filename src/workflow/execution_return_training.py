"""Paired primary return-label alignment on a common native-price cohort."""
from dataclasses import asdict,replace
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.workflow.minute_pit_allocator import _primary_dataset, _auxiliary_dataset, _prediction_dates, _ensemble
from src.model.minute_pit_allocator import MinutePitAllocator
from src.model.tangent_allocator import TangentHeadAllocator
from src.data.allocation_execution_labels import paired_execution_returns
from src.data.allocation_feature_contract import disjoint_momentum_inputs, reconstruct_momentum
from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.backtest.full_path_tiered import replay_full_path
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.analysis.integrated_allocation import paired_metrics, annual_paired_metrics, stationary_bootstrap_excess
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def _initial_hash(model):
    state=model._initialize().state_dict()
    return hashlib.sha256(b''.join(k.encode()+v.detach().numpy().tobytes() for k,v in sorted(state.items()))).hexdigest()


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed ablation plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    names=['inputs/daily_frame','inputs/features','inputs/common','inputs/timing','inputs/auxiliary','inputs/audit','inputs/plan','analysis/outer_predictions',*[f'models/base_outer_{seed}' for seed in plan['seeds']]]
    training,train_audit=_source(root,plan['training_source'],names)
    source,audit=_source(root,plan['source'],['inputs/frame',*[f'analysis/native/cash_start_2023_{family}_{slip}' for family in ['current_anchor','soxx'] for slip in plan['slippage_bps']]])
    controller,control_audit=_source(root,plan['price_source'],['inputs/controller','inputs/plan','inputs/frame'])
    action,action_audit=_source(root,plan['action_source'],['inputs/plan','inputs/market','inputs/source_targets','controller/source_recovery_selection'])
    if controller['inputs/plan']['tariff']!=plan['tariff']:raise ValueError('tariff mismatch')
    frozen,frozen_audit=_source(root,plan['frozen_source'],['inputs/plan',*[f'analysis/native/{family}_{slip}' for family in ['control','candidate','current_anchor','soxx'] for slip in plan['slippage_bps']]])
    for key in ['source','price_source','action_source','training_source','tariff','initial_cash','start','end','as_of','slippage_bps','maximum_soxl_weight','minimum_cash_weight','maximum_drawdown','seeds']:
        if frozen['inputs/plan'][key]!=plan[key]:raise ValueError('reference context differs: '+key)
    original=training[f'models/base_outer_{plan["seeds"][0]}'];source_plan=training['inputs/plan']
    inputs={name:training['inputs/'+key] for name,key in [('frame','daily_frame'),('features','features'),('common','common'),('timing','timing')]};inputs['spec']=original.spec
    dataset=_primary_dataset(inputs,train_end=str(original.train_end.date()),predict_bounds=source_plan['outer_window'])
    old_frame,new_frame,label_audit=paired_execution_returns(inputs['frame'],inputs['timing'],controller['inputs/frame'])
    cohort_spec=replace(original.spec,train_start=plan['cohort_train_start'])
    datasets={family:_primary_dataset({**inputs,'frame':f,'spec':cohort_spec},train_end=str(original.train_end.date()),predict_bounds=source_plan['outer_window']) for family,f in [('control',old_frame),('candidate',new_frame)]}
    # Native labels use the same marks as the evaluation account, not a second price plane.
    native=controller['inputs/frame'].set_index('signal_date');evaluation=source['inputs/frame'].set_index('signal_date')
    for field in ['soxl_open','soxx_open','soxl_exit_open','soxx_exit_open']:
        shared=native.index.intersection(evaluation.index);np.testing.assert_allclose(native.loc[shared,field],evaluation.loc[shared,field],rtol=0,atol=1e-10)
    auxiliary=_auxiliary_dataset(training['inputs/auxiliary']['outer'],str(original.train_end.date()))
    tracking=card/'qlib_recorder';progress_path=card/'run_state.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':digest,'fitted':[],'native':[]}
    if progress['plan_sha256']!=digest:raise ValueError('progress plan differs')
    kwargs={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kwargs.update(recorder_id=progress['recorder_id'],resume=True)
    predictions={};training_rows=[]
    with R.start(**kwargs):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        rec.log_params(plan_sha256=digest,maximum_new_model_fits=6,maximum_new_native_cases=6)
        rec.save_objects(artifact_path='inputs',plan=plan,training_audit=train_audit,source_audit=audit,controller_audit=control_audit,action_audit=action_audit,frozen_reference_audit=frozen_audit,label_audit=label_audit,original_labels=old_frame,execution_labels=new_frame)
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),model=(root/'src/model/minute_pit_allocator.py').read_text(),representation=(root/'src/data/allocation_feature_contract.py').read_text(),candidate_model=(root/plan.get('candidate_model_source','src/model/minute_pit_allocator.py')).read_text(),label_contract=(root/'src/data/allocation_execution_labels.py').read_text())
        for seed in plan['seeds']:
            original_control=training[f'models/base_outer_{seed}'];spec={**asdict(original_control.spec),'root':str(root)}
            base_prediction=_prediction_dates(original_control.predict(dataset));np.testing.assert_allclose(base_prediction,training['analysis/outer_predictions'][('base',seed)],atol=1e-12,rtol=1e-12)
            spec={**spec,'train_start':plan['cohort_train_start']}
            models={family:TangentHeadAllocator(base_features=original_control.base_features,extra_features=[],feature_masks=original_control.feature_masks,spec=spec,seed=seed,cutoff=str(original_control.cutoff),train_end=str(original_control.train_end.date()),minimum_observations=original_control.minimum_observations) for family in ['control','candidate']}
            identity=_initial_hash(models['control'])
            if identity!=_initial_hash(models['candidate']):raise ValueError('paired initialization differs')
            initialization_audit={'identical_initial_parameters':True,'sha256':identity}
            for family,model in models.items():
                name=f'{family}_{seed}'
                if name in progress['fitted']:model=rec.load_object('models/'+name);models[family]=model
                else:
                    if len(progress['fitted'])>=plan['maximum_model_fits']:raise ValueError('fit budget exhausted')
                    model.fit(datasets[family],auxiliary_dataset=auxiliary);rec.save_objects(artifact_path='models',**{name:model});progress['fitted'].append(name);atomic_json(progress,progress_path)
                predictions[(family,seed)]=_prediction_dates(model.predict(datasets[family]));training_rows.append({'family':family,'seed':seed,'initialization_audit':initialization_audit,**model.audit});rec.log_metrics(completed_model_fits=len(progress['fitted']));print('fit',name,flush=True)
            a,b=models['control'],models['candidate']
            for key in ['training_rows','first_training_signal','last_training_signal','last_mature_label_at','auxiliary_rows','epochs','checkpoint_policy']:
                if a.audit[key]!=b.audit[key]:raise ValueError('paired training domain differs: '+key)
            if a.primary_scaler!=b.primary_scaler or a.common_scaler!=b.common_scaler:raise ValueError('paired scalers differ')
        rec.save_objects(artifact_path='analysis',predictions=predictions,training=training_rows)
    frame=source['inputs/frame'];frame=frame.loc[frame.signal_date.between(plan['start'],plan['end'])].copy()
    market=action['inputs/market'];dates=pd.DatetimeIndex(market.signal_date);returns=market.set_index('signal_date')[['soxl_return','soxx_return']]
    targets={family:_smooth_weights(_ensemble(predictions,family,plan['seeds']).reindex(dates),spec=original.spec,candidate=original.candidate,asset_returns=returns) for family in ['control','candidate']}
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);action_spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    provider=root/'data/qlib/full_path_tiered_allocation_20261013';books={};seed_rows=[]
    for (family,seed),prediction in predictions.items():
        seed_rows.append({'family':family,'seed':seed,'mean_raw_cash_pct':100*prediction.reindex(pd.DatetimeIndex(frame.signal_date)).cash_weight.mean()})
    for family in ['control','candidate']:
        data=frame.copy();aligned=targets[family].reindex(pd.DatetimeIndex(data.signal_date))
        if aligned.isna().any().any():raise ValueError('missing allocation on required account day')
        for col in aligned:data[col]=aligned[col].to_numpy()
        for slip in plan['slippage_bps']:
            key=f'{family}_{slip}'
            if key in progress['native']:
                book=rec.load_object('analysis/native/'+key)
            else:
                if len(progress['native'])>=plan['maximum_new_native_cases']:raise ValueError('native budget exhausted')
                replay=replay_full_path(data,models=None,plan=plan,slippage_bps=slip,controller=controller['inputs/controller'],recovery_spec=recovery,action_spec=action_spec,selection=action['controller/source_recovery_selection'])
                book=native_tiered_stage1(data,replay,provider_uri=provider,tariff=plan['tariff'],slippage_bps=slip,initial_cash=plan['initial_cash'],freq='1min',opening_offset=pd.Timedelta(hours=9,minutes=47),position_type=QlibContinuousPosition)
                book['targets']=replay['targets'];rec.save_objects(artifact_path='analysis/native',**{key:book});progress['native'].append(key);atomic_json(progress,progress_path)
            books[(family,slip)]=book;rec.log_metrics(completed_new_native_cases=len(progress['native']));print('account',key,flush=True)
    for slip in plan['slippage_bps']:
        for family,key in [('bare_tangent','candidate'),('original_control','control'),('current_anchor','current_anchor'),('soxx','soxx')]:
            books[(family,slip)]=frozen[f'analysis/native/{key}_{slip}'];rec.save_objects(artifact_path='analysis/native',**{f'{family}_{slip}':books[(family,slip)]})
    rows=[];annual=[];uncertainty=[]
    for (family,slip),book in books.items():
        path=book['authority'];benchmark=books[('soxx',slip)]['authority'];control=books[('control',slip)]['authority']
        if not path.trade_date.equals(benchmark.trade_date):raise ValueError('benchmark calendar differs')
        relative=np.log1p(path.strategy_return)-np.log1p(benchmark.strategy_return)
        rows.append({'family':family,'slippage_bps':slip,**paired_metrics(path,benchmark),'monitored_drawdown_pct':-100*path.monitored_drawdown.max(),'control_relative_wealth':float(path.equity.iloc[-1]/control.equity.iloc[-1]),'average_cash_pct':100*path.ending_cash_weight.mean(),'orders':len(book['orders']),'broker_fees_usd':path.broker_and_external_fees_usd.sum(),'slippage_usd':path.slippage_reserve_usd.sum(),'positive_126d_fraction':relative.rolling(126).sum().dropna().gt(0).mean()})
        annual.append(annual_paired_metrics(path,benchmark).assign(family=family,slippage_bps=slip))
        if family=='candidate':
            for comparator in ['soxx','control','bare_tangent']:
                uncertainty.append({'slippage_bps':slip,'comparator':comparator,**stationary_bootstrap_excess(path,books[(comparator,slip)]['authority'],mean_block=63,resamples=2000,alpha=.025,seed=20261014)})
    table=pd.DataFrame(rows);annual=pd.concat(annual,ignore_index=True);uncertain=pd.DataFrame(uncertainty);candidate=table.loc[table.family.eq('candidate')]
    gates={'net_soxx_excess_all_costs':bool(candidate.relative_wealth.gt(1+1e-8).all()),'drawdown_30pct':bool(candidate.max_drawdown_pct.ge(-30).all() and candidate.monitored_drawdown_pct.ge(-30).all()),'cvar_no_worse':bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),'beat_paired_control_all_costs':bool(candidate.control_relative_wealth.gt(1+1e-8).all()),'rolling_70pct':bool(candidate.positive_126d_fraction.ge(.7).all()),'annual_70pct':bool(annual.loc[annual.family.eq('candidate')].groupby('slippage_bps').relative_wealth.apply(lambda x:x.gt(1).mean()).ge(.7).all()),'bootstrap_positive':bool(uncertain.lower_bound_positive.all()),'independent_development':False,'qualified_forward':False}
    gates['beat_original_all_costs']=bool(candidate.set_index('slippage_bps').terminal_wealth.gt(table.loc[table.family.eq('current_anchor')].set_index('slippage_bps').terminal_wealth).all())
    gates['beat_bare_tangent_all_costs']=bool(candidate.set_index('slippage_bps').terminal_wealth.gt(table.loc[table.family.eq('bare_tangent')].set_index('slippage_bps').terminal_wealth).all())
    historical=all(v for k,v in gates.items() if k not in ['independent_development','qualified_forward'])
    summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'needs_independent_validation' if historical else plan.get('failure_decision','not_promoted_fixed_representation_failed'),'gates':gates,'objective_achieved':False,'new_fits':len(progress['fitted']),'new_native_cases':len(progress['native']),'reused_reference_models':3,'reused_native_cases':12,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(provider),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/model.html','scope':plan.get('scope','Only primary momentum coordinates changed; original common encoder retained. Same dimension and initialization. Paired base models/accounts reused, original f696 and SOXX retained as comparators. Fixed final epoch and equal-weight three-seed ensemble; reused history, no independent development or forward qualification.'),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','models/*','analysis/predictions','analysis/native/*']}}
    rec.save_objects(artifact_path='analysis',summary=summary,comparison=table,annual=annual,uncertainty=uncertain,seed_cash=pd.DataFrame(seed_rows),targets=targets)
    artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    for name,value in [('backtest_benchmark_comparison',table),('annual_metrics',annual),('uncertainty',uncertain),('seed_cash',pd.DataFrame(seed_rows))]:value.to_csv(card/(name+'.csv'),index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

"""Bounded paired allocator training and full-policy net-cost evaluation."""
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.workflow.minute_pit_allocator import _primary_dataset, _auxiliary_dataset, _prediction_dates, _ensemble
from src.model.minute_pit_allocator import MinutePitAllocator
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
    controller,control_audit=_source(root,plan['price_source'],['inputs/controller','inputs/plan'])
    action,action_audit=_source(root,plan['action_source'],['inputs/plan','inputs/market','inputs/source_targets','controller/source_recovery_selection'])
    if controller['inputs/plan']['tariff']!=plan['tariff']:raise ValueError('tariff mismatch')
    paired,paired_audit=_source(root,plan['paired_control_source'],['inputs/plan',*[f'analysis/native/control_{slip}' for slip in plan['slippage_bps']]])
    for key in ['source','price_source','action_source','training_source','tariff','initial_cash','start','end','as_of','slippage_bps','maximum_soxl_weight','minimum_cash_weight','maximum_drawdown','seeds']:
        if paired['inputs/plan'][key]!=plan[key]:raise ValueError('paired account context differs: '+key)
    original=training[f'models/base_outer_{plan["seeds"][0]}'];source_plan=training['inputs/plan']
    inputs={name:training['inputs/'+key] for name,key in [('frame','daily_frame'),('features','features'),('common','common'),('timing','timing')]};inputs['spec']=original.spec
    dataset=_primary_dataset(inputs,train_end=str(original.train_end.date()),predict_bounds=source_plan['outer_window'])
    representation=plan.get('input_representation','disjoint_primary_momentum')
    if representation=='disjoint_primary_momentum':
        transformed,mapping=disjoint_momentum_inputs(inputs['features'].add_prefix('primary:'))
        restored=reconstruct_momentum(transformed).rename(columns=lambda c:c.removeprefix('primary:'))
        np.testing.assert_allclose(restored,inputs['features'][list(restored.columns)],rtol=1e-12,atol=1e-12,equal_nan=True)
        replacement={r.source.removeprefix('primary:'):r.target.removeprefix('primary:') for r in mapping.itertuples()}
        transformed=transformed.rename(columns=lambda c:c.removeprefix('primary:')).reindex(columns=[replacement.get(c,c) for c in inputs['features'].columns])
        untouched=[c for c in inputs['features'] if c not in replacement]
        pd.testing.assert_frame_equal(transformed[untouched],inputs['features'][untouched])
        candidate_inputs={**inputs,'features':transformed}
    elif representation=='original':
        replacement={};mapping=pd.DataFrame();candidate_inputs=inputs
    else:raise ValueError('unknown fixed input representation')
    routing=plan.get('encoder_routing','shared')
    if routing=='shared':model_class=MinutePitAllocator
    elif routing=='allocation_only':
        from src.model.allocation_encoder_routing import AllocationEncoderAllocator
        model_class=AllocationEncoderAllocator
    else:raise ValueError('unknown fixed encoder routing')
    candidate_dataset=_primary_dataset(candidate_inputs,train_end=str(original.train_end.date()),predict_bounds=source_plan['outer_window'])
    auxiliary=_auxiliary_dataset(training['inputs/auxiliary']['outer'],str(original.train_end.date()))
    tracking=card/'qlib_recorder';progress_path=card/'run_state.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':digest,'fitted':[],'native':[]}
    if progress['plan_sha256']!=digest:raise ValueError('progress plan differs')
    kwargs={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kwargs.update(recorder_id=progress['recorder_id'],resume=True)
    predictions={};training_rows=[]
    with R.start(**kwargs):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        rec.log_params(plan_sha256=digest,maximum_new_model_fits=3,maximum_new_native_cases=3)
        rec.save_objects(artifact_path='inputs',plan=plan,training_audit=train_audit,source_audit=audit,controller_audit=control_audit,action_audit=action_audit,paired_control_audit=paired_audit,representation_mapping=mapping)
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),model=(root/'src/model/minute_pit_allocator.py').read_text(),representation=(root/'src/data/allocation_feature_contract.py').read_text(),candidate_model=(root/plan.get('candidate_model_source','src/model/minute_pit_allocator.py')).read_text())
        for seed in plan['seeds']:
            control=training[f'models/base_outer_{seed}'];spec={**asdict(control.spec),'root':str(root)}
            if control.extra_features or control.spec.turnover_loss_weight!=.006:raise ValueError('control is not intended base family')
            candidate=model_class(base_features=[replacement.get(c,c) for c in control.base_features],extra_features=[],feature_masks=control.feature_masks,spec=spec,seed=seed,cutoff=str(control.cutoff),train_end=str(control.train_end.date()),minimum_observations=control.minimum_observations)
            identity=_initial_hash(candidate)
            if identity!=_initial_hash(control):raise ValueError('initialization differs')
            prediction=_prediction_dates(control.predict(dataset));saved=training['analysis/outer_predictions'][('base',seed)]
            np.testing.assert_allclose(prediction,saved,atol=1e-12,rtol=1e-12)
            if not prediction.index.equals(saved.index):raise ValueError('saved control dates differ')
            predictions[('control',seed)]=prediction
            name=f'candidate_{seed}'
            if name in progress['fitted']:
                candidate=rec.load_object('models/'+name)
            else:
                if len(progress['fitted'])>=plan['maximum_model_fits']:raise ValueError('fit budget exhausted')
                candidate.fit(candidate_dataset,auxiliary_dataset=auxiliary)
                rec.save_objects(artifact_path='models',**{name:candidate});progress['fitted'].append(name);atomic_json(progress,progress_path)
            for key in ['training_rows','first_training_signal','last_training_signal','last_mature_label_at','auxiliary_rows','epochs','checkpoint_policy']:
                if candidate.audit[key]!=control.audit[key]:raise ValueError('paired training domain differs: '+key)
            if candidate.common_scaler!=control.common_scaler:raise ValueError('unchanged common scaler differs')
            for kind,values in control.primary_scaler.items():
                for feature in control.base_features:
                    if feature not in replacement and candidate.primary_scaler[kind][feature]!=values[feature]:raise ValueError('untouched primary scaler differs')
            predictions[('candidate',seed)]=_prediction_dates(candidate.predict(candidate_dataset))
            training_rows.append({'seed':seed,'initialization_sha256':identity,'representation':representation,'encoder_routing':routing,'primary_dimensions':len(candidate.base_features),**candidate.audit})
            rec.log_metrics(completed_model_fits=len(progress['fitted']));print('fit',seed,flush=True)
        rec.save_objects(artifact_path='analysis',predictions=predictions,training=training_rows)
    frame=source['inputs/frame'];frame=frame.loc[frame.signal_date.between(plan['start'],plan['end'])].copy()
    market=action['inputs/market'];dates=pd.DatetimeIndex(market.signal_date);returns=market.set_index('signal_date')[['soxl_return','soxx_return']]
    targets={family:_smooth_weights(_ensemble(predictions,family,plan['seeds']).reindex(dates),spec=original.spec,candidate=original.candidate,asset_returns=returns) for family in ['control','candidate']}
    np.testing.assert_allclose(targets['control'],action['inputs/source_targets']['base'],rtol=1e-10,atol=1e-10)
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
            if family=='control':
                book=paired['analysis/native/'+key];rec.save_objects(artifact_path='analysis/native',**{key:book})
            elif key in progress['native']:
                book=rec.load_object('analysis/native/'+key)
            else:
                if len(progress['native'])>=plan['maximum_new_native_cases']:raise ValueError('native budget exhausted')
                replay=replay_full_path(data,models=None,plan=plan,slippage_bps=slip,controller=controller['inputs/controller'],recovery_spec=recovery,action_spec=action_spec,selection=action['controller/source_recovery_selection'])
                book=native_tiered_stage1(data,replay,provider_uri=provider,tariff=plan['tariff'],slippage_bps=slip,initial_cash=plan['initial_cash'],freq='1min',opening_offset=pd.Timedelta(hours=9,minutes=47),position_type=QlibContinuousPosition)
                book['targets']=replay['targets'];rec.save_objects(artifact_path='analysis/native',**{key:book});progress['native'].append(key);atomic_json(progress,progress_path)
            books[(family,slip)]=book;rec.log_metrics(completed_new_native_cases=len(progress['native']));print('account',key,flush=True)
    for slip in plan['slippage_bps']:
        for family in ['current_anchor','soxx']:
            books[(family,slip)]=source[f'analysis/native/cash_start_2023_{family}_{slip}'];rec.save_objects(artifact_path='analysis/native',**{f'{family}_{slip}':books[(family,slip)]})
    rows=[];annual=[];uncertainty=[]
    for (family,slip),book in books.items():
        path=book['authority'];benchmark=books[('soxx',slip)]['authority'];control=books[('control',slip)]['authority']
        if not path.trade_date.equals(benchmark.trade_date):raise ValueError('benchmark calendar differs')
        relative=np.log1p(path.strategy_return)-np.log1p(benchmark.strategy_return)
        rows.append({'family':family,'slippage_bps':slip,**paired_metrics(path,benchmark),'monitored_drawdown_pct':-100*path.monitored_drawdown.max(),'control_relative_wealth':float(path.equity.iloc[-1]/control.equity.iloc[-1]),'average_cash_pct':100*path.ending_cash_weight.mean(),'orders':len(book['orders']),'broker_fees_usd':path.broker_and_external_fees_usd.sum(),'slippage_usd':path.slippage_reserve_usd.sum(),'positive_126d_fraction':relative.rolling(126).sum().dropna().gt(0).mean()})
        annual.append(annual_paired_metrics(path,benchmark).assign(family=family,slippage_bps=slip))
        if family=='candidate':
            for comparator in ['soxx','control']:
                uncertainty.append({'slippage_bps':slip,'comparator':comparator,**stationary_bootstrap_excess(path,books[(comparator,slip)]['authority'],mean_block=63,resamples=2000,alpha=.025,seed=20261014)})
    table=pd.DataFrame(rows);annual=pd.concat(annual,ignore_index=True);uncertain=pd.DataFrame(uncertainty);candidate=table.loc[table.family.eq('candidate')]
    gates={'net_soxx_excess_all_costs':bool(candidate.relative_wealth.gt(1+1e-8).all()),'drawdown_30pct':bool(candidate.max_drawdown_pct.ge(-30).all() and candidate.monitored_drawdown_pct.ge(-30).all()),'cvar_no_worse':bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),'beat_paired_control_all_costs':bool(candidate.control_relative_wealth.gt(1+1e-8).all()),'rolling_70pct':bool(candidate.positive_126d_fraction.ge(.7).all()),'annual_70pct':bool(annual.loc[annual.family.eq('candidate')].groupby('slippage_bps').relative_wealth.apply(lambda x:x.gt(1).mean()).ge(.7).all()),'bootstrap_positive':bool(uncertain.lower_bound_positive.all()),'independent_development':False,'qualified_forward':False}
    gates['beat_original_all_costs']=bool(candidate.set_index('slippage_bps').terminal_wealth.gt(table.loc[table.family.eq('current_anchor')].set_index('slippage_bps').terminal_wealth).all())
    historical=all(v for k,v in gates.items() if k not in ['independent_development','qualified_forward'])
    summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'needs_independent_validation' if historical else plan.get('failure_decision','not_promoted_fixed_representation_failed'),'gates':gates,'objective_achieved':False,'new_fits':len(progress['fitted']),'new_native_cases':len(progress['native']),'reused_control_models':3,'reused_native_cases':9,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(provider),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/model.html','scope':plan.get('scope','Only primary momentum coordinates changed; original common encoder retained. Same dimension and initialization. Paired base models/accounts reused, original f696 and SOXX retained as comparators. Fixed final epoch and equal-weight three-seed ensemble; reused history, no independent development or forward qualification.'),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','models/*','analysis/predictions','analysis/native/*']}}
    rec.save_objects(artifact_path='analysis',summary=summary,comparison=table,annual=annual,uncertainty=uncertain,seed_cash=pd.DataFrame(seed_rows),targets=targets)
    artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    for name,value in [('backtest_benchmark_comparison',table),('annual_metrics',annual),('uncertainty',uncertain),('seed_cash',pd.DataFrame(seed_rows))]:value.to_csv(card/(name+'.csv'),index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

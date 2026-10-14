"""Fixed annual expanding refits, with continuous full-policy net-cost accounts."""
from pathlib import Path
from dataclasses import asdict,replace
import hashlib,json
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.workflow.minute_pit_allocator import _primary_dataset,_auxiliary_dataset,_prediction_dates,_ensemble
from src.workflow.pit_allocation_retrain import _training_auxiliary
from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.model.minute_pit_allocator import MinutePitAllocator
from src.model.tangent_allocator import TangentHeadAllocator,audit_tangent_initialization
from src.model.annual_allocator_clock import annual_decision_groups
from src.backtest.full_path_tiered import replay_full_path
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.analysis.integrated_allocation import paired_metrics,annual_paired_metrics,stationary_bootstrap_excess
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def auxiliary_fingerprint(provider):
    h=hashlib.sha256()
    for folder in ['calendars','instruments','features']:
        for p in sorted((Path(provider)/folder).rglob('*')):
            if p.is_file():h.update(str(p.relative_to(provider)).encode());h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('frozen plan changed')
        return old
    aux_provider=root/plan['auxiliary_provider']
    if auxiliary_fingerprint(aux_provider)!=plan['auxiliary_store_sha256']:raise ValueError('auxiliary source changed')
    training,ta=_source(root,plan['training_source'],['inputs/daily_frame','inputs/features','inputs/common','inputs/timing','inputs/plan',*[f'models/base_outer_{seed}' for seed in plan['seeds']]])
    frozen,fa=_source(root,plan['frozen_source'],['inputs/plan','analysis/predictions',*[f'models/candidate_{seed}' for seed in plan['seeds']],*[f'analysis/native/{family}_{cost}' for family in ['control','candidate','current_anchor','soxx'] for cost in plan['slippage_bps']]])
    source,sa=_source(root,plan['source'],['inputs/frame']);controller,ca=_source(root,plan['price_source'],['inputs/controller','inputs/plan']);action,aa=_source(root,plan['action_source'],['inputs/plan','inputs/market','controller/source_recovery_selection'])
    for key in ['tariff','initial_cash','start','end','as_of','slippage_bps','maximum_soxl_weight','minimum_cash_weight','maximum_drawdown','seeds','training_source','source','price_source','action_source']:
        if frozen['inputs/plan'][key]!=plan[key]:raise ValueError('frozen control context differs: '+key)
    original=training[f'models/base_outer_{plan["seeds"][0]}'];inputs={name:training['inputs/'+key] for name,key in [('frame','daily_frame'),('features','features'),('common','common'),('timing','timing')]};inputs['spec']=original.spec
    market=action['inputs/market'];dates=pd.DatetimeIndex(market.signal_date);groups=annual_decision_groups(inputs['timing'],dates)
    if sorted(groups)!=[2023,2024,2025,2026]:raise ValueError('unexpected annual domain')
    tracking=card/'qlib_recorder';progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':digest,'fitted':[],'auxiliary':[],'native':[]}
    if progress['plan_sha256']!=digest:raise ValueError('progress identity differs')
    prepared_auxiliary={}
    for year,group in groups.items():
        if year==2023:continue
        aux_key=str(year);train_end=inputs['features'].index[inputs['features'].index<group['cutoff'].tz_localize(None)].max()
        if aux_key in progress['auxiliary']:
            with R.uri_context(str(tracking)):
                prior=R.get_recorder(recorder_id=progress['recorder_id'],experiment_name=card.name)
                pair=(prior.load_object('inputs/auxiliary_'+aux_key),prior.load_object('inputs/auxiliary_audit_'+aux_key))
        else:pair=_training_auxiliary(replace(original.spec,root=root,development_end=str(train_end.date())))
        prepared_auxiliary[year]=pair
    import qlib
    qlib.init(provider_uri=str(root/'data/qlib/full_path_tiered_allocation_20261013'),region='us',kernels=1)
    kwargs={'experiment_name':card.name,'uri':str(tracking)}
    if progress.get('recorder_id'):kwargs.update(recorder_id=progress['recorder_id'],resume=True)
    parts={(f,seed):[] for f in ['annual_original','annual_tangent'] for seed in plan['seeds']};audits=[];clock_rows=[]
    with R.start(**kwargs):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audits={'training':ta,'frozen':fa,'prices':sa,'controller':ca,'action':aa})
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),annual_clock=(root/'src/model/annual_allocator_clock.py').read_text(),tangent_model=(root/'src/model/tangent_allocator.py').read_text(),base_model=(root/'src/model/minute_pit_allocator.py').read_text())
        for year,group in groups.items():
            cutoff=group['cutoff'];selected=group['signal_dates'];train_end=inputs['features'].index[inputs['features'].index<cutoff.tz_localize(None)].max();clock_rows.append({'decision_year':year,'cutoff':str(cutoff),'train_observation_end':str(train_end),'first_signal':str(selected.min()),'first_decision':str(inputs['timing'].decision_at.reindex(selected).min()),'last_signal':str(selected.max()),'forecast_rows':len(selected)})
            if year==2023:
                for seed in plan['seeds']:
                    for family,key in [('annual_original','control'),('annual_tangent','candidate')]:
                        p=frozen['analysis/predictions'][(key,seed)].reindex(selected);assert p.notna().all().all();parts[(family,seed)].append(p)
                continue
            aux_key=str(year)
            auxiliary_frame,aux_audit=prepared_auxiliary[year]
            if aux_key not in progress['auxiliary']:
                rec.save_objects(artifact_path='inputs',**{'auxiliary_'+aux_key:auxiliary_frame,'auxiliary_audit_'+aux_key:aux_audit});progress['auxiliary'].append(aux_key);atomic_json(progress,progress_path)
            auxiliary=_auxiliary_dataset(auxiliary_frame,aux_audit['raw_observation_end']);dataset=_primary_dataset(inputs,train_end=str(train_end.date()),predict_bounds=[str(selected.min().date()),str(selected.max().date())])
            for seed in plan['seeds']:
                pair={};spec={**asdict(original.spec),'root':str(root),'development_end':str(train_end.date())}
                for family,cls in [('annual_original',MinutePitAllocator),('annual_tangent',TangentHeadAllocator)]:
                    name=f'{family}_{year}_{seed}'
                    if name in progress['fitted']:model=rec.load_object('models/'+name)
                    else:
                        if len(progress['fitted'])>=plan['maximum_model_fits']:raise ValueError('fit budget exhausted')
                        model=cls(base_features=original.base_features,extra_features=[],feature_masks=original.feature_masks,spec=spec,seed=seed,cutoff=str(cutoff),train_end=str(train_end.date()),minimum_observations=original.minimum_observations)
                        model.fit(dataset,auxiliary_dataset=auxiliary);rec.save_objects(artifact_path='models',**{name:model});progress['fitted'].append(name);atomic_json(progress,progress_path)
                    if not pd.Timestamp(model.audit['last_mature_label_at'])<cutoff:raise ValueError('unmatured primary training label')
                    prediction=_prediction_dates(model.predict(dataset)).reindex(selected)
                    if prediction.isna().any().any():raise ValueError('missing annual prediction')
                    parts[(family,seed)].append(prediction);pair[family]=model;audits.append({'family':family,'decision_year':year,'seed':seed,**model.audit});rec.log_metrics(completed_model_fits=len(progress['fitted']));print('fit',name,flush=True)
                x,y=pair['annual_original'],pair['annual_tangent'];audit_tangent_initialization(x._initialize(),y._initialize())
                if x.primary_scaler!=y.primary_scaler or x.common_scaler!=y.common_scaler:raise ValueError('paired annual scalers differ')
                for key in ['training_rows','auxiliary_rows','epochs','last_mature_label_at','last_training_signal']:
                    if x.audit[key]!=y.audit[key]:raise ValueError('paired annual domain differs')
        predictions={k:pd.concat(v).sort_index().reindex(dates) for k,v in parts.items()}
        if any(p.isna().any().any() or p.index.has_duplicates for p in predictions.values()):raise ValueError('annual prediction coverage invalid')
        rec.save_objects(artifact_path='analysis',predictions=predictions,training=pd.DataFrame(audits),annual_clock=pd.DataFrame(clock_rows))
    frame=source['inputs/frame'];frame=frame.loc[frame.signal_date.between(plan['start'],plan['end'])].copy();returns=market.set_index('signal_date')[['soxl_return','soxx_return']]
    targets={f:_smooth_weights(_ensemble(predictions,f,plan['seeds']).reindex(dates),spec=original.spec,candidate=original.candidate,asset_returns=returns) for f in ['annual_original','annual_tangent']}
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);action_spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec']);provider=root/'data/qlib/full_path_tiered_allocation_20261013';books={}
    for family in targets:
        data=frame.copy();aligned=targets[family].reindex(pd.DatetimeIndex(data.signal_date))
        if aligned.isna().any().any():raise ValueError('missing native account targets')
        for c in aligned:data[c]=aligned[c].to_numpy()
        for cost in plan['slippage_bps']:
            key=f'{family}_{cost}'
            if key in progress['native']:book=rec.load_object('analysis/native/'+key)
            else:
                if len(progress['native'])>=plan['maximum_new_native_cases']:raise ValueError('account budget exhausted')
                replay=replay_full_path(data,models=None,plan=plan,slippage_bps=cost,controller=controller['inputs/controller'],recovery_spec=recovery,action_spec=action_spec,selection=action['controller/source_recovery_selection'])
                book=native_tiered_stage1(data,replay,provider_uri=provider,tariff=plan['tariff'],slippage_bps=cost,initial_cash=plan['initial_cash'],freq='1min',opening_offset=pd.Timedelta(hours=9,minutes=47),position_type=QlibContinuousPosition);book['targets']=replay['targets'];rec.save_objects(artifact_path='analysis/native',**{key:book});progress['native'].append(key);atomic_json(progress,progress_path)
            books[(family,cost)]=book;rec.log_metrics(completed_native_cases=len(progress['native']));print('account',key,flush=True)
    for cost in plan['slippage_bps']:
        for family,key in [('frozen_original','control'),('frozen_tangent','candidate'),('current_anchor','current_anchor'),('soxx','soxx')]:
            books[(family,cost)]=frozen[f'analysis/native/{key}_{cost}'];rec.save_objects(artifact_path='analysis/native',**{f'{family}_{cost}':books[(family,cost)]})
    rows=[];annual=[];uncertainty=[]
    for (family,cost),book in books.items():
        path=book['authority'];benchmark=books[('soxx',cost)]['authority'];assert path.trade_date.equals(benchmark.trade_date);relative=np.log1p(path.strategy_return)-np.log1p(benchmark.strategy_return)
        rows.append({'family':family,'slippage_bps':cost,**paired_metrics(path,benchmark),'monitored_drawdown_pct':-100*path.monitored_drawdown.max(),'average_cash_pct':100*path.ending_cash_weight.mean(),'positive_126d_fraction':relative.rolling(126).sum().dropna().gt(0).mean(),'orders':len(book['orders'])})
        annual.append(annual_paired_metrics(path,benchmark).assign(family=family,slippage_bps=cost))
        if family=='annual_tangent':
            for comparator in ['soxx','frozen_tangent','annual_original']:
                uncertainty.append({'slippage_bps':cost,'comparator':comparator,**stationary_bootstrap_excess(path,books[(comparator,cost)]['authority'],mean_block=63,resamples=2000,alpha=.025,seed=20261014)})
    table=pd.DataFrame(rows);annual=pd.concat(annual,ignore_index=True);uncertainty=pd.DataFrame(uncertainty);candidate=table.loc[table.family.eq('annual_tangent')].set_index('slippage_bps')
    gates={'net_soxx_excess_all_costs':bool(candidate.relative_wealth.gt(1+1e-8).all()),'drawdown_30pct':bool(candidate.max_drawdown_pct.ge(-30).all() and candidate.monitored_drawdown_pct.ge(-30).all()),'cvar_no_worse':bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),'annual_70pct':bool(annual.loc[annual.family.eq('annual_tangent')].groupby('slippage_bps').relative_wealth.apply(lambda x:x.gt(1).mean()).ge(.7).all()),'rolling_70pct':bool(candidate.positive_126d_fraction.ge(.7).all()),'bootstrap_positive':bool(uncertainty.lower_bound_positive.all()),'independent_development':False,'qualified_forward':False}
    for family in ['frozen_tangent','annual_original','frozen_original','current_anchor']:gates['beat_'+family+'_all_costs']=bool(candidate.terminal_wealth.gt(table.loc[table.family.eq(family)].set_index('slippage_bps').terminal_wealth).all())
    historical=all(v for k,v in gates.items() if k not in ['independent_development','qualified_forward']);summary={'experiment':card.name,'plan_sha256':digest,'decision':'needs_independent_validation' if historical else 'not_promoted_fixed_annual_refit_failed','gates':gates,'objective_achieved':False,'new_fits':len(progress['fitted']),'new_native_cases':len(progress['native']),'reused_models_2023':6,'reused_native_cases':12,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(provider),'region':'us','as_of':plan['as_of'],'scope':plan['scope'],'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','models/*','analysis/*']}}
    rec.save_objects(artifact_path='analysis',summary=summary,comparison=table,annual=annual,uncertainty=uncertainty,targets=targets);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    if auxiliary_fingerprint(aux_provider)!=plan['auxiliary_store_sha256']:raise ValueError('auxiliary source changed during run')
    assert hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()==plan['release_sha256']
    artifacts=Path(summary['recorder_local_dir'])/'artifacts';atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    for name,data in [('backtest_benchmark_comparison',table),('annual_metrics',annual),('uncertainty',uncertainty),('training_audit',pd.DataFrame(audits)),('annual_clock',pd.DataFrame(clock_rows))]:data.to_csv(card/(name+'.csv'),index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':card.name,'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False));summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=summary['experiment_id']);atomic_json(summary,card/'summary.json');return summary

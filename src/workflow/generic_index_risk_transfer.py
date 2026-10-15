"""Fixed six-fit family-data comparison with reused raw-accounting execution."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from dataclasses import asdict
from src.model.generic_index_risk_transfer import IndexTailAllocator
from src.model.minute_pit_allocator import TARGETS
from src.workflow.minute_pit_allocator import _primary_dataset,_prediction_dates
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES,_standardize_pair
from src.factor import build_constituent_2x_tail_training_frame
WEIGHTS=["soxl_weight","soxx_weight","cash_weight"]
from src.workflow.multi_3x_etf_normalized_nested_wf import _attach_normalized_features
from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
from src.workflow.minute_action_value import _source
from src.data.minute_intake import atomic_json
from src.backtest.raw_corporate_policy import replay_raw_package
from src.backtest.native_raw_corporate import native_raw_validate
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.data.nas_market_data import archive_nas_recorder

NAME='generic_index_risk_transfer_20261015'
PLAN_SHA='79528e43569e199bf608469b8869b6287f294d5f40f6a30237f0053fa8c0e4b9'

def external_labels(opening):
    daily=opening.shift(-2)/opening.shift(-1)-1
    forward=pd.concat([daily.shift(-j) for j in range(5)],axis=1)
    label=((1+forward).cumprod(axis=1)-1).min(axis=1).le(-.06).astype(float)
    label[forward.isna().any(axis=1)]=np.nan
    return label

def prepare_external(root,plan):
    import qlib
    from qlib.data import D
    groups={};audit={}
    for index,cfg in plan['external_indices'].items():
        provider=root/cfg['uri'];assert hashlib.sha256((provider/'qlib_store_manifest.json').read_bytes()).hexdigest()==cfg['manifest_sha256']
        qlib.init(provider_uri=str(provider),region='us',kernels=1)
        calendar=pd.DatetimeIndex(D.calendar(start_time='2010-03-11',end_time='2022-12-30',freq='day'))
        raw=D.features([index],['$open','$high','$low','$close','$volume'],start_time=calendar[0],end_time=calendar[-1],freq='day').reset_index()
        pd.testing.assert_index_equal(pd.DatetimeIndex(raw.datetime).rename(None),calendar.rename(None))
        if not np.isfinite(raw[['$open','$high','$low','$close','$volume']]).all().all() or not raw[['$open','$high','$low','$close']].gt(0).all().all():raise ValueError('invalid source prices')
        raw['$adj_close']=raw['$close'];raw['$pit_weight_pct']=100.;raw['$pit_available']=0.;raw['$pit_member']=1.;raw['$instrument_type']=0.;raw['$leverage_multiple']=1.;raw['$is_actual_2x']=0.
        features=build_constituent_2x_tail_training_frame(raw).set_index('datetime').sort_index()
        opening=raw.set_index('datetime')['$open'];features['tail_label']=external_labels(opening)
        features['mature_at']=pd.Series(calendar,index=calendar).shift(-6)+pd.Timedelta('9h31min')
        eligible=features.index.to_series().ge('2011-01-03') & features.mature_at.notna() & features.mature_at.lt(pd.Timestamp('2023-01-01'))
        g=features.loc[eligible,list(COMMON_FEATURES)+['tail_label','mature_at']].copy()
        if not g.tail_label.notna().all():raise ValueError('mature label unknown')
        groups[index]=g;audit[index]={'rows':len(g),'positives':int(g.tail_label.sum()),'first_signal':str(g.index[0]),'last_signal':str(g.index[-1]),'last_mature_at':str(g.mature_at.max()),'manifest_sha256':cfg['manifest_sha256']}
    return groups,audit

def run(root):
    import qlib,yaml
    from qlib.workflow import R
    from src.analysis.family_transfer_assessment import analyze_transfer
    root=Path(root).resolve();card=root/'research'/NAME;plan=json.loads((card/'plan.json').read_text())
    assert hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()==PLAN_SHA
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    qlib.init(provider_uri=str(root/'data/qlib/frozen_policy_raw_cashbook_20261015'),region='us',kernels=1)
    torch.set_num_threads(1)
    for f,h in plan['source_code_sha256'].items():assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h,f
    costs=[0,2,5]
    names=['inputs/frame','inputs/raw_points','inputs/schedule','inputs/plan']+[f'native/{family}_{c}' for family in ['raw_candidate27baa','raw_SOXX'] for c in costs]
    raw,ra=_source(root,plan['raw_source'],names)
    training,ta=_source(root,plan['training_source'],plan['training_artifacts'])
    ctrl,cta=_source(root,plan['controller_source'],['inputs/controller']);action,aa=_source(root,plan['action_source'],['inputs/plan','inputs/market','controller/source_recovery_selection'])
    frame=raw['inputs/frame'].reset_index(drop=True).copy();assert len(frame)==944
    dates=pd.DatetimeIndex(frame.signal_date);original=training['models/base_outer_71']
    if original.base_features!=plan['features']['primary_order'] or list(COMMON_FEATURES)!=plan['features']['common_order']:raise ValueError('feature order changed')
    inputs={name:training['inputs/'+key] for name,key in [('frame','daily_frame'),('features','features'),('common','common'),('timing','timing')]};inputs['spec']=original.spec
    dataset=_primary_dataset(inputs,train_end=str(original.train_end.date()),predict_bounds=training['inputs/plan']['outer_window'])
    cached=json.loads((card/'run_state.json').read_text()) if (card/'run_state.json').exists() else {}
    if cached.get('sources_saved'):
        with R.uri_context(str(card/'qlib_recorder')):
            saved=R.get_recorder(experiment_name=NAME,recorder_id=cached['recorder_id'])
            groups=saved.load_object('inputs/external_groups');data_audit=saved.load_object('inputs/data_audit')
    else:groups,data_audit=prepare_external(root,plan)
    primary=dataset.prepare('train',col_set=['feature','label','common','meta']);primary.index=primary.index.get_level_values('datetime')
    maturity=pd.to_datetime(primary[('meta','label_available_at')]);mask=maturity.notna()&maturity.lt(original.cutoff)&(primary.index<=original.train_end)&np.isfinite(primary['label'][list(TARGETS)]).all(axis=1)
    train=primary.loc[mask];assert len(train)==3014
    for g in groups.values():pd.testing.assert_index_equal(g.index.rename(None),pd.DatetimeIndex(train.index).rename(None))
    _,_,ps=_standardize_pair(train['feature'],train['feature'],original.base_features);_,_,cs=_standardize_pair(train['common'],train['common'],COMMON_FEATURES)
    scaler={'primary':ps,'common':cs};models={};initials={}
    for seed in plan['model']['seeds']:
        template=training[f'models/base_outer_{seed}']
        for arm in ['control','candidate']:
            models[(arm,seed)]=IndexTailAllocator(external_tail_enabled=arm=='candidate',base_features=template.base_features,extra_features=[],feature_masks={},spec={**asdict(template.spec),'root':str(root)},seed=seed,cutoff=str(template.cutoff),train_end=str(template.train_end.date()),minimum_observations=template.minimum_observations)
        left,right=models[('control',seed)]._initialize(),models[('candidate',seed)]._initialize()
        for k,v in left.state_dict().items():torch.testing.assert_close(v,right.state_dict()[k],rtol=0,atol=0)
        x=torch.zeros((2,39));z=torch.zeros((2,16))
        for u,v in zip(left(x,z),right(x,z)):torch.testing.assert_close(u,v,rtol=0,atol=0)
        initials[str(seed)]=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in left.state_dict().values())).hexdigest()
    if cached.get('sources_saved'):
        assert initials==saved.load_object('inputs/initialization')
        pd.testing.assert_frame_equal(train,saved.load_object('inputs/train'))
    provider=root/'data/qlib/frozen_policy_raw_cashbook_20261015';points=raw['inputs/raw_points'];schedule=raw['inputs/schedule'];execution=raw['inputs/plan']
    assert len(points)==3778 and points.index.max()==pd.Timestamp(frame.exit_date.iloc[-1])+pd.Timedelta('9h47min')
    qlib.init(provider_uri=str(provider),region='us',kernels=1)
    progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':PLAN_SHA,'fit_attempts':[],'fitted':[],'path_attempts':[],'paths_done':[],'native_attempts':[],'native_done':[]}
    assert progress['plan_sha256']==PLAN_SHA
    tracking=card/'qlib_recorder';kw={'experiment_name':NAME,'uri':str(tracking)}
    if progress.get('recorder_id'):kw.update(recorder_id=progress['recorder_id'],resume=True)
    predictions={}
    with R.start(**kw):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        if not progress.get('sources_saved'):
            files=['src/model/generic_index_risk_transfer.py','src/workflow/generic_index_risk_transfer.py',*plan['source_code_sha256']]
            rec.save_objects(artifact_path='sources',**{f.replace('/','__'):(root/f).read_text() for f in files})
            rec.save_objects(artifact_path='inputs',plan=plan,execution_plan=execution,frame=frame,raw_points=points,schedule=schedule,train=train,external_groups=groups,scaler=scaler,data_audit=data_audit,source_audits={'raw':ra,'training':ta,'controller':cta,'action':aa},initialization=initials)
            progress['sources_saved']=True;atomic_json(progress,progress_path)
        if not progress.get('setup_repair_source_saved'):
            rec.save_objects(artifact_path='sources/setup_repair_1',model=(root/'src/model/generic_index_risk_transfer.py').read_text(),workflow=Path(__file__).read_text(),receipt=json.loads((card/'implementation_repair.json').read_text()))
            progress['setup_repair_source_saved']=True;atomic_json(progress,progress_path)
        for seed in plan['model']['seeds']:
            for arm in ['control','candidate']:
                name=f'{arm}_{seed}';model=models[(arm,seed)]
                if name in progress['fitted']:model=rec.load_object('models/'+name)
                else:
                    if name in progress['fit_attempts']:
                        if name!='control_71' or progress.get('setup_retry_attempts'):raise ValueError('fit setup retry budget exhausted')
                        progress['setup_retry_attempts']=[name]
                    else:
                        if len(progress['fit_attempts'])>=6:raise ValueError('fit attempt budget exhausted')
                        progress['fit_attempts'].append(name)
                    atomic_json(progress,progress_path)
                    model.fit(dataset,auxiliary_dataset=groups);rec.save_objects(artifact_path='models',**{name:model});progress['fitted'].append(name);atomic_json(progress,progress_path)
                pred=_prediction_dates(model.predict(dataset)).reindex(dates)
                for key in ['median','mean','scale']:
                    pd.testing.assert_series_equal(pd.Series(model.primary_scaler[key]),ps[key],check_names=False)
                    pd.testing.assert_series_equal(pd.Series(model.common_scaler[key]),cs[key],check_names=False)
                if pred.isna().any().any():raise ValueError('missing model output')
                predictions[(arm,seed)]=pred;rec.save_objects(artifact_path='predictions',**{name:pred});rec.log_metrics(completed_fits=len(progress['fitted']));print('completed fit',name,model.audit,flush=True)
    returns=action['inputs/market'].set_index('signal_date')[['soxl_return','soxx_return']].reindex(dates)
    targets={arm:_smooth_weights(sum(predictions[(arm,s)] for s in plan['model']['seeds'])/3,spec=SimpleNamespace(smoothing_span=3,rebalance_threshold=.02),asset_returns=returns) for arm in ['control','candidate']}
    rec.save_objects(artifact_path='analysis',targets=targets)
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);action_spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    books={(f,c):raw[f'native/{f}_{c}'] for f in ['raw_candidate27baa','raw_SOXX'] for c in costs}
    for arm in ['control','candidate']:
        family='raw_'+arm;local=frame.copy()
        for col in targets[arm]:local[col]=targets[arm][col].to_numpy()
        if family in progress['paths_done']:paths=rec.load_object('paths/'+family)
        else:
            if family in progress['path_attempts'] or len(progress['path_attempts'])>=2:raise ValueError('path attempt budget exhausted')
            progress['path_attempts'].append(family);atomic_json(progress,progress_path)
            paths,trace=replay_raw_package(local,point_prices=points,schedule=schedule,plan=execution,controller=ctrl['inputs/controller'],recovery_spec=recovery,action_spec=action_spec,selection=action['controller/source_recovery_selection'])
            rec.save_objects(artifact_path='paths',**{family:paths});rec.save_objects(artifact_path='traces',**{family:trace});progress['paths_done'].append(family);atomic_json(progress,progress_path);print('completed raw package',family,flush=True)
        for cost in costs:
            name=f'{family}_{cost}'
            if name in progress['native_done']:book=rec.load_object('native/'+name)
            else:
                if name in progress['native_attempts'] or len(progress['native_attempts'])>=6:raise ValueError('native attempt budget exhausted')
                progress['native_attempts'].append(name);atomic_json(progress,progress_path)
                book=native_raw_validate(paths[cost],provider_uri=provider,point_prices=points,schedule=schedule,tariff=execution['tariff'],cost=cost,initial_cash=execution['initial_cash'])
                rec.save_objects(artifact_path='native',**{name:book});progress['native_done'].append(name);atomic_json(progress,progress_path);print('completed native',name,flush=True)
            books[(family,cost)]=book;rec.log_metrics(completed_native=len(progress['native_done']))
    results=analyze_transfer(books,frame,initial_cash=execution['initial_cash'],costs=costs)
    for name,value in results.items():
        if isinstance(value,pd.DataFrame):value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
        elif name=='rolling':rec.save_objects(artifact_path='analysis/rolling',**value)
    gates=results['gates'];passed=all(v for k,v in gates.items() if k not in ['fullminute_risk_qualified','qualified_forward','independent_development'])
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','models/*','predictions/*','paths/*','traces/*','native/*','analysis/rolling/*']}
    summary={'experiment':NAME,'plan_sha256':PLAN_SHA,'decision':'historical_candidate_requires_risk_and_independent_forward' if passed else 'not_promoted_fixed_index_tail_transfer','objective_achieved':False,'model_fits':6,'fit_invocation_attempts':len(progress['fit_attempts'])+len(progress.get('setup_retry_attempts',[])),'policy_paths':6,'native_validations':6,'new_fetches':0,'training_audit':data_audit,'gates':gates,'historical_PIT_qualified':False,'currency_contract':'USD gross paper assumption, noDRIP, same raw basis all arms','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy}
    rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['raw_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'plan':'plan.json','module':'src.workflow.generic_index_risk_transfer','provider_uri':str(provider),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');progress_path.rename(card/'execution_attempts.json');return summary

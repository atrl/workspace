"""Fixed six-fit family-data comparison with reused raw-accounting execution."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from src.model.family_balanced_direct_utility import FamilyUtilityModel,WEIGHTS
from src.workflow.multi_3x_etf_normalized_nested_wf import _attach_normalized_features
from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
from src.workflow.minute_action_value import _source
from src.data.minute_intake import atomic_json
from src.backtest.raw_corporate_policy import replay_raw_package
from src.backtest.native_raw_corporate import native_raw_validate
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.data.nas_market_data import archive_nas_recorder

NAME='family_balanced_direct_utility_training_20261015'
PLAN_SHA='766c18e6e706a3c8de716910cadf6189f7eb19cfb7c5c980f39658058a247b80'

def prepare(root,plan,dates):
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader
    features=plan['features']['ordered_model'];pieces=[];audits={};evaluation=None
    for asset,underlying in plan['families'].items():
        cfg=plan['providers'][asset];provider=root/cfg['uri']
        assert hashlib.sha256((provider/'qlib_store_manifest.json').read_bytes()).hexdigest()==cfg['manifest_sha256']
        qlib.init(provider_uri=str(provider),region='us',kernels=1)
        end='2026-10-08' if asset=='SOXL' else '2022-12-30'
        calendar=pd.DatetimeIndex(D.calendar(start_time='2010-03-11',end_time=end,freq='day'))
        fields=['$open','$high','$low','$close','$volume','$vix']
        data=D.features([asset,underlying],fields,start_time=calendar[0],end_time=calendar[-1],freq='day')
        a=data.xs(asset,level='instrument').sort_index();b=data.xs(underlying,level='instrument').sort_index()
        pd.testing.assert_index_equal(pd.DatetimeIndex(a.index).rename(None),pd.DatetimeIndex(calendar).rename(None));pd.testing.assert_index_equal(a.index,b.index)
        for v in [a,b]:
            if not np.isfinite(v[['$open','$high','$low','$close']]).all().all() or not v[['$open','$high','$low','$close']].gt(0).all().all():raise ValueError('invalid required source prices '+asset)
        raw=a.rename(columns=lambda c:c[1:]).copy();raw['underlying_open']=b['$open'];raw['underlying_close']=b['$close'];raw['datetime']=raw.index;raw['instrument']=asset;raw=raw.reset_index(drop=True)
        trainraw=raw[raw.datetime.le('2022-12-30')].copy();g=_attach_normalized_features(trainraw)
        sessions=pd.DatetimeIndex(g.datetime)
        g['decision_at']=pd.Series(sessions).shift(-1)+pd.Timedelta('9h29min');g['mature_at']=pd.Series(sessions).shift(-6)+pd.Timedelta('9h31min')
        g['leveraged_return']=g.open.shift(-2)/g.open.shift(-1)-1;g['underlying_return']=g.underlying_open.shift(-2)/g.underlying_open.shift(-1)-1
        mask=g.datetime.ge('2011-01-03') & g.mature_at.lt(pd.Timestamp('2023-01-01')) & g.mature_at.notna()
        train=g.loc[mask].copy()
        if not np.isfinite(train[['leveraged_return','underlying_return']]).all().all() or not train[['leveraged_return','underlying_return']].gt(-1).all().all():raise ValueError('invalid mature labels')
        if not (train.decision_at<train.mature_at).all():raise ValueError('bad label maturity')
        pieces.append(train);audits[asset]={'provider_manifest':cfg['manifest_sha256'],'rows':len(train),'first_signal':str(train.datetime.iloc[0]),'last_signal':str(train.datetime.iloc[-1]),'last_mature_at':str(train.mature_at.iloc[-1]),'source_end_read':end}
        if asset=='SOXL':
            full=_attach_normalized_features(raw).set_index('datetime');evaluation=full.reindex(dates).rename_axis("datetime").reset_index()
            if evaluation.instrument.isna().any():raise ValueError('missing fixed evaluation dates')
    train=pd.concat(pieces,ignore_index=True);primary=train[train.instrument.eq('SOXL')]
    x=primary[features].replace([np.inf,-np.inf],np.nan);median=x.median()
    if median.isna().any():raise ValueError('all-missing training feature')
    filled=x.fillna(median);scaler={'median':median,'mean':filled.mean(),'scale':filled.std(ddof=1).replace(0,1).fillna(1)}
    def static(g):
        idx=pd.MultiIndex.from_arrays([g.datetime,g.instrument],names=['datetime','instrument'])
        f=g[features].copy();f.index=idx
        lab=g.reindex(columns=['leveraged_return','underlying_return']).copy();lab.index=idx
        return pd.concat({'feature':f,'label':lab},axis=1)
    allrows=pd.concat([static(train),static(evaluation)]).sort_index()
    dataset=DatasetH(handler=DataHandlerLP(data_loader=StaticDataLoader(config=allrows),infer_processors=[],learn_processors=[]),segments={'train':('2011-01-03','2022-12-30'),'test':(str(dates[0].date()),str(dates[-1].date()))})
    assert len(dataset.prepare('train',col_set='feature'))==len(train)
    return dataset,train,evaluation,scaler,audits

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
    raw,ra=_source(root,plan['raw_source'],names);context,ca=_source(root,plan['market_context_source'],['inputs/frame'])
    ctrl,cta=_source(root,plan['controller_source'],['inputs/controller']);action,aa=_source(root,plan['action_source'],['inputs/plan','inputs/market','controller/source_recovery_selection'])
    frame=raw['inputs/frame'].reset_index(drop=True).copy();market=context['inputs/frame'].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[['signal_date','trade_date','exit_date']],market[['signal_date','trade_date','exit_date']]);assert len(frame)==944
    for col in ['alpha_probability','tail_probability']:frame[col]=market[col].to_numpy()
    dates=pd.DatetimeIndex(frame.signal_date);dataset,train,evaluation,scaler,data_audit=prepare(root,plan,dates)
    models={}
    for seed in plan['model']['seeds']:
        for arm,weights in plan['arms'].items():models[(arm,seed)]=FamilyUtilityModel(plan['features']['ordered_model'],scaler,seed,weights)
        assert models[('control',seed)].initial_sha==models[('candidate',seed)].initial_sha
        for model in [models[('control',seed)],models[('candidate',seed)]]:model.tensor(train);model.tensor(evaluation)
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
            files=['src/model/family_balanced_direct_utility.py','src/workflow/family_balanced_direct_utility.py','src/analysis/family_transfer_assessment.py',*plan['source_code_sha256']]
            rec.save_objects(artifact_path='sources',**{f.replace('/','__'):(root/f).read_text() for f in files})
            rec.save_objects(artifact_path='inputs',plan=plan,execution_plan=execution,frame=frame,raw_points=points,schedule=schedule,train=train,evaluation=evaluation,scaler=scaler,data_audit=data_audit,source_audits={'raw':ra,'context':ca,'controller':cta,'action':aa},initialization={str(k):m.initial_sha for k,m in models.items()})
            progress['sources_saved']=True;atomic_json(progress,progress_path)
        for seed in plan['model']['seeds']:
            for arm in plan['arms']:
                name=f'{arm}_{seed}';model=models[(arm,seed)]
                if name in progress['fitted']:model=rec.load_object('models/'+name)
                else:
                    if name in progress['fit_attempts'] or len(progress['fit_attempts'])>=6:raise ValueError('fit attempt budget exhausted')
                    progress['fit_attempts'].append(name);atomic_json(progress,progress_path)
                    model.fit(dataset);rec.save_objects(artifact_path='models',**{name:model});progress['fitted'].append(name);atomic_json(progress,progress_path)
                pred=model.predict(dataset).xs('SOXL',level='instrument').reindex(dates)
                if pred.isna().any().any():raise ValueError('missing model output')
                predictions[(arm,seed)]=pred;rec.save_objects(artifact_path='predictions',**{name:pred});rec.log_metrics(completed_fits=len(progress['fitted']));print('completed fit',name,model.audit,flush=True)
    returns=action['inputs/market'].set_index('signal_date')[['soxl_return','soxx_return']].reindex(dates)
    targets={arm:_smooth_weights(sum(predictions[(arm,s)] for s in plan['model']['seeds'])/3,spec=SimpleNamespace(smoothing_span=3,rebalance_threshold=.02),asset_returns=returns) for arm in plan['arms']}
    rec.save_objects(artifact_path='analysis',targets=targets)
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);action_spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    books={(f,c):raw[f'native/{f}_{c}'] for f in ['raw_candidate27baa','raw_SOXX'] for c in costs}
    for arm in plan['arms']:
        family='raw_'+arm;local=frame.copy()
        for col in WEIGHTS:local[col]=targets[arm][col].to_numpy()
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
    summary={'experiment':NAME,'plan_sha256':PLAN_SHA,'decision':'historical_candidate_requires_risk_and_independent_forward' if passed else 'not_promoted_fixed_family_data_comparison','objective_achieved':False,'model_fits':6,'policy_paths':6,'native_validations':6,'new_fetches':0,'training_audit':data_audit,'gates':gates,'historical_PIT_qualified':False,'currency_contract':'USD gross paper assumption, noDRIP, same raw basis all arms','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy}
    rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['raw_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'plan':'plan.json','module':'src.workflow.family_balanced_direct_utility','provider_uri':str(provider),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');progress_path.rename(card/'execution_attempts.json');return summary

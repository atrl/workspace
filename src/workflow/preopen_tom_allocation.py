"""Stage-separated, train-only paired allocation study for one preopen proxy."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
from src.model.minute_pit_allocator import TARGETS, _flat_index, _scaled
from src.model.preopen_tom_allocator import PreopenTomAllocator, PREOPEN_TOM_FEATURES, PREOPEN_TOM_MASKS, initialization_contract
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES, _standardize_pair

NAME='stage1_preopen_tom_20261018'
ARMS={'B_preopen_tom_context':'neutral','C_preopen_tom':'observed'}
SEEDS=[71,172,273]
CANDIDATE='C_preopen_tom'
REQUIRED=('A_active_research_replay','B_shared','B_preopen_tom_context','SOXX','strong9097')
FAMILIES=(*REQUIRED,CANDIDATE,'additional27baa')
DATES=['signal_date','trade_date','exit_date']
OUTPUTS=['soxl_weight','soxx_weight','cash_weight','tail_probability','alpha_probability']
CODE_FILES=['src/workflow/preopen_tom_allocation.py','src/data/preopen_tom_features.py','src/model/preopen_tom_allocator.py',
 'src/model/minute_pit_allocator.py','src/workflow/minute_pit_allocator.py','src/workflow/pit_constituent_auxiliary_training_allocation.py',
 'src/workflow/calendar_tom_source.py','src/workflow/aux_projection_preflight.py','src/workflow/stage1_domain_heads.py',
 'src/backtest/stage1_domain_heads_raw.py','src/backtest/native_raw_corporate.py','src/backtest/raw_corporate_policy.py',
 'src/backtest/qlib_continuous_position.py','src/backtest/ibkr_tiered.py','src/workflow/stage1_open_action_value.py',
 'src/workflow/intraday_recovery_direct_allocation.py','src/analysis/stage1_domain_heads.py','src/analysis/integrated_allocation.py',
 'src/core/recorder_sources.py','src/data/minute_intake.py','src/data/qlib_native.py','src/data/nas_market_data.py',
 'tests/test_preopen_tom_features.py','tests/test_preopen_tom_allocator.py','tests/test_preopen_tom_workflow.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def code_identity(root):return {name:sha(Path(root)/name) for name in CODE_FILES}


def _verify_bytes(root,source):
    card=root/source['card'];summary=json.loads((card/'summary.json').read_text())
    if summary['recorder_id']!=source['recorder_id'] or sha(card/'verification_manifest.json')!=source['inventory_sha256']:
        raise ValueError('source Recorder/manifest identity differs')
    if json.loads((card/'nas_binding.json').read_text())['evidence']['sha256']!=source['archive']:
        raise ValueError('source archive differs')
    for name,digest in source['artifact_sha256'].items():
        if sha(Path(summary['recorder_local_dir'])/'artifacts'/name)!=digest:raise ValueError('artifact bytes changed: '+name)


def _read(root,source):
    _verify_bytes(root,source)
    return load_verified_recorder_objects(root/source['card'],expected_recorder_id=source['recorder_id'],
        artifacts=source['artifacts'],expected_manifest_sha256=source['inventory_sha256'],
        expected_artifact_sha256=source['artifact_sha256'])


def audit_training_residency(dataset,*,expected_rows,expected_columns,maximum_signal_date,role):
    """Inspect all actual resident frames, not only a named Dataset segment."""
    if type(dataset).__name__!='DatasetH' or set(dataset.segments)!= {'train'}:
        raise ValueError('only a train-only native Dataset is permitted')
    handler=dataset.handler
    if type(handler).__name__!='DataHandlerLP' or type(handler.data_loader).__name__!='StaticDataLoader':
        raise ValueError('unexpected training handler/loader; never call setup_data')
    for key in ('_data','_infer','_learn'):
        if not isinstance(getattr(handler,key,None),pd.DataFrame):raise ValueError('missing self-contained handler cache: '+key)
    frames=[];seen=set()
    def walk(value,path):
        if isinstance(value,pd.DataFrame):
            idx=value.index.get_level_values('datetime') if isinstance(value.index,pd.MultiIndex) else value.index
            if not isinstance(idx,pd.DatetimeIndex) or idx.tz is not None:
                raise ValueError('resident training frame lacks exact naive date index: '+path)
            if len(value)!=expected_rows or len(value.columns)!=expected_columns or idx.max()>pd.Timestamp(maximum_signal_date):
                raise ValueError('resident training frame contains unexpected/future rows: '+path)
            frames.append({'path':path,'rows':len(value),'columns':len(value.columns),
                'first_date':str(idx.min()),'last_date':str(idx.max())})
            return
        if value is None or isinstance(value,(str,int,float,bool,bytes,Path,np.ndarray,torch.Tensor)):return
        if id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,dict):
            for key,item in value.items():walk(item,path+'.'+str(key))
        elif isinstance(value,(list,tuple)):
            for i,item in enumerate(value):walk(item,path+f'[{i}]')
        elif type(value).__module__.startswith('qlib.'):
            for key,item in vars(value).items():walk(item,path+'.'+key)
    loader=handler.data_loader
    if isinstance(getattr(loader,'_config',None),(str,Path)):
        raise ValueError('external loader reload reference forbidden')
    walk(dataset,'dataset')
    if len(frames)<3:raise ValueError('all handler resident caches must be checked')
    return {'role':role,'resident_frames':frames,'setup_data_called':False,'evaluation_object_deserialized':False}


def _new_model(old,mode,root):
    new=PreopenTomAllocator(mode=mode,base_features=old.base_features,spec={**asdict(old.spec),'root':str(root)},
        seed=old.seed,cutoff=str(old.cutoff),train_end=str(old.train_end.date()),minimum_observations=old.minimum_observations)
    if asdict(new.candidate)!=asdict(old.candidate):raise ValueError('complete original candidate/loss coefficients differ')
    if {k:v for k,v in asdict(new.spec).items() if k!='root'}!={k:v for k,v in asdict(old.spec).items() if k!='root'}:
        raise ValueError('original spec differs')
    for name in ('base_features','seed','cutoff','train_end','minimum_observations'):
        if getattr(new,name)!=getattr(old,name):raise ValueError('model metadata differs: '+name)
    if old.extra_features or len(old.base_features)!=39:raise ValueError('not the original39-input model')
    return new


def _native_dataset(frame,segments):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    dataset=DatasetH(handler=DataHandlerLP.from_df(frame),segments=segments)
    dataset.config(dump_all=True,recursive=True)
    return dataset


def _append_four(original,canonical):
    dates=original.index.get_level_values('datetime')
    if not dates.isin(canonical.index).all():raise ValueError('canonical proxy omitted original rows')
    aligned=canonical.reindex(dates)
    result=original.copy()
    for name in PREOPEN_TOM_MASKS:result[('feature',name)]=aligned[name].to_numpy(dtype=float)
    result[('feature','preopen_tom_centered')]=aligned.preopen_tom_proxy_nullable.to_numpy(dtype=float)-.5
    if list(result.columns[:len(original.columns)])!=list(original.columns):raise ValueError('original columns reordered')
    if len(result)!=len(original) or len(result.columns)!=len(original.columns)+4:raise ValueError('four-slot append changed rows')
    return result


def _original_tensors(train,aux,old):
    flat=_flat_index(train)
    primary,_,ps=_standardize_pair(flat['feature'],flat['feature'],old.base_features)
    common,_,cs=_standardize_pair(flat['common'],flat['common'],COMMON_FEATURES)
    p={k:v.to_dict() for k,v in ps.items()};c={k:v.to_dict() for k,v in cs.items()}
    tensors={'primary':primary,'common':common,'auxiliary':_scaled(aux['feature'],list(COMMON_FEATURES),c),
      'returns':torch.tensor(flat['label'].loc[:,list(TARGETS[:2])].to_numpy(np.float32)),
      'tail':torch.tensor(flat[('label','tail_event_5d')].to_numpy(np.float32)),
      'alpha':torch.tensor(flat[('label','relative_alpha_5d')].gt(0).to_numpy(np.float32)),
      'cash':torch.tensor(flat[('label','tail_cash_target_5d')].to_numpy(np.float32)),
      'years':torch.tensor(flat.index.year.to_numpy(),dtype=torch.int64),
      'aux_tail':torch.tensor(aux[('label','synthetic_2x_crash_label')].to_numpy(np.float32)),
      'aux_alpha':torch.tensor(aux[('label','synthetic_2x_upside_label')].to_numpy(np.float32)),
      'aux_weight':torch.tensor(aux[('label','training_weight')].to_numpy(np.float32))}
    tensors['aux_weight']/=tensors['aux_weight'].mean()
    return tensors,p,c


def _identifiability(canonical,dates):
    data=canonical.reindex(dates);rows=[]
    for pattern,group in data.groupby(list(PREOPEN_TOM_MASKS),dropna=False):
        values=group.preopen_tom_proxy_nullable
        rows.append({**dict(zip(PREOPEN_TOM_MASKS,[bool(x) for x in pattern])), 'rows':len(group),
            'known_zero':int(values.eq(0).sum()),'known_one':int(values.eq(1).sum()),'unknown':int(values.isna().sum())})
    table=pd.DataFrame(rows)
    return table,bool(((table.known_zero>0)&(table.known_one>0)).any())


def _common_guards(root,card,plan):
    if not plan.get('frozen_before_preflight'):raise ValueError('executable plan not frozen')
    for path,digest in plan['frozen_review_files'].items():
        if sha(root/path)!=digest:raise ValueError('frozen review/contract changed: '+path)
    if sha(root/plan['active_release']['path'])!=plan['active_release']['sha256']:raise ValueError('active release changed')
    from src.workflow.aux_projection_preflight import _software
    torch.set_num_threads(1)
    if _software()!=plan['software']:raise ValueError('backend differs from matched source')


def _stage_approval(root,card,plan,stage):
    _common_guards(root,card,plan)
    file='preflight_execution_approval.json' if stage=='preflight' else 'execution_approval.json'
    env='PREOPEN_TOM_PREFLIGHT' if stage=='preflight' else 'PREOPEN_TOM_EXECUTE'
    if os.environ.get(env)!='1':raise ValueError(stage+' disabled')
    approval=json.loads((card/file).read_text());digest=sha(card/'plan.json')
    review_file='preflight_design_review.json' if stage=='preflight' else 'design_review.json'
    review=json.loads((card/review_file).read_text())
    verdict='APPROVE_NO_FIT_PREFLIGHT' if stage=='preflight' else 'APPROVE_DESIGN'
    if review.get('verdict')!=verdict or review.get('plan_sha256')!=digest or review.get('code_sha256')!=code_identity(root):
        raise ValueError(stage+' independent code/plan review absent or stale')
    if approval.get('review_sha256')!=sha(card/review_file):raise ValueError('stage review identity differs')
    if approval.get('approved') is not True or approval.get('plan_sha256')!=digest or approval.get('code_sha256')!=code_identity(root):
        raise ValueError(stage+' owner approval absent/stale')
    if stage=='preflight' and not (approval.get('source_feature_generation') is True and approval.get('training_only_no_fit') is True
                                  and approval.get('fits') is False and approval.get('accounts') is False):
        raise ValueError('preflight stage permissions differ')
    if stage=='fit':
        expected={'preflight_manifest_sha256':sha(card/'preflight/verification_manifest.json'),
                  'preflight_summary_sha256':sha(card/'preflight/summary.json')}
        if any(approval.get(key)!=value or review.get(key)!=value for key,value in expected.items()):
            raise ValueError('both final design review and execution GO must bind completed preflight')
    return {'approval_sha256':sha(card/file),'plan_sha256':digest,'code_sha256':code_identity(root)}


def _seal(rec,root,card,summary,source_card):
    from src.data.nas_market_data import archive_nas_recorder
    artifact_root=Path(summary['recorder_local_dir'])/'artifacts'
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':sha(p),'bytes':p.stat().st_size}
        for p in sorted(artifact_root.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    transport=json.loads((root/source_card/'nas_binding.json').read_text())
    atomic_json({'schema_version':'recorder_evidence_binding_v1','authority':'yanxuan_nas','ssh':transport['ssh'],
        'source_id':NAME+'-'+summary['plan_sha256'],'as_of':'2026-10-09'},card/'nas_binding.json')
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,
        recorder_id=rec.id,experiment_id=str(rec.experiment_id))
    atomic_json(summary,card/'summary.json')
    return summary


def prepare(*,root,card):
    """Only after separate GO: derive calendar values and inspect train-only objects."""
    import qlib
    from qlib.workflow import R
    from src.data.preopen_tom_features import build_preopen_tom_features
    root,card=Path(root),Path(card);plan=json.loads((card/'plan.json').read_text())
    approval=_stage_approval(root,card,plan,'preflight');out=card/'preflight';out.mkdir(exist_ok=True)
    attempt=out/'run_state.json'
    if attempt.exists():raise ValueError('preflight attempt exists; no silent repeat')
    state={'plan_sha256':approval['plan_sha256'],'status':'started','new_fits':0,'new_accounts':0};atomic_json(state,attempt)
    try:
        qlib.init(provider_uri=str(root/plan['provider_init']),region='us',kernels=1)
        calendar,audit_calendar=_read(root,plan['preflight_sources']['calendar'])
        source_plan=json.loads((root/plan['calendar_source_plan']['path']).read_text())
        if sha(root/plan['calendar_source_plan']['path'])!=plan['calendar_source_plan']['sha256']:raise ValueError('source plan changed')
        canonical=build_preopen_tom_features(calendar=calendar['source/actual_calendar'],timing=calendar['source/original_timing'],
            ledger=calendar['source/knowledge_ledger'],amendment_gaps=source_plan['calendar_amendment_gaps'],
            as_of='2026-10-09',prefix_start=plan['prefix_start'],prefix_gaps=tuple(plan['prefix_gaps']))
        pd.testing.assert_index_equal(canonical.index,calendar['source/original_timing'].index,check_names=False)
        pd.testing.assert_series_equal(canonical.original_decision_at,calendar['source/original_timing'].decision_at,check_names=False)
        from src.workflow.calendar_tom_source import _mapping_check
        mapping=_mapping_check(calendar['source/original_timing'],calendar['source/actual_calendar'])
        # Account eligibility is descriptive metadata, never a proxy mask input.
        canonical['account_eligible']=(canonical.index>=pd.Timestamp('2023-01-03'))&(canonical.index<=pd.Timestamp('2026-10-07'))
        if len(canonical.loc['2023-01-03':'2026-10-09'])!=946 or int(canonical.account_eligible.sum())!=944:
            raise ValueError('canonical row/tail identity differs')
        primary,audit_primary=_read(root,plan['preflight_sources']['training_only'])
        originals,audit_models=_read(root,plan['preflight_sources']['original_models'])
        sealed,audit_tensors=_read(root,plan['preflight_sources']['mechanical_tensors'])
        dataset,auxiliary=primary['inputs/training_dataset'],primary['inputs/auxiliary_dataset']
        residency=[audit_training_residency(dataset,expected_rows=3014,expected_columns=85,maximum_signal_date='2022-12-21',role='primary'),
          audit_training_residency(auxiliary,expected_rows=20741,expected_columns=20,maximum_signal_date='2022-12-22',role='auxiliary')]
        train=dataset.prepare('train',col_set='__all');aux=auxiliary.prepare('train',col_set='__all')
        dates=train.index.get_level_values('datetime');maturity=train[('meta','label_available_at')]
        if not (maturity.notna() & maturity.lt(pd.Timestamp(plan['training']['cutoff']))).all() or not np.isfinite(train['label'][list(TARGETS)]).all().all():
            raise ValueError('original mature/finite training cohort differs')
        if pd.to_datetime(aux[('meta','observations_end')]).max()>pd.Timestamp(plan['training']['train_end']):raise ValueError('aux maturity differs')
        tensors,ps,cs=_original_tensors(train,aux,originals['models/base_outer_71'])
        for name,value in tensors.items():
            if not torch.equal(value,sealed['inputs/tensors'][name]):raise ValueError('original training tensor differs: '+name)
            if hashlib.sha256(value.numpy().tobytes()).hexdigest()!=primary['inputs/training_audit']['tensor_sha256'][name]:
                raise ValueError('training tensor hash differs: '+name)
        augmented=_append_four(train,canonical)
        paired_dataset=_native_dataset(augmented,{'train':('2011-01-03','2022-12-30')})
        auxiliary.config(dump_all=True,recursive=True)
        initial=[]
        for seed in SEEDS:
            old=originals[f'models/base_outer_{seed}']
            if old.seed!=seed or old.train_end!=pd.Timestamp('2022-12-30') or old.cutoff!=pd.Timestamp(plan['training']['cutoff']) or old.spec.epochs!=300:
                raise ValueError('source model training cutoff/seed/final epoch differs')
            if ps!=old.primary_scaler or cs!=old.common_scaler:raise ValueError('original full scaler differs')
            left,right=_new_model(old,'neutral',root),_new_model(old,'observed',root)
            left._fit_extra_scaler(augmented['feature']);right._fit_extra_scaler(augmented['feature'])
            if not torch.equal(left._extra_tensor(augmented['feature'])[:,:3],right._extra_tensor(augmented['feature'])[:,:3]):
                raise ValueError('paired masks differ')
            initial.append({'seed':seed,**initialization_contract(old,left,right)})
        support,identifiable=_identifiability(canonical,dates)
        coverage=[]
        for basis,years in [('signal_year',pd.Series(canonical.index.year,index=canonical.index)),
                            ('decision_year',canonical.decision_date.dt.year)]:
            counted=pd.DataFrame({'year':years,'known':canonical.proxy_known.astype(int),
                'zero':canonical.preopen_tom_proxy_nullable.eq(0).astype(int),
                'one':canonical.preopen_tom_proxy_nullable.eq(1).astype(int)})
            table=counted.groupby('year',dropna=False).agg(rows=('known','size'),known=('known','sum'),
                known_zero=('zero','sum'),known_one=('one','sum')).reset_index()
            table['basis']=basis;coverage.append(table)
        coverage=pd.concat(coverage,ignore_index=True)
        for source in plan['evaluation_sources'].values():_verify_bytes(root,source)  # no deserialization
        index=pd.MultiIndex.from_arrays([canonical.index,['NYSE_PREOPEN_TOM']*len(canonical)],names=['datetime','instrument'])
        native_canonical=canonical.copy();native_canonical.index=index
        feature_dataset=_native_dataset(pd.concat({'meta':native_canonical},axis=1),{'all':(None,None)})
        tracking=out/'qlib_recorder'
        with R.start(experiment_name=NAME+'_preflight',uri=str(tracking)):
            rec=R.get_recorder();state.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(state,attempt)
            rec.save_objects(artifact_path='inputs',plan=plan,approval=approval,primary_dataset=paired_dataset,auxiliary_dataset=auxiliary,
                canonical=canonical,canonical_dataset=feature_dataset,actual_calendar=calendar['source/actual_calendar'],
                original_models=originals,training_audit=primary['inputs/training_audit'])
            rec.save_objects(artifact_path='analysis',residency=residency,initialization=initial,within_mask_support=support,
                coverage=coverage,sources={'calendar':audit_calendar,'training':audit_primary,'models':audit_models,'tensors':audit_tensors})
            for name,expected,segment in [('primary_dataset',augmented,'train'),('auxiliary_dataset',aux,'train'),
                                         ('canonical_dataset',pd.concat({'meta':native_canonical},axis=1),'all')]:
                pd.testing.assert_frame_equal(rec.load_object('inputs/'+name).prepare(segment,col_set='__all'),expected)
            summary={'experiment':NAME+'_preflight','plan_sha256':approval['plan_sha256'],'status':'completed',
                'decision':'eligible_for_formal_design_review' if identifiable else 'nonidentifiable_fixed_definition',
                'fit_eligible':identifiable,'primary_rows':3014,'auxiliary_rows':20741,'original_columns':85,'augmented_columns':89,
                'all_original_training_tensors_scalers_exact':True,'all_resident_frames_train_only':True,
                'canonical_rows':len(canonical),'outer_rows':946,'account_rows':944,'evaluation_objects_deserialized':0,
                'timestamp_only_mapping':mapping,
                'new_fits':0,'new_accounts':0,'new_source_HTTP':0,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),
                'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),
                'initialization':initial,'artifact_policy':plan['artifact_policy'],'historical_PIT_qualified':False}
            rec.save_objects(artifact_path='analysis',summary=summary)
            rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in (tracking/str(rec.experiment_id)/rec.id/'artifacts').rglob('*')))
        support.to_csv(card/'preflight_mask_support.csv',index=False)
        coverage.to_csv(card/'preflight_coverage.csv',index=False)
        summary=_seal(rec,root,out,summary,plan['preflight_sources']['training_only']['card'])
        atomic_json(summary,card/'preflight.json');state.update(status='completed',fit_eligible=identifiable);atomic_json(state,attempt)
        (card/'conclusion.md').write_text('# 09:29条件日历代理：无拟合预检\n\n实际完成固定source-feature和train-only检查；未拟合模型、未读评价标签/收益或回放账户。原3014/20741、85/20列与全部训练tensor/scaler核验一致，新主Dataset只追加4列。全部驻留handler/loader帧均在训练范围，未调用setup_data。\n\n'+
            '同mask组内0/1数值支持：'+str(identifiable)+'；若false，固定定义不可辨识并停止，不能改mask/窗口/venue救援。覆盖按signal年及decision年分别保留，unknown行不删除。\n\n'+
            '结论：'+summary['decision']+'。正式6fits/6原生账户仍需独立完整设计评审和owner GO；本结果不证明经济有效性或PIT资格。\n\nPreflight Recorder：`'+rec.id+'`。\n')
        return summary
    except BaseException as error:
        state.update(status='failed',exception_type=type(error).__name__,exception=str(error));atomic_json(state,attempt)
        raise


def _own_preflight_source(card):
    pre=card/'preflight';s=json.loads((pre/'summary.json').read_text());m=json.loads((pre/'verification_manifest.json').read_text())
    if not s.get('fit_eligible'):raise ValueError('fixed numeric proxy is nonidentifiable; no rescue fit')
    names=['inputs/primary_dataset','inputs/auxiliary_dataset','inputs/canonical','inputs/actual_calendar',
           'inputs/original_models','inputs/training_audit']
    return {'card':str(pre),'recorder_id':s['recorder_id'],'archive':json.loads((pre/'nas_binding.json').read_text())['evidence']['sha256'],
        'inventory_sha256':sha(pre/'verification_manifest.json'),'artifacts':names,
        'artifact_sha256':{name:next(x['sha256'] for x in m['entries'] if x['path'].endswith('/artifacts/'+name)) for name in names}}


def _claim(state,path,phase,key,maximum):
    if key in state[phase+'_attempts'] or len(state[phase+'_attempts'])>=maximum:raise ValueError('consumed '+phase+' attempt; no hidden retry')
    state[phase+'_attempts'].append(key);atomic_json(state,path)


def _model_barrier(rec,artifact_root,state):
    expected=[f'{family}_{seed}' for family in ARMS for seed in SEEDS]
    if state['fits_done']!=expected or set(state['model_sha256'])!=set(expected):raise ValueError('six-model barrier is incomplete')
    for key in expected:
        if sha(artifact_root/'models'/key)!=state['model_sha256'][key]:raise ValueError('sealed final model changed: '+key)
    return {key:rec.load_object('models/'+key) for key in expected}


def _evaluate(root,plan,models,prepared):
    """Only called after all six final model hashes are durable and verified."""
    from src.workflow.minute_pit_allocator import _prediction_dates, _ensemble
    from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
    from src.analysis.stage1_domain_heads import information_metrics
    if len(models)!=6:raise ValueError('incomplete model set')
    sources,audits={},{}
    for name,binding in plan['evaluation_sources'].items():sources[name],audits[name]=_read(root,binding)
    original=sources['prediction_dataset']['evaluation/dataset'].prepare('predict',col_set='__all')
    if len(original)!=946 or len(original.columns)!=85:raise ValueError('original evaluation dataset differs')
    augmented=_append_four(original,prepared['inputs/canonical'])
    dataset=_native_dataset(augmented,{'predict':('2023-01-03','2026-10-09')})
    predictions={(family,seed):_prediction_dates(models[f'{family}_{seed}'].predict(dataset)) for family in ARMS for seed in SEEDS}
    for values in predictions.values():
        if len(values)!=946 or not values.index.equals(prepared['inputs/canonical'].loc['2023-01-03':'2026-10-09'].index):
            raise ValueError('prediction rows/tails changed')
    ensembles={family:_ensemble(predictions,family,SEEDS) for family in ARMS}
    a=sources['account_inputs'];frame=a['inputs/frame'].reset_index(drop=True).copy();market=a['inputs/market'].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[DATES],market[DATES])
    points=a['inputs/raw_points'];schedule=a['inputs/schedule'];raw=sources['strong_native']
    pd.testing.assert_frame_equal(points,raw['inputs/raw_points'])
    if plan['evaluation_sources']['account_inputs']['artifact_sha256']['inputs/schedule']!=plan['evaluation_sources']['strong_native']['artifact_sha256']['inputs/schedule']:
        raise ValueError('corporate schedule identity differs')
    if len(frame)!=944 or len(points)!=3778:raise ValueError('original raw account calendar differs')
    dates=pd.DatetimeIndex(frame.signal_date);canonical=prepared['inputs/canonical'].reindex(dates)
    if not canonical.account_eligible.eq(True).all() or canonical.original_decision_at.isna().any():raise ValueError('unexecutable tail entered account')
    mapped=pd.DatetimeIndex(canonical.original_decision_at.dt.tz_localize(None).dt.normalize())
    if not mapped.equals(pd.DatetimeIndex(frame.trade_date)):raise ValueError('preopen proxy-to-account date mismatch')
    for source_plan in (a['inputs/plan'],raw['inputs/plan']):
        account=source_plan.get('accounts',source_plan)
        if source_plan['tariff']!=plan['tariff'] or account['initial_cash']!=100000 or account['slippage_bps']!=[0,2,5]:
            raise ValueError('reference cost/capital contract differs')
    references={}
    for item in plan['reference_books']:
        book=sources[item['source_role']][item['artifact']]
        pd.testing.assert_frame_equal(book['authority'][DATES].reset_index(drop=True),frame[DATES])
        if book['audit']['status']!='passed' or not pd.DatetimeIndex(book['points'].stamp).equals(points.index):raise ValueError('reference native/grid differs')
        np.testing.assert_allclose(book['points'].iloc[0].nav,100000,rtol=0,atol=1e-8)
        references[(item['family'],item['cost'])]=book
    if len(references)!=15:raise ValueError('all fifteen sealed reference books required')
    returns=market.set_index('signal_date')[['soxl_return','soxx_return']]
    base=prepared['inputs/original_models']['models/base_outer_71']
    targets={family:_smooth_weights(values.reindex(dates),spec=base.spec,candidate=base.candidate,asset_returns=returns)
             for family,values in ensembles.items()}
    info=information_metrics(predictions,ensembles,dataset,as_of=plan['accounts']['terminal'])
    e={'plan':plan,'frame':frame,'market':market,'points':points,'schedule':schedule,'guard':a['inputs/guard'],
       'controller':a['inputs/controller'],'selection':a['inputs/selection'],'recovery_spec':a['inputs/recovery_spec'],
       'action_spec':a['inputs/action_spec'],'references':references,
       'daily_calendar_frame':pd.DataFrame({'trade_date':prepared['inputs/actual_calendar']})}
    return e,dataset,predictions,ensembles,targets,info,audits


def run(*,root,card):
    import qlib
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder
    root,card=Path(root),Path(card);plan=json.loads((card/'plan.json').read_text())
    approval=_stage_approval(root,card,plan,'fit');digest=approval['plan_sha256'];state_path=card/'run_state.json'
    if state_path.exists():
        prior=json.loads(state_path.read_text())
        if prior.get('completed'):return json.loads((card/'summary.json').read_text())
        raise ValueError('formal attempt already exists; explicit scoped recovery required')
    qlib.init(provider_uri=str(root/plan['provider_init']),region='us',kernels=1)
    prepared,prefit_audit=_read(root,_own_preflight_source(card))
    audit_training_residency(prepared['inputs/primary_dataset'],expected_rows=3014,expected_columns=89,maximum_signal_date='2022-12-21',role='augmented_primary')
    audit_training_residency(prepared['inputs/auxiliary_dataset'],expected_rows=20741,expected_columns=20,maximum_signal_date='2022-12-22',role='auxiliary')
    state={'plan_sha256':digest,'fit_attempts':[],'fits_done':[],'path_attempts':[],'paths_done':[],
           'native_attempts':[],'native_done':[],'model_sha256':{},'phase':'training'}
    atomic_json(state,state_path);tracking=card/'qlib_recorder'
    with R.start(experiment_name=NAME,uri=str(tracking)):
        rec=R.get_recorder();state.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(state,state_path)
        artifacts=tracking/str(rec.experiment_id)/rec.id/'artifacts'
        rec.save_objects(artifact_path='inputs',plan=plan,approval=approval,preflight_audit=prefit_audit,
            primary_dataset=prepared['inputs/primary_dataset'],auxiliary_dataset=prepared['inputs/auxiliary_dataset'],
            canonical=prepared['inputs/canonical'])
        rec.save_objects(artifact_path='sources',**{name.replace('/','__'):(root/name).read_text() for name in CODE_FILES})
        training=[]
        for family,mode in ARMS.items():
            for seed in SEEDS:
                key=f'{family}_{seed}';_claim(state,state_path,'fit',key,6)
                old=prepared['inputs/original_models'][f'models/base_outer_{seed}'];model=_new_model(old,mode,root)
                try:
                    model.fit(prepared['inputs/primary_dataset'],auxiliary_dataset=prepared['inputs/auxiliary_dataset'])
                    for name in ('training_rows','auxiliary_rows','last_mature_label_at','epochs','checkpoint_policy'):
                        if model.audit[name]!=old.audit[name]:raise ValueError('original fit contract differs: '+name)
                    if model.primary_scaler!=old.primary_scaler or model.common_scaler!=old.common_scaler:raise ValueError('original scaler drift')
                    rec.save_objects(artifact_path='models',**{key:model})
                    state['model_sha256'][key]=sha(artifacts/'models'/key);state['fits_done'].append(key);atomic_json(state,state_path)
                    training.append({'family':family,'seed':seed,**model.audit});rec.log_metrics(completed_fits=len(state['fits_done']))
                    print('completed fit',key,flush=True)
                except BaseException as error:
                    failed={'exception_type':type(error).__name__,'message':str(error),'model_audit':getattr(model,'audit',None),
                        'optimizer_state_available':False,'optimizer_state_reconstructed':False,'completed_step_count':'not exposed by unchanged original fit'}
                    if getattr(model,'network',None) is not None:failed['network_state']={k:v.detach().clone() for k,v in model.network.state_dict().items()}
                    rec.save_objects(artifact_path='failed_fits',**{key:failed})
                    state.update(failed=True,failed_phase='training',exception=str(error));atomic_json(state,state_path)
                    raise
        models=_model_barrier(rec,artifacts,state)
        state.update(phase='models_fixed_before_evaluation',all_six_models_fixed=True,evaluation_unlocked=True);atomic_json(state,state_path)
        rec.save_objects(artifact_path='inputs',evaluation_unlock={'model_sha256':state['model_sha256'],
            'evaluation_label_return_prediction_objects_read_before_barrier':0})
    # Source readers and native validators must run without active global R.start.
    rec.start_run()
    try:
        state['phase']='evaluation';atomic_json(state,state_path)
        e,dataset,predictions,ensembles,targets,info,audits=_evaluate(root,plan,models,prepared)
        rec.save_objects(artifact_path='evaluation',dataset=dataset,prepared=e,source_audits=audits)
        rec.save_objects(artifact_path='predictions',**{f'{f}_{s}':v for (f,s),v in predictions.items()})
        rec.save_objects(artifact_path='analysis',ensembles=ensembles,smoothed_targets=targets,classification=info,training=pd.DataFrame(training))
        from src.workflow.stage1_domain_heads import _provider
        from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
        from src.backtest.native_raw_corporate import native_raw_validate
        from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
        from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
        from src.analysis.stage1_domain_heads import analyze_paired_accounts
        provider,manifest=_provider(root,e);rec.save_objects(artifact_path='inputs',native_provider_manifest=manifest)
        recovery=IntradayRecoveryDirectAllocationSpec(root=root,**e['recovery_spec']);action=Stage1OpenActionValueSpec(root=root,**e['action_spec'])
        execution={'initial_cash':100000,'tariff':plan['tariff']};books=dict(e['references'])
        for family in ARMS:
            frame=e['frame'].copy()
            for name in OUTPUTS:frame[name]=targets[family][name].to_numpy(copy=True)
            for cost in (0,2,5):
                key=f'{family}_{cost}';state['phase']='path';_claim(state,state_path,'path',key,6)
                path=replay_stage1_raw(frame,point_prices=e['points'],schedule=e['schedule'],plan=execution,controller=e['controller'],
                    recovery_spec=recovery,action_spec=action,selection=e['selection'],slippage_bps=cost)
                rec.save_objects(artifact_path='paths',**{key:path});state['paths_done'].append(key);atomic_json(state,state_path)
                state['phase']='native';_claim(state,state_path,'native',key,6)
                book=native_raw_validate(path,provider_uri=provider,point_prices=e['points'],schedule=e['schedule'],tariff=plan['tariff'],cost=cost,initial_cash=100000)
                rec.save_objects(artifact_path='native',**{key:book});state['native_done'].append(key);atomic_json(state,state_path)
                books[(family,cost)]=book;rec.log_metrics(completed_native_accounts=len(state['native_done']))
                print('completed native',key,flush=True)
        state['phase']='analysis';state['analysis_attempted']=True;atomic_json(state,state_path)
        results=analyze_paired_accounts(books,e['daily_calendar_frame'],plan,candidate_family=CANDIDATE,required_comparators=REQUIRED,expected_families=FAMILIES)
        for name,value in results.items():
            if name=='rolling':rec.save_objects(artifact_path='analysis/rolling',**value)
            else:rec.save_objects(artifact_path='analysis',**{name:value})
            if name in ('metrics','paired_comparison','annual_metrics','quarterly_metrics','rolling_summary','uncertainty'):
                if len(value)>500:raise ValueError('slim report unexpectedly large')
                value.to_csv(card/(name+'.csv'),index=False)
        info.to_csv(card/'information_metrics.csv',index=False);pd.DataFrame(training).to_csv(card/'training_summary.csv',index=False)
        economic=all(v for k,v in results['gates'].items() if k not in ('qualified_independent_forward','fresh_fullminute_risk_qualified'))
        summary={'experiment':NAME,'plan_sha256':digest,'decision':'historical_economic_pass_needs_risk_and_independent_forward' if economic else 'not_promoted_fixed_preopen_TOM',
            'objective_achieved':False,'promotion_approved':False,'gates':results['gates'],'coarse_economic_gates_pass':economic,
            'new_fits':6,'new_native_accounts':6,'reused_native_accounts':15,'total_comparison_accounts':21,
            'required_paired_intervals':15,'model_sha256':state['model_sha256'],'sessions_each':944,'native_points_each':3778,
            'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),
            'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(provider),'as_of':'2026-10-09',
            'historical_PIT_qualified':False,'independent_forward_qualified':False,'new_source_HTTP':0,'production_changed':False,
            'history_status':'reused_history_not_independent_validation','artifact_policy':plan['artifact_policy'],'independent_evaluation':'pending'}
        rec.save_objects(artifact_path='analysis',summary=summary);rec.save_objects(artifact_path='execution',receipt=state)
        rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
        rec.end_run(status=Recorder.STATUS_FI)
        summary=_seal(rec,root,card,summary,plan['preflight_sources']['training_only']['card'])
        _conclusion(card,summary,results['metrics']);state.update(completed=True,phase='completed');atomic_json(state,state_path)
        return summary
    except BaseException as error:
        state.update(failed=True,failed_phase=state['phase'],exception_type=type(error).__name__,exception=str(error));atomic_json(state,state_path)
        rec.save_objects(artifact_path='execution',failure=state);rec.end_run(status=Recorder.STATUS_FA)
        raise


def _conclusion(card,summary,metrics):
    lines=['# 固定09:29条件NYSE日历代理：配对实验','',
      '完成B/C各3seed固定300轮拟合，全部6模型保存固定SHA后才加载评价标签/收益。使用原train-only3014/20741样本与39/16输入，4新增allocation槽中B/C共享3known masks，仅C读取已知centered±0.5，unknown在canonical保留NaN。原shared encoder/aux/loss/optimizer和Stage1政策不改。',
      '', '6个新ensemble费用账户、15封存参考、21组比较及15CI均保留；不按分类分数/seed选择。来源为09:29有条件计划，不是当前日已开市确认、机构资金或PIT证据。',
      '', '|方案|bp|累计净收益%|收盘DD%|CVaR5%|','|---|---:|---:|---:|---:|']
    for row in metrics.itertuples():lines.append(f'|{row.family}|{row.slippage_bps}|{row.return_pct:.3f}|{row.close_max_drawdown_pct:.3f}|{row.cvar5_pct:.3f}|')
    lines+=['','决定：'+summary['decision']+'，执行者不自签晋升。失败或未取得的门：'+', '.join(k for k,v in summary['gates'].items() if not v)+'。',
      '', '全部年度/季度和126日滚动保留，2024/2025完整、2023/2026partial分列。B的masks本身含部分日历信息，C/B仅检验其外数值增量；还必须对A、旧B、SOXX和强参考满足全门。重复历史和3seed不构成独立市场证据。',
      '', '日历vintage/临时闭市/修订缺口与条件前缀假设保留；费用会改变自身状态，不能把路径差异解释成单控制器或简单费用因果。生产未改，源数据请求0。',
      '', '下一步为独立结果评审；固定失败版本不得调整venue、窗口、mask、seed、编码或控制器救援。历史通过也仅是待风险/新前向资格的候选，整体稳定SOXX超额目标未完成。',
      '', 'Recorder：`'+summary['recorder_id']+'`；完整证据见summary/verification_manifest/nas_binding。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')

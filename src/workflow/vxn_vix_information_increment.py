"""Native matched-arm input contracts for one initial-release information feature."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.convex_allocation import FEATURE_COLUMNS,LABEL_COLUMNS,METADATA_COLUMNS
from src.data.generic_horizon import predictor_clock,attach_horizon_scope
from src.data.integrated_allocation import build_native_dataset,frame_sha256
from src.data.provider_content_hash import provider_content_hash


def _require(condition,message):
    if not condition:
        raise ValueError('blocked: '+message)


def _sha_list(index):
    return hashlib.sha256('\n'.join(str(x) for x in index).encode()).hexdigest()


def _flags(values):
    a=np.asarray(values)
    _require(np.isin(a,[0,1]).all(),'source known flag must be explicit0/1 without missing')
    return a.astype(bool)


def _clock(values,name):
    values=list(values)
    _require(all(pd.isna(v) or pd.Timestamp(v).tzinfo is not None for v in values),
             name+' requires explicit timezone')
    return pd.DatetimeIndex(pd.to_datetime(values,utc=True))


def combine_feature_availability(price_available_at,source_available_at,source_known):
    index=price_available_at.index if isinstance(price_available_at,pd.Series) else pd.RangeIndex(len(price_available_at))
    price=pd.Series(_clock(price_available_at,'price clock'),index=index)
    source=pd.Series(_clock(source_available_at,'source clock'),index=index)
    known=_flags(source_known)
    _require(len(price)==len(source)==len(known),'availability lengths differ')
    _require(price[known].notna().all() and source[known].notna().all(),'known feature has missing clock')
    combined=pd.Series(pd.NaT,index=index,dtype='datetime64[ns, UTC]')
    combined.loc[known]=pd.concat([price,source],axis=1).max(axis=1).loc[known]
    return combined


def source_eligibility_model(*,signal_dates,source_known,label_available_at,cutoff,feature_available_at,prediction_at):
    signal=pd.DatetimeIndex(signal_dates)
    _require(not signal.hasnans and signal.is_unique and signal.is_monotonic_increasing,'signal dates must be unique ordered')
    known=_flags(source_known)
    label=_clock(label_available_at,'label clock')
    feature=_clock(feature_available_at,'feature clock')
    prediction=_clock(prediction_at,'prediction clock')
    _require(len(signal)==len(known)==len(label)==len(feature)==len(prediction),'eligibility lengths differ')
    _require(prediction.equals(predictor_clock(signal)),'prediction clock differs from signal16:15NY')
    cut=pd.Timestamp(cutoff)
    _require(cut.tzinfo is not None,'cutoff requires explicit timezone')
    mature=label.notna() & (label<cut)
    ready=feature.notna() & (feature<=prediction) & (feature<cut)
    eligible=known & mature & ready
    audit=dict(rows_total=len(signal),rows_eligible=int(eligible.sum()),rows_excluded=int((~eligible).sum()),
        exclusion_reasons=dict(source_unknown=int((~known).sum()),label_unmatured_or_nat=int((known&~mature).sum()),
                              feature_clock_missing_late_or_after_cutoff=int((known&mature&~ready).sum())),
        eligible_index_sha256=_sha_list(signal[eligible]),excluded_index_sha256=_sha_list(signal[~eligible]),
        cutoff=cut.isoformat(),no_row_deleted=True)
    return dict(eligible_mask=eligible,audit=audit)


def _load_source_frame(store,signal_dates):
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(Path(store).resolve()),region='us',kernels=1)
    calendar=pd.DatetimeIndex(D.calendar(freq='day'))
    raw=D.features(['TECH_IV_REL'],['$known','$log_vxn_vix','$availability_calendar_index'],freq='day')
    _require(set(raw.index.get_level_values('instrument'))=={'TECH_IV_REL'},'source instrument identity differs')
    raw=raw.droplevel('instrument')
    _require(raw.index.equals(calendar) and calendar.equals(pd.DatetimeIndex(signal_dates)),'source and price signal calendars differ')
    known=_flags(raw['$known'])
    pos=raw.loc[known,'$availability_calendar_index'].to_numpy(float)
    _require(np.isfinite(pos).all() and np.array_equal(pos,pos.astype(int)) and (pos>=0).all() and (pos<len(calendar)).all(),'invalid source calendar positions')
    times=pd.Series(pd.NaT,index=calendar,dtype='datetime64[ns, UTC]')
    times.loc[known]=(calendar[pos.astype(int)].tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=31)).tz_convert('UTC')
    _require(raw.loc[~known,'$log_vxn_vix'].isna().all(),'unknown ratio was filled')
    result=pd.DataFrame(dict(source_known=known.astype(int),source_ratio=raw['$log_vxn_vix'],source_available_at=times),index=calendar)
    return result,calendar


def build_matched_arm_datasets(*,root,plan,as_of='2026-09-25'):
    """Keep full calendars and unknowns; no training or portfolio replay."""
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root);store=root/plan['source_binding']['provider_uri']
    expected=plan['source_binding']['provider_content_sha256']
    _require(provider_content_hash(store)[0]==expected,'qualified information store identity differs')
    from src.data.oos_horizon_inputs import prepare_horizon_inputs
    parent=json.loads((root/'research/purged_ridge_regularization_20261009/plan.json').read_text())
    bundle=prepare_horizon_inputs(root,parent)
    frame=bundle['frame'].copy();calendar=pd.DatetimeIndex(bundle['calendar'])
    source,source_calendar=_load_source_frame(store,frame.index)
    frame['source_known']=source.source_known
    frame['source_available_at']=source.source_available_at
    frame['source_ratio']=source.source_ratio
    frame['base_feature_available_at']=frame.feature_available_at
    frame['feature_available_at']=combine_feature_availability(frame.feature_available_at,frame.source_available_at,frame.source_known)
    known=_flags(frame.source_known)
    _require(np.isfinite(frame.loc[known,'source_ratio']).all() and frame.loc[~known,'source_ratio'].isna().all(),'known/unknown ratio contract differs')
    _require((frame.loc[known,'feature_available_at']<=pd.to_datetime(frame.loc[known,'decision_at'],utc=True)).all(),'source feature is not available at prediction')
    extra=['source_known','source_available_at','base_feature_available_at']
    meta=tuple(METADATA_COLUMNS)+tuple(extra)
    candidate_features=tuple(FEATURE_COLUMNS)+('source_ratio',)
    output=dict(frame=frame,calendar=calendar,source_calendar=source_calendar,covariance=bundle['covariance'],
                control_features=list(FEATURE_COLUMNS),candidate_features=list(candidate_features),
                rows=len(frame),known_rows=int(known.sum()),unknown_rows=int((~known).sum()),
                source_ratio_nan_unknown=int(frame.loc[~known,'source_ratio'].isna().sum()),fit_performed=False)
    for arm,features in [('control',FEATURE_COLUMNS),('candidate',candidate_features)]:
        dataset=build_native_dataset(frame,feature_columns=features,label_columns=LABEL_COLUMNS,metadata_columns=meta,as_of=as_of)
        attach_horizon_scope(dataset,5);dataset.calendar=calendar
        prepared=dataset.prepare('all',col_set=['feature','label','metadata'])
        _require(pd.DatetimeIndex(prepared.index.get_level_values('datetime')).equals(frame.index),'DatasetH deleted or reordered source dates')
        output[arm]=dataset;output[arm+'_prepared']=prepared
    output['rowset_equal']=output['control_prepared'].index.equals(output['candidate_prepared'].index)
    output['rowset_sha256']=_sha_list(frame.index)
    output['eligible_rowset_sha256']=_sha_list(frame.index[known])
    pd.testing.assert_frame_equal(output['control_prepared']['metadata'],output['candidate_prepared']['metadata'],check_exact=True)
    _require(provider_content_hash(store)[0]==expected,'source store changed during readback')
    return output


def fit_budget(plan):
    return 2*len(plan['schedule']['outer_years'])*5


def restore_frozen_arm_datasets(bundle,as_of='2026-09-25'):
    """DatasetH omits cached private data by default; restore from verified Recorder rows."""
    result=dict(bundle)
    meta=tuple(METADATA_COLUMNS)+('source_known','source_available_at','base_feature_available_at')
    for arm in ('control','candidate'):
        dataset=build_native_dataset(bundle['frame'],feature_columns=bundle[arm+'_features'],
            label_columns=LABEL_COLUMNS,metadata_columns=meta,as_of=as_of)
        attach_horizon_scope(dataset,5);dataset.calendar=bundle['calendar']
        prepared=dataset.prepare('all',col_set=['feature','label','metadata'])
        pd.testing.assert_frame_equal(prepared,bundle[arm+'_prepared'],check_exact=True)
        result[arm]=dataset
    return result


def qualify_matched_annual_inputs(bundle, plan):
    """Both arms use identical dates/labels; this qualification performs zero fits."""
    from src.model.generic_annual_ridge_selection import qualify_annual_horizon_inputs
    rows=[]
    for year in plan['schedule']['outer_years']:
        audits={}
        for arm in ('control','candidate'):
            result=qualify_annual_horizon_inputs(bundle[arm],bundle['frame'],outer_year=year,
                feature_columns=bundle[arm+'_features'],label_columns=LABEL_COLUMNS,
                availability_column='h5_available_at',eligibility_column='source_known')
            audits[arm]=result['audit']
        for key in ('inner_identity','outer_identity','validation_identity'):
            _require(audits['control'][key]==audits['candidate'][key],f'{year} matched {key} differs')
        rows.append(dict(matched_identity=True,**audits['candidate']))
    # The registered fallback must actually be below the unchanged inner minimum.
    fallback={}
    for year in plan['schedule']['fallback_outer_years']:
        frame=bundle['frame'];cut=pd.Timestamp(f'{year-1}-01-01',tz='America/New_York')
        result=source_eligibility_model(signal_dates=frame.index,source_known=frame.source_known,
            label_available_at=frame.h5_available_at,cutoff=cut,
            feature_available_at=frame.feature_available_at,prediction_at=frame.decision_at)
        _require(result['audit']['rows_eligible']<252,'fallback year has sufficient training rows')
        fallback[year]=result['audit']
    return dict(annual_audits=rows,fallback_audits=fallback,fit_performed=False,
                all_arm_identities_equal=True)


def verify_source_binding(root,plan):
    """Verify immutable source bytes, parent identities and real source Recorder."""
    from src.core.recorder_sources import load_verified_recorder_objects
    root=Path(root);binding=plan['source_binding'];digests={}
    paths=dict(binding['raw_zip_sha256'],**binding['source_code_identity'])
    paths[binding['provider_uri']+'/qlib_store_manifest.json']=binding['store_manifest_sha256']
    for name,identity in binding['reference_parents'].items():
        paths[f'research/{name}/plan.json']=identity['plan_sha256']
        paths[f'research/{name}/summary.json']=identity['summary_sha256']
    for name,expected in paths.items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        _require(actual==expected,'source binding differs: '+name);digests[name]=actual
    for name,identity in binding['price_execution_providers'].items():
        store=root/identity['provider_uri']
        actual=provider_content_hash(store)[0]
        _require(actual==identity['content_sha256'],'price provider differs: '+name)
        _require(hashlib.sha256((store/'qlib_store_manifest.json').read_bytes()).hexdigest()==identity['manifest_sha256'],
                 'price provider manifest differs: '+name)
        digests[name]=actual
    _require(provider_content_hash(root/binding['provider_uri'])[0]==binding['provider_content_sha256'],'information provider differs')
    qualified=binding['source_qualification_latest_recorder']
    load_verified_recorder_objects(root/qualified['card'],expected_recorder_id=qualified['recorder_id'],
        artifacts=list(qualified['object_sha256']),expected_artifact_sha256=qualified['object_sha256'])
    return digests


def _manifest(card,summary):
    local=Path(summary['recorder_local_dir'])/'artifacts'
    entries=[dict(path=str(p.relative_to(card)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
             for p in sorted(local.rglob('*')) if p.is_file() and not p.name.startswith('code_')]
    (card/'verification_manifest.json').write_text(json.dumps(dict(recorder_id=summary['recorder_id'],entries=entries),indent=2)+'\n')
    (card/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prepare_information_increment_card(*,root,card):
    """Executable notebook stage: source and matched DatasetH qualification, zero fits."""
    import qlib,yaml
    from qlib.workflow import R
    root=Path(root).resolve();card=Path(card).resolve()
    plan=json.loads((card/'plan.json').read_text())
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    qlib.init(provider_uri=str(root/plan['source_binding']['price_execution_providers']['published_futu_affine']['provider_uri']),region='us')
    freeze=verify_source_binding(root,plan)
    bundle=build_matched_arm_datasets(root=root,plan=plan)
    audit=qualify_matched_annual_inputs(bundle,plan)
    uri=str(card/'qlib_recorder')
    with R.start(experiment_name=plan['experiment'],recorder_name='native_prefit',uri=uri):
        rec=R.get_recorder()
        rec.log_params(stage='native_prefit',fit_calls=0,native_cases=0,
            plan_sha256=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest(),
            source_provider=plan['source_binding']['provider_uri'])
        rec.save_objects(artifact_path='prefit',**{'inputs.pkl':bundle,'qualification.pkl':audit,'source_binding.pkl':freeze,'plan.pkl':plan})
        rid=rec.id;local=rec.get_local_dir()
    _require(rec.status=='FINISHED','prefit Recorder did not finish')
    policy=dict(policy='slim_card_recorder_first',workflow_config='workflow.yaml',
                recorder_only_artifacts=['prefit/inputs.pkl','prefit/qualification.pkl','prefit/source_binding.pkl','prefit/plan.pkl'])
    summary=dict(experiment=plan['experiment'],status='prefit_qualified_not_promoted',promotion_decision='not_promoted',
        promotion_eligible=False,fit_calls=0,native_cases=0,rows=bundle['rows'],known_rows=bundle['known_rows'],unknown_rows=bundle['unknown_rows'],
        all_arm_identities_equal=True,recorder_id=rid,recorder_uri=uri,recorder_local_dir=local,recorder_status='FINISHED',
        source=dict(provider_uri=plan['source_binding']['provider_uri'],region='us',as_of='2026-09-25'),
        artifact_policy=policy,plan_sha256=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest())
    _manifest(card,summary)
    flat=[]
    for item in audit['annual_audits']:
        row={k:v for k,v in item.items() if not isinstance(v,dict)}
        for k,v in item.items():
            if isinstance(v,dict): row.update({k+'_'+a:b for a,b in v.items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(card/'sample_eligibility.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump(dict(experiment=plan['experiment'],source=summary['source'],
        recorder_uri=uri,recorder_id=rid,artifact_policy=policy,method='fixed H5 matched25/26 source-eligible DatasetH'),sort_keys=False))
    (card/'selected_model.json').write_text(json.dumps(dict(model=None,status='no_fit',fixed_future_fit_budget=100,promotion_decision='not_promoted'),indent=2)+'\n')
    text='\n'.join(['# VXN/VIX 信息增量：拟合前资格','',
        '已通过原生 DatasetH 和 Recorder 读回：3956 个交易日，3126 日已知、830 日未知；两组年度训练、验证日期和标签身份相同。',
        '只增加 log(VXN/VIX) 一项信息，来源为 ALFRED Initial Release Only。观察日不等于发布日；2018-10-08 异常观察在 2018-10-09 信号中保留为未知。',
        '2016 年内层严格成熟样本不足，两组均按原 H5 调仓节奏退回 SOXX，不拟合、不伪造预测。',
        '本阶段 0 次拟合、0 次回放，尚无收益、风险或增量结论；不晋升。后续只能执行冻结的100次拟合、12个原生场景，失败即停止此路线。',
        '来源只有日期级初次发布资格，没有验证日内发布秒数。历史仍是复用诊断，不能替代新的独立前向证据。',f'Recorder：{rid}；FINISHED。'])+'\n'
    (card/'conclusion.md').write_text(text)
    return summary


def _archive_run(summary,objects):
    from qlib.workflow import R
    with R.start(experiment_name=summary['experiment'],recorder_id=summary['recorder_id'],
                 uri=summary['recorder_uri'],resume=True):
        rec=R.get_recorder()
        _require(rec.id==summary['recorder_id'],'resumed Recorder identity differs')
        rec.save_objects(artifact_path='aggregate',**objects)
    _require(rec.status=='FINISHED','experiment Recorder did not finish')


def verify_execution_plan(root,card,execution):
    root=Path(root);card=Path(card)
    _require(execution['authorization']['historical_fit_authorized'] is True and
             execution['authorization']['replay_authorized'] is True,'one-shot research execution not authorized')
    _require(execution['fixed_fit_budget']==100 and execution['fixed_native_cases']==12,'fixed budgets differ')
    _require(execution['horizon_sessions']==5 and execution['costs_bps']==[10.,25.] and execution['robust_radius']==1.,
             'fixed method differs')
    _require(hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()==execution['preregister_sha256'],
             'preregistration differs')
    for name,digest in execution['source_sha256'].items():
        _require(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'execution source differs: '+name)


def run_information_increment(*,root,card,execution_plan_path):
    """One fixed 100-fit/12-native-case run; preserve all failures and forbid repeats."""
    import qlib
    from qlib.workflow import R
    from src.core.recorder_sources import load_verified_recorder_objects
    from src.model.generic_annual_ridge_selection import select_annual_horizon
    from src.backtest.generic_horizon_window import replay_horizon_window,native_verify_window
    from src.workflow.benchmark_evidence import evaluate_registered_paths
    from src.analysis.excess_resolution import apply_excess_resolution
    from src.analysis.forecast_information_increment import paired_forecast_information
    root=Path(root).resolve();card=Path(card).resolve()
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    execution=json.loads(Path(execution_plan_path).read_text());plan=json.loads((card/'plan.json').read_text())
    verify_execution_plan(root,card,execution)
    attempt=card/'execution_attempt.json'
    _require(not attempt.exists(),'one-shot execution already attempted; do not refit')
    price=plan['source_binding']['price_execution_providers']
    qlib.init(provider_uri=str(root/price['published_futu_affine']['provider_uri']),region='us',kernels=1)
    freeze=verify_source_binding(root,plan)
    prefit=execution['prefit']
    objects,_=load_verified_recorder_objects(card,expected_recorder_id=prefit['recorder_id'],
        artifacts=['prefit/inputs.pkl','prefit/qualification.pkl','prefit/plan.pkl'],
        expected_artifact_sha256=prefit['object_sha256'])
    bundle=restore_frozen_arm_datasets(objects['prefit/inputs.pkl'])
    _require(objects['prefit/plan.pkl']==plan,'frozen prefit plan differs')
    _require(qualify_matched_annual_inputs(bundle,plan)==objects['prefit/qualification.pkl'],'prefit annual masks differ')
    frame=bundle['frame'];calendar=bundle['calendar'];sigma=bundle['covariance'];uri=str(card/'execution_recorder')
    with attempt.open('x') as f: json.dump(dict(status='starting',fit_calls=0,native_cases=0),f)
    with R.start(experiment_name=plan['experiment'],recorder_name='fixed_matched_increment',uri=uri):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',**{'inputs.pkl':bundle,'execution_plan.pkl':execution})
        rec.log_params(horizon_sessions=5,fixed_fit_budget=100,fixed_native_cases=12,source='ALFRED InitialReleaseOnly')
        summary=dict(experiment=plan['experiment'],recorder_id=rec.id,recorder_uri=uri,recorder_local_dir=rec.get_local_dir())
    selections={arm:{} for arm in ('control','candidate')};cases={};fit_calls=0;stage='fit';fitted_objects={}
    def checkpoint():
        attempt.write_text(json.dumps(dict(status='running',stage=stage,fit_calls=fit_calls,native_cases=len(cases),**summary),indent=2)+'\n')
        _archive_run(summary,{'selections.pkl':selections,'case_results.pkl':cases,'completed_fits.pkl':fitted_objects})
    try:
        for year in plan['schedule']['outer_years']:
            for arm in ('control','candidate'):
                stage=f'fit_{year}_{arm}'
                def fitted(kind,name,model):
                    nonlocal fit_calls
                    fitted_objects[(arm,year,kind,name)]=model
                    fit_calls+=1
                result=select_annual_horizon(bundle[arm],frame,outer_year=year,
                    feature_columns=bundle[arm+'_features'],label_columns=LABEL_COLUMNS,
                    availability_column='h5_available_at',eligibility_column='source_known',fit_callback=fitted)
                selections[arm][year]=result
                print(f'{arm} selected {year}: {result["selected_name"]}; fits={fit_calls}',flush=True)
            for key in ('inner_identity','outer_identity','validation_identity'):
                _require(selections['control'][year]['audit'][key]==selections['candidate'][year]['audit'][key],
                         'fitted arms differ: '+key)
            checkpoint()
        _require(fit_calls==100,'completed fit budget differs')
        exits=pd.to_datetime(frame.exit_date);unsettled=exits.isna().to_numpy()
        _require(int(unsettled.sum())==2 and unsettled[-2:].all(),'unsettled rows must be terminal2')
        paths={};forecasts={};windows={}
        for name,years in [('development',plan['schedule']['development_years']),('reused_diagnostic',plan['schedule']['reused_diagnostic_years'])]:
            mask=frame.index.year.isin(years)&~unsettled
            windows[name]=dict(frame=frame.loc[mask].reset_index(drop=True),covariance=sigma[mask])
        for name,parts in windows.items():
            for cost in execution['costs_bps']:
                for arm in ('control','candidate','soxx'):
                    key=f'{name}::{arm}::{int(cost)}';stage=key
                    result=replay_horizon_window(parts['frame'],{} if arm=='soxx' else selections[arm],parts['covariance'],
                        calendar=calendar,cost_bps=cost,horizon_sessions=5,robust_radius=1.,benchmark_only=arm=='soxx',
                        feature_columns=bundle.get(arm+'_features',()),availability_column='h5_available_at',
                        eligibility_column='source_known',fallback_years=(2016,))
                    cases[key]=result
                    result=native_verify_window(parts['frame'],result,provider_uri=root/price['legacy_soxl_execution']['provider_uri'],
                        cost_bps=cost,initial_cash=100000.)
                    _require(result['native']['actual_position_audit']['status']=='passed','native actual accounting failed')
                    cases[key]=result;paths[(name,arm,cost)]=result['path']
                    if arm!='soxx':
                        if (name,arm) in forecasts: pd.testing.assert_frame_equal(forecasts[(name,arm)],result['forecast'])
                        forecasts[(name,arm)]=result['forecast']
                    print('native passed '+key,flush=True);checkpoint()
        _require(len(cases)==12,'native case inventory incomplete')
        stage='statistics'
        combined={(arm,cost):pd.concat([paths[(w,arm,cost)] for w in windows],ignore_index=True)
                  for arm in ('candidate','control','soxx') for cost in execution['costs_bps']}
        registered=dict(development_years=plan['schedule']['development_years'],costs_bps=execution['costs_bps'],
            as_of='2026-09-25',variants=[{'name':'candidate'},{'name':'control'}],candidate_budget=1,
            gates=plan['gates']['copied_from_parent'])
        tables,evidence=evaluate_registered_paths(combined,plan=registered,calendar=calendar,candidate_controls={'candidate':'control'})
        dev={c:paths[('development','candidate',c)] for c in execution['costs_bps']}
        bench={c:paths[('development','soxx',c)] for c in execution['costs_bps']}
        resolved=apply_excess_resolution(evidence['candidate'],dev,bench,log_excess_floor=1e-8)
        tables['raw_gates.csv']=tables['gates.csv'].copy();tables['gates.csv']=resolved['gates']
        tables['excess_resolution.csv']=pd.DataFrame([resolved['excess_resolution']])
        tables['resolved_annual_metrics.csv']=resolved['annual'];tables['bootstrap_analysis.csv']=resolved['bootstrap']
        tables['rolling_summary.csv']=resolved['rolling_summary']
        diagnostic=tables['backtest_benchmark_comparison.csv'].query("strategy=='candidate' and window=='reused_diagnostic'")
        diag_pass=bool((diagnostic.log_excess_sum>1e-8).all() and
            (diagnostic.max_drawdown_pct>=diagnostic.benchmark_max_drawdown_pct-1e-10).all() and
            (diagnostic.cvar5_pct>=diagnostic.benchmark_cvar5_pct-1e-10).all())
        incremental_portfolio=bool((tables['ablation_comparison.csv'].log_excess_sum>1e-8).all())
        c=pd.concat([forecasts[(w,'candidate')] for w in windows]);b=pd.concat([forecasts[(w,'control')] for w in windows])
        info,info_pass=paired_forecast_information(c,b,cutoff=pd.Timestamp('2026-09-25 16:15',tz='America/New_York'),
            windows={w:plan['schedule'][w+'_years'] for w in windows})
        tables['forecast_information.csv']=info
        tables['model_selection.csv']=pd.DataFrame([dict(arm=arm,year=y,selected=r['selected_name'])
            for arm,years in selections.items() for y,r in years.items()])
        tables['native_audit.csv']=pd.DataFrame([dict(case=k,**v['native']['actual_position_audit']) for k,v in cases.items()])
        tables['risk_analysis.csv']=pd.DataFrame([dict(case=k,metric=str(m),value=float(row['risk']))
            for k,v in cases.items() for m,row in v['native']['risk_analysis_next_open'].iterrows()])
        verify_execution_plan(root,card,execution);_require(verify_source_binding(root,plan)==freeze,'source changed during experiment')
        passed=bool(resolved['historical_gate_pass'] and diag_pass and incremental_portfolio and info_pass)
        summary.update(status='complete_not_promoted',promotion_decision='not_promoted',promotion_eligible=False,
            scientific_promotion=False,fit_calls=fit_calls,native_cases=12,historical_gate_pass=bool(resolved['historical_gate_pass']),
            diagnostic_gate_pass=diag_pass,incremental_portfolio_pass=incremental_portfolio,incremental_information_pass=info_pass,
            route_decision='historical_pass_requires_independent_forward' if passed else 'end_vxn_vix_route_no_further_search',
            source=dict(provider_uri=price['published_futu_affine']['provider_uri'],execution_provider_uri=price['legacy_soxl_execution']['provider_uri'],
                        information_provider_uri=plan['source_binding']['provider_uri'],region='us',as_of='2026-09-25'),
            recorder_status='FINISHED',plan_sha256=execution['preregister_sha256'],
            execution_plan_sha256=hashlib.sha256(Path(execution_plan_path).read_bytes()).hexdigest(),
            artifact_policy=dict(policy='slim_card_recorder_first',workflow_config='workflow.yaml',recorder_only_artifacts=[
                'inputs/inputs.pkl','inputs/execution_plan.pkl','aggregate/selections.pkl','aggregate/case_results.pkl',
                'aggregate/tables.pkl','aggregate/raw_evidence.pkl','aggregate/resolved_evidence.pkl','aggregate/final_summary.pkl']))
        _archive_run(summary,{'selections.pkl':selections,'case_results.pkl':cases,'tables.pkl':tables,
            'raw_evidence.pkl':evidence,'resolved_evidence.pkl':resolved,'final_summary.pkl':summary})
        for name,table in tables.items(): table.to_csv(card/name,index=False)
        _manifest(card,summary)
        _write_result_conclusion(card,summary,tables)
        attempt.write_text(json.dumps(dict(status='completed',fit_calls=fit_calls,native_cases=12,**{k:summary[k] for k in ('recorder_id','recorder_uri')}),indent=2)+'\n')
        return summary
    except Exception as error:
        failure={**summary,'status':'blocked','stage':stage,'error':repr(error),'fit_calls':fit_calls,'native_cases':len(cases)}
        _archive_run(summary,{'selections.pkl':selections,'case_results.pkl':cases,'completed_fits.pkl':fitted_objects,'failure.pkl':failure})
        attempt.write_text(json.dumps(failure,indent=2)+'\n')
        raise


def _write_result_conclusion(card,summary,tables):
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump(dict(experiment=summary['experiment'],source=summary['source'],
        recorder_uri=summary['recorder_uri'],recorder_id=summary['recorder_id'],artifact_policy=summary['artifact_policy']),sort_keys=False))
    (card/'selected_model.json').write_text(json.dumps(dict(model='native_nested_annual_prior_ridge',horizon_sessions=5,
        selections=tables['model_selection.csv'].to_dict('records'),promotion_decision='not_promoted'),indent=2)+'\n')
    lines=['# VXN/VIX 单一信息增量：结果','',
        '固定100次拟合、12个原生账户场景全部完成；控制组25特征与候选26特征使用相同来源资格、严格成熟样本。',
        '2016两组按H5节奏退回SOXX；未知行不预测、不删日历，非调仓日精确HOLD。','']
    for row in tables['backtest_benchmark_comparison.csv'].query("strategy=='candidate'").itertuples():
        lines.append(f'- {row.window}/{row.cost_bps:g}bp：净收益 {row.return_pct:.2f}%，相对SOXX终值 {row.relative_wealth:.6f}，最大回撤 {row.max_drawdown_pct:.2f}%。')
    lines += ['',f'历史门 {summary["historical_gate_pass"]}；复用诊断门 {summary["diagnostic_gate_pass"]}；相对匹配控制组组合增量 {summary["incremental_portfolio_pass"]}；固定超额响应MSE增量 {summary["incremental_information_pass"]}。',
        f'不晋升；路线决定：{summary["route_decision"]}。',
        '逐窗口SOXL、SOXX及超额响应MSE/IC见 forecast_information.csv；IC仅为描述，未做显著性认证。历史复用、日期级初次发布、固定HAC5均不构成独立前向证据。',
        '失败即停止此来源路线，不改变期限、alpha、阈值、特征符号或风险门。若全部历史门通过，仍需新冻结身份和至少60日独立前向证据，不能据此立即上线。',f'Recorder：{summary["recorder_id"]}；FINISHED。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')

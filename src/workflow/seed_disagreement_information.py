"""Six fixed Ridge calibrations on saved annual OOF raw allocation outputs."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.workflow.minute_action_value import _source
from src.model.nominal_beta_composition import cancel_leverage_cash
from src.analysis.seed_disagreement_information import mature_before,fit_scaler,transform,full_calendar_bootstrap
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def prepare(root,plan):
    from qlib.data import D
    objects={};audits={}
    for key,source in plan['sources'].items():objects[key],audits[key]=_source(root,source,plan['source_artifacts'][key])
    model=objects['later_predictions']['models/2020_71']
    if model.base_features!=plan['market_features']:raise ValueError('original feature order changed')
    predictions={**objects['early_predictions']['analysis/predictions'],**objects['later_predictions']['analysis/predictions']}
    training=objects['early_predictions']['analysis/training']+objects['later_predictions']['analysis/training']
    records={(str(x['year']),x['base_seed']):x for x in training}
    full=objects['frame']['inputs/full_reference_frame'].copy()
    frame=full.loc[full.trade_date.between(plan['input_window']['first_trade'],plan['input_window']['last_trade'])].copy().reset_index(drop=True)
    if len(frame)!=plan['input_window']['expected_rows']:raise ValueError('source input coverage differs')
    calendar=pd.DatetimeIndex(D.calendar(start_time=frame.trade_date.min(),end_time=frame.trade_date.max(),freq='day'))
    pd.testing.assert_index_equal(calendar,pd.DatetimeIndex(frame.trade_date),check_names=False)
    source_available=pd.to_datetime(frame.label_available_at).dt.tz_localize('America/New_York')
    conservative=(frame.exit_date+pd.Timedelta(hours=9,minutes=49)).dt.tz_localize('America/New_York')
    available=source_available.where(source_available.ge(conservative),conservative)
    feature_available=pd.to_datetime(frame.feature_available_at).dt.tz_localize('America/New_York')
    decision=pd.to_datetime(frame.decision_at).dt.tz_localize('America/New_York')
    if not feature_available.le(decision).all():raise ValueError('market features unavailable at decision')
    raw=[];source_checks=[]
    for year in [2019,2020,2021,2022]:
        required=pd.DatetimeIndex(frame.loc[frame.trade_date.dt.year.eq(year),'signal_date'])
        group=[]
        for seed in plan['base_seeds']:
            values=predictions[(str(year),seed)];audit=records[(str(year),seed)]
            pd.testing.assert_index_equal(pd.DatetimeIndex(values.index),required,check_names=False)
            if not np.isfinite(values.to_numpy(float)).all():raise ValueError('missing raw prediction')
            cutoff=pd.Timestamp(audit['cutoff']);maturity=pd.Timestamp(audit['last_mature_label_at'])
            if cutoff!=pd.Timestamp(f'{year}-01-01',tz='America/New_York') or not maturity<cutoff:raise ValueError('OOF model timing invalid')
            if not cutoff<decision.loc[frame.trade_date.dt.year.eq(year)].min():raise ValueError('OOF model unavailable at first trade')
            source_checks.append({'year':year,'base_seed':seed,'actual_seed':audit['actual_seed'],'rows':len(values),'model_cutoff':str(cutoff),'last_training_label_available_at':str(maturity)})
            group.append(values)
        means=sum(group)/len(group);betas=np.stack([3*x.soxl_weight.to_numpy()+x.soxx_weight.to_numpy() for x in group],axis=1)
        means['mean_raw_beta']=betas.mean(axis=1);means['mean_raw_cash']=means.cash_weight
        means['mean_alpha_probability']=means.alpha_probability;means['mean_tail_probability']=means.tail_probability
        means['raw_beta_seed_std_ddof0']=betas.std(axis=1,ddof=0);raw.append(means)
    mean=pd.concat(raw).reindex(pd.DatetimeIndex(frame.signal_date));features=frame[plan['market_features']].copy()
    for column in [*plan['baseline_extra_features'],plan['candidate_extra_feature']]:features[column]=mean[column].to_numpy()
    mapped=np.array([cancel_leverage_cash(w) for w in mean[['soxl_weight','soxx_weight','cash_weight']].to_numpy()])
    mature=mature_before(available,plan['score_as_of']);label=np.full(len(frame),np.nan)
    # Do not read the final2023 exit-price outcome into the2022 information score.
    asset_returns=frame.loc[mature,['soxl_exit_open','soxx_exit_open']].to_numpy()/frame.loc[mature,['soxl_open','soxx_open']].to_numpy()-1
    label[mature]=(mapped[mature.to_numpy(),:2]*asset_returns).sum(axis=1)-asset_returns[:,1]
    if not np.isfinite(label[mature]).all():raise ValueError('invalid mature label')
    metadata=frame[['signal_date','trade_date','exit_date']].copy();metadata['label_available_at']=available;metadata['score_eligible']=mature
    return features,pd.Series(label,name='active_score'),metadata,audits,pd.DataFrame(source_checks)


def run(*,root,card,plan):
    from qlib.workflow import R
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed information plan changed')
        return old
    features,label,meta,source_audits,source_checks=prepare(root,plan)
    baseline=[*plan['market_features'],*plan['baseline_extra_features']];candidate=[*baseline,plan['candidate_extra_feature']]
    data=pd.concat({'feature':features,'label':label.to_frame(),'meta':meta},axis=1)
    data.index=pd.MultiIndex.from_arrays([meta.trade_date,['SOXL_SOXX_CASH']*len(meta)],names=['datetime','instrument'])
    progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':digest,'attempts':[],'fitted':[]}
    if progress['plan_sha256']!=digest:raise ValueError('fit progress identity changed')
    tracking=card/'qlib_recorder';kw={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kw.update(recorder_id=progress['recorder_id'],resume=True)
    scored=[];folds=[];coefficients=[];preprocess=[]
    with R.start(**kw):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        if not progress.get('source_saved'):
            rec.save_objects(artifact_path='inputs',plan=plan,data=data,source_audits=source_audits,source_model_timing=source_checks)
            rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),analysis=(root/'src/analysis/seed_disagreement_information.py').read_text())
            progress['source_saved']=True;atomic_json(progress,progress_path)
        for year in [2020,2021,2022]:
            dataset=DatasetH(handler=DataHandlerLP.from_df(data),segments={'train':('2019-01-01',f'{year-1}-12-31'),'test':(f'{year}-01-01',f'{year}-12-31')})
            train=dataset.prepare('train',col_set=['feature','label','meta']);test=dataset.prepare('test',col_set=['feature','label','meta'])
            train=train.loc[mature_before(train[('meta','label_available_at')],plan['forecast_cutoffs'][str(year)])]
            if len(train)!=plan['expected_training_rows'][str(year)] or len(test)!=plan['expected_evaluation_rows'][str(year)]:raise ValueError('derived fold counts differ; never trim rows to force counts')
            if train['label'].isna().any().any():raise ValueError('eligible training label missing')
            means,scales=fit_scaler(train['feature'][candidate]);prediction={}
            for arm,columns in [('baseline',baseline),('plus_disagreement',candidate)]:
                count=len(columns);key=f'{year}_{arm}';x=transform(train['feature'][columns],means[:count],scales[:count]);xt=transform(test['feature'][columns],means[:count],scales[:count])
                if key in progress['fitted']:model=rec.load_object('models/'+key)
                else:
                    if key in progress['attempts'] or len(progress['attempts'])>=plan['maximum_statistical_fits']:raise ValueError('statistical fit attempt budget exhausted')
                    progress['attempts'].append(key);atomic_json(progress,progress_path)
                    model=Ridge(alpha=plan['estimator']['alpha'],fit_intercept=plan['estimator']['fit_intercept'],solver=plan['estimator']['solver'])
                    model.fit(x,train[('label','active_score')].to_numpy(float));rec.save_objects(artifact_path='models',**{key:model});progress['fitted'].append(key);atomic_json(progress,progress_path)
                prediction[arm]=model.predict(xt)
                coefficients.extend({'year':year,'arm':arm,'feature':name,'standardized_coefficient':float(coef),'training_mean':float(means[i]),'training_std':float(scales[i])} for i,(name,coef) in enumerate(zip(columns,model.coef_)))
                rec.log_metrics(completed_statistical_fits=len(progress['fitted']))
            part=test['meta'].reset_index(drop=True).copy();part['target']=test[('label','active_score')].to_numpy(float)
            for arm,values in prediction.items():part[arm]=values
            part['paired_loss_improvement']=(part.target-part.baseline)**2-(part.target-part.plus_disagreement)**2
            scored.append(part);valid=part.target.notna()
            folds.append({'year':year,'training_rows':len(train),'evaluation_calendar_rows':len(test),'scored_rows':int(valid.sum()),'unknown_rows':int((~valid).sum()),
                'baseline_mse':float(np.mean((part.loc[valid,'target']-part.loc[valid,'baseline'])**2)),
                'candidate_mse':float(np.mean((part.loc[valid,'target']-part.loc[valid,'plus_disagreement'])**2)),
                'paired_mse_improvement':float(part.paired_loss_improvement.mean()),'last_training_label_available_at':str(train[('meta','label_available_at')].max()),'cutoff':plan['forecast_cutoffs'][str(year)]})
            preprocess.append({'year':year,'features':candidate,'means':means,'scales':scales,'training_trade_dates':train.index.get_level_values('datetime')})
        daily=pd.concat(scored,ignore_index=True)
        if len(daily)!=sum(plan['expected_evaluation_rows'].values()) or daily.target.notna().sum()!=plan['expected_mature_scored_rows']:raise ValueError('full evaluation calendar/maturity changed')
        b=plan['bootstrap'];bootstrap,indices,replicate=full_calendar_bootstrap(daily.paired_loss_improvement.to_numpy(),**{k:b[k] for k in ['mean_block','resamples','seed','alpha']})
        fold_table=pd.DataFrame(folds);coefficient_table=pd.DataFrame(coefficients)
        information_pass=bool(fold_table.paired_mse_improvement.gt(0).all() and bootstrap['lower_bound']>0)
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'conditional_information_pass_requires_separate_policy_test' if information_pass else 'not_promoted_seed_disagreement_information_failed','information_gate_pass':information_pass,'objective_achieved':False,'new_statistical_fits':len(progress['fitted']),'new_allocator_fits':0,'new_native_cases':0,'evaluation_as_of':plan['evaluation_as_of'],'bootstrap':bootstrap,'provider_uri':str(root/plan['provider_uri']),'region':'us','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'interpretation':plan['interpretation'],'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','models/*','analysis/daily','analysis/bootstrap_indices','analysis/bootstrap_replicates','analysis/preprocessing']}}
        rec.save_objects(artifact_path='analysis',daily=daily,bootstrap_indices=indices,bootstrap_replicates=replicate,preprocessing=preprocess,fold_metrics=fold_table,coefficients=coefficient_table,summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')))
    for name,value in [('fold_metrics',fold_table),('coefficients',coefficient_table),('source_model_timing',source_checks),('bootstrap_summary',pd.DataFrame([bootstrap]))]:value.to_csv(card/(name+'.csv'),index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['sources']['frame']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

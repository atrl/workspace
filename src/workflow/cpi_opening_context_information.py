"""Official48-header gate before any fixed CPI-context statistical fitting."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.data.cpi_release_calendar import intake
from src.workflow.minute_action_value import _source
from src.workflow.seed_disagreement_information import prepare as prepare_saved_outputs
from src.analysis.seed_disagreement_information import fit_scaler,transform,mature_before,full_calendar_bootstrap
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def fit_information(root,card,plan,rec,events,progress,save_progress):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    parent={**plan,'baseline_extra_features':plan['baseline_extra_features'][:4],'candidate_extra_feature':'raw_beta_seed_std_ddof0'}
    features,_,meta,audits,source_checks=prepare_saved_outputs(root,parent)
    opening,oa=_source(root,plan['sources']['opening'],plan['source_artifacts']['opening'])
    prices,pa=_source(root,plan['sources']['frame'],['inputs/full_reference_frame'])
    index=pd.DatetimeIndex(meta.signal_date);opened=opening['inputs/opening_features'].reindex(index);timing=opening['inputs/opening_timing'].reindex(index)
    if not timing.feature_available_at.le(timing.decision_at).all() or not timing.latest_bar_end.eq(timing.decision_at-pd.Timedelta(minutes=1)).all():raise ValueError('opening timing mismatch')
    if not pd.DatetimeIndex(timing.decision_at.dt.tz_localize(None).dt.normalize()).equals(pd.DatetimeIndex(meta.trade_date)):raise ValueError('opening trade-date mismatch')
    release={pd.Timestamp(x.release_date):pd.Timestamp(x.release_at) for x in events.itertuples()}
    flags=np.array([day in release for day in meta.trade_date],float)
    features['cpi_released_before_decision']=flags;features['opening_soxx_return15']=opened.opening_soxx_return15.to_numpy();features['opening_state_ready']=opened.opening_state_ready.to_numpy()
    features[plan['candidate_extra_feature']]=np.where(flags==0,0.,features.opening_soxx_return15.where(features.opening_state_ready.eq(1)))
    frame=prices['inputs/full_reference_frame'].set_index('trade_date').reindex(pd.DatetimeIndex(meta.trade_date));label=pd.Series(np.nan,index=features.index,name='target')
    eligible=meta.score_eligible.to_numpy();label.loc[eligible]=frame.loc[eligible,'soxx_exit_open'].to_numpy()/frame.loc[eligible,'soxx_open'].to_numpy()-1
    base=plan['market_features']+plan['baseline_extra_features'];full=base+[plan['candidate_extra_feature']]
    data=pd.concat({'feature':features[full],'label':label.to_frame(),'meta':meta},axis=1);data.index=pd.MultiIndex.from_arrays([meta.trade_date,['SOXL_SOXX_CASH']*len(meta)],names=['datetime','instrument'])
    rec.save_objects(artifact_path='inputs',data=data,source_audits=audits,opening_audit=oa,price_audit=pa,source_model_timing=source_checks)
    folds=[];daily=[];coefficients=[];scalers=[]
    for year in [2020,2021,2022]:
        ds=DatasetH(handler=DataHandlerLP.from_df(data),segments={'train':('2019-01-01',f'{year-1}-12-31'),'test':(f'{year}-01-01',f'{year}-12-31')})
        train=ds.prepare('train',col_set=['feature','label','meta']);test=ds.prepare('test',col_set=['feature','label','meta'])
        train=train.loc[mature_before(train[('meta','label_available_at')],plan['forecast_cutoffs'][str(year)])]
        if len(train)!=plan['expected_training_rows'][str(year)] or len(test)!=plan['expected_evaluation_rows'][str(year)]:raise ValueError('fold coverage changed')
        mean,scale=fit_scaler(train['feature'][full]);pred={}
        for arm,columns in [('baseline',base),('plus_cpi_interaction',full)]:
            key=f'{year}_{arm}';count=len(columns)
            if key in progress['fitted']:model=rec.load_object('models/'+key)
            else:
                if key in progress['fit_attempts'] or len(progress['fit_attempts'])>=6:raise ValueError('fit attempt budget exhausted')
                progress['fit_attempts'].append(key);save_progress()
                model=Ridge(alpha=10.,fit_intercept=True,solver='svd').fit(transform(train['feature'][columns],mean[:count],scale[:count]),train[('label','target')].to_numpy(float))
                rec.save_objects(artifact_path='models',**{key:model});progress['fitted'].append(key);save_progress()
            pred[arm]=model.predict(transform(test['feature'][columns],mean[:count],scale[:count]))
            coefficients.extend({'year':year,'arm':arm,'feature':x,'coefficient':float(v)} for x,v in zip(columns,model.coef_))
        part=test['meta'].reset_index(drop=True).copy();part['target']=test[('label','target')].to_numpy();part['cpi_flag']=test[('feature','cpi_released_before_decision')].to_numpy()
        for arm,value in pred.items():part[arm]=value
        part['paired_loss_improvement']=(part.target-part.baseline)**2-(part.target-part.plus_cpi_interaction)**2;daily.append(part)
        folds.append({'year':year,'training_rows':len(train),'evaluation_rows':len(test),'scored_rows':int(part.target.notna().sum()),'train_cpi_events':int(train[('feature','cpi_released_before_decision')].sum()),'test_cpi_events':int(part.cpi_flag.sum()),'baseline_mse':float(((part.target-part.baseline)**2).mean()),'candidate_mse':float(((part.target-part.plus_cpi_interaction)**2).mean()),'paired_mse_improvement':float(part.paired_loss_improvement.mean())})
        scalers.append({'year':year,'means':mean,'scales':scale,'features':full})
    daily=pd.concat(daily,ignore_index=True);folds=pd.DataFrame(folds)
    if len(daily)!=756 or daily.target.notna().sum()!=755:raise ValueError('full-calendar maturity differs')
    b=plan['bootstrap'];ci,indices,replicates=full_calendar_bootstrap(daily.paired_loss_improvement.to_numpy(),**{k:b[k] for k in ['mean_block','resamples','seed','alpha']})
    passed=bool(folds.paired_mse_improvement.gt(0).all() and ci['lower_bound']>0)
    groups=daily.groupby([daily.trade_date.dt.year,'cpi_flag']).paired_loss_improvement.agg(['mean','count']).reset_index()
    rec.save_objects(artifact_path='analysis',daily=daily,fold_metrics=folds,coefficients=pd.DataFrame(coefficients),scalers=scalers,bootstrap_indices=indices,bootstrap_replicates=replicates)
    for name,value in [('fold_metrics',folds),('coefficients',pd.DataFrame(coefficients)),('event_group_diagnostics',groups),('bootstrap_summary',pd.DataFrame([ci]))]:value.to_csv(card/(name+'.csv'),index=False)
    return {'information_gate_pass':passed,'bootstrap':ci,'decision':'conditional_information_pass_requires_separate_policy_test' if passed else 'not_promoted_cpi_context_information_failed'}


def run(*,root,card,plan):
    from qlib.workflow import R
    from qlib.data import D
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    path=card/'run_state.json';progress=json.loads(path.read_text()) if path.exists() else {'plan_sha256':digest,'request_attempts':[],'source_completed':False,'fit_attempts':[],'fitted':[]}
    if progress['plan_sha256']!=digest:raise ValueError('frozen plan differs')
    def save():atomic_json(progress,path)
    tracking=card/'qlib_recorder';kwargs={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kwargs.update(recorder_id=progress['recorder_id'],resume=True)
    with R.start(**kwargs):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));save()
        if not progress.get('source_saved'):
            rec.save_objects(artifact_path='inputs',plan=plan)
            rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),intake=(root/'src/data/cpi_release_calendar.py').read_text())
            progress['source_saved']=True;save()
        if progress['source_completed']:
            events=rec.load_object('sources/events');gate=rec.load_object('sources/gate')
        else:
            def attempt(url):
                if url in progress['request_attempts'] or len(progress['request_attempts'])>=49:raise ValueError('source request budget exhausted/no retry')
                progress['request_attempts'].append(url);save()
            calendar=pd.DatetimeIndex(D.calendar(start_time='2019-01-01',end_time='2022-12-31',freq='day'))
            corpus,receipts,events,gate=intake(plan,calendar,record_attempt=attempt)
            rec.save_objects(artifact_path='sources',raw_corpus=corpus,receipts=receipts,events=events,gate=gate)
            progress['source_completed']=True;save();atomic_json(gate,card/'source_gate.json');atomic_json(receipts,card/'source_receipts.json')
            events.to_csv(card/'event_manifest.csv',index=False)
    # Verified source reads require no active writable Recorder.
    result=fit_information(root,card,plan,rec,events,progress,save) if gate['passed'] else {'information_gate_pass':False,'decision':'needs_data_cpi_source_gate_failed_before_fit'}
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['sources/raw_corpus','inputs/*','models/*','analysis/daily','analysis/scalers','analysis/bootstrap_indices','analysis/bootstrap_replicates']}
    summary={'experiment':plan['experiment'],'plan_sha256':digest,**result,'source_gate':gate,'objective_achieved':False,'global_goal_blocked':False,'new_statistical_fits':len(progress['fitted']),'new_allocator_fits':0,'new_native_cases':0,'evaluation_as_of':plan['evaluation_as_of'],'provider_uri':str(root/plan['provider_uri']),'region':'us','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy}
    rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['sources']['frame']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

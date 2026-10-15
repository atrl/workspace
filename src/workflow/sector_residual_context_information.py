"""Fourteen fixed Ridge fits for one generic joint-history coordinate."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.workflow.minute_action_value import _source
from src.analysis.sector_residual_context import context_features,EXTRA,FEATURE
from src.analysis.seed_disagreement_information import fit_scaler,transform,full_calendar_bootstrap
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
NAME='sector_residual_context_information_20261015'
SHA='a13b47d17e1a2a629bb3fd4a55654027e99d5a9fed1833e87c439c790c961d66'

def ny(series):
    x=pd.to_datetime(series)
    return x.dt.tz_localize('America/New_York') if x.dt.tz is None else x.dt.tz_convert('America/New_York')

def run(root):
    import qlib,yaml
    from qlib.data import D
    from qlib.workflow import R
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader
    root=Path(root).resolve();card=root/'research'/NAME;plan=json.loads((card/'plan.json').read_text())
    assert hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()==SHA
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    for f,h in plan['source_code_sha256'].items():assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h
    qlib.init(provider_uri=str(root/'data/qlib/soxl'),region='us',kernels=1)
    src,audit=_source(root,plan['label_source'],plan['label_artifacts'])
    # Never persist or inspect later source labels; mask forward targets first.
    base=src['inputs/daily_frame'].loc['2011-01-03':'2022-12-30'];original=src['inputs/features'].loc['2011-01-03':'2022-12-30'];timing=src['inputs/timing'].loc['2011-01-03':'2022-12-30']
    maturity=ny(timing.label_available_at);asof=pd.Timestamp('2022-12-30 09:31',tz='America/New_York')
    target=base.soxx_forward_open_return.where(maturity.notna()&maturity.le(asof))
    if np.isinf(target.to_numpy(float)).any():raise ValueError('infinite H1 label')
    if not target.dropna().gt(-1).all():raise ValueError('invalid H1 label')
    prices={};calendars={}
    for symbol,cfg in plan['price_sources'].items():
        provider=root/cfg['uri'];assert hashlib.sha256((provider/'qlib_store_manifest.json').read_bytes()).hexdigest()==cfg['manifest_sha256']
        qlib.init(provider_uri=str(provider),region='us',kernels=1)
        calendar=pd.DatetimeIndex(D.calendar(start_time='2010-03-11',end_time='2022-12-30',freq='day')).rename('datetime')
        close=D.features([symbol],['$close'],start_time=calendar[0],end_time=calendar[-1],freq='day').xs(symbol,level='instrument')['$close']
        pd.testing.assert_index_equal(close.index.rename('datetime'),calendar);prices[symbol]=close;calendars[symbol]=calendar
    pd.testing.assert_index_equal(calendars['SOXX'],calendars['SPY'])
    closes=pd.DataFrame(prices);context,trace=context_features(closes)
    calendar=calendars['SOXX'][calendars['SOXX']>=pd.Timestamp('2011-01-03')]
    for x in [base,original,timing]:pd.testing.assert_index_equal(x.index.rename('datetime'),calendar)
    baseline=plan['features']['ordered_baseline'];full=plan['features']['ordered_candidate'];features=original[baseline[:39]].join(context[EXTRA+[FEATURE]],how='left').reindex(calendar)
    if np.isinf(features.to_numpy(float)).any():raise ValueError('infinite feature')
    meta=pd.DataFrame({'signal_date':calendar,'mature_at':maturity.reindex(calendar).to_numpy(),'decision_at':ny(timing.decision_at).reindex(calendar).to_numpy()},index=calendar)
    label=pd.DataFrame({'target':target.reindex(calendar)},index=calendar)
    static=pd.concat({'feature':features,'label':label,'meta':meta},axis=1);static.index=pd.MultiIndex.from_arrays([calendar,['SOXX']*len(calendar)],names=['datetime','instrument'])
    dataset=DatasetH(handler=DataHandlerLP(data_loader=StaticDataLoader(config=static),infer_processors=[],learn_processors=[]),segments={'all':('2011-01-03','2022-12-30')})
    source=dataset.prepare('all',col_set=['feature','label','meta']);source.index=source.index.get_level_values('datetime');assert len(source)==len(calendar)
    progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':SHA,'fit_attempts':[],'fitted':[]}
    assert progress['plan_sha256']==SHA
    tracking=card/'qlib_recorder';kw={'experiment_name':NAME,'uri':str(tracking)}
    if progress.get('recorder_id'):kw.update(recorder_id=progress['recorder_id'],resume=True)
    rows=[];folds=[];coefs=[];scalers=[]
    with R.start(**kw):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        if not progress.get('sources_saved'):
            rec.save_objects(artifact_path='inputs',plan=plan,source_audit=audit,closes=closes,feature_trace=trace,dataset_frame=source,original_plan=src['inputs/plan'])
            rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),feature=(root/'src/analysis/sector_residual_context.py').read_text(),bootstrap=(root/'src/analysis/seed_disagreement_information.py').read_text())
            progress['sources_saved']=True;atomic_json(progress,progress_path)
        for year in plan['model']['folds']:
            cutoff=pd.Timestamp(f'{year}-01-01',tz='America/New_York');mature=pd.to_datetime(source[('meta','mature_at')]);trainmask=(source.index<pd.Timestamp(f'{year}-01-01'))&mature.notna()&mature.lt(cutoff)&source[('label','target')].notna()
            train=source.loc[trainmask];test=source.loc[source.index.year==year]
            expected=calendar[calendar.year==year];pd.testing.assert_index_equal(test.index.rename('datetime'),expected.rename('datetime'))
            if not len(train) or not len(test):raise ValueError('empty fixed fold')
            mean,scale=fit_scaler(train['feature'][full]);base_mean,base_scale=fit_scaler(train['feature'][baseline]);np.testing.assert_array_equal(mean[:45],base_mean);np.testing.assert_array_equal(scale[:45],base_scale)
            predictions={}
            for arm,columns in [('baseline',baseline),('candidate',full)]:
                name=f'{year}_{arm}'
                if name in progress['fitted']:model=rec.load_object('models/'+name)
                else:
                    if name in progress['fit_attempts'] or len(progress['fit_attempts'])>=14:raise ValueError('fixed fit budget exhausted')
                    progress['fit_attempts'].append(name);atomic_json(progress,progress_path)
                    model=Ridge(alpha=10.,fit_intercept=True,solver='svd').fit(transform(train['feature'][columns],mean[:len(columns)],scale[:len(columns)]),train[('label','target')].to_numpy(float))
                    rec.save_objects(artifact_path='models',**{name:model});progress['fitted'].append(name);atomic_json(progress,progress_path)
                predictions[arm]=model.predict(transform(test['feature'][columns],mean[:len(columns)],scale[:len(columns)]))
                coefs.extend({'year':year,'arm':arm,'feature':k,'coefficient':float(v)} for k,v in zip(columns,model.coef_))
            part=test['meta'].reset_index(drop=True).copy();part['target']=test[('label','target')].to_numpy(float)
            for arm,pred in predictions.items():part[arm]=pred
            part['paired_loss_improvement']=(part.target-part.baseline)**2-(part.target-part.candidate)**2;rows.append(part)
            folds.append({'year':year,'train_rows':len(train),'first_train_signal':str(train.index[0]),'last_train_signal':str(train.index[-1]),'last_train_maturity':str(mature[trainmask].max()),'calendar_rows':len(test),'scored_rows':int(part.target.notna().sum()),'baseline_mse':float(((part.target-part.baseline)**2).mean()),'candidate_mse':float(((part.target-part.candidate)**2).mean()),'paired_mse_improvement':float(part.paired_loss_improvement.mean()),'train_residual_unknown':int(train[('feature',FEATURE)].isna().sum()),'test_residual_unknown':int(test[('feature',FEATURE)].isna().sum())})
            scalers.append({'year':year,'features':full,'means':mean,'scales':scale,'training_dates':train.index})
            rec.save_objects(artifact_path='folds',**{str(year):{'daily':part,'metrics':folds[-1],'scaler':scalers[-1]}});rec.log_metrics(completed_fits=len(progress['fitted']));print('completed fold',year,folds[-1],flush=True)
        daily=pd.concat(rows,ignore_index=True);expected=calendar[calendar.year>=2016];pd.testing.assert_index_equal(pd.DatetimeIndex(daily.signal_date).rename(None),expected.rename(None))
        ci,indices,replicates=full_calendar_bootstrap(daily.paired_loss_improvement.to_numpy(),**plan['primary_test']['bootstrap'])
        fold_table=pd.DataFrame(folds);passed=bool(fold_table.paired_mse_improvement.gt(0).sum()>=5 and ci['lower_bound']>0)
        quarter=daily.groupby(daily.signal_date.dt.to_period('Q')).agg(calendar_rows=('target','size'),scored_rows=('target','count'),paired_mse_improvement=('paired_loss_improvement','mean')).reset_index().rename(columns={'signal_date':'quarter'});quarter['quarter']=quarter.quarter.astype(str)
        tables={'fold_metrics':fold_table,'quarter_metrics':quarter,'coefficients':pd.DataFrame(coefs),'bootstrap_summary':pd.DataFrame([ci])}
        rec.save_objects(artifact_path='analysis',daily=daily,scalers=scalers,bootstrap_indices=indices,bootstrap_replicates=replicates,**tables)
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','models/*','folds/*','analysis/daily','analysis/scalers','analysis/bootstrap_indices','analysis/bootstrap_replicates']}
        summary={'experiment':NAME,'plan_sha256':SHA,'decision':'conditional_information_pass_requires_separate_policy_test' if passed else 'not_promoted_fixed_beta_adjusted_context_information','information_gate_pass':passed,'positive_heldout_years':int(fold_table.paired_mse_improvement.gt(0).sum()),'all_heldout_years_improve':bool(fold_table.paired_mse_improvement.gt(0).all()),'bootstrap':ci,'Ridge_fits':14,'new_policy_accounts':0,'new_fetches':0,'evaluation_asof':str(asof),'evaluation_calendar_rows':len(daily),'mature_scored_rows':int(daily.target.notna().sum()),'objective_achieved':False,'fresh_independent_validation':False,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    for key,value in tables.items():value.to_csv(card/(key+'.csv'),index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['label_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'plan':'plan.json','module':'src.workflow.sector_residual_context_information','artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');progress_path.rename(card/'execution_attempts.json');return summary

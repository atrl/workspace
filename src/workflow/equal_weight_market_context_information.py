"""Fixed four-request source gate followed, only on pass, by fourteen Ridge fits."""
import hashlib,json,datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from src.data.equal_weight_source import parse_chart
from src.data import build_qlib_store_from_frame
from src.workflow.minute_action_value import _source
from src.analysis.seed_disagreement_information import fit_scaler,transform,full_calendar_bootstrap
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
NAME='equal_weight_market_context_information_20261015'
SHA='d459a7c53fe187c59e49300a025ee6adf3df04d23fb1ed886108aeb7fed02208'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ny(x):
    x=pd.to_datetime(x);return x.dt.tz_localize('America/New_York') if x.dt.tz is None else x.dt.tz_convert('America/New_York')

def fit_information(root,card,plan,rec,progress,prices):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader
    src,audit=_source(root,plan['label_source'],plan['label_artifacts'])
    base=src['inputs/daily_frame'].loc['2011-01-03':'2022-12-30'];original=src['inputs/features'].loc['2011-01-03':'2022-12-30'];timing=src['inputs/timing'].loc['2011-01-03':'2022-12-30']
    maturity=ny(timing.label_available_at);asof=pd.Timestamp('2022-12-30 09:31',tz='America/New_York');target=base.soxx_forward_open_return.where(maturity.notna()&maturity.le(asof))
    if np.isinf(target.to_numpy(float)).any() or not target.dropna().gt(-1).all():raise ValueError('invalid masked H1 target')
    calendar=prices.index[prices.index>=pd.Timestamp('2011-01-03')]
    for x in [base,original,timing]:pd.testing.assert_index_equal(x.index.rename(None),calendar.rename(None))
    momentum=np.log(prices/prices.shift(20));context=pd.DataFrame({'spy_vendor_logmom20':momentum.SPY,'rsp_minus_spy_vendor_logmom20':momentum.RSP-momentum.SPY}).reindex(calendar)
    if not np.isfinite(context).all().all():raise ValueError('source gate: unknown new momentum after warmup')
    baseline=plan['features']['ordered_baseline'];full=plan['features']['ordered_candidate'];features=original[baseline[:39]].join(context)
    if np.isinf(features.to_numpy(float)).any():raise ValueError('infinite original feature')
    meta=pd.DataFrame({'signal_date':calendar,'mature_at':maturity.reindex(calendar).to_numpy(),'decision_at':ny(timing.decision_at).reindex(calendar).to_numpy()},index=calendar)
    static=pd.concat({'feature':features,'label':pd.DataFrame({'target':target},index=calendar),'meta':meta},axis=1);static.index=pd.MultiIndex.from_arrays([calendar,['SOXX']*len(calendar)],names=['datetime','instrument'])
    dataset=DatasetH(handler=DataHandlerLP(data_loader=StaticDataLoader(config=static),infer_processors=[],learn_processors=[]),segments={'all':('2011-01-03','2022-12-30')})
    source=dataset.prepare('all',col_set=['feature','label','meta']);source.index=source.index.get_level_values('datetime')
    rec.save_objects(artifact_path='inputs',label_source_audit=audit,dataset_frame=source,original_plan=src['inputs/plan'])
    daily=[];folds=[];scalers=[];coefs=[]
    for year in plan['model']['folds']:
        cutoff=pd.Timestamp(f'{year}-01-01',tz='America/New_York');mature=pd.to_datetime(source[('meta','mature_at')]);mask=(source.index<pd.Timestamp(f'{year}-01-01'))&mature.notna()&mature.lt(cutoff)&source[('label','target')].notna();train=source.loc[mask];test=source.loc[source.index.year==year]
        pd.testing.assert_index_equal(test.index.rename(None),calendar[calendar.year==year].rename(None));mean,scale=fit_scaler(train['feature'][full]);m0,s0=fit_scaler(train['feature'][baseline]);np.testing.assert_array_equal(mean[:40],m0);np.testing.assert_array_equal(scale[:40],s0)
        predictions={}
        for arm,columns in [('baseline',baseline),('candidate',full)]:
            key=f'{year}_{arm}';n=len(columns)
            if key in progress['fitted']:model=rec.load_object('models/'+key)
            else:
                if key in progress['fit_attempts'] or len(progress['fit_attempts'])>=14:raise ValueError('fixed fit budget exhausted')
                progress['fit_attempts'].append(key);atomic_json(progress,card/'run_state.json')
                model=Ridge(alpha=10.,fit_intercept=True,solver='svd').fit(transform(train['feature'][columns],mean[:n],scale[:n]),train[('label','target')].to_numpy(float));rec.save_objects(artifact_path='models',**{key:model});progress['fitted'].append(key);atomic_json(progress,card/'run_state.json')
            predictions[arm]=model.predict(transform(test['feature'][columns],mean[:n],scale[:n]));coefs.extend({'year':year,'arm':arm,'feature':k,'coefficient':float(v)} for k,v in zip(columns,model.coef_))
        part=test['meta'].reset_index(drop=True).copy();part['target']=test[('label','target')].to_numpy(float)
        for arm,pred in predictions.items():part[arm]=pred
        part['paired_loss_improvement']=(part.target-part.baseline)**2-(part.target-part.candidate)**2;daily.append(part)
        folds.append({'year':year,'train_rows':len(train),'last_train_signal':str(train.index[-1]),'last_train_maturity':str(mature[mask].max()),'calendar_rows':len(test),'scored_rows':int(part.target.notna().sum()),'baseline_mse':float(((part.target-part.baseline)**2).mean()),'candidate_mse':float(((part.target-part.candidate)**2).mean()),'paired_mse_improvement':float(part.paired_loss_improvement.mean())});scalers.append({'year':year,'features':full,'mean':mean,'scale':scale,'training_dates':train.index});rec.save_objects(artifact_path='folds',**{str(year):{'daily':part,'scaler':scalers[-1],'metrics':folds[-1]}});rec.log_metrics(completed_fits=len(progress['fitted']));print('completed fold',year,folds[-1],flush=True)
    daily=pd.concat(daily,ignore_index=True);pd.testing.assert_index_equal(pd.DatetimeIndex(daily.signal_date).rename(None),calendar[calendar.year>=2016].rename(None))
    ci,indices,reps=full_calendar_bootstrap(daily.paired_loss_improvement.to_numpy(),**plan['primary_test']['bootstrap']);folds=pd.DataFrame(folds);positive=int(folds.paired_mse_improvement.gt(0).sum());passed=positive>=5 and ci['lower_bound']>0
    quarter=daily.groupby(daily.signal_date.dt.to_period('Q')).agg(calendar_rows=('target','size'),scored_rows=('target','count'),paired_mse_improvement=('paired_loss_improvement','mean')).reset_index().rename(columns={'signal_date':'quarter'});quarter['quarter']=quarter.quarter.astype(str)
    tables={'fold_metrics':folds,'quarter_metrics':quarter,'coefficients':pd.DataFrame(coefs),'bootstrap_summary':pd.DataFrame([ci])};rec.save_objects(artifact_path='analysis',daily=daily,scalers=scalers,bootstrap_indices=indices,bootstrap_replicates=reps,**tables)
    for key,value in tables.items():value.to_csv(card/(key+'.csv'),index=False)
    return {'decision':'conditional_information_pass_requires_separate_policy_test' if passed else 'not_promoted_fixed_equal_weight_context_information','information_gate_pass':bool(passed),'positive_heldout_years':positive,'bootstrap':ci,'Ridge_fits':14,'evaluation_calendar_rows':len(daily),'mature_scored_rows':int(daily.target.notna().sum())}

def run(root):
    import qlib,yaml
    from qlib.data import D
    from qlib.workflow import R
    root=Path(root).resolve();card=root/'research'/NAME;plan=json.loads((card/'plan.json').read_text());assert hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()==SHA
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    for f,h in plan['source_code_sha256'].items():assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h
    assert hashlib.sha256((root/'data/qlib/soxl/qlib_store_manifest.json').read_bytes()).hexdigest()==plan['Qlib_source_contract']['calendar_manifest_sha256']
    qlib.init(provider_uri=str(root/'data/qlib/soxl'),region='us',kernels=1);calendar=pd.DatetimeIndex(D.calendar(start_time='2010-03-11',end_time='2022-12-30',freq='day'))
    progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':SHA,'http_attempts':[],'fit_attempts':[],'fitted':[]}
    assert progress['plan_sha256']==SHA;tracking=card/'qlib_recorder';kwargs={'experiment_name':NAME,'uri':str(tracking)}
    if progress.get('recorder_id'):kwargs.update(recorder_id=progress['recorder_id'],resume=True)
    with R.start(**kwargs):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
        if not progress.get('sources_saved'):
            rec.save_objects(artifact_path='inputs',plan=plan,calendar=calendar);rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),parser=(root/'src/data/equal_weight_source.py').read_text(),bootstrap=(root/'src/analysis/seed_disagreement_information.py').read_text());progress['sources_saved']=True;atomic_json(progress,progress_path)
    frames=[];actions=[];metadata={};failure=None
    for symbol in plan['source_intake']['symbols']:
        success=False
        for host in plan['source_intake']['hosts_in_order']:
            key=f'{symbol}_{host.split(".")[0]}'
            if any(a['key']==key for a in progress['http_attempts']):raise ValueError('recorded request cannot be repeated silently')
            if len(progress['http_attempts'])>=4:raise ValueError('HTTP budget exhausted')
            url=plan['source_intake']['path_template'].format(host=host,symbol=symbol);receipt={'key':key,'symbol':symbol,'requested_url':url,'params':plan['source_intake']['params'],'request_started_at_utc':now(),'completed':False};progress['http_attempts'].append(receipt);atomic_json(progress,progress_path);rec.save_objects(artifact_path='source_attempts',**{key:dict(receipt)});body=None;payload=None;fallback=False
            try:
                response=requests.get(url,params=plan['source_intake']['params'],headers=plan['source_intake']['headers'],timeout=30,allow_redirects=False);body=response.content;receipt.update(http_status=response.status_code,resolved_url=response.url,response_received_at_utc=now(),response_bytes=len(body),response_sha256=hashlib.sha256(body).hexdigest())
                if response.status_code!=200:receipt['failure']='HTTP_'+str(response.status_code);fallback=True
                else:
                    try:payload=json.loads(body)
                    except (ValueError,UnicodeDecodeError) as error:receipt['failure']='nonJSON:'+str(error);fallback=True
            except requests.RequestException as error:receipt.update(failure=repr(error),response_received_at_utc=now());fallback=True
            receipt['completed']=True;rec.save_objects(artifact_path='source_attempts',**{key:{'receipt':dict(receipt),'raw_response_bytes':body}});atomic_json(progress,progress_path);print('source attempt',key,receipt.get('http_status'),receipt.get('failure'),flush=True)
            if fallback:continue
            try:frame,event,meta=parse_chart(payload,symbol,calendar)
            except (ValueError,AssertionError,KeyError,TypeError) as error:failure={'symbol':symbol,'stage':'valid_JSON_semantic_source_gate','error':repr(error)};break
            frames.append(frame);actions.append(event);metadata[symbol]=meta;rec.save_objects(artifact_path='qualified_source',**{symbol:{'frame':frame,'events':event,'metadata':meta,'response_sha256':receipt['response_sha256']}});success=True;break
        if not success:
            if failure is None:failure={'symbol':symbol,'stage':'HTTP_transport_format_budget','error':'all fixed same-vendor hosts failed'}
            break
    if failure is None:
        data=pd.concat(frames,ignore_index=True)
        for field in ['open','high','low','close']:data[field]=data['quote_'+field]
        fields=['open','high','low','close','volume','quote_open','quote_high','quote_low','quote_close','vendor_adj_close'];provider=root/plan['Qlib_source_contract']['derived_provider']
        manifest=build_qlib_store_from_frame(data,provider_uri=provider,freq='day',region='us',as_of='2022-12-30',field_map={x:x for x in fields},source='Fixed Yahoo RSP/SPY current-vintage quote+adjusted proxy; not physical-share/total-return/PIT qualification')
        qlib.init(provider_uri=str(provider),region='us',kernels=1);read=D.features(['RSP','SPY'],['$vendor_adj_close','$quote_close'],start_time=calendar[0],end_time=calendar[-1],freq='day');errors=[]
        for symbol in ['RSP','SPY']:
            actual=read.xs(symbol,level='instrument');expected=data[data.instrument.eq(symbol)].set_index('datetime');pd.testing.assert_index_equal(actual.index.rename(None),calendar.rename(None))
            for field in ['vendor_adj_close','quote_close']:
                np.testing.assert_allclose(actual['$'+field].to_numpy(),expected[field].to_numpy(),rtol=1e-7,atol=0);errors.append({'symbol':symbol,'field':field,'max_relative_error':float(np.max(np.abs(actual['$'+field].to_numpy()/expected[field].to_numpy()-1)))})
        rec.save_objects(artifact_path='qualified_source',provider_manifest=manifest,Qlib_readback=read,quantization=pd.DataFrame(errors),actions=pd.concat(actions,ignore_index=True));atomic_json(manifest,card/'provider_manifest.json');prices=read['$vendor_adj_close'].unstack('instrument')[['RSP','SPY']].astype(float)
        result=fit_information(root,card,plan,rec,progress,prices)
    else:result={'decision':'needs_data_fixed_RSP_SPY_source_gate','source_failure':failure,'information_gate_pass':False,'Ridge_fits':0}
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','source_attempts/*','qualified_source/*','models/*','folds/*','analysis/daily','analysis/scalers','analysis/bootstrap_indices','analysis/bootstrap_replicates']}
    summary={'experiment':NAME,'plan_sha256':SHA,**result,'HTTP_attempts':len(progress['http_attempts']),'source_gate_pass':failure is None,'new_policy_accounts':0,'objective_achieved':False,'historical_PIT_qualified':False,'total_return_account_qualified':False,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy}
    rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json(progress['http_attempts'],card/'source_attempts.json');atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['label_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json');(card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'plan':'plan.json','module':'src.workflow.equal_weight_market_context_information','artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');progress_path.rename(card/'execution_attempts.json');return summary

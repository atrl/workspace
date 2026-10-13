"""Fixed clock-corrected Asian information screen on unelapsed US RTH returns."""
import json,hashlib,io
from pathlib import Path
import pandas as pd
import numpy as np
from src.data.asia_same_session import build_asia_features
from src.model.revenue_forecast import RevenueForecast
from src.workflow.minute_action_value import _source
from src.workflow.fed_capacity_information import _bootstrap
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed Asia plan changed')
        return old
    source,audit=_source(root,plan['source'],['inputs/grouped'])
    prices,price_audit=_source(root,plan['price_source'],['inputs/frame'])
    grouped=source['inputs/grouped'].copy();dates=pd.DatetimeIndex(grouped.index.get_level_values('datetime'));frame=prices['inputs/frame'].reindex(dates)
    price_calendar=pd.DatetimeIndex(D.calendar(freq='1min'));close_times=pd.Series(price_calendar,index=price_calendar.normalize()).groupby(level=0).max().reindex(dates)
    if close_times.isna().any():raise ValueError('native session close missing')
    raw_bytes=(root/plan['raw_csv']).read_bytes();manifest_bytes=(root/plan['raw_manifest']).read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest()!=plan['raw_csv_sha256'] or hashlib.sha256(manifest_bytes).hexdigest()!=plan['raw_manifest_sha256']:raise ValueError('Asian raw identity differs')
    raw=pd.read_csv(io.BytesIO(raw_bytes),index_col=0,parse_dates=True);manifest=json.loads(manifest_bytes)
    if raw.index.max()>pd.Timestamp(plan['as_of']):raise ValueError('raw source exceeds as_of')
    for market,zone in plan['markets'].items():
        if manifest['series'][market]['timezone']!=zone:raise ValueError('source timezone differs')
    provider=root/'data/qlib'/plan['experiment'];features,alignment,store=build_asia_features(raw,dates,plan=plan,provider=provider)
    for col in features:grouped[('feature',col)]=features[col].to_numpy()
    soxx=np.log(frame.close_soxx.to_numpy(float)/frame.soxx_open.to_numpy(float));soxl=np.log(frame.close_soxl.to_numpy(float)/frame.soxl_open.to_numpy(float))
    grouped[('label','cash_relative_rth')]=-soxx;grouped[('label','leverage_relative_rth')]=soxl-soxx
    grouped[('meta','label_available_at')]=(close_times+pd.Timedelta(minutes=1)).to_numpy()
    if not np.isfinite(grouped['label'][plan['targets']]).all().all():raise ValueError('RTH target missing')
    controls=[m+'_'+f for m in plan['markets'] for f in ['known','gap_days']];values=[m+'_return_value' for m in plan['markets']]
    handler=DataHandlerLP.from_df(grouped);tracking=card/'qlib_recorder';models={};predictions={};metrics=[];increments=[];losses={};audits=[]
    with R.start(experiment_name=plan['experiment'],recorder_name='same_local_date_remaining_session',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,maximum_model_fits=6,new_backtests=0)
        rec.save_objects(artifact_path='inputs',plan=plan,grouped=grouped,source_audit=audit,price_audit=price_audit,alignment=alignment,provider_manifest=store,provider_files={str(p.relative_to(provider)):p.read_bytes() for p in provider.rglob('*') if p.is_file()})
        rec.save_objects(artifact_path='sources',raw_csv=raw_bytes,raw_manifest=manifest_bytes,workflow=Path(__file__).read_text(),data=(root/'src/data/asia_same_session.py').read_text())
        for fold in plan['folds']:
            cutoff=pd.Timestamp(fold['cutoff']);name=fold['name'];dataset=DatasetH(handler=handler,segments={'train':(None,cutoff-pd.Timedelta(nanoseconds=1)),'test':(fold['start'],fold['end'])})
            test=dataset.prepare('test',col_set=['label','meta']);truth=test['label'][plan['targets']];valid=pd.to_datetime(test['meta'].label_available_at).le(pd.Timestamp(plan['as_of'])+pd.Timedelta(hours=16,minutes=1))
            for family,columns in [('control',plan['base_features']+controls),('asia',plan['base_features']+controls+values)]:
                if len(models)>=plan['maximum_model_fits']:raise ValueError('fit budget exhausted')
                model=RevenueForecast(columns,plan['targets'],cutoff,plan['ridge_alpha']).fit(dataset);models[(name,family)]=model
                prediction=model.predict(dataset);predictions[(name,family)]={'prediction':prediction,'truth':truth,'label_mature':valid};audits.append({'fold':name,'family':family,**model.audit})
                rec.save_objects(artifact_path='models',**{name+'_'+family:model})
                for target in plan['targets']:
                    error=prediction.loc[valid,target]-truth.loc[valid,target];metrics.append({'fold':name,'family':family,'target':target,'scored_rows':int(valid.sum()),'rmse':float(np.sqrt(np.mean(error**2))),'bias':float(error.mean()),'rank_ic':float(prediction.loc[valid,target].corr(truth.loc[valid,target],method='spearman'))})
            left,right=[models[(name,f)] for f in ['control','asia']]
            if left.audit['training_rows']!=right.audit['training_rows'] or left.audit['latest_label_available_at']!=right.audit['latest_label_available_at']:raise ValueError('paired training domain differs')
            normalized={family:((predictions[(name,family)]['prediction'].loc[valid].to_numpy()-truth.loc[valid].to_numpy())/left.y_scaler.scale_)**2 for family in ['control','asia']}
            losses[name]=(normalized['control']-normalized['asia']).mean(axis=1);increments.append({'fold':name,**_bootstrap(losses[name],plan)});rec.log_metrics(completed_fits=len(models));print('completed',name,flush=True)
        pooled=_bootstrap(np.concatenate([v for k,v in losses.items() if k.startswith('development')]),plan);increments.append({'fold':'pooled_development',**pooled})
        passed=all(r['mean_joint_normalized_loss_improvement']>0 for r in increments if r['fold'].startswith('development')) and pooled['lower_bound']>0
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'information_pass_needs_full_account_test' if passed else 'not_promoted_fixed_asia_remaining_session_failed','information_gate_pass':bool(passed),'objective_achieved':False,'new_fits':len(models),'new_backtests':0,'rows':len(grouped),'unknown_feature_rows':{m:int(features[m+'_known'].eq(0).sum()) for m in plan['markets']},'actual_provider_first_receipt_certified':False,'source_timezone_labels_only':True,'availability_role':plan['availability_role'],'provider_uri':str(provider),'price_provider_uri':str(root/'data/qlib/full_path_tiered_allocation_20261013'),'region':'us','as_of':plan['as_of'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/data.html','artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','models/*','analysis/predictions','analysis/losses']}}
        rec.save_objects(artifact_path='analysis',summary=summary,predictions=predictions,losses=losses,training=audits,metrics=pd.DataFrame(metrics),increments=pd.DataFrame(increments))
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    pd.DataFrame(metrics).to_csv(card/'forecast_metrics.csv',index=False);pd.DataFrame(increments).to_csv(card/'information_increment.csv',index=False)
    pd.DataFrame([{k:r[k] for k in ['fold','family','training_rows','latest_label_available_at','cutoff']} for r in audits]).to_csv(card/'training_audit.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

"""Paired fixed SKEW risk/return information gate, without portfolio claims."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.data.skew_information import lagged_skew
from src.analysis.allocator_cash_information import cash_labels
from src.model.revenue_forecast import RevenueForecast
from src.workflow.fed_capacity_information import _bootstrap
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed SKEW plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    source,audit=_source(root,plan['source'],['inputs/grouped','analysis/predictions'])
    skew,skew_audit=_source(root,plan['skew_source'],['inputs/native','sources/reference_calendar'])
    prices,price_audit=_source(root,plan['price_source'],['inputs/frame'])
    grouped=source['inputs/grouped'].copy();dates=pd.DatetimeIndex(grouped.index.get_level_values('datetime'))
    aligned=lagged_skew(skew['inputs/native'],skew['sources/reference_calendar'],dates,lag=plan['source_lag_sessions'],eligible_after=plan['source_eligible_after'])
    for col in ['skew_known','skew_scaled']:grouped[('feature',col)]=aligned[col].to_numpy()
    risk=cash_labels(prices['inputs/frame'],horizons=[1,5],tail_threshold=-.05).set_index('trade_date').reindex(dates)
    grouped[('label','downside_h5')]=risk.downside_h5.to_numpy()
    avail=pd.DataFrame({'parent':pd.to_datetime(grouped[('meta','label_available_at')]).to_numpy(),'risk':pd.to_datetime(risk.label_available_h5).to_numpy()},index=grouped.index)
    grouped[('meta','label_available_at')]=avail.max(axis=1).where(avail.risk.notna())
    handler=DataHandlerLP.from_df(grouped);tracking=card/'qlib_recorder';metrics=[];increments=[];losses={};predictions={};models={};audits=[]
    with R.start(experiment_name=plan['experiment'],recorder_name='fixed_skew_ridge10',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,maximum_model_fits=6,new_backtests=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audit=audit,skew_audit=skew_audit,price_audit=price_audit,grouped=grouped,alignment=aligned)
        for fold in plan['folds']:
            name=fold['name'];saved=source['analysis/predictions'][(name,'price_calendar')];index=saved['prediction'].index
            cutoff=pd.Timestamp(fold['cutoff']);dataset=DatasetH(handler=handler,segments={'train':(None,cutoff-pd.Timedelta(nanoseconds=1)),'test':(index.get_level_values('datetime').min(),index.get_level_values('datetime').max())})
            truth=grouped.loc[index,'label'][plan['targets']]
            valid=saved['label_mature'] & truth.notna().all(axis=1) & pd.to_datetime(grouped.loc[index,('meta','label_available_at')]).le(pd.Timestamp(plan['as_of'])+pd.Timedelta(hours=16))
            for family,columns in [('control',plan['features']+['skew_known']),('skew',plan['features']+['skew_known','skew_scaled'])]:
                if len(models)>=plan['maximum_model_fits']:raise ValueError('fit budget exhausted')
                model=RevenueForecast(columns,plan['targets'],cutoff,plan['ridge_alpha']).fit(dataset);models[(name,family)]=model
                prediction=model.predict(dataset).loc[index];predictions[(name,family)]={'prediction':prediction,'truth':truth,'label_mature':valid}
                audits.append({'fold':name,'family':family,**model.audit});rec.save_objects(artifact_path='models',**{name+'_'+family:model})
                for target in plan['targets']:
                    error=prediction.loc[valid,target]-truth.loc[valid,target]
                    metrics.append({'fold':name,'family':family,'target':target,'scored_rows':int(valid.sum()),'rmse':float(np.sqrt(np.mean(error**2))),'bias':float(error.mean()),'rank_ic':float(prediction.loc[valid,target].corr(truth.loc[valid,target],method='spearman'))})
            left,right=[models[(name,f)] for f in ['control','skew']]
            if left.audit['training_rows']!=right.audit['training_rows'] or left.audit['latest_label_available_at']!=right.audit['latest_label_available_at']:raise ValueError('paired sample maturity differs')
            normalized={family:((predictions[(name,family)]['prediction'].loc[valid].to_numpy()-truth.loc[valid].to_numpy())/left.y_scaler.scale_)**2 for family in ['control','skew']}
            for role,columns in [('risk',[2]),('returns',[0,1])]:
                value=(normalized['control'][:,columns]-normalized['skew'][:,columns]).mean(axis=1);losses[(name,role)]=value;increments.append({'fold':name,'role':role,**_bootstrap(value,plan)})
            rec.log_metrics(completed_fits=len(models));print('completed',name,flush=True)
        for role in ['risk','returns']:
            value=np.concatenate([loss for (fold,r),loss in losses.items() if fold.startswith('development') and r==role]);increments.append({'fold':'pooled_development','role':role,**_bootstrap(value,plan)})
        table=pd.DataFrame(increments);gates={role:bool(table.loc[table.role.eq(role)&table.fold.str.startswith('development'),'mean_joint_normalized_loss_improvement'].gt(0).all() and table.loc[table.role.eq(role)&table.fold.eq('pooled_development'),'lower_bound'].gt(0).all()) for role in ['risk','returns']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'information_pass_needs_full_account_and_vintage_validation' if all(gates.values()) else 'not_promoted_fixed_skew_information_failed','information_gates':gates,'objective_achieved':False,'new_fits':len(models),'new_backtests':0,'rows':len(grouped),'unknown_skew_rows':int(aligned.skew_known.eq(0).sum()),'historical_vintage_certified':False,'normalization':'same control training target scale for both methods','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/'data/qlib/full_path_tiered_allocation_20261013'),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/model.html','artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','models/*','analysis/predictions','analysis/losses','sources/*']}}
        rec.save_objects(artifact_path='analysis',predictions=predictions,losses=losses,metrics=pd.DataFrame(metrics),increments=table,training=audits,summary=summary)
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),alignment=(root/'src/data/skew_information.py').read_text(),model=(root/'src/model/revenue_forecast.py').read_text())
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    pd.DataFrame(metrics).to_csv(card/'forecast_metrics.csv',index=False);table.to_csv(card/'information_increment.csv',index=False)
    pd.DataFrame([{k:r[k] for k in ['fold','family','training_rows','latest_label_available_at','cutoff']} for r in audits]).to_csv(card/'training_audit.csv',index=False)
    aligned.loc[aligned.skew_known.eq(0),['source_date','assumed_available_at']].to_csv(card/'unknown_sessions.csv',index_label='trade_date')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

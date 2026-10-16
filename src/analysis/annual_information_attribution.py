"""Exact accounting of frozen paired monthly forecasts, without model inference."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import pandas as pd

NAME='etf_annual_information_attribution_20261016'
PLAN_SHA='0723e0378b324066a2f685731f3e7caabd68ce1300959f8114c6bac72f7a7c36'


def saved_transform_arithmetic(model, frame):
    values=frame[model.features].replace([np.inf,-np.inf],np.nan).fillna(model.medians)
    values=(values-model.means)/model.scales
    values.loc[:,~model.active]=0.
    if not np.isfinite(values.to_numpy(float)).all():raise ValueError('nonfinite stored-transform arithmetic')
    return values


def decomposition(e,d):
    e=np.asarray(e,float);d=np.asarray(d,float)
    mean_alignment=2*e.mean()*d.mean()
    covariance_alignment=2*np.mean((e-e.mean())*(d-d.mean()))
    mean_shift_penalty=d.mean()**2
    variation_penalty=np.mean((d-d.mean())**2)
    result={'mean_control_residual':float(e.mean()),'mean_forecast_delta':float(d.mean()),
            'mean_residual_delta_alignment':float(mean_alignment),'covariance_alignment':float(covariance_alignment),
            'mean_shift_penalty':float(mean_shift_penalty),'delta_variation_penalty':float(variation_penalty),
            'total_alignment':float(np.mean(2*e*d)),'total_shift_penalty':float(np.mean(d*d)),
            'paired_improvement':float(np.mean(2*e*d-d*d))}
    np.testing.assert_allclose(mean_alignment+covariance_alignment-mean_shift_penalty-variation_penalty,result['paired_improvement'],atol=1e-12)
    return result


def run(root):
    import qlib,yaml
    from qlib.workflow import R
    from src.workflow.minute_action_value import _source
    from src.data.minute_intake import atomic_json
    from src.data.nas_market_data import archive_nas_recorder
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root).resolve();card=root/'research'/NAME
    plan=json.loads((card/'plan.json').read_text());review=json.loads((card/'design_review.json').read_text())
    if hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()!=PLAN_SHA or review.get('plan_sha256')!=PLAN_SHA or not str(review.get('decision','')).startswith('approved'):raise ValueError('design identity/approval differs')
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    for name,digest in plan['source_code_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('frozen model class definition differs')
    qlib.init(provider_uri=plan['provider_uri'],region='us',kernels=1)
    objects,audit=_source(root,plan['source'],plan['source_artifacts'])
    scores=objects['analysis/monthly_scores'];frame=objects['inputs/labelled_frame'];heads=plan['heads'];groups=plan['groups']
    if len(scores)!=31 or scores.index.has_duplicates or not scores.index.is_monotonic_increasing:raise ValueError('saved31month calendar differs')
    if not scores.index.isin(frame.index).all():raise ValueError('saved feature rows missing')
    entry_periods=pd.DatetimeIndex(scores.entry_at).tz_localize(None).to_period('M')
    if not entry_periods.equals(pd.period_range('2024-01','2026-07',freq='M')):raise ValueError('monthly calendar changed')
    rows=[];group_rows=[];coefficients=[];chronology=[];max_prediction_error=0.
    for year in [2024,2025,2026]:
        saved=scores.loc[scores.execution_year.eq(year)];x=frame.loc[saved.index];cutoff=saved.signal_at.iloc[0]
        train=frame.loc[frame.source_qualified & frame.market_complete & frame.exit_at.lt(cutoff)]
        models={arm:objects[f'models/{arm}_{year}'] for arm in ['control','candidate']}
        transforms={arm:saved_transform_arithmetic(model,x) for arm,model in models.items()}
        market=groups['market_refit'];candidate=models['candidate'];control=models['control']
        if list(control.features)!=market or list(control.targets)!=heads or list(candidate.targets)!=heads:raise ValueError('frozen feature/target ordering differs')
        if set(candidate.features)!=set(sum([v for k,v in groups.items() if k!='intercept_shift'],[])):raise ValueError('group coverage differs')
        for attribute in ['label_means','label_scales']:
            pd.testing.assert_series_equal(getattr(control,attribute),getattr(candidate,attribute))
        for attribute in ['medians','means','scales']:
            pd.testing.assert_series_equal(getattr(control,attribute),getattr(candidate,attribute).loc[market])
        pd.testing.assert_frame_equal(transforms['control'],transforms['candidate'][market])
        for feature in candidate.features:
            values=train[feature].replace([np.inf,-np.inf],np.nan);test_values=x[feature].replace([np.inf,-np.inf],np.nan)
            chronology.append({'execution_year':year,'feature':feature,'training_rows':len(train),'training_nonmissing':int(values.notna().sum()),'heldout_rows':len(x),'heldout_nonmissing':int(test_values.notna().sum()),'frozen_active':bool(candidate.active[feature]),'training_median':float(candidate.medians[feature]),'training_mean':float(candidate.means[feature]),'training_scale':float(candidate.scales[feature]),'heldout_standardized_mean':float(transforms['candidate'][feature].mean()),'heldout_max_abs_standardized':float(transforms['candidate'][feature].abs().max()),'first_heldout_available':str(test_values.first_valid_index()),'training_cutoff':str(cutoff)})
        reconstructed={}
        for arm,model in models.items():
            coef=np.asarray(model.regressor.coef_,float);intercept=np.asarray(model.regressor.intercept_,float)
            normalized=transforms[arm].to_numpy(float)@coef.T+intercept
            raw=normalized*model.label_scales.to_numpy(float)+model.label_means.to_numpy(float)
            reconstructed[arm]=raw
            for j,head in enumerate(heads):
                for k,feature in enumerate(model.features):coefficients.append({'execution_year':year,'arm':arm,'head':head,'feature':feature,'standardized_coefficient':float(coef[j,k]),'feature_scale':float(model.scales[feature]),'label_training_std':float(model.label_scales[head]),'active':bool(model.active[feature]),'training_rows':len(train)})
                coefficients.append({'execution_year':year,'arm':arm,'head':head,'feature':'intercept','standardized_coefficient':float(intercept[j]),'feature_scale':1.,'label_training_std':float(model.label_scales[head]),'active':True,'training_rows':len(train)})
        fallback=~saved.source_qualified.to_numpy(bool)
        reconstructed['candidate'][fallback]=reconstructed['control'][fallback]
        for j,head in enumerate(heads):
            std=float(control.label_scales[head]);np.testing.assert_allclose(saved['training_std_'+head],std,atol=1e-14)
            for arm in models:
                error=np.abs(reconstructed[arm][:,j]-saved[arm+'_'+head].to_numpy(float));max_prediction_error=max(max_prediction_error,float(error.max()))
                np.testing.assert_allclose(reconstructed[arm][:,j],saved[arm+'_'+head],rtol=0,atol=1e-10)
            e=(saved[head].to_numpy(float)-saved['control_'+head].to_numpy(float))/std
            d=(saved['candidate_'+head].to_numpy(float)-saved['control_'+head].to_numpy(float))/std
            ccoef=pd.Series(candidate.regressor.coef_[j],index=candidate.features)
            bcoef=pd.Series(control.regressor.coef_[j],index=control.features)
            gd={'intercept_shift':np.full(len(saved),float(candidate.regressor.intercept_[j]-control.regressor.intercept_[j])),
                'market_refit':transforms['candidate'][market].to_numpy(float)@(ccoef[market]-bcoef).to_numpy(float)}
            for group,features in groups.items():
                if group in gd:continue
                gd[group]=transforms['candidate'][features].to_numpy(float)@ccoef[features].to_numpy(float)
            for values in gd.values():values[fallback]=0.
            np.testing.assert_allclose(sum(gd.values()),d,rtol=0,atol=1e-10)
            improvement=e**2-(e-d)**2
            np.testing.assert_allclose(improvement,2*e*d-d*d,rtol=0,atol=1e-10)
            for i,date in enumerate(saved.index):
                row={'signal_date':date,'signal_at':saved.signal_at.iloc[i],'entry_at':saved.entry_at.iloc[i],'exit_at':saved.exit_at.iloc[i],'execution_year':year,'head':head,'source_qualified':bool(saved.source_qualified.iloc[i]),'training_std':std,'actual_label':float(saved[head].iloc[i]),'control_prediction':float(saved['control_'+head].iloc[i]),'candidate_prediction':float(saved['candidate_'+head].iloc[i]),'raw_forecast_delta':float(d[i]*std),'normalized_control_residual':float(e[i]),'normalized_forecast_delta':float(d[i]),'alignment':float(2*e[i]*d[i]),'shift_penalty':float(d[i]*d[i]),'control_mse':float(e[i]**2),'candidate_mse':float((e[i]-d[i])**2),'paired_improvement':float(improvement[i]),'raw_rank_available':bool(pd.notna(x.annual_ep_rank.iloc[i])),'raw_rank_prior_count':int(x.rank_prior_count.iloc[i])}
                rows.append(row)
                for group,values in gd.items():group_rows.append({'signal_date':date,'execution_year':year,'head':head,'group':group,'delta_group':float(values[i]),'alignment_contribution':float(2*e[i]*values[i]),'cross_penalty_allocation':float(d[i]*values[i]),'loss_improvement_contribution':float((2*e[i]-d[i])*values[i])})
    daily=pd.DataFrame(rows);group_daily=pd.DataFrame(group_rows)
    pooled=daily.groupby('signal_date').paired_improvement.mean().reindex(scores.index)
    np.testing.assert_allclose(pooled,scores.paired_improvement,rtol=0,atol=1e-10)
    allocated=group_daily.groupby(['signal_date','head']).loss_improvement_contribution.sum()
    np.testing.assert_allclose(allocated,daily.set_index(['signal_date','head']).paired_improvement.sort_index(),rtol=0,atol=1e-10)
    selections=[('all','all',daily.index,False)]
    for year,g in daily.groupby('execution_year'):selections.append(('year',str(year),g.index,year==2026))
    periods=pd.DatetimeIndex(daily.entry_at).tz_localize(None).to_period('Q')
    for period in periods.unique():selections.append(('quarter',str(period),daily.index[periods==period],period==periods[-1]))
    totals=[];components=[]
    for kind,period,index,partial in selections:
        selected=daily.loc[index];dates=selected.signal_date.unique();parts=[];group_parts=[]
        for head,g in selected.groupby('head',sort=False):
            result={'kind':kind,'period':period,'partial_period':partial,'head':head,'months':len(g),**decomposition(g.normalized_control_residual,g.normalized_forecast_delta),'control_mse':float(g.control_mse.mean()),'candidate_mse':float(g.candidate_mse.mean())};parts.append(result);totals.append(result)
            for group in groups:
                gd=group_daily.loc[group_daily.signal_date.isin(dates)&group_daily['head'].eq(head)&group_daily['group'].eq(group)]
                r={'kind':kind,'period':period,'partial_period':partial,'head':head,'group':group,'months':len(gd),'mean_delta_group':float(gd.delta_group.mean()),'alignment_contribution':float(gd.alignment_contribution.mean()),'cross_penalty_allocation':float(gd.cross_penalty_allocation.mean()),'loss_improvement_contribution':float(gd.loss_improvement_contribution.mean())};components.append(r);group_parts.append(r)
        numeric=[k for k,v in parts[0].items() if isinstance(v,(int,float)) and k not in ['months','partial_period']]
        totals.append({'kind':kind,'period':period,'partial_period':partial,'head':'pooled_equal_heads','months':len(dates),**{k:float(np.mean([x[k] for x in parts])) for k in numeric}})
        for group in groups:
            gs=[x for x in group_parts if x['group']==group]
            components.append({'kind':kind,'period':period,'partial_period':partial,'head':'pooled_equal_heads','group':group,'months':len(dates),**{k:float(np.mean([x[k] for x in gs])) for k in ['mean_delta_group','alignment_contribution','cross_penalty_allocation','loss_improvement_contribution']}})
    tables={'score_identity_summary':pd.DataFrame(totals),'feature_group_summary':pd.DataFrame(components),'coefficient_summary':pd.DataFrame(coefficients),'feature_chronology':pd.DataFrame(chronology)}
    overall=tables['score_identity_summary'].query("kind=='all' and head=='pooled_equal_heads'").iloc[0]
    np.testing.assert_allclose(overall.paired_improvement,objects['analysis/summary']['paired_normalized_MSE']['paired_mean_loss_improvement'],rtol=0,atol=1e-10)
    tracking=card/'qlib_recorder';policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/monthly_head_identity','analysis/monthly_group_identity']}
    with R.start(experiment_name=NAME,uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,source_audit=audit,saved_source_objects=objects)
        rec.save_objects(artifact_path='sources',analysis=Path(__file__).read_text())
        rec.save_objects(artifact_path='analysis',monthly_head_identity=daily,monthly_group_identity=group_daily,**tables)
        summary={'experiment':NAME,'decision':'saved_forecast_failure_attribution_only_parent_remains_closed','objective_achieved':False,'new_fits':0,'new_inference':0,'new_accounts':0,'new_fetches':0,'evaluation_months':31,'head_observations':len(daily),'saved_models':6,'maximum_prediction_identity_error':max_prediction_error,'paired_normalized_mse_improvement':float(overall.paired_improvement),'mean_residual_delta_alignment':float(overall.mean_residual_delta_alignment),'covariance_alignment':float(overall.covariance_alignment),'mean_shift_penalty':float(overall.mean_shift_penalty),'delta_variation_penalty':float(overall.delta_variation_penalty),'parent_ci_reused':objects['analysis/summary']['paired_normalized_MSE'],'parent_gates_unchanged':objects['analysis/summary']['gates'],'provider_uri':plan['provider_uri'],'region':'us','as_of':plan['as_of'],'plan_sha256':PLAN_SHA,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/recorder.html','limitations':['Historicalforecastaccountingnotcausalfeatureimportance','Groupcrosspenaltyallocationisadditiveidentity,notremoval/refitcounterfactual','Rankavailability/modelactivationchronologydoesnotidentifybetweenyearcausality','Partial2026sevenmonthsnotfullannualqualification','No newstrategy orindependentforwardevidence; fixedannualinformationroute remainsnotpromoted']}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),monthly_identity_rows=len(daily))
    for name,table in tables.items():table.to_csv(card/(name+'.csv'),index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['source']['card']/'nas_binding.json').read_text());binding.pop('evidence',None);atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'module':'src.analysis.annual_information_attribution','plan':'plan.json','artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

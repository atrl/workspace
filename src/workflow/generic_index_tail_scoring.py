"""Proper scoring of fixed saved risk forecasts, without inference or calibration."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.analysis.seed_disagreement_information import full_calendar_bootstrap
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
NAME='generic_index_tail_scoring_20261015'
SHA='ae4ef830a9d8e9995c495c6a12db0dcb8cf46109d2fb502557c44f81324e29f8'

def run(root):
    import qlib,yaml
    from qlib.workflow import R
    root=Path(root).resolve();card=root/'research'/NAME;plan=json.loads((card/'plan.json').read_text())
    assert hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()==SHA
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    for f,h in plan['source_code_sha256'].items():assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h
    qlib.init(provider_uri=str(root/'data/qlib/frozen_policy_raw_cashbook_20261015'),region='us',kernels=1)
    forecasts,fa=_source(root,plan['forecast_source'],plan['forecast_artifacts']);labels,la=_source(root,plan['label_source'],plan['label_artifacts'])
    sourceplan=labels['inputs/plan'];assert pd.Timestamp(sourceplan['as_of'])<=pd.Timestamp('2026-10-09')
    frame=forecasts['inputs/frame'].reset_index(drop=True);dates=pd.DatetimeIndex(frame.signal_date);assert len(dates)==944 and dates.is_unique and dates.is_monotonic_increasing
    original=labels['inputs/daily_frame'];timing=labels['inputs/timing'];assert dates.isin(original.index).all() and dates.isin(timing.index).all()
    y=original.tail_event_5d.reindex(dates).to_numpy(float);finite=np.isfinite(y)
    if not np.isin(y[finite],[0,1]).all():raise ValueError('label not binary')
    available=pd.to_datetime(timing.label_available_at.reindex(dates))
    if available.dt.tz is None:available=available.dt.tz_localize('America/New_York')
    else:available=available.dt.tz_convert('America/New_York')
    asof=pd.Timestamp('2026-10-09 09:47',tz='America/New_York');known=available.notna()&available.le(asof);mask=finite&known.to_numpy()
    if not mask.any():raise ValueError('no mature observations')
    daily=frame[['signal_date','trade_date','exit_date']].copy();daily['label_available_at']=available.to_numpy();daily['original_label']=y;daily['scored']=mask;daily['label']=np.where(mask,y,np.nan)
    daily['mask_reason']=np.select([~finite,available.isna().to_numpy(),~known.to_numpy()],['original_label_unknown','maturity_unknown','after_parent_asof'],default='scored')
    clipping=[]
    for arm in ['control','candidate']:
        target=forecasts['analysis/targets'][arm]
        if not dates.isin(target.index).all():raise ValueError('missing fixed prediction date')
        p=target.tail_probability.reindex(dates).to_numpy(float)
        if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('invalid probabilities')
        clipped=np.clip(p,1e-7,1-1e-7);daily[arm+'_probability']=p
        daily[arm+'_logloss']=np.where(mask,-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)),np.nan)
        daily[arm+'_brier']=np.where(mask,(p-y)**2,np.nan)
        clipping.append({'arm':arm,'clipped_rows':int((clipped!=p).sum()),'maximum_adjustment':float(np.abs(clipped-p).max())})
    daily['paired_logloss_improvement']=daily.control_logloss-daily.candidate_logloss;daily['paired_brier_improvement']=daily.control_brier-daily.candidate_brier
    rows=[];contributions=[]
    selections=[('all','all',np.ones(len(daily),bool),False)]
    for freq,kind in [('Y','year'),('Q','quarter')]:
        periods=daily.signal_date.dt.to_period(freq)
        for period in periods.unique():selections.append((kind,str(period),periods.eq(period).to_numpy(),period in [periods.iloc[0],periods.iloc[-1]]))
    for kind,period,select,partial in selections:
        g=daily.loc[select];scored=g[g.scored];n=len(scored)
        for arm in ['control','candidate']:
            rows.append({'kind':kind,'period':period,'arm':arm,'calendar_rows':len(g),'scored_rows':n,'unknown_rows':len(g)-n,'partial_boundary_period':partial,'event_count':int(scored.label.sum()),'event_rate':float(scored.label.mean()),'mean_probability_all_rows':float(g[arm+'_probability'].mean()),'mean_probability_scored_rows':float(scored[arm+'_probability'].mean()),'logloss':float(scored[arm+'_logloss'].mean()),'brier':float(scored[arm+'_brier'].mean()),'paired_logloss_improvement':float(scored.paired_logloss_improvement.mean()),'paired_brier_improvement':float(scored.paired_brier_improvement.mean())})
            for event in [0,1]:
                part=scored[scored.label.eq(event)];contributions.append({'kind':kind,'period':period,'arm':arm,'event':event,'rows':len(part),'scored_denominator':n,'mean_probability':float(part[arm+'_probability'].mean()),'conditional_logloss':float(part[arm+'_logloss'].mean()),'logloss_contribution':float(part[arm+'_logloss'].sum()/n) if n else np.nan,'conditional_brier':float(part[arm+'_brier'].mean()),'brier_contribution':float(part[arm+'_brier'].sum()/n) if n else np.nan})
            if n:
                recent=contributions[-2:];np.testing.assert_allclose(sum(x['logloss_contribution'] for x in recent),scored[arm+'_logloss'].mean(),rtol=1e-12,atol=1e-12);np.testing.assert_allclose(sum(x['brier_contribution'] for x in recent),scored[arm+'_brier'].mean(),rtol=1e-12,atol=1e-12)
    b=plan['bootstrap'];ci,indices,replicates=full_calendar_bootstrap(daily.paired_logloss_improvement.to_numpy(),**{k:b[k] for k in ['mean_block','resamples','seed','alpha']})
    tables={'score_summary':pd.DataFrame(rows),'event_contributions':pd.DataFrame(contributions),'probability_clipping':pd.DataFrame(clipping),'bootstrap_summary':pd.DataFrame([ci])}
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=NAME,uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,source_audits={'forecasts':fa,'labels':la},forecast_targets=forecasts['analysis/targets'],frame=frame,original_label_frame=original,original_timing=timing,original_plan=sourceplan)
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),bootstrap=(root/'src/analysis/seed_disagreement_information.py').read_text())
        rec.save_objects(artifact_path='analysis',daily=daily,bootstrap_indices=indices,bootstrap_replicates=replicates,**tables)
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/daily','analysis/bootstrap_indices','analysis/bootstrap_replicates']}
        summary={'experiment':NAME,'plan_sha256':SHA,'decision':'conditional_saved_tail_forecast_improvement_only' if ci['lower_bound']>0 else 'no_reliable_saved_tail_forecast_improvement','primary_logloss_ci':ci,'calendar_rows':len(daily),'scored_rows':int(mask.sum()),'unknown_rows':int((~mask).sum()),'event_count':int(daily.label.sum()),'last_scored_signal':str(daily.loc[mask,'signal_date'].iloc[-1]),'last_used_maturity':str(available[mask].max()),'parent_asof':str(asof),'parent_portfolio_route_remains_closed':True,'new_inference':0,'new_fits':0,'new_policy_accounts':0,'new_fetches':0,'objective_achieved':False,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    for key,value in tables.items():value.to_csv(card/(key+'.csv'),index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['forecast_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'plan':'plan.json','module':'src.workflow.generic_index_tail_scoring','artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

"""Read frozen native accounts, audit new conditional labels, retain all evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.analysis.tiered_action_labels import own_path_labels
from src.workflow.minute_action_value import _source
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def summarize(frame):
    rows=[]
    for slip,full in frame.groupby('slippage_bps'):
        for year in ['all',*sorted(full.trade_date.dt.year.unique())]:
            f=full if year=='all' else full.loc[full.trade_date.dt.year.eq(year)]
            for group in ['all','execute','hold']:
                mask=f.eligible & f.tiered_advantage.notna()
                if group!='all':mask &= f.execute.eq(group=='execute')
                part=f.loc[mask];labels=part.tiered_advantage
                rows.append({'slippage_bps':slip,'year':year,'group':group,'n':len(part),'mean_tiered_bp':1e4*labels.mean(),'mean_flat25_bp':1e4*part.flat25_advantage.mean(),'positive_tiered_fraction':labels.gt(0).mean(),'fee_projection_sign_flip_fraction':labels.gt(0).ne(part.flat25_advantage.gt(0)).mean(),'rank_ic':part.probability.corr(labels,method='spearman') if len(part)>2 else np.nan,'auc':roc_auc_score(labels.gt(0),part.probability) if labels.gt(0).nunique()==2 else np.nan})
    return pd.DataFrame(rows)


def block_intervals(frame):
    rows=[];rng=np.random.default_rng(20261014)
    for slip,full in frame.groupby('slippage_bps'):
        # Keep the calendar, including forced/unknown rows, during resampling.
        n=len(full);starts=rng.integers(0,n,size=(2000,int(np.ceil(n/63))))
        indices=((starts[:,:,None]+np.arange(63))%n).reshape(2000,-1)[:,:n]
        for group in ['execute','hold']:
            mask=full.eligible & full.tiered_advantage.notna() & full.execute.eq(group=='execute')
            values=np.where(mask,full.tiered_advantage,0.)
            count=mask.to_numpy(int)[indices].sum(axis=1);good=count>0
            samples=values[indices].sum(axis=1)[good]/count[good]*1e4
            rows.append({'slippage_bps':slip,'group':group,'lower_mean_bp':np.quantile(samples,.025),'upper_mean_bp':np.quantile(samples,.975),'valid_resamples':int(good.sum()),'block_sessions':63,'resamples':2000})
    return pd.DataFrame(rows)


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed label plan changed')
        return old
    if hashlib.sha256((root/'production/stage1_futu_sim_v1/runtime_release.json').read_bytes()).hexdigest()!=plan['release_sha256']:raise ValueError('release changed')
    source,audit=_source(root,plan['source'],['inputs/frame',*[f'analysis/native/cash_start_2023_current_anchor_{s}' for s in plan['slippage_bps']]])
    controller,control_audit=_source(root,plan['price_source'],['inputs/controller','inputs/plan'])
    action,action_audit=_source(root,plan['action_source'],['inputs/plan','controller/source_recovery_selection'])
    if controller['inputs/plan']['tariff']!=plan['tariff']:raise ValueError('fee identity differs')
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);action_spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec'])
    frame=source['inputs/frame'];frame=frame.loc[frame.signal_date.between(plan['start'],plan['end'])]
    results=[]
    for slip in plan['slippage_bps']:
        results.append(own_path_labels(frame,source[f'analysis/native/cash_start_2023_current_anchor_{slip}'],controller=controller['inputs/controller'],recovery_spec=recovery,action_spec=action_spec,selection=action['controller/source_recovery_selection'],plan=plan,slippage_bps=slip))
        print('labelled',slip,flush=True)
    labels=pd.concat(results,ignore_index=True);table=summarize(labels);intervals=block_intervals(labels)
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='own_path_tiered_labels',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_strategy_accounts=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audit=audit,controller_audit=control_audit,action_audit=action_audit)
        rec.save_objects(artifact_path='sources',labels=(root/'src/analysis/tiered_action_labels.py').read_text(),workflow=Path(__file__).read_text())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'diagnostic_only_no_promotion','objective_achieved':False,'new_fits':0,'new_strategy_accounts':0,'reused_native_accounts':3,'reference_sessions_per_cost':len(frame),'eligible_rows':int(labels.eligible.sum()),'scored_rows':int(labels.tiered_advantage.notna().sum()),'unknown_eligible_rows':int((labels.eligible & labels.tiered_advantage.isna()).sum()),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/'data/qlib/full_path_tiered_allocation_20261013'),'region':'us','as_of':plan['as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/recorder.html','limits':['reference own-state one-session reset proxy, not causal full policy value','flat25bp projection changes fees only on current states, not original training-label reproduction','frozen balanced-class logistic score is not calibrated probability','history reused; no independent forward or deployment qualification'],'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/labels']}}
        rec.save_objects(artifact_path='analysis',labels=labels,comparison=table,intervals=intervals,summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    table.to_csv(card/'conditional_value.csv',index=False);intervals.to_csv(card/'block_intervals.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

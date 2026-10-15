"""Read saved accounts only; explain failed composition without new executions."""
import hashlib,json
from pathlib import Path
import pandas as pd
from src.workflow.minute_action_value import _source
from src.analysis.three_asset_failure_attribution import analyze
from src.analysis.composition_path_diagnosis import sampled_drawdown,cost_divergence
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    from qlib.data import D
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('frozen diagnostic changed')
        return old
    costs=plan['slippage_bps'];source,audit=_source(root,plan['sources']['candidate'],['inputs/frame','inputs/plan',*[f'analysis/native/candidate_{c}' for c in costs]])
    reference,ra=_source(root,plan['sources']['reference'],[f'analysis/native/strong_always_{c}' for c in costs])
    benchmark,ba=_source(root,plan['sources']['benchmark'],[f'analysis/native/soxx_{c}' for c in costs])
    frame=source['inputs/frame'].reset_index(drop=True)
    candidate={c:source[f'analysis/native/candidate_{c}'] for c in costs}
    books={(f,c):v for c in costs for f,v in [('candidate',candidate[c]),('strong_always',reference[f'analysis/native/strong_always_{c}'])]}
    bench={c:benchmark[f'analysis/native/soxx_{c}'] for c in costs}
    calendar=pd.DatetimeIndex(D.calendar(start_time=frame.trade_date.min(),end_time=frame.exit_date.max(),freq='day'))
    pd.testing.assert_index_equal(calendar,pd.DatetimeIndex([*frame.trade_date,frame.exit_date.iloc[-1]]),check_names=False)
    tables=analyze(frame,books,bench,initial_cash=plan['initial_cash'],cash_reference=plan['cash_reference'],rolling_sessions=126,pairs=[['candidate','strong_always']])
    action,action_audit=_source(root,source['inputs/plan']['action_source'],['inputs/plan'])
    recovery_spec=action['inputs/plan']['recovery_spec']
    events,differences=cost_divergence(candidate,frame,tolerances=plan['divergence_tolerances'],recovery_spec=recovery_spec)
    all_marks={};episodes=[];breaches=[]
    for cost,book in candidate.items():
        native_times=pd.DatetimeIndex(book['native_report'].index)
        close_times=pd.Series(native_times,index=native_times.normalize()).groupby(level=0).last()
        marks,episode=sampled_drawdown(frame,book,initial_cash=plan['initial_cash'],close_times=close_times,tolerance=plan['divergence_tolerances']['native_reconstruction_relative'])
        all_marks[str(cost)]=marks;episodes.append(episode.assign(slippage_bps=cost));breaches.append(marks.loc[marks.breach30].assign(slippage_bps=cost))
    tables.update(first_divergences=events,drawdown_episodes=pd.concat(episodes,ignore_index=True),all_breach_marks=pd.concat(breaches,ignore_index=True))
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='saved_composition_failure_diagnosis',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audits={'candidate':audit,'reference':ra,'benchmark':ba,'action':action_audit},recovery_spec=recovery_spec)
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text(),diagnostic=(root/'src/analysis/composition_path_diagnosis.py').read_text(),cross_account=(root/'src/analysis/three_asset_failure_attribution.py').read_text(),accounting=(root/'src/analysis/decision_accounting.py').read_text())
        for name,table in tables.items():
            if isinstance(table,dict):rec.save_objects(artifact_path='analysis/daily/'+name,**{f'{f}_{c}':v for (f,c),v in table.items()})
            else:table.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:table})
        rec.save_objects(artifact_path='analysis/daily/cost_differences',**differences);rec.save_objects(artifact_path='analysis/daily/sampled_marks',**all_marks)
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'diagnostic_only_no_promotion','objective_achieved':False,'new_fits':0,'new_backtests':0,'reused_native_accounts':9,'sessions':len(frame),'source_identities':plan['sources'],'provider_uri':str(root/plan['provider_uri']),'region':'us','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':plan['artifact_policy'],'interpretation':plan['eligibility'],'clock_labels':plan['clock_labels'],'superseded_run':json.loads((card/'superseded_run.json').read_text())['summary']['recorder_id'] if (card/'superseded_run.json').exists() else None}
        rec.save_objects(artifact_path='analysis',summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')))
    binding=json.loads((root/plan['sources']['candidate']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':plan['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary

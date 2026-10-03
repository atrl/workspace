"""Recorder-backed audit of frozen DEVELOPMENT training checkpoints only."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
from dataclasses import asdict
import tempfile

import numpy as np
import pandas as pd

from src.workflow.stable_excess_integration import CANONICAL_ROOT, json_write, sha
from src.analysis.integrated_allocation import _dates

EXPERIMENT='benchmark_anchor_risk_feasibility_20261003'


def _integer(value):
    return isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_))


def expected_identities(plan):
    p=plan['inventory']
    for key in ('variants','training_cutoff_years','seeds','saved_epochs'):
        values=p[key]
        if not values or len(values)!=len(set(values)):
            raise ValueError('blocked: duplicate or empty registered inventory axis')
        if key=='variants':
            if any(not isinstance(v,str) or not v for v in values):raise ValueError('blocked: invalid variant identity')
        elif any(not _integer(v) for v in values):raise ValueError('blocked: inventory identity must be integral')
    identities={(v,int(y),int(seed)) for v in p['variants'] for y in p['training_cutoff_years'] for seed in p['seeds']}
    if len(identities)!=p['runs'] or tuple(p['saved_epochs'])!=tuple(range(121)):
        raise ValueError('blocked: registered checkpoint inventory differs')
    return identities


def checked_parent_load(recorder,parent_card,manifest,artifact):
    """Verify the committed parent artifact bytes before Qlib deserializes them."""
    relative_artifact=Path(artifact)
    if relative_artifact.is_absolute() or '..' in relative_artifact.parts:
        raise ValueError('blocked: artifact path must remain within the pinned Recorder')
    base=(Path(recorder.get_local_dir())/'artifacts').resolve()
    if not (base/relative_artifact).resolve().is_relative_to(base):
        raise ValueError('blocked: artifact resolves outside the pinned Recorder')
    relative=Path(recorder.get_local_dir()).relative_to(Path(recorder.uri))
    wanted=str(Path('qlib_recorder')/relative/'artifacts'/artifact)
    entries=[entry for entry in manifest['entries'] if entry['path']==wanted]
    if len(entries)!=1:raise ValueError('blocked: parent artifact absent or duplicate in manifest')
    path=Path(recorder.get_local_dir())/'artifacts'/artifact
    entry=entries[0]
    if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
        raise ValueError('blocked: parent artifact bytes differ before deserialization')
    value=recorder.load_object(artifact)
    if sha(path)!=entry['sha256']:raise ValueError('blocked: parent artifact changed during read')
    return value,{'artifact':artifact,'sha256':entry['sha256'],'bytes':entry['bytes'],'parent_manifest_path':wanted}


def _same_evidence(left,right):
    if isinstance(left,np.ndarray) or isinstance(right,np.ndarray):
        return isinstance(left,np.ndarray) and isinstance(right,np.ndarray) and np.array_equal(left,right,equal_nan=True)
    if isinstance(left,dict) or isinstance(right,dict):
        return isinstance(left,dict) and isinstance(right,dict) and left.keys()==right.keys() and all(_same_evidence(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)) or isinstance(right,(list,tuple)):
        return type(left) is type(right) and len(left)==len(right) and all(_same_evidence(a,b) for a,b in zip(left,right))
    return type(left) is type(right) and bool(left==right)


def load_registered_inputs(root,plan):
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.benchmark_anchor_utility import build_anchor_dataset,prepare_training_rows,transform_features
    from src.data.integrated_allocation import frame_sha256
    root=Path(root);source=plan['source'];card=root/'research'/source['experiment']
    parent=json.loads((card/'summary.json').read_text());manifest=json.loads((card/'artifact_manifest.json').read_text())
    if sha(card/'artifact_manifest.json')!=source['parent_artifact_manifest_sha256']:
        raise ValueError('blocked: parent evidence manifest identity changed')
    summary_entries=[e for e in manifest['entries'] if e['path'].endswith('/artifacts/card/summary.json')]
    if len(summary_entries)!=1 or sha(card/'summary.json')!=summary_entries[0]['sha256']:
        raise ValueError('blocked: local parent summary differs from its pinned Recorder snapshot')
    if parent['recorder_id']!=source['recorder_id'] or parent['plan_sha256']!=source['parent_plan_sha256']:
        raise ValueError('blocked: parent summary identity differs')
    qlib.init(provider_uri=str(root/parent['provider_uri']),region='us',kernels=1)
    calendar=pd.DatetimeIndex(D.calendar(freq='day'))
    provider_manifest_path=root/parent['provider_uri']/'price_basis_manifest.json'
    provider_manifest=json.loads(provider_manifest_path.read_text())
    if sha(provider_manifest_path)!=parent['source_audit']['provider_manifest_sha256']:
        raise ValueError('blocked: frozen provider manifest identity differs')
    if (calendar.empty or calendar[-1]!=pd.Timestamp(source['provider_as_of'])
            or calendar.strftime('%Y-%m-%d').tolist()!=provider_manifest['reference_calendar']['dates']):
        raise ValueError('blocked: complete frozen provider calendar differs')
    loaded=[]
    with R.uri_context(source['recorder_uri']):
        rec=R.get_recorder(experiment_name=source['experiment'],recorder_id=source['recorder_id'])
        if rec.client.get_run(rec.id).info.status!=source['required_status']:
            raise ValueError('blocked: parent Recorder is not complete')
        # Plan is plain text and is checked without invoking any pickle loader.
        plan_path=Path(rec.get_local_dir())/'artifacts'/source['parent_registration_artifact']
        if sha(plan_path)!=source['parent_plan_sha256']:raise ValueError('blocked: parent registration bytes differ')
        parent_plan=json.loads(plan_path.read_text())
        from src.workflow.benchmark_anchor_utility import training_spec
        expected_spec=asdict(training_spec(parent_plan))
        loaded.append({'artifact':source['parent_registration_artifact'],'sha256':sha(plan_path),'bytes':plan_path.stat().st_size,'serialization':'json'})
        hashes,identity=checked_parent_load(rec,card,manifest,source['source_hash_artifact']);loaded.append(identity)
        if hashes!=parent['source_sha256'] or any(sha(root/path)!=digest for path,digest in hashes.items()):
            raise ValueError('blocked: checkpoint inference/accounting source differs from parent')
        if hashes['src/model/benchmark_anchor_utility.py']!=source['model_implementation_sha256']:
            raise ValueError('blocked: model implementation differs from registration')
        folds,identity=checked_parent_load(rec,card,manifest,source['input_fold_artifact']);loaded.append(identity)
        expected=expected_identities(plan)
        expected_folds={(v,y) for v,y,s in expected}
        if set(folds)!=expected_folds:raise ValueError('blocked: development folds differ from exact registered inventory')
        maturity=[]
        for (variant,year),fold in folds.items():
            rows=fold['rows'];_dates(rows)
            if fold['invalid_training'] or not fold['scaler']['eligible']:
                raise ValueError('blocked: parent fold was not valid for this integrity diagnostic')
            cutoff=pd.Timestamp(f'{year}-01-01')
            expected_dates=calendar[:-2][(calendar[:-2]>=pd.Timestamp(plan['data_contract']['train_start'])) & (calendar[2:]<cutoff)]
            if not rows.index.equals(expected_dates):raise ValueError('blocked: training rows omit a mature native calendar session')
            dataset=build_anchor_dataset(rows,as_of=source['provider_as_of'])
            rebuilt,audit=prepare_training_rows(dataset,year=year,min_rows=plan['data_contract']['minimum_rows'])
            if not rebuilt.equals(rows) or frame_sha256(rows)!=fold['audit']['training_rows_sha256']:
                raise ValueError('blocked: native Dataset training rows differ from pinned parent')
            transformed=transform_features(rows,fold['scaler'],variant=variant)
            if transformed['invalid_forecast_mask'].any() or not np.array_equal(transformed['features'],fold['features']):
                raise ValueError('blocked: parent standardized/masked inputs differ')
            if not np.array_equal(rows[['soxl_return','soxx_return']].to_numpy(float),fold['returns']):
                raise ValueError('blocked: parent daily labels differ')
            if (fold['scaler']['model_year']!=year or fold['scaler']['variant']!=variant
                    or pd.Timestamp(fold['scaler']['cutoff'])!=pd.Timestamp(audit['cutoff'])):
                raise ValueError('blocked: parent scaler/year/variant/cutoff mismatch')
            maturity.append({'variant':variant,'year':year,'rows':len(rows),
                'return_available_at_max':audit['return_available_at_max'],'cutoff':audit['cutoff'],
                'training_rows_sha256':audit['training_rows_sha256'],'scaler_sha256':fold['scaler']['scaler_sha256']})
        audits={}
        for variant,year,seed in sorted(expected):
            artifact=source['checkpoint_artifact_template'].format(variant=variant,year=year,seed=seed)
            audit,identity=checked_parent_load(rec,card,manifest,artifact);loaded.append(identity)
            if (audit['variant']!=variant or audit['year']!=year or audit['seed']!=seed or audit['failure'] is not None):
                raise ValueError('blocked: checkpoint run identity or parent fit failure')
            if 'epochs' not in audit or 'history' not in audit or not _same_evidence(audit['epochs'],audit['history']):
                raise ValueError('blocked: checkpoint history and epoch aliases disagree')
            fold=folds[(variant,year)]
            if audit['spec']!=expected_spec or audit['training_rows']!=len(fold['rows']):
                raise ValueError('blocked: stored optimizer/utility spec or row count differs from parent registration')
            for field,array in [('training_feature_sha256',fold['features']),('training_return_sha256',fold['returns'])]:
                if audit[field]!=hashlib.sha256(np.asarray(array,dtype=np.float64).tobytes()).hexdigest():
                    raise ValueError('blocked: checkpoint training input hash differs')
            audits[(variant,year,seed)]=audit
    return {'parent_summary':parent,'parent_plan':parent_plan,'parent_source_hashes':hashes,
        'source_artifacts':loaded,'folds':folds,'training_audits':audits,'maturity':pd.DataFrame(maturity)}


def _evaluate_job(payload):
    from src.analysis.anchor_checkpoint_feasibility import evaluate_checkpoint_fit
    identity,fold,audit,spec=payload
    variant,year,seed=identity
    result=evaluate_checkpoint_fit(fold['features'],fold['returns'],fold['rows'],audit,
        variant=variant,model_year=year,seed=seed,spec=spec)
    return identity,result


def verify_inventory(results,plan):
    expected=expected_identities(plan)
    if set(results)!=expected:raise ValueError('blocked: completed run identity inventory differs')
    metrics=pd.concat([r['checkpoint_metrics'] for r in results.values()],ignore_index=True)
    statuses=pd.concat([r['checkpoint_status'] for r in results.values()],ignore_index=True)
    idcols=['variant','model_year','seed','epoch']
    null=plan['inventory']['null_epoch_label']
    def canonical(frame,has_cost):
        needed=idcols+['explicit_null']+(['cost_bps'] if has_cost else [])
        if set(needed)-set(frame):raise ValueError('blocked: missing result identity columns')
        values=[]
        for row in frame.itertuples(index=False):
            if (not isinstance(row.variant,str) or not _integer(row.model_year) or not _integer(row.seed)
                    or not (_integer(row.epoch) or (isinstance(row.epoch,str) and row.epoch==null))
                    or not isinstance(row.explicit_null,(bool,np.bool_))
                    or bool(row.explicit_null)!=(row.epoch==null)):
                raise ValueError('blocked: noncanonical checkpoint identity or explicit null marker')
            identity=(row.variant,int(row.model_year),int(row.seed),str(row.epoch))
            if has_cost:
                cost=row.cost_bps
                if (isinstance(cost,(bool,np.bool_)) or not isinstance(cost,(float,int,np.integer,np.floating))
                        or not np.isfinite(cost)):
                    raise ValueError('blocked: invalid cost identity')
                identity=(*identity,float(cost))
            values.append(identity)
        if len(values)!=len(set(values)):raise ValueError('blocked: duplicate canonical result identity')
        return set(values)
    actual=canonical(metrics,True);actual_joint=canonical(statuses,False)
    wanted={(v,y,seed,str(e),float(c)) for v,y,seed in expected for e in [null,*plan['inventory']['saved_epochs']] for c in plan['recompute']['cost_bps']}
    if actual!=wanted or len(metrics)!=len(wanted):raise ValueError('blocked: incomplete or extraneous checkpoint-by-cost records')
    wanted_joint={(v,y,seed,e) for v,y,seed,e,c in wanted}
    if actual_joint!=wanted_joint or len(statuses)!=len(wanted_joint):raise ValueError('blocked: incomplete or extraneous checkpoint joint records')
    return metrics,statuses


def run_experiment(root,card=None,*,workers=3):
    from qlib.workflow import R
    from src.analysis.anchor_checkpoint_feasibility import CheckpointFeasibilitySpec
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts
    root=Path(root).resolve();card=Path(card) if card else root/'research'/EXPERIMENT
    plan=json.loads((card/'selection_plan.json').read_text());plan_hash=sha(card/'selection_plan.json')
    sources=['src/analysis/anchor_checkpoint_feasibility.py','src/workflow/anchor_checkpoint_feasibility.py']
    hashes={p:sha(root/p) for p in sources}
    bundle=load_registered_inputs(root,plan)
    spec=CheckpointFeasibilitySpec(objective_match_tolerance=plan['recompute']['recompute_vs_stored_utility_absolute_tolerance'],
        risk_tolerance_percentage_points=plan['feasibility']['risk_tolerance'],positive_utility_margin=plan['feasibility']['positive_utility_margin'],
        expected_epochs=tuple(plan['inventory']['saved_epochs']))
    uri=CANONICAL_ROOT/'research'/EXPERIMENT/'qlib_recorder';uri.mkdir(parents=True,exist_ok=True)
    R.set_uri(str(uri));results={}
    with R.start(experiment_name=EXPERIMENT):
        recorder=R.get_recorder()
        R.log_params(plan_sha256=plan_hash,parent_recorder_id=plan['source']['recorder_id'],scope='training_diagnostic_only_no_new_policy',market_fits=False,out_of_sample_evaluation=False)
        recorder.log_artifact(str(card/'selection_plan.json'),artifact_path='inputs')
        with tempfile.TemporaryDirectory(prefix='anchor-feasibility-source-') as temporary:
            manifest_file=Path(temporary)/'source_artifact_manifest.json'
            json_write(manifest_file,bundle['source_artifacts'])
            recorder.log_artifact(str(manifest_file),artifact_path='inputs')
        checkpoint_index=pd.DataFrame([{'variant':v,'model_year':y,'seed':seed,'epoch':state['epoch'],
            'tensor_sha256':state['checkpoint']['tensor_sha256'],
            'training_feature_sha256':audit['training_feature_sha256'],'training_return_sha256':audit['training_return_sha256']}
            for (v,y,seed),audit in bundle['training_audits'].items() for state in audit['epochs']])
        R.save_objects(artifact_path='inputs',**{'checkpoint_index.pkl':checkpoint_index,'source_hashes.pkl':hashes,'parent_source_hashes.pkl':bundle['parent_source_hashes'],
            'maturity.pkl':bundle['maturity'],'parent_plan.pkl':bundle['parent_plan']})
        payloads=[(identity,bundle['folds'][identity[:2]],bundle['training_audits'][identity],spec) for identity in sorted(expected_identities(plan))]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(_evaluate_job,payload) for payload in payloads]
            for future in as_completed(futures):
                try:identity,result=future.result()
                except Exception as error:
                    evidence=getattr(error,'evidence',None)
                    if evidence is not None:R.save_objects(artifact_path='integrity_failure',**{'offending_evidence.pkl':evidence})
                    raise
                if identity in results:raise ValueError('blocked: duplicate completed diagnostic chunk')
                results[identity]=result
                recorder.save_objects(artifact_path='completed_chunks/'+('/'.join(map(str,identity))),**{'result.pkl':result})
                print(f'Checkpoint feasibility {identity} completed ({len(results)}/{len(payloads)})',flush=True)
        metrics,statuses=verify_inventory(results,plan)
        summaries=pd.DataFrame([r['fit_summary'] for r in results.values()])
        audits=[r['audit'] for r in results.values()]
        reproduction=pd.concat([r['utility_reproduction'] for r in results.values()],ignore_index=True)
        reports={identity:result['benchmark_risk_reports'] for identity,result in results.items()}
        if sha(card/'selection_plan.json')!=plan_hash or any(sha(root/p)!=h for p,h in {**bundle['parent_source_hashes'],**hashes}.items()):
            raise ValueError('blocked: fixed source or diagnostic plan changed during evaluation')
        trajectory_metrics=metrics.loc[~metrics.explicit_null].reset_index(drop=True)
        trajectory_statuses=statuses.loc[~statuses.explicit_null].reset_index(drop=True)
        null_metrics=metrics.loc[metrics.explicit_null].reset_index(drop=True)
        null_statuses=statuses.loc[statuses.explicit_null].reset_index(drop=True)
        R.save_objects(artifact_path='analysis',**{'checkpoint_cost_metrics.pkl':trajectory_metrics,'checkpoint_joint_feasibility.pkl':trajectory_statuses,
            'null_metrics.pkl':null_metrics,'null_joint_feasibility.pkl':null_statuses,
            'run_summaries.pkl':summaries,'integrity_audits.pkl':audits,'utility_reproduction.pkl':reproduction,'benchmark_risk_reports.pkl':reports})
        recdir=Path(recorder.get_local_dir())
        pd.testing.assert_frame_equal(trajectory_metrics,recorder.load_object('analysis/checkpoint_cost_metrics.pkl'))
        pd.testing.assert_frame_equal(null_metrics,recorder.load_object('analysis/null_metrics.pkl'))
        R.log_metrics(training_runs=len(results),trajectory_cost_rows=len(trajectory_metrics),null_cost_rows=len(null_metrics),
            trajectory_joint_rows=len(trajectory_statuses),new_strategy_selected=0)
    _remove_qlib_code_snapshot_artifacts(recdir)
    summaries.to_csv(card/'run_summary.csv',index=False);bundle['maturity'].to_csv(card/'maturity_summary.csv',index=False)
    nonnull=statuses.loc[~statuses.explicit_null.astype(bool)]
    columns=['risk_feasible','risk_and_nonnegative_J','risk_and_strict_positive_J','risk_and_positive_J_margin']
    counts=nonnull.groupby('variant',sort=True)[columns].sum().reset_index()
    counts['trajectory_checkpoints']=nonnull.groupby('variant').size().reindex(counts.variant).to_numpy()
    counts.to_csv(card/'feasibility_counts.csv',index=False)
    integrity={'all_runs_completed':len(results)==plan['inventory']['runs'],'checkpoint_models':len(nonnull),
        'explicit_nulls':int(statuses.explicit_null.sum()),'cost_rows_including_null':len(metrics),
        'trajectory_cost_rows':len(trajectory_metrics),'null_cost_rows':len(null_metrics),'audits':audits,
        'no_market_fits':True,'no_out_of_sample_evaluation':True,'source_and_plan_unchanged':True}
    R.set_uri(str(uri))
    json_write(card/'integrity_summary.json',{k:v for k,v in integrity.items() if k!='audits'})
    summary={'experiment':EXPERIMENT,'campaign_status':'active_research','objective_achieved':False,
        'decision':'training_feasibility_diagnostic_only_no_new_strategy','new_strategy_selected':False,
        'parent_recorder_id':plan['source']['recorder_id'],'parent_plan_sha256':plan['source']['parent_plan_sha256'],
        'parent_artifact_manifest_sha256':plan['source']['parent_artifact_manifest_sha256'],
        'source_sha256':hashes,'parent_source_sha256':bundle['parent_source_hashes'],'plan_sha256':plan_hash,
        'source_as_of':plan['source']['provider_as_of'],'provider_uri':bundle['parent_summary']['provider_uri'],'region':'us','market':'all',
        'recorder_id':recorder.id,'recorder_uri':str(uri),'recorder_local_dir':str(recdir),
        'recorder_relative_dir':str(recdir.relative_to(CANONICAL_ROOT/'research'/EXPERIMENT)),
        'counts':counts.to_dict('records'),'integrity':{k:v for k,v in integrity.items() if k!='audits'},
        'historical_qualification_pass':False,'new_forward_qualified':False,'original_phase5_selection_unchanged':True,
        'market_model_fits':False,'out_of_sample_evaluation':False,'real_broker_orders':False,'existing_sim_change':False,
        'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml',
            'recorder_only_artifacts':['inputs/*.pkl','completed_chunks/*','analysis/checkpoint_cost_metrics.pkl','analysis/checkpoint_joint_feasibility.pkl','analysis/utility_reproduction.pkl','analysis/benchmark_risk_reports.pkl']},
        'sidecars':['run_summary.csv','maturity_summary.csv','feasibility_counts.csv','integrity_summary.json'],
        'qlib_docs_reference':plan['qlib_docs'],
        'limitations':['Only saved optimization trajectories are inspected; absence of a feasible improving checkpoint cannot prove the whole model class infeasible.',
            'Training feasibility or a positive training utility is not out-of-sample excess or a promotion decision.',
            'Repeated checkpoints/seeds share the same market observations; counts are not statistical sample sizes.']}
    json_write(card/'summary.json',summary)
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':EXPERIMENT,'parent_recorder_id':plan['source']['recorder_id'],
        'provider_uri':summary['provider_uri'],'recorder_uri':str(uri),'region':'us','selection_plan':'selection_plan.json',
        'artifact_policy':summary['artifact_policy'],'mode':'cached_training_feasibility_no_model_fitting'},sort_keys=False))
    lines=['# 训练检查点风险可行性结论','',f'读取63条已保存开发期优化轨迹，核对{len(nonnull)}个训练状态及63个精确SOXX控制。没有重拟合、替代样本外回放或新策略选择。',
        '', '同时满足两档费用下回撤与CVaR门，并且原训练效用J有超过1e-8数值余量的状态数：']
    for row in counts.itertuples():lines.append(f'- {row.variant}：{int(row.risk_and_positive_J_margin)}/{int(row.trajectory_checkpoints)}；仅风险可行数{int(row.risk_feasible)}。')
    lines+=['','具体年度与种子差异见run_summary.csv。最佳/首次状态仅为训练诊断标识，不是已选策略。','',
        '原价格版的历史资格未通过，稳定净超额目标仍在进行。此诊断只能判断现有轨迹是否漏选可行状态，不能证明整个模型类无解。',
        '下一步依据可行状态分布决定是否登记新的模型选择实验或约束训练；所有原费用、尾部、年度稳定性和前向要求不变。',f'Qlib Recorder：{recorder.id}。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
    return summary

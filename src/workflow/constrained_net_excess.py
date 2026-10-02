"""Registered mean-net-excess training under unchanged SOXX tail constraints."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import paired_metrics,annual_paired_metrics,native_allocation_reconciliation
from src.workflow.stable_excess_integration import CANONICAL_ROOT,benchmark_path,choose_family,json_write,sha
from src.workflow.benchmark_anchor_utility import prepare_folds,replay_models,qualification,independent_path_audit

EXPERIMENT='constrained_net_excess_20261003'
SEEDS=(17,41,71)
ASSETS=('soxl','soxx','cash')

def training_spec(plan):
    from src.model.constrained_net_excess import ConstrainedNetExcessSpec
    p = plan['training']
    if tuple(p['seed_ids']) != SEEDS or plan['architecture']['feature_count'] != 27 or plan['architecture']['hidden_units'] != 8:
        raise ValueError('blocked: implementation architecture or seeds differ from plan')
    return ConstrainedNetExcessSpec(updates=p['updates'], learning_rate=p['learning_rate'],
        gradient_clip=p['gradient_clip'], risk_group_length=p['risk_group_length'],
        gradient_chunk_length=p['gradient_scored_chunk'], context_sessions=p['gradient_prior_context'],
        cost_bps=tuple(p['cost_bps']), minimum_rows=p['minimum_rows'], torch_threads=p['torch_threads'],
        downside_coefficient=p['downside_coefficient'], rms_epsilon=p['rms_epsilon'],
        dual_learning_rate=p['dual_learning_rate'],risk_tolerance_percentage_points=plan['constraints']['risk_tolerance_percentage_points'])


def _fit_job(payload):
    from src.model.constrained_net_excess import fit_policy
    from src.model.benchmark_anchor_utility import policy_to_dict
    variant, year, seed, x, returns, spec = payload
    start = time.monotonic()
    model, metadata = fit_policy(x, returns, seed=seed, spec=spec)
    metadata['elapsed_seconds'] = time.monotonic() - start
    return variant, year, seed, policy_to_dict(model), metadata


def _validate_retained(payloads,metadata,plan):
    """Separate feasible economic incumbents from failed accounting-only nulls."""
    from src.model.benchmark_anchor_utility import policy_from_dict
    costs=tuple(plan['training']['cost_bps'])
    def scalar(value):
        return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)) and bool(np.isfinite(value))
    flag=metadata.get('training_feasibility_passed')
    if not isinstance(flag,(bool,np.bool_)) or not bool(flag):
        raise ValueError('blocked: retained training feasibility needs a true boolean')
    F=metadata.get('centered_worst_utility');parts=metadata.get('centered_utility_by_cost')
    if not scalar(F) or F<0 or not isinstance(parts,dict) or set(parts)!=set(costs) or any(not scalar(v) for v in parts.values()) or F!=min(parts.values()):
        raise ValueError('blocked: retained mean-excess evidence is nonfinite, negative or inconsistent')
    policies=[policy_from_dict(payload) for payload in payloads]
    failure=metadata.get('failure')
    if failure is None:
        if metadata.get('qualification_eligible') is not True:
            raise ValueError('blocked: healthy retained fit lacks qualification integrity')
    else:
        if (not isinstance(failure,dict) or failure.get('accounting_anchor_only') is not True
                or metadata.get('qualification_eligible') is not False or F!=0 or any(v!=0 for v in parts.values())
                or any(np.count_nonzero(payload['state_dict']['theta']) for payload in payloads)):
            raise ValueError('blocked: numerical failure did not preserve an explicit exact accounting null')
    risks=metadata.get('selected_risks_by_cost')
    if not isinstance(risks,dict) or set(risks)!=set(costs):raise ValueError('blocked: missing retained raw risk evidence')
    for risk in risks.values():
        keys=('max_drawdown_pct','cvar5_pct','benchmark_max_drawdown_pct','benchmark_cvar5_pct')
        if any(not scalar(risk.get(k)) for k in keys):raise ValueError('blocked: nonfinite retained risk values')
        gaps=np.asarray(risk.get('raw_percentage_point_gaps'),dtype=float)
        if gaps.shape!=(2,) or not np.isfinite(gaps).all():raise ValueError('blocked: invalid retained risk gaps')
        implied=np.array([risk['benchmark_max_drawdown_pct']-risk['max_drawdown_pct'],risk['benchmark_cvar5_pct']-risk['cvar5_pct']])
        if not np.allclose(gaps,implied,atol=1e-12,rtol=0) or (gaps>plan['constraints']['risk_tolerance_percentage_points']).any():
            raise ValueError('blocked: retained raw tail constraints fail')
        if failure is None:
            normalizers=np.asarray(risk.get('normalizers'),dtype=float)
            expected_reference=-np.array([risk['benchmark_max_drawdown_pct'],risk['benchmark_cvar5_pct']])/100
            if (risk.get('normalizers_valid_for_optimization') is not True or normalizers.shape!=(2,)
                    or not np.isfinite(normalizers).all() or (normalizers<=0).any()
                    or not np.allclose(normalizers,expected_reference,atol=1e-14,rtol=0)):
                raise ValueError('blocked: healthy retained fit has invalid risk normalizers')
    return policies


def fit_folds(folds, plan, recorder, *, workers=3):
    from src.model.benchmark_anchor_utility import (BenchmarkAnchorPolicy, policy_to_dict,
        policy_from_dict)
    from src.model.constrained_net_excess import evaluate_ensemble
    spec = training_spec(plan)
    fitted = {}; diagnostics = []; ensemble_audits = []
    jobs = []
    for (name, year), fold in folds.items():
        fitted.setdefault(name, {})[year] = {'scaler':fold['scaler'], 'seed_models':{},
            'variant':name,'model_year':year,'scaler_sha256':fold['scaler']['scaler_sha256'],
            'training_feature_sha256':hashlib.sha256(np.asarray(fold['features'],dtype=np.float64).tobytes()).hexdigest(),
            'training_return_sha256':hashlib.sha256(np.asarray(fold['returns'],dtype=np.float64).tobytes()).hexdigest(),
            'invalid_training':fold['invalid_training'], 'invalid_training_dates':fold['invalid_training_dates']}
        for seed in SEEDS:
            if fold['invalid_training']:
                null = BenchmarkAnchorPolicy(seed=seed,anchor=True)
                payload=policy_to_dict(null)
                fitted[name][year]['seed_models'][seed] = payload
                source_failure={'variant':name,'year':year,'seed':seed,'failure':'invalid_available_training_volatility',
                    'qualification_eligible':False,'accounting_anchor_only':True,'selected_epoch':'anchor',
                    'centered_worst_utility':0.,'centered_utility_by_cost':{c:0. for c in plan['training']['cost_bps']},
                    'null_selected':True,'epochs':[],'invalid_training_dates':fold['invalid_training_dates']}
                diagnostics.append(source_failure)
                recorder.save_objects(artifact_path=f'fit_checkpoints/{name}/{year}/{seed}',
                    **{'model.pkl':payload,'training_audit.pkl':source_failure})
            else:
                jobs.append((name,year,seed,fold['features'],fold['returns'],spec))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(_fit_job, job) for job in jobs]
        for future in as_completed(pending):
            name,year,seed,model,metadata = future.result()
            for field in ('training_feature_sha256','training_return_sha256'):
                if metadata[field] != fitted[name][year][field]:
                    raise ValueError('blocked: fitted worker consumed different mature inputs')
            _validate_retained([model],metadata,plan)
            fitted[name][year]['seed_models'][seed] = model
            diagnostic = {'variant':name,'year':year,'seed':seed,**metadata}
            diagnostics.append(diagnostic)
            recorder.save_objects(artifact_path=f'fit_checkpoints/{name}/{year}/{seed}',
                **{'model.pkl':model,'training_audit.pkl':diagnostic})
            print(f'Constrained excess {name} year={year} seed={seed} epoch={metadata["selected_epoch"]} F={metadata["centered_worst_utility"]:.8g} seconds={metadata["elapsed_seconds"]:.1f}',flush=True)
    for (name,year),fold in folds.items():
        record = fitted[name][year]
        if set(record['seed_models']) != set(SEEDS):
            raise ValueError('blocked: missing registered seed fit')
        models = [policy_from_dict(record['seed_models'][seed]) for seed in SEEDS]
        accepted,audit = evaluate_ensemble(models,fold['features'],fold['returns'],spec=spec)
        _validate_retained([policy_to_dict(model) for model in accepted],audit,plan)
        record['ensemble_models'] = [policy_to_dict(model) for model in accepted]
        record['ensemble_audit'] = {k:v for k,v in audit.items() if k not in ('training_paths','rejected_ensemble_training_paths')}
        record['ensemble_audit']['recorder_artifact']=f'fit_checkpoints/{name}/{year}/ensemble/training_audit.pkl'
        record['numerical_failure'] = bool(audit.get('failure') is not None or any(
            d.get('failure') is not None and d.get('failure')!='invalid_available_training_volatility'
            for d in diagnostics if d['variant']==name and d['year']==year))
        ensemble_audits.append({'variant':name,'year':year,**audit})
        recorder.save_objects(artifact_path=f'fit_checkpoints/{name}/{year}/ensemble',
            **{'model_bundle.pkl':record,'training_audit.pkl':audit})
    return fitted,diagnostics,ensemble_audits


def _training_index(diagnostics):
    fields=('variant','year','seed','selected_epoch','failure','qualification_eligible',
        'training_feasibility_passed','centered_worst_utility','centered_utility_by_cost','legacy_J',
        'null_selected','training_feature_sha256','training_return_sha256','optimizer_diagnostics')
    return [{**{k:a.get(k) for k in fields},
             'recorder_artifact':f"fit_checkpoints/{a['variant']}/{a['year']}/{a['seed']}/training_audit.pkl"}
            for a in diagnostics]


def _ensemble_index(audits):
    omitted=('training_paths','rejected_ensemble_training_paths')
    return [{**{k:v for k,v in a.items() if k not in omitted},
             'recorder_artifact':f"fit_checkpoints/{a['variant']}/{a['year']}/ensemble/training_audit.pkl"}
            for a in audits]


def _source_hashes(root):
    sources=['src/objective/benchmark_tail_constraints.py','src/model/constrained_net_excess.py',
        'src/data/constrained_net_excess.py','src/workflow/constrained_net_excess.py','src/workflow/anchor_checkpoint_feasibility.py','src/data/benchmark_anchor_utility.py','src/model/benchmark_anchor_utility.py','src/workflow/benchmark_anchor_utility.py',
        'src/model/integrated_allocation.py','src/data/conditional_variance.py','src/workflow/conditional_variance.py',
        'src/data/integrated_allocation.py','src/data/convex_allocation.py','src/analysis/integrated_allocation.py',
        'src/analysis/convex_qlib.py','src/backtest/qlib_continuous_position.py','src/workflow/stable_excess_integration.py',
        'src/backtest/position_accounting.py','src/workflow/stage1_economic_action.py','src/data/qlib_native.py','src/data/price_basis.py',
        'src/factor/soxx_pit_tail_factors.py','src/workflow/pit_constituent_auxiliary_training_allocation.py']
    return {path:sha(root/path) for path in sources}


def _finalize_native(recorder_id,uri,build_reports):
    """Native backtest may reinitialize Qlib; run it outside an active Recorder.

    Explicitly resume the original identity/URI to persist the reports. A native
    failure is also propagated inside that resumed context, retaining FAILED.
    """
    from qlib.workflow import R
    try:
        reports=build_reports()
    except Exception:
        with R.start(experiment_name=EXPERIMENT,recorder_id=recorder_id,uri=uri,resume=True):
            raise
    with R.start(experiment_name=EXPERIMENT,recorder_id=recorder_id,uri=uri,resume=True):
        if R.get_recorder().id!=recorder_id:
            raise ValueError('blocked: native reports resumed a different Recorder')
        R.save_objects(artifact_path='portfolio_analysis',**{'native_reconciliation.pkl':reports})
    return reports


def run_experiment(root,card=None,*,workers=3):
    import qlib
    from qlib.workflow import R
    from src.data.constrained_net_excess import load_constrained_inputs
    from src.workflow.conditional_variance import hac_attribution
    from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES
    from src.analysis.convex_qlib import native_open_reconciliation as precise_native
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts
    root=Path(root).resolve();card=Path(card) if card else root/'research'/EXPERIMENT
    plan=json.loads((card/'selection_plan.json').read_text());plan_hash=sha(card/'selection_plan.json')
    hashes=_source_hashes(root);source=plan['source'];costs=plan['evaluation']['cost_bps']
    bundle=load_constrained_inputs(root,plan)
    raw=bundle['raw'];prices=bundle['prices'];forecasts=bundle['forecasts']
    forecast_audit=bundle['forecast_audit'];forecast_models=bundle['forecast_models'];dataset=bundle['dataset']
    uri=CANONICAL_ROOT/'research'/EXPERIMENT/'qlib_recorder';uri.mkdir(parents=True,exist_ok=True)
    R.set_uri(str(uri))
    with R.start(experiment_name=EXPERIMENT):
        recorder=R.get_recorder()
        R.log_params(plan_sha256=plan_hash,source_recorder_id=source['recorder_id'],
            parent_input_recorder_id=plan['parent_input_source']['recorder_id'],objective='mean_net_log_excess_with_benchmark_tail_constraints',campaign_status='active_research')
        recorder.log_artifact(str(card/'selection_plan.json'),artifact_path='inputs')
        R.save_objects(artifact_path='inputs',**{'source_hashes.pkl':hashes,'source_audit.pkl':bundle['source_audit'],
            'primary_frame.pkl':raw,'execution_prices.pkl':prices})
        R.save_objects(artifact_path='inputs',**{'source_details.pkl':bundle['source_details']})
        valid=raw[['soxl_return','soxx_return']].notna().all(axis=1)
        development=forecasts.loc[valid & raw.index.year.isin(plan['training']['development_signal_years'])]
        outer=forecasts.loc[valid & raw.index.year.isin(plan['training']['diagnostic_signal_years'])]
        variants=[v['name'] for v in plan['variants']]
        folds=prepare_folds(dataset,variants,plan['training']['development_signal_years'],plan)
        for identity,fold in folds.items():
            previous=bundle['parent_dev_folds'][identity]
            if (fold['scaler']!=previous['scaler'] or not np.array_equal(fold['features'],previous['features'])
                    or not np.array_equal(fold['returns'],previous['returns'])):
                raise ValueError('blocked: constrained study changed prepared development inputs')
        fitted,diagnostics,ensemble_audits=fit_folds(folds,plan,recorder,workers=workers)
        R.save_objects(artifact_path='inputs',**{'forecasts.pkl':forecasts,'forecast_fit_audit.pkl':forecast_audit,
            'development_fold_inputs.pkl':folds})
        R.save_objects(artifact_path='models',**{'forecast_models.pkl':forecast_models,'development.pkl':fitted})
        benchmarks={c:benchmark_path(development,cost_bps=c) for c in costs}
        evidence={};paths={};replays=[];seed_rows=[];path_audits=[]
        for name in variants:
            candidate={}
            for cost in costs:
                path=replay_models(development,fitted[name],variant=name,cost_bps=cost)
                paths[(name,cost)]=path;candidate[cost]=path
                replays.append(path.assign(variant=name,seed='ensemble',segment='development'))
                path_audits.append({'variant':name,'seed':'ensemble',**independent_path_audit(development,path,cost_bps=cost)})
                for seed in SEEDS:
                    trial=replay_models(development,fitted[name],variant=name,cost_bps=cost,seed=seed)
                    seed_rows.append({'variant':name,'seed':seed,'cost_bps':cost,**paired_metrics(trial,benchmarks[cost])})
                    replays.append(trial.assign(variant=name,seed=str(seed),segment='development'))
                    path_audits.append({'variant':name,'seed':seed,**independent_path_audit(development,trial,cost_bps=cost)})
            evidence[name]=qualification(candidate,benchmarks,pd.DataFrame(seed_rows).query('variant==@name'),plan=plan)
        selected,ranking=choose_family(evidence)
        seal={'plan_sha256':plan_hash,'selected':selected,'selection_uses_diagnostic_signal_years':False,
            'development_last_exit':str(development.exit_date.max().date()),'campaign_status':'active_research'}
        json_write(card/'selection_seal.json',seal);recorder.log_artifact(str(card/'selection_seal.json'),artifact_path='inputs')
        print(f'Constrained excess development sealed: {selected}',flush=True)
        chosen=selected['variant']
        outer_folds=prepare_folds(dataset,[chosen],plan['training']['diagnostic_signal_years'],plan)
        outer_fitted,outer_diagnostics,outer_ensemble=fit_folds(outer_folds,plan,recorder,workers=workers)
        diagnostics+=outer_diagnostics;ensemble_audits+=outer_ensemble
        comparisons=[r['metrics'].assign(variant=name,segment='development') for name,r in evidence.items()]
        annuals=[r['annual'].assign(variant=name,segment='development') for name,r in evidence.items()]
        controls=[];attribution=[];windows=[]
        for name in variants:
            for cost in costs:
                path=paths[(name,cost)]
                if name!='constant': controls.append({'variant':name,'reference':'constant','cost_bps':cost,
                    **paired_metrics(path,paths[('constant',cost)])})
                attribution.append({'variant':name,'cost_bps':cost,'signal_year':'all',**hac_attribution(path,benchmarks[cost])})
                for year in plan['training']['development_signal_years']:
                    mask=path.signal_date.dt.year.eq(year)
                    attribution.append({'variant':name,'cost_bps':cost,'signal_year':str(year),
                        **hac_attribution(path.loc[mask].reset_index(drop=True),benchmarks[cost].loc[mask].reset_index(drop=True))})
        outer_frames={}
        for mode in ('annual_refit','frozen2022_parameters'):
            frame=outer.copy();frozen=None
            if mode=='frozen2022_parameters':
                frozen=2023
                if chosen=='price_vol':
                    prediction=forecast_models[2023].predict(frame.loc[:,list(COMMON_PRIMARY_FEATURES)])
                    frame.loc[:,['cash_advantage_prediction','future_vol_prediction']]=prediction
                    frame.loc[:,'forecast_available']=1.
                    frame['forecast_model_available_at']=pd.Timestamp('2023-01-01',tz='America/New_York').tz_convert('UTC')
            outer_frames[mode]=frame
            for cost in costs:
                reference=benchmark_path(frame,cost_bps=cost)
                path=replay_models(frame,outer_fitted[chosen],variant=chosen,cost_bps=cost,frozen_year=frozen)
                name=chosen+'_'+mode
                replays.append(path.assign(variant=name,seed='ensemble',segment='reused_diagnostic'))
                comparisons.append(pd.DataFrame([{'variant':name,'segment':'reused_diagnostic','cost_bps':cost,**paired_metrics(path,reference)}]))
                annuals.append(annual_paired_metrics(path,reference).assign(variant=name,segment='reused_diagnostic',cost_bps=cost))
                path_audits.append({'variant':name,'seed':'ensemble',**independent_path_audit(frame,path,cost_bps=cost)})
                mask=path.trade_date.ge('2026-08-03')
                if mask.any():
                    windows.append({'variant':name,'cost_bps':cost,'window':'trade_date>=2026-08-03; through2026-09-25open','start_position':'carried from diagnostic2023start',
                        **paired_metrics(path.loc[mask],reference.loc[mask])})
                    fresh=replay_models(frame.loc[mask.to_numpy()],outer_fitted[chosen],variant=chosen,cost_bps=cost,frozen_year=frozen)
                    fresh_ref=benchmark_path(frame.loc[mask.to_numpy()],cost_bps=cost)
                    windows.append({'variant':name,'cost_bps':cost,'window':'trade_date>=2026-08-03; through2026-09-25open','start_position':'CASH at first window trade',
                        **paired_metrics(fresh,fresh_ref)})
        for cost in costs:
            replays.append(benchmarks[cost].assign(variant='SOXX_buy_hold',seed='benchmark',segment='development'))
            replays.append(benchmark_path(outer,cost_bps=cost).assign(variant='SOXX_buy_hold',seed='benchmark',segment='reused_diagnostic'))
        comparison=pd.concat(comparisons,ignore_index=True);annual=pd.concat(annuals,ignore_index=True)
        gates=pd.concat([r['gates'].assign(variant=n) for n,r in evidence.items()],ignore_index=True)
        bootstrap=pd.concat([r['bootstrap'].assign(variant=n) for n,r in evidence.items()],ignore_index=True)
        if sha(card/'selection_plan.json')!=plan_hash or _source_hashes(root)!=hashes:
            raise ValueError('blocked: registered implementation changed during experiment')
        R.save_objects(artifact_path='models',**{'selected_annual_refits.pkl':outer_fitted})
        R.save_objects(artifact_path='inputs',**{'diagnostic_fold_inputs.pkl':outer_folds,'diagnostic_frames.pkl':outer_frames})
        R.save_objects(artifact_path='analysis',**{'portfolio_replays.pkl':pd.concat(replays,ignore_index=True),
            'comparison.pkl':comparison,'annual.pkl':annual,'gates.pkl':gates,'bootstrap.pkl':bootstrap,
            'seed_metrics.pkl':pd.DataFrame(seed_rows),'training_diagnostics.pkl':_training_index(diagnostics),
            'ensemble_audits.pkl':_ensemble_index(ensemble_audits),'independent_path_audits.pkl':path_audits,
            'controls.pkl':pd.DataFrame(controls),'attribution.pkl':pd.DataFrame(attribution),'window_metrics.pkl':pd.DataFrame(windows)})
        R.log_metrics(historical_qualification_pass=float(selected['historical_qualification_pass']),registered_variants=3,seeds_per_variant=3)
        recdir=Path(recorder.get_local_dir())
        pd.testing.assert_frame_equal(comparison,recorder.load_object('analysis/comparison.pkl'))
    native=_finalize_native(recorder.id,str(uri),lambda: {c:native_allocation_reconciliation(
        development.reset_index(drop=True),paths[(chosen,c)],provider_uri=root/source['provider_uri'],
        cost_bps=c,native_reconciler=precise_native) for c in costs})
    _remove_qlib_code_snapshot_artifacts(recdir)
    training_rows=[]
    for audit in diagnostics:
        item={k:audit.get(k) for k in ('variant','year','seed','failure','selected_epoch','centered_worst_utility','legacy_J','training_feasibility_passed','null_selected','elapsed_seconds')}
        epochs=audit.get('epochs',[])
        item['feasible_epochs']=sum(bool(e['feasible']) for e in epochs)
        item['positive_F_feasible_epochs']=sum(bool(e['feasible']) and e['F']>0 for e in epochs)
        item['F_above_numerical_margin_feasible_epochs']=sum(bool(e['feasible']) and e['F']>1e-8 for e in epochs)
        item['last_trajectory_feasible']=bool(epochs[-1]['feasible']) if epochs else None
        item['updates_completed']=audit.get('optimizer_diagnostics',{}).get('updates_completed')
        item['selected_maximum_risk_excess_percentage_points']=max([0.]+[float(gap) for risk in audit.get('selected_risks_by_cost',{}).values() for gap in risk['raw_percentage_point_gaps']]) if audit.get('selected_risks_by_cost') else None
        training_rows.append(item)
    tables={'model_selection.csv':ranking,'backtest_benchmark_comparison.csv':comparison,'annual_metrics.csv':annual,
        'development_gates.csv':gates,'bootstrap_analysis.csv':bootstrap,'seed_metrics.csv':pd.DataFrame(seed_rows),
        'training_summary.csv':pd.DataFrame(training_rows),'forecast_training_audit.csv':forecast_audit,
        'control_comparison.csv':pd.DataFrame(controls),'beta_alpha_attribution.csv':pd.DataFrame(attribution),
        'window_metrics.csv':pd.DataFrame(windows)}
    for filename,table in tables.items():table.to_csv(card/filename,index=False)
    summary={'experiment':EXPERIMENT,'campaign_status':'active_research','objective_achieved':False,
        'decision':'historical_candidate_requires_source_sensitivity_and_new_forward' if selected['historical_qualification_pass'] else 'continue_research_no_historical_qualification',
        'selected':selected,'historical_qualification_pass':bool(selected['historical_qualification_pass']),
        'new_forward_qualified':False,'source_sensitivity_completed':False,'source':source,'source_audit':bundle['source_audit'],
        'seed_scope':'Optimization-start stability only; constant masked inputs and fixed theta/bias make the three seed decision paths structurally identical. Seeds are never independent market observations.',
        'source_sha256':hashes,'plan_sha256':plan_hash,'provider_uri':source['provider_uri'],'region':'us','market':'all',
        'training_objective':'F=min_c63mean_net_log_excess; legacy_J_report_only',
        'training_constraints':'same_benchmark_anchored_MaxDD_and_exact_CVaR5_at_both_costs',
        'normalized_cost_state_scope':'Post-fee normalized targets are cost-blind; fee funding changes NAV/trade amounts, not necessarily normalized holdings.',
        'recorder_id':recorder.id,'recorder_uri':str(uri),'recorder_local_dir':str(recdir),
        'recorder_relative_dir':str(recdir.relative_to(CANONICAL_ROOT/'research'/EXPERIMENT)),
        'selection_uses_diagnostic_signal_years':False,'development_last_exit':str(development.exit_date.max().date()),
        'native_qlib_audit':{str(c):v['audit'] for c,v in native.items()},'independent_accounting_audit':path_audits,
        'production_change':False,'real_broker_orders':False,'sidecars':list(tables),
        'training_audit_storage':'Complete epoch states/selected paths/optimizer state only in per-fit Recorder artifacts; aggregate analysis files are small indices with exact artifact pointers.',
        'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml',
            'recorder_only_artifacts':['inputs/*.pkl','models/*.pkl','fit_checkpoints/*','analysis/portfolio_replays.pkl','analysis/training_diagnostics.pkl','analysis/ensemble_audits.pkl','portfolio_analysis/*.pkl']},
        'qlib_docs_reference':plan['qlib_docs'],
        'limitations':['Historical years have been reviewed; no fresh forward evidence or guaranteed future excess.',
            'Global expert coefficient budget restricts state-dependent expressivity.',
            '21-session state-gradient truncation and nonsmooth fee gradients are optimizer approximations; exact incumbent forward audits do not certify optimality.',
            'Retained training models satisfy historical mean-excess and tail constraints; this does not guarantee future wealth/risk or optimizer convergence.',
            'Source basis retains published-versus-server adjustment differences and16secondary observations.',
            'Historical open prices and continuous adjusted shares are not evidence of actual spread, impact or partial fills.']}
    json_write(card/'summary.json',summary);json_write(card/'selected_model.json',{'selected':selected,'research_only':True,'new_forward_qualified':False})
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':EXPERIMENT,'provider_uri':source['provider_uri'],'region':'us',
        'recorder_uri':str(uri),'model_selection':'selection_plan.json','artifact_policy':summary['artifact_policy'],
        'segments':{k:plan['training'][k] for k in ('development_signal_years','diagnostic_signal_years')}},sort_keys=False))
    lines=['# 净超额与基准尾部约束训练结论','',f'完成3组信息对照、每组3个种子；完整持仓路径计入所有费用，直接优化最差成本下平均净对数超额，并保留同时满足四个基准风险门的状态；原季度短缺效用仅作诊断。历史资格通过={selected["historical_qualification_pass"]}；稳定净超额目标仍未完成。','',
        '开发期2016–2022，10bp：']
    for row in comparison[comparison.segment.eq('development') & comparison.cost_bps.eq(10)].itertuples():
        lines.append(f'- {row.variant}：收益{row.return_pct:.2f}%，相对SOXX终值{row.relative_wealth:.4f}，MaxDD{row.max_drawdown_pct:.2f}%，CVaR5{row.cvar5_pct:.2f}%。')
    lines+=['','各年差异见annual_metrics.csv，8月3日实际交易日起的持续持仓与独立现金起点分别见window_metrics.csv。三种子不是三条独立市场历史。',
        f'决定：{summary["decision"]}。2023以后为已查看的诊断期，不能作为新留出样本；原SIM未切换。',
        '风险：有限模型容量、截断梯度、零换手边界导数、行情口径和真实成交差异。训练期净超额与尾部可行性不保证样本外超额或风险；对偶迭代没有全局收敛证明。',
        '下一步：按失败门与训练诊断继续定位；若历史门通过，先完成来源敏感性，再冻结进入新的前向证据期。',f'Qlib Recorder：{recorder.id}。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
    return summary

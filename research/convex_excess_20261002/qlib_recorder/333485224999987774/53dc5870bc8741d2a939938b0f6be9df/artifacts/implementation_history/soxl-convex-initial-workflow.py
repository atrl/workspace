"""Registered benchmark-feasible convex allocation, Qlib evidence and gates."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.workflow.stable_excess_integration import (
    CANONICAL_ROOT, benchmark_path, choose_family, json_write, load_frozen_bundle, sha,
)
from src.workflow.benchmark_risk_excess import COVARIANCE_FIELDS, expanding_covariance, attach_covariance

EXPERIMENT = 'convex_excess_20261002'
SOURCE = 'stable_excess_integration_20261002'
ASSETS = ('soxl', 'soxx', 'cash')
REPEATS = (1, 2, 3)


def fit_predict_repeat(dataset, frame, years):
    """Fresh annual fits; no model cache or post-cutoff fit information."""
    from src.data.convex_allocation import fit_annual_estimator, predict_annual
    models, predictions, audits = {}, {}, []
    for year in years:
        fitted = fit_annual_estimator(dataset, year=year, ridge_alpha=10., minimum_training_rows=252, hac_lags=5)
        market = frame.loc[frame.index.year == year]
        if market.empty:
            raise ValueError('blocked: missing registered prediction year')
        forecast = predict_annual(fitted, market)
        models[year] = fitted
        predictions[year] = forecast
        audits.append({'year': year, **fitted['audit'], 'prediction_audit': forecast['audit']})
    return models, assemble_predictions(frame, predictions), audits


def assemble_predictions(frame, yearly):
    result = pd.DataFrame(index=frame.index)
    for year, forecast in yearly.items():
        mask = frame.index.year == year
        count = int(mask.sum())
        if forecast['prior'].shape != (count, 2) or forecast['ridge'].shape != (count, 2) or forecast['Vmean'].shape != (count, 2, 2):
            raise ValueError('blocked: annual predictions do not cover the declared decisions')
        for j, asset in enumerate(ASSETS[:2]):
            result.loc[mask, f'prior_{asset}'] = forecast['prior'][:, j]
            result.loc[mask, f'ridge_{asset}'] = forecast['ridge'][:, j]
        for i, j in ((0, 0), (0, 1), (1, 1)):
            result.loc[mask, f'mean_cov_{i}{j}'] = forecast['Vmean'][:, i, j]
        result.loc[mask, 'model_year'] = year
    if not np.isfinite(result.to_numpy()).all() or len(result.columns) != 8:
        raise ValueError('blocked: incomplete prediction coverage')
    return result


def prediction_inputs(predictions, variant):
    mu = predictions[[f"{variant['mean']}_{a}" for a in ASSETS[:2]]].to_numpy(dtype=float)
    error = np.empty((len(predictions), 2, 2))
    error[:, 0, 0] = predictions.mean_cov_00
    error[:, 0, 1] = error[:, 1, 0] = predictions.mean_cov_01
    error[:, 1, 1] = predictions.mean_cov_11
    return mu, error


def verify_repetitions(reference, current, tolerance=1e-8):
    """All values and dates must match; repeated runs are not new observations."""
    if isinstance(reference, pd.DataFrame):
        if not reference.index.equals(current.index) or not reference.columns.equals(current.columns):
            raise ValueError('blocked: deterministic reproduction axes changed')
        maximum = 0.
        for column in reference:
            if pd.api.types.is_numeric_dtype(reference[column]) and not pd.api.types.is_bool_dtype(reference[column]):
                a, b = reference[column].to_numpy(dtype=float), current[column].to_numpy(dtype=float)
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('blocked: nonfinite repeated evidence')
                maximum = max(maximum, float(np.abs(a-b).max(initial=0.)))
            elif not reference[column].equals(current[column]):
                raise ValueError(f'blocked: deterministic reproduction differs in {column}')
    else:
        a, b = np.asarray(reference), np.asarray(current)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('blocked: invalid repeated numerical evidence')
        maximum = float(np.abs(a-b).max(initial=0.))
    if maximum > tolerance:
        raise ValueError(f'blocked: deterministic reproduction differs by {maximum}')
    return {'all_values_match': True, 'maximum_absolute_difference': maximum, 'tolerance': tolerance,
            'independent_statistical_evidence': False}


def replay_predictions(frame, expected_returns, *, cost_bps, mean_error_covariance=None, robust_radius=0.):
    from src.model.convex_allocation import ConvexAllocationSolver
    from src.backtest.position_accounting import settle_target_session
    from src.analysis.integrated_allocation import validate_replay
    mu = np.asarray(expected_returns, dtype=float)
    if mu.shape != (len(frame), 2) or not np.isfinite(mu).all() or (mu <= -1).any():
        raise ValueError('blocked: invalid H5 forecasts; no clipping or dropped observations')
    errors = np.zeros((len(frame), 2, 2)) if mean_error_covariance is None else np.asarray(mean_error_covariance, dtype=float)
    if errors.shape != (len(frame), 2, 2) or not np.isfinite(errors).all():
        raise ValueError('blocked: invalid H5 mean-error covariance')
    solver = ConvexAllocationSolver(cost_bps=cost_bps, tolerance=1e-8)
    current = np.array([0., 0., 1.]); equity = 1.; rows, audits = [], []
    for number, (_, market) in enumerate(frame.iterrows()):
        sigma = market.loc[list(COVARIANCE_FIELDS)].to_numpy(dtype=float)
        solved = solver.solve(current, mu[number], sigma, mean_error_covariance_H=errors[number], robust_radius=robust_radius)
        target = solved['target_net_weights']
        settled = settle_target_session(current, open_target_weights=solved['execution_target_weights'],
            asset_returns=[market.soxl_return, market.soxx_return, 0.], cost_bps=cost_bps)
        if not np.allclose(settled['open_net_weights'], target, atol=1e-10, rtol=0):
            raise ValueError('blocked: optimizer and actual cash-fee settlement disagree')
        w = settled['open_net_weights'][:2]
        variance = w[0]**2*sigma[0]+2*w[0]*w[1]*sigma[1]+w[1]**2*sigma[2]
        risk_ratio = variance/sigma[2] if sigma[2] > 0 else 0.
        if (sigma[2] > 0 and risk_ratio > 1+1e-8) or (sigma[2] == 0 and variance > 1e-20):
            raise ValueError('blocked: independently settled weights violate benchmark variance')
        equity *= settled['net_equity']
        rows.append({'signal_date':market.signal_date,'trade_date':market.trade_date,'exit_date':market.exit_date,
            'strategy_return':settled['net_return'],'turnover':settled['turnover'],'fees':settled['fees'],
            'cost_bps':cost_bps,'equity':equity,'execute_open_target':True,'execute_open_trade':settled['turnover']>1e-12,
            'objective':solved['objective'],'null_gap':solved['null_gap'],'risk_ratio':risk_ratio,
            'optimality_verified':solved['optimality_verified'],'solution_source':solved['solution_source'],
            **{f'{a}_weight':float(target[i]) for i,a in enumerate(ASSETS)},
            **{f'pretrade_{a}_weight':float(current[i]) for i,a in enumerate(ASSETS)},
            **{f'ending_{a}_weight':float(settled['ending_weights'][i]) for i,a in enumerate(ASSETS)}})
        audits.append({'signal_date':market.signal_date, 'solver':solved['solver_audit'], 'nulls':solved['null_objectives'],
            'status':solved['status'],'optimality_verified':solved['optimality_verified']})
        current = settled['ending_weights']
    path = validate_replay(pd.DataFrame(rows))
    audit = {'all_optimality_verified':bool(path.optimality_verified.all()),'fallback_sessions':int((~path.optimality_verified).sum()),
        'minimum_null_gap':float(path.null_gap.min()),'maximum_predicted_variance_ratio':float(path.risk_ratio.max()),
        'mean_soxl_pct':float(100*path.soxl_weight.mean()),'mean_soxx_pct':float(100*path.soxx_weight.mean()),
        'mean_cash_pct':float(100*path.cash_weight.mean()),'mean_nominal_beta':float((3*path.soxl_weight+path.soxx_weight).mean()),
        'solver_audits':audits}
    return path, audit


def deterministic_gates(paths, benchmarks, repeated_metrics, *, solver_verified, repeats_verified, family_size=10):
    from src.analysis.integrated_allocation import StabilityGateSpec, stable_excess_gates
    spec = replace(StabilityGateSpec(), family_size=family_size, expected_seeds=REPEATS, minimum_positive_seeds=3)
    result = stable_excess_gates(paths, benchmarks, spec=spec, seed_table=repeated_metrics.rename(columns={'repeat_id':'seed'}))
    gates = result['gates'].rename(columns={'minimum_positive_seeds_at_both_costs':'all_repeats_positive_at_both_costs',
        'positive_seeds_at_this_cost':'positive_repeats_at_this_cost','seeds_positive_at_both_costs':'repeats_positive_at_both_costs',
        'seed_evidence_complete':'repeat_evidence_complete'})
    gates['all_three_fresh_runs_match'] = bool(repeats_verified)
    gates['all_convex_solutions_verified'] = bool(solver_verified)
    gates['measurable_log_excess_above_numerical_tolerance'] = result['metrics'].log_excess_sum.to_numpy() > 1e-8
    gates['all_pass'] &= gates.all_three_fresh_runs_match & gates.all_convex_solutions_verified & gates.measurable_log_excess_above_numerical_tolerance
    result['gates'] = gates
    result['historical_qualification_pass'] = bool(gates.all_pass.all())
    result['reproduction_evidence'] = 'three fresh deterministic fits and solves; no random initialization and no extra statistical observations'
    return result


def run_experiment(*, root):
    import qlib
    from qlib.workflow import R
    from src.data.convex_allocation import build_horizon_frame, build_horizon_dataset, predict_annual
    from src.analysis.integrated_allocation import paired_metrics, annual_paired_metrics, native_allocation_reconciliation
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts
    root = Path(root).resolve(); card = root/'research'/EXPERIMENT
    plan = json.loads((card/'selection_plan.json').read_text()); plan_hash = sha(card/'selection_plan.json')
    source_plan = json.loads((root/'research'/SOURCE/'selection_plan.json').read_text())
    source_summary = json.loads((root/'research'/SOURCE/'summary.json').read_text())
    if source_summary['recorder_id'] != plan['source_recorder_id']:
        raise ValueError('blocked: convex study frozen input identity changed')
    bundle = load_frozen_bundle(root, source_plan)
    horizon, data_audit = build_horizon_frame(bundle['primary_frame'], bundle['execution_prices'])
    dataset = build_horizon_dataset(horizon, as_of=plan['source_as_of'])
    covariance = expanding_covariance(bundle['execution_prices'])
    frame = attach_covariance(horizon, covariance)
    valid = frame[['soxl_return','soxx_return']].notna().all(axis=1)
    development = frame.loc[valid & frame.index.year.isin(plan['training']['development_signal_years'])]
    outer = frame.loc[valid & frame.index.year.isin(plan['training']['diagnostic_signal_years'])]
    source_files = ('src/data/convex_allocation.py','src/model/convex_allocation.py','src/workflow/convex_excess.py',
        'src/data/integrated_allocation.py','src/analysis/integrated_allocation.py','src/workflow/benchmark_risk_excess.py',
        'src/workflow/stable_excess_integration.py','src/backtest/position_accounting.py','src/workflow/stage1_economic_action.py')
    source_hashes = {p:sha(root/p) for p in source_files}
    costs = plan['evaluation']['cost_bps']; variants = plan['variants']
    benchmarks = {cost:benchmark_path(development,cost_bps=cost) for cost in costs}
    models, forecasts, paths, solver_details = {}, {}, {}, {}
    fit_audits, repetition_audits, metrics, risk_rows, replays = [], [], [], [], []
    for repeat in REPEATS:
        fitted, predicted, audits = fit_predict_repeat(dataset, development, plan['training']['development_signal_years'])
        models[repeat], forecasts[repeat] = fitted, predicted
        fit_audits.extend({'repeat_id':repeat,'segment':'development',**a} for a in audits)
        if repeat > 1:
            repetition_audits.append({'repeat_id':repeat,'kind':'development_predictions',**verify_repetitions(forecasts[1],predicted)})
        for variant in variants:
            name = variant['name']; mu, error = prediction_inputs(predicted,variant)
            for cost in costs:
                path, audit = replay_predictions(development,mu,cost_bps=cost,mean_error_covariance=error,robust_radius=variant['robust_radius'])
                key = (repeat,name,cost); paths[key] = path
                solver_details[key] = audit['solver_audits']
                risk_rows.append({'repeat_id':repeat,'variant':name,'cost_bps':cost,'segment':'development',**{k:v for k,v in audit.items() if k!='solver_audits'}})
                if repeat > 1:
                    repetition_audits.append({'repeat_id':repeat,'kind':'development_path','variant':name,'cost_bps':cost,
                        **verify_repetitions(paths[(1,name,cost)],path)})
                metrics.append({'repeat_id':repeat,'variant':name,'cost_bps':cost,**paired_metrics(path,benchmarks[cost])})
                replays.append(path.assign(variant=name,repeat_id=repeat,segment='development'))
                print(f'Convex development repeat={repeat} {name} cost={cost} verified={audit["all_optimality_verified"]}',flush=True)
    evidence = {}
    for variant in variants:
        name=variant['name']; relevant=pd.DataFrame(risk_rows).query('variant == @name')
        evidence[name] = deterministic_gates({cost:paths[(1,name,cost)] for cost in costs},benchmarks,
            pd.DataFrame(metrics).query('variant == @name'),solver_verified=bool(relevant.all_optimality_verified.all()),
            repeats_verified=True,family_size=plan['evaluation']['family_size_adjustment'])
    selected, ranking = choose_family(evidence)
    json_write(card/'selection_seal.json',{'plan_sha256':plan_hash,'selected':selected,'development_last_exit':str(development.exit_date.max().date()),
        'selection_uses_diagnostic_signal_years':False,'campaign_status':'active_research'})
    print(f'Convex development sealed: {selected}',flush=True)
    chosen = next(v for v in variants if v['name']==selected['variant'])
    comparisons = [r['metrics'].assign(variant=name,segment='development') for name,r in evidence.items()]
    annuals = [r['annual'].assign(variant=name,segment='development') for name,r in evidence.items()]
    outer_models, outer_forecasts, outer_paths = {}, {}, {}
    for repeat in REPEATS:
        fitted,predicted,audits = fit_predict_repeat(dataset,outer,plan['training']['diagnostic_signal_years'])
        outer_models[repeat] = fitted; outer_forecasts[(repeat,'annual_refit')] = predicted
        frozen = predict_annual(fitted[2023],outer)
        # Assemble by year without refitting or changing the frozen2023 scaler.
        frozen_table = pd.DataFrame(index=outer.index)
        for j,a in enumerate(ASSETS[:2]):
            for mean in ('prior','ridge'):frozen_table[f'{mean}_{a}']=frozen[mean][:,j]
        for i,j in ((0,0),(0,1),(1,1)):frozen_table[f'mean_cov_{i}{j}']=frozen['Vmean'][:,i,j]
        frozen_table['model_year']=2023
        outer_forecasts[(repeat,'frozen2022')] = frozen_table
        fit_audits.extend({'repeat_id':repeat,'segment':'reused_diagnostic',**a} for a in audits)
        for mode in ('annual_refit','frozen2022'):
            prediction = outer_forecasts[(repeat,mode)]
            if repeat>1:repetition_audits.append({'repeat_id':repeat,'kind':mode+'_predictions',**verify_repetitions(outer_forecasts[(1,mode)],prediction)})
            mu,error=prediction_inputs(prediction,chosen)
            for cost in costs:
                path,audit=replay_predictions(outer,mu,cost_bps=cost,mean_error_covariance=error,robust_radius=chosen['robust_radius'])
                key=(repeat,mode,cost);outer_paths[key]=path;solver_details[('outer',*key)]=audit['solver_audits']
                risk_rows.append({'repeat_id':repeat,'variant':chosen['name']+'_'+mode,'cost_bps':cost,'segment':'reused_diagnostic',**{k:v for k,v in audit.items() if k!='solver_audits'}})
                if repeat>1:repetition_audits.append({'repeat_id':repeat,'kind':mode+'_path','cost_bps':cost,**verify_repetitions(outer_paths[(1,mode,cost)],path)})
                replays.append(path.assign(variant=chosen['name']+'_'+mode,repeat_id=repeat,segment='reused_diagnostic'))
                if repeat==1:
                    reference=benchmark_path(outer,cost_bps=cost)
                    comparisons.append(pd.DataFrame([{'variant':chosen['name']+'_'+mode,'segment':'reused_diagnostic','cost_bps':cost,**paired_metrics(path,reference)}]))
                    annuals.append(annual_paired_metrics(path,reference).assign(variant=chosen['name']+'_'+mode,segment='reused_diagnostic',cost_bps=cost))
                print(f'Convex diagnostic repeat={repeat} {mode} cost={cost}',flush=True)
    for cost in costs:
        replays.append(benchmarks[cost].assign(variant='SOXX_buy_hold',repeat_id=0,segment='development'))
        replays.append(benchmark_path(outer,cost_bps=cost).assign(variant='SOXX_buy_hold',repeat_id=0,segment='reused_diagnostic'))
    native = {cost:native_allocation_reconciliation(development.reset_index(drop=True),paths[(1,chosen['name'],cost)],
        provider_uri=root/'data/qlib'/SOURCE,cost_bps=cost) for cost in costs}
    controls = pd.DataFrame([{'variant':v['name'],'cost_bps':cost,'reference':'prior_h5',
        **paired_metrics(paths[(1,v['name'],cost)],paths[(1,'prior_h5',cost)])} for v in variants for cost in costs])
    comparison=pd.concat(comparisons,ignore_index=True);annual=pd.concat(annuals,ignore_index=True)
    gates=pd.concat([r['gates'].assign(variant=name) for name,r in evidence.items()],ignore_index=True)
    bootstrap=pd.concat([r['bootstrap'].assign(variant=name) for name,r in evidence.items()],ignore_index=True)
    if sha(card/'selection_plan.json')!=plan_hash or any(sha(root/p)!=v for p,v in source_hashes.items()):
        raise ValueError('blocked: registered convex plan or implementation changed during run')
    uri=CANONICAL_ROOT/'research'/EXPERIMENT/'qlib_recorder';uri.mkdir(parents=True,exist_ok=True)
    qlib.init(provider_uri=str(root/'data/qlib'/SOURCE),region='us',kernels=1);R.set_uri(str(uri))
    with R.start(experiment_name=EXPERIMENT):
        recorder=R.get_recorder();R.log_params(plan_sha256=plan_hash,source_recorder_id=plan['source_recorder_id'],campaign_status='active_research')
        R.log_metrics(historical_qualification_pass=float(selected['historical_qualification_pass']),registered_variants=3,actual_fresh_repetitions=3)
        R.save_objects(artifact_path='inputs',**{'horizon_frame.pkl':frame,'data_audit.pkl':data_audit,'source_data_audit.pkl':bundle['audit'],
            'covariance.pkl':covariance,'fit_audits.pkl':fit_audits})
        R.save_objects(artifact_path='models',**{'development.pkl':models,'diagnostic.pkl':outer_models})
        R.save_objects(artifact_path='predictions',**{'development.pkl':forecasts,'diagnostic.pkl':outer_forecasts})
        R.save_objects(artifact_path='analysis',**{'portfolio_replays.pkl':pd.concat(replays,ignore_index=True),'comparison.pkl':comparison,
            'annual.pkl':annual,'gates.pkl':gates,'bootstrap.pkl':bootstrap,'repeated_metrics.pkl':pd.DataFrame(metrics),
            'reproduction.pkl':pd.DataFrame(repetition_audits),'risk_solver_audit.pkl':pd.DataFrame(risk_rows),
            'daily_solver_audits.pkl':solver_details,'prior_control.pkl':controls})
        R.save_objects(artifact_path='portfolio_analysis',**{'native_reconciliation.pkl':native})
        recorder.log_artifact(str(card/'selection_plan.json'),artifact_path='inputs');recorder.log_artifact(str(card/'selection_seal.json'),artifact_path='inputs')
        recdir=Path(recorder.get_local_dir());pd.testing.assert_frame_equal(comparison,recorder.load_object('analysis/comparison.pkl'))
    _remove_qlib_code_snapshot_artifacts(recdir)
    tables={'model_selection.csv':ranking,'backtest_benchmark_comparison.csv':comparison,'annual_metrics.csv':annual,
        'development_gates.csv':gates,'bootstrap_analysis.csv':bootstrap,'repeated_metrics.csv':pd.DataFrame(metrics),
        'reproduction_audit.csv':pd.DataFrame(repetition_audits),'risk_solver_audit.csv':pd.DataFrame(risk_rows),'prior_control_comparison.csv':controls}
    for filename,table in tables.items():table.to_csv(card/filename,index=False)
    summary={'experiment':EXPERIMENT,'campaign_status':'active_research','objective_achieved':False,
        'decision':'historical_candidate_requires_new_forward' if selected['historical_qualification_pass'] else 'continue_research_no_historical_qualification',
        'selected':selected,'plan_sha256':plan_hash,'source_as_of':plan['source_as_of'],'source_recorder_id':plan['source_recorder_id'],
        'provider_uri':f'data/qlib/{SOURCE}','region':'us','market':'all','data_audit':data_audit,'source_sha256':source_hashes,
        'historical_qualification_pass':bool(selected['historical_qualification_pass']),'new_forward_qualified':False,
        'selection_uses_diagnostic_signal_years':False,'development_last_exit':str(development.exit_date.max().date()),
        'deterministic_reproduction':{'fresh_fits_and_solves':3,'all_match':True,'tolerance':1e-8,'independent_statistical_observations':False},
        'recorder_id':recorder.id,'recorder_uri':str(uri),'recorder_local_dir':str(recdir),
        'recorder_relative_dir':str(recdir.relative_to(CANONICAL_ROOT/'research'/EXPERIMENT)),
        'native_qlib_audit':{str(cost):value['audit'] for cost,value in native.items()},'production_change':False,'real_broker_orders':False,
        'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml',
            'recorder_only_artifacts':['inputs/*.pkl','models/*.pkl','predictions/*.pkl','analysis/portfolio_replays.pkl','analysis/daily_solver_audits.pkl','portfolio_analysis/*.pkl']},
        'limitations':['Previously inspected history is not fresh OOS; family10 correction excludes older adaptive search.',
            'H5 utility assumes no future trading but actual policy reoptimizes daily; it is a rolling planning approximation.',
            'Estimated variance bounds do not ensure realized tail-risk bounds; actual drawdown/CVaR gates remain mandatory.',
            'One SEM robust radius is not a confidence guarantee; excludes ridge bias and regime change.',
            'A failed experimental batch does not complete the stable-excess objective.']}
    json_write(card/'summary.json',summary)
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':EXPERIMENT,'provider_uri':summary['provider_uri'],'region':'us',
        'recorder_uri':str(uri),'model_selection':'selection_plan.json','artifact_policy':summary['artifact_policy'],
        'segments':plan['training']},sort_keys=False))
    lines=['# 可达基准的凸配置实验结论','',f"完成3个确定性版本，每版3次真实重拟合与求解，历史资格通过={selected['historical_qualification_pass']}；稳定超额目标仍在进行。",'',
        '年度仅用严格成熟的五日开盘收益训练均值模型；每日在现金费用和因果SOXX方差预算下求解，纯SOXX作为显式可行基准。三次复现不增加统计样本。','', '开发期2016–2022信号年，10bp：']
    for row in comparison[comparison.segment.eq('development')&comparison.cost_bps.eq(10)].itertuples():
        lines.append(f'- {row.variant}：收益{row.return_pct:.2f}%，相对SOXX终值{row.relative_wealth:.4f}，MaxDD{row.max_drawdown_pct:.2f}%，CVaR5{row.cvar5_pct:.2f}%。')
    lines+=['','年度差异见 annual_metrics.csv；2023以后已看过的诊断窗口在开发选择封存后运行，未用于排名。费用压力、留一年、区块bootstrap和尾部风险门均保留。','',
        f"推广决定：{summary['decision']}；没有修改现有SIM或发出委托。下一步检查预测误差、实际换手和相对无信号对照的增量，登记下一项独立机制检验。",
        '风险：历史反复检视、均值估计不稳定、H5计划与每日重配差异、扩展风险估计的滞后。数学求解正确不等于真实收益可靠。',f'Qlib recorder：{recorder.id}。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
    return summary

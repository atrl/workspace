"""Qlib experiment for an attainable benchmark and complete-path net utility.

Annual fitting uses only strictly matured Dataset rows. Saved models are
training-utility incumbents, never certificates of global optimality or alpha.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import hashlib
import time

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import (
    StabilityGateSpec, stable_excess_gates, paired_metrics, annual_paired_metrics,
    native_allocation_reconciliation, validate_replay, _dates,
)
from src.workflow.stable_excess_integration import CANONICAL_ROOT, benchmark_path, choose_family, json_write, sha

EXPERIMENT = 'benchmark_anchor_utility_20261002'
ASSETS = ('soxl', 'soxx', 'cash')
SEEDS = (17, 41, 71)


def training_spec(plan):
    from src.model.benchmark_anchor_utility import AnchorUtilitySpec
    p = plan['training']
    if tuple(p['seed_ids']) != SEEDS or plan['architecture']['feature_count'] != 27 or plan['architecture']['hidden_units'] != 8:
        raise ValueError('blocked: implementation architecture or seeds differ from plan')
    return AnchorUtilitySpec(updates=p['updates'], learning_rate=p['learning_rate'],
        gradient_clip=p['gradient_clip'], risk_group_length=p['risk_group_length'],
        gradient_chunk_length=p['gradient_scored_chunk'], context_sessions=p['gradient_prior_context'],
        cost_bps=tuple(p['cost_bps']), minimum_rows=p['minimum_rows'], torch_threads=p['torch_threads'],
        downside_coefficient=p['downside_coefficient'], rms_epsilon=p['rms_epsilon'])


def _fit_job(payload):
    from src.model.benchmark_anchor_utility import fit_policy, policy_to_dict
    variant, year, seed, x, returns, spec = payload
    start = time.monotonic()
    model, metadata = fit_policy(x, returns, seed=seed, spec=spec)
    metadata['elapsed_seconds'] = time.monotonic() - start
    return variant, year, seed, policy_to_dict(model), metadata


def prepare_folds(dataset, variants, years, plan):
    from src.data.benchmark_anchor_utility import prepare_training_rows, fit_feature_scaler, transform_features
    folds = {}
    for year in years:
        rows, maturity = prepare_training_rows(dataset, year=year, min_rows=plan['training']['minimum_rows'])
        _dates(rows)
        for name in variants:
            scaler = fit_feature_scaler(rows, maturity, variant=name)
            transformed = transform_features(rows, scaler, variant=name)
            invalid = np.asarray(transformed['invalid_forecast_mask'], dtype=bool)
            folds[(name, year)] = {'rows': rows, 'features': transformed['features'],
                'returns': rows[['soxl_return','soxx_return']].to_numpy(float), 'scaler': scaler,
                'invalid_training': bool(name == 'price_vol' and (invalid.any() or not scaler['eligible'])),
                'invalid_training_dates': scaler['invalid_training_forecast_dates'],
                'audit': maturity, 'transform_audit': transformed['audit']}
    return folds


def fit_folds(folds, plan, recorder, *, workers=3):
    from src.model.benchmark_anchor_utility import (BenchmarkAnchorPolicy, policy_to_dict,
        policy_from_dict, evaluate_ensemble)
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
                fitted[name][year]['seed_models'][seed] = policy_to_dict(null)
                diagnostics.append({'variant':name,'year':year,'seed':seed,'failure':'invalid_available_training_volatility',
                    'selected_epoch':'anchor','invalid_training_dates':fold['invalid_training_dates']})
            else:
                jobs.append((name,year,seed,fold['features'],fold['returns'],spec))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(_fit_job, job) for job in jobs]
        for future in as_completed(pending):
            name,year,seed,model,metadata = future.result()
            for field in ('training_feature_sha256','training_return_sha256'):
                if metadata[field] != fitted[name][year][field]:
                    raise ValueError('blocked: fitted worker consumed different mature inputs')
            fitted[name][year]['seed_models'][seed] = model
            diagnostic = {'variant':name,'year':year,'seed':seed,**metadata}
            diagnostics.append(diagnostic)
            recorder.save_objects(artifact_path=f'fit_checkpoints/{name}/{year}/{seed}',
                **{'model.pkl':model,'training_audit.pkl':diagnostic})
            print(f'Anchor utility {name} year={year} seed={seed} epoch={metadata["selected_epoch"]} J={metadata["centered_worst_utility"]:.8g} seconds={metadata["elapsed_seconds"]:.1f}',flush=True)
    for (name,year),fold in folds.items():
        record = fitted[name][year]
        if set(record['seed_models']) != set(SEEDS):
            raise ValueError('blocked: missing registered seed fit')
        models = [policy_from_dict(record['seed_models'][seed]) for seed in SEEDS]
        accepted,audit = evaluate_ensemble(models,fold['features'],fold['returns'],spec=spec)
        record['ensemble_models'] = [policy_to_dict(model) for model in accepted]
        record['ensemble_audit'] = {k:v for k,v in audit.items() if k!='training_paths'}
        record['numerical_failure'] = any(d.get('failure') is not None for d in diagnostics
            if d['variant']==name and d['year']==year)
        ensemble_audits.append({'variant':name,'year':year,**audit})
        recorder.save_objects(artifact_path=f'fit_checkpoints/{name}/{year}/ensemble',
            **{'model_bundle.pkl':record,'training_audit.pkl':audit})
    return fitted,diagnostics,ensemble_audits


def replay_models(frame, annual, *, variant, cost_bps, seed=None, frozen_year=None):
    """Keep holdings across annual refits; invalid observed vol is causal only."""
    from src.data.benchmark_anchor_utility import transform_features
    from src.model.benchmark_anchor_utility import policy_from_dict, numpy_alphas, rollout_alphas
    market = _dates(frame)
    dates = pd.DatetimeIndex(market.signal_date)
    current = np.array([0.,0.,1.]); parts=[]
    for year in sorted(dates.year.unique()):
        use_year = int(frozen_year or year)
        if use_year not in annual: raise ValueError('blocked: missing annual allocation model')
        record=annual[use_year]; positions=np.flatnonzero(dates.year==year)
        rows=frame.iloc[positions]
        expected_cutoff=pd.Timestamp(f'{use_year}-01-01',tz='America/New_York')
        if (record.get('variant')!=variant or record.get('model_year')!=use_year
                or record['scaler'].get('model_year')!=use_year
                or record.get('scaler_sha256')!=record['scaler'].get('scaler_sha256')
                or pd.Timestamp(record['scaler']['cutoff'])!=expected_cutoff
                or expected_cutoff > pd.Timestamp(rows.signal_date.min()).tz_localize('America/New_York')):
            raise ValueError('blocked: allocation bundle/scaler is mismatched or unavailable at signal')
        transformed=transform_features(rows,record['scaler'],variant=variant)
        stored=record['ensemble_models'] if seed is None else [record['seed_models'][seed]]
        models=[policy_from_dict(value) for value in stored]
        alpha=numpy_alphas(models,transformed['features'])
        bad=np.asarray(transformed['invalid_forecast_mask'],bool)
        if variant!='price_vol' and bad.any(): raise ValueError('blocked: unused forecast affected control')
        alpha[bad]=0. # Exact SOXX on that date; never retroactive whole-year substitution.
        result=rollout_alphas(alpha,rows[['soxl_return','soxx_return']].to_numpy(float),cost_bps=cost_bps,initial_weights=current)
        part=market.iloc[positions].loc[:,['signal_date','trade_date','exit_date']].copy()
        part['strategy_return']=result['net_equity']-1
        part['fees']=result['fees'];part['turnover']=result['turnover'];part['cost_bps']=cost_bps
        part['execute_open_target']=True;part['execute_open_trade']=result['turnover']>1e-12
        part['observed_forecast_fallback']=bad;part['invalid_training_fallback']=bool(record['invalid_training'])
        part['numerical_failure']=bool(record['numerical_failure']);part['model_year']=use_year
        part['ensemble_anchor_selected']=bool(record['ensemble_audit']['anchor_selected']) if seed is None else False
        for number,asset in enumerate(ASSETS):
            part[f'{asset}_weight']=result['open_net_weights'][:,number]
            part[f'pretrade_{asset}_weight']=result['pretrade_weights'][:,number]
            part[f'ending_{asset}_weight']=result['ending_weights'][:,number]
        current=result['ending_weights'][-1].copy();parts.append(part)
    path=pd.concat(parts,ignore_index=True);path['equity']=np.cumprod(1+path.strategy_return.to_numpy())
    return validate_replay(path)


def qualification(paths,benchmarks,seed_table,*,plan):
    spec=StabilityGateSpec(family_size=plan['evaluation']['family_size_adjustment'])
    result=stable_excess_gates(paths,benchmarks,spec=spec,seed_table=seed_table)
    for cost,path in paths.items():
        row=result['gates'].cost_bps.eq(cost)
        flags=path[['observed_forecast_fallback','invalid_training_fallback','numerical_failure']]
        if flags.isna().any().any() or any(flags[c].dtype!=bool for c in flags):
            raise ValueError('blocked: qualification failure flags must be complete booleans')
        valid=not flags.any().any()
        result['gates'].loc[row,'zero_invalid_source_or_numerical_events']=bool(valid)
        result['gates'].loc[row,'above_numerical_null']=paired_metrics(path,benchmarks[cost])['log_excess_sum']>1e-8
    for column in ('zero_invalid_source_or_numerical_events','above_numerical_null'):
        result['gates'][column]=result['gates'][column].astype(bool)
    checks=[c for c in result['gates'] if result['gates'][c].dtype==bool and c not in ('all_pass','seed_evidence_complete')]
    result['gates']['all_pass']=result['gates'][checks].all(axis=1)
    result['historical_qualification_pass']=bool(result['gates'].all_pass.all())
    return result


def independent_path_audit(frame,path,*,cost_bps):
    """Independent legacy-free fee funding and settlement of every saved action."""
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target
    market=_dates(frame);path=validate_replay(path)
    if not market[['signal_date','trade_date','exit_date']].equals(path[['signal_date','trade_date','exit_date']]):
        raise ValueError('blocked: audit dates differ')
    held=np.array([0.,0.,1.]);errors={'pretrade':0.,'ending':0.,'return':0.,'fee':0.,'turnover':0.}
    for i,row in path.iterrows():
        target=row[[f'{asset}_weight' for asset in ASSETS]].to_numpy(float)
        pre=row[[f'pretrade_{asset}_weight' for asset in ASSETS]].to_numpy(float)
        execution=fee_affordable_target(target,pretrade_weights=held,cost_bps=cost_bps)
        settled=settle_target_session(held,open_target_weights=execution,
            asset_returns=[market.soxl_return.iloc[i],market.soxx_return.iloc[i],0.],cost_bps=cost_bps)
        errors['pretrade']=max(errors['pretrade'],float(np.max(abs(pre-held))))
        errors['ending']=max(errors['ending'],float(np.max(abs(row[[f'ending_{asset}_weight' for asset in ASSETS]].to_numpy(float)-settled['ending_weights']))))
        for key,field in [('return','net_return'),('fee','fees'),('turnover','turnover')]:
            observed=row.strategy_return if key=='return' else row[field]
            errors[key]=max(errors[key],abs(float(observed)-settled[field]))
        held=settled['ending_weights']
    if max(errors.values())>1e-10: raise ValueError(f'blocked: independent settlement differs {errors}')
    return {'sessions':len(path),'cost_bps':cost_bps,'maximum_absolute_errors':errors,'passed':True}


def _source_hashes(root):
    sources=['src/data/benchmark_anchor_utility.py','src/model/benchmark_anchor_utility.py','src/workflow/benchmark_anchor_utility.py',
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
    from src.data.conditional_variance import load_qualified_primary
    from src.data.benchmark_anchor_utility import build_anchor_dataset
    from src.workflow.conditional_variance import fit_oof_forecasts,hac_attribution
    from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES
    from src.analysis.convex_qlib import native_open_reconciliation as precise_native
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts
    root=Path(root).resolve();card=Path(card) if card else root/'research'/EXPERIMENT
    plan=json.loads((card/'selection_plan.json').read_text());plan_hash=sha(card/'selection_plan.json')
    hashes=_source_hashes(root);source=plan['source'];costs=plan['evaluation']['cost_bps']
    bundle=load_qualified_primary(root,provider_uri=source['provider_uri'],parent_summary=f"research/{source['experiment']}/summary.json",
        expected_recorder_id=source['recorder_id'],as_of=source['as_of'],train_start=source['model_signal_start'])
    raw=bundle['primary_frame'];prices=bundle['execution_prices']
    uri=CANONICAL_ROOT/'research'/EXPERIMENT/'qlib_recorder';uri.mkdir(parents=True,exist_ok=True)
    R.set_uri(str(uri))
    with R.start(experiment_name=EXPERIMENT):
        recorder=R.get_recorder()
        R.log_params(plan_sha256=plan_hash,source_recorder_id=source['recorder_id'],campaign_status='active_research')
        recorder.log_artifact(str(card/'selection_plan.json'),artifact_path='inputs')
        R.save_objects(artifact_path='inputs',**{'source_hashes.pkl':hashes,'source_audit.pkl':bundle['audit'],
            'primary_frame.pkl':raw,'execution_prices.pkl':prices})
        forecasts,forecast_audit,forecast_models=fit_oof_forecasts(raw,as_of=source['as_of'])
        dataset=build_anchor_dataset(forecasts,as_of=source['as_of'])
        valid=raw[['soxl_return','soxx_return']].notna().all(axis=1)
        development=forecasts.loc[valid & raw.index.year.isin(plan['training']['development_signal_years'])]
        outer=forecasts.loc[valid & raw.index.year.isin(plan['training']['diagnostic_signal_years'])]
        variants=[v['name'] for v in plan['variants']]
        folds=prepare_folds(dataset,variants,plan['training']['development_signal_years'],plan)
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
        print(f'Anchor utility development sealed: {selected}',flush=True)
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
                mask=path.signal_date.ge('2026-08-03')
                if mask.any():
                    windows.append({'variant':name,'cost_bps':cost,'window':'2026-08-03 onward','start_position':'carried from diagnostic2023start',
                        **paired_metrics(path.loc[mask],reference.loc[mask])})
                    fresh=replay_models(frame.loc[mask.to_numpy()],outer_fitted[chosen],variant=chosen,cost_bps=cost,frozen_year=frozen)
                    fresh_ref=benchmark_path(frame.loc[mask.to_numpy()],cost_bps=cost)
                    windows.append({'variant':name,'cost_bps':cost,'window':'2026-08-03 onward','start_position':'CASH at first window trade',
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
            'seed_metrics.pkl':pd.DataFrame(seed_rows),'training_diagnostics.pkl':diagnostics,
            'ensemble_audits.pkl':ensemble_audits,'independent_path_audits.pkl':path_audits,
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
        training_rows.append({k:audit.get(k) for k in ('variant','year','seed','failure','selected_epoch','centered_worst_utility','elapsed_seconds')})
    tables={'model_selection.csv':ranking,'backtest_benchmark_comparison.csv':comparison,'annual_metrics.csv':annual,
        'development_gates.csv':gates,'bootstrap_analysis.csv':bootstrap,'seed_metrics.csv':pd.DataFrame(seed_rows),
        'training_summary.csv':pd.DataFrame(training_rows),'forecast_training_audit.csv':forecast_audit,
        'control_comparison.csv':pd.DataFrame(controls),'beta_alpha_attribution.csv':pd.DataFrame(attribution),
        'window_metrics.csv':pd.DataFrame(windows)}
    for filename,table in tables.items():table.to_csv(card/filename,index=False)
    summary={'experiment':EXPERIMENT,'campaign_status':'active_research','objective_achieved':False,
        'decision':'historical_candidate_requires_source_sensitivity_and_new_forward' if selected['historical_qualification_pass'] else 'continue_research_no_historical_qualification',
        'selected':selected,'historical_qualification_pass':bool(selected['historical_qualification_pass']),
        'new_forward_qualified':False,'source_sensitivity_completed':False,'source':source,'source_audit':bundle['audit'],
        'seed_scope':'Optimization-start stability only; constant masked inputs and fixed theta/bias make the three seed decision paths structurally identical. Seeds are never independent market observations.',
        'source_sha256':hashes,'plan_sha256':plan_hash,'provider_uri':source['provider_uri'],'region':'us','market':'all',
        'recorder_id':recorder.id,'recorder_uri':str(uri),'recorder_local_dir':str(recdir),
        'recorder_relative_dir':str(recdir.relative_to(CANONICAL_ROOT/'research'/EXPERIMENT)),
        'selection_uses_diagnostic_signal_years':False,'development_last_exit':str(development.exit_date.max().date()),
        'native_qlib_audit':{str(c):v['audit'] for c,v in native.items()},'independent_accounting_audit':path_audits,
        'production_change':False,'real_broker_orders':False,'sidecars':list(tables),
        'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml',
            'recorder_only_artifacts':['inputs/*.pkl','models/*.pkl','fit_checkpoints/*','analysis/portfolio_replays.pkl','analysis/training_diagnostics.pkl','analysis/ensemble_audits.pkl','portfolio_analysis/*.pkl']},
        'qlib_docs_reference':plan['qlib_docs'],
        'limitations':['Historical years have been reviewed; no fresh forward evidence or guaranteed future excess.',
            'Global expert coefficient budget restricts state-dependent expressivity.',
            '21-session state-gradient truncation and nonsmooth fee gradients are optimizer approximations; exact incumbent forward audits do not certify optimality.',
            'Training null protection concerns complete training utility only, not future wealth or tail risk.',
            'Source basis retains published-versus-server adjustment differences and16secondary observations.',
            'Historical open prices and continuous adjusted shares are not evidence of actual spread, impact or partial fills.']}
    json_write(card/'summary.json',summary);json_write(card/'selected_model.json',{'selected':selected,'research_only':True,'new_forward_qualified':False})
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':EXPERIMENT,'provider_uri':source['provider_uri'],'region':'us',
        'recorder_uri':str(uri),'model_selection':'selection_plan.json','artifact_policy':summary['artifact_policy'],
        'segments':{k:plan['training'][k] for k in ('development_signal_years','diagnostic_signal_years')}},sort_keys=False))
    lines=['# 基准可达连续净收益训练结论','',f'完成3组信息对照、每组3个种子；完整持仓路径计入所有费用，训练候选与精确SOXX基准比较。历史资格通过={selected["historical_qualification_pass"]}；稳定净超额目标仍未完成。','',
        '开发期2016–2022，10bp：']
    for row in comparison[comparison.segment.eq('development') & comparison.cost_bps.eq(10)].itertuples():
        lines.append(f'- {row.variant}：收益{row.return_pct:.2f}%，相对SOXX终值{row.relative_wealth:.4f}，MaxDD{row.max_drawdown_pct:.2f}%，CVaR5{row.cvar5_pct:.2f}%。')
    lines+=['','各年差异见annual_metrics.csv，8月3日起持续持仓与独立现金起点分别见window_metrics.csv。三种子不是三条独立市场历史。',
        f'决定：{summary["decision"]}。2023以后为已查看的诊断期，不能作为新留出样本；原SIM未切换。',
        '风险：有限模型容量、截断梯度、零换手边界导数、行情口径和真实成交差异。训练期不劣于基准的utility不保证样本外超额或回撤。',
        '下一步：按失败门与训练诊断继续定位；若历史门通过，先完成来源敏感性，再冻结进入新的前向证据期。',f'Qlib Recorder：{recorder.id}。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
    return summary

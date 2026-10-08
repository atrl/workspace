"""Registered variance-information study with one shared calibration and budget."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import paired_metrics, annual_paired_metrics, native_allocation_reconciliation, validate_replay
from src.data.conditional_variance import (load_qualified_primary, build_raw_variances, build_calibration_dataset,
    calibrate_annual, apply_calibration, RISK_UNITS)
from src.workflow.convex_excess import deterministic_gates, verify_repetitions
from src.workflow.stable_excess_integration import CANONICAL_ROOT, benchmark_path, choose_family, json_write, sha

EXPERIMENT='conditional_variance_20261002'
REPEATS=(1,2,3)
ASSETS=('soxl','soxx','cash')


def fit_oof_forecasts(frame, *, as_of):
    """Same Ridge10 estimator; trained-but-invalid predictions stay available.

    Prediction validity is distinct from training availability so an unusable
    learned estimator cannot silently remove dates from the controls' fit set.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.data.integrated_allocation import (build_native_dataset,prepare_mature_rows,PRIMARY_FEATURES,
        PRIMARY_LABELS,PRIMARY_METADATA,COMMON_PRIMARY_FEATURES,FORECAST_LABELS)
    dataset=build_native_dataset(frame,feature_columns=PRIMARY_FEATURES,label_columns=PRIMARY_LABELS,
        metadata_columns=PRIMARY_METADATA,as_of=as_of)
    result=frame.copy();models={};audits=[]
    result['forecast_model_available_at']=pd.Series(pd.NaT,index=result.index,dtype='datetime64[ns, UTC]')
    for year in sorted(frame.index.year.unique()):
        cutoff=pd.Timestamp(f'{year}-01-01',tz='America/New_York')
        mature=frame.index>=pd.Timestamp('2011-01-03')
        for field in ('feature_available_at','return_available_at','forecast_available_at'):
            mature &= pd.to_datetime(frame[field],utc=True).lt(cutoff).to_numpy()
        required=frame.loc[mature,[*PRIMARY_FEATURES,*FORECAST_LABELS,'soxl_return','soxx_return']]
        if (not np.isfinite(required.to_numpy(float)).all() or (required[['soxl_return','soxx_return']]<=-1).any().any()
                or (required.future_vol_5d<0).any()):
            raise ValueError('blocked: invalid mature OOF source rows cannot be dropped')
        train,audit=prepare_mature_rows(dataset,decision_at=cutoff,label_columns=FORECAST_LABELS,availability_column='forecast_available_at')
        if len(train)!=int(mature.sum()):raise ValueError('blocked: OOF Dataset maturity unexpectedly dropped eligible rows')
        ready=len(train)>=252;invalid=[]
        if ready:
            model=make_pipeline(StandardScaler(),Ridge(alpha=10.))
            model.fit(train.loc[:,list(COMMON_PRIMARY_FEATURES)],train.loc[:,list(FORECAST_LABELS)])
            mask=frame.index.year==year
            prediction=model.predict(frame.loc[mask,list(COMMON_PRIMARY_FEATURES)])
            if prediction.shape!=(int(mask.sum()),2):raise ValueError('blocked: OOF prediction shape differs')
            result.loc[mask,['cash_advantage_prediction','future_vol_prediction']]=prediction
            result.loc[mask,'forecast_available']=1.
            result.loc[mask,'forecast_model_available_at']=cutoff.tz_convert('UTC')
            invalid=frame.index[mask][~np.isfinite(prediction[:,1]) | (prediction[:,1]<=0)].strftime('%Y-%m-%d').tolist()
            models[int(year)]=model
        audits.append({'model_year':int(year),'available':ready,'ridge_alpha':10.,'minimum_training_rows':252,
            'invalid_vol_prediction_count':len(invalid),'invalid_vol_prediction_dates':','.join(invalid),**audit})
    return result,pd.DataFrame(audits),models


def prepare_policy_inputs(market, raw, calibrations, variant, *, frozen_year=None):
    if not market.index.equals(raw.index):raise ValueError('blocked: policy risk rows differ')
    n=len(market);covariance=np.empty((n,2,2));budget=np.empty(n);means=np.empty((n,2));fallback=np.zeros(n,bool);support=np.zeros(n,bool)
    center=np.empty(n);fit_year=np.empty(n,int);audit=[];provenance=[None]*n
    for year in sorted(market.index.year.unique()):
        selected=frozen_year if frozen_year is not None else int(year)
        calibration=calibrations[selected]
        indices=np.flatnonzero(market.index.year==year)
        applied=apply_calibration(calibration,raw.iloc[indices],variant)
        covariance[indices]=applied['covariance'];budget[indices]=applied['benchmark_variance']
        means[indices]=calibration['mu_H5'];fallback[indices]=applied['fallback'];support[indices]=applied['out_of_support']
        center[indices]=calibration['audit']['target_soxx_return_mean'];fit_year[indices]=selected
        identity={'model_year':selected,'cutoff':calibration['cutoff'],'common_dates_sha256':calibration['audit']['common_dates_sha256'],
            'units':RISK_UNITS,'parameter_frozen_at_cutoff':True}
        for index in indices:provenance[index]=identity
        audit.append({'signal_year':int(year),'calibration_year':selected,**applied['audit']})
    if not np.isfinite(means).all() or (means<=-1).any():raise ValueError('blocked: nonphysical H5 prior mean')
    return {'signal_dates':market.index.copy(),'covariance':covariance,'budget':budget,'means':means,'estimator_fallback':fallback,'support':support,
        'return_center':center,'calibration_year':fit_year,'budget_provenance':provenance,'audit':audit}


def replay_policy(market, inputs, *, cost_bps):
    from src.model.convex_allocation import ConvexAllocationSolver
    from src.backtest.position_accounting import settle_target_session
    if 'signal_dates' not in inputs or not pd.DatetimeIndex(inputs['signal_dates']).equals(market.index):
        raise ValueError('blocked: policy inputs are not bound to exact signal dates')
    if (inputs['covariance'].shape!=(len(market),2,2) or inputs['budget'].shape!=(len(market),)
            or inputs['means'].shape!=(len(market),2) or len(inputs['budget_provenance'])!=len(market)):
        raise ValueError('blocked: policy inputs lack exact date coverage')
    solver=ConvexAllocationSolver(cost_bps=cost_bps,tolerance=1e-8,budget_mode='external_positive')
    current=np.array([0.,0.,1.]);equity=1.;rows=[];details=[]
    for i,(_,row) in enumerate(market.iterrows()):
        sigma=inputs['covariance'][i];B=float(inputs['budget'][i])
        result=solver.solve(current,inputs['means'][i],sigma,robust_radius=0.,risk_budget_variance=B,budget_provenance=inputs['budget_provenance'][i])
        returns=np.array([row.soxl_return,row.soxx_return,0.])
        if not np.isfinite(returns).all() or (returns<=-1).any():raise ValueError('blocked: invalid portfolio return labels')
        settled=settle_target_session(current,open_target_weights=result['execution_target_weights'],asset_returns=returns,cost_bps=cost_bps)
        target=settled['open_net_weights'];ratio=float(target[:2]@sigma@target[:2]/B)
        if not np.allclose(target,result['target_net_weights'],atol=1e-10,rtol=0) or ratio>1+1e-8:
            raise ValueError('blocked: independent settlement risk/funding disagrees with solver')
        equity*=settled['net_equity']
        rows.append({'signal_date':row.signal_date,'trade_date':row.trade_date,'exit_date':row.exit_date,
            'strategy_return':settled['net_return'],'turnover':settled['turnover'],'fees':settled['fees'],'equity':equity,'cost_bps':cost_bps,
            'execute_open_target':True,'execute_open_trade':settled['turnover']>1e-12,'objective':result['objective'],'null_gap':result['null_gap'],
            'risk_budget_ratio':ratio,'risk_budget_variance':B,'current_soxx_variance':sigma[1,1],
            'optimality_verified':result['optimality_verified'],'solution_source':result['solution_source'],
            'estimator_fallback':bool(inputs['estimator_fallback'][i]),'out_of_support':bool(inputs['support'][i]),
            **{f'{asset}_weight':float(target[j]) for j,asset in enumerate(ASSETS)},
            **{f'pretrade_{asset}_weight':float(current[j]) for j,asset in enumerate(ASSETS)},
            **{f'ending_{asset}_weight':float(settled['ending_weights'][j]) for j,asset in enumerate(ASSETS)}})
        details.append({'signal_date':row.signal_date,'solver':result['solver_audit'],'nulls':result['null_objectives'],
            'status':result['status'],'optimality_verified':result['optimality_verified']})
        current=settled['ending_weights']
    path=validate_replay(pd.DataFrame(rows))
    audit={'all_optimality_verified':bool(path.optimality_verified.all()),'solver_fallback_sessions':int((~path.optimality_verified).sum()),
        'estimator_fallback_sessions':int(path.estimator_fallback.sum()),'out_of_support_sessions':int(path.out_of_support.sum()),
        'risk_budget_binding_sessions':int((abs(path.risk_budget_ratio-1)<1e-6).sum()),
        'maximum_variance_budget_ratio':float(path.risk_budget_ratio.max()),'minimum_null_gap':float(path.null_gap.min()),
        'mean_nominal_beta':float((3*path.soxl_weight+path.soxx_weight).mean()),
        **{f'mean_{asset}_pct':float(100*path[f'{asset}_weight'].mean()) for asset in ASSETS}}
    return path,audit,details


def paired_block_mean(values, *, block, resamples=2000, alpha=.05/14, seed=20261002):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or len(values)<2*block:
        raise ValueError('blocked: invalid/insufficient paired forecast-loss observations')
    rng=np.random.default_rng(seed);indices=rng.integers(len(values),size=resamples);total=values[indices].copy()
    for _ in range(1,len(values)):
        restart=rng.random(resamples)<1/block;indices=(indices+1)%len(values)
        indices[restart]=rng.integers(len(values),size=int(restart.sum()));total+=values[indices]
    means=total/len(values)
    return {'mean_difference':float(values.mean()),'lower_bound':float(np.quantile(means,alpha)),
        'upper_bound':float(np.quantile(means,1-alpha)),'mean_block':block,'resamples':resamples,'alpha':alpha,
        'scope':'descriptive paired stationary mean-loss difference, one path; not latent variance truth or new evidence'}


def forecast_loss_evidence(market, inputs_by_variant, plan):
    first=next(iter(inputs_by_variant.values()));target=np.square(market.soxx_return.to_numpy()-first['return_center'])
    losses={};metrics=[];differences=[];annual=[]
    for name,inputs in inputs_by_variant.items():
        if not np.array_equal(inputs['return_center'],first['return_center']):raise ValueError('blocked: forecast-loss centers differ')
        variance=inputs['covariance'][:,1,1]
        if not np.isfinite(variance).all() or (variance<=0).any():raise ValueError('blocked: nonphysical evaluated variance')
        qlike=target/variance+np.log(variance);mse=np.square(target-variance);losses[name]={'qlike':qlike,'mse':mse}
        for year in sorted(market.index.year.unique()):
            mask=market.index.year==year
            annual.append({'variant':name,'signal_year':int(year),'rows':int(mask.sum()),'mean_predicted_variance':float(variance[mask].mean()),
                'mean_centered_squared_return':float(target[mask].mean()),'realized_proxy_to_prediction_ratio':float(target[mask].sum()/variance[mask].sum()),
                'mean_qlike':float(qlike[mask].mean()),'mean_squared_loss':float(mse[mask].mean()),
                'estimator_fallback_sessions':int(inputs['estimator_fallback'][mask].sum())})
        metrics.append({'variant':name,'rows':len(market),'mean_qlike':float(qlike.mean()),'mean_squared_loss':float(mse.mean()),
            'estimator_fallback_sessions':int(inputs['estimator_fallback'].sum()),'scope':'prior-centered conditional second-moment proxy'})
    for candidate in losses:
        for reference in losses:
            if candidate==reference:continue
            for loss in ('qlike','mse'):
                for block in plan['evaluation']['mean_block_lengths']:
                    differences.append({'variant':candidate,'reference':reference,'loss':loss,'positive_favors_candidate':True,
                        **paired_block_mean(losses[reference][loss]-losses[candidate][loss],block=block,
                            resamples=plan['evaluation']['bootstrap_resamples'],alpha=.05/plan['evaluation']['family_size_adjustment'],seed=plan['evaluation']['bootstrap_seed'])})
    return pd.DataFrame(metrics),pd.DataFrame(annual),pd.DataFrame(differences),losses


def hac_attribution(path, benchmark):
    """NumPy OLS/Bartlett HAC20, matching the earlier attribution audit."""
    from statistics import NormalDist
    from src.analysis.integrated_allocation import _dates
    path,benchmark=_dates(path),_dates(benchmark)
    if not path[['signal_date','trade_date','exit_date']].equals(benchmark[['signal_date','trade_date','exit_date']]):
        raise ValueError('blocked: attribution dates differ')
    y=path.strategy_return.to_numpy(dtype=float);x=benchmark.strategy_return.to_numpy(dtype=float)
    if len(y)<=22 or not np.isfinite(np.r_[x,y]).all():raise ValueError('blocked: invalid HAC observations')
    design=np.c_[np.ones(len(x)),x]
    if np.linalg.matrix_rank(design)!=2:raise ValueError('blocked: HAC benchmark has no identifiable variation')
    coefficients=np.linalg.lstsq(design,y,rcond=None)[0]
    residual=y-design@coefficients;bread=np.linalg.inv(design.T@design)
    scores=design*residual[:,None];meat=scores.T@scores
    for lag in range(1,21):
        cross=scores[lag:].T@scores[:-lag]
        meat+=(1-lag/21)*(cross+cross.T)
    covariance=(bread@meat@bread)*len(x)/(len(x)-2)
    variance=float(covariance[0,0])
    if not np.isfinite(covariance).all() or variance < -1e-14*max(float(np.abs(covariance).max()),1e-30):
        raise ValueError('blocked: invalid HAC covariance')
    standard_error=np.sqrt(max(variance,0.));critical=NormalDist().inv_cdf(.975)
    return {'realized_beta':float(coefficients[1]),'annualized_arithmetic_alpha':float(coefficients[0]*252),
        'alpha95_lower':float((coefficients[0]-critical*standard_error)*252),
        'alpha95_upper':float((coefficients[0]+critical*standard_error)*252),
        'hac_lags':20,'hac_kernel':'Bartlett','small_sample_correction':'N/(N-2)','interval':'normal95percent',
        'sessions':len(y),'scope':'linear association with benchmark net returns, not causal or selection-adjusted proof',
        'method_reference':'https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html'}


def verify_forecast_repetitions(reference, current, tolerance=1e-8):
    if not reference.index.equals(current.index) or not reference.columns.equals(current.columns):
        raise ValueError('blocked: forecast reproduction axes changed')
    left,right=reference.copy(),current.copy();invalid=0
    for column in ('cash_advantage_prediction','future_vol_prediction'):
        a,b=left[column].to_numpy(float),right[column].to_numpy(float)
        for mask in (np.isnan,np.isposinf,np.isneginf):
            if not np.array_equal(mask(a),mask(b)):raise ValueError('blocked: invalid forecast reproduction masks changed')
        invalid+=int((~np.isfinite(a)).sum())
        # Comparison-only neutral placeholders; actual predictions are retained.
        left[column]=np.where(np.isfinite(a),a,0.)
        right[column]=np.where(np.isfinite(b),b,0.)
    result=verify_repetitions(left,right,tolerance=tolerance)
    return {**result,'matching_nonfinite_forecast_values':invalid,'predictions_modified':False}


def verify_model_repetitions(reference, current, tolerance=1e-8):
    if set(reference)!=set(current):raise ValueError('blocked: repeated model years differ')
    maximum=0.
    for year in reference:
        a,b=reference[year].named_steps,current[year].named_steps
        for step,fields in [('standardscaler',('mean_','scale_','var_','n_samples_seen_')),('ridge',('coef_','intercept_'))]:
            for field in fields:
                result=verify_repetitions(np.asarray(getattr(a[step],field)),np.asarray(getattr(b[step],field)),tolerance=tolerance)
                maximum=max(maximum,result['maximum_absolute_difference'])
        if a['standardscaler'].feature_names_in_.tolist()!=b['standardscaler'].feature_names_in_.tolist():
            raise ValueError('blocked: repeated model feature schema differs')
    return {'all_values_match':True,'maximum_absolute_difference':maximum,'tolerance':tolerance,'model_years':len(reference),
        'independent_statistical_evidence':False}


def variance_gates(paths, benchmarks, repeated_metrics, profile_audit, *, repeats_verified, family_size=14):
    if not isinstance(repeats_verified,(bool,np.bool_)):
        raise ValueError('blocked: repeat verification must be an explicit boolean')
    if not profile_audit.all_optimality_verified.map(lambda value:isinstance(value,(bool,np.bool_))).all():
        raise ValueError('blocked: solver verification flags must be explicit booleans')
    expected={(repeat,cost) for repeat in REPEATS for cost in paths}
    if profile_audit.duplicated(['repeat_id','cost_bps']).any() or set(zip(profile_audit.repeat_id,profile_audit.cost_bps))!=expected:
        raise ValueError('blocked: incomplete or duplicate repeated profile evidence')
    counts=profile_audit[['solver_fallback_sessions','estimator_fallback_sessions']].to_numpy(float)
    if not np.isfinite(counts).all() or (counts<0).any():raise ValueError('blocked: invalid profile failure counts')
    verified=bool(profile_audit.all_optimality_verified.all() and profile_audit.solver_fallback_sessions.eq(0).all())
    result=deterministic_gates(paths,benchmarks,repeated_metrics,solver_verified=verified,repeats_verified=repeats_verified,family_size=family_size)
    result['gates']['no_invalid_estimator_fallbacks']=bool(profile_audit.estimator_fallback_sessions.eq(0).all())
    result['gates']['all_pass'] &= result['gates'].no_invalid_estimator_fallbacks
    result['historical_qualification_pass']=bool(result['gates'].all_pass.all())
    return result


def _prediction_view(frame):
    return frame[['cash_advantage_prediction','future_vol_prediction','forecast_available','forecast_model_available_at']]


def _calibration_identity(values):
    return hashlib.sha256(json.dumps(values,sort_keys=True,default=str,allow_nan=False).encode()).hexdigest()


def run_experiment(*, root):
    import qlib
    from qlib.workflow import R
    from src.analysis.convex_qlib import native_open_reconciliation as precise_native
    from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts
    root=Path(root).resolve();card=root/'research'/EXPERIMENT
    plan=json.loads((card/'selection_plan.json').read_text());plan_hash=sha(card/'selection_plan.json');source=plan['source']
    bundle=load_qualified_primary(root,provider_uri=source['provider_uri'],parent_summary='research/'+source['experiment']+'/summary.json',
        expected_recorder_id=source['recorder_id'],as_of=source['as_of'],train_start=source['model_signal_start'])
    raw=bundle['primary_frame'];prices=bundle['execution_prices'];valid=raw[['soxl_return','soxx_return']].notna().all(axis=1)
    development=raw.loc[valid & raw.index.year.isin(plan['training']['development_signal_years'])]
    outer=raw.loc[valid & raw.index.year.isin(plan['training']['diagnostic_signal_years'])]
    sources=('src/data/conditional_variance.py','src/model/convex_allocation.py','src/workflow/conditional_variance.py',
        'src/data/integrated_allocation.py','src/data/convex_allocation.py','src/analysis/integrated_allocation.py',
        'src/workflow/convex_excess.py','src/analysis/convex_qlib.py','src/backtest/qlib_continuous_position.py',
        'src/workflow/benchmark_risk_excess.py','src/workflow/stable_excess_integration.py','src/backtest/position_accounting.py',
        'src/workflow/stage1_economic_action.py','src/data/qlib_native.py','src/data/price_basis.py',
        'src/factor/soxx_pit_tail_factors.py','src/workflow/pit_constituent_auxiliary_training_allocation.py')
    source_hashes={p:sha(root/p) for p in sources};costs=plan['evaluation']['cost_bps'];variants=plan['variants']
    benchmarks={cost:benchmark_path(development,cost_bps=cost) for cost in costs}
    forecasts={};models={};calibrations={};raw_variances={};datasets={};inputs={};paths={};details={}
    forecast_audits=[];calibration_audits=[];reproduction=[];profile_audits=[];replays=[];repeat_metrics=[]
    for repeat in REPEATS:
        forecast,fit_audit,fitted=fit_oof_forecasts(raw,as_of=source['as_of'])
        risk=build_raw_variances(forecast,prices);dataset=build_calibration_dataset(forecast,risk,as_of=source['as_of'])
        cals={year:calibrate_annual(dataset,year) for year in plan['training']['development_signal_years']}
        forecasts[repeat],models[repeat],calibrations[repeat],raw_variances[repeat],datasets[repeat]=forecast,fitted,cals,risk,dataset
        forecast_audits.append(fit_audit.assign(repeat_id=repeat))
        calibration_audits.extend({'repeat_id':repeat,'segment':'development','year':year,'calibration':value} for year,value in cals.items())
        if repeat>1:
            reproduction.append({'repeat_id':repeat,'kind':'oof_predictions',**verify_forecast_repetitions(_prediction_view(forecasts[1]),_prediction_view(forecast))})
            reproduction.append({'repeat_id':repeat,'kind':'model_parameters',**verify_model_repetitions(models[1],fitted)})
            if _calibration_identity(calibrations[1])!=_calibration_identity(cals):raise ValueError('blocked: fresh calibration repetition differs')
        for variant in variants:
            name=variant['name'];prepared=prepare_policy_inputs(development,risk.loc[development.index],cals,variant['variance_key'])
            inputs[(repeat,name)]=prepared
        if repeat==1:
            forecast_metrics,forecast_annual,forecast_bootstrap,forecast_losses=forecast_loss_evidence(development,
                {v['name']:inputs[(1,v['name'])] for v in variants},plan)
        for variant in variants:
            name=variant['name']
            for cost in costs:
                path,audit,daily=replay_policy(development,inputs[(repeat,name)],cost_bps=cost)
                paths[(repeat,name,cost)]=path;details[(repeat,name,cost)]=daily
                profile_audits.append({'repeat_id':repeat,'variant':name,'segment':'development','cost_bps':cost,**audit})
                replays.append(path.assign(repeat_id=repeat,variant=name,segment='development'))
                repeat_metrics.append({'repeat_id':repeat,'variant':name,'cost_bps':cost,**paired_metrics(path,benchmarks[cost])})
                if repeat>1:reproduction.append({'repeat_id':repeat,'kind':'development_path','variant':name,'cost_bps':cost,
                    **verify_repetitions(paths[(1,name,cost)],path)})
                print(f'Variance repeat={repeat} {name} cost={cost} verified={audit["all_optimality_verified"]} estimator_fallbacks={audit["estimator_fallback_sessions"]}',flush=True)
    evidence={}
    for variant in variants:
        name=variant['name'];audit=pd.DataFrame(profile_audits).query('variant==@name')
        result=variance_gates({cost:paths[(1,name,cost)] for cost in costs},benchmarks,pd.DataFrame(repeat_metrics).query('variant==@name'),
            audit,repeats_verified=True,family_size=plan['evaluation']['family_size_adjustment'])
        evidence[name]=result
    selected,ranking=choose_family(evidence)
    json_write(card/'selection_seal.json',{'plan_sha256':plan_hash,'selected':selected,'development_last_exit':str(development.exit_date.max().date()),
        'selection_uses_diagnostic_signal_years':False,'campaign_status':'active_research'})
    print(f'Variance development sealed: {selected}',flush=True)
    chosen=next(v for v in variants if v['name']==selected['variant'])
    comparisons=[r['metrics'].assign(variant=name,segment='development') for name,r in evidence.items()]
    annuals=[r['annual'].assign(variant=name,segment='development') for name,r in evidence.items()]
    control_rows=[];attribution=[]
    for variant in variants:
        name=variant['name']
        for cost in costs:
            path=paths[(1,name,cost)]
            for reference in variants:
                if reference['name']!=name:control_rows.append({'variant':name,'reference':reference['name'],'cost_bps':cost,
                    **paired_metrics(path,paths[(1,reference['name'],cost)])})
            attribution.append({'variant':name,'cost_bps':cost,'signal_year':'all',**hac_attribution(path,benchmarks[cost])})
            for year in plan['training']['development_signal_years']:
                mask=path.signal_date.dt.year.eq(year)
                attribution.append({'variant':name,'cost_bps':cost,'signal_year':str(year),**hac_attribution(path.loc[mask].reset_index(drop=True),benchmarks[cost].loc[mask].reset_index(drop=True))})
    outer_paths={};outer_inputs={};outer_forecasts={}
    for repeat in REPEATS:
        cals={year:calibrate_annual(datasets[repeat],year) for year in plan['training']['diagnostic_signal_years']}
        calibration_audits.extend({'repeat_id':repeat,'segment':'reused_diagnostic','year':year,'calibration':value} for year,value in cals.items())
        for mode in ('annual_refit','frozen2022_calibration'):
            if mode=='annual_refit':
                forecast=forecasts[repeat].loc[outer.index];risk=raw_variances[repeat].loc[outer.index];frozen_year=None
            else:
                forecast=raw.loc[outer.index].copy();prediction=models[repeat][2023].predict(forecast.loc[:,list(COMMON_PRIMARY_FEATURES)])
                forecast.loc[:,['cash_advantage_prediction','future_vol_prediction']]=prediction
                forecast.loc[:,'forecast_available']=1.;forecast['forecast_model_available_at']=pd.Timestamp('2023-01-01',tz='America/New_York').tz_convert('UTC')
                risk=build_raw_variances(forecast,prices);frozen_year=2023
            outer_forecasts[(repeat,mode)]=forecast
            prepared=prepare_policy_inputs(outer,risk,cals,chosen['variance_key'],frozen_year=frozen_year);outer_inputs[(repeat,mode)]=prepared
            if repeat>1:reproduction.append({'repeat_id':repeat,'kind':mode+'_predictions',**verify_forecast_repetitions(_prediction_view(outer_forecasts[(1,mode)]),_prediction_view(forecast))})
            for cost in costs:
                path,audit,daily=replay_policy(outer,prepared,cost_bps=cost);name=chosen['name']+'_'+mode
                outer_paths[(repeat,mode,cost)]=path;details[('outer',repeat,mode,cost)]=daily
                profile_audits.append({'repeat_id':repeat,'variant':name,'segment':'reused_diagnostic','cost_bps':cost,**audit})
                replays.append(path.assign(repeat_id=repeat,variant=name,segment='reused_diagnostic'))
                if repeat>1:reproduction.append({'repeat_id':repeat,'kind':mode+'_path','cost_bps':cost,
                    **verify_repetitions(outer_paths[(1,mode,cost)],path)})
                if repeat==1:
                    reference=benchmark_path(outer,cost_bps=cost)
                    comparisons.append(pd.DataFrame([{'variant':name,'segment':'reused_diagnostic','cost_bps':cost,**paired_metrics(path,reference)}]))
                    annuals.append(annual_paired_metrics(path,reference).assign(variant=name,segment='reused_diagnostic',cost_bps=cost))
                print(f'Variance diagnostic repeat={repeat} {mode} cost={cost}',flush=True)
    native={cost:native_allocation_reconciliation(development.reset_index(drop=True),paths[(1,chosen['name'],cost)],
        provider_uri=root/source['provider_uri'],cost_bps=cost,native_reconciler=precise_native) for cost in costs}
    for cost in costs:
        replays.append(benchmarks[cost].assign(repeat_id=0,variant='SOXX_buy_hold',segment='development'))
        replays.append(benchmark_path(outer,cost_bps=cost).assign(repeat_id=0,variant='SOXX_buy_hold',segment='reused_diagnostic'))
    comparison=pd.concat(comparisons,ignore_index=True);annual=pd.concat(annuals,ignore_index=True)
    gates=pd.concat([r['gates'].assign(variant=name) for name,r in evidence.items()],ignore_index=True)
    bootstrap=pd.concat([r['bootstrap'].assign(variant=name) for name,r in evidence.items()],ignore_index=True)
    control=pd.DataFrame(control_rows)
    learned=control[control.variant.eq('learned_vol5') & control.reference.isin(['observed_open20','observed_close20'])]
    learned_increment=bool(len(learned)==4 and learned.log_excess_sum.gt(1e-8).all())
    learned_profiles=pd.DataFrame(profile_audits).query("variant=='learned_vol5' and segment=='development'")
    learned_integrity=bool(learned_profiles.all_optimality_verified.all() and learned_profiles.solver_fallback_sessions.eq(0).all()
        and learned_profiles.estimator_fallback_sessions.eq(0).all())
    if sha(card/'selection_plan.json')!=plan_hash or any(sha(root/p)!=v for p,v in source_hashes.items()):raise ValueError('blocked: registered plan or implementation changed during run')
    uri=CANONICAL_ROOT/'research'/EXPERIMENT/'qlib_recorder';uri.mkdir(parents=True,exist_ok=True)
    qlib.init(provider_uri=str(root/source['provider_uri']),region='us',kernels=1);R.set_uri(str(uri))
    with R.start(experiment_name=EXPERIMENT):
        recorder=R.get_recorder();R.log_params(plan_sha256=plan_hash,source_recorder_id=source['recorder_id'],campaign_status='active_research')
        R.log_metrics(historical_qualification_pass=float(selected['historical_qualification_pass']),registered_variants=4,fresh_repetitions=3)
        R.save_objects(artifact_path='inputs',**{'primary_frame.pkl':raw,'execution_prices.pkl':prices,'source_audit.pkl':bundle['audit'],
            'forecast_frames.pkl':forecasts,'raw_variances.pkl':raw_variances,'annual_calibrations.pkl':calibration_audits})
        R.save_objects(artifact_path='models',**{'annual_forecasters.pkl':models})
        R.save_objects(artifact_path='policy',**{'development_inputs.pkl':inputs,'diagnostic_inputs.pkl':outer_inputs,'diagnostic_forecasts.pkl':outer_forecasts})
        R.save_objects(artifact_path='analysis',**{'portfolio_replays.pkl':pd.concat(replays,ignore_index=True),'comparison.pkl':comparison,'annual.pkl':annual,
            'gates.pkl':gates,'bootstrap.pkl':bootstrap,'repeated_metrics.pkl':pd.DataFrame(repeat_metrics),'reproduction.pkl':pd.DataFrame(reproduction),
            'profile_audits.pkl':pd.DataFrame(profile_audits),'daily_solver_audits.pkl':details,'controls.pkl':control,'attribution.pkl':pd.DataFrame(attribution),
            'forecast_metrics.pkl':forecast_metrics,'forecast_annual.pkl':forecast_annual,'forecast_bootstrap.pkl':forecast_bootstrap,'forecast_losses.pkl':forecast_losses})
        R.save_objects(artifact_path='portfolio_analysis',**{'native_reconciliation.pkl':native})
        recorder.log_artifact(str(card/'selection_plan.json'),artifact_path='inputs');recorder.log_artifact(str(card/'selection_seal.json'),artifact_path='inputs')
        recdir=Path(recorder.get_local_dir());pd.testing.assert_frame_equal(comparison,recorder.load_object('analysis/comparison.pkl'))
    _remove_qlib_code_snapshot_artifacts(recdir)
    tables={'model_selection.csv':ranking,'backtest_benchmark_comparison.csv':comparison,'annual_metrics.csv':annual,'development_gates.csv':gates,
        'bootstrap_analysis.csv':bootstrap,'repeated_metrics.csv':pd.DataFrame(repeat_metrics),'reproduction_audit.csv':pd.DataFrame(reproduction),
        'risk_solver_audit.csv':pd.DataFrame(profile_audits),'control_comparison.csv':control,'beta_alpha_attribution.csv':pd.DataFrame(attribution),
        'forecast_metrics.csv':forecast_metrics,'forecast_annual_metrics.csv':forecast_annual,'forecast_loss_bootstrap.csv':forecast_bootstrap,
        'forecast_training_audit.csv':pd.concat(forecast_audits,ignore_index=True)}
    for filename,table in tables.items():table.to_csv(card/filename,index=False)
    summary={'experiment':EXPERIMENT,'campaign_status':'active_research','objective_achieved':False,
        'decision':'historical_candidate_requires_source_sensitivity_and_new_forward' if selected['historical_qualification_pass'] else 'continue_research_no_historical_qualification',
        'selected':selected,'historical_qualification_pass':bool(selected['historical_qualification_pass']),'learned_increment_over_both_observed_controls_at_both_costs':learned_increment,
        'learned_attribution_integrity_eligible':bool(learned_increment and learned_integrity),
        'learned_increment_scope':'descriptive net path differences; neither fallback policy nor positive point estimates alone establish learned alpha',
        'new_forward_qualified':False,'source_sensitivity_completed':False,'source':source,'source_audit':bundle['audit'],'source_sha256':source_hashes,'plan_sha256':plan_hash,
        'provider_uri':source['provider_uri'],'region':'us','market':'all','recorder_id':recorder.id,'recorder_uri':str(uri),'recorder_local_dir':str(recdir),
        'recorder_relative_dir':str(recdir.relative_to(CANONICAL_ROOT/'research'/EXPERIMENT)),
        'development_last_exit':str(development.exit_date.max().date()),'selection_uses_diagnostic_signal_years':False,
        'native_qlib_audit':{str(k):v['audit'] for k,v in native.items()},'deterministic_reproduction':{'fresh_runs':3,'all_match':True,'tolerance':1e-8,'independent_statistical_observations':False},
        'production_change':False,'real_broker_orders':False,'sidecars':list(tables),
        'qlib_docs_reference':['https://qlib.readthedocs.io/en/latest/component/data.html','https://qlib.readthedocs.io/en/latest/component/workflow.html'],
        'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*.pkl','models/*.pkl','policy/*.pkl','analysis/portfolio_replays.pkl','analysis/daily_solver_audits.pkl','portfolio_analysis/*.pkl']},
        'limitations':['Reviewed historical data is not new forward evidence; family14does not correct all adaptive research.',
            'Annual B scale cancels with common moment calibration; conditional time-varying variance ratios remain. This is normalized variance timing.',
            'Prior-centered squared returns are a conditional second-moment proxy, not observed true conditional variance.',
            'H5planning ignores future trades; actual daily costs, realized risk and qualification gates remain binding.',
            'Published-affine price basis and16secondary quotes retain documented source uncertainty.']}
    json_write(card/'summary.json',summary);json_write(card/'selected_model.json',{'selected':selected,'variant_spec':chosen,'research_only':True,'new_forward_qualified':False})
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':EXPERIMENT,'provider_uri':source['provider_uri'],'region':'us','recorder_uri':str(uri),
        'model_selection':'selection_plan.json','artifact_policy':summary['artifact_policy'],'segments':plan['training']},sort_keys=False))
    lines=['# 条件波动比较结论','',f'完成4组对照、3次独立重拟合/重算及10/25bp实际费用路径。历史资格通过={selected["historical_qualification_pass"]}；稳定超额目标仍在进行。',
        '', '统一使用合格Qlib价格口径、同一H5先前均值、共同成熟日期及风险形状。学习预测与开盘/收盘20日对照分开，检验条件波动信息而不是调高风险预算。', '', '开发期2016–2022信号年，10bp：']
    for row in comparison[comparison.segment.eq('development') & comparison.cost_bps.eq(10)].itertuples():
        lines.append(f'- {row.variant}：收益{row.return_pct:.2f}%，相对SOXX终值{row.relative_wealth:.4f}，MaxDD{row.max_drawdown_pct:.2f}%，CVaR5{row.cvar5_pct:.2f}%。')
    lines += ['',f'学习版本记账路径相对两个20日观察对照、两档费用均有正增量={learned_increment}；估计与求解完整性通过={learned_integrity}。正点估计或回退路径本身不能证明学到了alpha。预测损失、各年差异及费用压力分别保留，三次复现不增加统计样本。',
        '',f'推广决定：{summary["decision"]}。2023以后已看过的诊断期未参与排名；数据敏感性及新的前向证据尚未完成。',
        '', '风险包括波动预测误差、归一化校准的跨年变化、H5规划与日重配差异、交易成本及报价来源不确定性。下一步依据失败门和相对对照的信息增量继续研究；没有切换原SIM或提交真实委托。',f'Qlib Recorder：{recorder.id}。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
    return summary

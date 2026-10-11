"""Selected native horizon forecasts, certified targets and complete native execution QA."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.analysis.generic_native_reconciliation import reconcile_native_horizon
from src.backtest.generic_horizon_replay import replay_matched_horizon
from src.data.convex_allocation import FEATURE_COLUMNS
from src.data.generic_horizon import allocation_clock, unit_tags
from src.model.benchmark_terminal_value import transform_benchmark_terminal_inputs
from src.model.generic_horizon_estimator import predict_horizon


def replay_horizon_window(frame, selections, daily_covariances, *, calendar, cost_bps,
                          horizon_sessions=20, robust_radius=1., initial_cash=100000.,
                          benchmark_only=False, feature_columns=FEATURE_COLUMNS,
                          availability_column=None, eligibility_column=None, fallback_years=()):
    """No fitting. The caller aligns all daily covariances to this exact window."""
    market = frame.copy()
    market.index = pd.DatetimeIndex(market.signal_date)
    market['prediction_at'] = market.decision_at
    market['decision_at'] = allocation_clock(market.trade_date)
    means = np.zeros((len(market),2))
    uncertainty = np.zeros((len(market),2,2))
    forecast_rows = []
    model_used = np.zeros(len(market), dtype=bool)
    if not benchmark_only:
        if not set(market.index.year).issubset(set(selections) | set(fallback_years)):
            raise ValueError('blocked: window contains a signal year without a registered fitted model')
        known = np.ones(len(market), dtype=bool)
        if eligibility_column is not None:
            if eligibility_column not in market or not market[eligibility_column].isin([0,1]).all():
                raise ValueError('blocked: window eligibility must be explicit0/1')
            known = market[eligibility_column].to_numpy(bool)
        availability_column = availability_column or f'label_h{horizon_sessions}_available_at'
        table = pd.DataFrame(dict(signal_date=market.signal_date,prediction_at=market.prediction_at,
            label_available_at=market[availability_column],model_year=np.nan,
            predicted_soxl=np.nan,predicted_soxx=np.nan,prior_soxl=np.nan,prior_soxx=np.nan,
            actual_soxl=market[f'soxl_return_h{horizon_sessions}'],actual_soxx=market[f'soxx_return_h{horizon_sessions}'],
            model_used=False,not_used_reason=np.where(known,'registered_fallback_year','source_unknown')),index=market.index)
        for year in sorted(market.index.year.unique()):
            if int(year) in fallback_years:
                if int(year) in selections:
                    raise ValueError('blocked: fallback year must have no fitted model')
                continue
            mask = (market.index.year == year) & known
            part = market.loc[mask]
            if part.empty:
                continue
            model = selections[int(year)]['selected_fit']
            predicted = predict_horizon(model, part, feature_columns=feature_columns,
                                        horizon_sessions=horizon_sessions)
            means[mask], uncertainty[mask] = predicted['ridge'], predicted['Vmean']
            model_used[mask] = True
            table.loc[mask,['predicted_soxl','predicted_soxx']]=predicted['ridge']
            table.loc[mask,['prior_soxl','prior_soxx']]=predicted['prior']
            table.loc[mask,'model_year']=int(year)
            table.loc[mask,'model_used']=True
            table.loc[mask,'not_used_reason']=''
        forecast_rows.append(table)
        transformed = transform_benchmark_terminal_inputs(means, uncertainty, cost_bps=cost_bps)
        effective_means, effective_covariance = transformed['transformed_mu'], transformed['transformed_covariance']
        from src.model.certified_convex_allocation import CertifiedConvexAllocationSolver
        optimizer = CertifiedConvexAllocationSolver(cost_bps=cost_bps, tolerance=1e-8,
                                                    forecast_horizon_sessions=horizon_sessions)
        solutions = []
        rebalance_call = 0
        def solver(held, mu, vmean, sigma):
            nonlocal rebalance_call
            position = rebalance_call * horizon_sessions
            rebalance_call += 1
            if not model_used[position]:
                from src.workflow.stage1_economic_action import fee_affordable_target
                target = np.array([0.,1.,0.])
                result = dict(target_net_weights=target,
                    execution_target_weights=fee_affordable_target(target,pretrade_weights=held,cost_bps=cost_bps),
                    decision_source='pre_registered_SOXX_fallback',model_used=False)
                solutions.append(result)
                return result
            result = optimizer.solve(held, mu, sigma, mean_error_covariance_H=vmean, robust_radius=robust_radius)
            if result.get('optimality_verified') is not True:
                raise ValueError('blocked: horizon optimizer lacks a verified optimality certificate')
            solutions.append(result)
            return result
    else:
        effective_means, effective_covariance = means, uncertainty
        transformed, solutions = {}, []
        from src.workflow.stage1_economic_action import fee_affordable_target
        def solver(held, mu, vmean, sigma):
            target = np.array([0.,1.,0.])
            return dict(target_net_weights=target,
                        execution_target_weights=fee_affordable_target(target,pretrade_weights=held,cost_bps=cost_bps))
    path, audit = replay_matched_horizon(market, effective_means, effective_covariance, daily_covariances,
        solver=solver, calendar=calendar, horizon_sessions=horizon_sessions, cost_bps=cost_bps,
        unit_tags=unit_tags(horizon_sessions), buy_and_hold=benchmark_only)
    if benchmark_only:
        if not (path.fees.iloc[1:] == 0).all() or not (path.turnover.iloc[1:] == 0).all():
            raise ValueError('blocked: SOXX benchmark traded after initial entry')
        path['execute_open_target'] = np.arange(len(path)) == 0
    path['model_used'] = model_used
    path['fallback_rebalance'] = (~model_used) & path.execute.to_numpy() & (not benchmark_only)
    path['planning_placeholder_not_prediction'] = ~model_used
    return dict(path=path, forecast=pd.concat(forecast_rows) if forecast_rows else pd.DataFrame(),
                forecast_mean_covariance=uncertainty, solver_details=solutions, replay_audit=audit,
                transform_audit=transformed.get('audit'), input_covariance=np.asarray(daily_covariances))


def native_verify_window(frame, result, *, provider_uri, cost_bps, initial_cash):
    """Archive this result only after native cash, quantities, fees and weights all pass."""
    native = reconcile_native_horizon(frame, result['path'], provider_uri=provider_uri,
                                      cost_bps=cost_bps, account=initial_cash)
    from qlib.contrib.evaluate import risk_analysis
    native['risk_analysis_next_open'] = risk_analysis(
        native['reconciliation'].native_next_open_return, N=252, freq=None, mode='product')
    path = result['path'].copy()
    path['research_strategy_return'] = path.strategy_return
    path['strategy_return'] = native['reconciliation'].native_next_open_return.to_numpy()
    path['equity'] = np.cumprod(1 + path.strategy_return.to_numpy())
    result = {**result, 'path':path, 'native':native}
    return result

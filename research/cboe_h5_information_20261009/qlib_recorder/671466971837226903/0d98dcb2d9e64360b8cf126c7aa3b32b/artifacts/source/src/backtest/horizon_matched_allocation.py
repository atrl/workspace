"""Fixed-cadence convex targets with owned HOLD states and independent funding.

H5 forecasts retain their original units. Cadence one is the daily-replanning
control; cadence five implements five-session holdings. Estimated variance is
constrained at decisions only; subsequent HOLD drift is measured, not hidden.
Qlib native replay remains required before these ledgers become evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import ASSETS, validate_replay
from src.backtest.benchmark_relative_replay import _settings, _step, _target, validate_market
from src.backtest.position_accounting import settle_target_session
from src.model.convex_allocation import _psd_matrix_root
from src.model.certified_convex_allocation import CertifiedConvexAllocationSolver as ConvexAllocationSolver


def _covariances(values, count, name):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"blocked: invalid {name} array") from error
    if array.shape not in ((count, 2, 2), (count, 3)):
        raise ValueError(f"blocked: {name} must contain one two-asset matrix per session")
    matrices, repairs = [], []
    for value in array:
        matrix, _, audit = _psd_matrix_root(value, name=name)
        matrices.append(matrix)
        repairs.append(audit["maximum_roundoff_repair"])
    return np.stack(matrices), max(repairs, default=0.)


def replay_horizon_allocation(frame, expected_returns, mean_covariances, daily_covariances, *,
                              cost_bps, rebalance_sessions, robust_radius=1.):
    """Replay a complete dated path; no label enters the target solver.

    Forecast and covariance arrays are positional in the checked frame's exact
    row order. Their availability/provenance must be sealed upstream. Every row
    is validated, including rows on which no optimization takes place.
    """
    market = validate_market(frame)
    _settings(cost_bps, rebalance_sessions)
    if isinstance(robust_radius, (bool, np.bool_)) or not np.isscalar(robust_radius) \
            or not np.isfinite(robust_radius) or robust_radius < 0:
        raise ValueError("blocked: invalid robust radius")
    expected_decisions = (pd.DatetimeIndex(market.signal_date).tz_localize("America/New_York")
                          + pd.Timedelta(hours=16, minutes=15)).tz_convert("UTC")
    if not pd.DatetimeIndex(market.decision_at).equals(expected_decisions):
        raise ValueError("blocked: horizon allocation requires signal-day 16:15 decisions")
    if (not np.array_equal(market.trade_date.to_numpy()[:-1], market.signal_date.to_numpy()[1:])
            or not np.array_equal(market.exit_date.to_numpy()[:-2], market.signal_date.to_numpy()[2:])):
        raise ValueError("blocked: horizon replay needs consecutive signal/trade/exit sessions")
    try:
        means = np.asarray(expected_returns, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("blocked: invalid expected returns") from error
    if means.shape != (len(market), 2) or not np.isfinite(means).all() or (means <= -1).any():
        raise ValueError("blocked: complete finite two-asset H5 forecasts required")
    uncertainties, mean_repair = _covariances(mean_covariances, len(market), "H5 mean covariance")
    covariances, risk_repair = _covariances(daily_covariances, len(market), "daily risk covariance")
    solver = ConvexAllocationSolver(cost_bps=cost_bps, tolerance=1e-8, forecast_horizon_sessions=5)
    held, equity = np.array([0., 0., 1.]), 1.
    records, details = [], []
    maximum_accounting_error = 0.
    for index, row in enumerate(market.itertuples()):
        execute = index % rebalance_sessions == 0
        solution = None
        if execute:
            solution = solver.solve(held.copy(), means[index].copy(), covariances[index].copy(),
                mean_error_covariance_H=uncertainties[index].copy(), robust_radius=robust_radius)
            if solution.get("optimality_verified") is not True:
                raise ValueError(f"blocked: unverified convex solve at {row.signal_date}; fallback cannot qualify")
            target = _target(solution["target_net_weights"])
            executable = _target(solution["execution_target_weights"])
        else:
            target, executable = held.copy(), None

        sigma = covariances[index]
        variance = float(target[:2] @ sigma @ target[:2])
        budget = float(sigma[1, 1])
        ratio = variance / budget if budget > 0 else (0. if variance == 0 else None)
        exceeds = variance > budget * (1 + 1e-8) if budget > 0 else variance > 1e-20
        if execute and exceeds:
            raise ValueError("blocked: independently checked rebalance target exceeds SOXX variance")

        record, ending, next_equity = _step(row, held, target, cost_bps, equity, execute)
        independent = settle_target_session(held, open_target_weights=executable,
            asset_returns=[row.soxl_return, row.soxx_return, 0.], cost_bps=cost_bps)
        errors = [np.max(np.abs(independent["open_net_weights"] - target)),
                  np.max(np.abs(independent["ending_weights"] - ending)),
                  abs(independent["net_return"] - record["strategy_return"]),
                  abs(independent["fees"] - record["fees"]),
                  abs(independent["turnover"] - record["turnover"])]
        if execute:
            errors.extend([abs(float(solution["fees"]) - record["fees"]),
                           abs(float(solution["turnover"]) - record["turnover"])])
        if not np.isfinite(errors).all() or max(errors) > 1e-10:
            raise ValueError("blocked: optimizer target or independent cash/fee settlement differs")
        if not execute and (record["fees"] != 0 or record["turnover"] != 0):
            raise ValueError("blocked: exact HOLD unexpectedly traded")
        maximum_accounting_error = max(maximum_accounting_error, *errors)
        record.update(risk_ratio=ratio, portfolio_variance=variance, benchmark_variance=budget,
            estimated_risk_exceeds_benchmark=bool(exceeds), risk_constraint_enforced=bool(execute),
            optimizer_called=bool(execute), optimality_verified=True if execute else None,
            solution_source=solution["solution_source"] if execute else "exact_HOLD",
            execute_open_trade=bool(record["turnover"] > 1e-12),
            decision_at=row.decision_at, return_available_at=row.return_available_at)
        records.append(record)
        if execute:
            details.append({"row_number": index, "signal_date": row.signal_date,
                            "decision_at": row.decision_at, "solution": solution})
        held, equity = ending.copy(), next_equity

    path = validate_replay(pd.DataFrame(records))
    hold = ~path.optimizer_called
    return path, {
        "sessions": len(path), "solver_calls": len(details), "rebalance_sessions": rebalance_sessions,
        "forecast_horizon_sessions": 5, "cost_bps": float(cost_bps), "robust_radius": float(robust_radius),
        "all_solves_optimality_verified": True, "solver_fallback_sessions": 0,
        "rebalance_risk_pass": True, "hold_sessions": int(hold.sum()),
        "hold_risk_exceedance_sessions": int((hold & path.estimated_risk_exceeds_benchmark).sum()),
        "maximum_accounting_error": float(maximum_accounting_error),
        "mean_covariance_maximum_roundoff_repair": mean_repair,
        "daily_covariance_maximum_roundoff_repair": risk_repair,
        "risk_scope": "same-SOXX estimated variance only at rebalance; HOLD drift reported without forced trades",
        "input_availability_scope": "caller must verify forecast/covariance availability before each decision",
        "native_qlib_verified": False, "solver_details": details,
    }

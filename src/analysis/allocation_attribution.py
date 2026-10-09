"""Descriptive attribution of recorded allocations, without rerunning a policy.

Gross and fee components keep the realized positions fixed. They are not a
counterfactual strategy with different transaction costs. Realized betas use
future outcomes within the reporting window and must never become signals.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import ASSETS, DATES, _dates, validate_replay


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def attribute_allocations(paths, market, *, windows, benchmark="soxx", tolerance=1e-10):
    """Return one small attribution row per path, cost and explicit window.

    Input paths must already have native Qlib verification and identical dates.
    This function verifies the accounting arithmetic, not external provenance.
    Cash earns zero; fees are a fraction of pretrade NAV and open weights are
    normalized after payment. No joins, missing-row removal or fitting occurs.
    """
    _require(isinstance(tolerance, Real) and not isinstance(tolerance, bool)
             and np.isfinite(tolerance) and tolerance > 0, "invalid identity tolerance")
    _require(isinstance(paths, Mapping) and paths and isinstance(windows, Mapping) and windows,
             "explicit nonempty paths and windows are required")
    frame = _dates(market)
    returns = frame[["soxl_return", "soxx_return"]].to_numpy(float)
    _require(np.isfinite(returns).all() and (returns > -1).all(), "invalid complete market returns")
    checked = {}
    for key, supplied in paths.items():
        _require(isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str)
                 and isinstance(key[1], Real) and not isinstance(key[1], bool)
                 and np.isfinite(key[1]) and 0 <= key[1] < 10000, "invalid strategy/cost identity")
        path = validate_replay(supplied)
        _require(path[list(DATES)].equals(frame[list(DATES)]) and path.cost_bps.eq(key[1]).all(),
                 "path dates or cost differ from their declared identity")
        weights = path[[f"open_{asset}_weight" for asset in ASSETS]].to_numpy(float)
        fees = path.fees.to_numpy(float)
        _require(np.isfinite(weights).all() and (weights >= 0).all()
                 and np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12)
                 and np.isfinite(fees).all() and (fees >= 0).all() and (fees < 1).all(),
                 "invalid funded weights or paid fees")
        gross = (weights[:, :2] * returns).sum(axis=1)
        net = path.strategy_return.to_numpy(float)
        error = np.abs((1-fees)*(1+gross)-1-net)
        _require(error.max() <= tolerance, "recorded returns do not match held positions and current fees")
        checked[key] = (path, weights, fees, gross, net, error)
    costs = {key[1] for key in checked}
    names = {key[0] for key in checked}
    _require(all((name, cost) in checked for name in names for cost in costs)
             and benchmark in names, "strategy/cost inventory or explicit benchmark is incomplete")
    rows = []
    for window, bounds in windows.items():
        _require(isinstance(window, str) and isinstance(bounds, (tuple, list)) and len(bounds) == 2,
                 "invalid named window")
        start, end = map(pd.Timestamp, bounds)
        _require(not pd.isna(start) and not pd.isna(end) and start.tzinfo is None and end.tzinfo is None
                 and start == start.normalize() and end == end.normalize() and start <= end,
                 "window bounds require ordered session dates")
        mask = frame.signal_date.between(start, end).to_numpy()
        _require(mask.sum() >= 3, "attribution window needs at least three observations")
        rl, rx = returns[mask].T
        centered = rx-rx.mean()
        _require(centered @ centered > 0, "SOXX realized variance is zero")
        beta = float(centered @ (rl-rl.mean()) / (centered @ centered))
        residual = rl-beta*rx
        for (name, cost), (path, weights, fees, gross, net, error) in checked.items():
            _, bw, bf, bg, bn, _ = checked[(benchmark, cost)]
            dw = weights[mask, :2]-bw[mask, :2]
            exposure = beta*dw[:, 0]+dw[:, 1]
            components = np.column_stack([
                exposure.mean()*rx,
                (exposure-exposure.mean())*rx,
                dw[:, 0]*residual,
                -fees[mask]*(1+gross[mask])+bf[mask]*(1+bg[mask]),
            ])
            daily_error = float(np.max(np.abs(components.sum(axis=1)-(net[mask]-bn[mask]))))
            _require(daily_error <= 2*tolerance, "simple-return attribution fails accounting identity")
            gross_log = float((np.log1p(gross[mask])-np.log1p(bg[mask])).sum())
            fee_log = float((np.log1p(-fees[mask])-np.log1p(-bf[mask])).sum())
            net_log = float((np.log1p(net[mask])-np.log1p(bn[mask])).sum())
            _require(np.isfinite([gross_log, fee_log, net_log]).all(), "nonfinite log attribution")
            _require(abs(gross_log+fee_log-net_log) <= mask.sum()*4*tolerance,
                     "log attribution fails accounting identity")
            summed = components.sum(axis=0)*10000
            with np.errstate(over="ignore", invalid="ignore"):
                wealth = float(np.exp(net_log))
                strategy_beta = float(centered @ (net[mask]-net[mask].mean())/(centered @ centered))
            _require(np.isfinite([wealth, beta, strategy_beta, *summed]).all() and wealth > 0,
                     "nonrepresentable wealth or return attribution")
            rows.append({"strategy": name, "cost_bps": cost, "window": window,
                "sessions": int(mask.sum()), "relative_wealth": wealth,
                "net_log_excess": net_log, "fixed_positions_gross_log_excess": gross_log,
                "incremental_fee_log_drag": fee_log, "log_identity_error": gross_log+fee_log-net_log,
                "realized_soxl_beta": beta, "mean_soxl_weight": float(weights[mask, 0].mean()),
                "mean_soxx_weight": float(weights[mask, 1].mean()), "mean_cash_weight": float(weights[mask, 2].mean()),
                "mean_nominal_beta": float((3*weights[mask, 0]+weights[mask, 1]).mean()),
                "realized_strategy_beta": strategy_beta,
                "beta_level_simple_bp": summed[0], "beta_timing_simple_bp": summed[1],
                "leveraged_residual_simple_bp": summed[2], "fees_simple_bp": summed[3],
                "net_simple_difference_bp": float((net[mask]-bn[mask]).sum()*10000),
                "turnover": float(path.turnover.to_numpy(float)[mask].sum()),
                "maximum_accounting_error": float(error[mask].max()), "maximum_attribution_error": daily_error,
                "component_units": "sum of daily simple-return differences in bp, not compounded",
                "scope": "ex_post_fixed_realized_positions_no_new_forecast_or_native_provenance_certificate"})
    return pd.DataFrame(rows)

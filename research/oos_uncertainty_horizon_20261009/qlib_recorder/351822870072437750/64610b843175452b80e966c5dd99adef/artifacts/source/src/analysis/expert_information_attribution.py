"""Ex-post attribution of already reconciled Qlib portfolio evidence.

The caller loads market rows and portfolio paths from a Qlib Recorder and
binds their identities. This module does not fetch data, backtest, predict,
certify provenance, select a strategy, or authorize promotion.

``fees`` is cash paid divided by pretrade NAV. ``open_*_weight`` is the
post-fee funded allocation, normalized to post-fee NAV, including on HOLD
sessions. With zero cash yield, net return is (1-fees)*(1+weights@returns)-1.
This convention must not be confused with weights on pretrade NAV.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import ASSETS, DATES, _dates, validate_replay


OPEN_WEIGHTS = tuple(f"open_{asset}_weight" for asset in ASSETS)
SCOPE = "ex_post_descriptive_not_prediction_or_promotion_evidence"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _numeric(frame, columns, label):
    _require(not (set(columns) - set(frame)), f"{label} lacks required numeric fields")
    _require(all(pd.api.types.is_numeric_dtype(frame[name])
                 and not pd.api.types.is_bool_dtype(frame[name]) for name in columns),
             f"{label} fields must be numeric")
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    _require(np.isfinite(values).all(), f"{label} fields must be finite")
    return values


def _correlation(x, y):
    # Constant exposure differences have no defined correlation, not zero IC.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _slope(y, x):
    centered_x = x - x.mean()
    return float(centered_x @ (y - y.mean()) / (centered_x @ centered_x))


def attribute_expert_information(
    paths: Mapping[tuple[str, float], pd.DataFrame],
    market: pd.DataFrame,
    *,
    windows: Mapping[str, tuple[str, str]],
    candidate_name: str,
    reference_name: str,
    benchmark_name: str,
    leveraged_name: str,
    defensive_name: str,
    identity_tolerance: float = 1e-10,
) -> dict[str, pd.DataFrame]:
    """Return small structure, attribution and accounting tables.

    ``windows`` supplies inclusive signal-date bounds; every selected window
    must be contiguous and contain at least three complete observations.
    All supplied paths must match the full market DATES exactly, including
    paths not selected for the comparison. No inner join or row dropping is
    permitted. Every cost needs all five explicitly named strategies.

    Within each window, b = Cov(SOXL, SOXX)/Var(SOXX) is a realized OLS slope,
    and e = SOXL - b*SOXX retains the intercept. With d denoting candidate
    minus reference and B=b*d_w_SOXL+d_w_SOXX, the exact simple-return sum is:

      mean(B)*sum(SOXX) + sum((B-mean(B))*SOXX)
      + sum(d_w_SOXL*e) - sum(d_fee_drag),

    where fee_drag=fees*(1+gross_return). Components are cumulative sums in
    basis points, NOT compounded or annualized returns. Compounded relative
    wealth is reported separately. Estimated b and exposure/return
    correlations use realized outcomes; they cannot be strategy inputs or
    claims of out-of-sample information. Zero cash yield is explicit here.
    """
    _require(isinstance(identity_tolerance, Real) and not isinstance(identity_tolerance, bool)
             and np.isfinite(identity_tolerance) and identity_tolerance > 0,
             "identity tolerance must be finite and positive")
    _require(isinstance(market, pd.DataFrame) and not market.columns.has_duplicates,
             "market must have unique columns")
    frame = _dates(market)
    returns = _numeric(frame, ("soxl_return", "soxx_return"), "market")
    _require((returns > -1).all(), "market returns must exceed -1")
    _require(isinstance(paths, Mapping) and bool(paths), "paths must be a nonempty mapping")
    names = (candidate_name, reference_name, benchmark_name, leveraged_name, defensive_name)
    _require(all(isinstance(name, str) and name.strip() for name in names)
             and len(set(names)) == len(names), "strategy names must be unique nonempty strings")
    checked, accounting = {}, []
    for key, supplied in paths.items():
        _require(isinstance(key, tuple) and len(key) == 2, "path keys require strategy and cost")
        name, cost = key
        _require(isinstance(name, str) and name.strip() and isinstance(cost, Real)
                 and not isinstance(cost, bool) and np.isfinite(cost) and 0 <= cost < 10000,
                 "path key contains an invalid strategy or cost")
        _require(isinstance(supplied, pd.DataFrame) and not supplied.columns.has_duplicates,
                 "paths must have unique columns")
        path = validate_replay(supplied)
        _require(path.loc[:, list(DATES)].equals(frame.loc[:, list(DATES)]),
                 "path and market must have identical signal/trade/exit dates")
        weights = _numeric(path, OPEN_WEIGHTS, "allocation")
        fee = _numeric(path, ("fees",), "fees")[:, 0]
        declared_cost = _numeric(path, ("cost_bps",), "cost")[:, 0]
        _require((declared_cost == cost).all(), "path cost differs from mapping key")
        _require((weights >= 0).all() and np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12),
                 "post-fee weights must be a funded nonnegative simplex")
        _require(((fee >= 0) & (fee < 1)).all(), "fees must be a fraction of pretrade NAV")
        gross = (weights[:, :2] * returns).sum(axis=1)
        drag = fee * (1 + gross)
        error = float(np.max(np.abs(gross - drag - path.strategy_return.to_numpy(float))))
        _require(error <= identity_tolerance,
                 "net return does not match funded open weights, market returns and fees")
        checked[key] = (path, weights, drag)
        accounting.append({"strategy": name, "cost_bps": cost, "sessions": len(path),
            "maximum_net_reconstruction_error": error, "identity_tolerance": identity_tolerance,
            "cash_yield": 0., "provenance_verified_by_this_function": False,
            "evidence_scope": SCOPE})
    costs = sorted({cost for _, cost in checked})
    _require(all((name, cost) in checked for name in names for cost in costs),
             "each cost must contain all named strategies")
    _require(isinstance(windows, Mapping) and bool(windows), "windows must be explicit")
    masks = {}
    for name, bounds in windows.items():
        _require(isinstance(name, str) and name.strip() and isinstance(bounds, (tuple, list))
                 and len(bounds) == 2, "window requires a name and two signal-date bounds")
        try:
            start, end = (pd.Timestamp(value) for value in bounds)
        except (TypeError, ValueError) as error:
            raise ValueError("blocked: invalid window bounds") from error
        _require(not pd.isna(start) and not pd.isna(end) and start.tzinfo is None
                 and end.tzinfo is None and start == start.normalize() and end == end.normalize()
                 and start <= end, "window bounds must be ordered daily dates")
        mask = frame.signal_date.between(start, end).to_numpy()
        _require(mask.sum() >= 3, "attribution window requires at least three observations")
        _dates(frame.loc[mask])
        _require(np.var(returns[mask, 1]) > 0, "SOXX variance is zero in attribution window")
        masks[name] = mask

    structure, attribution = [], []
    for window, mask in masks.items():
        r_l, r_x = returns[mask].T
        beta = _slope(r_l, r_x)
        residual = r_l - beta * r_x
        pair_corr = _correlation(r_l, r_x)
        shared = {"window": window, "sessions": int(mask.sum()),
            "start_signal": str(frame.loc[mask, "signal_date"].iloc[0].date()),
            "last_exit": str(frame.loc[mask, "exit_date"].iloc[-1].date()),
            "realized_soxl_beta_to_soxx": beta, "evidence_scope": SCOPE}
        for cost in costs:
            candidate, cw, cd = checked[(candidate_name, cost)]
            reference, rw, rd = checked[(reference_name, cost)]
            benchmark = checked[(benchmark_name, cost)][0].strategy_return.to_numpy(float)[mask]
            lever = checked[(leveraged_name, cost)][0].strategy_return.to_numpy(float)[mask] - benchmark
            defensive = checked[(defensive_name, cost)][0].strategy_return.to_numpy(float)[mask] - benchmark
            experts = np.column_stack([lever, defensive])
            centered = experts - experts.mean(axis=0)
            eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(centered))
            cr = candidate.strategy_return.to_numpy(float)[mask]
            rr = reference.strategy_return.to_numpy(float)[mask]
            candidate_corr = _correlation(cr, r_x)
            structure.append({**shared, "cost_bps": cost,
                "leveraged_expert": leveraged_name, "defensive_expert": defensive_name,
                "benchmark": benchmark_name, "soxl_soxx_correlation": pair_corr,
                "soxl_regression_r_squared": None if pair_corr is None else pair_corr ** 2,
                "leveraged_residual_daily_mean_bp": float(residual.mean() * 10000),
                "leveraged_residual_daily_std_bp": float(residual.std() * 10000),
                "expert_active_return_correlation": _correlation(lever, defensive),
                "expert_active_first_pc_variance_share": float(eigenvalues[-1] / eigenvalues.sum())
                    if eigenvalues.sum() > 0 else None,
                "candidate_beta_to_soxx": _slope(cr, r_x),
                "reference_beta_to_soxx": _slope(rr, r_x),
                "candidate_regression_r_squared": None if candidate_corr is None else candidate_corr ** 2})
            dw_l, dw_x = (cw[mask, :2] - rw[mask, :2]).T
            delta_beta = beta * dw_l + dw_x
            beta_level = np.full(len(r_x), delta_beta.mean()) * r_x
            beta_timing = (delta_beta - delta_beta.mean()) * r_x
            residual_term = dw_l * residual
            fee_term = -(cd[mask] - rd[mask])
            difference = cr - rr
            reconstructed = beta_level + beta_timing + residual_term + fee_term
            maximum_error = float(np.max(np.abs(reconstructed - difference)))
            _require(maximum_error <= 2 * identity_tolerance,
                     "attribution components do not reconstruct net return differences")
            relative_log = float((np.log1p(cr) - np.log1p(rr)).sum())
            _require(np.isfinite(np.exp(relative_log)), "relative wealth is not finite")
            attribution.append({**shared, "cost_bps": cost, "candidate": candidate_name,
                "reference": reference_name, "component_unit": "summed_daily_simple_return_bp_not_compounded",
                "mean_beta_exposure_component_bp": float(beta_level.sum() * 10000),
                "beta_timing_component_bp": float(beta_timing.sum() * 10000),
                "leveraged_residual_component_bp": float(residual_term.sum() * 10000),
                "fee_difference_component_bp": float(fee_term.sum() * 10000),
                "net_difference_sum_bp": float(difference.sum() * 10000),
                "maximum_daily_identity_error": maximum_error,
                "compounded_relative_return_bp": float(np.expm1(relative_log) * 10000),
                "mean_delta_soxl_weight_pp": float(dw_l.mean() * 100),
                "minimum_delta_soxl_weight_pp": float(dw_l.min() * 100),
                "maximum_delta_soxl_weight_pp": float(dw_l.max() * 100),
                "mean_delta_beta": float(delta_beta.mean()),
                "delta_beta_future_soxx_return_correlation": _correlation(delta_beta, r_x),
                "delta_soxl_weight_future_residual_correlation": _correlation(dw_l, residual),
                "candidate_turnover": float(candidate.turnover.to_numpy(float)[mask].sum()),
                "reference_turnover": float(reference.turnover.to_numpy(float)[mask].sum()),
                "candidate_sum_fee_rates_bp": float(candidate.fees.to_numpy(float)[mask].sum() * 10000),
                "reference_sum_fee_rates_bp": float(reference.fees.to_numpy(float)[mask].sum() * 10000)})
    return {"structure": pd.DataFrame(structure), "attribution": pd.DataFrame(attribution),
            "accounting": pd.DataFrame(accounting)}

"""Benchmark-conversion terminal valuation for a receding-horizon planner.

The transformation lets the existing funded solver value post-entry holdings
as if they were converted to SOXX at the forecast horizon. It neither executes
that conversion nor charges its hypothetical fees to an actual account. It is
not an exact multi-period optimizer or an instruction to liquidate periodically.

Asset order is SOXL, SOXX, cash; mean/error-covariance order is SOXL, SOXX.
Cash earns zero. One common proportional rate applies to buying and selling;
there are no minimum fees, taxes, impact, shorts or account borrowing here.
"""
from __future__ import annotations

from numbers import Real

import numpy as np

from src.data.convex_allocation import _psd_roundoff


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _cost(cost_bps):
    _require(isinstance(cost_bps, Real) and not isinstance(cost_bps, (bool, np.bool_))
             and np.isfinite(cost_bps) and 0 <= cost_bps < 10000,
             "cost_bps must be finite in [0,10000)")
    return float(cost_bps) / 10000


def _array(values, name):
    try:
        raw = np.asarray(values)
        _require(raw.dtype.kind in "fiu", f"{name} requires real numeric values")
        result = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"blocked: invalid {name}") from error
    _require(np.isfinite(result).all(), f"{name} must be finite")
    return result


def _means(values):
    means = _array(values, "horizon means")
    _require(means.shape == (2,) or (means.ndim == 2 and means.shape[0] > 0 and means.shape[1] == 2),
             "horizon means require shape (2,) or (n,2)")
    _require((means > -1).all(), "horizon simple-return means must exceed -1")
    return means


def direct_terminal_value(post_fee_values, mu, cost_bps):
    """Value a fixed holding after hypothetical sell-SOXL/buy-SOXX trades.

    ``post_fee_values`` are actual currency values, or fractions of pretrade
    NAV, after funding the entry trade; they do not need to sum to one. Input
    shapes must match: values (3,) with means (2,), or (n,3) with (n,2).
    This function accepts point forecasts or individual realized scenarios.
    Existing SOXX is retained. SOXL sale proceeds and cash fund new SOXX
    purchases including their proportional fee.
    """
    cost, means = _cost(cost_bps), _means(mu)
    values = _array(post_fee_values, "post-fee values")
    _require(values.shape == (*means.shape[:-1], 3), "post-fee values and means have different shapes")
    _require((values >= 0).all() and (values.sum(axis=-1) > 0).all(),
             "post-fee values must be nonnegative with positive total NAV")
    try:
        with np.errstate(over="raise", invalid="raise"):
            retained_soxx = values[..., 1] * (1 + means[..., 1])
            sale_proceeds = (1 - cost) * values[..., 0] * (1 + means[..., 0])
            new_soxx_value = (values[..., 2] + sale_proceeds) / (1 + cost)
            terminal = retained_soxx + new_soxx_value
    except FloatingPointError as error:
        raise ValueError("blocked: terminal value is not representable") from error
    _require(np.isfinite(terminal).all() and (terminal > 0).all(), "terminal value must be finite and positive")
    return float(terminal) if means.ndim == 1 else terminal


def transform_benchmark_terminal_inputs(mu, V, *, cost_bps):
    """Transform horizon means and their estimation-error covariance only.

    For post-entry values x=(xL,xX,xC), c=cost_bps/10000 and q=1/(1+c):

      E[terminal value] = xX*(1+muX) + (xC+(1-c)*xL*(1+muL))/(1+c)
                        = q*(sum(x) + transformed_mu @ x[:2]).

    transformed_mu = diag(1-c,1+c) @ mu + (-c,c), with covariance
    transformed_V = D @ V @ D. V is conditional mean-estimation uncertainty,
    NOT the daily market covariance used for risk feasibility. For a solver's
    active uncertainty penalty P in these transformed units, q*objective is
    E[terminal value]-q*P. This merely normalizes a planning value function.

    Inputs support one (2,)/(2,2) pair or aligned (n,2)/(n,2,2) batches.
    Material asymmetry or negative eigenvalues fail closed. The existing
    project helper repairs floating-point PSD/symmetry roundoff only and all
    such adjustments are reported; no economic covariance floor is added.
    """
    cost, means = _cost(cost_bps), _means(mu)
    covariance = _array(V, "mean-error covariance")
    _require(covariance.shape == (*means.shape[:-1], 2, 2),
             "mean-error covariance must align exactly with horizon means")
    single = means.ndim == 1
    matrices = covariance[None, ...] if single else covariance
    diagonal = np.array([1 - cost, 1 + cost])
    repaired, input_audits, output_audits = [], [], []
    for matrix in matrices:
        checked, before = _psd_roundoff(matrix)
        try:
            with np.errstate(over="raise", invalid="raise"):
                transformed = diagonal[:, None] * checked * diagonal[None, :]
        except FloatingPointError as error:
            raise ValueError("blocked: transformed covariance is not representable") from error
        transformed, after = _psd_roundoff(transformed)
        repaired.append(transformed)
        input_audits.append(before)
        output_audits.append(after)
    try:
        with np.errstate(over="raise", invalid="raise"):
            # Expanded form avoids cancellation and is exact at zero cost.
            effective_means = means * diagonal + np.array([-cost, cost])
    except FloatingPointError as error:
        raise ValueError("blocked: transformed means are not representable") from error
    _require(np.isfinite(effective_means).all() and (effective_means > -1).all(),
             "transformed means must be finite and exceed -1")
    transformed_covariance = np.stack(repaired)
    audits = input_audits + output_audits
    return {"transformed_mu": effective_means,
        "transformed_covariance": transformed_covariance[0] if single else transformed_covariance,
        "wealth_scale": 1 / (1 + cost), "audit": {
            "mechanism": "receding_horizon_terminal_conversion_value_to_SOXX",
            "cost_bps": float(cost_bps), "wealth_scale": 1 / (1 + cost),
            "response_order": ["SOXL", "SOXX"], "asset_order": ["SOXL", "SOXX", "cash"],
            "rows": len(matrices), "covariance_diagonal_transform": diagonal.tolist(),
            "mean_formula": "mu_eff=((1-c)*mu_SOXL-c,(1+c)*mu_SOXX+c)",
            "covariance_formula": "D @ conditional_mean_error_covariance @ D",
            "scaled_objective": "q*objective=expected_terminal_value-q*transformed_active_uncertainty_penalty",
            "daily_risk_covariance_changed": False, "cash_yield": 0.,
            "input_psd_roundoff": input_audits, "output_psd_roundoff": output_audits,
            "maximum_covariance_roundoff_adjustment": max(a["maximum_entry_adjustment"] for a in audits),
            "future_conversion_fees_executed": False, "literal_terminal_holdings_enforced": False,
            "exact_multi_period_optimality_claim": False,
            "limitation": "Planning assumption only; no future fee debit, execution change or performance guarantee."}}

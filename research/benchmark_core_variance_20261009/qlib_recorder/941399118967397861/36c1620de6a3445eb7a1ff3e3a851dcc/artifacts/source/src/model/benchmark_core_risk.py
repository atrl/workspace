"""Pure two-asset risk representations, with no mean or return prediction.

Asset order is SOXL, SOXX, cash. Budgets and covariance entries are in the same
one-session return-variance units. The caller supplies past-available inputs;
this mapping has no data source, training, clock, optimizer or future returns.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _numeric(value, name):
    try:
        raw = np.asarray(value)
        _require(raw.dtype.kind in "fiu", name + " requires real numeric values")
        result = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("blocked: invalid " + name) from error
    _require(np.isfinite(result).all(), name + " must be finite")
    return result


def _decimal(value):
    return Decimal.from_float(float(value))


def _checked_covariance(matrix):
    """Keep variances exact; repair only tiny symmetry/correlation roundoff.

    The off-diagonal bound is computed without multiplying float64 variances,
    so valid very small or large covariances do not underflow or overflow.
    A representable cross-covariance is rounded toward the PSD interval when
    necessary. No diagonal floor or shrinkage is introduced.
    """
    L, X = _decimal(matrix[0, 0]), _decimal(matrix[1, 1])
    _require(L >= 0 and X >= 0, "risk covariance variances must be nonnegative")
    upper, lower = _decimal(matrix[0, 1]), _decimal(matrix[1, 0])
    product = L * X
    bound = product.sqrt()
    tolerance = _decimal(128 * np.finfo(float).eps) * max(bound, abs(upper), abs(lower))
    _require(abs(upper - lower) <= tolerance, "risk covariance is materially asymmetric")
    cross = (upper + lower) / 2
    _require(abs(cross) <= bound + tolerance, "risk covariance is not positive semidefinite")
    if product == 0:
        _require(cross == 0, "zero variance requires zero cross-covariance")
    value = float(cross)
    # A PSD boundary may fall between adjacent float64 values. Verify the
    # chosen stored value by products rather than an inexact square root.
    repairs = 0
    while Fraction.from_float(value) ** 2 > Fraction(L) * Fraction(X):
        value = float(np.nextafter(value, 0.))
        repairs += 1
        _require(repairs <= 256, "PSD roundoff correction exceeded the numerical envelope")
    checked = np.array([[float(L), value], [value, float(X)]])
    return checked, (L, _decimal(value), X), {
        "maximum_covariance_roundoff_adjustment": float(np.max(np.abs(checked - matrix))),
        "symmetry_error": float(abs(upper - lower)), "cross_psd_rounding_steps": repairs,
        "diagonal_variances_changed": False, "economic_covariance_floor_added": False,
        "psd_policy": "nonnegative exact diagonals; symmetry/cross-bound roundoff only at128float64eps"}


def _weights(amount, mode):
    if mode == "soxl_cash":
        return np.array([amount, 0., 1. - amount])
    if mode == "soxx_cash":
        return np.array([0., amount, 1. - amount])
    return np.array([amount, 1. - amount, 0.])


def _variance(weights, covariance):
    L, C, X = map(Fraction, covariance)
    a, b = (Fraction.from_float(float(value)) for value in weights[:2])
    return L * a * a + 2 * C * a * b + X * b * b


def _funded_feasible(amount, mode, covariance, budget):
    """Round the continuous solution toward feasibility, never raise risk."""
    original = float(amount)
    budget = Fraction(budget)
    value = original
    weights = _weights(value, mode)
    variance = _variance(weights, covariance)
    steps = 0
    while variance > budget and steps < 4:
        value = float(np.nextafter(value, 0.))
        weights = _weights(value, mode)
        variance = _variance(weights, covariance)
        steps += 1
    bisections = 0
    if variance > budget:
        # Zero exposure in the chosen coordinate is feasible in every branch
        # reaching here. A convex quadratic with a feasible left endpoint has
        # one right boundary, even if variance initially decreases.
        low, high = 0., value
        _require(_variance(_weights(low, mode), covariance) <= budget,
                 "numerical risk root has no feasible left endpoint")
        for _ in range(128):
            middle = low + (high - low) / 2
            if middle == low or middle == high:
                break
            if _variance(_weights(middle, mode), covariance) <= budget:
                low = middle
            else:
                high = middle
            bisections += 1
        value = low
        weights = _weights(value, mode)
        variance = _variance(weights, covariance)
    _require(variance >= 0 and variance <= budget, "returned target is not risk feasible")
    _require((weights >= 0).all() and abs(float(weights.sum()) - 1.) <= 4 * np.finfo(float).eps,
             "returned allocation is not a funded simplex")
    return weights, variance, {"exposure_roundoff_change": original - value,
        "feasibility_nextafter_steps": steps, "feasibility_bisection_steps": bisections}


def _one(matrix, budget, representation):
    with localcontext() as context:
        context.prec = 80
        checked, covariance, audit = _checked_covariance(matrix)
        L, C, X = covariance
        B = _decimal(budget)
        if representation == "lever_cash":
            amount = Decimal(1) if L <= B else (B / L).sqrt()
            mode, branch, targeted = "soxl_cash", "soxl_cash_risk_fraction", L >= B
            if L < B:
                branch = "full_soxl_budget_underused"
        elif X >= B:
            amount = Decimal(1) if X == B else (B / X).sqrt()
            mode, branch, targeted = "soxx_cash", "soxx_cash_risk_fraction", True
            if X == B:
                branch = "exact_soxx_at_budget"
        elif L <= B:
            amount, mode, targeted = Decimal(1), "soxl_cash", L == B
            branch = "full_soxl_at_budget" if L == B else "full_soxl_budget_underused"
        else:
            # V(a)=A*a^2+b*a+X, V(0)<B<V(1). PSD makes A nonnegative,
            # but b can be negative: do not assume variance is monotone.
            A, b, d = L + X - 2 * C, 2 * (C - X), B - X
            _require(A >= 0, "PSD quadratic curvature is negative")
            if A == 0:
                _require(b > 0, "degenerate risk boundary has no finite root")
                amount = d / b
            else:
                discriminant = (b * b + 4 * A * d).sqrt()
                amount = 2 * d / (b + discriminant) if b >= 0 else (discriminant - b) / (2 * A)
            _require(0 <= amount <= 1, "maximum core risk root is outside the simplex")
            mode, branch, targeted = "soxl_soxx", "maximum_soxl_core_root", True
            audit["risk_curve_initially_decreases"] = b < 0
            audit["continuous_maximum_soxl_weight"] = float(amount)
        weights, variance, correction = _funded_feasible(amount, mode, covariance, B)
        audit.update(correction)
        audit.update({"branch": branch, "certified_risk_feasible": True,
            "certificate": "exact rational evaluation of returned float64 weights and checked covariance versus the input float64 budget",
            "root_arithmetic": "80-digit decimal quadratic root followed by conservative float64 feasibility adjustment",
            "variance_exact_ratio": f"{variance.numerator}/{variance.denominator}", "risk_budget_decimal": str(B),
            "budget_targeted": bool(targeted), "no_return_mean_or_future_outcome_used": True})
        return {"weights": weights, "covariance": checked, "portfolio_variance": float(variance),
            "risk_budget_variance": float(B), "budget_used_fraction": float(variance / Fraction(B)),
            "budget_unspent_variance": float(Fraction(B) - variance), "budget_targeted": bool(targeted),
            "branch": branch, "audit": audit}


def benchmark_core_allocation(covariance, risk_budget, *, representation):
    """Map past risk inputs to ``core`` or its matched ``lever_cash`` control.

    Covariance is (2,2) or (n,2,2). Budget is positive and is a scalar or an
    aligned (n,) vector for batches. Each row is independent. Core first uses
    SOXX/cash when its variance meets/exceeds B; exact equality returns pure
    SOXX. Otherwise it maximizes SOXL within the SOXL/SOXX fully invested mix,
    capping at pure SOXL and reporting any unused budget. No 3x-beta assumption
    or statistical guarantee of realized returns is made.
    """
    _require(isinstance(representation, str) and representation in ("core", "lever_cash"), "unknown risk representation")
    matrices = _numeric(covariance, "risk covariance")
    single = matrices.shape == (2, 2)
    _require(single or (matrices.ndim == 3 and matrices.shape[0] > 0 and matrices.shape[1:] == (2, 2)),
             "risk covariance must have shape(2,2) or(n,2,2)")
    if single:
        matrices = matrices[None, ...]
    budgets = _numeric(risk_budget, "risk budget")
    _require(budgets.ndim == 0 or (not single and budgets.shape == (len(matrices),)),
             "risk budgets must align with covariance rows")
    budgets = np.full(len(matrices), float(budgets)) if budgets.ndim == 0 else budgets
    _require((budgets > 0).all(), "risk budgets must be strictly positive")
    rows = [_one(matrix, budget, representation) for matrix, budget in zip(matrices, budgets)]
    if single:
        return {"representation": representation, **rows[0]}
    return {"representation": representation,
        **{key: np.stack([row[key] for row in rows]) for key in ("weights", "covariance")},
        **{key: np.array([row[key] for row in rows]) for key in
           ("portfolio_variance", "risk_budget_variance", "budget_used_fraction", "budget_unspent_variance", "budget_targeted")},
        "branch": [row["branch"] for row in rows], "audit": {
            "rows": len(rows), "row_audits": [row["audit"] for row in rows],
            "all_certified_risk_feasible": True, "risk_prediction_added": False,
            "maximum_covariance_roundoff_adjustment": max(row["audit"]["maximum_covariance_roundoff_adjustment"] for row in rows)}}

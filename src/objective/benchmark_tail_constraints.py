"""Complete-path benchmark tail losses and daily log-multiplier adjoints.

Inputs are already fee-funded chronological equity multipliers in log form.
The module neither constructs a portfolio path nor fits or selects a policy.
Risk values match the project's initial-capital drawdown and exact fractional
5% lower-return-tail convention. Ties use the explicitly declared subgradients;
neither these choices nor downstream truncated state gradients certify an
optimizer or the risk of a future path.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from numbers import Integral, Real

import numpy as np

TAIL_FRACTION = .05
RISK_ORDER = ("drawdown_loss", "cvar5_loss")
REFERENCE_SCHEMA = "benchmark_tail_reference_v1"
CVAR_REFERENCE = "https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf"


class TailConstraintError(ValueError):
    """A nonphysical path, invalid reference, or nonfinite risk calculation."""


def _vector(value, name):
    try:
        original = np.asarray(value)
        if np.iscomplexobj(original) or original.dtype == np.bool_:
            raise ValueError("complex or boolean values")
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise TailConstraintError(f"blocked: {name} must contain real numerical values") from error
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise TailConstraintError(f"blocked: {name} must be a nonempty finite one-dimensional path")
    return result


def evaluate_risk(log_multipliers):
    """Return positive drawdown loss, signed CVaR loss, and their log adjoints.

    NAV index0 is the initial capital. A selected peak/trough pair (s,t)
    differentiates log-multiplier array entries ``[s:t]``. At drawdown ties the
    first worst trough and its earliest prefix peak are used; drawdown0 returns
    a zero subgradient. CVaR cutoff ties share the remaining tail mass equally.
    CVaR loss can be negative and is never clamped.
    """
    logs = _vector(log_multipliers, "log multipliers")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        multipliers = np.exp(logs)
        returns = np.expm1(logs)
        log_nav = np.r_[0., np.cumsum(logs)]
        nav = np.exp(log_nav)
    if (not np.isfinite(multipliers).all() or (multipliers <= 0).any()
            or not np.isfinite(returns).all() or (returns <= -1).any()
            or not np.isfinite(log_nav).all() or not np.isfinite(nav).all() or (nav <= 0).any()):
        raise TailConstraintError("blocked: risk path has nonphysical returns or nonfinite/nonpositive NAV")

    peak_log_nav = np.maximum.accumulate(log_nav)
    underwater = log_nav - peak_log_nav
    trough = int(np.argmin(underwater))  # First minimum includes initial NAV.
    peak = int(np.flatnonzero(log_nav[:trough + 1] == peak_log_nav[trough])[0])
    drawdown = float(-np.expm1(underwater[trough]))
    drawdown_adjoint = np.zeros(len(logs), dtype=np.float64)
    if drawdown > 0:
        drawdown_adjoint[peak:trough] = -np.exp(underwater[trough])
    else:
        drawdown = 0.

    ordered = np.sort(returns)
    mass = len(logs) * TAIL_FRACTION
    whole = int(np.floor(mass))
    fraction = mass - whole
    # Compute the value with precisely the existing fractional order-statistic
    # convention; do not replace it with a quantile-based conditional mean.
    cvar_return = float((ordered[:whole].sum() + fraction * ordered[whole]) / mass)
    boundary = float(ordered[int(np.ceil(mass)) - 1])
    below, tied = returns < boundary, returns == boundary
    remaining_mass = mass - int(below.sum())
    tail_weights = np.zeros(len(logs), dtype=np.float64)
    tail_weights[below] = 1 / mass
    tail_weights[tied] = remaining_mass / (int(tied.sum()) * mass)
    cvar_adjoint = -tail_weights * multipliers
    cvar_loss = -cvar_return
    if (not np.isfinite([drawdown, cvar_loss]).all()
            or not np.isfinite(drawdown_adjoint).all() or not np.isfinite(cvar_adjoint).all()
            or not np.isfinite(tail_weights).all() or (tail_weights < 0).any()
            or abs(float(tail_weights.sum()) - 1.) > 64 * np.finfo(float).eps):
        raise TailConstraintError("blocked: nonfinite or invalid complete-path tail risk")
    return {
        "sessions": len(logs), "drawdown_loss": drawdown, "cvar5_loss": cvar_loss,
        "max_drawdown_pct": -100 * drawdown, "cvar5_pct": -100 * cvar_loss,
        "drawdown_adjoint": drawdown_adjoint, "cvar5_adjoint": cvar_adjoint,
        "tail_weights": tail_weights, "tail_mass": mass, "tail_boundary_return": boundary,
        "tail_boundary_count": int(tied.sum()), "tail_remaining_mass": remaining_mass,
        "peak_index": peak, "trough_index": trough,
        "drawdown_tie_rule": "earliest worst trough, earliest prefix maximum; zero at D=0",
        "cvar_tie_rule": "remaining exact tail mass shared equally among boundary ties",
        "log_multipliers_sha256": hashlib.sha256(logs.tobytes()).hexdigest(),
        "adjoint_units": "risk_loss_fraction_per_daily_log_net_equity_multiplier",
    }


def reference_risk(log_multipliers):
    """Freeze one cost's benchmark normalizers; no zero-risk epsilon budget."""
    risk = evaluate_risk(log_multipliers)
    normalizers = np.array([risk[name] for name in RISK_ORDER], dtype=np.float64)
    if not np.isfinite(normalizers).all() or (normalizers <= 0).any():
        raise TailConstraintError("blocked: benchmark tail-risk normalizers must be strictly positive and finite")
    normalizers.setflags(write=False)
    return {**risk, "schema_version": REFERENCE_SCHEMA, "normalizers": normalizers,
            "normalizer_order": RISK_ORDER, "normalization_changes_raw_feasibility": False}


def _reference_values(reference, sessions):
    if not isinstance(reference, Mapping) or reference.get("schema_version") != REFERENCE_SCHEMA:
        raise TailConstraintError("blocked: expected an explicit benchmark tail-risk reference")
    try:
        normalizers = np.asarray(reference["normalizers"], dtype=np.float64)
        declared = np.array([reference[name] for name in RISK_ORDER], dtype=np.float64)
        order = tuple(reference.get("normalizer_order", ()))
    except (KeyError, ValueError, TypeError) as error:
        raise TailConstraintError("blocked: incomplete benchmark risk reference") from error
    count = reference.get("sessions")
    if (not isinstance(count, Integral) or isinstance(count, (bool, np.bool_)) or count != sessions or order != RISK_ORDER
            or normalizers.shape != (2,) or not np.isfinite(normalizers).all() or (normalizers <= 0).any()
            or not np.array_equal(normalizers, declared)):
        raise TailConstraintError("blocked: benchmark reference coverage, order or positive normalizers differ")
    return normalizers


def evaluate_constraints(log_multipliers, reference, *, risk_tolerance_percentage_points=1e-10):
    """Compare one full policy path with one fixed, same-cost benchmark.

    All gaps have the loss direction: positive is worse. Raw feasibility uses
    percentage points only; the normalized violations/adjoints condition the
    optimizer and never supply a different financial tolerance.
    """
    tolerance = risk_tolerance_percentage_points
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance < 0:
        raise TailConstraintError("blocked: invalid raw risk tolerance in percentage points")
    risk = evaluate_risk(log_multipliers)
    normalizers = _reference_values(reference, risk["sessions"])
    gaps = np.array([risk[name] for name in RISK_ORDER]) - normalizers
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        percentage_point_gaps = 100 * gaps
        normalized = gaps / normalizers
        adjoints = np.stack((risk["drawdown_adjoint"], risk["cvar5_adjoint"])) / normalizers[:, None]
    if not np.isfinite(percentage_point_gaps).all() or not np.isfinite(normalized).all() or not np.isfinite(adjoints).all():
        raise TailConstraintError("blocked: nonfinite normalized benchmark risk constraints")
    passed = percentage_point_gaps <= tolerance
    return {"risk": risk, "normalizers": normalizers.copy(), "risk_order": RISK_ORDER,
            "raw_loss_gaps": gaps, "raw_percentage_point_gaps": percentage_point_gaps,
            "normalized_g": normalized, "normalized_adjoints": adjoints,
            "risk_pass": passed, "feasible": bool(passed.all()),
            "risk_tolerance_percentage_points": float(tolerance)}


def combined_daily_adjoints(objective_adjoints, constraints, multipliers):
    """Maximization adjoints F' - lambda*g' for one cost, including inactive F.

    The caller orders costs10 then25 and updates the four dual variables only
    after its single complete primal update. This helper never clips, projects,
    updates, or discards the constraints of an inactive objective-cost branch.
    """
    objective = _vector(objective_adjoints, "objective adjoints")
    dual = _vector(multipliers, "risk multipliers")
    try:
        adjoints = np.asarray(constraints["normalized_adjoints"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise TailConstraintError("blocked: missing normalized risk adjoints") from error
    if dual.shape != (2,) or (dual < 0).any() or adjoints.shape != (2, len(objective)) or not np.isfinite(adjoints).all():
        raise TailConstraintError("blocked: invalid risk multiplier order, signs or adjoint coverage")
    with np.errstate(over="ignore", invalid="ignore"):
        combined = objective - dual @ adjoints
    if not np.isfinite(combined).all():
        raise TailConstraintError("blocked: nonfinite combined objective/risk adjoints")
    return combined

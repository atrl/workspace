"""Single frozen action-risk quantile fit conditional on a saved state baseline."""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from src.model.stage1_on_policy_critics import (
    FEATURE_COUNT, RIDGE, SOLVER_OPTIONS, TAU, WeightedStateScaler,
    _action_linear_design, _data, _psd_raw_used, _quadratic_design, _vector,
)


@dataclass(frozen=True)
class FittedConditionalActionRisk:
    scaler: WeightedStateScaler
    c0: np.ndarray
    C: np.ndarray
    H_raw: np.ndarray
    H_used: np.ndarray
    objective: float
    solver_status: str
    solver_iterations: int

    def correction(self, z, u):
        x = self.scaler.transform(z)
        action = np.asarray(u, float)
        if action.shape != (len(x), 2) or not np.isfinite(action).all():
            raise ValueError("physical action must be finite N x 2")
        linear = np.sum(action * (self.c0 + x @ self.C.T), axis=1)
        quadratic = .5 * np.einsum("ni,ij,nj->n", action, self.H_used, action)
        return linear + quadratic

    def predict(self, z, u, saved_state_raw, received_drawdown):
        baseline = _vector(saved_state_raw, len(z), "saved state raw risk")
        floor = _vector(received_drawdown, len(z), "received drawdown")
        if (floor < 0).any() or (floor > 1).any():
            raise ValueError("received drawdown outside [0,1]")
        return np.maximum(baseline + self.correction(z, u), floor)


def fit_conditional_action_risk(z, u, risk, saved_state_raw, weights,
                                *, saved_scaler: WeightedStateScaler):
    """One tau=.95 CLARABEL fit; zero physical action has zero correction."""
    x, action, absolute, weight = _data(z, u, risk, weights)
    if (absolute < 0).any() or (absolute > 1).any():
        raise ValueError("absolute segment risk outside [0,1]")
    baseline = _vector(saved_state_raw, len(x), "saved state raw risk")
    standardized = saved_scaler.transform(x)
    if standardized.shape != (len(x), FEATURE_COUNT):
        raise ValueError("saved state scaler does not match frozen feature count")
    c0 = cp.Variable(2)
    C = cp.Variable((2, FEATURE_COUNT))
    H = cp.Variable((2, 2), PSD=True)
    correction = (action @ c0
                  + _action_linear_design(standardized, action) @ cp.vec(C, order="C")
                  + _quadratic_design(action) @ cp.vec(H, order="C"))
    residual = absolute - baseline - correction
    pinball = cp.maximum(TAU * residual, (TAU - 1.) * residual)
    penalty = RIDGE * (cp.sum_squares(c0) + cp.sum_squares(C) + cp.sum_squares(H))
    problem = cp.Problem(cp.Minimize(cp.sum(cp.multiply(weight, pinball)) + penalty))
    value = problem.solve(**SOLVER_OPTIONS)
    if problem.status != cp.OPTIMAL or not np.isfinite(value):
        raise RuntimeError("conditional action risk fit did not reach OPTIMAL")
    raw, used, _, _ = _psd_raw_used(H.value)
    return FittedConditionalActionRisk(
        saved_scaler, np.asarray(c0.value).copy(), np.asarray(C.value).copy(),
        raw, used, float(value), problem.status, int(problem.solver_stats.num_iters),
    )

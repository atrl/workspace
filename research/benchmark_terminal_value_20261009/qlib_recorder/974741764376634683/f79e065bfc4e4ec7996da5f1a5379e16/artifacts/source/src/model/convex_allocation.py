"""Cash-funded convex allocation with an explicit, feasible SOXX reference.

The forecast horizon describes expected asset returns, not a holding lock.
The same problem is re-solved on actual holdings each decision. Future trades
are not assumed in this rolling-planning approximation. Forecast estimation
and causal covariance/mean-error estimation belong to the Qlib workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter
from typing import Any

import cvxpy as cp
import numpy as np

from src.backtest.position_accounting import settle_target_session
from src.workflow.stage1_economic_action import fee_affordable_target


BENCHMARK_RISKY_WEIGHTS = np.array([0., 1.])


def _numerical_certificate(solution, data: dict, *, tolerance: float) -> dict[str, Any]:
    """Inspect the public raw Clarabel result and unscaled SOCP data.

    This is a finite-tolerance numerical certificate, not an exact-arithmetic
    theorem bound. In particular, the dual wealth bound is approximate when
    stationarity or cone membership have nonzero numerical residuals.
    """
    required = ("x", "z", "s", "obj_val", "obj_val_dual", "r_prim", "r_dual", "status")
    if any(not hasattr(solution, name) or getattr(solution, name) is None for name in required):
        raise ValueError("missing raw Clarabel numerical certificate")
    primal = np.asarray(solution.x, dtype=float)
    dual = np.asarray(solution.z, dtype=float)
    slack = np.asarray(solution.s, dtype=float)
    matrix, rhs, linear, dims = data["A"], np.asarray(data["b"]), np.asarray(data["c"]), data["dims"]
    if (primal.shape != linear.shape or dual.shape != rhs.shape or slack.shape != rhs.shape
            or matrix.shape != (len(rhs), len(linear))):
        raise ValueError("unexpected SOCP numerical certificate dimensions")
    values = np.r_[primal, dual, slack, rhs, linear, matrix.data,
                   solution.obj_val, solution.obj_val_dual, solution.r_prim, solution.r_dual]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite raw Clarabel numerical certificate")
    quadratic = data.get("P")
    if quadratic is not None and np.any(quadratic.data != 0):
        raise ValueError("numerical certificate only supports the registered linear-objective SOCP")
    if getattr(dims, "exp", 0) or getattr(dims, "psd", []) or getattr(dims, "p3d", []):
        raise ValueError("unregistered cone in allocation numerical certificate")
    offset = int(dims.zero)
    primal_cone = float(np.abs(slack[:offset]).max(initial=0.))
    dual_cone = 0.  # Equality-cone duals are unrestricted.
    nonnegative = int(dims.nonneg)
    primal_cone = max(primal_cone, float(np.maximum(-slack[offset:offset+nonnegative], 0).max(initial=0.)))
    dual_cone = max(dual_cone, float(np.maximum(-dual[offset:offset+nonnegative], 0).max(initial=0.)))
    offset += nonnegative
    for length in dims.soc:
        length = int(length)
        primal_cone = max(primal_cone, float(max(0., np.linalg.norm(slack[offset+1:offset+length]) - slack[offset])))
        dual_cone = max(dual_cone, float(max(0., np.linalg.norm(dual[offset+1:offset+length]) - dual[offset])))
        offset += length
    if offset != len(rhs):
        raise ValueError("SOCP cone dimensions do not cover the numerical certificate")
    primal_objective = float(linear @ primal)
    dual_objective = float(-rhs @ dual)
    checks = {
        "primal_residual_abs": float(np.abs(matrix @ primal + slack - rhs).max(initial=0.)),
        "dual_residual_abs": float(np.abs(matrix.T @ dual + linear).max(initial=0.)),
        "primal_cone_violation": primal_cone, "dual_cone_violation": dual_cone,
        "absolute_duality_gap": abs(primal_objective - dual_objective),
        "primal_objective_identity_error": abs(primal_objective - float(solution.obj_val)),
        "dual_objective_identity_error": abs(dual_objective - float(solution.obj_val_dual)),
        "reported_primal_residual": abs(float(solution.r_prim)),
        "reported_dual_residual": abs(float(solution.r_dual)),
    }
    supported_status = str(solution.status) in {"Solved", "AlmostSolved"}
    return {**checks, "raw_status": str(solution.status),
        "canonical_primal_objective": primal_objective, "canonical_dual_objective": dual_objective,
        "raw_primal_objective": float(solution.obj_val), "raw_dual_objective": float(solution.obj_val_dual),
        "numerical_dual_upper_wealth": 1. - float(solution.obj_val_dual),
        "all_checks_pass": bool(supported_status and all(value <= tolerance for value in checks.values())),
        "tolerance": tolerance, "exact_arithmetic_bound": False,
        "public_api": "get_problem_data -> SolvingChain.solve_via_data -> Problem.unpack_results",
        "api_reference": "https://www.cvxpy.org/tutorial/advanced/index.html#getting-the-standard-form",
        "raw_solution_reference": "https://clarabel.org/stable/python/getting_started_py/#Results"}


def _psd_matrix_root(value, *, name: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (3,):
        matrix = np.array([[matrix[0], matrix[1]], [matrix[1], matrix[2]]])
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError(f"blocked: {name} requires a finite two-asset covariance")
    scale = max(float(np.abs(matrix).max()), np.finfo(float).tiny)
    roundoff = 128 * np.finfo(float).eps * scale
    if np.abs(matrix - matrix.T).max() > roundoff:
        raise ValueError(f"blocked: {name} is not symmetric")
    symmetric = .5 * (matrix + matrix.T)
    if (np.diag(symmetric) < 0).any():
        raise ValueError(f"blocked: {name} has negative variance")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if eigenvalues.min() < -roundoff:
        raise ValueError(f"blocked: {name} is not positive semidefinite")
    # The row-scaled transpose is important: root.T @ root = covariance.
    root = np.sqrt(np.maximum(eigenvalues, 0.))[:, None] * eigenvectors.T
    repaired = root.T @ root
    # Keep exact zero-variance coordinates exact. PSD then requires their
    # cross-covariances to be zero, up to the accepted rounding tolerance.
    for index in range(2):
        if symmetric[index, index] == 0:
            if abs(symmetric[index, 1 - index]) > roundoff:
                raise ValueError(f"blocked: {name} has covariance with a zero-variance asset")
            repaired[index, :] = repaired[:, index] = 0.
    if np.any(np.diag(symmetric) == 0):
        eigenvalues, eigenvectors = np.linalg.eigh(repaired)
        root = np.sqrt(np.maximum(eigenvalues, 0.))[:, None] * eigenvectors.T
    return repaired, root, {
        "minimum_input_eigenvalue": float(np.linalg.eigvalsh(symmetric).min()),
        "maximum_roundoff_repair": float(np.abs(repaired - symmetric).max()),
        "root_identity_error": float(np.abs(root.T @ root - repaired).max()),
    }


def _holdings(value) -> tuple[np.ndarray, float]:
    original = np.asarray(value, dtype=np.float64)
    if original.shape != (3,) or not np.isfinite(original).all() or original.min() < -1e-12:
        raise ValueError("blocked: convex allocation requires finite long-only three-asset holdings")
    if not np.isclose(original.sum(), 1., atol=1e-10, rtol=0):
        raise ValueError("blocked: convex allocation holdings must sum to one")
    weights = np.maximum(original, 0.)
    weights /= weights.sum()
    return weights, float(np.abs(weights - original).max())


@dataclass(frozen=True)
class ConvexSolverSettings:
    cost_bps: float = 10.
    tolerance: float = 1e-8
    solver_tolerance: float = 1e-10
    max_iter: int = 200
    forecast_horizon_sessions: int = 5
    budget_mode: str = "same_soxx"

    def __post_init__(self):
        if not np.isfinite([self.cost_bps, self.tolerance, self.solver_tolerance]).all():
            raise ValueError("blocked: nonfinite convex solver settings")
        if not 0 <= self.cost_bps < 10000 or not 0 < self.solver_tolerance <= self.tolerance:
            raise ValueError("blocked: invalid convex cost or numerical tolerances")
        if self.max_iter < 1 or self.forecast_horizon_sessions < 1:
            raise ValueError("blocked: invalid convex horizon or iteration limit")
        if self.budget_mode not in {"same_soxx", "external_positive"}:
            raise ValueError("blocked: unknown convex risk-budget mode")


class DPPConvexAllocationSolver:
    """Reusable serial solver; use a separate instance for each replay worker.

    ``x`` is post-fee asset value relative to pretrade NAV, not normalized
    weights. Auxiliary absolute-trade variables cannot burn cash at an optimum:
    scaling a slack-fee solution upward preserves its homogeneous risk/norm
    constraints and improves its positive objective. A feasible SOXX target
    supplies that positive objective when its forecast is greater than -100%.
    """
    def __init__(self, cost_bps: float = 10., tolerance: float = 1e-8, *,
                 solver_tolerance: float = 1e-10, max_iter: int = 200,
                 forecast_horizon_sessions: int = 5, budget_mode: str = "same_soxx"):
        self.settings = ConvexSolverSettings(cost_bps, tolerance, solver_tolerance, max_iter, forecast_horizon_sessions, budget_mode)
        if "CLARABEL" not in cp.installed_solvers():
            raise RuntimeError("blocked: CLARABEL is unavailable for convex allocation")
        self._x = cp.Variable(3, nonneg=True, name="post_fee_values")
        self._trades = cp.Variable(2, nonneg=True, name="absolute_risky_trades")
        self._held = cp.Parameter(2, name="actual_risky_holdings")
        self._mu = cp.Parameter(2, name="expected_asset_returns_H")
        self._rate = cp.Parameter(nonneg=True, name="cost_per_risky_turnover", value=cost_bps / 10000.)
        self._risk_root = cp.Parameter((2, 2), name="benchmark_normalized_risk_root")
        self._uncertainty_root = cp.Parameter((2, 2), name="scaled_mean_error_root_H")
        net = cp.sum(self._x)
        difference = self._x[:2] - net * BENCHMARK_RISKY_WEIGHTS
        # Funding gives net = 1 - rate*sum(trades). Remove the additive
        # constant one to avoid subtracting near-unit primal/dual objectives
        # when the economically relevant improvement is only a few basis
        # points. This changes conditioning, not the optimization problem.
        objective = cp.Maximize(self._mu @ self._x[:2] - self._rate * cp.sum(self._trades)
                                - cp.norm(self._uncertainty_root @ difference, 2))
        common = [self._trades >= self._x[:2] - self._held,
                  self._trades >= self._held - self._x[:2],
                  net + self._rate * cp.sum(self._trades) == 1.]
        self._positive_branch = ("positive_benchmark_variance" if budget_mode == "same_soxx"
                                 else "positive_external_risk_budget")
        self._problems = {
            self._positive_branch: cp.Problem(objective, [*common, cp.norm(self._risk_root @ self._x[:2], 2) <= net]),
            "zero_benchmark_positive_soxl_variance": cp.Problem(objective, [*common, self._x[0] == 0.]),
            "all_zero_variance": cp.Problem(objective, common),
        }
        if not all(problem.is_dcp(dpp=True) for problem in self._problems.values()):
            raise RuntimeError("blocked: convex allocation problem is not DPP compliant")
        self._calls = dict.fromkeys(self._problems, 0)
        self._last_certificate = None
        self._last_attempts = []

    @property
    def is_dpp(self) -> bool:
        return all(problem.is_dcp(dpp=True) for problem in self._problems.values())

    def _invoke(self, problem):
        """Isolated for deterministic failure-path testing; no market fallback."""
        options = {"tol_gap_abs": self.settings.solver_tolerance, "tol_gap_rel": self.settings.solver_tolerance,
                   "tol_feas": self.settings.solver_tolerance, "max_iter": self.settings.max_iter}
        # Passing solver_opts here is also required by CVXPY 1.9.3's Clarabel
        # inverse-data adapter when unpacking a raw result.
        data, chain, inverse_data = problem.get_problem_data(cp.CLARABEL, enforce_dpp=True, solver_opts=options)
        self._last_attempts = []
        for warm_start in (True, False):
            raw = chain.solve_via_data(problem, data, warm_start=warm_start, solver_opts=options)
            self._last_certificate = _numerical_certificate(raw, data, tolerance=self.settings.tolerance)
            self._last_attempts.append({"warm_start": warm_start, "certificate": self._last_certificate})
            if self._last_certificate["all_checks_pass"]:
                break
            # Exactly one numerical restart. The parameters, objective and
            # requested tolerances are identical; no financial outcome is used
            # to select an attempt. A failed cold certificate remains a failure.
        problem.unpack_results(raw, chain, inverse_data)
        return problem.value

    def _risk_violation(self, weights: np.ndarray, *, matrix: np.ndarray, normalized_root: np.ndarray | None,
                        branch: str, budget: float) -> float:
        if branch == self._positive_branch:
            risky = weights[:2]
            variance = (risky[0]**2 * matrix[0, 0] + 2 * risky[0]*risky[1]*matrix[0, 1]
                        + risky[1]**2 * matrix[1, 1])
            return max(0., float(variance / budget) - 1.)
        if branch == "zero_benchmark_positive_soxl_variance":
            return float(abs(weights[0]))
        return 0.

    def _score_target(self, target: np.ndarray, *, held: np.ndarray, mu: np.ndarray,
                      uncertainty_root: np.ndarray, covariance: np.ndarray, budget: float) -> dict[str, Any]:
        execution = fee_affordable_target(target, pretrade_weights=held, cost_bps=self.settings.cost_bps)
        settled = settle_target_session(held, open_target_weights=execution,
            asset_returns=[0., 0., 0.], cost_bps=self.settings.cost_bps)
        post = settled["ending_values"]
        net = float(post.sum())
        expected = net + float(mu @ post[:2])
        penalty = float(np.linalg.norm(uncertainty_root @ (post[:2] - net * BENCHMARK_RISKY_WEIGHTS)))
        variance = float(target[:2] @ covariance @ target[:2])
        soxx_variance = float(covariance[1, 1])
        budget_ratio = variance / budget if budget > 0 else None
        return {"target_net_weights": target.copy(), "execution_target_weights": execution,
            "post_fee_values": post, "net_equity_after_fees": net,
            "turnover": float(settled["turnover"]), "fees": float(settled["fees"]),
            "expected_terminal_wealth": expected, "robust_penalty": penalty,
            "objective": expected - penalty, "portfolio_variance": max(0., variance),
            "benchmark_variance": soxx_variance, "risk_budget_variance": budget,
            "soxx_variance_same_covariance": soxx_variance,
            "predicted_variance_ratio": budget_ratio,
            "predicted_variance_ratio_to_budget": budget_ratio,
            "predicted_variance_ratio_to_current_soxx": variance / soxx_variance if soxx_variance > 0 else None}

    def solve(self, pretrade_weights, expected_returns_H, covariance_daily, *, benchmark_variance=None,
              mean_error_covariance_H=None, robust_radius: float = 1., risk_budget_variance=None,
              budget_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        """Solve with one explicitly resolved daily-variance budget.

        The default retains the original same-SOXX constraint. External mode
        requires ``risk_budget_variance>0`` and ``robust_radius=0``; it does not
        infer a budget from the current covariance. Budget provenance is a
        caller declaration: its annual cutoff and data maturity must be
        verified by the Qlib workflow, not inferred by this numerical solver.
        """
        held, holding_repair = _holdings(pretrade_weights)
        mu = np.asarray(expected_returns_H, dtype=np.float64)
        if mu.shape != (2,) or not np.isfinite(mu).all() or (mu <= -1.).any():
            raise ValueError("blocked: expected asset returns must be finite and greater than -100 percent")
        if not np.isfinite(robust_radius) or robust_radius < 0:
            raise ValueError("blocked: invalid mean-uncertainty radius")
        if budget_provenance is not None and not isinstance(budget_provenance, dict):
            raise ValueError("blocked: budget provenance must be a metadata mapping")
        provenance = deepcopy(budget_provenance or {})
        covariance, risk_root, covariance_audit = _psd_matrix_root(covariance_daily, name="daily covariance")
        soxx_variance = float(covariance[1, 1])
        if self.settings.budget_mode == "same_soxx":
            if risk_budget_variance is not None:
                raise ValueError("blocked: explicit external budget requires external_positive mode")
            budget = soxx_variance
            if benchmark_variance is not None:
                supplied = float(benchmark_variance)
                if not np.isfinite(supplied) or supplied < 0 or (budget == 0 and supplied != 0) or (budget > 0 and not np.isclose(supplied, budget, rtol=1e-12, atol=0)):
                    raise ValueError("blocked: risk budget must equal the same covariance's SOXX variance")
        else:
            if benchmark_variance is not None:
                raise ValueError("blocked: external mode cannot also specify benchmark_variance")
            if risk_budget_variance is None:
                raise ValueError("blocked: external mode requires an explicit positive risk budget")
            budget = float(risk_budget_variance)
            if not np.isfinite(budget) or budget <= 0:
                raise ValueError("blocked: external risk budget must be finite and strictly positive")
            if robust_radius != 0:
                raise ValueError("blocked: external risk budgets require robust_radius=0")
        mean_error, error_root, uncertainty_audit = _psd_matrix_root(
            np.zeros((2, 2)) if mean_error_covariance_H is None else mean_error_covariance_H,
            name="mean-error covariance")
        uncertainty_root = robust_radius * error_root
        if budget > 0:
            branch = self._positive_branch
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                normalized_root = risk_root / np.sqrt(budget)
            if not np.isfinite(normalized_root).all():
                raise ValueError("blocked: covariance normalization is nonfinite at the declared risk budget")
        elif covariance[0, 0] > 0:
            branch = "zero_benchmark_positive_soxl_variance"
            normalized_root = None
        else:
            branch = "all_zero_variance"
            normalized_root = None
        nulls, null_results = {}, {}
        # HOLD wins an exact objective tie, without adding a trade threshold.
        null_targets = [("HOLD", held), ("SOXX", np.array([0., 1., 0.])), ("CASH", np.array([0., 0., 1.]))]
        scaled_soxx_fraction = 1.
        positive_reference = "SOXX"
        if self.settings.budget_mode == "external_positive":
            # Separate square roots avoid prematurely underflowing B/Sigma22.
            if soxx_variance > budget:
                scaled_soxx_fraction = float(np.sqrt(budget) / np.sqrt(soxx_variance))
            null_targets.insert(2, ("SCALED_SOXX", np.array([0., scaled_soxx_fraction, 1. - scaled_soxx_fraction])))
            positive_reference = "SCALED_SOXX"
        for name, target in null_targets:
            risk_violation = self._risk_violation(target, matrix=covariance, normalized_root=normalized_root, branch=branch, budget=budget)
            feasible = risk_violation <= self.settings.tolerance if budget > 0 else risk_violation == 0
            nulls[name] = {"feasible": bool(feasible), "risk_violation": risk_violation, "objective": None}
            if feasible:
                scored = self._score_target(target, held=held, mu=mu, uncertainty_root=uncertainty_root, covariance=covariance, budget=budget)
                null_results[name] = scored
                nulls[name]["objective"] = scored["objective"]
        best_name = max(null_results, key=lambda name: null_results[name]["objective"])
        best_null = null_results[best_name]
        if positive_reference not in null_results or null_results[positive_reference]["objective"] <= 0:
            raise ValueError("blocked: positive feasible benchmark wealth required for fee-tightness proof")
        self._held.value, self._mu.value = held[:2], mu
        self._uncertainty_root.value = uncertainty_root
        self._risk_root.value = np.zeros((2, 2)) if normalized_root is None else normalized_root
        problem = self._problems[branch]
        first_call = self._calls[branch] == 0
        self._calls[branch] += 1
        self._last_certificate = None
        self._last_attempts = []
        started = perf_counter()
        error = None
        try:
            self._invoke(problem)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        elapsed = perf_counter() - started
        audit = {"solver": "CLARABEL", "cvxpy_version": cp.__version__, "dpp": True,
            "branch": branch, "first_call_for_branch": first_call, "branch_solve_count": self._calls[branch],
            "wall_seconds": elapsed, "solver_status": problem.status if error is None else "solver_exception",
            "solver_exception": error, "tolerance": self.settings.tolerance,
            "solver_tolerance": self.settings.solver_tolerance, "pretrade_roundoff_repair": holding_repair,
            "covariance": covariance_audit, "mean_error_covariance": uncertainty_audit,
            "budget_mode": self.settings.budget_mode, "risk_budget_variance": budget,
            "soxx_variance_same_covariance": soxx_variance,
            "risk_budget_units": "one_session_simple_return_variance",
            "budget_provenance": deepcopy(provenance),
            "budget_provenance_scope": "caller declared; annual cutoff and data maturity verified by workflow",
            "full_soxx_feasible": nulls["SOXX"]["feasible"], "scaled_soxx_fraction": scaled_soxx_fraction,
            "chosen_positive_feasible_reference": positive_reference,
            "robust_radius": float(robust_radius), "dual_gap_available": self._last_certificate is not None,
            "risk_tolerance_units": "variance_ratio_excess; same units as independent workflow audit",
            "optimization_objective_offset": 1.,
            "optimization_objective": "mu_H*xRisky - cost*sum(absolute_trades) - robust_penalty; add one for terminal wealth",
            "numerical_certificate": self._last_certificate,
            "numerical_attempts": self._last_attempts,
            "cold_retry_used": len(self._last_attempts) == 2,
            "cold_retry_policy": "one same-input cold restart only after a failed numerical certificate; identical solver tolerances",
            "certification_scope": "measured primal/dual residuals, cone feasibility, numerical dual gap and exact-funded wealth within the numerical dual bound; finite-tolerance certificate, not exact arithmetic"}
        if error is None and problem.solver_stats is not None:
            audit.update(solver_seconds=problem.solver_stats.solve_time,
                         setup_seconds=problem.solver_stats.setup_time, iterations=problem.solver_stats.num_iters,
                         compilation_seconds=getattr(problem, "compilation_time", None))
        candidate = None
        failure = error
        if error is None and problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            failure = f"solver status is {problem.status}, no supported primal/dual solution"
        if failure is None:
            try:
                raw = np.asarray(self._x.value, dtype=float)
                trades = np.asarray(self._trades.value, dtype=float)
                if raw.shape != (3,) or trades.shape != (2,) or not np.isfinite(np.r_[raw, trades, problem.value]).all() or raw.sum() <= 0:
                    raise ValueError("nonfinite or empty solver allocation")
                actual_turnover = float(np.abs(raw[:2] - held[:2]).sum())
                rate = self.settings.cost_bps / 10000.
                primal = {
                    "long_only_violation": max(0., -float(raw.min()), -float(trades.min())),
                    "absolute_trade_violation": max(0., float((np.abs(raw[:2] - held[:2]) - trades).max())),
                    "solver_funding_residual": abs(float(raw.sum() + rate * trades.sum() - 1.)),
                    "actual_funding_residual": abs(float(raw.sum() + rate * actual_turnover - 1.)),
                    "artificial_fee_slack": rate * float(trades.sum() - actual_turnover),
                }
                target = np.maximum(raw, 0.)
                target /= target.sum()
                violation = self._risk_violation(target, matrix=covariance, normalized_root=normalized_root, branch=branch, budget=budget)
                primal["risk_violation_before_roundoff_repair"] = violation
                if any(abs(value) > self.settings.tolerance for value in primal.values()):
                    audit.update(primal)
                    raise ValueError("solver allocation violates independent feasibility or fee-tightness tolerance")
                uncorrected = target.copy()
                if branch == "zero_benchmark_positive_soxl_variance":
                    target[0] = 0.
                    target /= target.sum()
                elif violation > 0:
                    target[:2] /= np.linalg.norm(normalized_root @ target[:2])
                    target[2] = 1. - target[:2].sum()
                primal["target_roundoff_repair"] = float(np.abs(target - uncorrected).max())
                candidate = self._score_target(target, held=held, mu=mu, uncertainty_root=uncertainty_root, covariance=covariance, budget=budget)
                primal["post_fee_reconstruction_difference"] = float(np.abs(candidate["post_fee_values"] - raw).max())
                solver_wealth_objective = 1. + float(problem.value)
                primal["objective_reconstruction_difference"] = abs(float(candidate["objective"] - solver_wealth_objective))
                primal["feasible_null_objective_gap"] = candidate["objective"] - best_null["objective"]
                audit.update(primal, solver_objective=solver_wealth_objective,
                             solver_centered_objective=float(problem.value), reconstructed_objective=candidate["objective"])
                if primal["post_fee_reconstruction_difference"] > self.settings.tolerance or primal["objective_reconstruction_difference"] > self.settings.tolerance:
                    raise ValueError("exact execution differs from solver objective or asset values")
                if primal["feasible_null_objective_gap"] < -self.settings.tolerance:
                    raise ValueError("solver allocation is dominated by an explicit feasible null")
                certificate = self._last_certificate
                if certificate is None or not certificate["all_checks_pass"]:
                    raise ValueError("missing or failed primal/dual numerical certificate")
                chosen_objective = max(candidate["objective"], best_null["objective"])
                bound_gap = certificate["numerical_dual_upper_wealth"] - chosen_objective
                audit["exact_funded_wealth_to_numerical_dual_bound_gap"] = bound_gap
                if abs(bound_gap) > self.settings.tolerance:
                    raise ValueError("exact-funded feasible objective is not within the numerical dual-bound tolerance")
            except (ValueError, FloatingPointError) as exception:
                failure = str(exception)
        verified = failure is None and candidate is not None
        if verified and candidate["objective"] >= best_null["objective"]:
            chosen = candidate
            source = "socp"
        else:
            chosen = best_null
            source = f"feasible_null:{best_name}"
        audit["verification_failure"] = failure
        audit["null_substitution_within_tolerance"] = bool(verified and source != "socp")
        return {**chosen, "status": "solved" if verified else "fallback_feasible_null",
            "optimality_verified": bool(verified), "solution_source": source,
            "pretrade_weights": held, "expected_returns_H": mu.copy(),
            "covariance_daily": covariance, "mean_error_covariance_H": mean_error,
            "budget_mode": self.settings.budget_mode, "budget_provenance": provenance,
            "risk_budget_units": "one_session_simple_return_variance",
            "budget_fit_year": provenance.get("model_year"),
            "budget_available_at": provenance.get("estimation_cutoff"),
            "budget_source_sha256": provenance.get("training_source_hash"),
            "full_soxx_feasible": nulls["SOXX"]["feasible"], "scaled_soxx_fraction": scaled_soxx_fraction,
            "chosen_positive_feasible_reference": positive_reference,
            "cost_bps": self.settings.cost_bps, "forecast_horizon_sessions": self.settings.forecast_horizon_sessions,
            "reoptimization": "every decision; forecast horizon is not a holding lock",
            "future_transactions_assumed": False, "null_objectives": nulls,
            "best_feasible_null": best_name, "best_null_objective": best_null["objective"],
            "null_gap": chosen["objective"] - best_null["objective"], "solver_audit": audit}


ConvexAllocationSolver = DPPConvexAllocationSolver

__all__ = ["ConvexSolverSettings", "DPPConvexAllocationSolver", "ConvexAllocationSolver"]

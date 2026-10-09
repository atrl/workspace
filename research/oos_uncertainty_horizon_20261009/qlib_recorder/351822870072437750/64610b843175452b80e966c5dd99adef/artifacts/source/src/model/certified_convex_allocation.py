"""Same convex allocation with bounded, certificate-driven numerical retries.

No portfolio objective, feasibility tolerance, forecast, fee or risk input is
changed. Escalation depends only on the original-unit numerical certificate.
"""
from __future__ import annotations

import cvxpy as cp

from src.model.convex_allocation import DPPConvexAllocationSolver, _numerical_certificate


class CertifiedConvexAllocationSolver(DPPConvexAllocationSolver):
    def solve(self, *args, **kwargs):
        result = super().solve(*args, **kwargs)
        attempts = result["solver_audit"].get("numerical_attempts", [])
        result["solver_audit"].update(
            cold_retry_used=any(not attempt["warm_start"] for attempt in attempts),
            cold_retry_policy="bounded original warm/cold, refined cold, unequilibrated refined cold; original-unit certificate only",
            numerical_escalation_used=any(attempt.get("numerical_method") in
                {"refined_cold", "unequilibrated_refined_cold"} for attempt in attempts),
            financial_tolerances_unchanged=True)
        return result

    def _invoke(self, problem):
        base = {"tol_gap_abs": self.settings.solver_tolerance,
            "tol_gap_rel": self.settings.solver_tolerance,
            "tol_feas": self.settings.solver_tolerance, "max_iter": self.settings.max_iter}
        refinement = {"iterative_refinement_enable": True, "iterative_refinement_max_iter": 20,
            "iterative_refinement_abstol": 1e-14, "iterative_refinement_reltol": 1e-14}
        attempts = [("original_warm", True, base), ("original_cold", False, base),
            ("refined_cold", False, base | refinement),
            ("unequilibrated_refined_cold", False, base | refinement | {"equilibrate_enable": False})]
        self._last_attempts = []
        for name, warm, options in attempts:
            data, chain, inverse = problem.get_problem_data(cp.CLARABEL, enforce_dpp=True, solver_opts=options)
            raw = chain.solve_via_data(problem, data, warm_start=warm, solver_opts=options)
            certificate = _numerical_certificate(raw, data, tolerance=self.settings.tolerance)
            self._last_certificate = certificate
            self._last_attempts.append({"numerical_method": name, "warm_start": warm,
                "solver_options": dict(options), "certificate": certificate,
                "financial_inputs_unchanged": True})
            if certificate["all_checks_pass"]:
                break
        problem.unpack_results(raw, chain, inverse)
        return problem.value

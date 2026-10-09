"""Numerical certificates may improve; financial acceptance never weakens."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import certified_convex_allocation as certified
from src.model.convex_allocation import DPPConvexAllocationSolver


SIGMA = np.array([[.0009, .00028], [.00028, .0001]])
METHODS = ["original_warm", "original_cold", "refined_cold", "unequilibrated_refined_cold"]


@pytest.fixture
def numerical_case():
    return json.loads((Path(__file__).parent / "fixtures/convex_refinement_regression.json").read_text())


def fixed_solve(solver, case):
    return solver.solve(case["held"], case["expected_returns_H"], case["covariance_daily"],
        mean_error_covariance_H=case["mean_error_covariance_H"], robust_radius=case["robust_radius"])


def test_original_fixed_failure_gets_raw_unit_certificate_without_changing_null(numerical_case):
    original = fixed_solve(DPPConvexAllocationSolver(**numerical_case["settings"]), numerical_case)
    repaired = fixed_solve(certified.CertifiedConvexAllocationSolver(**numerical_case["settings"]), numerical_case)
    assert original["optimality_verified"] is False
    assert original["solver_audit"]["numerical_certificate"]["dual_residual_abs"] > 1e-8
    assert repaired["optimality_verified"] is True
    audit = repaired["solver_audit"]
    attempts = audit["numerical_attempts"]
    assert [a["numerical_method"] for a in attempts] == METHODS[:3]
    assert all(a["certificate"]["all_checks_pass"] is False for a in attempts[:2])
    assert attempts[-1]["certificate"]["all_checks_pass"] is True
    assert attempts[-1]["certificate"]["dual_residual_abs"] <= 1e-8
    np.testing.assert_array_equal(repaired["target_net_weights"], numerical_case["exact_feasible_null_target"])
    assert repaired["objective"] == original["objective"] == numerical_case["exact_feasible_null_objective"]
    assert audit["tolerance"] == 1e-8 and audit["solver_tolerance"] == 1e-10
    assert audit["cold_retry_used"] and audit["numerical_escalation_used"]
    assert audit["financial_tolerances_unchanged"] is True
    assert abs(audit["exact_funded_wealth_to_numerical_dual_bound_gap"]) <= 1e-8
    for attempt in attempts:
        assert attempt["certificate"]["tolerance"] == 1e-8
        for key in ("tol_gap_abs", "tol_gap_rel", "tol_feas"):
            assert attempt["solver_options"][key] == 1e-10
        assert attempt["solver_options"]["max_iter"] == 200
        assert attempt["financial_inputs_unchanged"] is True


@pytest.mark.parametrize("cost", [10., 25.])
def test_successful_original_sequence_preserves_targets_and_objectives_exactly(cost):
    old = DPPConvexAllocationSolver(cost_bps=cost)
    new = certified.CertifiedConvexAllocationSolver(cost_bps=cost)
    random = np.random.default_rng(83)
    uncertainty = np.diag([.004**2, .001**2])
    for _ in range(8):
        held, means = random.dirichlet(np.ones(3)), random.normal(.004, .01, size=2)
        before = (held.copy(), means.copy(), uncertainty.copy())
        reference = old.solve(held, means, SIGMA, mean_error_covariance_H=uncertainty)
        actual = new.solve(held, means, SIGMA, mean_error_covariance_H=uncertainty)
        assert reference["optimality_verified"] and actual["optimality_verified"]
        np.testing.assert_array_equal(actual["target_net_weights"], reference["target_net_weights"])
        assert actual["objective"] == reference["objective"]
        assert actual["fees"] == reference["fees"]
        assert not actual["solver_audit"]["numerical_escalation_used"]
        for a, b in zip((held, means, uncertainty), before):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("scale", [0., 1e-12, 1.])
def test_zero_and_small_covariances_preserve_original_unit_acceptance(scale):
    result = certified.CertifiedConvexAllocationSolver(cost_bps=10.).solve(
        [0., 1., 0.], [-.001, .002], SIGMA * scale,
        mean_error_covariance_H=np.eye(2) * 1e-8 * scale)
    assert result["optimality_verified"] is True
    assert result["solver_audit"]["numerical_certificate"]["all_checks_pass"] is True
    assert result["solver_audit"]["numerical_certificate"]["tolerance"] == 1e-8
    assert result["target_net_weights"].min() >= 0
    assert result["target_net_weights"].sum() == pytest.approx(1., abs=1e-12)


def _forced_certificates(monkeypatch, *, fail_count):
    original = certified._numerical_certificate
    seen = []

    def check(raw, data, *, tolerance):
        seen.append({"tolerance": tolerance, "A": data["A"].copy(),
                     "b": np.array(data["b"]), "c": np.array(data["c"])})
        result = original(raw, data, tolerance=tolerance)
        if len(seen) <= fail_count:
            result = deepcopy(result)
            result["dual_residual_abs"] = 2e-8
            result["all_checks_pass"] = False
        return result

    monkeypatch.setattr(certified, "_numerical_certificate", check)
    return seen


def test_first_passing_certificate_stops_retry_before_any_later_candidate(monkeypatch):
    seen = _forced_certificates(monkeypatch, fail_count=2)
    result = certified.CertifiedConvexAllocationSolver(cost_bps=10.).solve([0., 1., 0.], [-.01, .02], SIGMA)
    assert result["optimality_verified"]
    assert len(seen) == 3
    assert [a["numerical_method"] for a in result["solver_audit"]["numerical_attempts"]] == METHODS[:3]
    assert all(item["tolerance"] == 1e-8 for item in seen)
    for item in seen[1:]:
        np.testing.assert_array_equal(item["A"].toarray(), seen[0]["A"].toarray())
        np.testing.assert_array_equal(item["b"], seen[0]["b"])
        np.testing.assert_array_equal(item["c"], seen[0]["c"])


def test_four_failed_certificates_remain_unverified_and_adapter_blocks(monkeypatch):
    from src.backtest.horizon_matched_allocation import replay_horizon_allocation

    seen = _forced_certificates(monkeypatch, fail_count=100)
    result = certified.CertifiedConvexAllocationSolver(cost_bps=10.).solve([0., 1., 0.], [-.01, .02], SIGMA)
    assert result["optimality_verified"] is False
    assert result["status"] == "fallback_feasible_null"
    attempts = result["solver_audit"]["numerical_attempts"]
    assert len(seen) == 4
    assert [a["numerical_method"] for a in attempts] == METHODS
    assert [a["warm_start"] for a in attempts] == [True, False, False, False]
    assert all(not a["certificate"]["all_checks_pass"] for a in attempts)
    for item in seen[1:]:
        np.testing.assert_array_equal(item["A"].toarray(), seen[0]["A"].toarray())
        np.testing.assert_array_equal(item["b"], seen[0]["b"])
        np.testing.assert_array_equal(item["c"], seen[0]["c"])
        assert item["tolerance"] == 1e-8
    assert attempts[-1]["solver_options"]["equilibrate_enable"] is False
    for attempt in attempts[2:]:
        options = attempt["solver_options"]
        assert options["iterative_refinement_enable"] is True
        assert options["iterative_refinement_max_iter"] == 20
        assert options["iterative_refinement_abstol"] == options["iterative_refinement_reltol"] == 1e-14
    dates = pd.bdate_range("2024-01-02", periods=4)
    frame = pd.DataFrame({"signal_date": dates[:2], "trade_date": dates[1:3], "exit_date": dates[2:],
        "decision_at": dates[:2].tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15),
        "return_available_at": dates[2:].tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31),
        "soxl_return": .01, "soxx_return": .003})
    with pytest.raises(ValueError, match="unverified convex solve"):
        replay_horizon_allocation(frame, np.tile([.004, .002], (2, 1)), np.zeros((2, 2, 2)),
            np.tile(SIGMA, (2, 1, 1)), cost_bps=10., rebalance_sessions=5)
    assert len(seen) == 8

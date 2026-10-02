import numpy as np
import pytest
from types import SimpleNamespace
from scipy import sparse

from src.backtest.position_accounting import settle_target_session
from src.workflow.stage1_economic_action import fee_affordable_target
from src.model.convex_allocation import DPPConvexAllocationSolver, _numerical_certificate, _psd_matrix_root


SIGMA = np.array([[.0009, .00028], [.00028, .0001]])


def _objective(values, mu, uncertainty=None):
    uncertainty = np.zeros((2, 2)) if uncertainty is None else uncertainty
    net = values.sum()
    return net + mu @ values[:2] - np.linalg.norm(uncertainty @ (values[:2] - net * np.array([0., 1.])))


def test_root_orientation_and_rank_deficient_covariance_are_correct():
    matrix, root, audit = _psd_matrix_root(SIGMA, name="test")
    np.testing.assert_allclose(root.T @ root, SIGMA, atol=1e-18, rtol=0)
    weights = np.array([.15, .6])
    assert np.linalg.norm(root @ weights)**2 == pytest.approx(weights @ SIGMA @ weights, abs=1e-18)
    singular = np.array([[.0009, .0003], [.0003, .0001]])
    _, singular_root, _ = _psd_matrix_root(singular, name="test")
    np.testing.assert_allclose(singular_root.T @ singular_root, singular, atol=1e-18, rtol=0)
    assert audit["root_identity_error"] < 1e-18


@pytest.mark.parametrize("cost", [0., 10., 25.])
@pytest.mark.parametrize("robust", [False, True])
def test_dpp_solution_matches_actual_cash_accounting_and_dominates_feasible_nulls(cost, robust):
    solver = DPPConvexAllocationSolver(cost_bps=cost)
    rng = np.random.default_rng(83)
    error_covariance = np.diag([.004**2, .001**2]) if robust else None
    _, error_root, _ = _psd_matrix_root(error_covariance if robust else np.zeros((2, 2)), name="mean error")
    assert solver.is_dpp
    for _ in range(8):
        held = rng.dirichlet(np.ones(3))
        mu = rng.normal(.004, .01, size=2)
        decision = solver.solve(held, mu, SIGMA, mean_error_covariance_H=error_covariance)
        assert decision["status"] == "solved", decision["solver_audit"]
        assert decision["optimality_verified"]
        assert decision["target_net_weights"].min() >= 0
        assert decision["target_net_weights"].sum() == pytest.approx(1., abs=1e-12)
        assert decision["predicted_variance_ratio"] <= 1 + 1e-12
        settled = settle_target_session(held, open_target_weights=decision["execution_target_weights"],
            asset_returns=[0., 0., 0.], cost_bps=cost)
        np.testing.assert_allclose(settled["ending_values"], decision["post_fee_values"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(settled["open_net_weights"], decision["target_net_weights"], atol=1e-12, rtol=0)
        assert decision["fees"] == pytest.approx(cost / 10000 * decision["turnover"], abs=1e-14)
        expected = _objective(settled["ending_values"], mu, error_root)
        assert decision["objective"] == pytest.approx(expected, abs=1e-12)
        assert decision["null_gap"] >= 0
        for null in decision["null_objectives"].values():
            if null["feasible"]:
                assert decision["objective"] >= null["objective"] - 1e-8
        assert decision["forecast_horizon_sessions"] == 5
        assert decision["future_transactions_assumed"] is False
        assert decision["solver_audit"]["dual_gap_available"]
        assert decision["solver_audit"]["numerical_certificate"]["all_checks_pass"]
        assert abs(decision["solver_audit"]["exact_funded_wealth_to_numerical_dual_bound_gap"]) <= 1e-8
        assert decision["solver_audit"]["solver_objective"] == pytest.approx(
            1 + decision["solver_audit"]["solver_centered_objective"], abs=1e-15)
        assert decision["solver_audit"]["objective_reconstruction_difference"] <= 1e-8


def test_exact_soxx_is_representable_and_can_be_selected_as_no_alpha_reference():
    solver = DPPConvexAllocationSolver(cost_bps=10.)
    # Cash and SOXL have worse five-session forecasts; already owning SOXX
    # avoids fees and must dominate each alternative under the same risk cone.
    decision = solver.solve([0., 1., 0.], [-.01, .02], SIGMA)
    assert decision["status"] == "solved"
    assert decision["objective"] >= 1.02 - 1e-10
    np.testing.assert_allclose(decision["target_net_weights"], [0., 1., 0.], atol=1e-8, rtol=0)
    assert decision["null_objectives"]["SOXX"]["objective"] == pytest.approx(1.02)


def test_zero_alpha_hold_is_free_and_drifted_infeasible_hold_is_excluded():
    solver = DPPConvexAllocationSolver(cost_bps=25.)
    held = np.array([.1, .3, .6])
    decision = solver.solve(held, [0., 0.], SIGMA)
    assert decision["status"] == "solved"
    np.testing.assert_allclose(decision["target_net_weights"], held, atol=1e-8, rtol=0)
    assert decision["objective"] >= 1. - 1e-10
    assert decision["fees"] < 1e-10
    unsafe = solver.solve([1., 0., 0.], [0., 0.], SIGMA)
    assert not unsafe["null_objectives"]["HOLD"]["feasible"]
    assert unsafe["predicted_variance_ratio"] <= 1 + 1e-12
    assert unsafe["fees"] > 0


def test_hold_variance_ratio_tolerance_matches_the_strict_workflow_audit():
    solver = DPPConvexAllocationSolver(cost_bps=25.)
    # For the rank-one 3x/1x covariance, beta=1+7.5e-9 means
    # volatility-ratio excess <1e-8 but variance-ratio excess >1e-8.
    covariance = np.array([[.0009, .0003], [.0003, .0001]])
    held = np.array([.2, .4 + 7.5e-9, .4 - 7.5e-9])
    decision = solver.solve(held, [0., 0.], covariance)
    assert not decision["null_objectives"]["HOLD"]["feasible"]
    assert decision["null_objectives"]["HOLD"]["risk_violation"] > 1e-8
    actual = decision["target_net_weights"][:2]
    assert actual @ covariance @ actual / covariance[1, 1] <= 1 + 1e-8
    assert decision["solver_audit"]["risk_tolerance_units"].startswith("variance_ratio")


@pytest.mark.parametrize("covariance", [np.zeros((2, 2)), np.array([[.0009, 0.], [0., 0.]])])
def test_zero_benchmark_variance_uses_no_artificial_risk_budget(covariance):
    solver = DPPConvexAllocationSolver(cost_bps=10.)
    decision = solver.solve([.3, .4, .3], [.05, .01], covariance, benchmark_variance=0.)
    assert decision["status"] == "solved", decision["solver_audit"]
    assert decision["benchmark_variance"] == 0
    assert decision["portfolio_variance"] == 0
    assert decision["predicted_variance_ratio"] is None
    assert decision["solver_audit"]["branch"] != "positive_benchmark_variance"
    if covariance[0, 0] > 0:
        assert decision["target_net_weights"][0] == 0.
        assert not decision["null_objectives"]["HOLD"]["feasible"]


def test_solver_exception_fallback_is_feasible_but_never_called_solved(monkeypatch):
    solver = DPPConvexAllocationSolver()
    monkeypatch.setattr(solver, "_invoke", lambda problem: (_ for _ in ()).throw(RuntimeError("injected failure")))
    decision = solver.solve([.8, .1, .1], [.01, .01], SIGMA)
    assert decision["status"] == "fallback_feasible_null"
    assert not decision["optimality_verified"]
    assert decision["solution_source"].startswith("feasible_null:")
    assert decision["null_gap"] == 0
    assert decision["predicted_variance_ratio"] <= 1 + 1e-12
    assert "injected failure" in decision["solver_audit"]["verification_failure"]


def _raw_certificate_fixture():
    # min y subject to y>=1; exact primal and dual optimum are both one.
    raw = SimpleNamespace(x=[1.], z=[1.], s=[0.], obj_val=1., obj_val_dual=1.,
                          r_prim=0., r_dual=0., status="Solved")
    data = {"A": sparse.csc_matrix([[-1.]]), "b": np.array([-1.]), "c": np.array([1.]),
            "dims": SimpleNamespace(zero=0, nonneg=1, soc=[], exp=0, psd=[], p3d=[])}
    return raw, data


def test_raw_certificate_checks_independent_residuals_and_dual_objective():
    raw, data = _raw_certificate_fixture()
    assert _numerical_certificate(raw, data, tolerance=1e-8)["all_checks_pass"]
    raw.status = "AlmostSolved"
    assert _numerical_certificate(raw, data, tolerance=1e-8)["all_checks_pass"]
    raw.z = [.5]
    failed = _numerical_certificate(raw, data, tolerance=1e-8)
    assert not failed["all_checks_pass"] and failed["dual_residual_abs"] == .5
    raw, data = _raw_certificate_fixture()
    raw.obj_val_dual = .99
    failed = _numerical_certificate(raw, data, tolerance=1e-8)
    assert not failed["all_checks_pass"] and failed["dual_objective_identity_error"] > .009
    raw.obj_val_dual = None
    with pytest.raises(ValueError, match="missing raw"):
        _numerical_certificate(raw, data, tolerance=1e-8)


def test_optimal_inaccurate_requires_an_actual_good_certificate(monkeypatch):
    solver = DPPConvexAllocationSolver()
    invoke = solver._invoke

    def certified_inaccurate(problem):
        invoke(problem)
        problem._status = "optimal_inaccurate"
        solver._last_certificate["raw_status"] = "AlmostSolved"

    monkeypatch.setattr(solver, "_invoke", certified_inaccurate)
    result = solver.solve([0., 1., 0.], [.01, .02], SIGMA)
    assert result["status"] == "solved" and result["optimality_verified"]

    def missing_certificate(problem):
        invoke(problem)
        problem._status = "optimal_inaccurate"
        solver._last_certificate = None

    monkeypatch.setattr(solver, "_invoke", missing_certificate)
    result = solver.solve([0., 1., 0.], [.01, .02], SIGMA)
    assert result["status"] == "fallback_feasible_null"
    assert "certificate" in result["solver_audit"]["verification_failure"]


def test_exact_execution_must_match_the_numerical_dual_bound(monkeypatch):
    solver = DPPConvexAllocationSolver()
    invoke = solver._invoke

    def loose_bound(problem):
        invoke(problem)
        solver._last_certificate["numerical_dual_upper_wealth"] += .001

    monkeypatch.setattr(solver, "_invoke", loose_bound)
    result = solver.solve([0., 1., 0.], [.01, .02], SIGMA)
    assert result["status"] == "fallback_feasible_null"
    assert "dual-bound tolerance" in result["solver_audit"]["verification_failure"]


def test_one_cold_retry_after_failed_certificate_retains_both_attempts(monkeypatch):
    from src.model import convex_allocation as module

    original = module._numerical_certificate
    calls = []

    def fail_first(solution, data, *, tolerance):
        certificate = original(solution, data, tolerance=tolerance)
        calls.append(tolerance)
        if len(calls) == 1:
            certificate["dual_residual_abs"] = 2e-8
            certificate["all_checks_pass"] = False
        return certificate

    monkeypatch.setattr(module, "_numerical_certificate", fail_first)
    result = DPPConvexAllocationSolver().solve([0., 1., 0.], [.01, .02], SIGMA)
    assert result["status"] == "solved"
    assert calls == [1e-8, 1e-8]
    attempts = result["solver_audit"]["numerical_attempts"]
    assert [a["warm_start"] for a in attempts] == [True, False]
    assert not attempts[0]["certificate"]["all_checks_pass"]
    assert attempts[1]["certificate"]["all_checks_pass"]
    assert result["solver_audit"]["cold_retry_used"]


def test_cold_retry_is_bounded_and_cannot_override_failed_certificate(monkeypatch):
    from src.model import convex_allocation as module

    original = module._numerical_certificate
    calls = []

    def fail_every_attempt(solution, data, *, tolerance):
        certificate = original(solution, data, tolerance=tolerance)
        calls.append(tolerance)
        certificate["dual_residual_abs"] = 2e-8
        certificate["all_checks_pass"] = False
        return certificate

    monkeypatch.setattr(module, "_numerical_certificate", fail_every_attempt)
    result = DPPConvexAllocationSolver().solve([0., 1., 0.], [.01, .02], SIGMA)
    assert result["status"] == "fallback_feasible_null"
    assert not result["optimality_verified"]
    assert len(calls) == 2
    assert result["solver_audit"]["solver_tolerance"] == 1e-10


def test_optimal_status_with_artificial_fee_burn_is_rejected(monkeypatch):
    solver = DPPConvexAllocationSolver(cost_bps=25.)

    def bad_solution(problem):
        solver._x.value = np.array([0., .5, .49])
        solver._trades.value = np.array([2., 2.])
        problem._status = "optimal"
        problem._value = -.01

    monkeypatch.setattr(solver, "_invoke", bad_solution)
    decision = solver.solve([0., 1., 0.], [0., 0.], SIGMA)
    assert decision["status"] == "fallback_feasible_null"
    assert not decision["optimality_verified"]
    assert decision["solver_audit"]["artificial_fee_slack"] > 1e-4


def test_optimal_but_null_dominated_point_is_rejected(monkeypatch):
    solver = DPPConvexAllocationSolver(cost_bps=10.)
    held = np.array([0., 1., 0.])
    cash_execution = fee_affordable_target([0., 0., 1.], pretrade_weights=held, cost_bps=10.)
    settled = settle_target_session(held, open_target_weights=cash_execution, asset_returns=[0., 0., 0.], cost_bps=10.)

    def bad_solution(problem):
        solver._x.value = settled["ending_values"].copy()
        solver._trades.value = np.abs(solver._x.value[:2] - held[:2])
        problem._status = "optimal"
        problem._value = settled["net_equity"] - 1.

    monkeypatch.setattr(solver, "_invoke", bad_solution)
    decision = solver.solve(held, [.01, .02], SIGMA)
    assert decision["status"] == "fallback_feasible_null"
    assert "dominated" in decision["solver_audit"]["verification_failure"]
    assert decision["objective"] == pytest.approx(1.02)


def test_fresh_deterministic_replays_and_cost_paths_match_actual_quantities():
    rng = np.random.default_rng(801)
    forecasts = rng.normal(.001, .008, size=(12, 2))
    returns = rng.normal(.001, .02, size=(12, 2))

    def replay(cost):
        solver = DPPConvexAllocationSolver(cost_bps=cost)
        weights = np.array([0., 0., 1.]); equity = 1.
        evidence = []
        for mu, realized in zip(forecasts, returns):
            decision = solver.solve(weights, mu, SIGMA)
            assert decision["status"] == "solved", decision["solver_audit"]
            settled = settle_target_session(weights, open_target_weights=decision["execution_target_weights"],
                asset_returns=np.r_[realized, 0.], cost_bps=cost)
            equity *= settled["net_equity"]
            weights = settled["ending_weights"]
            evidence.append(np.r_[decision["target_net_weights"], weights, settled["fees"], equity])
        return np.array(evidence)

    first, repeated, stress = replay(10.), replay(10.), replay(25.)
    np.testing.assert_allclose(first, repeated, atol=1e-10, rtol=0)
    assert not np.allclose(first[:, -1], stress[:, -1])


def test_robust_penalty_vanishes_at_soxx_and_uses_horizon_error_covariance():
    solver = DPPConvexAllocationSolver()
    uncertainty = np.array([[.04, .01], [.01, .01]])
    decision = solver.solve([0., 1., 0.], [.021, .02], SIGMA, mean_error_covariance_H=uncertainty)
    assert decision["status"] == "solved"
    assert decision["null_objectives"]["SOXX"]["objective"] == pytest.approx(1.02)
    assert decision["robust_penalty"] < 1e-8
    assert decision["null_objectives"]["CASH"]["objective"] < 1.


@pytest.mark.parametrize("kwargs,match", [
    ({"expected_returns_H": [-1., .01]}, "greater than"),
    ({"expected_returns_H": [np.nan, .01]}, "finite"),
    ({"pretrade_weights": [-.1, .6, .5]}, "long-only"),
    ({"pretrade_weights": [.1, .1, .1]}, "sum to one"),
    ({"covariance_daily": [[.0001, .0002], [.0002, .0001]]}, "positive semidefinite"),
    ({"benchmark_variance": .0002}, "same covariance"),
    ({"mean_error_covariance_H": [[1., 2.], [2., 1.]]}, "positive semidefinite"),
])
def test_invalid_inputs_fail_closed_without_forecast_clipping(kwargs, match):
    arguments = {"pretrade_weights": [0., 0., 1.], "expected_returns_H": [.01, .005], "covariance_daily": SIGMA}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        DPPConvexAllocationSolver().solve(**arguments)


@pytest.mark.parametrize("cost", [10., 25.])
def test_external_lower_budget_matches_analytic_scaled_soxx_and_exact_cash_funding(cost):
    variance = .0001
    covariance = variance * np.array([[9., 3.], [3., 1.]])
    budget = variance / 4
    solver = DPPConvexAllocationSolver(cost_bps=cost, budget_mode="external_positive")
    result = solver.solve([0., 0., 1.], [0., .05], covariance,
        risk_budget_variance=budget, robust_radius=0.)
    assert result["status"] == "solved", result["solver_audit"]
    assert not result["full_soxx_feasible"]
    assert not result["null_objectives"]["SOXX"]["feasible"]
    assert result["null_objectives"]["SCALED_SOXX"]["feasible"]
    assert result["scaled_soxx_fraction"] == pytest.approx(.5)
    np.testing.assert_allclose(result["target_net_weights"], [0., .5, .5], atol=1e-7, rtol=0)
    expected_net = 1. / (1. + cost / 10000. * .5)
    assert result["objective"] == pytest.approx(expected_net * 1.025, abs=1e-8)
    assert result["benchmark_variance"] == pytest.approx(variance)
    assert result["risk_budget_variance"] == budget
    assert result["predicted_variance_ratio_to_budget"] == pytest.approx(1., abs=1e-7)
    assert result["predicted_variance_ratio_to_current_soxx"] == pytest.approx(.25, abs=1e-7)
    assert result["chosen_positive_feasible_reference"] == "SCALED_SOXX"


@pytest.mark.parametrize("cost", [10., 25.])
def test_external_higher_budget_reports_more_risk_than_current_soxx_without_mislabeling(cost):
    variance = .0001
    covariance = variance * np.array([[9., 3.], [3., 1.]])
    solver = DPPConvexAllocationSolver(cost_bps=cost, budget_mode="external_positive")
    result = solver.solve([0., 0., 1.], [.3, .01], covariance,
        risk_budget_variance=4 * variance, robust_radius=0.)
    assert result["status"] == "solved", result["solver_audit"]
    np.testing.assert_allclose(result["target_net_weights"], [2/3, 0., 1/3], atol=1e-7, rtol=0)
    assert result["full_soxx_feasible"] and result["scaled_soxx_fraction"] == 1.
    assert result["null_objectives"]["SOXX"]["objective"] == result["null_objectives"]["SCALED_SOXX"]["objective"]
    assert result["predicted_variance_ratio_to_current_soxx"] == pytest.approx(4., abs=1e-7)
    assert result["predicted_variance_ratio_to_budget"] <= 1 + 1e-8
    assert result["solver_audit"]["branch"] == "positive_external_risk_budget"


def test_external_fixed_budget_does_not_follow_a_changed_current_covariance():
    variance = .0001
    covariance = variance * np.array([[9., 3.], [3., 1.]])
    solver = DPPConvexAllocationSolver(budget_mode="external_positive")
    first = solver.solve([0., 0., 1.], [0., .05], covariance,
        risk_budget_variance=variance/4, robust_radius=0.)
    second = solver.solve([0., 0., 1.], [0., .05], 4*covariance,
        risk_budget_variance=variance/4, robust_radius=0.)
    assert first["risk_budget_variance"] == second["risk_budget_variance"] == variance/4
    assert first["scaled_soxx_fraction"] == pytest.approx(.5)
    assert second["scaled_soxx_fraction"] == pytest.approx(.25)
    assert second["target_net_weights"][1] == pytest.approx(.25, abs=1e-7)
    assert second["soxx_variance_same_covariance"] == pytest.approx(4*variance)


@pytest.mark.parametrize("covariance", [np.zeros((2, 2)), np.diag([.0009, 0.])])
def test_external_positive_budget_keeps_zero_current_soxx_variance_distinct(covariance):
    budget = .0001
    result = DPPConvexAllocationSolver(budget_mode="external_positive").solve(
        [0., 0., 1.], [.05, .01], covariance, risk_budget_variance=budget, robust_radius=0.)
    assert result["status"] == "solved", result["solver_audit"]
    assert result["benchmark_variance"] == 0.
    assert result["risk_budget_variance"] == budget
    assert result["full_soxx_feasible"] and result["scaled_soxx_fraction"] == 1.
    assert result["predicted_variance_ratio_to_current_soxx"] is None
    assert result["predicted_variance_ratio_to_budget"] <= 1+1e-8
    if covariance[0, 0] > 0:
        assert result["target_net_weights"][0] <= 1/3 + 1e-8


def test_external_nulls_and_settlement_use_budget_not_sigma22():
    covariance = np.diag([.0009, .0001])
    held = np.array([0., .7, .3])
    result = DPPConvexAllocationSolver(cost_bps=25., budget_mode="external_positive").solve(
        held, [.01, .02], covariance, risk_budget_variance=.000025, robust_radius=0.)
    assert result["status"] == "solved", result["solver_audit"]
    assert not result["null_objectives"]["HOLD"]["feasible"]
    assert not result["null_objectives"]["SOXX"]["feasible"]
    assert result["null_objectives"]["CASH"]["feasible"]
    actual = settle_target_session(held, open_target_weights=result["execution_target_weights"],
        asset_returns=[0., 0., 0.], cost_bps=25.)
    weights = actual["open_net_weights"]
    assert weights[:2] @ covariance @ weights[:2] / .000025 <= 1+1e-8
    np.testing.assert_allclose(actual["ending_values"], result["post_fee_values"], atol=1e-12, rtol=0)
    assert result["null_gap"] >= 0.


def test_external_tiny_positive_budget_is_not_replaced_by_an_epsilon():
    budget = 1e-16
    result = DPPConvexAllocationSolver(budget_mode="external_positive").solve(
        [0., 0., 1.], [0., .01], np.diag([.0009, .0001]),
        risk_budget_variance=budget, robust_radius=0.)
    assert result["risk_budget_variance"] == budget
    assert result["scaled_soxx_fraction"] == pytest.approx(1e-6, rel=1e-12)
    assert result["portfolio_variance"] <= budget * (1+1e-8)
    assert result["null_objectives"]["SCALED_SOXX"]["feasible"]


@pytest.mark.parametrize("budget", [None, 0., -1., np.nan, np.inf])
def test_external_budget_requires_an_explicit_finite_positive_value(budget):
    with pytest.raises(ValueError, match="(explicit positive|finite and strictly positive)"):
        DPPConvexAllocationSolver(budget_mode="external_positive").solve(
            [0., 0., 1.], [.01, .01], SIGMA, risk_budget_variance=budget, robust_radius=0.)


def test_external_mode_rejects_ambiguity_and_any_positive_robust_radius():
    solver = DPPConvexAllocationSolver(budget_mode="external_positive")
    with pytest.raises(ValueError, match="benchmark_variance"):
        solver.solve([0., 0., 1.], [.01, .01], SIGMA,
            benchmark_variance=.0001, risk_budget_variance=.0001, robust_radius=0.)
    for radius in (1., 1e-300):
        with pytest.raises(ValueError, match="robust_radius=0"):
            solver.solve([0., 0., 1.], [.01, .01], SIGMA,
                risk_budget_variance=.0001, robust_radius=radius)
    with pytest.raises(ValueError, match="requires external_positive"):
        DPPConvexAllocationSolver().solve([0., 0., 1.], [.01, .01], SIGMA, risk_budget_variance=.0001)
    with pytest.raises(ValueError, match="unknown convex risk-budget"):
        DPPConvexAllocationSolver(budget_mode="implicit_external")


def test_external_provenance_is_retained_without_claiming_solver_verified_maturity():
    provenance = {"model_year": 2020, "estimation_cutoff": "2020-01-01T00:00:00-05:00",
                  "last_observation_at": "2019-12-31T14:30:00Z", "training_source_hash": "a"*64,
                  "method": "synthetic prior-only variance", "units": "one_session_simple_return_variance"}
    result = DPPConvexAllocationSolver(budget_mode="external_positive").solve(
        [0., 0., 1.], [.01, .01], SIGMA, risk_budget_variance=.0002,
        robust_radius=0., budget_provenance=provenance)
    assert result["budget_provenance"] == provenance
    assert result["budget_fit_year"] == 2020 and result["budget_source_sha256"] == "a"*64
    assert "caller declared" in result["solver_audit"]["budget_provenance_scope"]
    provenance["model_year"] = 2021
    assert result["budget_provenance"]["model_year"] == 2020


def test_default_and_explicit_same_soxx_modes_have_identical_numerical_paths():
    first = DPPConvexAllocationSolver()
    explicit = DPPConvexAllocationSolver(budget_mode="same_soxx")
    rng = np.random.default_rng(238)
    for _ in range(8):
        held = rng.dirichlet(np.ones(3)); mu = rng.normal(.005, .01, 2)
        a, b = first.solve(held, mu, SIGMA), explicit.solve(held, mu, SIGMA)
        np.testing.assert_array_equal(a["target_net_weights"], b["target_net_weights"])
        assert a["objective"] == b["objective"]
        assert list(a["null_objectives"]) == ["HOLD", "SOXX", "CASH"]
        assert a["benchmark_variance"] == a["risk_budget_variance"]

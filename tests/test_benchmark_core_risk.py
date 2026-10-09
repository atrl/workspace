from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from src.model.benchmark_core_risk import benchmark_core_allocation


def _map(covariance, budget, representation="core"):
    return benchmark_core_allocation(covariance, budget, representation=representation)


def test_same_soxx_budget_returns_exact_benchmark_without_invented_beta():
    for matrix in (np.array([[.003, .00041], [.00041, .0002]]),
                   np.array([[.0001, -.00004], [-.00004, .0002]])):
        result = _map(matrix, matrix[1, 1])
        np.testing.assert_array_equal(result["weights"], [0., 1., 0.])
        assert result["portfolio_variance"] == matrix[1, 1]
        assert result["budget_used_fraction"] == 1.
        assert result["branch"] == "exact_soxx_at_budget"


def test_same_risk_input_two_representations_have_distinct_owned_exposures():
    matrix, budget = np.array([[.0009, .0003], [.0003, .0001]]), .000225
    core, lever = _map(matrix, budget), _map(matrix, budget, "lever_cash")
    np.testing.assert_allclose(core["weights"], [.25, .75, 0.], atol=1e-14)
    np.testing.assert_allclose(lever["weights"], [.5, 0., .5], atol=1e-14)
    for result in (core, lever):
        assert result["portfolio_variance"] <= budget
        assert result["portfolio_variance"] == pytest.approx(budget, abs=1e-17)


def test_high_soxx_risk_reduces_soxx_into_cash_and_cap_underuse_is_explicit():
    matrix = np.array([[.0009, .0002], [.0002, .0004]])
    result = _map(matrix, .0001)
    np.testing.assert_allclose(result["weights"], [0., .5, .5], atol=1e-15)
    under = _map(matrix, .001)
    np.testing.assert_array_equal(under["weights"], [1., 0., 0.])
    assert under["portfolio_variance"] == matrix[0, 0]
    assert under["budget_unspent_variance"] == pytest.approx(.0001)
    assert not under["budget_targeted"]
    assert under["branch"] == "full_soxl_budget_underused"


def test_general_nonmonotone_quadratic_uses_rightmost_feasible_root():
    matrix, budget = np.array([[4., -1.5], [-1.5, 1.]]), 2.
    result = _map(matrix, budget)
    # V(a)=8a^2-5a+1 first decreases; feasible right boundary is (5+sqrt57)/16.
    expected = (5 + np.sqrt(57)) / 16
    np.testing.assert_allclose(result["weights"], [expected, 1 - expected, 0.], atol=1e-15)
    assert result["audit"]["risk_curve_initially_decreases"]
    assert result["portfolio_variance"] <= budget
    larger = result["weights"].copy()
    larger[0] += 1e-9
    larger[1] -= 1e-9
    assert larger[:2] @ matrix @ larger[:2] > budget


@pytest.mark.parametrize("representation", ["core", "lever_cash"])
def test_random_psd_matrices_are_funded_feasible_and_core_roots_are_maximal(representation):
    rng = np.random.default_rng(719)
    factors = rng.normal(size=(400, 2, 2))
    matrices = factors @ factors.transpose(0, 2, 1)
    budgets = rng.uniform(.01, 1.4, len(matrices)) * np.max(np.diagonal(matrices, axis1=1, axis2=2), axis=1)
    result = _map(matrices, budgets, representation)
    weights = result["weights"]
    assert np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-15, rtol=0)
    assert (result["portfolio_variance"] <= budgets).all()
    actual = np.einsum("ni,nij,nj->n", weights[:, :2], result["covariance"], weights[:, :2])
    np.testing.assert_allclose(actual, result["portfolio_variance"], atol=2e-14, rtol=1e-14)
    if representation == "core":
        for i, branch in enumerate(result["branch"]):
            if branch == "maximum_soxl_core_root":
                a = weights[i, 0]
                test = np.array([a + 1e-8, 1 - a - 1e-8])
                assert test @ result["covariance"][i] @ test > budgets[i]
                assert weights[i, 2] == 0.
    assert result["audit"]["all_certified_risk_feasible"]


@pytest.mark.parametrize("matrix,budget,expected", [
    ([[0., 0.], [0., 0.]], 1., [1., 0., 0.]),
    ([[0., 0.], [0., 1.]], .25, [0., .5, .5]),
    ([[1., 0.], [0., 0.]], .25, [.5, .5, 0.]),
    ([[1., 1.], [1., 1.]], 2., [1., 0., 0.]),
    ([[4., -2.], [-2., 1.]], 2., [(1 + np.sqrt(2)) / 3, 1 - (1 + np.sqrt(2)) / 3, 0.]),
    ([[4., 1.], [1., 1.]], 4., [1., 0., 0.]),
])
def test_psd_degenerate_and_zero_variance_cases(matrix, budget, expected):
    result = _map(matrix, budget)
    np.testing.assert_allclose(result["weights"], expected, atol=1e-15)
    assert result["portfolio_variance"] <= budget
    if matrix[0][0] == 0:
        lever = _map(matrix, budget, "lever_cash")
        np.testing.assert_array_equal(lever["weights"], [1., 0., 0.])
        assert lever["portfolio_variance"] == 0.


def test_only_roundoff_cross_covariance_is_repaired_and_diagonals_stay_exact():
    matrix = np.array([[1., 1. + 2e-15], [1. + 2e-15, 1.]])
    result = _map(matrix, 1.)
    np.testing.assert_array_equal(result["weights"], [0., 1., 0.])
    np.testing.assert_array_equal(np.diag(result["covariance"]), [1., 1.])
    assert result["audit"]["maximum_covariance_roundoff_adjustment"] > 0
    assert not result["audit"]["economic_covariance_floor_added"]
    assert not result["audit"]["diagonal_variances_changed"]
    with pytest.raises(ValueError, match="positive semidefinite"):
        _map([[1., 1.001], [1.001, 1.]], 1.)


def test_extreme_legal_scales_do_not_overflow_or_underflow_the_exposure_formula():
    smallest, largest = np.nextafter(0., 1.), np.finfo(float).max
    leveraged = _map([[largest, 0.], [0., 1.]], smallest, "lever_cash")
    assert 0 < leveraged["weights"][0] < 1e-300
    assert leveraged["portfolio_variance"] <= smallest
    tiny = _map([[smallest, smallest], [smallest, smallest]], smallest)
    np.testing.assert_array_equal(tiny["weights"], [0., 1., 0.])
    mixed = _map([[largest, 0.], [0., smallest]], 1.)
    assert mixed["weights"][0] > 0 and mixed["weights"][2] == 0.
    assert mixed["portfolio_variance"] <= 1.
    # Independently evaluate the returned float64 holdings at high precision.
    with localcontext() as context:
        context.prec = 100
        a = Decimal.from_float(float(mixed["weights"][0]))
        b = Decimal.from_float(float(mixed["weights"][1]))
        variance = Decimal.from_float(largest) * a * a + Decimal.from_float(smallest) * b * b
        assert variance <= 1


def test_subnormal_secondary_variance_cannot_hide_an_exact_budget_violation():
    smallest = np.nextafter(0., 1.)
    result = _map([[4., 0.], [0., smallest]], 1.)
    a, b = (Fraction.from_float(float(value)) for value in result["weights"][:2])
    exact = 4 * a * a + Fraction.from_float(smallest) * b * b
    assert exact <= 1
    assert result["weights"][0] < .5  # At exactly .5 the tiny second term still exceeds B.
    assert Fraction(result["audit"]["variance_exact_ratio"]) == exact


def test_batch_rows_are_independent_and_input_arrays_are_not_mutated():
    rng = np.random.default_rng(23)
    factors = rng.normal(size=(30, 2, 2))
    matrices = factors @ factors.transpose(0, 2, 1)
    budgets = np.linspace(.1, 2., 30)
    before = matrices.copy()
    for representation in ("core", "lever_cash"):
        original = _map(matrices, budgets, representation)
        altered, changed_budget = matrices.copy(), budgets.copy()
        altered[20:] *= 100.
        changed_budget[20:] *= .01
        later = _map(altered, changed_budget, representation)
        np.testing.assert_array_equal(original["weights"][:20], later["weights"][:20])
        np.testing.assert_array_equal(original["portfolio_variance"][:20], later["portfolio_variance"][:20])
        assert original["audit"]["row_audits"][:20] == later["audit"]["row_audits"][:20]
    np.testing.assert_array_equal(matrices, before)


@pytest.mark.parametrize("matrix,budget,representation", [
    ([[1., 0.], [0., 1.]], 0., "core"),
    ([[1., 0.], [0., 1.]], -1., "core"),
    ([[1., 0.], [0., 1.]], np.inf, "core"),
    ([[1., 0.], [0., 1.]], True, "core"),
    ([[1., 0.], [0., 1.]], [1., 2.], "core"),
    ([[1., 0.], [0., 1.]], 1., "unknown"),
    ([[1., np.nan], [np.nan, 1.]], 1., "core"),
    ([[-1., 0.], [0., 1.]], 1., "core"),
    ([[1., .3], [.1, 1.]], 1., "core"),
    ([[0., 1e-20], [1e-20, 1.]], 1., "core"),
    ([[True, False], [False, True]], 1., "core"),
    ([1., 0., 1.], 1., "core"),
    (np.empty((0, 2, 2)), 1., "core"),
    (np.tile(np.eye(2), (3, 1, 1)), [1., 2.], "core"),
])
def test_invalid_covariance_budget_or_representation_fails_closed(matrix, budget, representation):
    with pytest.raises(ValueError, match="blocked:"):
        _map(matrix, budget, representation)

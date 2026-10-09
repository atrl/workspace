"""Algebra and synthetic solver tests; no market fit, forecast or replay."""
import numpy as np
import pytest

from src.model.benchmark_terminal_value import (
    direct_terminal_value, transform_benchmark_terminal_inputs,
)
from src.model.certified_convex_allocation import CertifiedConvexAllocationSolver


def funded_values(solution):
    # The old solver can leave a ~1e-16 negative raw cash residual at the
    # exact-SOXX boundary. Its owned normalized target is nonnegative. Keep
    # the helper's physical input contract strict and audit this roundoff.
    values = solution["net_equity_after_fees"] * solution["target_net_weights"]
    np.testing.assert_allclose(values, solution["post_fee_values"], rtol=0, atol=1e-12)
    return values


def test_zero_cost_identity_and_input_immutability():
    mu = np.array([[.008, .0028], [-.2, -.07]])
    covariance = np.array([[[.004, .001], [.001, .002]], [[0., 0.], [0., 0.]]])
    before = mu.copy(), covariance.copy()
    result = transform_benchmark_terminal_inputs(mu, covariance, cost_bps=0.)
    np.testing.assert_array_equal(result["transformed_mu"], mu)
    np.testing.assert_array_equal(result["transformed_covariance"], covariance)
    assert result["wealth_scale"] == 1.
    assert result["audit"]["future_conversion_fees_executed"] is False
    assert result["audit"]["literal_terminal_holdings_enforced"] is False
    assert result["audit"]["exact_multi_period_optimality_claim"] is False
    np.testing.assert_array_equal(mu, before[0])
    np.testing.assert_array_equal(covariance, before[1])


@pytest.mark.parametrize("cost", [0., 10., 25., 9500.])
def test_scenario_terminal_formula_equals_independent_share_and_cash_settlement(cost):
    values = np.array([17000., 51000., 30000.])
    scenarios = np.array([[.12, .04], [-.36, -.13], [.01, -.02], [-.98, -.99]])
    for realized in scenarios:
        # Entry prices and owned shares are arbitrary, since values already
        # reflect the entry fee. Execute the hypothetical terminal trade.
        shares_l, shares_x = values[0] / 40., values[1] / 100.
        final_l_price, final_x_price = np.array([40., 100.]) * (1 + realized)
        sold_notional = shares_l * final_l_price
        cash_after_sale = values[2] + sold_notional - cost / 10000 * sold_notional
        purchased_shares = cash_after_sale / (final_x_price * (1 + cost / 10000))
        actual_terminal_value = (shares_x + purchased_shares) * final_x_price
        expected = direct_terminal_value(values, realized, cost)
        assert expected == pytest.approx(actual_terminal_value, rel=1e-13, abs=1e-10)


def test_terminal_value_is_linear_in_scenario_probabilities_and_covariance_transform():
    scenarios = np.array([[.10, .04], [-.20, -.06], [.03, .01]])
    probabilities = np.array([.2, .3, .5])
    mu = probabilities @ scenarios
    centered = scenarios - mu
    covariance = centered.T @ (probabilities[:, None] * centered)
    values, cost = np.array([.2, .7, .09]), 25.
    realized_values = direct_terminal_value(np.tile(values, (3, 1)), scenarios, cost)
    assert probabilities @ realized_values == pytest.approx(direct_terminal_value(values, mu, cost), abs=1e-14)
    transformed = transform_benchmark_terminal_inputs(mu, covariance, cost_bps=cost)
    scenario_transformed = transform_benchmark_terminal_inputs(scenarios, np.tile(covariance, (3, 1, 1)), cost_bps=cost)
    changes = scenario_transformed["transformed_mu"] - transformed["transformed_mu"]
    empirical = changes.T @ (probabilities[:, None] * changes)
    np.testing.assert_allclose(transformed["transformed_covariance"], empirical, rtol=1e-13, atol=1e-16)
    normalized_value = values.sum() + transformed["transformed_mu"] @ values[:2]
    assert transformed["wealth_scale"] * normalized_value == pytest.approx(probabilities @ realized_values, abs=1e-14)


@pytest.mark.parametrize("robust_radius", [0., 1.])
def test_shared_solver_objective_has_the_declared_terminal_value_scale(robust_radius):
    mu = np.array([.012, .004])
    uncertainty = np.array([[.000016, .000002], [.000002, .000004]])
    daily_risk = np.array([[.0009, .00028], [.00028, .0001]])
    source_risk = daily_risk.copy()
    transformed = transform_benchmark_terminal_inputs(mu, uncertainty, cost_bps=25.)
    solution = CertifiedConvexAllocationSolver(cost_bps=25., forecast_horizon_sessions=5).solve(
        [.05, .85, .10], transformed["transformed_mu"], daily_risk,
        mean_error_covariance_H=transformed["transformed_covariance"], robust_radius=robust_radius)
    assert solution["optimality_verified"] is True
    terminal = direct_terminal_value(funded_values(solution), mu, 25.)
    scale = transformed["wealth_scale"]
    assert scale * solution["expected_terminal_wealth"] == pytest.approx(terminal, abs=1e-12)
    assert scale * solution["objective"] == pytest.approx(terminal - scale * solution["robust_penalty"], abs=1e-12)
    np.testing.assert_array_equal(daily_risk, source_risk)


def test_synthetic_cheap_entry_bias_is_removed_by_terminal_conversion_valuation():
    # A prescribed synthetic 3:1 risk pair, not fitted to observed market data.
    means = np.array([.008, .0028])
    risk = np.array([[9., 3.], [3., 1.]]) * .0001
    uncertainty = np.zeros((2, 2))
    raw = CertifiedConvexAllocationSolver(cost_bps=25., forecast_horizon_sessions=5).solve(
        [0., 0., 1.], means, risk, mean_error_covariance_H=uncertainty, robust_radius=0.)
    transformed = transform_benchmark_terminal_inputs(means, uncertainty, cost_bps=25.)
    terminal = CertifiedConvexAllocationSolver(cost_bps=25., forecast_horizon_sessions=5).solve(
        [0., 0., 1.], transformed["transformed_mu"], risk,
        mean_error_covariance_H=transformed["transformed_covariance"], robust_radius=0.)
    assert raw["optimality_verified"] and terminal["optimality_verified"]
    assert raw["target_net_weights"][0] > .30
    assert raw["target_net_weights"][1] < 1e-6
    np.testing.assert_allclose(terminal["target_net_weights"], [0., 1., 0.], atol=1e-7, rtol=0)
    assert raw["fees"] < terminal["fees"]
    assert direct_terminal_value(funded_values(terminal), means, 25.) > direct_terminal_value(funded_values(raw), means, 25.)


@pytest.mark.parametrize("cost", [-1., 10000., np.inf, np.nan, True, [25.]])
def test_invalid_costs_fail_closed(cost):
    with pytest.raises(ValueError, match="blocked:"):
        transform_benchmark_terminal_inputs([.008, .0028], np.eye(2), cost_bps=cost)


@pytest.mark.parametrize("mu,covariance", [
    ([-1., 0.], np.eye(2)), ([np.nan, 0.], np.eye(2)), ([np.inf, 0.], np.eye(2)),
    ([True, False], np.eye(2)), ([.1 + .1j, 0.], np.eye(2)), ([.1], np.eye(2)),
    (np.zeros((0, 2)), np.zeros((0, 2, 2))), (np.ones((3, 2)), np.tile(np.eye(2), (2, 1, 1))),
    ([0., 0.], [[1., .5], [0., 1.]]), ([0., 0.], [[1., 0.], [0., -.01]]),
    ([0., 0.], [[np.inf, 0.], [0., 1.]]), ([0., 0.], np.ones(3)),
])
def test_invalid_mean_and_covariance_inputs_fail_closed(mu, covariance):
    with pytest.raises(ValueError, match="blocked:"):
        transform_benchmark_terminal_inputs(mu, covariance, cost_bps=25.)


@pytest.mark.parametrize("values", [[-.1, .8, .3], [0., 0., 0.], [np.nan, 0., 1.], [.5, .5], [[.2, .7, .1]]])
def test_direct_value_rejects_invalid_or_misaligned_holdings(values):
    with pytest.raises(ValueError, match="blocked:"):
        direct_terminal_value(values, [.01, .003], 25.)


def test_only_floating_point_psd_repairs_are_allowed_and_audited():
    covariance = np.diag([1., -1e-15])
    result = transform_benchmark_terminal_inputs([.01, .003], covariance, cost_bps=25.)
    assert result["audit"]["input_psd_roundoff"][0]["clipped_roundoff_eigenvalues"] == 1
    assert result["audit"]["maximum_covariance_roundoff_adjustment"] == pytest.approx(1e-15)
    assert np.linalg.eigvalsh(result["transformed_covariance"]).min() >= 0
    assert covariance[1, 1] == -1e-15

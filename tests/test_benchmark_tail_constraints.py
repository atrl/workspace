"""Synthetic value, adjoint and raw-feasibility contracts for training risk."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.analysis.integrated_allocation import _cvar, _drawdown
from src.objective.benchmark_tail_constraints import (
    TailConstraintError, combined_daily_adjoints, evaluate_constraints,
    evaluate_risk, reference_risk,
)


@pytest.mark.parametrize("length", [1, 7, 20, 21, 40, 41, 63, 64, 252])
def test_values_match_existing_anchored_drawdown_and_exact_fractional_cvar(length):
    logs = np.random.default_rng(240 + length).normal(-.001, .022, length)
    before = logs.copy()
    result = evaluate_risk(logs)
    assert result["drawdown_loss"] == -_drawdown(logs)
    assert result["cvar5_loss"] == -_cvar(np.expm1(logs))
    assert result["tail_weights"].sum() == pytest.approx(1., abs=2e-15)
    np.testing.assert_array_equal(logs, before)


def test_drawdown_chooses_earliest_trough_and_earliest_equal_prefix_peak():
    # Exact binary fractions make the repeated log peaks/troughs exact ties.
    repeated_trough = evaluate_risk([.125, -.25, .25, -.25])
    assert (repeated_trough["peak_index"], repeated_trough["trough_index"]) == (1, 2)
    np.testing.assert_array_equal(repeated_trough["drawdown_adjoint"], [0., -np.exp(-.25), 0., 0.])
    repeated_peak = evaluate_risk([.125, -.125, .125, -.25])
    assert (repeated_peak["peak_index"], repeated_peak["trough_index"]) == (1, 4)
    np.testing.assert_array_equal(repeated_peak["drawdown_adjoint"], [0., -np.exp(-.25), -np.exp(-.25), -np.exp(-.25)])


def test_flat_cash_has_zero_drawdown_subgradient_and_equal_cvar_tie_weights():
    result = evaluate_risk(np.zeros(100))
    assert result["drawdown_loss"] == result["cvar5_loss"] == 0.
    assert result["peak_index"] == result["trough_index"] == 0
    np.testing.assert_array_equal(result["drawdown_adjoint"], np.zeros(100))
    np.testing.assert_array_equal(result["tail_weights"], np.full(100, .01))
    np.testing.assert_array_equal(result["cvar5_adjoint"], np.full(100, -.01))


def test_fractional_cutoff_ties_share_only_remaining_tail_mass_and_are_permutation_equivariant():
    returns = np.r_[-.12, np.full(4, -.08), np.full(45, .01)]
    result = evaluate_risk(np.log1p(returns))
    assert result["tail_mass"] == 2.5
    assert result["cvar5_loss"] == pytest.approx(.096)
    np.testing.assert_allclose(result["tail_weights"][:5], [.4, .15, .15, .15, .15], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(result["tail_weights"][5:], np.zeros(45))
    permutation = np.random.default_rng(81).permutation(len(returns))
    permuted = evaluate_risk(np.log1p(returns[permutation]))
    np.testing.assert_array_equal(permuted["tail_weights"], result["tail_weights"][permutation])
    np.testing.assert_array_equal(permuted["cvar5_adjoint"], result["cvar5_adjoint"][permutation])


def test_mostly_cash_path_does_not_dilute_one_loss_across_every_zero_day():
    result = evaluate_risk(np.log1p(np.r_[-.1, np.zeros(99)]))
    assert result["cvar5_loss"] == pytest.approx(.02)
    assert result["tail_weights"][0] == .2
    np.testing.assert_allclose(result["tail_weights"][1:], 4 / (99 * 5), rtol=0, atol=1e-16)


@pytest.mark.parametrize("cost_bps", [10., 25.])
def test_initial_fee_enters_anchored_drawdown_and_tail_loss(cost_bps):
    rate = cost_bps / 10000
    logs = np.zeros(100)
    logs[0] = -np.log1p(rate)
    result = evaluate_risk(logs)
    assert result["peak_index"] == 0 and result["trough_index"] == 1
    assert result["drawdown_loss"] == pytest.approx(rate / (1 + rate), rel=1e-14)
    assert result["cvar5_loss"] == pytest.approx(rate / (1 + rate) / 5, rel=1e-14)
    assert result["drawdown_adjoint"][0] == pytest.approx(-1 / (1 + rate), rel=1e-14)
    np.testing.assert_array_equal(result["drawdown_adjoint"][1:], np.zeros(99))


def _finite_difference(function, logs, epsilon=1e-7):
    result = np.empty(len(logs))
    for i in range(len(logs)):
        step = np.zeros(len(logs));step[i] = epsilon
        result[i] = (function(logs + step) - function(logs - step)) / (2 * epsilon)
    return result


def test_full_path_risk_adjoints_match_finite_differences_away_from_ties():
    logs = np.random.default_rng(171).normal(-.0003, .025, 41)
    result = evaluate_risk(logs)
    # Independent legacy value functions provide the perturbation reference.
    drawdown = _finite_difference(lambda x: -_drawdown(x), logs)
    cvar = _finite_difference(lambda x: -_cvar(np.expm1(x)), logs)
    np.testing.assert_allclose(result["drawdown_adjoint"], drawdown, rtol=1e-7, atol=2e-9)
    np.testing.assert_allclose(result["cvar5_adjoint"], cvar, rtol=1e-7, atol=2e-9)


def test_raw_percentage_point_tolerance_is_not_a_normalized_risk_allowance():
    reference = reference_risk(np.log1p(np.r_[-.0001, np.zeros(99)]))
    exact = evaluate_constraints(np.log1p(np.r_[-.0001, np.zeros(99)]), reference)
    np.testing.assert_array_equal(exact["raw_loss_gaps"], [0., 0.])
    assert exact["feasible"]
    inside = evaluate_constraints(np.log1p(np.r_[-.0001 - 5e-13, np.zeros(99)]), reference)
    outside = evaluate_constraints(np.log1p(np.r_[-.0001 - 2e-12, np.zeros(99)]), reference)
    assert inside["normalized_g"][0] > 1e-10
    assert inside["raw_percentage_point_gaps"][0] < 1e-10 and inside["feasible"]
    assert outside["raw_percentage_point_gaps"][0] > 1e-10 and not outside["feasible"]


def test_inactive_objective_cost_still_receives_risk_gradients_with_correct_sign_and_scale():
    logs = np.random.default_rng(93).normal(-.0002, .022, 41)
    reference = reference_risk(np.random.default_rng(33).normal(-.001, .018, 41))
    constraints = evaluate_constraints(logs, reference)
    dual = np.array([.3, .7])
    combined = combined_daily_adjoints(np.zeros(len(logs)), constraints, dual)
    assert np.count_nonzero(combined) > 0 and combined.min() >= 0
    normalizers = reference["normalizers"]
    def scalar(x):
        losses = np.array([-_drawdown(x), -_cvar(np.expm1(x))])
        return -dual @ ((losses - normalizers) / normalizers)
    expected = _finite_difference(scalar, logs)
    np.testing.assert_allclose(combined, expected, rtol=1e-7, atol=2e-8)
    objective = np.linspace(.1, .3, len(logs))
    np.testing.assert_array_equal(combined_daily_adjoints(objective, constraints, [0., 0.]), objective)


@pytest.mark.parametrize("logs", [np.zeros(100), np.full(100, .01), np.log1p(np.r_[-1e-6, np.full(99, .01)])])
def test_nonpositive_benchmark_risk_never_gets_an_epsilon_floor(logs):
    risk = evaluate_risk(logs)
    assert risk["drawdown_loss"] <= 0 or risk["cvar5_loss"] <= 0
    with pytest.raises(TailConstraintError, match="strictly positive"):
        reference_risk(logs)


def test_gain_only_cvar_loss_stays_negative():
    result = evaluate_risk(np.full(100, .01))
    assert result["cvar5_loss"] == pytest.approx(-np.expm1(.01))
    assert result["cvar5_loss"] < 0


@pytest.mark.parametrize("logs", [[], [[0., 0.]], [np.nan], [np.inf], [True, False], [1 + 2j], [1000.], [-1000.], [700., 700.], np.full(1000, -1.)])
def test_invalid_or_nonrepresentable_equity_paths_fail_closed(logs):
    with pytest.raises(TailConstraintError):
        evaluate_risk(logs)


@pytest.mark.parametrize("fault", ["coverage", "normalizer", "order", "nonpositive"])
def test_reference_identity_shape_and_positive_normalizers_are_rechecked(fault):
    logs = np.log1p(np.r_[-.02, np.zeros(99)])
    reference = copy.deepcopy(reference_risk(logs))
    if fault == "coverage": reference["sessions"] += 1
    if fault == "normalizer": reference["normalizers"] = reference["normalizers"] * 2
    if fault == "order": reference["normalizer_order"] = ("cvar5_loss", "drawdown_loss")
    if fault == "nonpositive": reference["normalizers"] = np.array([0., reference["cvar5_loss"]])
    with pytest.raises(TailConstraintError, match="reference"):
        evaluate_constraints(logs, reference)


@pytest.mark.parametrize("tolerance", [-1., np.nan, True, "1e-10"])
def test_invalid_tolerance_does_not_silently_change_feasible_set(tolerance):
    logs = np.log1p(np.r_[-.02, np.zeros(99)])
    with pytest.raises(TailConstraintError, match="tolerance"):
        evaluate_constraints(logs, reference_risk(logs), risk_tolerance_percentage_points=tolerance)


@pytest.mark.parametrize("dual", [[-1., 0.], [np.inf, 0.], [True, False], [0., 0., 0., 0.]])
def test_invalid_dual_values_cannot_be_clipped_or_used(dual):
    logs = np.log1p(np.r_[-.02, np.zeros(99)])
    constraints = evaluate_constraints(logs, reference_risk(logs))
    with pytest.raises(TailConstraintError):
        combined_daily_adjoints(np.zeros(100), constraints, dual)


def test_normalization_overflow_fails_without_floor_or_changed_budget():
    logs = np.r_[-1e-310, np.zeros(99)]
    reference = reference_risk(logs)
    assert reference["normalizers"].min() > 0
    with pytest.raises(TailConstraintError, match="nonfinite normalized"):
        evaluate_constraints(logs, reference)

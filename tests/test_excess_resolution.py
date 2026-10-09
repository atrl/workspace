"""Cumulative-unit resolution removes sign noise without editing economic paths."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.analysis.benchmark_relative_protocol import evaluate_benchmark_relative
from src.analysis.excess_resolution import apply_excess_resolution


YEARS = tuple(range(2016, 2023))


def histories(daily_log_increment):
    dates = pd.bdate_range("2016-01-04", "2022-12-30")
    baseline = np.sin(np.arange(len(dates))) * .004
    candidate, benchmark = {}, {}
    for cost in (10., 25.):
        frame = pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
            "exit_date": dates + pd.offsets.BDay(2), "strategy_return": baseline,
            "cost_bps": cost, "turnover": 0., "fees": 0.})
        benchmark[cost] = frame
        candidate[cost] = frame.assign(strategy_return=np.expm1(np.log1p(baseline) + daily_log_increment))
    return candidate, benchmark


def evidence_for(candidate, benchmark):
    return evaluate_benchmark_relative(candidate, benchmark, expected_years=YEARS,
        minimum_positive_years=6, bootstrap_resamples=200,
        expected_signal_dates=candidate[10.].signal_date)


def test_sub_ulp_scaled_nav_soxx_excess_is_not_positive_evidence():
    candidate, benchmark = histories(0.)
    count = len(candidate[10.])
    # The same flat owned asset with a different account scale, represented by
    # adjacent floats, can yield a spurious positive return of about 2e-16.
    reference_nav = np.ones(count + 1)
    scaled_nav = np.empty(count + 1)
    scaled_nav[0] = 1e6
    for i in range(count): scaled_nav[i + 1] = np.nextafter(scaled_nav[i], np.inf)
    for cost in candidate:
        candidate[cost]["strategy_return"] = scaled_nav[1:] / scaled_nav[:-1] - 1
        benchmark[cost]["strategy_return"] = reference_nav[1:] / reference_nav[:-1] - 1
    evidence = evidence_for(candidate, benchmark)
    resolved = apply_excess_resolution(evidence, candidate, benchmark)
    assert resolved["gates"].raw_aggregate_positive_net_log_excess.all()
    assert resolved["gates"].raw_positive_years.eq(7).all()
    assert not resolved["gates"].aggregate_positive_net_log_excess.any()
    assert resolved["gates"].positive_years.eq(0).all()
    assert resolved["gates"].within_resolution_years.eq(7).all()
    assert resolved["rolling_summary"].positive_windows.eq(0).all()
    assert not resolved["historical_gate_pass"] and not resolved["promotion_eligible"]
    assert resolved["bootstrap"].within_resolution_cumulative_lower_bound.all()


def test_tiny_deterministic_increment_cannot_masquerade_as_alpha():
    candidate, benchmark = histories(1e-16)
    evidence = evidence_for(candidate, benchmark)
    assert evidence["historical_gate_pass"]
    result = apply_excess_resolution(evidence, candidate, benchmark)
    assert result["raw_historical_gate_pass"] is True
    assert result["historical_gate_pass"] is False
    assert not result["bootstrap"].lower_bound_positive.any()
    assert result["annual"].within_resolution.all()
    assert result["rolling_summary"].raw_positive_excess_rate.eq(1.).all()
    assert result["rolling_summary"].positive_excess_rate.eq(0.).all()


def test_material_positive_result_and_risk_gates_preserved_without_resampling(monkeypatch):
    candidate, benchmark = histories(.0003)
    evidence = evidence_for(candidate, benchmark)
    before, old_candidate, old_benchmark = deepcopy(evidence), deepcopy(candidate), deepcopy(benchmark)
    def forbidden(*args, **kwargs): pytest.fail("resolution must never draw a new bootstrap")
    monkeypatch.setattr("src.analysis.integrated_allocation.stationary_bootstrap_excess", forbidden)
    monkeypatch.setattr("src.analysis.benchmark_relative_protocol.stationary_bootstrap_excess", forbidden)
    result = apply_excess_resolution(evidence, candidate, benchmark)
    assert result["historical_gate_pass"] is True and result["promotion_eligible"] is False
    for column in ("anchored_max_drawdown_not_worse_than_soxx", "cvar5_not_worse_than_soxx", "minimum_positive_years"):
        pd.testing.assert_series_equal(result["gates"][column], evidence["gates"][column])
    for name in ("metrics", "annual"):
        pd.testing.assert_frame_equal(result[name][evidence[name].columns], evidence[name], check_exact=True)
    for column in evidence["bootstrap"]:
        if column != "lower_bound_positive":
            pd.testing.assert_series_equal(result["bootstrap"][column], evidence["bootstrap"][column], check_exact=True)
    assert result["excess_resolution"]["new_bootstrap_draws"] == 0
    for name in ("metrics", "annual", "rolling_summary", "bootstrap", "gates"):
        pd.testing.assert_frame_equal(evidence[name], before[name], check_exact=True)
    for cost in candidate:
        pd.testing.assert_frame_equal(candidate[cost], old_candidate[cost], check_exact=True)
        pd.testing.assert_frame_equal(benchmark[cost], old_benchmark[cost], check_exact=True)


def test_bootstrap_floor_uses_cumulative_not_daily_units():
    candidate, benchmark = histories(1e-11)
    evidence = evidence_for(candidate, benchmark)
    assert (evidence["bootstrap"].daily_lower_bound < 1e-8).all()
    assert (evidence["bootstrap"].daily_lower_bound * len(candidate[10.]) > 1e-8).all()
    result = apply_excess_resolution(evidence, candidate, benchmark)
    assert result["bootstrap"].lower_bound_positive.all()
    assert result["gates"].aggregate_positive_net_log_excess.all()
    # Annual/252-session effects still do not clear their own cumulative floor.
    assert result["gates"].positive_years.eq(0).all()
    assert result["rolling_summary"].positive_windows.eq(0).all()
    np.testing.assert_array_equal(result["bootstrap"].cumulative_lower_bound,
                                  evidence["bootstrap"].sessions * evidence["bootstrap"].daily_lower_bound)


def test_failing_risk_gate_cannot_be_repaired_by_resolution():
    candidate, benchmark = histories(.0003)
    for cost in candidate:
        candidate[cost]["strategy_return"] = benchmark[cost].strategy_return * 3 + .001
    evidence = evidence_for(candidate, benchmark)
    result = apply_excess_resolution(evidence, candidate, benchmark)
    assert not evidence["gates"].cvar5_not_worse_than_soxx.any()
    pd.testing.assert_series_equal(result["gates"].cvar5_not_worse_than_soxx, evidence["gates"].cvar5_not_worse_than_soxx)
    assert not result["historical_gate_pass"]


def test_floor_boundary_is_exclusive_and_negative_noise_counts_as_unresolved():
    candidate, benchmark = histories(.0003)
    evidence = evidence_for(candidate, benchmark)
    floor = float(evidence["annual"].iloc[0].log_excess_sum)
    result = apply_excess_resolution(evidence, candidate, benchmark, log_excess_floor=floor)
    assert result["annual"].iloc[0].within_resolution
    assert not result["annual"].iloc[0].resolved_positive_excess
    dates = candidate[10.].signal_date
    increments = np.where(dates.dt.year.eq(2016), -1e-12, .0003)
    for cost in candidate:
        candidate[cost]["strategy_return"] = np.expm1(np.log1p(benchmark[cost].strategy_return) + increments)
    evidence = evidence_for(candidate, benchmark)
    result = apply_excess_resolution(evidence, candidate, benchmark)
    negative = result["annual"].loc[result["annual"].year.eq(2016)]
    assert negative.within_resolution.all() and not negative.raw_positive_excess.any()
    assert not negative.resolved_positive_excess.any()


@pytest.mark.parametrize("floor", [0., -1., np.nan, np.inf, True, "1e-8", [1e-8]])
def test_invalid_floor_fails_closed(floor):
    with pytest.raises(ValueError, match="positive finite numeric"):
        apply_excess_resolution({}, {}, {}, log_excess_floor=floor)


@pytest.fixture
def bound_evidence():
    candidate, benchmark = histories(.0003)
    return evidence_for(candidate, benchmark), candidate, benchmark


@pytest.mark.parametrize("corruption", ["metric", "annual", "rolling", "bootstrap_mean", "bootstrap_units",
    "bootstrap_count", "duplicate_cost", "cost", "date", "nan_return", "gate_count", "gate_sign", "sample_count"])
def test_mismatched_evidence_or_paths_are_rejected(bound_evidence, corruption):
    evidence, candidate, benchmark = bound_evidence
    if corruption == "metric": evidence["metrics"].loc[0, "log_excess_sum"] += .001
    elif corruption == "annual": evidence["annual"].loc[0, "return_pct"] += 1.
    elif corruption == "rolling": evidence["rolling_summary"].loc[0, "positive_windows"] -= 1
    elif corruption == "bootstrap_mean": evidence["bootstrap"].loc[0, "daily_mean_log_excess"] += .01
    elif corruption == "bootstrap_units": evidence["bootstrap"].loc[0, "annualized_lower_bound"] *= 2
    elif corruption == "bootstrap_count": evidence["bootstrap"].loc[0, "sessions"] -= 1
    elif corruption == "duplicate_cost": evidence["metrics"] = pd.concat([evidence["metrics"], evidence["metrics"].iloc[[0]]], ignore_index=True)
    elif corruption == "cost": candidate.pop(25.)
    elif corruption == "date": candidate[10.].loc[len(candidate[10.]) - 1, "exit_date"] += pd.offsets.BDay(1)
    elif corruption == "nan_return": candidate[10.].loc[0, "strategy_return"] = np.nan
    elif corruption == "gate_count": evidence["gates"].loc[0, "positive_years"] -= 1
    elif corruption == "gate_sign": evidence["gates"].loc[0, "aggregate_positive_net_log_excess"] = False
    else: evidence["expected_signal_sessions"] -= 1
    with pytest.raises(ValueError, match="blocked"):
        apply_excess_resolution(evidence, candidate, benchmark)


def test_resolution_cannot_be_reapplied_with_a_new_threshold(bound_evidence):
    evidence, candidate, benchmark = bound_evidence
    result = apply_excess_resolution(evidence, candidate, benchmark)
    with pytest.raises(ValueError, match="applied twice"):
        apply_excess_resolution(result, candidate, benchmark, log_excess_floor=1e-9)

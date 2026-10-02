"""Cross-layer checks for the registered conditional-variance experiment."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.data.conditional_variance import (
    apply_calibration, build_calibration_dataset, build_raw_variances,
    calibrate_annual, primary_frame_from_qualified_qlib,
)
from src.model.convex_allocation import DPPConvexAllocationSolver
from src.workflow import conditional_variance as workflow


def _calibrated_case():
    dates = pd.bdate_range("2016-01-04", periods=3)
    risk = pd.DataFrame({
        "cov_soxl_soxl": .0009, "cov_soxl_soxx": .0002, "cov_soxx_soxx": .0001,
        "m_expanding": .0001, "m_open20": .0001, "m_close20": .0001,
        "m_learned": [.0001, .0004, .000025],
        "forecast_available": 1., "future_vol_prediction": [.01, .02, .005],
    }, index=dates)
    calibration = {
        "cutoff": "2016-01-01T00:00:00-05:00", "B": .0001,
        "mu_H5": [0., .05],
        "audit": {"target_soxx_return_mean": .0003, "common_dates_sha256": "fixture-2016"},
        "c": {name: 1. for name in ("expanding", "open20", "close20", "learned")},
        "support": {name: {"min": .00005, "max": .0003}
                    for name in ("expanding", "open20", "close20", "learned")},
    }
    return calibration, risk


def _solve_risk(prepared):
    solver = DPPConvexAllocationSolver(cost_bps=25., budget_mode="external_positive")
    return [solver.solve([0., 0., 1.], [0., .05], sigma, risk_budget_variance=budget, robust_radius=0.)
            for sigma, budget in zip(prepared["covariance"], prepared["benchmark_variance"])]


def test_calibrated_risk_timing_survives_solver_and_support_flags_do_not_clip():
    calibration, risk = _calibrated_case()
    prepared = apply_calibration(calibration, risk, "learned")
    assert prepared["out_of_support"].tolist() == [False, True, True]
    assert not prepared["fallback"].any()
    results = _solve_risk(prepared)
    assert all(row["optimality_verified"] for row in results)
    weights = np.asarray([row["target_net_weights"] for row in results])
    # A fourfold variance increases predicted volatility twofold: the same
    # annual budget admits half SOXX, while low variance permits full SOXX.
    np.testing.assert_allclose(weights[:, 1], [1., .5, 1.], atol=1e-7, rtol=0)
    for weight, sigma, budget in zip(weights, prepared["covariance"], prepared["benchmark_variance"]):
        assert weight[:2] @ sigma @ weight[:2] / budget <= 1 + 1e-8


def test_common_calibration_budget_scale_is_not_an_independent_exposure_parameter():
    calibration, risk = _calibrated_case()
    before = apply_calibration(calibration, risk, "learned")
    scaled = copy.deepcopy(calibration)
    scaled["B"] *= 7.
    scaled["c"] = {name: coefficient * 7. for name, coefficient in scaled["c"].items()}
    after = apply_calibration(scaled, risk, "learned")
    np.testing.assert_allclose(
        before["covariance"] / before["benchmark_variance"][:, None, None],
        after["covariance"] / after["benchmark_variance"][:, None, None], rtol=1e-15)
    original = _solve_risk(before)
    transformed = _solve_risk(after)
    assert all(row["optimality_verified"] for row in original + transformed)
    np.testing.assert_allclose([row["target_net_weights"] for row in original],
                               [row["target_net_weights"] for row in transformed], atol=1e-8, rtol=0)


@pytest.fixture(scope="module")
def primary_history():
    dates = pd.bdate_range("2010-01-01", "2014-12-31", name="datetime")
    t = np.arange(len(dates))
    parts = []
    for symbol, amplitude in (("SOXL", .024), ("SOXX", .009)):
        opening = 25. * np.exp(np.cumsum(.0002 + amplitude * np.sin(t * .37)))
        closing = opening * (1 + .005 * np.cos(t * .13))
        part = pd.DataFrame({"$open": opening, "$close": closing,
                             "$high": np.maximum(opening, closing) * 1.01,
                             "$low": np.minimum(opening, closing) * .99,
                             "$volume": 1e6 * np.exp(.2 * np.sin(t * .19)),
                             "instrument": symbol}, index=dates)
        parts.append(part.reset_index().set_index(["instrument", "datetime"]))
    raw = pd.concat(parts).sort_index()
    frame, prices, _ = primary_frame_from_qualified_qlib(raw, dates)
    return frame, prices


def test_oof_repetitions_freshly_fit_and_future_labels_cannot_affect_earlier_models(primary_history):
    frame, _ = primary_history
    results = [workflow.fit_oof_forecasts(frame, as_of="2014-12-31") for _ in range(3)]
    assert all({2013, 2014}.issubset(models) for _, _, models in results)
    assert all(set(models) == set(results[0][2]) for _, _, models in results)
    assert len({id(models[2013]) for _, _, models in results}) == 3
    for augmented, audit, _ in results[1:]:
        pd.testing.assert_frame_equal(results[0][0], augmented)
        for row in audit.loc[audit.available].itertuples():
            assert pd.Timestamp(row.label_available_at_max) < pd.Timestamp(row.decision_at)
    assert workflow.verify_model_repetitions(results[0][2], results[1][2])["all_values_match"]
    changed = frame.copy()
    future = pd.to_datetime(changed.forecast_available_at, utc=True).ge(pd.Timestamp("2013-01-01", tz="America/New_York"))
    changed.loc[future, ["cash_advantage_1d", "future_vol_5d"]] = [.8, 2.]
    updated, _, altered_models = workflow.fit_oof_forecasts(changed, as_of="2014-12-31")
    columns = ["cash_advantage_prediction", "future_vol_prediction", "forecast_available"]
    pd.testing.assert_frame_equal(results[0][0].loc["2013", columns], updated.loc["2013", columns])
    np.testing.assert_array_equal(results[0][2][2013][-1].coef_, altered_models[2013][-1].coef_)


@pytest.mark.parametrize("bad_prediction", [np.nan, -0.01])
def test_invalid_oof_output_preserves_common_dates_and_control_calibration(primary_history, monkeypatch, bad_prediction):
    from sklearn.linear_model import Ridge

    frame, prices = primary_history
    normal, _, _ = workflow.fit_oof_forecasts(frame, as_of="2014-12-31")
    normal_risk = build_raw_variances(normal, prices)
    expected = calibrate_annual(build_calibration_dataset(normal, normal_risk, as_of="2014-12-31"), 2014)
    original = Ridge.predict

    def corrupted(model, features):
        value = original(model, features)
        value[:, 1] = bad_prediction
        return value

    monkeypatch.setattr(Ridge, "predict", corrupted)
    forecast, audit, models = workflow.fit_oof_forecasts(frame, as_of="2014-12-31")
    assert forecast.loc["2013", "forecast_available"].eq(1).all()
    assert audit.loc[audit.model_year.eq(2013), "invalid_vol_prediction_count"].item() == len(forecast.loc["2013"])
    assert 2013 in models
    risk = build_raw_variances(forecast, prices)
    actual = calibrate_annual(build_calibration_dataset(forecast, risk, as_of="2014-12-31"), 2014)
    assert actual["audit"]["common_dates"] == expected["audit"]["common_dates"]
    assert actual["B"] == expected["B"]
    assert actual["c"]["learned"] is None
    for control in ("expanding", "open20", "close20"):
        assert actual["c"][control] == expected["c"][control]


def test_frozen_calibration_keeps_all_parameters_while_observed_risk_can_move():
    calibration, risk = _calibrated_case()
    dates = pd.to_datetime(["2016-12-30", "2017-01-02", "2017-01-03"])
    risk.index = dates
    market = pd.DataFrame(index=dates)
    newer = copy.deepcopy(calibration)
    newer.update({"cutoff": "2017-01-01T00:00:00-05:00", "B": .0002, "mu_H5": [.04, .02]})
    newer["c"] = {name: 2. for name in newer["c"]}
    newer["audit"].update({"target_soxx_return_mean": .0007, "common_dates_sha256": "fixture-2017"})
    annual = workflow.prepare_policy_inputs(market, risk, {2016: calibration, 2017: newer}, "learned")
    frozen = workflow.prepare_policy_inputs(market, risk, {2016: calibration, 2017: newer}, "learned", frozen_year=2016)
    np.testing.assert_array_equal(annual["calibration_year"], [2016, 2017, 2017])
    np.testing.assert_array_equal(frozen["calibration_year"], [2016] * 3)
    np.testing.assert_array_equal(frozen["budget"], [.0001] * 3)
    np.testing.assert_array_equal(frozen["means"], [calibration["mu_H5"]] * 3)
    np.testing.assert_array_equal(frozen["return_center"], [.0003] * 3)
    assert all(item["common_dates_sha256"] == "fixture-2016" for item in frozen["budget_provenance"])
    assert frozen["covariance"][1, 1, 1] == pytest.approx(4 * frozen["covariance"][0, 1, 1])


def _market_and_inputs():
    calibration, risk = _calibrated_case()
    dates = risk.index
    market = pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
                           "exit_date": dates + pd.offsets.BDay(2),
                           "soxl_return": [.06, -.12, .03], "soxx_return": [.02, -.04, .01]}, index=dates)
    inputs = workflow.prepare_policy_inputs(market, risk, {2016: calibration}, "learned")
    return market, inputs


def test_each_cost_path_carries_own_drift_and_estimator_fallback_is_never_dropped():
    market, inputs = _market_and_inputs()
    inputs["estimator_fallback"][1] = True
    paths = {}
    for cost in (10., 25.):
        path, audit, details = workflow.replay_policy(market, inputs, cost_bps=cost)
        assert len(path) == len(details) == len(market)
        assert audit["all_optimality_verified"]
        assert audit["estimator_fallback_sessions"] == 1
        assert path.estimator_fallback.tolist() == [False, True, False]
        np.testing.assert_allclose(path.fees, path.turnover * cost / 1e4, rtol=0, atol=1e-14)
        np.testing.assert_allclose(path[[f"pretrade_{asset}_weight" for asset in workflow.ASSETS]].iloc[1:],
                                   path[[f"ending_{asset}_weight" for asset in workflow.ASSETS]].iloc[:-1], rtol=0, atol=0)
        np.testing.assert_allclose(path.equity, np.cumprod(1 + path.strategy_return), rtol=0, atol=1e-14)
        assert path.risk_budget_ratio.le(1 + 1e-8).all()
        paths[cost] = path
    assert paths[25.].equity.iloc[-1] < paths[10.].equity.iloc[-1]


def test_replay_rejects_same_length_inputs_from_different_dates():
    market, inputs = _market_and_inputs()
    shifted = market.copy()
    shifted.index = shifted.index + pd.offsets.BDay(1)
    with pytest.raises(ValueError, match="exact signal dates"):
        workflow.replay_policy(shifted, inputs, cost_bps=10.)


def test_forecast_reproduction_keeps_invalid_values_explicit_and_never_masks_changes():
    dates = pd.bdate_range("2016-01-04", periods=4)
    reference = pd.DataFrame({"cash_advantage_prediction": [0., .01, .02, .03],
                              "future_vol_prediction": [np.nan, np.inf, -np.inf, .02],
                              "forecast_available": 1.,
                              "forecast_model_available_at": pd.Timestamp("2016-01-01", tz="UTC")}, index=dates)
    original = reference.copy(deep=True)
    result = workflow.verify_forecast_repetitions(reference, reference.copy())
    assert result["all_values_match"] and result["matching_nonfinite_forecast_values"] == 3
    assert result["predictions_modified"] is False
    pd.testing.assert_frame_equal(reference, original)
    for number, replacement in ((0, 0.), (1, -np.inf), (2, np.nan), (3, .03)):
        changed = reference.copy()
        changed.iloc[number, changed.columns.get_loc("future_vol_prediction")] = replacement
        with pytest.raises(ValueError, match="reproduction"):
            workflow.verify_forecast_repetitions(reference, changed)
    changed = reference.copy()
    changed.iloc[0, changed.columns.get_loc("forecast_available")] = 0.
    with pytest.raises(ValueError, match="reproduction"):
        workflow.verify_forecast_repetitions(reference, changed)


def test_model_reproduction_checks_parameters_and_feature_schema(primary_history):
    models = workflow.fit_oof_forecasts(primary_history[0], as_of="2014-12-31")[2]
    changed = copy.deepcopy(models)
    changed[2013][-1].coef_[0, 0] += .1
    with pytest.raises(ValueError, match="reproduction"):
        workflow.verify_model_repetitions(models, changed)
    changed = copy.deepcopy(models)
    changed[2013][0].feature_names_in_[0] = "wrong_input"
    with pytest.raises(ValueError, match="feature schema"):
        workflow.verify_model_repetitions(models, changed)


@pytest.mark.parametrize("column,value", [("soxx_return", -1.), ("future_vol_5d", -.01), ("future_vol_5d", np.nan)])
def test_oof_fit_rejects_bad_mature_rows_instead_of_shortening_training(primary_history, column, value):
    frame = primary_history[0].copy()
    frame.loc["2011-06-01", column] = value
    with pytest.raises(ValueError, match="mature OOF source rows cannot be dropped"):
        workflow.fit_oof_forecasts(frame, as_of="2014-12-31")


def _profiles():
    return pd.DataFrame([{"repeat_id": repeat, "cost_bps": cost,
                          "all_optimality_verified": True, "solver_fallback_sessions": 0,
                          "estimator_fallback_sessions": 0}
                         for repeat in (1, 2, 3) for cost in (10., 25.)])


def _economic_pass(monkeypatch):
    from src.analysis import integrated_allocation as analysis

    def passing(paths, benchmarks, *, spec, seed_table):
        assert spec.family_size == 14
        assert spec.expected_seeds == (1, 2, 3)
        assert spec.minimum_positive_seeds == 3
        return {"gates": pd.DataFrame({"cost_bps": [10., 25.], "all_pass": True}),
                "metrics": pd.DataFrame({"cost_bps": [10., 25.], "log_excess_sum": [1., 1.]}),
                "historical_qualification_pass": True}

    monkeypatch.setattr(analysis, "stable_excess_gates", passing)


@pytest.mark.parametrize("failure", ["estimator", "solver", "solver_flag", "repetition"])
def test_economic_success_never_overrides_any_integrity_failure(monkeypatch, failure):
    _economic_pass(monkeypatch)
    profiles = _profiles()
    if failure == "estimator": profiles.loc[0, "estimator_fallback_sessions"] = 1
    if failure == "solver": profiles.loc[0, "solver_fallback_sessions"] = 1
    if failure == "solver_flag": profiles.loc[0, "all_optimality_verified"] = False
    result = workflow.variance_gates({10.: None, 25.: None}, {}, pd.DataFrame(), profiles,
                                     repeats_verified=failure != "repetition")
    assert not result["historical_qualification_pass"]
    assert not result["gates"].all_pass.any()


@pytest.mark.parametrize("failure", ["missing", "duplicate", "negative_count", "missing_count", "missing_solver_flag", "string_solver_flag"])
def test_incomplete_or_malformed_profile_evidence_fails_closed(monkeypatch, failure):
    _economic_pass(monkeypatch)
    profiles = _profiles()
    if failure == "missing": profiles = profiles.iloc[:-1]
    if failure == "duplicate": profiles = pd.concat([profiles, profiles.iloc[:1]])
    if failure == "negative_count": profiles.loc[0, "estimator_fallback_sessions"] = -1
    if failure == "missing_count": profiles.loc[0, "solver_fallback_sessions"] = np.nan
    if failure in {"missing_solver_flag", "string_solver_flag"}:
        profiles["all_optimality_verified"] = profiles.all_optimality_verified.astype(object)
        profiles.loc[0, "all_optimality_verified"] = np.nan if failure == "missing_solver_flag" else "False"
    with pytest.raises(ValueError, match="profile|explicit booleans"):
        workflow.variance_gates({10.: None, 25.: None}, {}, pd.DataFrame(), profiles, repeats_verified=True)


@pytest.mark.parametrize("value", [np.nan, "False"])
def test_missing_or_string_reproduction_flag_cannot_claim_verification(monkeypatch, value):
    _economic_pass(monkeypatch)
    with pytest.raises(ValueError, match="boolean|bool|reproduction|repeat"):
        workflow.variance_gates({10.: None, 25.: None}, {}, pd.DataFrame(), _profiles(), repeats_verified=value)


def _attribution_paths():
    n = 96
    t = np.arange(n)
    benchmark_return = .001 + .012 * np.sin(.43 * t) + .007 * np.cos(.19 * t)
    innovations = np.random.default_rng(803).normal(0., .003, n)
    residual = np.zeros(n)
    for number in range(1, n):
        residual[number] = .7 * residual[number - 1] + innovations[number] * (1 + abs(benchmark_return[number]) * 25)
    dates = pd.bdate_range("2016-01-04", periods=n)
    benchmark = pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
                              "exit_date": dates + pd.offsets.BDay(2), "strategy_return": benchmark_return})
    path = benchmark.copy()
    path["strategy_return"] = .0004 + 1.35 * benchmark_return + residual
    return path, benchmark


def test_hac_attribution_matches_centered_ols_and_dense_bartlett_reference():
    path, benchmark = _attribution_paths()
    result = workflow.hac_attribution(path, benchmark)
    x = benchmark.strategy_return.to_numpy()
    y = path.strategy_return.to_numpy()
    n = len(x)
    centered = x - x.mean()
    beta = np.dot(centered, y - y.mean()) / np.dot(centered, centered)
    alpha = y.mean() - beta * x.mean()
    residual = y - alpha - beta * x
    # Intercept's linear OLS influence weights give the same sandwich without
    # using a Gram inverse, score matrix, or lag-sum implementation.
    influence = (1 / n - x.mean() * centered / np.dot(centered, centered)) * residual
    distance = abs(np.subtract.outer(np.arange(n), np.arange(n)))
    kernel = np.maximum(1 - distance / 21., 0.)
    reference_variance = influence @ kernel @ influence * n / (n - 2)
    critical = 1.959963984540054
    reference_bounds = (alpha + np.array([-1., 1.]) * critical * np.sqrt(reference_variance)) * 252
    assert result["realized_beta"] == pytest.approx(beta, rel=1e-13)
    assert result["annualized_arithmetic_alpha"] == pytest.approx(alpha * 252, abs=1e-14)
    np.testing.assert_allclose([result["alpha95_lower"], result["alpha95_upper"]],
                               reference_bounds, rtol=1e-12, atol=1e-14)
    assert result["small_sample_correction"] == "N/(N-2)"
    assert result["hac_lags"] == 20 and result["sessions"] == n


@pytest.mark.parametrize("failure", ["rank", "nonfinite", "mismatched_dates", "missing_dates", "duplicate_dates", "reversed_dates"])
def test_hac_rejects_unidentifiable_or_invalid_ordered_observations(failure):
    path, benchmark = _attribution_paths()
    if failure == "rank": benchmark["strategy_return"] = .001
    if failure == "nonfinite": path.loc[0, "strategy_return"] = np.inf
    if failure == "mismatched_dates": path.loc[0, "signal_date"] += pd.offsets.BDay(1)
    if failure == "missing_dates":
        path.loc[0, "signal_date"] = benchmark.loc[0, "signal_date"] = pd.NaT
    if failure == "duplicate_dates":
        for frame in (path, benchmark):
            frame.loc[1, ["signal_date", "trade_date", "exit_date"]] = frame.loc[0, ["signal_date", "trade_date", "exit_date"]]
    if failure == "reversed_dates":
        path, benchmark = path.iloc[::-1].reset_index(drop=True), benchmark.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="HAC|attribution|date|ordered"):
        workflow.hac_attribution(path, benchmark)

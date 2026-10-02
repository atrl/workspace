import numpy as np
import pandas as pd
import pytest

from src.data.convex_allocation import FEATURE_COLUMNS, LABEL_COLUMNS, build_horizon_dataset, build_horizon_frame
from src.data.integrated_allocation import PRIMARY_FEATURES
from src.workflow import convex_excess as workflow


@pytest.fixture(scope="module")
def horizon():
    rng = np.random.default_rng(23)
    calendar = pd.bdate_range("2011-01-03", periods=1050)
    dates = pd.Series(calendar, index=calendar)
    common = rng.normal(.0003, .01, len(calendar))
    prices = pd.DataFrame({("open", "SOXL"): 20 * np.cumprod(1 + 3 * common),
                           ("open", "SOXX"): 100 * np.cumprod(1 + common)}, index=calendar)
    frame = pd.DataFrame(rng.normal(size=(len(calendar), len(FEATURE_COLUMNS))), index=calendar, columns=FEATURE_COLUMNS)
    frame.index.name = "datetime"
    for name in PRIMARY_FEATURES[-3:]:
        frame[name] = 0.
    frame["signal_date"] = dates
    frame["trade_date"], frame["exit_date"] = dates.shift(-1), dates.shift(-2)
    for name, days, hour, minute in [("feature_available_at", dates, 16, 0), ("decision_at", dates.shift(-1), 9, 31),
                                    ("return_available_at", dates.shift(-2), 9, 30), ("forecast_available_at", dates.shift(-6), 9, 30),
                                    ("auxiliary_available_at", dates.shift(-5), 16, 0)]:
        frame[name] = pd.Series(pd.DatetimeIndex(days).tz_localize("America/New_York") + pd.Timedelta(hours=hour, minutes=minute), index=calendar)
    for symbol in ("SOXL", "SOXX"):
        frame[f"{symbol.lower()}_return"] = prices[("open", symbol)].shift(-2) / prices[("open", symbol)].shift(-1) - 1
    result, _ = build_horizon_frame(frame, prices)
    return result


def test_repeated_forecast_runs_really_refit_and_future_labels_cannot_enter(horizon, monkeypatch):
    from src.data import convex_allocation as data

    observed = []
    original = data.fit_annual_estimator

    def fitted(*args, **kwargs):
        fit = original(*args, **kwargs)
        observed.append(fit)
        return fit

    monkeypatch.setattr(data, "fit_annual_estimator", fitted)
    dataset = build_horizon_dataset(horizon, as_of=str(horizon.index.max().date()))
    market = horizon.loc[horizon.index.year == 2013]
    runs = [workflow.fit_predict_repeat(dataset, market, [2013]) for _ in range(3)]
    assert len(observed) == 3 and len({id(fit) for fit in observed}) == 3
    for _, predictions, audits in runs[1:]:
        assert workflow.verify_repetitions(runs[0][1], predictions)["all_values_match"]
        assert pd.Timestamp(audits[0]["last_label_available_at"]) < pd.Timestamp("2013-01-01", tz="America/New_York")
    changed = horizon.copy()
    changed.loc[changed.h5_available_at.ge(pd.Timestamp("2013-01-01", tz="America/New_York")), list(LABEL_COLUMNS)] = .9
    altered = build_horizon_dataset(changed, as_of=str(changed.index.max().date()))
    _, prediction, _ = workflow.fit_predict_repeat(altered, market, [2013])
    pd.testing.assert_frame_equal(runs[0][1], prediction)


def _market():
    dates = pd.bdate_range("2016-01-04", periods=4)
    return pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
                         "exit_date": dates + pd.offsets.BDay(2), "soxl_return": [.03, -.06, .015, .03],
                         "soxx_return": [.01, -.02, .005, .01],
                         "cov_soxl_soxl": .0009, "cov_soxl_soxx": .0003, "cov_soxx_soxx": .0001}, index=dates)


def test_fresh_convex_replays_match_and_feasible_soxx_has_exact_fee_accounting():
    frame = _market()
    means = np.tile([0., .05], (len(frame), 1))
    runs = [workflow.replay_predictions(frame, means, cost_bps=25.) for _ in range(3)]
    reference = workflow.benchmark_path(frame, cost_bps=25.)
    for path, audit in runs:
        assert audit["all_optimality_verified"] and audit["fallback_sessions"] == 0
        assert audit["minimum_null_gap"] >= -1e-8
        assert audit["maximum_predicted_variance_ratio"] <= 1 + 1e-8
        np.testing.assert_allclose(path.strategy_return, reference.strategy_return, rtol=0, atol=1e-8)
        assert workflow.verify_repetitions(runs[0][0], path)["all_values_match"]


def test_failed_solver_is_retained_as_unverified_not_dropped_or_promoted(monkeypatch):
    from src.model.convex_allocation import DPPConvexAllocationSolver

    def failure(*args):
        raise RuntimeError("solver failure fixture")

    monkeypatch.setattr(DPPConvexAllocationSolver, "_invoke", failure)
    frame = _market()
    path, audit = workflow.replay_predictions(frame, np.tile([0., .05], (len(frame), 1)), cost_bps=10.)
    assert len(path) == len(frame)
    assert not audit["all_optimality_verified"]
    assert audit["fallback_sessions"] == len(frame)
    assert not path.optimality_verified.any()
    assert all(row["status"] == "fallback_feasible_null" for row in audit["solver_audits"])


@pytest.mark.parametrize("mu", [np.array([[0., np.nan]] * 4), np.array([[0., -1.]] * 4), np.zeros((3, 2))])
def test_invalid_forecasts_abort_instead_of_clipping_or_dropping(mu):
    with pytest.raises(ValueError, match="invalid H5 forecasts"):
        workflow.replay_predictions(_market(), mu, cost_bps=10.)


def test_reproduction_rejects_changed_dates_flags_values_and_nonfinite_data():
    reference = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "target": [.2, .3], "verified": [True, True]})
    for column, replacement in (("date", pd.Timestamp("2020-01-04")), ("target", .21), ("verified", False), ("target", np.nan)):
        changed = reference.copy()
        changed.loc[0, column] = replacement
        with pytest.raises(ValueError, match="reproduction|nonfinite"):
            workflow.verify_repetitions(reference, changed)
    changed = reference.copy()
    changed.loc[0, "target"] += 5e-9
    result = workflow.verify_repetitions(reference, changed)
    assert result["all_values_match"] and result["independent_statistical_evidence"] is False


def test_missing_annual_predictions_cannot_silently_shorten_coverage():
    dates = pd.to_datetime(["2016-12-30", "2017-01-03"])
    frame = pd.DataFrame(index=dates)
    forecast = {"prior": np.zeros((1, 2)), "ridge": np.zeros((1, 2)), "Vmean": np.zeros((1, 2, 2))}
    with pytest.raises(ValueError, match="incomplete prediction coverage"):
        workflow.assemble_predictions(frame, {2016: forecast})


@pytest.mark.parametrize("solver_verified,repeats_verified", [(False, True), (True, False), (False, False)])
def test_economic_pass_cannot_override_unverified_solver_or_reproduction(monkeypatch, solver_verified, repeats_verified):
    from src.analysis import integrated_allocation as analysis

    seen = []
    def qualified(paths, benchmarks, *, spec, seed_table):
        seen.append(spec)
        return {"gates": pd.DataFrame({"cost_bps": [10., 25.], "all_pass": True,
                    "minimum_positive_seeds_at_both_costs": True, "seed_evidence_complete": True}),
                "metrics": pd.DataFrame({"cost_bps": [10., 25.], "log_excess_sum": [1., 1.]}),
                "historical_qualification_pass": True}
    monkeypatch.setattr(analysis, "stable_excess_gates", qualified)
    repeated = pd.DataFrame([{"repeat_id": repeat, "cost_bps": cost, "log_excess_sum": 1.}
                             for repeat in (1, 2, 3) for cost in (10., 25.)])
    result = workflow.deterministic_gates({}, {}, repeated, solver_verified=solver_verified,
                                          repeats_verified=repeats_verified)
    assert not result["historical_qualification_pass"] and not result["gates"].all_pass.any()
    assert seen[0].minimum_positive_seeds == 3 and seen[0].expected_seeds == (1, 2, 3)
    assert seen[0].family_size == 10
    assert "no extra statistical observations" in result["reproduction_evidence"]


def test_numerical_benchmark_noise_is_not_qualified_as_measurable_excess(monkeypatch):
    from src.analysis import integrated_allocation as analysis

    def qualified(*args, **kwargs):
        return {"gates": pd.DataFrame({"cost_bps": [10., 25.], "all_pass": True}),
                "metrics": pd.DataFrame({"log_excess_sum": [1e-12, 1e-12]}),
                "historical_qualification_pass": True}

    monkeypatch.setattr(analysis, "stable_excess_gates", qualified)
    result = workflow.deterministic_gates({}, {}, pd.DataFrame(), solver_verified=True, repeats_verified=True)
    assert not result["historical_qualification_pass"]
    assert not result["gates"].measurable_log_excess_above_numerical_tolerance.any()

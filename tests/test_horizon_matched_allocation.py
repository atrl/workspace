"""Fixed-horizon execution contracts on synthetic Qlib observations."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import horizon_matched_allocation as replay
from src.data.benchmark_relative_data import build_relative_dataset
from src.data.qlib_native import build_qlib_store_from_frame


@pytest.fixture(scope="module")
def native_inputs(tmp_path_factory):
    dates = pd.bdate_range("2024-03-04", periods=17)
    source = []
    for symbol, prices in (
        ("SOXL", [100, 101, 125, 120, 140, 130, 135, 138, 128, 133, 143, 140, 138, 145, 146, 144, 145]),
        ("SOXX", [100, 101, 108, 106, 112, 109, 111, 112, 109, 111, 114, 113, 112, 114, 115, 114, 115]),
    ):
        for date, price in zip(dates, prices):
            source.append({"datetime": date, "instrument": symbol, "open": float(price),
                "close": float(price), "high": price * 1.01, "low": price * .99, "volume": 1e6})
    provider = tmp_path_factory.mktemp("horizon_provider") / "provider"
    build_qlib_store_from_frame(pd.DataFrame(source), provider_uri=provider,
        as_of=str(dates[-1].date()), source="synthetic H5 funding and HOLD test")
    _, frame, _ = build_relative_dataset(provider_uri=provider,
        signal_start=str(dates[0].date()), as_of=str(dates[-1].date()))
    # Leave a native calendar right boundary after the final evaluated exit.
    frame = frame.iloc[:-1].copy()
    count = len(frame)
    means = np.tile([.04, .008], (count, 1))
    uncertainty = np.zeros((count, 2, 2))
    risk = np.tile(np.array([[.0009, .0003], [.0003, .0001]]), (count, 1, 1))
    return provider, frame, means, uncertainty, risk


def run(inputs, **kwargs):
    _, frame, means, uncertainty, risk = inputs
    options = dict(cost_bps=25., rebalance_sessions=5)
    options.update(kwargs)
    return replay.replay_horizon_allocation(frame, means, uncertainty, risk, **options)


@pytest.mark.parametrize("cost", [10., 25.])
def test_real_solver_and_native_qlib_reconcile_complete_funded_holdings(native_inputs, cost):
    from src.analysis.convex_qlib import native_open_reconciliation

    provider, frame, _, _, _ = native_inputs
    path, audit = run(native_inputs, cost_bps=cost)
    native = native_open_reconciliation(frame, path, provider_uri=provider, cost_bps=cost)
    assert native["audit"]["status"] == "passed"
    assert native["audit"]["maximum_session_return_error"] <= 1e-7
    assert audit["maximum_accounting_error"] < 1e-10
    assert audit["solver_calls"] == 3
    assert audit["all_solves_optimality_verified"] is True
    assert audit["native_qlib_verified"] is False  # only the external native call certifies it
    assert path.optimizer_called.tolist() == [index % 5 == 0 for index in range(len(frame))]
    assert (path.loc[path.optimizer_called, "risk_ratio"] <= 1 + 1e-8).all()


def test_hold_is_exact_and_drift_is_reported_without_rebalance(native_inputs, monkeypatch):
    real_solver = replay.ConvexAllocationSolver
    observed = []

    class TracedSolver:
        def __init__(self, **kwargs):
            self.inner = real_solver(**kwargs)

        def solve(self, held, means, covariance, **kwargs):
            observed.append(held.copy())
            return self.inner.solve(held, means, covariance, **kwargs)

    monkeypatch.setattr(replay, "ConvexAllocationSolver", TracedSolver)
    path, audit = run(native_inputs)
    hold = ~path.optimizer_called
    assert len(observed) == 3
    np.testing.assert_array_equal(path.loc[hold, "fees"], 0.)
    np.testing.assert_array_equal(path.loc[hold, "turnover"], 0.)
    for asset in ("soxl", "soxx", "cash"):
        np.testing.assert_array_equal(path.loc[hold, f"open_{asset}_weight"],
                                      path.loc[hold, f"pretrade_{asset}_weight"])
        np.testing.assert_array_equal(path[f"pretrade_{asset}_weight"].iloc[1:],
                                      path[f"ending_{asset}_weight"].iloc[:-1])
    assert audit["hold_risk_exceedance_sessions"] > 0
    assert not path.loc[hold, "risk_constraint_enforced"].any()
    assert path.loc[hold, "optimality_verified"].isna().all()
    assert path.loc[hold, "solution_source"].eq("exact_HOLD").all()


def test_future_forecasts_covariances_and_outcomes_cannot_change_prior_path(native_inputs):
    provider, frame, means, uncertainty, risk = native_inputs
    original, _ = run(native_inputs)
    changed_frame, changed_means = frame.copy(), means.copy()
    changed_uncertainty, changed_risk = uncertainty.copy(), risk.copy()
    changed_means[8:] = [-.03, .01]
    changed_uncertainty[8:] = np.eye(2) * .001
    changed_risk[8:] *= 2
    changed_frame.loc[changed_frame.index[8:], ["soxl_return", "soxx_return"]] = [.02, -.01]
    revised, _ = run((provider, changed_frame, changed_means, changed_uncertainty, changed_risk))
    pd.testing.assert_frame_equal(original.iloc[:8], revised.iloc[:8], check_exact=True)
    # Outcomes only mark an already selected allocation; they cannot set its target.
    np.testing.assert_allclose(original.loc[8, ["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]].to_numpy(float),
        revised.loc[8, ["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]].to_numpy(float), atol=0, rtol=0)


@pytest.mark.parametrize("corruption", ["unverified", "bad_fee", "wrong_execution", "risk_violation"])
def test_bad_solver_result_cannot_be_silently_replaced_by_benchmark(native_inputs, monkeypatch, corruption):
    real_solver = replay.ConvexAllocationSolver

    class CorruptSolver:
        def __init__(self, **kwargs):
            self.inner = real_solver(**kwargs)

        def solve(self, *args, **kwargs):
            result = self.inner.solve(*args, **kwargs)
            if corruption == "unverified":
                result["optimality_verified"] = False
            elif corruption == "bad_fee":
                result["fees"] += .01
            elif corruption == "wrong_execution":
                result["execution_target_weights"] = np.array([0., 1., 0.])
            else:
                result["target_net_weights"] = np.array([1., 0., 0.])
            return result

    monkeypatch.setattr(replay, "ConvexAllocationSolver", CorruptSolver)
    with pytest.raises(ValueError, match="blocked"):
        run(native_inputs)


@pytest.mark.parametrize("corruption", ["forecast_nan", "forecast_short", "forecast_impossible", "mean_nonpsd", "daily_nonpsd", "asymmetric", "nan_on_hold", "calendar_gap", "late_decision"])
def test_invalid_complete_inputs_fail_before_any_solver(native_inputs, monkeypatch, corruption):
    provider, frame, means, uncertainty, risk = native_inputs
    frame, means, uncertainty, risk = frame.copy(), means.copy(), uncertainty.copy(), risk.copy()
    if corruption == "forecast_nan": means[1, 0] = np.nan
    elif corruption == "forecast_short": means = means[:-1]
    elif corruption == "forecast_impossible": means[1, 0] = -1.
    elif corruption == "mean_nonpsd": uncertainty[1] = [[1, 2], [2, 1]]
    elif corruption == "daily_nonpsd": risk[1] = [[1, 2], [2, 1]]
    elif corruption == "asymmetric": risk[1, 0, 1] += .1
    elif corruption == "nan_on_hold": uncertainty[1, 0, 0] = np.nan
    elif corruption == "calendar_gap": frame = frame.drop(frame.index[3])
    else: frame["decision_at"] = pd.DatetimeIndex(frame.trade_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)

    def forbidden(**kwargs):
        pytest.fail("invalid full input must be rejected before constructing solver")

    monkeypatch.setattr(replay, "ConvexAllocationSolver", forbidden)
    with pytest.raises(ValueError, match="blocked"):
        run((provider, frame, means, uncertainty, risk))


@pytest.mark.parametrize("settings", [{"rebalance_sessions": 0}, {"rebalance_sessions": True},
    {"cost_bps": -1}, {"robust_radius": -1}, {"robust_radius": float("nan")}])
def test_invalid_settings_fail_closed(native_inputs, settings):
    with pytest.raises(ValueError, match="blocked"):
        run(native_inputs, **settings)


def test_daily_control_and_compact_covariance_are_explicit(native_inputs):
    provider, frame, means, uncertainty, risk = native_inputs
    compact = risk[:, [0, 0, 1], [0, 1, 1]]
    path, audit = run((provider, frame, means, uncertainty, compact), rebalance_sessions=1)
    assert path.optimizer_called.all()
    assert audit["solver_calls"] == len(frame)
    assert audit["forecast_horizon_sessions"] == 5
    assert audit["rebalance_sessions"] == 1

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.horizon_matched_allocation import replay_horizon_allocation
from src.model.benchmark_terminal_value import transform_benchmark_terminal_inputs
from src.workflow import benchmark_terminal_experiment as workflow


@pytest.fixture
def plan():
    return json.loads((Path(__file__).resolve().parents[1] /
        "research/benchmark_terminal_value_20261009/plan.json").read_text())


@pytest.fixture
def synthetic(plan):
    calendar = pd.bdate_range("2016-01-04", periods=16)
    dates = pd.Series(calendar, index=calendar)
    frame = pd.DataFrame(index=calendar)
    frame.index.name = "datetime"
    frame["signal_date"] = dates
    frame["trade_date"], frame["exit_date"] = dates.shift(-1), dates.shift(-2)
    frame["decision_at"] = calendar.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
    frame["return_available_at"] = pd.DatetimeIndex(frame.exit_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    frame["h5_available_at"] = pd.DatetimeIndex(dates.shift(-6)).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    frame["soxl_return"], frame["soxx_return"] = np.sin(np.arange(16)) * .025, np.sin(np.arange(16)) * .008
    market = frame.loc[frame.exit_date.notna()]
    count = len(market)
    rows = pd.DataFrame({"signal_date": market.signal_date, "forecast_available_at": market.decision_at,
        "label_available_at": market.h5_available_at, "model_year": 2016,
        "predicted_soxl": .025, "predicted_soxx": .009}, index=market.index)
    error = np.tile(np.array([[3e-6, .5e-6], [.5e-6, 1e-6]]), (count, 1, 1))
    sigma = np.tile(np.array([[.0016, .00035], [.00035, .00016]]), (len(frame), 1, 1))
    model = {"model_year": 2016, "model_available_at": "2016-01-01T00:00:00-05:00",
             "audit": {"last_label_available_at": "2015-12-31T09:31:00-05:00"}}
    bundle = {"frame": frame, "calendar": calendar, "covariance": sigma, "audit": {"synthetic": True}}
    objects = {"inputs/frame.pkl": frame.copy(deep=True),
        "forecast/forecasts.pkl": {"validation_selected": {"rows": rows, "Vmean": error}},
        "models/annual_fits.pkl": {2016: {"validation_selected": model}}, "analysis/paths.pkl": {}}
    return bundle, objects


def test_parent_reader_requests_all_four_preregistered_digests_before_any_objects(plan, monkeypatch, tmp_path):
    calls = []

    def verified(card, **kwargs):
        calls.append((card, kwargs))
        assert kwargs["artifacts"] == list(workflow.PARENT_ARTIFACTS)
        assert kwargs["expected_artifact_sha256"] == plan["parent"]["artifact_sha256"]
        return {"verified": True}, {"hashes_checked_before_unpickling": True}

    monkeypatch.setattr(workflow, "load_verified_recorder_objects", verified)
    objects, audit = workflow._load_parent(tmp_path, plan)
    assert len(calls) == 1 and objects["verified"] and audit["hashes_checked_before_unpickling"]


@pytest.mark.parametrize("case", ["frame", "forecast_date", "forecast_clock", "model_year", "model_cutoff", "uncertainty_shape"])
def test_frozen_source_forecasts_and_clocks_fail_closed(plan, synthetic, case):
    bundle, objects = deepcopy(synthetic)
    forecast = objects["forecast/forecasts.pkl"]["validation_selected"]
    if case == "frame":
        objects["inputs/frame.pkl"].iloc[0, -1] += .01
    elif case == "forecast_date":
        forecast["rows"] = forecast["rows"].iloc[1:]
    elif case == "forecast_clock":
        forecast["rows"].iloc[0, forecast["rows"].columns.get_loc("forecast_available_at")] += pd.Timedelta(minutes=1)
    elif case == "model_year":
        forecast["rows"]["model_year"] = 2017
    elif case == "model_cutoff":
        objects["models/annual_fits.pkl"][2016]["validation_selected"]["audit"]["last_label_available_at"] = "2016-01-01T00:00:00-05:00"
    else:
        forecast["Vmean"] = forecast["Vmean"][:-1]
    with pytest.raises(ValueError, match="blocked:"):
        workflow._frozen_inputs(bundle, objects, plan)


@pytest.mark.parametrize("cost", [10., 25.])
def test_real_synthetic_solver_direct_terminal_identity_and_current_fees(plan, synthetic, cost):
    bundle, objects = synthetic
    before = deepcopy(objects)
    market, means, error, sigma, audit = workflow._frozen_inputs(bundle, objects, plan)
    transformed = transform_benchmark_terminal_inputs(means, error, cost_bps=cost)
    path, solved = replay_horizon_allocation(market, transformed["transformed_mu"], transformed["transformed_covariance"],
        sigma, cost_bps=cost, rebalance_sessions=5, robust_radius=1.)
    verified = workflow.verify_terminal_valuations(path, solved["solver_details"], means, transformed,
        cost_bps=cost, daily_covariances=sigma)
    assert verified.objective_identity_error.max() <= 1e-12
    assert verified.post_fee_values_error.max() <= 1e-12
    assert not verified.future_conversion_fee_debited.any()
    np.testing.assert_allclose(path.fees, path.turnover * cost / 10000., rtol=0, atol=1e-12)
    np.testing.assert_array_equal(path.loc[~path.optimizer_called, ["fees", "turnover"]], 0.)
    assert audit["model_fits"] == 0 and audit["unknown_terminal_h5_predictions_retained"] == 4
    pd.testing.assert_frame_equal(objects["forecast/forecasts.pkl"]["validation_selected"]["rows"],
                                  before["forecast/forecasts.pkl"]["validation_selected"]["rows"], check_exact=True)
    np.testing.assert_array_equal(error, before["forecast/forecasts.pkl"]["validation_selected"]["Vmean"])


def test_direct_valuation_rejects_wrong_funded_values_or_objective(plan, synthetic):
    bundle, objects = synthetic
    market, means, error, sigma, _ = workflow._frozen_inputs(bundle, objects, plan)
    transformed = transform_benchmark_terminal_inputs(means, error, cost_bps=25.)
    path, audit = replay_horizon_allocation(market, transformed["transformed_mu"], transformed["transformed_covariance"],
        sigma, cost_bps=25., rebalance_sessions=5)
    for key in ("post_fee_values", "objective"):
        details = deepcopy(audit["solver_details"])
        details[0]["solution"][key] += 1e-5
        with pytest.raises(ValueError, match="blocked:"):
            workflow.verify_terminal_valuations(path, details, means, transformed, cost_bps=25., daily_covariances=sigma)
    details = deepcopy(audit["solver_details"])
    details[-1] = deepcopy(details[0])
    with pytest.raises(ValueError, match="exactly once"):
        workflow.verify_terminal_valuations(path, details, means, transformed, cost_bps=25., daily_covariances=sigma)


def test_resolution_replaces_only_qualification_tables_and_preserves_raw_statistics(plan, monkeypatch):
    raw = {"metrics": pd.DataFrame({"estimate": [1.]}), "annual": pd.DataFrame({"return_pct": [2.]}),
        "bootstrap": pd.DataFrame({"daily_lower_bound": [1e-12], "lower_bound_positive": [True]}),
        "rolling_summary": pd.DataFrame({"positive_windows": [4]}), "gates": pd.DataFrame({"all_pass": [True]})}
    tables = {"gates.csv": raw["gates"].copy(), "rolling_summary.csv": raw["rolling_summary"].copy(),
        "bootstrap_analysis.csv": raw["bootstrap"].copy(), "annual_metrics.csv": pd.DataFrame({"all_strategies": ["soxx", "control", "candidate"]})}
    paths = {(name, cost): pd.DataFrame({"signal_date": pd.to_datetime(["2016-01-04", "2023-01-03"])})
             for name in ("benchmark_terminal", "soxx") for cost in (10., 25.)}

    def resolve(evidence, candidates, benchmarks, **kwargs):
        assert evidence is raw and kwargs["log_excess_floor"] == 1e-8
        assert all(len(path) == 1 for path in [*candidates.values(), *benchmarks.values()])
        output = deepcopy(evidence)
        output["gates"]["all_pass"] = False
        output["rolling_summary"]["positive_windows"] = 0
        output["bootstrap"]["lower_bound_positive"] = False
        return output

    monkeypatch.setattr(workflow, "apply_excess_resolution", resolve)
    result, evidence = workflow.resolve_evidence(tables, {"benchmark_terminal": raw}, paths, plan)
    assert not result["gates.csv"].all_pass.any() and result["raw_gates.csv"].all_pass.all()
    pd.testing.assert_frame_equal(result["annual_metrics.csv"], tables["annual_metrics.csv"], check_exact=True)
    np.testing.assert_array_equal(result["bootstrap_analysis.csv"].daily_lower_bound, [1e-12])
    assert result["resolved_annual_metrics.csv"].strategy.eq("benchmark_terminal").all()


def _mock_workflow(tmp_path, monkeypatch, plan, synthetic, *, fail_native=False):
    import qlib.workflow
    import src.data.convex_allocation

    card = tmp_path / "card"
    card.mkdir()
    (card / "plan.json").write_text(json.dumps(plan))
    (card / "workflow.yaml").write_text("test: mocked_orchestration\n")
    (card / "mechanism.md").write_text("合成测试，无市场结论。\n")
    bundle, objects = deepcopy(synthetic)
    market, means, error, sigma, _ = workflow._frozen_inputs(bundle, objects, plan)
    for cost in plan["costs_bps"]:
        objects["analysis/paths.pkl"][("validation_selected", cost)] = replay_horizon_allocation(
            market, means, error, sigma, cost_bps=cost, rebalance_sessions=5)[0]
    saved, contexts, native_calls = {}, [], []

    class Recorder:
        id = "synthetic-recorder"
        experiment_id = "synthetic-experiment"

        def log_params(self, **kwargs):
            pass

        def save_objects(self, artifact_path, **kwargs):
            saved.update({artifact_path + "/" + name: deepcopy(value) for name, value in kwargs.items()})

        def log_artifact(self, path, artifact_path=None):
            assert Path(path).is_file()

        def get_local_dir(self):
            return str(tmp_path / "recorder-local")

    recorder = Recorder()

    class FakeR:
        active = False
        status = None

        @contextmanager
        def start(self, **kwargs):
            assert not self.active
            assert kwargs.get("recorder_id", recorder.id) == recorder.id
            contexts.append(kwargs)
            self.active = True
            try:
                yield recorder
            except Exception:
                self.status = "FAILED"
                raise
            else:
                self.status = "FINISHED"
            finally:
                self.active = False

        def get_recorder(self):
            return recorder

    fake = FakeR()
    monkeypatch.setattr(qlib.workflow, "R", fake)

    def prepare(*args):
        assert not fake.active
        return deepcopy(bundle)

    def parent(*args):
        assert not fake.active
        return deepcopy(objects), {"hashes_checked_before_unpickling": True, "recorder_id": plan["parent"]["recorder_id"]}

    def native(frame, path, provider, cost, account):
        assert not fake.active
        native_calls.append(cost)
        if fail_native:
            raise ValueError("blocked: synthetic numerical failure")
        return path.copy(deep=True), {"audit": {"status": "passed", "cost_bps": cost, "sessions": len(path)}}

    tables = {name: pd.DataFrame({"test": [1]}) for name in ("backtest_benchmark_comparison.csv", "annual_metrics.csv", "gates.csv", "rolling_summary.csv", "bootstrap_analysis.csv")}
    evidence = {"benchmark_terminal": {"historical_gate_pass": False, "excess_resolution": {"log_excess_floor": 1e-8}}}
    monkeypatch.setattr(workflow, "_hashes", lambda root: {})
    monkeypatch.setattr(workflow, "prepare_horizon_inputs", prepare)
    monkeypatch.setattr(workflow, "load_legacy_execution_view", lambda root: ((root / plan["source"]["execution_provider_uri"]).resolve(), {"test": True}))
    monkeypatch.setattr(workflow, "_load_parent", parent)
    monkeypatch.setattr(workflow, "_bundle_identity", lambda bundle: {"test": "immutable"})
    monkeypatch.setattr(workflow, "_native_path", native)
    monkeypatch.setattr(workflow, "evaluate_registered_paths", lambda *args, **kwargs: (deepcopy(tables), deepcopy(evidence)))
    monkeypatch.setattr(workflow, "resolve_evidence", lambda *args: (deepcopy(tables), deepcopy(evidence)))
    monkeypatch.setattr(workflow, "_conclusion", lambda *args: "中文合成测试结论。")
    monkeypatch.setattr(src.data.convex_allocation, "fit_annual_estimator", lambda *a, **k: pytest.fail("model fitting is forbidden"))
    return card, saved, contexts, native_calls, fake


def test_orchestration_keeps_native_outside_recorder_reproduces_parent_and_archives_before_summary(tmp_path, monkeypatch, plan, synthetic):
    card, saved, contexts, native_calls, fake = _mock_workflow(tmp_path, monkeypatch, plan, synthetic)
    summary, tables, conclusion = workflow.run_benchmark_terminal_experiment(source_root=tmp_path,
        evidence_root=tmp_path, card_dir=card, recorder_uri=tmp_path / "tracking")
    assert len(native_calls) == 6 and fake.status == "FINISHED" and not fake.active
    assert len(contexts) == 2 and contexts[-1]["resume"]
    assert summary["model_fits"] == 0 and summary["parent_control_reproduced"]
    assert (card / "summary.json").is_file() and len(summary["compressed_artifacts"]) == 3
    assert "analysis/raw_gates.pkl" in saved and "analysis/gates.pkl" in saved
    assert tables["parent_control_reproduction.csv"].maximum_difference.max() == 0.
    with pytest.raises(ValueError, match="immutable"):
        workflow.run_benchmark_terminal_experiment(source_root=tmp_path, evidence_root=tmp_path,
            card_dir=card, recorder_uri=tmp_path / "tracking")


def test_financial_failure_retains_frozen_inputs_and_failed_recorder_without_summary(tmp_path, monkeypatch, plan, synthetic):
    card, saved, contexts, native_calls, fake = _mock_workflow(tmp_path, monkeypatch, plan, synthetic, fail_native=True)
    with pytest.raises(ValueError, match="synthetic numerical failure"):
        workflow.run_benchmark_terminal_experiment(source_root=tmp_path, evidence_root=tmp_path,
            card_dir=card, recorder_uri=tmp_path / "tracking")
    assert fake.status == "FAILED" and not fake.active
    assert "inputs/binding.pkl" in saved and "forecast/frozen_parent.pkl" in saved
    assert saved["failure/exception.pkl"]["financial_recipe_changed"] is False
    assert saved["failure/exception.pkl"]["stage"] == "native_soxx_10bps"
    assert not (card / "summary.json").exists()


@pytest.mark.parametrize("key,value", [("model_fit_budget", 1), ("numerical_log_excess_floor", 0.),
    ("rebalance_sessions", 1), ("robust_radius", True), ("candidate_budget", 2)])
def test_unregistered_financial_changes_are_rejected(plan, key, value):
    plan[key] = value
    with pytest.raises(ValueError, match="registered terminal"):
        workflow._validate_plan(plan)

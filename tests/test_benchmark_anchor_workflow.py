"""Synthetic orchestration contracts; no registered market fits are run here."""
from __future__ import annotations

import copy
from concurrent.futures import Future
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.benchmark_anchor_utility import PRICE_FEATURES, build_anchor_dataset
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy, policy_to_dict
from src.workflow import benchmark_anchor_utility as workflow


@pytest.fixture(scope="module")
def plan():
    value = json.loads((Path(__file__).parents[1] / "research/benchmark_anchor_utility_20261002/selection_plan.json").read_text())
    # A zero-update synthetic API fixture, never a market-research alternative.
    value["training"]["updates"] = 0
    value["training"]["minimum_rows"] = 63
    return value


@pytest.fixture(scope="module")
def history():
    dates = pd.bdate_range("2014-01-01", "2017-01-10", name="datetime")
    t = np.arange(len(dates))
    frame = pd.DataFrame({name: np.sin(t * (.07 + i * .013)) + i * .01
                          for i, name in enumerate(PRICE_FEATURES)}, index=dates)
    frame["signal_date"] = dates
    frame["trade_date"] = dates + pd.offsets.BDay(1)
    frame["exit_date"] = dates + pd.offsets.BDay(2)
    for name, offset, hours, minutes in (("feature_available_at", 0, 16, 0), ("decision_at", 1, 9, 31),
                                         ("return_available_at", 2, 9, 30), ("forecast_available_at", 6, 9, 30),
                                         ("auxiliary_available_at", 5, 16, 0)):
        frame[name] = (dates + pd.offsets.BDay(offset)).tz_localize("America/New_York") + pd.Timedelta(hours=hours, minutes=minutes)
    frame["soxx_return"] = .001 + .004 * np.sin(t * .21)
    frame["soxl_return"] = .005 + .011 * np.sin(t * .21)
    frame["future_vol_prediction"] = .018 + .001 * np.sin(t * .11)
    frame["forecast_available"] = 1.
    frame["forecast_model_available_at"] = pd.to_datetime([f"{d.year}-01-01T05:00:00Z" for d in dates])
    return frame


class _InlinePool:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


class _Recorder:
    def __init__(self):
        self.objects = {}
        self.artifacts = []

    def save_objects(self, *, artifact_path, **objects):
        self.objects[artifact_path] = objects

    def log_artifact(self, path, *, artifact_path):
        self.artifacts.append((artifact_path, str(path)))


@pytest.fixture
def prepared(history, plan):
    dataset = build_anchor_dataset(history, as_of="2017-01-10")
    return workflow.prepare_folds(dataset, ["price", "price_vol"], [2015, 2016], plan)


def _record(fold, variant, year, *, left=.2, cash=.1, hold=.4):
    model = BenchmarkAnchorPolicy(seed=17, anchor=True)
    with torch.no_grad():
        model.theta[0, 0] = left
        model.theta[1, 0] = cash
        model.theta[2, 0] = hold
    payload = policy_to_dict(model)
    return {"variant": variant, "model_year": year, "scaler": fold["scaler"],
            "scaler_sha256": fold["scaler"]["scaler_sha256"],
            "training_feature_sha256": hashlib.sha256(fold["features"].tobytes()).hexdigest(),
            "training_return_sha256": hashlib.sha256(fold["returns"].tobytes()).hexdigest(),
            "seed_models": {seed: copy.deepcopy(payload) for seed in workflow.SEEDS},
            "ensemble_models": [copy.deepcopy(payload) for _ in workflow.SEEDS],
            "ensemble_audit": {"anchor_selected": left == cash == hold == 0.},
            "invalid_training": False, "invalid_training_dates": [], "numerical_failure": False}


def _annual(prepared, variant="price"):
    return {year: _record(prepared[(variant, year)], variant, year, left=.2 if year == 2015 else .35)
            for year in (2015, 2016)}


def test_annual_changes_keep_actual_holdings_and_each_cost_has_its_own_path(history, prepared):
    frame = history.loc["2015-12-29":"2016-01-07"]
    annual = _annual(prepared)
    runs = {cost: workflow.replay_models(frame, annual, variant="price", cost_bps=cost) for cost in (10., 25.)}
    for cost, path in runs.items():
        assert workflow.independent_path_audit(frame, path, cost_bps=cost)["passed"]
        np.testing.assert_array_equal(path[[f"pretrade_{a}_weight" for a in workflow.ASSETS]].iloc[1:],
                                      path[[f"ending_{a}_weight" for a in workflow.ASSETS]].iloc[:-1])
        boundary = path.index[path.signal_date.dt.year.eq(2016)][0]
        assert path.pretrade_cash_weight.iloc[boundary] < .9
        assert path.model_year.tolist() == frame.index.year.tolist()
        assert path.equity.iloc[-1] == pytest.approx(np.prod(1 + path.strategy_return))
    assert not np.array_equal(runs[10.].strategy_return, runs[25.].strategy_return)
    assert runs[25.].fees.iloc[0] > runs[10.].fees.iloc[0]


def test_future_invalid_volatility_changes_only_its_date_then_resumes_owned_state(history, prepared):
    frame = history.loc["2016-01-01":"2016-01-08"].copy()
    annual = _annual(prepared, "price_vol")
    good = workflow.replay_models(frame, annual, variant="price_vol", cost_bps=25.)
    point = frame.index[3]
    frame.loc[point, "future_vol_prediction"] = np.nan
    changed = workflow.replay_models(frame, annual, variant="price_vol", cost_bps=25.)
    pd.testing.assert_frame_equal(good.iloc[:3], changed.iloc[:3])
    assert changed.observed_forecast_fallback.tolist() == [False, False, False, True, False, False]
    np.testing.assert_allclose(changed.loc[3, [f"{a}_weight" for a in workflow.ASSETS]].to_numpy(float), [0., 1., 0.], rtol=0, atol=1e-14)
    np.testing.assert_allclose(changed.loc[4, [f"pretrade_{a}_weight" for a in workflow.ASSETS]].to_numpy(float), [0., 1., 0.], rtol=0, atol=1e-14)
    assert changed.soxl_weight.iloc[4] > 0
    assert workflow.independent_path_audit(frame, changed, cost_bps=25.)["passed"]
    before = history.loc[frame.index]
    price_annual = _annual(prepared)
    pd.testing.assert_frame_equal(
        workflow.replay_models(before, price_annual, variant="price", cost_bps=25.),
        workflow.replay_models(frame, price_annual, variant="price", cost_bps=25.))


def test_frozen_year_keeps_older_bundle_and_rejects_future_or_tampered_scaler(history, prepared):
    frame = history.loc["2016-01-01":"2016-01-08"]
    annual = _annual(prepared)
    frozen = workflow.replay_models(frame, annual, variant="price", cost_bps=10., frozen_year=2015)
    assert frozen.model_year.eq(2015).all()
    assert workflow.independent_path_audit(frame, frozen, cost_bps=10.)["passed"]
    for fault in ("future", "variant", "binding", "payload"):
        broken = copy.deepcopy(annual)
        if fault == "future": broken[2015] = broken[2016]
        if fault == "variant": broken[2015]["variant"] = "constant"
        if fault == "binding": broken[2015]["scaler_sha256"] = "different"
        if fault == "payload": broken[2015]["scaler"]["price_mean"][0] += 1.
        with pytest.raises(ValueError, match="scaler|bundle|identity"):
            workflow.replay_models(frame, broken, variant="price", cost_bps=10., frozen_year=2015)


def test_invalid_training_volatility_anchors_all_seeds_without_jobs_or_row_drops(history, plan, monkeypatch):
    frame = history.copy()
    frame.loc["2014-06-02", "future_vol_prediction"] = -1.
    dataset = build_anchor_dataset(frame, as_of="2017-01-10")
    folds = workflow.prepare_folds(dataset, ["price", "price_vol"], [2015], plan)
    assert len(folds[("price", 2015)]["rows"]) == len(folds[("price_vol", 2015)]["rows"])
    assert not folds[("price", 2015)]["invalid_training"]
    assert folds[("price_vol", 2015)]["invalid_training"]
    assert "2014-06-02" in folds[("price_vol", 2015)]["invalid_training_dates"]
    monkeypatch.setattr(workflow, "ProcessPoolExecutor", _InlinePool)
    monkeypatch.setattr(workflow, "_fit_job", lambda *args: pytest.fail("invalid fold started optimizer"))
    recorder = _Recorder()
    fitted, diagnostics, audits = workflow.fit_folds({("price_vol", 2015): folds[("price_vol", 2015)]}, plan, recorder)
    record = fitted["price_vol"][2015]
    assert set(record["seed_models"]) == set(workflow.SEEDS)
    assert len(diagnostics) == 3 and all(d["selected_epoch"] == "anchor" for d in diagnostics)
    assert record["invalid_training"] and record["ensemble_audit"]["anchor_selected"]
    assert audits[0]["all_three_seed_models_used"]
    path = workflow.replay_models(history.loc["2015-01-01":"2015-01-08"], fitted["price_vol"], variant="price_vol", cost_bps=10.)
    assert path.invalid_training_fallback.all()
    np.testing.assert_allclose(path.soxx_weight, 1., rtol=0, atol=1e-14)


def test_worker_training_input_identity_is_checked_before_retaining_model(prepared, plan, monkeypatch):
    def wrong_worker(payload):
        name, year, seed, features, returns, spec = payload
        return name, year, seed, policy_to_dict(BenchmarkAnchorPolicy(seed=seed, anchor=True)), {
            "training_feature_sha256": "wrong", "training_return_sha256": hashlib.sha256(returns.tobytes()).hexdigest()}
    monkeypatch.setattr(workflow, "ProcessPoolExecutor", _InlinePool)
    monkeypatch.setattr(workflow, "_fit_job", wrong_worker)
    with pytest.raises(ValueError, match="different mature inputs"):
        workflow.fit_folds({("price", 2015): prepared[("price", 2015)]}, plan, _Recorder())


def _passing_economics(monkeypatch):
    def passing(paths, benchmarks, *, spec, seed_table):
        assert spec.family_size == 17 and spec.expected_seeds == (17, 41, 71)
        assert spec.minimum_positive_seeds == 2
        return {"gates": pd.DataFrame({"cost_bps": [10., 25.], "economic_check": True, "all_pass": True}),
                "historical_qualification_pass": True}
    monkeypatch.setattr(workflow, "stable_excess_gates", passing)


@pytest.mark.parametrize("flag", ["observed_forecast_fallback", "invalid_training_fallback", "numerical_failure"])
def test_any_source_or_numerical_failure_overrides_positive_economics(history, prepared, plan, monkeypatch, flag):
    frame = history.loc["2016-01-01":"2016-01-08"]
    annual = _annual(prepared)
    paths = {c: workflow.replay_models(frame, annual, variant="price", cost_bps=c) for c in (10., 25.)}
    paths[25.].loc[0, flag] = True
    _passing_economics(monkeypatch)
    monkeypatch.setattr(workflow, "paired_metrics", lambda *args, **kwargs: {"log_excess_sum": 1.})
    result = workflow.qualification(paths, paths, pd.DataFrame(), plan=plan)
    assert not result["historical_qualification_pass"]
    assert not result["gates"].set_index("cost_bps").loc[25., "all_pass"]


@pytest.mark.parametrize("value", [np.nan, "False"])
def test_missing_or_string_failure_flags_cannot_pass_qualification(history, prepared, plan, monkeypatch, value):
    frame = history.loc["2016-01-01":"2016-01-08"]
    paths = {c: workflow.replay_models(frame, _annual(prepared), variant="price", cost_bps=c) for c in (10., 25.)}
    paths[10.]["numerical_failure"] = paths[10.].numerical_failure.astype(object)
    paths[10.].loc[0, "numerical_failure"] = value
    _passing_economics(monkeypatch)
    with pytest.raises(ValueError, match="complete booleans"):
        workflow.qualification(paths, paths, pd.DataFrame(), plan=plan)


def test_exact_null_cannot_qualify_even_with_mocked_positive_economic_flags(history, prepared, plan, monkeypatch):
    frame = history.loc["2016-01-01":"2016-01-08"]
    annual = {year: _record(prepared[("price", year)], "price", year, left=0., cash=0., hold=0.) for year in (2015, 2016)}
    paths = {c: workflow.replay_models(frame, annual, variant="price", cost_bps=c) for c in (10., 25.)}
    benchmarks = {c: workflow.benchmark_path(frame, cost_bps=c) for c in paths}
    _passing_economics(monkeypatch)
    result = workflow.qualification(paths, benchmarks, pd.DataFrame(), plan=plan)
    assert not result["historical_qualification_pass"]
    assert not result["gates"].above_numerical_null.any()


@pytest.mark.parametrize("same_positive_seeds,expected", [(False, False), (True, True)])
def test_seed_gate_requires_same_two_seeds_at_both_costs(plan, monkeypatch, same_positive_seeds, expected):
    from src.analysis import integrated_allocation as analysis
    dates = pd.bdate_range("2016-01-01", "2022-12-30")
    base = pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
                         "exit_date": dates + pd.offsets.BDay(2), "turnover": 0., "fees": 0.,
                         "observed_forecast_fallback": False, "invalid_training_fallback": False, "numerical_failure": False})
    paths = {c: base.assign(cost_bps=c, strategy_return=.002) for c in (10., 25.)}
    references = {c: base.assign(cost_bps=c, strategy_return=.001) for c in paths}
    positive = {10.: {17, 41}, 25.: {17, 41} if same_positive_seeds else {41, 71}}
    seeds = pd.DataFrame([{"seed": seed, "cost_bps": cost, "log_excess_sum": 1. if seed in positive[cost] else -1.}
                          for seed in workflow.SEEDS for cost in paths])
    monkeypatch.setattr(analysis, "stationary_bootstrap_excess", lambda *args, **kwargs: {"lower_bound_positive": True})
    result = workflow.qualification(paths, references, seeds, plan=plan)
    assert result["historical_qualification_pass"] is expected
    assert result["gates"].seeds_positive_at_both_costs.eq(2 if expected else 1).all()


def test_diagnostic_preparation_cannot_start_before_development_selection_is_sealed(history, plan, tmp_path, monkeypatch):
    import qlib.workflow as native_workflow
    from src.data import benchmark_anchor_utility as data
    from src.data import conditional_variance as source
    from src.workflow import conditional_variance as forecast

    local_plan = copy.deepcopy(plan)
    local_plan["training"]["development_signal_years"] = [2015, 2016]
    local_plan["training"]["diagnostic_signal_years"] = [2017]
    card = tmp_path / "card"
    card.mkdir()
    (card / "selection_plan.json").write_text(json.dumps(local_plan))
    recorder = _Recorder()

    class FakeWorkflow:
        @staticmethod
        def set_uri(uri): pass

        @staticmethod
        @contextmanager
        def start(**kwargs): yield

        @staticmethod
        def get_recorder(): return recorder

        @staticmethod
        def log_params(**kwargs): pass

        @staticmethod
        def save_objects(**kwargs): recorder.save_objects(**kwargs)

    class ReachedSealedDiagnostic(Exception):
        pass

    prepared_years = []

    def prepare(dataset, variants, years, current_plan):
        prepared_years.append(tuple(years))
        if years == [2017]:
            seal_path = card / "selection_seal.json"
            assert seal_path.exists()
            assert ("inputs", str(seal_path)) in recorder.artifacts
            seal = json.loads(seal_path.read_text())
            assert seal["selection_uses_diagnostic_signal_years"] is False
            raise ReachedSealedDiagnostic()
        return {}

    def path(frame, *, cost_bps, **kwargs):
        assert frame.index.year.max() <= 2016
        return frame[["signal_date", "trade_date", "exit_date"]].reset_index(drop=True).assign(
            strategy_return=0., turnover=0., fees=0., cost_bps=cost_bps)

    monkeypatch.setattr(native_workflow, "R", FakeWorkflow)
    monkeypatch.setattr(workflow, "CANONICAL_ROOT", tmp_path)
    monkeypatch.setattr(workflow, "_source_hashes", lambda root: {})
    monkeypatch.setattr(source, "load_qualified_primary", lambda *args, **kwargs: {
        "primary_frame": history.copy(), "execution_prices": pd.DataFrame(), "audit": {}})
    monkeypatch.setattr(forecast, "fit_oof_forecasts", lambda *args, **kwargs: (history.copy(), pd.DataFrame(), {}))
    monkeypatch.setattr(data, "build_anchor_dataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(workflow, "prepare_folds", prepare)
    monkeypatch.setattr(workflow, "fit_folds", lambda *args, **kwargs: ({v: {} for v in ("constant", "price", "price_vol")}, [], []))
    monkeypatch.setattr(workflow, "benchmark_path", path)
    monkeypatch.setattr(workflow, "replay_models", lambda frame, annual, **kwargs: path(frame, **kwargs))
    monkeypatch.setattr(workflow, "paired_metrics", lambda *args, **kwargs: {"log_excess_sum": 0.})
    monkeypatch.setattr(workflow, "independent_path_audit", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(workflow, "qualification", lambda *args, **kwargs: {})
    monkeypatch.setattr(workflow, "choose_family", lambda evidence: ({"variant": "constant", "historical_qualification_pass": False}, pd.DataFrame()))
    with pytest.raises(ReachedSealedDiagnostic):
        workflow.run_experiment(tmp_path, card, workers=1)
    assert prepared_years == [(2015, 2016), (2017,)]


@pytest.mark.parametrize("native_fails", [False, True])
def test_native_finalization_reinitializes_outside_active_recorder_then_resumes_exact_identity(tmp_path, monkeypatch, native_fails):
    import qlib
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    provider = tmp_path / "provider"
    provider.mkdir()
    uri = str(tmp_path / "original_recorder")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    with R.start(experiment_name=workflow.EXPERIMENT, uri=uri):
        recorder_id = R.get_recorder().id
        R.save_objects(artifact_path="inputs", **{"identity.pkl": {"id": recorder_id}})

    reports = {10: {"audit": {"status": "passed"}}, 25: {"audit": {"status": "passed"}}}

    def native_builder():
        assert R.exp_manager.active_experiment is None
        # The formerly failing API operation is real; no market provider or
        # native-return simulation is needed to reproduce this lifecycle rule.
        qlib.init(provider_uri=str(provider), region="us", kernels=1)
        R.set_uri(str(tmp_path / "different_default_uri"))
        if native_fails:
            raise RuntimeError("synthetic native report failure")
        return reports

    if native_fails:
        with pytest.raises(RuntimeError, match="synthetic native report failure"):
            workflow._finalize_native(recorder_id, uri, native_builder)
    else:
        assert workflow._finalize_native(recorder_id, uri, native_builder) == reports
    with R.uri_context(uri):
        retained = R.get_recorder(experiment_name=workflow.EXPERIMENT, recorder_id=recorder_id)
        assert retained.id == recorder_id
        assert retained.load_object("inputs/identity.pkl") == {"id": recorder_id}
        assert retained.status == (Recorder.STATUS_FA if native_fails else Recorder.STATUS_FI)
        if not native_fails:
            assert retained.load_object("portfolio_analysis/native_reconciliation.pkl") == reports
    assert R.exp_manager.active_experiment is None

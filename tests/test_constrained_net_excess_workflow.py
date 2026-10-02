"""Synthetic integration checks; no market model fitting is performed."""
from __future__ import annotations

import copy
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.model import constrained_net_excess as model
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy, policy_to_dict
from src.workflow import constrained_net_excess as workflow


@pytest.fixture(scope="module")
def plan():
    return json.loads((Path(__file__).parents[1] / "research/constrained_net_excess_20261003/selection_plan.json").read_text())


@pytest.fixture(scope="module")
def fold():
    # CASH beats the full-period benchmark and both absolute tail metrics, but
    # two rising quarters make the old quarter-shortfall objective negative.
    underlying = np.r_[np.full(126, .001), np.full(126, -.002)]
    return {"features": np.zeros((252, 27)), "returns": np.c_[3 * underlying, underlying],
            "scaler": {"scaler_sha256": "synthetic-frozen-scaler"},
            "invalid_training": False, "invalid_training_dates": []}


class _InlinePool:
    def __init__(self, max_workers): pass
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def submit(self, function, payload):
        future = Future()
        try:
            future.set_result(function(payload))
        except Exception as error:
            future.set_exception(error)
        return future


class _Recorder:
    def __init__(self):
        self.objects = {}
        self.artifacts = []
    def save_objects(self, *, artifact_path, **objects): self.objects[artifact_path] = objects
    def log_artifact(self, path, *, artifact_path): self.artifacts.append((artifact_path, str(path)))


def _worker(payload, *, anchor=False, failure=None):
    name, year, seed, features, returns, spec = payload
    policy = BenchmarkAnchorPolicy(seed=seed, anchor=True)
    if not anchor:
        with torch.no_grad(): policy.theta[1, 0] = 1.
    evaluated = model.evaluate_policy(policy, features, returns, spec=spec)
    metadata = {
        "seed": seed, "spec": asdict(spec), "training_rows": len(features),
        "training_feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
        "training_return_sha256": hashlib.sha256(returns.tobytes()).hexdigest(),
        "selected_epoch": "anchor" if anchor else 0, "elapsed_seconds": 0.,
        "failure": failure, "qualification_eligible": failure is None,
        "null_selected": anchor, "training_feasible": evaluated["feasible"],
        "training_feasibility_passed": evaluated["feasible"],
        "F": evaluated["F"], "selected_F": evaluated["F"], "centered_worst_utility": evaluated["F"],
        "centered_utility_by_cost": evaluated["A_by_cost"],
        "daily_mean_log_excess_by_cost": evaluated["daily_mean_log_excess_by_cost"],
        "aggregate_log_excess_by_cost": evaluated["aggregate_log_excess_by_cost"],
        "legacy_J": evaluated["legacy_J"],
        "selected_risks_by_cost": {cost: model._risk_record(value) for cost, value in evaluated["constraints_by_cost"].items()},
        "selected_normalized_g": evaluated["normalized_g"], "training_paths": evaluated["paths"],
    }
    return name, year, seed, policy_to_dict(policy), metadata


def _fit(fold, plan, monkeypatch, worker=_worker):
    recorder = _Recorder()
    monkeypatch.setattr(workflow, "ProcessPoolExecutor", _InlinePool)
    monkeypatch.setattr(workflow, "_fit_job", worker)
    return workflow.fit_folds({("price", 2016): fold}, plan, recorder, workers=1), recorder


def test_feasible_positive_F_negative_legacy_J_survives_seed_and_own_ensemble_acceptance(fold, plan, monkeypatch):
    (fitted, diagnostics, audits), recorder = _fit(fold, plan, monkeypatch)
    assert {d["seed"] for d in diagnostics} == {17, 41, 71}
    assert all(d["selected_F"] > 0 and d["legacy_J"] < 0 for d in diagnostics)
    record = fitted["price"][2016]
    assert set(record["seed_models"]) == {17, 41, 71}
    assert not record["numerical_failure"]
    assert audits[0]["F"] > 0 and audits[0]["legacy_J"] < 0
    assert audits[0]["all_three_seed_models_used"] and not audits[0]["anchor_selected"]
    assert len(recorder.objects) == 4  # All three seed checkpoints and the ensemble.


@pytest.mark.parametrize("fault", ["feasibility_false", "feasibility_string", "nonfinite_F", "missing_cost", "inconsistent_worst_cost", "failed_raw_risk", "false_normalizer_flag"])
def test_false_or_incomplete_training_acceptance_evidence_never_gets_persisted(fold, plan, monkeypatch, fault):
    recorder = _Recorder()
    def corrupted(payload):
        name, year, seed, policy, metadata = _worker(payload)
        if fault == "feasibility_false": metadata["training_feasibility_passed"] = False
        if fault == "feasibility_string": metadata["training_feasibility_passed"] = "False"
        if fault == "nonfinite_F":
            for key in ("F", "selected_F", "centered_worst_utility"): metadata[key] = np.nan
        if fault == "missing_cost": metadata["centered_utility_by_cost"].pop(25)
        if fault == "inconsistent_worst_cost": metadata["centered_utility_by_cost"][25] = -.001
        if fault == "failed_raw_risk":
            metric = metadata["selected_risks_by_cost"][25]
            metric["max_drawdown_pct"] = metric["benchmark_max_drawdown_pct"] - .1
            metric["raw_loss_gaps"][0] = .001
            metric["raw_percentage_point_gaps"][0] = .1
            metric["risk_pass"][0] = False
            metric["feasible"] = False
        if fault == "false_normalizer_flag":
            metadata["selected_risks_by_cost"][25]["normalizers"] = [0., -1.]
            metadata["selected_risks_by_cost"][25]["normalizers_valid_for_optimization"] = True
        return name, year, seed, policy, metadata
    monkeypatch.setattr(workflow, "ProcessPoolExecutor", _InlinePool)
    monkeypatch.setattr(workflow, "_fit_job", corrupted)
    with pytest.raises(ValueError, match="training|feasib|evidence|finite|risk|cost|boolean|objective|tail|normalizer"):
        workflow.fit_folds({("price", 2016): fold}, plan, recorder, workers=1)
    assert recorder.objects == {}


def test_invalid_training_source_anchors_all_seeds_without_optimizer_and_keeps_source_flag(fold, plan, monkeypatch):
    invalid = copy.deepcopy(fold)
    invalid["invalid_training"] = True
    invalid["invalid_training_dates"] = ["2015-06-01"]
    monkeypatch.setattr(workflow, "ProcessPoolExecutor", _InlinePool)
    monkeypatch.setattr(workflow, "_fit_job", lambda *args: pytest.fail("invalid source reached optimizer"))
    fitted, diagnostics, audits = workflow.fit_folds({("price_vol", 2016): invalid}, plan, _Recorder(), workers=1)
    record = fitted["price_vol"][2016]
    assert record["invalid_training"] and not record["numerical_failure"]
    assert set(record["seed_models"]) == {17, 41, 71}
    assert record["invalid_training_dates"] == ["2015-06-01"]
    assert all(d["failure"] == "invalid_available_training_volatility" for d in diagnostics)
    assert audits[0]["anchor_selected"] and audits[0]["F"] == 0


@pytest.mark.parametrize("failed", [False, True])
def test_numerical_fallback_and_valid_economic_null_remain_distinct(fold, plan, monkeypatch, failed):
    failure = {"epoch": 1, "reason": "synthetic numerical failure", "qualification_eligible": False,
               "accounting_anchor_only": True} if failed else None
    (fitted, diagnostics, audits), _ = _fit(fold, plan, monkeypatch,
        worker=lambda payload: _worker(payload, anchor=True, failure=failure))
    record = fitted["price"][2016]
    assert record["numerical_failure"] is failed
    assert record["ensemble_audit"]["anchor_selected"]
    assert all(d["qualification_eligible"] is (not failed) for d in diagnostics)
    assert all(np.count_nonzero(p["state_dict"]["theta"]) == 0 for p in record["ensemble_models"])
    assert audits[0]["F"] == 0


def test_independent_ensemble_numerical_failure_cannot_disappear_when_seed_fits_pass(fold, plan, monkeypatch):
    original = model.evaluate_ensemble
    def failed_ensemble(policies, features, returns, *, spec):
        anchors = [BenchmarkAnchorPolicy(seed=p.seed, anchor=True) for p in policies]
        accepted, audit = original(anchors, features, returns, spec=spec)
        audit["failure"] = {"reason": "synthetic ensemble-only numerical failure", "qualification_eligible": False,
                            "accounting_anchor_only": True}
        audit["qualification_eligible"] = False
        return accepted, audit
    monkeypatch.setattr(model, "evaluate_ensemble", failed_ensemble)
    (fitted, diagnostics, _), _ = _fit(fold, plan, monkeypatch)
    assert all(d["failure"] is None for d in diagnostics)
    assert fitted["price"][2016]["numerical_failure"]
    assert fitted["price"][2016]["ensemble_audit"]["failure"] is not None


def test_family20_uses_original_same_seed_and_cost_gates(plan, monkeypatch):
    from src.workflow import benchmark_anchor_utility as parent
    captured = []
    def passing(paths, benchmarks, *, spec, seed_table):
        captured.append(spec)
        return {"gates": pd.DataFrame({"cost_bps": [10., 25.], "original_economic_gate": True, "all_pass": True}),
                "historical_qualification_pass": True}
    monkeypatch.setattr(parent, "stable_excess_gates", passing)
    monkeypatch.setattr(parent, "paired_metrics", lambda *args, **kwargs: {"log_excess_sum": 1.})
    paths = {c: pd.DataFrame({"observed_forecast_fallback": [False], "invalid_training_fallback": [False],
                             "numerical_failure": [False]}) for c in (10., 25.)}
    result = workflow.qualification(paths, paths, pd.DataFrame(), plan=plan)
    spec = captured[0]
    assert spec.family_size == 20 and spec.expected_seeds == (17, 41, 71)
    assert spec.minimum_positive_seeds == 2 and spec.costs_bps == (10., 25.)
    assert spec.expected_years == tuple(range(2016, 2023)) and spec.minimum_positive_years == 5
    assert spec.mean_block_lengths == (20, 60) and spec.bootstrap_resamples == 2000
    assert result["historical_qualification_pass"]


@pytest.mark.parametrize("native_fails", [False, True])
def test_phase7_native_reports_resume_own_recorder_after_qlib_reinitialization(tmp_path, monkeypatch, native_fails):
    import qlib
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    provider = tmp_path / "provider";provider.mkdir()
    uri = str(tmp_path / "phase7_recorder")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    with R.start(experiment_name=workflow.EXPERIMENT, uri=uri):
        identifier = R.get_recorder().id
        R.save_objects(artifact_path="inputs", **{"marker.pkl": "retained phase7 evidence"})
    def build():
        assert R.exp_manager.active_experiment is None
        qlib.init(provider_uri=str(provider), region="us", kernels=1)
        R.set_uri(str(tmp_path / "different_default"))
        if native_fails: raise RuntimeError("synthetic native failure")
        return {"synthetic": True}
    if native_fails:
        with pytest.raises(RuntimeError, match="synthetic native failure"):
            workflow._finalize_native(identifier, uri, build)
    else:
        assert workflow._finalize_native(identifier, uri, build) == {"synthetic": True}
    with R.uri_context(uri):
        retained = R.get_recorder(experiment_name=workflow.EXPERIMENT, recorder_id=identifier)
        assert retained.status == (Recorder.STATUS_FA if native_fails else Recorder.STATUS_FI)
        assert retained.load_object("inputs/marker.pkl") == "retained phase7 evidence"
        if not native_fails:
            assert retained.load_object("portfolio_analysis/native_reconciliation.pkl") == {"synthetic": True}


@pytest.mark.parametrize("tamper_development", [False, True])
def test_parent_input_binding_seal_order_frozen_parameters_and_actual_aug3_trade_origin(plan, tmp_path, monkeypatch, tamper_development):
    """Exercise orchestration with sentinel models, never a financial replay."""
    import qlib.workflow as native_workflow
    from src.data import constrained_net_excess as data
    from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES
    from src.workflow import conditional_variance as prior_analysis

    card = tmp_path / "card";card.mkdir()
    (card / "selection_plan.json").write_text(json.dumps(plan))
    dates = pd.to_datetime(["2016-01-04", "2022-12-30", "2023-01-03", "2026-07-31", "2026-08-03", "2026-09-23"])
    frame = pd.DataFrame(0., index=dates, columns=list(COMMON_PRIMARY_FEATURES))
    frame["signal_date"] = dates;frame["trade_date"] = dates + pd.offsets.BDay(1);frame["exit_date"] = dates + pd.offsets.BDay(2)
    frame["soxl_return"] = .003;frame["soxx_return"] = .001
    frame["cash_advantage_prediction"] = 0.;frame["future_vol_prediction"] = .018;frame["forecast_available"] = 1.
    frame["forecast_model_available_at"] = pd.Timestamp("2016-01-01", tz="UTC")
    names = [v["name"] for v in plan["variants"]]
    dev_years = plan["training"]["development_signal_years"]
    outer_years = plan["training"]["diagnostic_signal_years"]
    parent_folds = {(name, year): {"scaler": {"fixture_year": year}, "features": np.zeros((1, 27)), "returns": np.zeros((1, 2))}
                    for name in names for year in dev_years}
    state = {"active": False, "prepared": [], "fit_calls": [], "outer_replays": [], "head_calls": []}

    class Recorder(_Recorder):
        id = "synthetic-phase7"
        def save_objects(self, *, artifact_path, **objects):
            for key, value in objects.items(): self.objects[f"{artifact_path}/{key}"] = value
        def load_object(self, name): return self.objects[name]
        def get_local_dir(self): return str(tmp_path / "synthetic_recorder")
    recorder = Recorder()

    class FakeWorkflow:
        @staticmethod
        def set_uri(uri): pass
        @staticmethod
        @contextmanager
        def start(**kwargs):
            state["active"] = True
            try: yield
            finally: state["active"] = False
        @staticmethod
        def get_recorder(): return recorder
        @staticmethod
        def log_params(**kwargs): pass
        @staticmethod
        def log_metrics(**kwargs): pass
        @staticmethod
        def save_objects(**kwargs): recorder.save_objects(**kwargs)

    class FrozenHead:
        def predict(self, features):
            assert list(features.columns) == list(COMMON_PRIMARY_FEATURES)
            state["head_calls"].append(features.index.tolist())
            return np.c_[np.zeros(len(features)), np.full(len(features), .123)]

    class ReachedNativeAfterClosedRecorder(Exception):
        pass

    outer_records = {year: {"scaler": {"year": year}, "seed_models": {seed: (year, seed) for seed in workflow.SEEDS},
                            "ensemble_models": [(year, seed) for seed in workflow.SEEDS],
                            "ensemble_audit": {"anchor_selected": year == 2023}}
                     for year in outer_years}

    def prepare(dataset, variants, years, current_plan):
        state["prepared"].append(tuple(years))
        if years == dev_years:
            result = copy.deepcopy(parent_folds)
            if tamper_development: result[(names[0], dev_years[0])]["features"][0, 0] = 1.
            return result
        assert years == outer_years and variants == ["price_vol"]
        seal = card / "selection_seal.json"
        assert seal.exists() and ("inputs", str(seal)) in recorder.artifacts
        assert json.loads(seal.read_text())["selection_uses_diagnostic_signal_years"] is False
        return {("price_vol", year): {} for year in outer_years}

    def fit(folds, current_plan, current_recorder, **kwargs):
        years = {year for _, year in folds}
        state["fit_calls"].append(years)
        return ({name: {} for name in names}, [], []) if years == set(dev_years) else ({"price_vol": outer_records}, [], [])

    def path(rows, *, cost_bps):
        return rows[["signal_date", "trade_date", "exit_date"]].reset_index(drop=True).assign(
            strategy_return=.001, fees=0., turnover=0., cost_bps=cost_bps)

    def replay(rows, annual, *, variant, cost_bps, seed=None, frozen_year=None):
        if rows.index.year.min() >= 2023:
            assert (card / "selection_seal.json").exists()
            assert annual is outer_records
            if frozen_year is not None:
                assert frozen_year == 2023
                assert annual[2023]["ensemble_audit"]["anchor_selected"] is True
                assert rows.future_vol_prediction.eq(.123).all()
                assert pd.to_datetime(rows.forecast_model_available_at, utc=True).eq(pd.Timestamp("2023-01-01", tz="America/New_York").tz_convert("UTC")).all()
            else:
                assert rows.future_vol_prediction.eq(.018).all()
            state["outer_replays"].append({"dates": rows.index.tolist(), "frozen_year": frozen_year, "cost": cost_bps})
        else:
            assert rows.index.year.max() <= 2022
        return path(rows, cost_bps=cost_bps)

    def metrics(left, right):
        return {"log_excess_sum": 0., "return_pct": 0., "relative_wealth": 1., "max_drawdown_pct": 0.,
                "cvar5_pct": 0., "sessions": len(left), "first_trade": str(left.trade_date.iloc[0].date())}

    def qualification(*args, **kwargs):
        return {"metrics": pd.DataFrame({"cost_bps": [10., 25.], "return_pct": 0., "relative_wealth": 1., "max_drawdown_pct": 0., "cvar5_pct": 0.}),
                "annual": pd.DataFrame({"signal_year": [2016], "log_excess_sum": [0.]}),
                "gates": pd.DataFrame({"cost_bps": [10., 25.], "all_pass": False}), "bootstrap": pd.DataFrame({"lower_bound": [-1.]})}

    def native_boundary(*args):
        assert state["active"] is False
        raise ReachedNativeAfterClosedRecorder()

    monkeypatch.setattr(native_workflow, "R", FakeWorkflow)
    monkeypatch.setattr(workflow, "CANONICAL_ROOT", tmp_path)
    monkeypatch.setattr(workflow, "_source_hashes", lambda root: {})
    monkeypatch.setattr(data, "load_constrained_inputs", lambda *args: {"raw": frame.copy(), "prices": pd.DataFrame(), "forecasts": frame.copy(),
        "forecast_audit": pd.DataFrame(), "forecast_models": {2023: FrozenHead()}, "dataset": object(),
        "source_audit": {}, "source_details": {}, "parent_dev_folds": parent_folds})
    monkeypatch.setattr(workflow, "prepare_folds", prepare)
    monkeypatch.setattr(workflow, "fit_folds", fit)
    monkeypatch.setattr(workflow, "benchmark_path", path)
    monkeypatch.setattr(workflow, "replay_models", replay)
    monkeypatch.setattr(workflow, "paired_metrics", metrics)
    monkeypatch.setattr(workflow, "annual_paired_metrics", lambda *args: pd.DataFrame({"signal_year": [2023], "log_excess_sum": [0.]}))
    monkeypatch.setattr(workflow, "independent_path_audit", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(workflow, "qualification", qualification)
    monkeypatch.setattr(workflow, "choose_family", lambda evidence: ({"variant": "price_vol", "historical_qualification_pass": False}, pd.DataFrame()))
    monkeypatch.setattr(prior_analysis, "hac_attribution", lambda *args: {"realized_beta": 1.})
    monkeypatch.setattr(workflow, "_finalize_native", native_boundary)
    if tamper_development:
        with pytest.raises(ValueError, match="changed prepared development inputs"):
            workflow.run_experiment(tmp_path, card, workers=1)
        assert state["fit_calls"] == []
        return
    with pytest.raises(ReachedNativeAfterClosedRecorder):
        workflow.run_experiment(tmp_path, card, workers=1)
    assert state["prepared"] == [tuple(dev_years), tuple(outer_years)]
    assert len(state["head_calls"]) == 1
    assert len(state["outer_replays"]) == 8  # Full/fresh x two costs x two modes.
    fresh = [r for r in state["outer_replays"] if r["dates"][0].year == 2026]
    assert len(fresh) == 4 and all(r["dates"][0] == pd.Timestamp("2026-07-31") for r in fresh)
    windows = recorder.objects["analysis/window_metrics.pkl"]
    assert len(windows) == 8 and windows.first_trade.eq("2026-08-03").all()

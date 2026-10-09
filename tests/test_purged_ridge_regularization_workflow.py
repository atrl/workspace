"""Nested-selector integration tests without fitting or scoring market paths."""
from contextlib import contextmanager
from copy import deepcopy
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.workflow import purged_ridge_regularization as workflow


REAL_FIT = workflow.fit_selected_forecasts


class Recorder:
    def __init__(self, path):
        self.id, self.experiment_id, self.path = "frozen-id", "1", path
        (path / "artifacts").mkdir(parents=True)
        self.active, self.status, self.starts, self.saved, self.archived = False, None, [], {}, []
        self.wrong_resume = False

    @contextmanager
    def start(self, **kwargs):
        assert not self.active
        self.active = True
        self.starts.append(kwargs)
        if kwargs.get("resume") and self.wrong_resume: self.id = "wrong-id"
        try:
            yield self
        except Exception:
            self.status = "FAILED"
            raise
        else:
            self.status = "FINISHED"
        finally:
            self.active = False

    def get_recorder(self):
        assert self.active
        return self

    def get_local_dir(self): return str(self.path)
    def log_params(self, **kwargs): assert self.active

    def save_objects(self, *, artifact_path, **objects):
        assert self.active
        self.saved.update({artifact_path + "/" + name: obj for name, obj in objects.items()})

    def log_artifact(self, path, *, artifact_path):
        assert self.active
        self.archived.append((Path(path).name, Path(path).read_bytes()))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import qlib
    import qlib.workflow

    source, root, card, uri = (tmp_path / name for name in ("source", "evidence", "card", "recorder"))
    card.mkdir(); root.mkdir()
    for name in workflow.NEW_SOURCES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen fixture\n")
    monkeypatch.setattr(workflow, "_hashes", lambda folder: {name: workflow._sha(folder / name) for name in workflow.NEW_SOURCES})
    plan = json.loads((Path(__file__).resolve().parents[1] / "research/purged_ridge_regularization_20261009/plan.json").read_text())
    plan["registered_at_utc"] = "2020-01-01T00:00:00+00:00"
    (card / "plan.json").write_text(json.dumps(plan))
    (card / "workflow.yaml").write_text("fixture: true\n")
    for name in plan["cause_sources"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"retained": true}\n')
    r = Recorder(uri / "1" / "frozen-id")
    monkeypatch.setattr(qlib.workflow, "R", r)
    calls = {name: [] for name in ("qlib", "prepare", "fit", "replay", "native", "evaluate")}
    def init(**kwargs):
        assert not r.active, "qlib.init while Recorder is active"
        calls["qlib"].append(kwargs)
    monkeypatch.setattr(qlib, "init", init)
    provider = root / plan["source"]["execution_provider_uri"]
    provider.mkdir(parents=True)
    provider_audit = {"identity": "frozen"}
    monkeypatch.setattr(workflow, "load_legacy_execution_view", lambda folder: (provider, deepcopy(provider_audit)))
    dates = pd.DatetimeIndex(["2011-01-03", "2015-12-31", "2016-01-04", "2017-01-03", "2022-12-30", "2023-01-03", "2026-09-23", "2026-09-24", "2026-09-25"])
    frame = pd.DataFrame({"signal_date": dates, "trade_date": pd.Series(dates).shift(-1).to_numpy(),
        "exit_date": pd.Series(dates).shift(-2).to_numpy(), "soxl_return_h5": .01, "soxx_return_h5": .003}, index=dates)
    native = frame.copy()
    native.columns = pd.MultiIndex.from_tuples([("metadata", name) for name in frame])
    native.index = pd.MultiIndex.from_arrays([dates, ["pair"] * len(dates)], names=["datetime", "instrument"])
    bundle = {"frame": frame, "calendar": dates, "dataset": SimpleNamespace(prepare=lambda *a, **k: native.copy()),
        "covariance": np.stack([np.eye(2) * (i + 1) for i in range(len(dates))]), "audit": {"source": "frozen"}}
    def prepare(folder, specification):
        assert folder == root
        qlib.init(provider_uri=str(root / specification["source"]["provider_uri"]))
        calls["prepare"].append(folder)
        return {**bundle, "frame": frame.copy(), "covariance": bundle["covariance"].copy(), "audit": deepcopy(bundle["audit"])}
    monkeypatch.setattr(workflow, "prepare_horizon_inputs", prepare)
    forecasts = {name: {"rows": pd.DataFrame({"predicted_soxl": np.arange(5) / 100 + extra,
        "predicted_soxx": np.arange(5) / 200 + extra}, index=dates[2:-2]),
        "Vmean": np.stack([np.eye(2) * (i + 1 + extra) for i in range(5)])}
        for name, extra in (("fixed_ridge10", 0.), ("validation_selected", 1.))}
    def fit(prepared, specification):
        assert r.active and prepared["dataset"].calendar.equals(prepared["calendar"])
        assert prepared["dataset"].calendar is not prepared["calendar"]
        calls["fit"].append(specification)
        return {}, {}, deepcopy(forecasts), pd.DataFrame({"information": [1.]}), pd.DataFrame({"candidate_name": ["prior", "ridge_10", "ridge_100", "ridge_1000"]}), pd.DataFrame({"outer_year": [2016], "selected_name": ["prior"]})
    monkeypatch.setattr(workflow, "fit_selected_forecasts", fit)
    monkeypatch.setattr(workflow, "analyze_forecast_increment", lambda *a, **k: {
        "metrics": pd.DataFrame({"value": [0]}), "increments": pd.DataFrame({"value": [0]}),
        "bootstrap": pd.DataFrame({"value": [0]}), "audit": {"scope": "diagnostic"}})
    def path(market, **kwargs):
        out = market[["signal_date", "trade_date", "exit_date"]].reset_index(drop=True).copy()
        out["cost_bps"], out["strategy_return"], out["fees"], out["turnover"] = kwargs["cost_bps"], .001, 0., 0.
        return out
    monkeypatch.setattr(workflow, "replay_fixed_target", path)
    def replay(market, means, errors, covariance, **kwargs):
        assert not r.active
        calls["replay"].append({**kwargs, "means": means.copy(), "errors": errors.copy(), "covariance": covariance.copy()})
        return path(market, **kwargs), {"solver_details": [], "solver_calls": 1}
    monkeypatch.setattr(workflow, "replay_horizon_allocation", replay)
    def native_replay(market, raw, provider, cost, account):
        qlib.init(provider_uri=str(provider))
        calls["native"].append(cost)
        return raw.copy(), {"audit": {"status": "passed", "cost_bps": cost}}
    monkeypatch.setattr(workflow, "_native_path", native_replay)
    def evaluate(paths, **kwargs):
        calls["evaluate"].append(kwargs)
        table = pd.DataFrame([{"strategy": name, "cost_bps": cost, "window": "development", "return_pct": 1.,
            "relative_wealth": 1., "max_drawdown_pct": -1., "cvar5_pct": -.1, "turnover": .1} for name, cost in paths])
        return {"backtest_benchmark_comparison.csv": table}, {"validation_selected": {"historical_gate_pass": False}}
    monkeypatch.setattr(workflow, "evaluate_registered_paths", evaluate)
    monkeypatch.setattr(workflow, "_verify_parent_control", lambda *args: pd.DataFrame({"passed": [True, True]}))
    monkeypatch.setattr(workflow, "save_compressed_object", lambda rec, **kwargs: {"name": kwargs["name"]})
    return {"source": source, "root": root, "card": card, "uri": uri, "plan": plan, "r": r, "calls": calls,
        "bundle": bundle, "provider_audit": provider_audit, "forecasts": forecasts,
        "kwargs": dict(source_root=source, evidence_root=root, card_dir=card, recorder_uri=uri)}


def test_one_selected_pipeline_fixed_execution_and_native_recorder_lifecycle(harness):
    h = harness
    summary, _, _ = workflow.run_purged_ridge_regularization(**h["kwargs"])
    assert len(h["calls"]["fit"]) == 1 and len(h["calls"]["replay"]) == 4
    assert len(h["calls"]["native"]) == 6 and len(h["calls"]["prepare"]) == 2
    assert summary["selected_by_year"] == {"2016": "prior"} and not summary["promotion_eligible"]
    assert h["calls"]["evaluate"][0]["candidate_controls"] == {"validation_selected": "fixed_ridge10"}
    for number, item in enumerate(h["calls"]["replay"]):
        name = "fixed_ridge10" if number % 2 == 0 else "validation_selected"
        assert item["cost_bps"] == (10. if number < 2 else 25.) and item["rebalance_sessions"] == 5
        np.testing.assert_array_equal(item["means"], h["forecasts"][name]["rows"].to_numpy())
        np.testing.assert_array_equal(item["errors"], h["forecasts"][name]["Vmean"])
        np.testing.assert_array_equal(item["covariance"], h["bundle"]["covariance"][2:-2])
    assert h["r"].status == "FINISHED" and h["r"].starts[-1]["recorder_id"] == "frozen-id"
    saved = next(value for name, value in h["r"].archived if name == "summary.json")
    assert json.loads(saved) == json.loads((h["card"] / "summary.json").read_text())
    assert len(h["r"].saved["forecast/selection_scores.pkl"]) == 4


@pytest.mark.parametrize("kind", ["source", "plan", "provider", "bundle", "upstream", "cause"])
def test_mutated_identity_never_generates_completed_card(harness, monkeypatch, kind):
    original = workflow.fit_selected_forecasts
    def change(bundle, plan):
        output = original(bundle, plan)
        if kind == "source": (harness["source"] / workflow.NEW_SOURCES[0]).write_text("# changed")
        elif kind == "plan":
            with (harness["card"] / "plan.json").open("a") as stream: stream.write("\n")
        elif kind == "provider": harness["provider_audit"]["identity"] = "changed"
        elif kind == "bundle": bundle["frame"].iloc[0, bundle["frame"].columns.get_loc("soxl_return_h5")] += .01
        elif kind == "upstream": harness["bundle"]["audit"]["source"] = "changed"
        else: (harness["root"] / plan["cause_sources"][0]).write_text('{"changed": true}')
        return output
    monkeypatch.setattr(workflow, "fit_selected_forecasts", change)
    with pytest.raises(ValueError, match="blocked"):
        workflow.run_purged_ridge_regularization(**harness["kwargs"])
    assert harness["r"].status == "FAILED" and not (harness["card"] / "summary.json").exists()


@pytest.mark.parametrize("kind", ["fit", "archive", "resume"])
def test_failed_fit_archive_or_wrong_resume_stays_failed(harness, monkeypatch, kind):
    if kind == "resume": harness["r"].wrong_resume = True
    else:
        def fail(*args, **kwargs): raise RuntimeError("injected failure")
        monkeypatch.setattr(workflow, "fit_selected_forecasts" if kind == "fit" else "save_compressed_object", fail)
    with pytest.raises((ValueError, RuntimeError)):
        workflow.run_purged_ridge_regularization(**harness["kwargs"])
    assert harness["r"].status == "FAILED" and not (harness["card"] / "summary.json").exists()
    if kind == "fit": assert len(harness["r"].starts) == 1


@pytest.mark.parametrize("key,value", [("candidate_budget", 2), ("inner_candidate_count", 5),
    ("inner_candidate_count", 4.), ("rebalance_sessions", 5.), ("forecast_horizon_sessions", 20),
    ("train_start", "2012-01-03"), ("benchmark", "SOXL"), ("alphas", [1., 10., 100.]),
    ("evidence_kind", "forward")])
def test_registered_parameters_reject_changes_before_work(harness, key, value):
    harness["plan"][key] = value
    (harness["card"] / "plan.json").write_text(json.dumps(harness["plan"]))
    with pytest.raises(ValueError, match="blocked"):
        workflow.run_purged_ridge_regularization(**harness["kwargs"])
    assert not harness["calls"]["qlib"] and not harness["r"].starts


def test_completed_card_cannot_be_reopened(harness):
    (harness["card"] / "summary.json").write_text('{"completed":true}')
    with pytest.raises(ValueError, match="immutable"):
        workflow.run_purged_ridge_regularization(**harness["kwargs"])
    assert not harness["r"].starts


@pytest.fixture
def kernel(monkeypatch):
    dates = pd.DatetimeIndex(["2016-01-04", "2016-01-05", "2017-01-03", "2017-01-04"])
    frame = pd.DataFrame({"signal_date": dates, "exit_date": dates + pd.offsets.BDay(2),
        "decision_at": dates.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15),
        "h5_available_at": (dates + pd.offsets.BDay(6)).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31),
        "soxl_return_h5": .01, "soxx_return_h5": .003}, index=dates)
    bundle = {"frame": frame, "calendar": dates, "dataset": SimpleNamespace()}
    plan = {"fit_years": [2016, 2017], "alphas": [10., 100., 1000.], "minimum_training_rows": 252,
        "minimum_validation_rows": 126, "hac_lags": 5}
    state = {"corruption": None, "selected_calls": [], "predict_calls": []}
    def select(dataset, source_frame, **kwargs):
        assert dataset.calendar.equals(dates) and dataset.calendar is not dates
        assert source_frame is frame
        assert kwargs["alphas"] == (10., 100., 1000.)
        year = kwargs["outer_year"]
        state["selected_calls"].append(year)
        control = {"selected": False, "year": year, "audit": {"training_frame_sha256": str(year)}, "prior_mean": np.array([.01, .003])}
        chosen = deepcopy(control); chosen["selected"] = True
        if state["corruption"] == "hash": chosen["audit"]["training_frame_sha256"] = "different"
        if state["corruption"] == "prior": chosen["prior_mean"] += .01
        names = ["prior", "ridge_10", "ridge_100", "ridge_1000"]
        return {"control_fit": control, "selected_fit": chosen, "inner_models": dict.fromkeys(names, "retained"),
            "candidate_scores": pd.DataFrame({"candidate_name": names, "selected": [True, False, False, False]}),
            "audit": {"outer_year": year, "selected_name": "prior", "outer_outcomes_used_for_selection": False}}
    def predict(model, part):
        state["predict_calls"].append((model["year"], model["selected"]))
        count = len(part)
        covariance_count = count
        if model["selected"] and state["corruption"] == "balanced_vmean": covariance_count += -1 if model["year"] == 2016 else 1
        prior = np.tile(model["prior_mean"], (count, 1))
        if model["selected"] and state["corruption"] == "prediction_prior": prior += .01
        return {"ridge": np.tile([.001 if model["selected"] else .02, .003], (count, 1)),
            "prior": prior, "Vmean": np.tile(np.eye(2), (covariance_count, 1, 1))}
    monkeypatch.setattr(workflow, "select_annual_ridge", select)
    monkeypatch.setattr(workflow, "predict_annual", predict)
    return bundle, plan, state


def test_selection_lives_only_in_inner_module_and_forecast_metadata_stays_matched(kernel):
    bundle, plan, state = kernel
    models, inner, forecasts, info, scores, audits = REAL_FIT(bundle, plan)
    assert state["selected_calls"] == [2016, 2017]
    assert len(state["predict_calls"]) == 4 and len(scores) == 8
    assert set(inner[2016]) == {"prior", "ridge_10", "ridge_100", "ridge_1000"}
    assert audits.selected_name.eq("prior").all() and not audits.outer_outcomes_used_for_selection.any()
    assert info.full_soxl.eq(.001).all() and info.base_soxl.eq(.02).all()
    assert info.index.equals(bundle["frame"].index)
    common = ["signal_date", "forecast_available_at", "label_available_at", "model_year", "prior_soxl", "prior_soxx"]
    pd.testing.assert_frame_equal(forecasts["fixed_ridge10"]["rows"][common], forecasts["validation_selected"]["rows"][common])


@pytest.mark.parametrize("corruption", ["hash", "prior", "prediction_prior", "balanced_vmean"])
def test_per_year_training_and_forecast_alignment_cannot_be_silently_repaired(kernel, corruption):
    bundle, plan, state = kernel
    state["corruption"] = corruption
    with pytest.raises(ValueError, match="blocked"):
        REAL_FIT(bundle, plan)


@pytest.fixture
def parent_control(tmp_path, monkeypatch):
    import qlib.workflow

    card_relative = Path("research/oos_uncertainty_horizon_fixture")
    card = tmp_path / card_relative
    local = card / "qlib_recorder/1/parent-id"
    artifact = local / "artifacts/analysis/paths.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"retained synthetic artifact identity")
    summary = {"experiment": "oos_uncertainty_horizon_fixture", "recorder_uri": str(card / "qlib_recorder"),
        "recorder_id": "parent-id", "as_of": "2026-09-25", "provider_uri": "data/qlib/qualified",
        "execution_provider_uri": "data/qlib/execution"}
    (card / "summary.json").write_text(json.dumps(summary))
    manifest = {"recorder_id": "parent-id", "as_of": "2026-09-25", "entries": [
        {"path": str(artifact.relative_to(tmp_path)), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "bytes": artifact.stat().st_size}]}
    (card / "verification_manifest.json").write_text(json.dumps(manifest))
    dates = pd.bdate_range("2016-01-04", periods=2)
    paths = {}
    for cost in (10., 25.):
        paths[("fixed_ridge10", cost)] = pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
            "exit_date": dates + pd.offsets.BDay(2), "cost_bps": cost, "strategy_return": .001,
            "turnover": 0., "fees": 0., "open_soxl_weight": 0., "open_soxx_weight": 1., "open_cash_weight": 0.})
    previous = {("hac_h5", cost): frame.copy() for (_, cost), frame in paths.items()}
    state = {"status": "FINISHED"}
    recorder = SimpleNamespace(id="parent-id", get_local_dir=lambda: str(local),
        load_object=lambda name: previous,
        client=SimpleNamespace(get_run=lambda rid: SimpleNamespace(info=SimpleNamespace(status=state["status"]))))
    @contextmanager
    def uri_context(uri):
        assert uri == summary["recorder_uri"]
        yield
    def get_recorder(**kwargs):
        assert kwargs == {"experiment_name": summary["experiment"], "recorder_id": summary["recorder_id"]}
        return recorder
    monkeypatch.setattr(qlib.workflow, "R", SimpleNamespace(uri_context=uri_context, get_recorder=get_recorder))
    plan = {"cause_sources": [str(card_relative / "summary.json")], "costs_bps": [10., 25.],
            "source": {"as_of": summary["as_of"], "provider_uri": summary["provider_uri"], "execution_provider_uri": summary["execution_provider_uri"]}}
    return dict(root=tmp_path, plan=plan, paths=paths, previous=previous, state=state,
                card=card, artifact=artifact, manifest=manifest)


def test_parent_control_readback_is_a_same_path_check_not_selection(parent_control):
    p = parent_control
    result = workflow._verify_parent_control(p["root"], p["plan"], p["paths"])
    assert len(result) == 2 and result.passed.all()
    assert result.maximum_difference.eq(0.).all() and result.tolerance.eq(1e-10).all()
    assert result.path_sha256.eq(hashlib.sha256(p["artifact"].read_bytes()).hexdigest()).all()


@pytest.mark.parametrize("corruption", ["source_missing", "unfinished", "dates", "values", "nonfinite", "artifact_hash", "manifest_source"])
def test_parent_control_rejects_source_identity_dates_or_values(parent_control, corruption):
    p = parent_control
    if corruption == "source_missing": p["plan"]["cause_sources"] = []
    elif corruption == "unfinished": p["state"]["status"] = "FAILED"
    elif corruption == "dates": p["previous"][("hac_h5", 10.)].loc[0, "signal_date"] += pd.offsets.BDay(1)
    elif corruption == "values": p["previous"][("hac_h5", 10.)].loc[0, "strategy_return"] += .001
    elif corruption == "nonfinite": p["previous"][("hac_h5", 10.)].loc[0, "strategy_return"] = np.nan
    elif corruption == "artifact_hash": p["artifact"].write_bytes(p["artifact"].read_bytes() + b"changed")
    else:
        p["manifest"]["recorder_id"] = "different-parent"
        (p["card"] / "verification_manifest.json").write_text(json.dumps(p["manifest"]))
    with pytest.raises(ValueError, match="blocked"):
        workflow._verify_parent_control(p["root"], p["plan"], p["paths"])

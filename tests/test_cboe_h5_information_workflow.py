"""Independent Cboe experiment and shared-evidence orchestration contracts."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.workflow import benchmark_evidence as evidence
from src.workflow import cboe_h5_information as workflow


REAL_FIT = workflow.fit_matched_models
METRIC = {"return_pct": 1., "relative_wealth": 1., "max_drawdown_pct": -1., "cvar5_pct": -.1, "turnover": .1}


class Recorder:
    def __init__(self, path):
        self.id, self.experiment_id, self.path = "retained-id", "1", path
        (path / "artifacts").mkdir(parents=True)
        self.active, self.status, self.starts, self.objects, self.archives = False, None, [], {}, []
        self.wrong_resume = False

    @contextmanager
    def start(self, **kwargs):
        assert not self.active
        self.starts.append(kwargs)
        self.active = True
        if kwargs.get("resume") and self.wrong_resume:
            self.id = "wrong-id"
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

    def get_local_dir(self):
        return str(self.path)

    def log_params(self, **kwargs):
        assert self.active

    def save_objects(self, *, artifact_path, **objects):
        assert self.active
        self.objects.update({artifact_path + "/" + key: value for key, value in objects.items()})

    def log_artifact(self, path, *, artifact_path):
        assert self.active
        self.archives.append((Path(path).name, Path(path).read_bytes()))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import qlib
    import qlib.workflow

    source, root, card, uri = (tmp_path / name for name in ("source", "evidence", "card", "recorder"))
    card.mkdir(); root.mkdir()
    for name in workflow.NEW_SOURCES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# isolated retained source\n")
    monkeypatch.setattr(workflow, "_hashes", lambda folder: {name: workflow._sha(folder / name) for name in workflow.NEW_SOURCES})
    plan = json.loads((Path(__file__).resolve().parents[1] / "research/cboe_h5_information_20261009/plan.json").read_text())
    plan["registered_at_utc"] = "2020-01-01T00:00:00+00:00"
    (card / "plan.json").write_text(json.dumps(plan))
    for name in ("source_intake.json", "source_contract.md", "workflow.yaml"):
        (card / name).write_text("fixture\n")
    r = Recorder(uri / "1" / "retained-id")
    monkeypatch.setattr(qlib.workflow, "R", r)
    qlib_calls, preparations, replays, native_calls, evaluations, fitcalls = [], [], [], [], [], []
    def init(**kwargs):
        assert not r.active, "qlib.init must remain outside active Recorder"
        qlib_calls.append(kwargs)
    monkeypatch.setattr(qlib, "init", init)
    provider = root / plan["source"]["execution_provider_uri"]
    provider.mkdir(parents=True)
    provider_audit = {"execution_source": "retained"}
    monkeypatch.setattr(workflow, "load_legacy_execution_view", lambda folder: (provider, deepcopy(provider_audit)))
    dates = pd.DatetimeIndex(["2014-02-04", "2015-12-29", "2016-01-04", "2017-01-03", "2022-12-30", "2023-01-03", "2026-09-23"])
    frame = pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
        "exit_date": dates + pd.offsets.BDay(2), "soxl_return_h5": .01, "soxx_return_h5": .003}, index=dates)
    native = frame.copy()
    native.columns = pd.MultiIndex.from_tuples([("metadata", key) for key in frame])
    native.index = pd.MultiIndex.from_arrays([dates, ["pair"] * len(dates)], names=["datetime", "instrument"])
    bundle = {"frame": frame, "calendar": dates,
        "dataset": SimpleNamespace(prepare=lambda *args, **kwargs: native.copy()),
        "covariance": np.stack([np.eye(2) * (i + 1) for i in range(len(dates))]),
        "audit": {"external": {"store_as_of": "2026-10-08", "market_as_of": "2026-09-25", "sha256": "frozen"}}}
    def prepare(folder, specification):
        assert folder == root
        qlib.init(provider_uri=str(root / specification["source"]["provider_uri"]))
        qlib.init(provider_uri=str(root / specification["external_source"]["provider_uri"]))
        preparations.append(folder)
        return {**bundle, "frame": frame.copy(), "covariance": bundle["covariance"].copy(), "audit": deepcopy(bundle["audit"])}
    monkeypatch.setattr(workflow, "prepare_cboe_horizon_dataset", prepare)
    prediction_dates = dates[1:]
    forecasts = {}
    for name, shift in (("base", 0.), ("base_cboe3", 1.)):
        rows = pd.DataFrame({"predicted_soxl": np.arange(len(prediction_dates)) / 100 + shift,
            "predicted_soxx": np.arange(len(prediction_dates)) / 200 + shift}, index=prediction_dates)
        forecasts[name] = {"rows": rows, "Vmean": np.stack([np.eye(2) * (i + 1 + shift) for i in range(len(rows))])}
    def fit(prepared, specification):
        assert r.active
        fitcalls.append(specification)
        return {}, deepcopy(forecasts), pd.DataFrame({"placeholder": [1.]}), pd.DataFrame({"fit_year": [2016]})
    monkeypatch.setattr(workflow, "fit_matched_models", fit)
    def statistics(info, **kwargs):
        assert kwargs["evaluation_cutoff"] == pd.Timestamp("2026-09-25T17:00:00", tz="America/New_York")
        return {"metrics": pd.DataFrame({"value": [0.]}), "increments": pd.DataFrame({"value": [0.]}),
                "bootstrap": pd.DataFrame({"value": [0.]}), "audit": {"historical_vintage_certified": False}}
    monkeypatch.setattr(workflow, "analyze_forecast_increment", statistics)
    def path(market, **kwargs):
        out = market[["signal_date", "trade_date", "exit_date"]].reset_index(drop=True).copy()
        out["strategy_return"], out["cost_bps"] = .001, kwargs["cost_bps"]
        out["fees"], out["turnover"] = 0., 0.
        out["pretrade_cash_weight"] = [1.] + [0.] * (len(out) - 1)
        return out
    monkeypatch.setattr(workflow, "replay_fixed_target", path)
    def horizon(market, means, errors, covariance, **kwargs):
        assert not r.active
        replays.append({**kwargs, "means": means.copy(), "errors": errors.copy(), "covariance": covariance.copy()})
        return path(market, **kwargs), {"solver_details": [], "solver_calls": 1}
    monkeypatch.setattr(workflow, "replay_horizon_allocation", horizon)
    def native_replay(market, raw, execution_provider, cost, account):
        qlib.init(provider_uri=str(execution_provider))
        native_calls.append(cost)
        return raw.copy(), {"audit": {"status": "passed", "cost_bps": cost}}
    monkeypatch.setattr(workflow, "_native_path", native_replay)
    def evaluate(paths, **kwargs):
        evaluations.append(kwargs)
        table = pd.DataFrame([{"strategy": name, "cost_bps": cost, "window": "development", **METRIC}
                              for name, cost in paths])
        return {"backtest_benchmark_comparison.csv": table}, {"base_cboe3": {"historical_gate_pass": False}}
    monkeypatch.setattr(workflow, "evaluate_registered_paths", evaluate)
    monkeypatch.setattr(workflow, "save_compressed_object", lambda rec, **kwargs: {"name": kwargs["name"]})
    return {"root": root, "source": source, "card": card, "uri": uri, "plan": plan, "r": r,
        "bundle": bundle, "forecasts": forecasts, "provider_audit": provider_audit,
        "preparations": preparations, "qlib_calls": qlib_calls, "native_calls": native_calls,
        "replays": replays, "evaluations": evaluations, "fitcalls": fitcalls,
        "kwargs": dict(source_root=source, evidence_root=root, card_dir=card, recorder_uri=uri)}


def test_registered_candidate_matched_indices_costs_cutoffs_and_archive_lifecycle(harness):
    h = harness
    summary, _, _ = workflow.run_cboe_h5_information(**h["kwargs"])
    assert summary["as_of"] == "2026-09-25" and summary["external_store_as_of"] == "2026-10-08"
    assert not summary["promotion_eligible"] and not summary["historical_vintage_certified"]
    assert len(h["fitcalls"]) == 1 and len(h["replays"]) == 4 and len(h["native_calls"]) == 6
    assert len(h["preparations"]) == 2
    for index, call in enumerate(h["replays"]):
        name = "base" if index % 2 == 0 else "base_cboe3"
        assert call["cost_bps"] == (10. if index < 2 else 25.) and call["rebalance_sessions"] == 5
        np.testing.assert_array_equal(call["means"], h["forecasts"][name]["rows"].iloc[1:].to_numpy())
        np.testing.assert_array_equal(call["errors"], h["forecasts"][name]["Vmean"][1:])
        np.testing.assert_array_equal(call["covariance"], h["bundle"]["covariance"][2:])
    assert h["evaluations"][0]["candidate_controls"] == {"base_cboe3": "base"}
    assert h["evaluations"][0]["plan"]["gates"]["family_size"] == 1
    assert h["r"].starts[-1]["recorder_id"] == "retained-id" and h["r"].status == "FINISHED"
    saved = next(raw for name, raw in h["r"].archives if name == "summary.json")
    assert json.loads(saved) == json.loads((h["card"] / "summary.json").read_text())
    assert all(p.pretrade_cash_weight.iloc[0] == 1 for p in h["r"].objects["analysis/paths.pkl"].values())


@pytest.mark.parametrize("changed", ["source", "plan", "execution", "external", "frame"])
def test_source_plan_or_matched_data_mutation_fails_without_summary(harness, monkeypatch, changed):
    original = workflow.fit_matched_models
    def mutate(bundle, plan):
        output = original(bundle, plan)
        if changed == "source": (harness["source"] / workflow.NEW_SOURCES[0]).write_text("# changed\n")
        elif changed == "plan":
            with (harness["card"] / "plan.json").open("a") as stream: stream.write("\n")
        elif changed == "execution": harness["provider_audit"]["execution_source"] = "changed"
        elif changed == "external": harness["bundle"]["audit"]["external"]["sha256"] = "changed"
        else: bundle["frame"].iloc[0, bundle["frame"].columns.get_loc("soxl_return_h5")] += .01
        return output
    monkeypatch.setattr(workflow, "fit_matched_models", mutate)
    with pytest.raises(ValueError, match="blocked"):
        workflow.run_cboe_h5_information(**harness["kwargs"])
    assert harness["r"].status == "FAILED" and not (harness["card"] / "summary.json").exists()


@pytest.mark.parametrize("failure", ["fit", "archive", "resume"])
def test_no_completed_summary_when_fit_archive_or_identity_fails(harness, monkeypatch, failure):
    if failure == "resume": harness["r"].wrong_resume = True
    else:
        def fail(*args, **kwargs): raise RuntimeError("injected failure")
        monkeypatch.setattr(workflow, "fit_matched_models" if failure == "fit" else "save_compressed_object", fail)
    with pytest.raises((RuntimeError, ValueError)):
        workflow.run_cboe_h5_information(**harness["kwargs"])
    assert harness["r"].status == "FAILED" and not (harness["card"] / "summary.json").exists()
    if failure == "fit": assert len(harness["r"].starts) == 1


@pytest.mark.parametrize("bad", ["variants", "family", "boolean_budget", "boolean_family", "matched_start", "cutoff", "extra_flag"])
def test_invalid_plan_is_rejected_before_data_loading(harness, bad):
    plan = harness["plan"]
    if bad == "variants": plan["variants"].reverse()
    elif bad == "family": plan["gates"]["family_size"] = 2
    elif bad == "boolean_budget": plan["candidate_budget"] = True
    elif bad == "boolean_family": plan["gates"]["family_size"] = True
    elif bad == "matched_start": plan["matched_training_start"] = "2014-02-05"
    elif bad == "cutoff": plan["external_source"]["market_as_of"] = plan["external_source"]["store_as_of"]
    else: plan["variants"][1]["extra_features"] = 1
    (harness["card"] / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="blocked"):
        workflow.run_cboe_h5_information(**harness["kwargs"])
    assert not harness["qlib_calls"] and not harness["r"].starts


@pytest.mark.parametrize("corruption", ["matched_rows_sha256", "label_rows_sha256", "training_dates_sha256", "prior", "prediction_dates", "prediction_clock", "vmean_shape"])
def test_matched_fit_kernel_rejects_different_cohorts_metadata_or_covariance(monkeypatch, corruption):
    dates = pd.DatetimeIndex(["2016-01-04", "2016-01-05"])
    bundle = {"dataset": object(), "frame": pd.DataFrame({"soxl_return_h5": .01, "soxx_return_h5": .003}, index=dates)}
    plan = {"variants": [{"name": "base", "extra_features": False}, {"name": "base_cboe3", "extra_features": True}],
        "fit_years": [2016], "ridge_alpha": 10., "minimum_training_rows": 252, "hac_lags": 5,
        "matched_training_start": "2014-02-04"}
    def fit(dataset, **kwargs):
        full = len(kwargs["feature_columns"]) > len(workflow.BASE_FEATURES)
        audit = dict.fromkeys(("matched_rows_sha256", "label_rows_sha256", "training_dates_sha256"), "matched")
        if full and corruption in audit: audit[corruption] = "different"
        return {"full": full, "audit": audit, "prior_mean": np.array([.01, .003]) + (.01 if full and corruption == "prior" else 0)}
    def predict(model, dataset, **kwargs):
        current = dates + pd.offsets.BDay(1) if model["full"] and corruption == "prediction_dates" else dates
        table = pd.DataFrame({"signal_date": current,
            "forecast_available_at": current.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15),
            "label_available_at": (current + pd.offsets.BDay(6)).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31),
            "model_year": 2016, "prior_soxl": .01, "prior_soxx": .003,
            "predicted_soxl": .01, "predicted_soxx": .003}, index=current)
        if model["full"] and corruption == "prediction_clock": table["forecast_available_at"] += pd.Timedelta(hours=1)
        count = 1 if model["full"] and corruption == "vmean_shape" else len(table)
        return {"predictions": table, "Vmean": np.zeros((count, 2, 2))}
    monkeypatch.setattr(workflow, "fit_matched_horizon_ridge", fit)
    monkeypatch.setattr(workflow, "predict_matched_horizon", predict)
    with pytest.raises(ValueError, match="blocked"):
        REAL_FIT(bundle, plan)


@pytest.mark.parametrize("corruption", ["missing_cost", "wrong_cost", "self_control", "future_exit"])
def test_shared_reporting_rejects_inventory_or_boundary_mismatch(monkeypatch, corruption):
    dates = pd.DatetimeIndex(["2016-01-04", "2023-01-03"])
    plan = {"costs_bps": [10., 25.], "development_years": list(range(2016, 2023)),
        "variants": [{"name": "base"}, {"name": "base_cboe3"}], "candidate_budget": 1,
        "gates": {"family_size": 1}, "source": {"as_of": "2026-09-25"}}
    paths = {(name, cost): pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
        "exit_date": dates + pd.offsets.BDay(2), "cost_bps": cost}) for name in ("soxx", "base", "base_cboe3") for cost in (10., 25.)}
    controls = {"base_cboe3": "base"}
    if corruption == "missing_cost": del paths[("base", 25.)]
    elif corruption == "wrong_cost": paths[("base", 25.)]["cost_bps"] = 10.
    elif corruption == "self_control": controls["base_cboe3"] = "base_cboe3"
    else: paths[("base_cboe3", 10.)].loc[1, "exit_date"] = pd.Timestamp("2026-10-08")
    dummy = pd.DataFrame({"all_pass": [False]})
    monkeypatch.setattr(evidence, "evaluate_benchmark_relative", lambda *args, **kwargs:
        {"gates": dummy, "rolling_summary": dummy, "bootstrap": dummy, "historical_gate_pass": False})
    monkeypatch.setattr(evidence, "paired_metrics", lambda *args: METRIC.copy())
    monkeypatch.setattr(evidence, "annual_paired_metrics", lambda *args: pd.DataFrame([METRIC]))
    with pytest.raises(ValueError, match="blocked"):
        evidence.evaluate_registered_paths(paths, plan=plan, calendar=dates, candidate_controls=controls)

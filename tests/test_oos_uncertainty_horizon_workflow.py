"""Independent lifecycle and alignment checks; no market fitting or search."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.workflow import oos_uncertainty_horizon as workflow


class RecordingFacade:
    def __init__(self, destination):
        self.id, self.experiment_id = "frozen-id", "experiment-1"
        self.destination = destination
        (destination / "artifacts").mkdir(parents=True)
        self.active, self.status, self.starts, self.objects, self.artifacts = False, None, [], {}, []
        self.override_identity = False

    @contextmanager
    def start(self, **kwargs):
        assert not self.active
        self.starts.append(kwargs)
        self.active = True
        if kwargs.get("resume") and self.override_identity:
            self.id = "wrong-id"
            self.override_identity = False
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
        return str(self.destination)

    def log_params(self, **kwargs):
        assert self.active

    def save_objects(self, *, artifact_path, **objects):
        assert self.active
        self.objects.update({artifact_path + "/" + name: value for name, value in objects.items()})

    def log_artifact(self, path, *, artifact_path):
        assert self.active
        self.artifacts.append((artifact_path, Path(path).name, Path(path).read_bytes()))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import qlib
    import qlib.workflow

    source, evidence, card, uri = (tmp_path / name for name in ("source", "evidence", "card", "recorder"))
    card.mkdir()
    evidence.mkdir()
    names = tuple(dict.fromkeys((*workflow.SOURCE_FILES, *workflow.ADDITIONAL_SOURCES)))
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# retained source\n")
    monkeypatch.setattr(workflow, "_source_hashes", lambda root: {name: workflow._sha(root / name) for name in names})
    plan = json.loads((Path(__file__).resolve().parents[1] /
                       "research/oos_uncertainty_horizon_20261009/plan.json").read_text())
    plan["registered_at_utc"] = "2020-01-01T00:00:00+00:00"
    (card / "plan.json").write_text(json.dumps(plan))
    for name in ("workflow.yaml", "diagnosis.md"):
        (card / name).write_text("test fixture\n")
    r = RecordingFacade(uri / "experiment-1" / "frozen-id")
    monkeypatch.setattr(qlib.workflow, "R", r)
    qlib_calls, preparation_calls, native_calls, replay_calls, gate_calls = [], [], [], [], []

    def init(**kwargs):
        assert not r.active, "qlib.init inside active Recorder"
        qlib_calls.append(kwargs)

    monkeypatch.setattr(qlib, "init", init)
    provider = evidence / plan["source"]["execution_provider_uri"]
    provider.mkdir(parents=True)
    audit = {"provider_sha256": "source-before"}
    monkeypatch.setattr(workflow, "load_legacy_execution_view", lambda root: (provider, deepcopy(audit)))

    def attribution(root, specification, execution_provider):
        assert root == evidence and execution_provider == provider
        qlib.init(provider_uri=str(provider))
        return {"information": pd.DataFrame({"source": ["retained experts"]})}, {"recorder_id": "attribution-parent"}

    monkeypatch.setattr(workflow, "_attribution", attribution)
    dates = pd.DatetimeIndex(["2011-01-03", "2013-01-02", "2014-01-02", "2015-01-02",
                              "2016-01-04", "2017-01-03", "2022-12-30", "2023-01-03", "2026-09-23"])
    frame = pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
                          "exit_date": dates + pd.offsets.BDay(2), "feature": np.arange(len(dates), dtype=float)}, index=dates)
    frame["decision_at"] = dates.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
    frame["return_available_at"] = pd.DatetimeIndex(frame.exit_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    covariance = np.stack([np.eye(2) * (i + 1) * .001 for i in range(len(dates))])
    native_frame = frame.copy()
    native_frame.columns = pd.MultiIndex.from_tuples([("metadata", name) for name in frame.columns])
    native_frame.index = pd.MultiIndex.from_arrays([dates, ["paired"] * len(dates)], names=["datetime", "instrument"])
    dataset = SimpleNamespace(prepare=lambda *args, **kwargs: native_frame.copy())
    bundle = {"frame": frame, "dataset": dataset, "calendar": dates, "covariance": covariance,
              "audit": {"native_frame_identity": "original-source", "transformed_frame_sha256": workflow.frame_sha256(frame)}}

    def prepare(root, specification):
        assert root == evidence
        qlib.init(provider_uri=str(evidence / specification["source"]["provider_uri"]))
        preparation_calls.append(root)
        return {**bundle, "frame": bundle["frame"].copy(), "covariance": bundle["covariance"].copy(), "audit": deepcopy(bundle["audit"])}

    monkeypatch.setattr(workflow, "prepare_horizon_inputs", prepare)
    prediction_dates = dates[1:]
    predictions = pd.DataFrame({"predicted_soxl": np.arange(len(prediction_dates)) / 1000,
                                "predicted_soxx": np.arange(len(prediction_dates)) / 2000}, index=prediction_dates)
    vmean = np.stack([np.eye(2) * (i + 1) * 1e-5 for i in range(len(prediction_dates))])

    def fit(prepared, specification):
        assert r.active
        return predictions.copy(), vmean.copy(), {"models": "fixed annual identities"}, pd.DataFrame({"year": [2013]})

    monkeypatch.setattr(workflow, "fit_annual_oos", fit)

    def floor(predicted, *, cutoff, minimum_rows, hac_lags):
        assert r.active and minimum_rows == 252 and hac_lags == 5
        return {"bias": np.array([0., 0.]), "floor": np.eye(2) * cutoff.year / 1e6,
                "audit": {"cutoff": str(cutoff), "strictly_mature": True}}

    monkeypatch.setattr(workflow, "fit_oos_mean_error", floor)

    def fixed(market, **kwargs):
        result = market.loc[:, ["signal_date", "trade_date", "exit_date"]].reset_index(drop=True).copy()
        result["strategy_return"] = kwargs["cost_bps"] / 100000.
        result["cost_bps"], result["fees"], result["turnover"] = kwargs["cost_bps"], 0., 0.
        result["pretrade_cash_weight"] = [1.] + [0.] * (len(result) - 1)
        return result

    monkeypatch.setattr(workflow, "replay_fixed_target", fixed)

    def horizon(market, means, errors, covariances, **kwargs):
        assert not r.active
        expected_decision = market.index.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
        assert pd.DatetimeIndex(market.decision_at).equals(expected_decision)
        replay_calls.append({**kwargs, "dates": market.index.copy(), "means": means.copy(),
                             "errors": errors.copy(), "covariance": covariances.copy()})
        return fixed(market, **kwargs), {"solver_details": ["sealed numerical certificate"], "solver_calls": 1}

    monkeypatch.setattr(workflow, "replay_horizon_allocation", horizon)

    def native(market, raw, source_provider, cost, account):
        assert not r.active
        qlib.init(provider_uri=str(source_provider))
        native_calls.append(cost)
        return raw.copy(), {"audit": {"status": "passed", "cost_bps": cost}}

    monkeypatch.setattr(workflow, "_native_path", native)

    def evaluate(paths, benchmarks, **kwargs):
        gate_calls.append(kwargs)
        assert set(paths) == set(benchmarks) == {10., 25.}
        return {"historical_gate_pass": False, "gates": pd.DataFrame({"all_pass": [False]}),
                "rolling_summary": pd.DataFrame({"positive_excess_rate": [.2]}),
                "bootstrap": pd.DataFrame({"lower_bound_positive": [False]})}

    monkeypatch.setattr(workflow, "evaluate_benchmark_relative", evaluate)
    metric = {"return_pct": 1., "relative_wealth": 1., "max_drawdown_pct": -1., "cvar5_pct": -.1, "turnover": 1.}
    monkeypatch.setattr(workflow, "paired_metrics", lambda *args: metric.copy())
    monkeypatch.setattr(workflow, "annual_paired_metrics", lambda *args: pd.DataFrame([{"year": 2016, **metric}]))
    monkeypatch.setattr(workflow, "save_compressed_object",
        lambda rec, *, name, artifact_path, value: {"name": name, "artifact_path": artifact_path})
    return {"source": source, "evidence": evidence, "card": card, "uri": uri, "plan": plan, "r": r,
        "audit": audit, "bundle": bundle, "predictions": predictions, "vmean": vmean,
        "replay_calls": replay_calls, "gate_calls": gate_calls, "native_calls": native_calls,
        "qlib_calls": qlib_calls, "preparation_calls": preparation_calls,
        "kwargs": dict(source_root=source, evidence_root=evidence, card_dir=card, recorder_uri=uri)}


def test_exact_candidates_costs_positional_covariance_and_recorder_lifecycle(harness):
    summary, _, _ = workflow.run_oos_uncertainty_horizon(**harness["kwargs"])
    h, r = harness, harness["r"]
    assert len(h["replay_calls"]) == 6 and len(h["native_calls"]) == 8
    assert [(c["cost_bps"], c["rebalance_sessions"]) for c in h["replay_calls"]] == [(10., 1), (10., 5), (10., 5), (25., 1), (25., 5), (25., 5)]
    replay_dates = h["bundle"]["frame"].index[h["bundle"]["frame"].index >= h["plan"]["signal_start"]]
    indices = h["predictions"].index.get_indexer(replay_dates)
    base = h["vmean"][indices]
    additional = np.stack([np.eye(2) * year / 1e6 for year in replay_dates.year])
    for i, call in enumerate(h["replay_calls"]):
        np.testing.assert_array_equal(call["means"], h["predictions"].iloc[indices].to_numpy())
        np.testing.assert_array_equal(call["errors"], base + additional if i % 3 == 2 else base)
        np.testing.assert_array_equal(call["covariance"], h["bundle"]["covariance"][4:])
    assert len(h["gate_calls"]) == 2
    assert all(c["family_size"] == 2 and c["minimum_positive_years"] == 6 for c in h["gate_calls"])
    assert summary["historical_gates"] == {"hac_h5": False, "oos_floor_h5": False}
    assert summary["promotion_eligible"] is False
    assert r.status == "FINISHED" and len(r.starts) == 2
    assert r.starts[-1]["recorder_id"] == "frozen-id" and r.starts[-1]["resume"]
    assert all(p.pretrade_cash_weight.iloc[0] == 1. for p in r.objects["analysis/paths.pkl"].values())
    assert set(r.objects["analysis/paths.pkl"]) == {(n, c) for n in ("soxx", "hac_daily", "hac_h5", "oos_floor_h5") for c in (10., 25.)}
    saved = next(data for folder, name, data in r.artifacts if folder == "card" and name == "summary.json")
    assert json.loads(saved) == json.loads((h["card"] / "summary.json").read_text())


@pytest.mark.parametrize("stage", ["fit", "calibration"])
def test_initial_fit_and_calibration_failure_retains_failed_recorder(harness, monkeypatch, stage):
    def fail(*args, **kwargs):
        assert harness["r"].active
        raise RuntimeError("fit-stage failure")
    monkeypatch.setattr(workflow, "fit_annual_oos" if stage == "fit" else "fit_oos_mean_error", fail)
    with pytest.raises(RuntimeError, match="fit-stage failure"):
        workflow.run_oos_uncertainty_horizon(**harness["kwargs"])
    assert harness["r"].status == "FAILED"
    assert len(harness["r"].starts) == 1
    assert not harness["replay_calls"]
    assert not (harness["card"] / "summary.json").exists()


@pytest.mark.parametrize("mutation", ["source", "plan", "execution_provider", "bundle_frame", "bundle_covariance", "canonical_source"])
def test_mutated_sources_fail_without_completed_summary(harness, monkeypatch, mutation):
    original = workflow.fit_annual_oos
    def changed(bundle, plan):
        result = original(bundle, plan)
        if mutation == "source":
            (harness["source"] / workflow.ADDITIONAL_SOURCES[0]).write_text("# changed\n")
        elif mutation == "plan":
            with (harness["card"] / "plan.json").open("a") as stream:
                stream.write("\n")
        elif mutation == "execution_provider":
            harness["audit"]["provider_sha256"] = "changed"
        elif mutation == "bundle_frame":
            bundle["frame"].iloc[0, bundle["frame"].columns.get_loc("feature")] += 1
        elif mutation == "bundle_covariance":
            bundle["covariance"][0] *= 2
        else:
            # Change only the upstream readback returned after the run, while
            # its already-frozen execution-provider audit remains identical.
            harness["bundle"]["frame"].iloc[0, harness["bundle"]["frame"].columns.get_loc("feature")] += 1
        return result
    monkeypatch.setattr(workflow, "fit_annual_oos", changed)
    with pytest.raises(ValueError, match="blocked"):
        workflow.run_oos_uncertainty_horizon(**harness["kwargs"])
    assert harness["r"].status == "FAILED"
    assert not (harness["card"] / "summary.json").exists()


def test_missing_prediction_date_fails_instead_of_using_last_array_row(harness):
    harness["predictions"].drop(index=pd.Timestamp("2017-01-03"), inplace=True)
    with pytest.raises(ValueError, match="prediction coverage"):
        workflow.run_oos_uncertainty_horizon(**harness["kwargs"])
    assert not harness["replay_calls"]
    assert harness["r"].status == "FAILED"


def test_failed_compressed_archive_never_publishes_completed_summary(harness, monkeypatch):
    def fail(*args, **kwargs):
        assert harness["r"].active
        raise RuntimeError("compression archive failed")
    monkeypatch.setattr(workflow, "save_compressed_object", fail)
    with pytest.raises(RuntimeError, match="compression archive failed"):
        workflow.run_oos_uncertainty_horizon(**harness["kwargs"])
    assert harness["r"].status == "FAILED"
    assert not (harness["card"] / "summary.json").exists()


def test_success_resume_rejects_wrong_recorder_identity(harness):
    harness["r"].override_identity = True
    with pytest.raises(ValueError, match="Recorder identity differs"):
        workflow.run_oos_uncertainty_horizon(**harness["kwargs"])
    assert not (harness["card"] / "summary.json").exists()
    assert "analysis/paths.pkl" not in harness["r"].objects


def test_completed_card_is_not_reopened(harness):
    (harness["card"] / "summary.json").write_text('{"immutable": true}\n')
    with pytest.raises(ValueError, match="immutable"):
        workflow.run_oos_uncertainty_horizon(**harness["kwargs"])
    assert not harness["r"].starts and not harness["qlib_calls"]


@pytest.mark.parametrize("mutation", ["family", "candidate_count", "costs", "clock_horizon", "boolean_cadence", "integer_flag", "source_keys"])
def test_invalid_registered_protocol_rejected_before_data_or_record(harness, mutation):
    plan = harness["plan"]
    if mutation == "family": plan["gates"]["family_size"] = 1
    elif mutation == "candidate_count": plan["candidate_budget"] = 3
    elif mutation == "costs": plan["costs_bps"] = [10.]
    elif mutation == "clock_horizon": plan["forecast_horizon_sessions"] = 20
    elif mutation == "boolean_cadence": plan["variants"][0]["rebalance_sessions"] = True
    elif mutation == "integer_flag": plan["variants"][2]["oos_floor"] = 1
    else: del plan["source"]["recorder_id"]
    (harness["card"] / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="blocked"):
        workflow.run_oos_uncertainty_horizon(**harness["kwargs"])
    assert not harness["r"].starts and not harness["qlib_calls"]

"""Runner lifecycle and evidence isolation; no market reruns in these tests."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.workflow import benchmark_relative_research as workflow


class Recorder:
    def __init__(self, recorder_id, local_dir):
        self.id, self.experiment_id = recorder_id, "experiment-1"
        self.local_dir = local_dir
        self.saved, self.logged, self.params, self.metrics = [], [], [], []
        self.fail_artifact_path = None
        (local_dir / "artifacts").mkdir(parents=True)
        for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
            (local_dir / "artifacts" / name).write_text("unrelated checkout snapshot")

    def get_local_dir(self):
        return str(self.local_dir)

    def log_params(self, **kwargs):
        self.params.append(kwargs)

    def log_metrics(self, **kwargs):
        self.metrics.append(kwargs)

    def save_objects(self, *, artifact_path, **objects):
        if artifact_path == self.fail_artifact_path:
            raise RuntimeError("recorder artifact write failed")
        self.saved.append((artifact_path, objects))

    def log_artifact(self, path, *, artifact_path):
        self.logged.append((str(path), artifact_path, Path(path).read_bytes()))


class RecorderFacade:
    def __init__(self, root):
        self.recorder = Recorder("sealed-recorder", root / "experiment-1" / "sealed-recorder")
        self.other = Recorder("wrong-recorder", root / "experiment-1" / "wrong-recorder")
        self.active = None
        self.starts, self.status = [], {}
        self.override_next_resume = False

    @contextmanager
    def start(self, **kwargs):
        assert self.active is None, "nested active Recorder context"
        self.starts.append(kwargs)
        selected = self.recorder
        if kwargs.get("resume") and self.override_next_resume:
            selected = self.other
            self.override_next_resume = False
        self.active = selected
        try:
            yield selected
        except Exception:
            self.status[selected.id] = "FAILED"
            raise
        else:
            self.status[selected.id] = "FINISHED"
        finally:
            self.active = None

    def get_recorder(self):
        assert self.active is not None
        return self.active


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import qlib
    import qlib.workflow

    source = tmp_path / "source"
    evidence = tmp_path / "read_only_evidence"
    card = tmp_path / "new_card"
    uri = tmp_path / "separate_recorder"
    card.mkdir()
    source.mkdir()
    evidence.mkdir()
    for name in workflow.SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# sealed source fixture\n")
    # Isolate runner lifecycle tests from imported-module location. The real
    # source-root binding gets its own failure-path test below.
    monkeypatch.setattr(workflow, "_source_hashes",
        lambda root: {name: workflow._sha(root / name) for name in workflow.SOURCE_FILES})
    plan = json.loads((Path(__file__).resolve().parents[1] /
        "research/benchmark_relative_architecture_20261009/plan.json").read_text())
    plan["registered_at_utc"] = "2020-01-01T00:00:00+00:00"
    (card / "plan.json").write_text(json.dumps(plan))
    (card / "workflow.yaml").write_text("workflow: explicit_test_fixture\n")
    provider = evidence / plan["provider_uri"]
    provider.mkdir(parents=True)
    protected = [evidence / "research/old_card/summary.json", evidence / "production/old_strategy/state.json",
                 source / "production/unchanged/state.json"]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"unchanged": true}\n')
    original = {path: path.read_bytes() for path in protected}
    recorder = RecorderFacade(uri)
    monkeypatch.setattr(qlib.workflow, "R", recorder)
    qlib_calls, source_reads, native_calls, gates_calls = [], [], [], []

    def init(**kwargs):
        assert recorder.active is None, "qlib.init must run outside active Recorder"
        qlib_calls.append(kwargs)

    monkeypatch.setattr(qlib, "init", init)
    audit = {"source_manifest_sha256": "frozen-provider-identity"}

    def load(root):
        assert Path(root) == evidence
        assert recorder.active is None
        source_reads.append(root)
        return provider, deepcopy(audit)

    monkeypatch.setattr(workflow, "load_legacy_execution_view", load)
    dates = pd.DatetimeIndex(["2016-01-04", "2022-12-30", "2023-01-03"])
    frame = pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.Timedelta(days=1),
        "exit_date": dates + pd.Timedelta(days=2)}, index=dates)

    def build(**kwargs):
        assert kwargs == {"provider_uri": provider, "signal_start": plan["signal_start"], "as_of": plan["as_of"]}
        qlib.init(provider_uri=str(provider), region="us")
        return object(), frame.copy(), dates

    monkeypatch.setattr(workflow, "build_relative_dataset", build)

    def replay(frame, **kwargs):
        assert recorder.active is None
        result = frame.reset_index(drop=True).copy()
        result["strategy_return"] = 0.001
        result["cost_bps"] = kwargs["cost_bps"]
        result["turnover"], result["fees"] = 0.0, 0.0
        return result

    monkeypatch.setattr(workflow, "replay_fixed_target", replay)
    monkeypatch.setattr(workflow, "replay_online_allocator",
        lambda frame, experts, **kwargs: (replay(frame, **kwargs), pd.DataFrame({"feedback_count": [0, 1, 2]})))

    def native(frame, path, **kwargs):
        assert recorder.active is None, "native backtest must not reset an active Recorder"
        qlib.init(provider_uri=str(kwargs["provider_uri"]), region="us")
        native_calls.append(kwargs)
        return {"audit": {"status": "passed", "cost_bps": kwargs["cost_bps"]},
            "reconciliation": pd.DataFrame({"native_next_open_return": path.strategy_return.to_numpy()})}

    monkeypatch.setattr(workflow, "native_open_reconciliation", native)

    def gate(paths, benchmarks, **kwargs):
        gates_calls.append(kwargs)
        return {"historical_gate_pass": False, "gates": pd.DataFrame({"all_pass": [False, False]}),
            "rolling_summary": pd.DataFrame({"positive_excess_rate": [0.3, 0.2]}),
            "bootstrap": pd.DataFrame({"lower_bound_positive": [False]})}

    monkeypatch.setattr(workflow, "evaluate_benchmark_relative", gate)
    metrics = {"return_pct": 1.0, "relative_wealth": 1.0, "max_drawdown_pct": -1.0,
               "cvar5_pct": -0.1, "turnover": 0.0}
    monkeypatch.setattr(workflow, "paired_metrics", lambda path, benchmark: metrics.copy())
    monkeypatch.setattr(workflow, "annual_paired_metrics",
        lambda path, benchmark: pd.DataFrame([{"year": 2016, **metrics}]))
    return {"source": source, "evidence": evidence, "card": card, "uri": uri,
        "plan": plan, "provider": provider, "audit": audit, "recorder": recorder,
        "source_reads": source_reads, "native_calls": native_calls, "qlib_calls": qlib_calls,
        "gates_calls": gates_calls, "protected": original,
        "kwargs": {"source_root": source, "evidence_root": evidence, "card_dir": card, "recorder_uri": uri}}


def test_explicit_destinations_isolate_old_evidence_and_native_lifecycle(harness):
    summary, tables, conclusion = workflow.run_benchmark_relative_research(**harness["kwargs"])
    r = harness["recorder"]
    assert summary["recorder_id"] == r.recorder.id
    assert summary["recorder_uri"] == str(harness["uri"].resolve())
    assert len(harness["native_calls"]) == 10
    assert len(harness["qlib_calls"]) == 11
    assert len(harness["source_reads"]) == 2
    assert r.status[r.recorder.id] == "FINISHED"
    assert r.starts == [
        {"experiment_name": harness["plan"]["experiment"], "uri": str(harness["uri"])},
        {"experiment_name": harness["plan"]["experiment"], "recorder_id": r.recorder.id,
         "uri": str(harness["uri"]), "resume": True},
    ]
    binding = json.loads((harness["card"] / "run_binding.json").read_text())
    assert binding["code_root"] == str(harness["source"])
    assert binding["read_only_evidence_root"] == str(harness["evidence"])
    assert binding["card_dir"] == str(harness["card"])
    assert binding["recorder_uri"] == str(harness["uri"])
    assert summary["promotion_eligible"] is False
    assert "不晋升" in conclusion
    assert all((harness["card"] / name).exists() for name in tables)
    assert not any((harness["card"] / name).exists() for name in ("predictions.csv", "signals.csv", "portfolio_report.csv"))
    assert {name for name, _ in r.recorder.saved} >= {"inputs", "analysis", "portfolio_analysis"}
    assert harness["gates_calls"][0]["minimum_positive_years"] == 6
    assert harness["gates_calls"][0]["expected_years"] == tuple(range(2016, 2023))
    for path, old_bytes in harness["protected"].items():
        assert path.read_bytes() == old_bytes
    assert not list(harness["source"].glob("research/**/summary.json"))
    for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
        assert not (r.recorder.local_dir / "artifacts" / name).exists()


def test_completed_card_is_immutable_before_any_source_or_recorder_work(harness):
    original = b'{"completed": true}\n'
    (harness["card"] / "summary.json").write_bytes(original)
    with pytest.raises(ValueError, match="completed card is immutable"):
        workflow.run_benchmark_relative_research(**harness["kwargs"])
    assert (harness["card"] / "summary.json").read_bytes() == original
    assert harness["source_reads"] == []
    assert harness["recorder"].starts == []
    assert not (harness["card"] / "run_binding.json").exists()


def test_declared_source_root_must_match_imported_modules(tmp_path):
    with pytest.raises(ValueError, match="imported implementation differs"):
        workflow._source_hashes(tmp_path)


def test_plan_provider_must_match_audited_read_only_source(harness):
    harness["plan"]["provider_uri"] = "data/qlib/different_provider"
    (harness["card"] / "plan.json").write_text(json.dumps(harness["plan"]))
    with pytest.raises(ValueError, match="plan provider differs"):
        workflow.run_benchmark_relative_research(**harness["kwargs"])
    assert harness["recorder"].starts == []
    assert not (harness["card"] / "run_binding.json").exists()


@pytest.mark.parametrize("key,value", [
    ("evidence_kind", "independent_forward"), ("candidate_budget", 2),
    ("costs_bps", [10.0]), ("anchor_weights", [1, 0, 0]),
    ("registered_before_replay", False), ("registered_at_utc", "2020-01-01"),
    ("registered_at_utc", "2999-01-01T00:00:00+00:00"),
    ("prior", [1, 1, 1]), ("share", -1),
])
def test_invalid_plan_fails_before_writable_run_or_native_replay(harness, key, value):
    harness["plan"][key] = value
    (harness["card"] / "plan.json").write_text(json.dumps(harness["plan"]))
    with pytest.raises(ValueError, match="blocked"):
        workflow.run_benchmark_relative_research(**harness["kwargs"])
    assert not (harness["card"] / "run_binding.json").exists()
    assert harness["source_reads"] == []
    assert harness["recorder"].starts == []


@pytest.mark.parametrize("changed", ["source", "plan", "provider"])
def test_changed_identity_fails_before_final_sidecars_or_summary(harness, monkeypatch, changed):
    original_native = workflow.native_open_reconciliation

    def mutating_native(*args, **kwargs):
        result = original_native(*args, **kwargs)
        if not getattr(mutating_native, "changed", False):
            mutating_native.changed = True
            if changed == "source":
                (harness["source"] / workflow.SOURCE_FILES[0]).write_text("# changed after binding\n")
            elif changed == "plan":
                with (harness["card"] / "plan.json").open("a") as stream:
                    stream.write("\n")
            else:
                harness["audit"]["source_manifest_sha256"] = "changed-provider"
        return result

    monkeypatch.setattr(workflow, "native_open_reconciliation", mutating_native)
    with pytest.raises(ValueError, match="source/plan changed"):
        workflow.run_benchmark_relative_research(**harness["kwargs"])
    assert harness["recorder"].status["sealed-recorder"] == "FAILED"
    assert not (harness["card"] / "summary.json").exists()
    assert not (harness["card"] / "backtest_benchmark_comparison.csv").exists()
    for path, old_bytes in harness["protected"].items():
        assert path.read_bytes() == old_bytes


def test_resume_must_return_the_original_recorder_identity(harness):
    harness["recorder"].override_next_resume = True
    with pytest.raises(ValueError, match="[Rr]ecorder|identity"):
        workflow.run_benchmark_relative_research(**harness["kwargs"])
    assert not harness["recorder"].other.saved
    assert not (harness["card"] / "summary.json").exists()


def test_recorder_archive_failure_must_not_publish_completed_summary(harness):
    harness["recorder"].recorder.fail_artifact_path = "portfolio_analysis"
    with pytest.raises(RuntimeError, match="artifact write failed"):
        workflow.run_benchmark_relative_research(**harness["kwargs"])
    assert harness["recorder"].status["sealed-recorder"] == "FAILED"
    assert not (harness["card"] / "summary.json").exists()


def test_native_failure_marks_the_original_recorder_failed(harness, monkeypatch):
    def fail(*args, **kwargs):
        assert harness["recorder"].active is None
        raise RuntimeError("native Qlib failure")

    monkeypatch.setattr(workflow, "native_open_reconciliation", fail)
    with pytest.raises(RuntimeError, match="native Qlib failure"):
        workflow.run_benchmark_relative_research(**harness["kwargs"])
    assert harness["recorder"].status["sealed-recorder"] == "FAILED"
    assert harness["recorder"].starts[-1]["recorder_id"] == "sealed-recorder"
    assert not (harness["card"] / "summary.json").exists()


def test_failure_resume_also_checks_the_exact_recorder_identity(harness, monkeypatch):
    harness["recorder"].override_next_resume = True

    def fail(*args, **kwargs):
        raise RuntimeError("native failure before resume")

    monkeypatch.setattr(workflow, "native_open_reconciliation", fail)
    with pytest.raises(ValueError, match="Recorder identity differs"):
        workflow.run_benchmark_relative_research(**harness["kwargs"])
    assert harness["recorder"].starts[-1]["recorder_id"] == "sealed-recorder"
    assert not harness["recorder"].other.saved
    assert not (harness["card"] / "summary.json").exists()

"""Synthetic provenance and inventory guards; no market checkpoint evaluation."""
from __future__ import annotations

import copy
from concurrent.futures import Future
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.workflow import anchor_checkpoint_feasibility as workflow


@pytest.fixture(scope="module")
def plan():
    path = Path(__file__).parents[1] / "research/benchmark_anchor_risk_feasibility_20261003/selection_plan.json"
    return json.loads(path.read_text())


class _RecorderSpy:
    def __init__(self, root):
        self.uri = str(root / "qlib_recorder")
        self.local = Path(self.uri) / "experiment" / "recorder"
        self.local.mkdir(parents=True)
        self.id = "recorder"
        self.calls = []
        self.mutate_during_load = False

    def get_local_dir(self):
        return str(self.local)

    def load_object(self, artifact):
        self.calls.append(artifact)
        if self.mutate_during_load:
            (self.local / "artifacts" / artifact).write_bytes(b"changed during load")
        return {"sentinel": "frozen object"}


def _entry(recorder, artifact, content):
    path = recorder.local / "artifacts" / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": str(Path("qlib_recorder/experiment/recorder/artifacts") / artifact),
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def test_parent_hash_is_verified_before_native_deserialization_and_after_read(tmp_path):
    recorder = _RecorderSpy(tmp_path)
    artifact = "inputs/frozen.pkl"
    entry = _entry(recorder, artifact, b"immutable fixture bytes")
    value, audit = workflow.checked_parent_load(recorder, tmp_path, {"entries": [entry]}, artifact)
    assert value == {"sentinel": "frozen object"}
    assert recorder.calls == [artifact]
    assert audit["sha256"] == entry["sha256"]
    recorder.calls.clear()
    recorder.mutate_during_load = True
    with pytest.raises(ValueError, match="changed during read"):
        workflow.checked_parent_load(recorder, tmp_path, {"entries": [entry]}, artifact)
    assert recorder.calls == [artifact]


@pytest.mark.parametrize("fault", ["content", "length", "missing_entry", "duplicate_entry"])
def test_unverified_parent_bytes_are_never_deserialized(tmp_path, fault):
    recorder = _RecorderSpy(tmp_path)
    artifact = "inputs/frozen.pkl"
    entry = _entry(recorder, artifact, b"good")
    manifest = {"entries": [entry]}
    if fault == "content": (recorder.local / "artifacts" / artifact).write_bytes(b"evil")
    if fault == "length": (recorder.local / "artifacts" / artifact).write_bytes(b"longer")
    if fault == "missing_entry": manifest["entries"] = []
    if fault == "duplicate_entry": manifest["entries"].append(copy.deepcopy(entry))
    with pytest.raises(ValueError, match="artifact|manifest|deserialization"):
        workflow.checked_parent_load(recorder, tmp_path, manifest, artifact)
    assert recorder.calls == []


@pytest.mark.parametrize("escape", ["parent", "absolute", "symlink"])
def test_artifact_paths_cannot_escape_the_pinned_recorder_tree(tmp_path, escape):
    recorder = _RecorderSpy(tmp_path)
    content = b"known bytes at an unauthorized location"
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(content)
    if escape == "parent":
        artifact = "../outside.pkl"
        (recorder.local / "outside.pkl").write_bytes(content)
    elif escape == "absolute": artifact = str(outside)
    else:
        artifact = "inputs/linked.pkl"
        link = recorder.local / "artifacts" / artifact
        link.parent.mkdir(parents=True)
        link.symlink_to(outside)
    entry = {"path": str(Path("qlib_recorder/experiment/recorder/artifacts") / artifact),
             "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    with pytest.raises(ValueError, match="artifact|path|escape|contain"):
        workflow.checked_parent_load(recorder, tmp_path, {"entries": [entry]}, artifact)
    assert recorder.calls == []


@pytest.fixture(scope="module")
def complete_results(plan):
    results = {}
    for variant, year, seed in sorted(workflow.expected_identities(plan)):
        common = {"variant": variant, "model_year": year, "seed": seed}
        epochs = [plan["inventory"]["null_epoch_label"], *plan["inventory"]["saved_epochs"]]
        statuses = [{**common, "epoch": epoch, "explicit_null": epoch == epochs[0]} for epoch in epochs]
        metrics = [{**status, "cost_bps": cost} for status in statuses for cost in plan["recompute"]["cost_bps"]]
        results[(variant, year, seed)] = {"checkpoint_metrics": pd.DataFrame(metrics),
                                         "checkpoint_status": pd.DataFrame(statuses)}
    return results


def test_exact_all_run_inventory_keeps_nulls_separate_and_never_deduplicates_models(plan, complete_results):
    costs, joint = workflow.verify_inventory(complete_results, plan)
    assert len(costs) == 15372 and len(joint) == 7686
    assert costs.explicit_null.sum() == 126 and joint.explicit_null.sum() == 63
    assert (~costs.explicit_null).sum() == 15246 and (~joint.explicit_null).sum() == 7623
    assert len(workflow.expected_identities(plan)) == 63


@pytest.mark.parametrize("fault", ["missing_run", "extra_run", "missing_cost", "duplicate_cost", "missing_joint", "duplicate_joint", "extra_cost"])
def test_missing_duplicate_or_extraneous_inventory_is_rejected(plan, complete_results, fault):
    results = copy.deepcopy(complete_results)
    key = next(iter(results))
    if fault == "missing_run": results.pop(key)
    if fault == "extra_run": results[("price", 2023, 17)] = copy.deepcopy(results[key])
    if fault == "missing_cost": results[key]["checkpoint_metrics"] = results[key]["checkpoint_metrics"].iloc[1:]
    if fault == "duplicate_cost": results[key]["checkpoint_metrics"] = pd.concat([results[key]["checkpoint_metrics"], results[key]["checkpoint_metrics"].iloc[:1]])
    if fault == "missing_joint": results[key]["checkpoint_status"] = results[key]["checkpoint_status"].iloc[1:]
    if fault == "duplicate_joint": results[key]["checkpoint_status"] = pd.concat([results[key]["checkpoint_status"], results[key]["checkpoint_status"].iloc[:1]])
    if fault == "extra_cost": results[key]["checkpoint_metrics"].loc[0, "cost_bps"] = 35
    with pytest.raises(ValueError, match="inventory|duplicate|incomplete|extraneous|identity"):
        workflow.verify_inventory(results, plan)


@pytest.mark.parametrize("fault", ["aliased_duplicate_epoch", "aliased_duplicate_year", "fractional_year", "fractional_seed", "wrong_null_flag", "missing_null_flag"])
def test_identity_coercion_and_bad_null_flags_cannot_hide_extra_or_mislabeled_records(plan, complete_results, fault):
    results = copy.deepcopy(complete_results)
    key = next(iter(results));metrics = results[key]["checkpoint_metrics"]
    row = metrics.index[metrics.epoch.eq(0)][0]
    if fault in {"aliased_duplicate_epoch", "aliased_duplicate_year"}:
        duplicate = metrics.loc[[row]].copy()
        column = "epoch" if fault == "aliased_duplicate_epoch" else "model_year"
        duplicate[column] = duplicate[column].astype(str)
        results[key]["checkpoint_metrics"] = pd.concat([metrics, duplicate], ignore_index=True)
    if fault == "fractional_year":
        metrics["model_year"] = metrics.model_year.astype(float)
        metrics.loc[row, "model_year"] += .5
    if fault == "fractional_seed":
        metrics["seed"] = metrics.seed.astype(float)
        metrics.loc[row, "seed"] += .5
    if fault == "wrong_null_flag": metrics.loc[row, "explicit_null"] = True
    if fault == "missing_null_flag":
        metrics["explicit_null"] = metrics.explicit_null.astype(object)
        metrics.loc[row, "explicit_null"] = None
    with pytest.raises(ValueError, match="identity|duplicate|integer|null|boolean|record|inventory"):
        workflow.verify_inventory(results, plan)


@pytest.mark.parametrize("fault", ["duplicate_variant", "duplicate_year", "duplicate_seed", "fractional_year", "boolean_epoch"])
def test_declared_identity_lists_cannot_be_silently_deduplicated_or_truncated(plan, fault):
    changed = copy.deepcopy(plan);inventory = changed["inventory"]
    if fault == "duplicate_variant": inventory["variants"].append(inventory["variants"][0])
    if fault == "duplicate_year": inventory["training_cutoff_years"].append(inventory["training_cutoff_years"][0])
    if fault == "duplicate_seed": inventory["seeds"].append(inventory["seeds"][0])
    if fault == "fractional_year": inventory["training_cutoff_years"][0] += .5
    if fault == "boolean_epoch": inventory["saved_epochs"][0] = False
    with pytest.raises(ValueError, match="inventory|duplicate|integer|identity|registered"):
        workflow.expected_identities(changed)


@pytest.mark.parametrize("fault", [None, "manifest_bytes", "calendar_interior", "summary_bytes"])
def test_provider_calendar_metadata_is_byte_pinned_before_any_parent_objects_load(tmp_path, plan, monkeypatch, fault):
    import qlib
    import qlib.data as native_data
    import qlib.workflow as native_workflow

    configured = copy.deepcopy(plan)
    source = configured["source"]
    card = tmp_path / "research" / source["experiment"]
    card.mkdir(parents=True)
    provider = tmp_path / "provider"
    provider.mkdir()
    calendar = pd.to_datetime(["2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25"])
    manifest_path = provider / "price_basis_manifest.json"
    # Deliberately formatted bytes: the parent's recorded SHA is not a hash
    # of an independently reserialized or compact canonical JSON object.
    manifest_path.write_text(json.dumps({"reference_calendar": {"dates": calendar.strftime("%Y-%m-%d").tolist()}}, indent=3) + "\n")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (card / "artifact_manifest.json").write_text('{"entries": []}\n')
    source["parent_artifact_manifest_sha256"] = hashlib.sha256((card / "artifact_manifest.json").read_bytes()).hexdigest()
    (card / "summary.json").write_text(json.dumps({"recorder_id": source["recorder_id"],
        "plan_sha256": source["parent_plan_sha256"], "provider_uri": "provider",
        "source_audit": {"provider_manifest_sha256": manifest_digest}}))
    summary_entry = {"path": "qlib_recorder/experiment/recorder/artifacts/card/summary.json",
                     "sha256": hashlib.sha256((card / "summary.json").read_bytes()).hexdigest()}
    (card / "artifact_manifest.json").write_text(json.dumps({"entries": [summary_entry]}))
    source["parent_artifact_manifest_sha256"] = hashlib.sha256((card / "artifact_manifest.json").read_bytes()).hexdigest()
    if fault == "manifest_bytes": manifest_path.write_text(manifest_path.read_text() + " ")
    if fault == "summary_bytes": (card / "summary.json").write_text((card / "summary.json").read_text() + " ")
    actual_calendar = calendar.delete(1) if fault == "calendar_interior" else calendar
    monkeypatch.setattr(qlib, "init", lambda **kwargs: None)
    monkeypatch.setattr(native_data, "D", SimpleNamespace(calendar=lambda **kwargs: actual_calendar))

    class MetadataVerifiedBeforeLoad(Exception):
        pass

    class FakeWorkflow:
        @staticmethod
        @contextmanager
        def uri_context(uri):
            raise MetadataVerifiedBeforeLoad()
            yield

    monkeypatch.setattr(native_workflow, "R", FakeWorkflow)
    monkeypatch.setattr(workflow, "checked_parent_load", lambda *args: pytest.fail("metadata guard reached object loading"))
    if fault is None:
        with pytest.raises(MetadataVerifiedBeforeLoad):
            workflow.load_registered_inputs(tmp_path, configured)
    else:
        with pytest.raises(ValueError, match="manifest|calendar|summary"):
            workflow.load_registered_inputs(tmp_path, configured)


def test_dispatch_rejects_diagnostic_year_before_any_checkpoint_evaluation():
    from src.analysis.anchor_checkpoint_feasibility import CheckpointFeasibilitySpec, CheckpointIntegrityError
    fold = {"features": None, "returns": None, "rows": None}
    with pytest.raises(CheckpointIntegrityError, match="development fit inventory"):
        workflow._evaluate_job((("price", 2023, 17), fold, {}, CheckpointFeasibilitySpec()))


def test_worker_integrity_failure_is_saved_and_never_becomes_a_partial_success(tmp_path, plan, monkeypatch):
    import qlib.workflow as native_workflow
    from src.analysis.anchor_checkpoint_feasibility import CheckpointIntegrityError

    card = tmp_path / "card"
    card.mkdir()
    (card / "selection_plan.json").write_text(json.dumps(plan))
    for relative in ("src/analysis/anchor_checkpoint_feasibility.py", "src/workflow/anchor_checkpoint_feasibility.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source identity fixture\n")
    identities = workflow.expected_identities(plan)
    audit_shape = {"epochs": [{"epoch": epoch, "checkpoint": {"tensor_sha256": "0" * 64}}
                              for epoch in plan["inventory"]["saved_epochs"]],
                   "training_feature_sha256": "a" * 64, "training_return_sha256": "b" * 64}
    bundle = {"source_artifacts": [], "parent_source_hashes": {}, "maturity": pd.DataFrame(), "parent_plan": {},
              "folds": {identity[:2]: {} for identity in identities},
              "training_audits": {identity: copy.deepcopy(audit_shape) for identity in identities}}
    state = {"objects": {}, "failed": False, "parameters": {}}
    evidence = {"synthetic": True, "reason": "stored utility mismatch", "offending_path": [1., .99]}

    class Recorder:
        id = "synthetic-new-recorder"

        @staticmethod
        def save_objects(*, artifact_path, **objects): state["objects"][artifact_path] = objects

        @staticmethod
        def log_artifact(path, *, artifact_path): pass

    class FakeWorkflow:
        @staticmethod
        def set_uri(uri): pass

        @staticmethod
        @contextmanager
        def start(**kwargs):
            try:
                yield
            except Exception:
                state["failed"] = True
                raise

        @staticmethod
        def get_recorder(): return Recorder()

        @staticmethod
        def log_params(**parameters): state["parameters"] = parameters

        @staticmethod
        def save_objects(**kwargs): Recorder.save_objects(**kwargs)

    class InlinePool:
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

    def failed_worker(payload):
        assert payload[0] in identities and payload[0][1] < 2023
        raise CheckpointIntegrityError("synthetic stored utility mismatch", evidence=evidence)

    monkeypatch.setattr(native_workflow, "R", FakeWorkflow)
    monkeypatch.setattr(workflow, "CANONICAL_ROOT", tmp_path)
    monkeypatch.setattr(workflow, "load_registered_inputs", lambda *args: bundle)
    monkeypatch.setattr(workflow, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(workflow, "_evaluate_job", failed_worker)
    with pytest.raises(CheckpointIntegrityError, match="synthetic stored utility mismatch"):
        workflow.run_experiment(tmp_path, card, workers=1)
    assert state["failed"]
    assert state["objects"]["integrity_failure"]["offending_evidence.pkl"] == evidence
    assert state["parameters"]["market_fits"] is False
    assert state["parameters"]["out_of_sample_evaluation"] is False
    assert "analysis" not in state["objects"]
    assert not (card / "summary.json").exists()

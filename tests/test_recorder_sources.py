"""Source reads verify bytes before loading and leave local Qlib state alone."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from src.core.recorder_sources import load_verified_recorder_objects


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(path):
    return {str(item.relative_to(path)): item.read_bytes() for item in path.rglob("*") if item.is_file()}


@pytest.fixture
def source(tmp_path, monkeypatch):
    import qlib
    import qlib.workflow

    card = tmp_path / "research/sample_card"
    uri, local = card / "qlib_recorder", card / "qlib_recorder/1/rec-1"
    names = ["analysis/rows.pkl", "models/frozen.pkl"]
    objects = {names[0]: {"values": [1., 2.]}, names[1]: {"kind": "frozen"}}
    for name in names:
        path = local / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(objects[name]))
    summary = {"experiment": "sample", "experiment_id": "1", "recorder_id": "rec-1",
        "recorder_uri": str(uri), "recorder_local_dir": str(local)}
    manifest = {"schema_version": "test_artifact_manifest_v1", "recorder_id": "rec-1", "entries": [
        {"path": str((local / "artifacts" / name).relative_to(tmp_path)),
         "sha256": sha(local / "artifacts" / name), "bytes": (local / "artifacts" / name).stat().st_size} for name in names]}
    state = {"status": "FINISHED", "experiment_name": "sample", "run_id": "rec-1", "experiment_id": "1",
             "uri": "/previous/read-only/default", "load_hook": None, "loaded": [], "contexts": [], "queries": []}
    recorder = SimpleNamespace(id="rec-1", experiment_id="1", uri=str(uri),
        artifact_uri=(local / "artifacts").as_uri(), get_local_dir=lambda: str(local))
    def get_run(rid):
        return SimpleNamespace(info=SimpleNamespace(status=state["status"], run_id=state["run_id"],
            experiment_id=state["experiment_id"], artifact_uri=recorder.artifact_uri))
    recorder.client = SimpleNamespace(get_run=get_run,
        get_experiment=lambda eid: SimpleNamespace(name=state["experiment_name"]))
    def load(name):
        state["loaded"].append(name)
        result = pickle.loads((local / "artifacts" / name).read_bytes())
        if state["load_hook"]: state["load_hook"](name)
        return result
    recorder.load_object = load
    @contextmanager
    def uri_context(value):
        assert isinstance(value, str) and Path(value).is_absolute() and not value.startswith("file:")
        previous = state["uri"]
        state["contexts"].append(value)
        state["uri"] = value
        try: yield
        finally: state["uri"] = previous
    def get_recorder(**kwargs):
        state["queries"].append(kwargs)
        return recorder
    manager = SimpleNamespace(active_experiment=None)
    facade = SimpleNamespace(uri_context=uri_context, get_recorder=get_recorder,
        exp_manager=manager, get_uri=lambda: state["uri"])
    monkeypatch.setattr(qlib.workflow, "R", facade)
    monkeypatch.setattr(qlib, "init", lambda *a, **k: pytest.fail("source reader must never initialize Qlib"))
    fixture = dict(card=card, uri=uri, local=local, names=names, objects=objects,
        summary=summary, manifest=manifest, state=state, recorder=recorder, manager=manager)
    write_metadata(fixture)
    return fixture


def write_metadata(source):
    (source["card"] / "summary.json").write_text(json.dumps(source["summary"]))
    (source["card"] / "verification_manifest.json").write_text(json.dumps(source["manifest"]))


def read(source, **kwargs):
    options = {"expected_recorder_id": "rec-1", "artifacts": source["names"]}
    options.update(kwargs)
    return load_verified_recorder_objects(source["card"], **options)


def test_verified_objects_are_read_only_with_restored_uri_and_byte_hashes(source):
    before = tree(source["card"])
    expected = {name: sha(source["local"] / "artifacts" / name) for name in source["names"]}
    objects, audit = read(source, expected_artifact_sha256=expected)
    assert objects == source["objects"]
    assert tree(source["card"]) == before
    assert source["state"]["uri"] == "/previous/read-only/default"
    assert source["manager"].active_experiment is None
    assert source["state"]["queries"] == [{"experiment_name": "sample", "recorder_id": "rec-1"}]
    assert audit["recorder_uri"] == str(source["uri"])
    assert audit["summary_sha256"] == sha(source["card"] / "summary.json")
    assert audit["verification_manifest_sha256"] == sha(source["card"] / "verification_manifest.json")
    assert audit["read_only"] and audit["hashes_checked_before_unpickling"]
    assert {name: row["sha256"] for name, row in audit["artifacts"].items()} == expected


def test_card_relative_manifest_entries_are_supported(source):
    for entry in source["manifest"]["entries"]:
        entry["path"] = str(Path(entry["path"]).relative_to("research/sample_card"))
    write_metadata(source)
    assert read(source)[0] == source["objects"]


@pytest.mark.parametrize("name", ["../rows.pkl", "/absolute.pkl", "analysis/../rows.pkl", "./rows.pkl", "analysis//rows.pkl", "analysis/", "C:\\rows.pkl", "https://host/object", ""])
def test_unsafe_artifact_paths_never_reach_unpickling(source, name):
    with pytest.raises(ValueError, match="blocked"):
        read(source, artifacts=[name])
    assert not source["state"]["loaded"]


@pytest.mark.parametrize("corruption", ["summary_id", "manifest_id", "entries_schema", "duplicate", "missing", "bad_digest", "wrong_digest", "bytes", "experiment", "remote_uri", "file_uri", "outside_local", "returned_id", "returned_uri", "returned_artifact", "returned_directory", "run_id", "run_experiment", "status", "active"])
def test_metadata_identity_schema_or_status_fails_before_unpickle(source, corruption, tmp_path):
    summary, manifest, state, recorder = (source[key] for key in ("summary", "manifest", "state", "recorder"))
    if corruption == "summary_id": summary["recorder_id"] = "other"
    elif corruption == "manifest_id": manifest["recorder_id"] = "other"
    elif corruption == "entries_schema": manifest["entries"] = {}
    elif corruption == "duplicate": manifest["entries"].append(deepcopy(manifest["entries"][0]))
    elif corruption == "missing": manifest["entries"].pop()
    elif corruption == "bad_digest": manifest["entries"][0]["sha256"] = "invalid"
    elif corruption == "wrong_digest": manifest["entries"][0]["sha256"] = "0" * 64
    elif corruption == "bytes": manifest["entries"][0]["bytes"] += 1
    elif corruption == "experiment": state["experiment_name"] = "other"
    elif corruption == "remote_uri": summary["recorder_uri"] = "https://tracking.example/store"
    elif corruption == "file_uri": summary["recorder_uri"] = source["uri"].as_uri()
    elif corruption == "outside_local": summary["recorder_local_dir"] = str(tmp_path)
    elif corruption == "returned_id": recorder.id = "other"
    elif corruption == "returned_uri": recorder.uri = str(tmp_path)
    elif corruption == "returned_artifact": recorder.artifact_uri = "s3://bucket/artifacts"
    elif corruption == "returned_directory": recorder.get_local_dir = lambda: str(tmp_path)
    elif corruption == "run_id": state["run_id"] = "other"
    elif corruption == "run_experiment": state["experiment_id"] = "2"
    elif corruption == "status": state["status"] = "FAILED"
    else: source["manager"].active_experiment = SimpleNamespace(active_recorder=object())
    write_metadata(source)
    with pytest.raises(ValueError, match="blocked"):
        read(source)
    assert not state["loaded"]
    assert state["uri"] == "/previous/read-only/default"


def test_explicit_digest_inventory_and_values_are_binding(source):
    with pytest.raises(ValueError, match="inventory differs"):
        read(source, expected_artifact_sha256={source["names"][0]: "0" * 64})
    with pytest.raises(ValueError, match="explicit artifact digest differs"):
        read(source, expected_artifact_sha256=dict.fromkeys(source["names"], "0" * 64))
    assert not source["state"]["loaded"]


def test_all_requested_hashes_are_checked_before_loading_the_first_object(source):
    path = source["local"] / "artifacts" / source["names"][1]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 differs before loading"):
        read(source)
    assert not source["state"]["loaded"]


@pytest.mark.parametrize("mutation", ["summary", "manifest", "first_artifact", "next_artifact", "status"])
def test_mutation_during_load_prevents_returning_objects_and_restores_context(source, mutation):
    def mutate(name):
        if len(source["state"]["loaded"]) != 1: return
        if mutation in ("summary", "manifest"):
            path = source["card"] / ("summary.json" if mutation == "summary" else "verification_manifest.json")
            path.write_bytes(path.read_bytes() + b"\n")
        elif mutation == "status": source["state"]["status"] = "RUNNING"
        else:
            index = 0 if mutation == "first_artifact" else 1
            path = source["local"] / "artifacts" / source["names"][index]
            path.write_bytes(path.read_bytes() + b"changed")
    source["state"]["load_hook"] = mutate
    with pytest.raises(ValueError, match="blocked"):
        read(source)
    assert source["state"]["uri"] == "/previous/read-only/default"
    if mutation == "next_artifact": assert len(source["state"]["loaded"]) == 1


def test_symbolic_link_artifact_cannot_escape_to_a_private_file(source, tmp_path):
    name = source["names"][0]
    path = source["local"] / "artifacts" / name
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes|symbolic"):
        read(source)
    assert not source["state"]["loaded"]


def test_real_local_qlib_roundtrip_does_not_write_source_or_reinitialize(tmp_path, monkeypatch):
    import qlib
    from qlib.workflow import R
    from src.data.qlib_native import build_qlib_store_from_frame

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    dates = pd.bdate_range("2024-01-02", periods=2)
    build_qlib_store_from_frame(pd.DataFrame({"datetime": dates, "instrument": "SOXX", "close": [100., 101.]}),
        provider_uri=tmp_path / "market", field_map={"close": "close"}, as_of="2024-01-03", source="test")
    card = tmp_path / "card"
    uri = card / "qlib_recorder"
    uri.mkdir(parents=True)
    with R.start(experiment_name="verified_read_test", uri=str(uri)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path="analysis", **{"value.pkl": {"test": [1, 2, 3]}})
        rid, eid, local = rec.id, rec.experiment_id, Path(rec.get_local_dir())
    summary = {"experiment": "verified_read_test", "experiment_id": eid, "recorder_id": rid,
        "recorder_uri": str(uri), "recorder_local_dir": str(local)}
    artifact = local / "artifacts/analysis/value.pkl"
    manifest = {"recorder_id": rid, "entries": [{"path": str(artifact.relative_to(card)), "sha256": sha(artifact), "bytes": artifact.stat().st_size}]}
    (card / "summary.json").write_text(json.dumps(summary))
    (card / "verification_manifest.json").write_text(json.dumps(manifest))
    before, original_uri = tree(card), R.get_uri()
    monkeypatch.setattr(qlib, "init", lambda *a, **k: pytest.fail("must not call qlib.init during a source read"))
    objects, audit = load_verified_recorder_objects(card, expected_recorder_id=rid, artifacts=["analysis/value.pkl"])
    assert objects == {"analysis/value.pkl": {"test": [1, 2, 3]}}
    assert tree(card) == before and R.get_uri() == original_uri
    assert R.exp_manager.active_experiment is None
    assert audit["recorder_uri"] == str(uri) and audit["read_only"] is True

"""Read selected completed stages from an explicitly sealed FAILED Recorder.

This is separate from the FINISHED-only source reader. It never resumes a run,
and validates the complete sealed inventory before any pickle is deserialized.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.core.recorder_sources import _local, _relative, _require, _sha


def read_completed_checkpoint(card, *, manifest_sha256, recorder_id,
                              experiment_id, experiment_name, artifacts,
                              manifest_name="failure_manifest.json", progress=None):
    from qlib.workflow import R

    card = Path(card).resolve()
    manifest_path = card / _relative(manifest_name, "manifest")
    _require(manifest_path.is_file() and not manifest_path.is_symlink()
             and _sha(manifest_path) == manifest_sha256, "checkpoint manifest SHA-256 differs")
    sealed_bytes = manifest_path.read_bytes()
    manifest = json.loads(sealed_bytes)
    _require(manifest.get("recorder_id") == recorder_id
             and str(manifest.get("experiment_id")) == experiment_id
             and manifest.get("recorder_status") == "FAILED", "checkpoint seal identity differs")
    uri = card / "qlib_recorder"
    local = uri / experiment_id / recorder_id
    prefix = ("qlib_recorder", experiment_id, recorder_id, "artifacts")
    entries = manifest.get("entries")
    _require(isinstance(entries, list) and bool(entries), "checkpoint inventory missing")
    recorded = {}
    for entry in entries:
        relative = _relative(entry.get("path"), "checkpoint entry")
        _require(relative.parts[:4] == prefix and len(relative.parts) > 4,
                 "checkpoint entry escapes exact run")
        name = Path(*relative.parts[4:]).as_posix()
        _require(name not in recorded, "duplicate checkpoint entry")
        path = card / relative
        _require(not any(item.is_symlink() for item in (path, *path.parents)
                         if item.is_relative_to(card)), "checkpoint symlinks unsupported")
        _require(path.is_file() and path.stat().st_size == entry.get("bytes")
                 and _sha(path) == entry.get("sha256"), "checkpoint artifact differs: " + name)
        recorded[name] = (path, entry)
    actual = {path.relative_to(local / "artifacts").as_posix()
              for path in (local / "artifacts").rglob("*") if path.is_file()}
    _require(actual == set(recorded), "checkpoint inventory changed")
    _require(isinstance(artifacts, (list, tuple)) and bool(artifacts)
             and len(set(artifacts)) == len(artifacts)
             and set(artifacts).issubset(recorded), "checkpoint requested inventory differs")
    meta = card / _relative(manifest.get("run_metadata_path"), "run metadata")
    _require(meta == local / "meta.yaml" and not meta.is_symlink()
             and _sha(meta) == manifest.get("run_metadata_sha256"), "checkpoint run metadata changed")
    original_uri, active = R.get_uri(), R.exp_manager.active_experiment
    _require(active is None or active.active_recorder is None,
             "checkpoint read requires no writable Recorder")
    result = {}
    with R.uri_context(str(uri)):
        # create=False prevents missing identities from silently creating an experiment.
        experiment = R.get_exp(experiment_id=experiment_id, create=False)
        _require(experiment.name == experiment_name, "checkpoint experiment name differs")
        recorder = experiment.get_recorder(recorder_id=recorder_id, create=False)
        _require(recorder.id == recorder_id and str(recorder.experiment_id) == experiment_id
                 and _local(recorder.get_local_dir(), "checkpoint local directory") == local
                 and _local(recorder.artifact_uri, "checkpoint artifacts", allow_file_uri=True)
                 == local / "artifacts", "checkpoint Qlib location differs")
        _require(recorder.client.get_run(recorder.id).info.status == "FAILED",
                 "checkpoint is no longer FAILED")
        for name in artifacts:
            path, entry = recorded[name]
            _require(manifest_path.read_bytes() == sealed_bytes and _sha(meta)
                     == manifest["run_metadata_sha256"] and _sha(path) == entry["sha256"],
                     "checkpoint changed before unpickling: " + name)
            if progress:
                progress("attempt", name)
            try:
                result[name] = recorder.load_object(name)
            except BaseException:
                if progress:
                    progress("unknown_partial", name)
                raise
            if progress:
                progress("completed", name)
        _require(recorder.client.get_run(recorder.id).info.status == "FAILED",
                 "checkpoint status changed while reading")
    _require(R.get_uri() == original_uri and R.exp_manager.active_experiment is active,
             "checkpoint read changed Qlib context")
    _require(manifest_path.read_bytes() == sealed_bytes
             and _sha(meta) == manifest["run_metadata_sha256"], "checkpoint metadata changed")
    for name, (path, entry) in recorded.items():
        _require(_sha(path) == entry["sha256"], "checkpoint artifact changed during read: " + name)
    return result, {"recorder_id": recorder_id, "experiment_id": experiment_id,
                    "recorder_status": "FAILED", "failure_manifest_sha256": manifest_sha256,
                    "artifacts": {name: recorded[name][1]["sha256"] for name in artifacts},
                    "hashes_checked_before_unpickling": True, "old_recorder_read_only": True}

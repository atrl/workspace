"""Verified read-only objects from trusted, card-local Qlib file Recorders.

Remote artifact stores are intentionally unsupported. Hashes establish the
retained local artifact identity; this is not an untrusted-pickle sandbox.
The caller must initialize Qlib before using this module.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlparse


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _sha(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError("blocked: source artifact cannot be read") from error
    return digest.hexdigest()


def _relative(value, name):
    _require(isinstance(value, str) and bool(value) and value == value.strip()
             and "\\" not in value and ":" not in value and "\x00" not in value,
             f"{name} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and all(part not in ("", ".", "..") for part in value.split("/"))
             and str(path) == value, f"{name} cannot be absolute or contain traversal/empty components")
    return path


def _local(value, name, *, allow_file_uri=False):
    _require(isinstance(value, str) and bool(value), f"missing {name}")
    parsed = urlparse(value)
    if parsed.scheme:
        _require(allow_file_uri and parsed.scheme == "file" and parsed.netloc in ("", "localhost")
                 and not parsed.query and not parsed.fragment, f"{name}: remote stores and URI-form metadata are unsupported")
        value = unquote(parsed.path)
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts, f"{name} must be an absolute local path")
    return path.resolve()


def _json_object(raw, name):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate metadata key in {name}")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"blocked: invalid {name} JSON") from error
    _require(isinstance(value, dict), f"{name} must contain a JSON object")
    return value


def _digest(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{name} requires a lowercase SHA-256 digest")
    return value


def load_verified_recorder_objects(card_dir, *, expected_recorder_id, artifacts: list[str],
                                   expected_artifact_sha256: dict[str, str] | None = None):
    """Load exact named objects after all requested bytes pass identity checks.

    Metadata uses absolute path strings, never ``file://`` or relative tracking
    URIs. Manifest entries may be card-relative or repository-relative paths
    ending in the exact card/local-Recorder artifact path. If optional expected
    digests are supplied, their keys must match the complete requested inventory.
    No source files or Recorder state are written, started, resumed or initialized.
    """
    card = Path(card_dir).resolve()
    _require(card.is_dir(), "source card directory is missing")
    rid = _relative(expected_recorder_id, "expected_recorder_id")
    _require(len(rid.parts) == 1, "expected_recorder_id must be one opaque component")
    _require(isinstance(artifacts, list) and len(artifacts) > 0
             and all(isinstance(name, str) for name in artifacts) and len(set(artifacts)) == len(artifacts),
             "artifacts must be a nonempty unique list")
    requested = {name: _relative(name, "artifact") for name in artifacts}
    if expected_artifact_sha256 is not None:
        _require(isinstance(expected_artifact_sha256, Mapping)
                 and set(expected_artifact_sha256) == set(requested), "expected digest inventory differs from requested artifacts")
        for name, digest in expected_artifact_sha256.items():
            _digest(digest, name)
    summary_path, manifest_path = card / "summary.json", card / "verification_manifest.json"
    _require(summary_path.is_file() and manifest_path.is_file(), "source summary or verification manifest is missing")
    _require(not summary_path.is_symlink() and not manifest_path.is_symlink(), "symbolic-link source metadata is unsupported")
    summary_bytes, manifest_bytes = summary_path.read_bytes(), manifest_path.read_bytes()
    summary = _json_object(summary_bytes, "summary")
    manifest = _json_object(manifest_bytes, "verification manifest")
    _require(summary.get("recorder_id") == manifest.get("recorder_id") == expected_recorder_id,
             "summary/manifest/expected Recorder identities differ")
    experiment_name = summary.get("experiment")
    _require(isinstance(experiment_name, str) and bool(experiment_name.strip()), "summary experiment name is missing")
    uri = _local(summary.get("recorder_uri"), "summary recorder_uri")
    local = _local(summary.get("recorder_local_dir"), "summary recorder_local_dir")
    _require(uri.is_dir() and uri.is_relative_to(card) and uri != card,
             "tracking filestore must already exist inside the source card")
    _require(local.is_dir() and local.is_relative_to(uri) and local.name == expected_recorder_id,
             "source Recorder directory must be inside its card-local filestore")
    artifact_root = (local / "artifacts").resolve()
    _require(artifact_root.is_dir() and artifact_root.is_relative_to(local), "artifact directory escapes the local Recorder")
    entries = manifest.get("entries")
    _require(isinstance(entries, list) and all(isinstance(item, dict) and "path" in item for item in entries),
             "verification manifest entries schema is invalid")
    parsed_entries = [(item, _relative(item["path"], "manifest entry")) for item in entries]
    identities = {}
    for name, relative in requested.items():
        lexical = local / "artifacts" / Path(relative)
        actual = lexical.resolve()
        _require(actual.is_file() and actual.is_relative_to(artifact_root), "artifact is missing or escapes the Recorder")
        _require(not any(part.is_symlink() for part in [lexical, *lexical.parents] if part.is_relative_to(local)),
                 "symbolic-link artifacts are unsupported")
        card_relative = PurePosixPath(lexical.relative_to(card).as_posix())
        suffix = (card.name, *card_relative.parts)
        matches = [item for item, entry in parsed_entries if entry == card_relative
                   or (len(entry.parts) >= len(suffix) and entry.parts[-len(suffix):] == suffix)]
        _require(len(matches) == 1, f"artifact must have exactly one matching manifest entry: {name}")
        registered = matches[0]
        digest = _digest(registered.get("sha256"), name)
        _require(_sha(actual) == digest, f"artifact SHA-256 differs before loading: {name}")
        if "bytes" in registered:
            _require(type(registered["bytes"]) is int and registered["bytes"] == actual.stat().st_size,
                     f"artifact byte count differs: {name}")
        if expected_artifact_sha256 is not None:
            _require(expected_artifact_sha256[name] == digest, f"explicit artifact digest differs: {name}")
        identities[name] = {"artifact": name, "artifact_id": f"{expected_recorder_id}:{name}",
            "sha256": digest, "bytes": actual.stat().st_size, "manifest_entry": registered["path"],
            "local_path": str(actual)}

    from qlib.workflow import R

    try:
        active = R.exp_manager.active_experiment
        original_uri = R.get_uri()
    except AttributeError as error:
        raise ValueError("blocked: Qlib must already be initialized for source reads") from error
    _require(active is None or active.active_recorder is None, "source reads require no active writable Recorder")
    objects = {}
    with R.uri_context(str(uri)):
        recorder = R.get_recorder(experiment_name=experiment_name, recorder_id=expected_recorder_id)
        _require(recorder.id == expected_recorder_id, "Qlib returned a different Recorder identity")
        _require(_local(recorder.uri, "returned Recorder URI", allow_file_uri=True) == uri,
                 "Qlib Recorder tracking URI differs from summary")
        _require(_local(recorder.artifact_uri, "returned artifact URI", allow_file_uri=True) == artifact_root,
                 "Qlib artifact URI differs from card-local artifacts")
        _require(_local(recorder.get_local_dir(), "returned Recorder local directory") == local,
                 "Qlib Recorder directory differs from summary")
        experiment = recorder.client.get_experiment(recorder.experiment_id)
        _require(experiment.name == experiment_name, "Qlib experiment name differs from summary")
        if "experiment_id" in summary:
            _require(str(summary["experiment_id"]) == str(recorder.experiment_id), "Qlib experiment ID differs from summary")
        run = recorder.client.get_run(recorder.id)
        _require(run.info.status == "FINISHED" and str(run.info.experiment_id) == str(recorder.experiment_id)
                 and run.info.run_id == expected_recorder_id, "source Recorder is unfinished or belongs to a different experiment")
        _require(_local(run.info.artifact_uri, "run artifact URI", allow_file_uri=True) == artifact_root,
                 "source run artifact location differs")
        # All requested digests were checked above; recheck immediately before
        # each public load as well, avoiding partial loading on an initial mismatch.
        for name, item in identities.items():
            _require(_sha(item["local_path"]) == item["sha256"], f"artifact changed before unpickling: {name}")
            objects[name] = recorder.load_object(name)
        for name, item in identities.items():
            _require(_sha(item["local_path"]) == item["sha256"], f"artifact changed during source read: {name}")
        _require(summary_path.read_bytes() == summary_bytes and manifest_path.read_bytes() == manifest_bytes,
                 "summary or verification manifest changed during source read")
        _require(recorder.client.get_run(recorder.id).info.status == "FINISHED", "source Recorder status changed during read")
    _require(R.get_uri() == original_uri and R.exp_manager.active_experiment is active,
             "Qlib source read did not restore its original context")
    audit = {"recorder_id": expected_recorder_id, "experiment": experiment_name,
        "experiment_id": str(recorder.experiment_id), "recorder_uri": str(uri), "recorder_local_dir": str(local),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "verification_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "artifacts": identities,
        "read_only": True, "local_filestore_only": True, "hashes_checked_before_unpickling": True,
        "metadata_and_artifact_hashes_rechecked": True,
        "source_api": ["qlib.workflow.R.get_recorder", "Recorder.load_object"]}
    return objects, audit

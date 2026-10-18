"""Three separately approved, immutable Qlib source/label qualification phases."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.day_night_labels import (CLOCKS, TARGETS, SYMBOLS, assert_reference_unchanged,
    basis_certificate, build_h5_mapping, common_log_plane, construct_labels,
    h1_vintage_diagnostic, original_target_digest, require, validate_prices)
from src.data.minute_intake import atomic_json

NAME = "day_night_label_source_20261018"
CODE_FILES = ["src/data/day_night_labels.py", "src/workflow/day_night_label_source.py",
    "src/core/recorder_sources.py", "src/data/minute_intake.py", "src/data/nas_market_data.py",
    "tests/test_day_night_label_source.py"]
PERMISSIONS = {
    "A": {"opaque_price_blob_materializations": 4, "D_calendar_calls": 0, "D_features_calls": 0,
          "PG_training_reference_loads": 0, "bounded_B_reference_loads": 0, "S_Z_rows": 0},
    "B": {"opaque_price_blob_materializations": 0, "D_calendar_calls": 1, "D_features_calls": 1,
          "PG_training_reference_loads": 1, "bounded_B_reference_loads": 0, "S_Z_rows": 0},
    "C": {"opaque_price_blob_materializations": 0, "D_calendar_calls": 0, "D_features_calls": 0,
          "PG_training_reference_loads": 0, "bounded_B_reference_loads": 1, "S_Z_rows": 3014},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity(root):
    result = {name: sha(Path(root) / name) for name in CODE_FILES}
    import qlib
    installed = Path(qlib.__file__).parent
    for relative in ("data/data.py", "data/storage/file_storage.py", "data/dataset/handler.py", "utils/serial.py"):
        result["installed_qlib/" + relative] = sha(installed / relative)
    return result


def native_dataset(frame):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments={"all": (None, None)})
    dataset.config(dump_all=True, recursive=True)
    pd.testing.assert_frame_equal(prepare_all_groups(pickle.loads(pickle.dumps(dataset))), frame)
    return dataset


def prepare_all_groups(dataset, segment="all"):
    from qlib.data.dataset.handler import DataHandler
    frame = dataset.prepare(segment, col_set=DataHandler.CS_RAW)
    require(isinstance(frame.columns, pd.MultiIndex) and frame.columns.nlevels == 2,
            "native full-group Dataset columns lost")
    return frame


def _init(provider):
    import qlib
    qlib.init(provider_uri=str(provider), region="us", kernels=1,
              expression_cache=None, dataset_cache=None, default_disk_cache=0)


@contextmanager
def raw_slice_guard(provider, calendar, start, end):
    """Record and constrain actual raw .bin slices before Qlib decodes values."""
    from qlib.data.storage.file_storage import FileFeatureStorage
    original = FileFeatureStorage.__getitem__
    provider = Path(provider).resolve()
    first, last = calendar.get_indexer([pd.Timestamp(start), pd.Timestamp(end)])
    require(first >= 0 and last >= first, "exact bounded-query endpoints absent from native calendar")
    allowed = {str((provider / "features" / symbol.lower() / (field + ".day.bin")).resolve())
               for symbol in SYMBOLS for field in ("open", "close")}
    reads = []

    def bounded(storage, item):
        path = str(storage.uri.resolve())
        require(path in allowed, "unapproved price field/provider requested")
        require(isinstance(item, slice) and item.step is None and item.start is not None and item.stop is not None,
                "unbounded raw price read is forbidden")
        require(first <= item.start <= item.stop - 1 <= last, "raw price slice expands outside approved interval")
        require(len(reads) < 4 and path not in {row["path"] for row in reads}, "duplicate/extra raw price slice")
        reads.append({"path": path, "start_index": int(item.start), "end_index": int(item.stop - 1),
                      "start_date": str(calendar[item.start].date()), "end_date": str(calendar[item.stop - 1].date())})
        return original(storage, item)

    FileFeatureStorage.__getitem__ = bounded
    try:
        yield reads
    finally:
        FileFeatureStorage.__getitem__ = original


def bounded_native_read(provider, *, start, end, before_call=None):
    """One public calendar metadata call and one raw-field D.features call."""
    from qlib.data import D
    if before_call:
        before_call("D_calendar_calls")
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    require(calendar.is_unique and calendar.is_monotonic_increasing and calendar.tz is None,
            "invalid complete native calendar metadata")
    with raw_slice_guard(provider, calendar, start, end) as reads:
        if before_call:
            before_call("D_features_calls")
        prices = D.features(list(SYMBOLS), ["$open", "$close"], start_time=start, end_time=end,
                            freq="day", disk_cache=0)
    require(len(reads) == 4, "one native slice per selected field was not observed")
    require(set(prices.index.names) == {"datetime", "instrument"}, "unexpected native price index")
    prices = prices.reorder_levels(["datetime", "instrument"]).sort_index()
    dates = prices.index.get_level_values("datetime")
    require(dates.min() >= pd.Timestamp(start) and dates.max() <= pd.Timestamp(end), "native output exceeds price bounds")
    require(all(dtype == np.dtype("float32") for dtype in prices.dtypes), "raw price dtype differs")
    return calendar, prices, {"public_D_calendar_calls": 1, "public_D_features_calls": 1,
        "raw_field_slices": reads, "calendar_metadata_may_extend_past_price_bound": True,
        "expression_cache": None, "dataset_cache": None, "disk_cache": 0, "kernels": 1,
        "price_decode_start": start, "price_decode_end": end}


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def _capture_blob(root, commit, path, expected_oid=None, expected_sha=None):
    oid = _git(root, "rev-parse", commit + ":" + path).decode().strip()
    if expected_oid:
        require(oid == expected_oid, "selected Git membership differs: " + path)
    body = _git(root, "cat-file", "blob", oid)
    object_hash = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    digest = hashlib.sha256(body).hexdigest()
    require(oid == object_hash, "Git object-content encoding differs")
    if expected_sha:
        require(digest == expected_sha, "selected raw bytes SHA differs: " + path)
    return body, {"git_commit": commit, "git_path": path, "git_blob_oid": oid,
                  "raw_byte_sha256": digest, "bytes": len(body)}


def _phase_manifest(card, phase):
    subcard = card / ("phase_" + phase)
    summary = json.loads((subcard / "summary.json").read_text())
    manifest = json.loads((subcard / "verification_manifest.json").read_text())
    require(summary["status"] == "completed_pending_independent_review", "preceding phase incomplete")
    return {"phase": phase, "recorder_id": summary["recorder_id"],
            "summary_sha256": sha(subcard / "summary.json"),
            "manifest_sha256": sha(subcard / "verification_manifest.json"),
            "archive_sha256": json.loads((subcard / "nas_binding.json").read_text())["evidence"]["sha256"]}, summary, manifest


def _approval(root, card, plan, phase):
    require(plan.get("frozen") is True, "source plan is not frozen")
    require(os.environ.get("DAY_NIGHT_SOURCE_EXECUTE") == "1", "real source execution is disabled")
    require(os.environ.get("DAY_NIGHT_SOURCE_PHASE") == phase, "phase environment differs")
    parent = int(os.environ.get("DAY_NIGHT_SOURCE_PARENT_PID", "0"))
    require(parent > 0 and os.getpid() != parent and os.getppid() == parent,
            "real phase must be a fresh notebook child process")
    identity_path = card / "implementation_identity.json"
    identity = json.loads(identity_path.read_text()); plan_hash = sha(card / "plan.json")
    current = code_identity(root)
    require(identity["plan_sha256"] == plan_hash and identity["code_sha256"] == current, "implementation identity changed")
    review_path = card / "implementation_review.json"
    review = json.loads(review_path.read_text())
    require(review.get("verdict") == "APPROVE_SOURCE_IMPLEMENTATION" and review.get("plan_sha256") == plan_hash
            and review.get("code_sha256") == current and phase in review.get("phases", []), "implementation review stale/missing")
    path = card / ("phase_" + phase + "_execution_approval.json")
    owner = json.loads(path.read_text())
    require(owner.get("approved") is True and owner.get("phase") == phase and owner.get("plan_sha256") == plan_hash,
            "phase owner approval missing/stale")
    require(owner.get("implementation_identity_sha256") == sha(identity_path)
            and owner.get("implementation_review_sha256") == sha(review_path), "phase implementation approval differs")
    expected = {**PERMISSIONS[phase], "new_source_HTTP": 0, "new_fits": 0, "new_accounts": 0}
    require(owner.get("maximum_counts") == expected, "phase permissions/budget differ")
    upstream = None
    if phase != "A":
        previous = "A" if phase == "B" else "B"
        upstream, _, _ = _phase_manifest(card, previous)
        result_path = card / ("phase_" + previous) / "source_result_review.json"
        result = json.loads(result_path.read_text())
        verdict = "ACCEPT_PHASE_A_IDENTITY_AND_BASIS" if phase == "B" else "ACCEPT_PHASE_B_BOUNDED_PRICE_AND_ALIGNMENT"
        require(result.get("verdict") == verdict and result.get("next_phase_eligible") is True
                and all(result.get(key) == value for key, value in upstream.items()), "preceding source result unaccepted/stale")
        upstream["review_sha256"] = sha(result_path)
        require(owner.get("upstream") == upstream, "owner approval does not bind accepted preceding phase")
    origin = plan["design_origin"]
    require(sha(root / origin["independent_review"]) == origin["independent_review_sha256"], "source design review changed")
    return {"plan_sha256": plan_hash, "implementation_identity_sha256": sha(identity_path),
            "implementation_review_sha256": sha(review_path), "owner_approval_sha256": sha(path),
            "code_sha256": current, "upstream": upstream}


def _read_phase(card, phase, names):
    subcard = card / ("phase_" + phase)
    identity, summary, manifest = _phase_manifest(card, phase)
    base = Path(summary["recorder_local_dir"]) / "artifacts"
    digests = {}
    for name in names:
        path = str((base / name).relative_to(subcard))
        rows = [item for item in manifest["entries"] if item["path"] == path]
        require(len(rows) == 1, "preceding phase object absent from exact manifest")
        digests[name] = rows[0]["sha256"]
    objects, audit = load_verified_recorder_objects(subcard, expected_recorder_id=identity["recorder_id"],
        artifacts=names, expected_manifest_sha256=identity["manifest_sha256"], expected_artifact_sha256=digests)
    return objects, audit


def _reference(root, plan):
    source = plan["phase_B_native_training_price_and_alignment"]["train_reference"]
    card = root / source["card"]
    require(json.loads((card / "nas_binding.json").read_text())["evidence"]["sha256"] == source["archive"],
            "original training archive identity differs")
    objects, audit = load_verified_recorder_objects(card, expected_recorder_id=source["recorder_id"],
        artifacts=source["artifacts"], expected_manifest_sha256=source["inventory_sha256"],
        expected_artifact_sha256=source["artifact_sha256"])
    dataset = objects["inputs/training_dataset"]
    require(set(dataset.segments) == {"train"}, "only the already accepted train-only Dataset may be loaded")
    frame = prepare_all_groups(dataset, "train")
    require(frame.shape == (3014, 85), "original train-only source shape differs")
    dates = frame.index.get_level_values("datetime")
    require(dates.max() <= pd.Timestamp("2022-12-21"), "unexpected reference signal range")
    # The whole admitted 85-column train-only object is resident. Selection here
    # does not pretend to prevent that; no eval or full-period source is loaded.
    reference = pd.concat([frame["meta"].loc[:, list(CLOCKS)], frame["label"].loc[:, list(TARGETS)]], axis=1).copy()
    reference.index = pd.DatetimeIndex(dates, name="signal_date")
    require(pd.to_datetime(reference.label_available_at).lt(pd.Timestamp("2023-01-01", tz="America/New_York")).all(),
            "original H5 cohort maturity differs")
    require(objects["inputs/training_audit"]["primary_rows"] == 3014, "sealed original training audit differs")
    return reference, {"source_audit": audit, "original_shape": [3014, 85],
                       "accepted_train_residency_reused": True, "whole_train_object_resident": True,
                       "full_period_or_evaluation_objects_loaded": 0}


def _phase_a(root, card, plan, out, state, count):
    selected = plan["source_choice_before_any_price_read"]["selected_alternative"]
    commit = selected["git_commit"]
    require(_git(root, "rev-parse", commit + "^{commit}").decode().strip() == commit, "selected commit differs")
    provider = root / "data" / "qlib" / NAME / ("selected_" + commit)
    require(not provider.exists(), "isolated source snapshot already exists; no implicit retry")
    provider.mkdir(parents=True)
    members = []; raw_bytes = {}
    for item in selected["selected_provider_members"]:
        relative = item["relative_path"]
        if relative.endswith(".bin"):
            count("opaque_price_blob_materializations")
        body, identity = _capture_blob(root, commit, selected["provider_tree_prefix"] + "/" + relative,
                                      expected_oid=item["expected_git_blob_oid"])
        destination = provider / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body); destination.chmod(0o444)
        require(sha(destination) == identity["raw_byte_sha256"], "opaque materialization bytes differ")
        members.append({**identity, "relative_path": relative, "materialized_path": str(destination),
                        "opaque_price_blob": relative.endswith(".bin")})
        raw_bytes[relative.replace("/", "__")] = body
    require(len(members) == 7 and state["opaque_price_blob_materializations"] == 4, "materialization inventory differs")
    require(sha(provider / "qlib_store_manifest.json") == selected["provider_manifest_sha256"], "provider manifest differs")
    raw_spec = selected["raw_source_manifest"]
    raw_body, raw_identity = _capture_blob(root, commit, raw_spec["git_path"].split(":", 1)[1],
        expected_oid=raw_spec["git_blob_oid"], expected_sha=raw_spec["sha256"])
    code = {}; code_audit = []
    for key in ("same_commit_builder", "same_commit_storage_builder"):
        spec = selected[key]; path = spec.get("path") or spec["git_path"].split(":", 1)[1]
        body, identity = _capture_blob(root, commit, path, expected_sha=spec["sha256"])
        code[path] = body.decode(); code_audit.append(identity)
    certificate = basis_certificate(json.loads(raw_body), plan["phase_A_content_identity_and_common_basis"]["fixed_metadata_coefficients"])
    certificate.update(raw_source_identity=raw_identity, code_identities=code_audit,
                       provider_manifest_sha256=selected["provider_manifest_sha256"],
                       selected_git_commit=commit, numerical_price_decoding=False,
                       independent_basis_acceptance_required_before_phase_B=True)
    for directory in sorted([p for p in provider.rglob("*") if p.is_dir()], reverse=True):
        directory.chmod(0o555)
    provider.chmod(0o555)
    _init(provider)
    artifacts = {"identity": {"selected_member_manifest": members},
        "source": {"opaque_selected_members": raw_bytes, "basis_certificate": certificate,
                   "immutable_metadata_and_code": {"raw_source_manifest": raw_body, "same_commit_code": code}},
        "analysis": {"source_identity_checks": {"git_oid_encoding_checked": True, "raw_sha256_checked": True,
                    "partial_provider": True, "original_provider_modified": False}}}
    summary = {"provider_uri": str(provider), "selected_members": 7, "selected_price_fields": 4,
               "price_values_decoded": 0, "train_label_objects_loaded": 0,
               "opaque_access_scope": "Entire selected source-member bytes captured; no numerical price decoding.",
               "partial_provider_disclosed": True, "basis_review_pending": True,
               "decision": "identity_and_declared_basis_complete_pending_independent_review"}
    return artifacts, summary


def _verify_snapshot(members):
    for item in members:
        path = Path(item["materialized_path"])
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["raw_byte_sha256"],
                "isolated source member changed")
        require(path.stat().st_mode & 0o222 == 0, "source member is not read-only")


def _phase_b(root, card, plan, out, state, count):
    prior = json.loads((card / "phase_A/summary.json").read_text())
    provider = Path(prior["provider_uri"]); _init(provider)
    admitted, source_audit = _read_phase(card, "A", ["identity/selected_member_manifest", "source/basis_certificate"])
    _verify_snapshot(admitted["identity/selected_member_manifest"])
    count("PG_training_reference_loads")
    reference, reference_audit = _reference(root, plan)
    original_hash = original_target_digest(reference)
    query = plan["phase_B_native_training_price_and_alignment"]["native_query"]
    calendar, prices, read_audit = bounded_native_read(provider, start=query["start_time"], end=query["end_time"], before_call=count)
    mapping = build_h5_mapping(calendar, reference)
    primitives = validate_prices(prices, mapping)
    paired, diagnostic = h1_vintage_diagnostic(prices, mapping, reference)
    assert_reference_unchanged(reference, original_hash)
    dataset = native_dataset(pd.concat({"price": prices}, axis=1))
    _verify_snapshot(admitted["identity/selected_member_manifest"])
    for item in reference_audit["source_audit"]["artifacts"].values():
        require(sha(item["local_path"]) == item["sha256"], "original training artifact changed after qualification")
    read_audit["opaque_snapshot_SHA_checks_before_and_after"] = True
    read_audit["opaque_hashing_semantics"] = "Full snapshot bytes are hashed without decoding before/after; only guarded D.features slices decode prices."
    artifacts = {"source": {"native_training_prices": dataset, "complete_calendar_metadata": calendar,
            "basis_certificate": admitted["source/basis_certificate"], "phase_A_source_audit": source_audit},
        "reference": {"original_train_index_clocks_and_targets": reference, "h5_mapping": mapping,
                      "reference_audit": reference_audit},
        "analysis": {"bounded_read_audit": read_audit, "primitive_checks": primitives,
                     "H1_cross_vintage_paired": paired, "H1_cross_vintage_diagnostic": diagnostic}}
    summary = {"provider_uri": str(provider), "bounded_native_price_rows": len(prices), "original_cohort_rows": len(reference),
        "original_target_sha256": original_hash, "original_targets_unchanged": True,
        "original_training_dataset_artifact_sha256": plan["phase_B_native_training_price_and_alignment"]["train_reference"]["artifact_sha256"]["inputs/training_dataset"],
        "H1_cross_vintage_diagnostic": diagnostic, "H1_diagnostic_has_numerical_gate": False,
        "mapping_rows": len(mapping), "last_maturity": str(reference.label_available_at.max()),
        "price_start": query["start_time"], "price_end": query["end_time"],
        "native_dataset_direct_prepare_roundtrip": "passed", "decision": "bounded_price_and_alignment_complete_pending_review"}
    return artifacts, summary


def _phase_c(root, card, plan, out, state, count):
    prior = json.loads((card / "phase_B/summary.json").read_text())
    _init(prior["provider_uri"])
    admitted, source_audit = _read_phase(card, "B", ["source/native_training_prices", "source/basis_certificate",
        "reference/original_train_index_clocks_and_targets", "reference/h5_mapping"])
    count("bounded_B_reference_loads")
    prices = prepare_all_groups(admitted["source/native_training_prices"])["price"]
    reference = admitted["reference/original_train_index_clocks_and_targets"]
    mapping = admitted["reference/h5_mapping"]
    original_hash = prior["original_target_sha256"]
    assert_reference_unchanged(reference, original_hash)
    require(mapping.index.equals(reference.index) and len(mapping) == 3014, "sealed B mapping/cohort differs")
    validate_prices(prices, mapping)
    log_plane = common_log_plane(prices, admitted["source/basis_certificate"])
    count("S_Z_rows", len(mapping))
    labels, checks = construct_labels(log_plane, mapping)
    assert_reference_unchanged(reference, original_hash)
    # Original five targets/clocks are copied exactly; S/Z get a separate group.
    original = reference.copy(); original.index = pd.MultiIndex.from_arrays([original.index, ["SOXL_SOXX"] * len(original)], names=["datetime", "instrument"])
    derived = labels.copy(); derived.index = original.index
    frame = pd.concat({"original": original, "fixed_day_night_label": derived}, axis=1)
    dataset = native_dataset(frame)
    artifacts = {"source": {"phase_B_source_audit": source_audit, "cached_common_log_plane": log_plane,
                            "basis_certificate": admitted["source/basis_certificate"]},
        "reference": {"original_train_index_clocks_and_targets": reference, "h5_mapping": mapping},
        "labels": {"fixed_H5_components_S_Z": dataset}, "analysis": {"source_label_checks": checks}}
    summary = {"provider_uri": prior["provider_uri"], "original_cohort_rows": len(reference), "fixed_label_rows": len(labels),
        "original_target_sha256": original_hash, "original_targets_unchanged": True,
        "algebra_identities_checked": len(checks), "algebra_all_passed": bool(checks.passed.all()),
        "max_absolute_residual": float(checks.residual.abs().max()), "all_bounds_finite": bool(np.isfinite(checks.bound).all()),
        "native_dataset_direct_prepare_roundtrip": "passed", "binary64_environment": checks.attrs["binary64_environment"],
        "decision": "conditional_training_label_source_qualified_not_economic_success",
        "label_plane": "distribution_adjusted_research_price_plane", "S_Z_predictive_usefulness_assessed": False}
    return artifacts, summary


def _seal(root, card, out, rec, summary, plan):
    from src.data.nas_market_data import archive_nas_recorder
    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts"
    atomic_json({"recorder_id": rec.id, "entries": [{"path": str(path.relative_to(out)), "sha256": sha(path),
        "bytes": path.stat().st_size} for path in sorted(artifact_root.rglob("*")) if path.is_file()]}, out / "verification_manifest.json")
    source = plan["phase_B_native_training_price_and_alignment"]["train_reference"]
    transport = json.loads((root / source["card"] / "nas_binding.json").read_text())
    atomic_json({"schema_version": "recorder_evidence_binding_v1", "authority": "yanxuan_nas", "ssh": transport["ssh"],
        "source_id": NAME + "-" + summary["phase"] + "-" + summary["plan_sha256"],
        "as_of": summary["source_snapshot_as_of"], "source_snapshot_as_of": summary["source_snapshot_as_of"],
        "training_observation_end": summary["training_observation_end"],
        "source_plan": "../plan.json"}, out / "nas_binding.json")
    summary["nas_evidence"] = archive_nas_recorder(out / "nas_binding.json", workspace_root=root, card=out,
        recorder_id=rec.id, experiment_id=str(rec.experiment_id))
    atomic_json(summary, out / "summary.json")


def run_phase(*, root, card, phase):
    """Worker entry only. Each accepted phase has its own immutable attempt/RID."""
    from qlib.workflow import R
    root, card = Path(root).resolve(), Path(card).resolve()
    plan = json.loads((card / "plan.json").read_text())
    approval = _approval(root, card, plan, phase)
    out = card / ("phase_" + phase); out.mkdir(exist_ok=True)
    path = out / "run_state.json"
    require(not path.exists(), "phase attempt exists; no implicit retry is authorized")
    state = {"phase": phase, "status": "started", "pid": os.getpid(), "parent_pid": os.getppid(),
             **{key: 0 for key in PERMISSIONS[phase]}, "new_source_HTTP": 0, "new_fits": 0, "new_accounts": 0,
             "approval": approval}
    atomic_json(state, path)
    def count(key, amount=1):
        require(state[key] + amount <= PERMISSIONS[phase][key], "phase operation budget exceeded: " + key)
        state[key] += amount; atomic_json(state, path)
    try:
        # All qlib.init/source loading is before the new writable R.start.
        artifacts, details = {"A": _phase_a, "B": _phase_b, "C": _phase_c}[phase](root, card, plan, out, state, count)
        tracking = out / "qlib_recorder"
        with R.start(experiment_name=NAME + "_phase_" + phase, uri=str(tracking)):
            rec = R.get_recorder(); state.update(recorder_id=rec.id, experiment_id=str(rec.experiment_id)); atomic_json(state, path)
            rec.save_objects(artifact_path="contract", plan=plan, approval=approval)
            for folder, objects in artifacts.items():
                rec.save_objects(artifact_path=folder, **objects)
            rec.save_objects(artifact_path="implementation", code={name: (root / name).read_text() for name in CODE_FILES})
            summary = {"experiment": NAME + "_phase_" + phase, "phase": phase,
                "status": "completed_pending_independent_review", "plan_sha256": approval["plan_sha256"],
                "recorder_id": rec.id, "experiment_id": str(rec.experiment_id), "recorder_uri": str(tracking),
                "recorder_local_dir": str(tracking / str(rec.experiment_id) / rec.id),
                "region": "us", "as_of": plan["source_choice_before_any_price_read"]["selected_alternative"]["declared_coverage"][-1],
                "source_snapshot_as_of": plan["source_choice_before_any_price_read"]["selected_alternative"]["declared_coverage"][-1],
                "training_observation_end": plan["phase_B_native_training_price_and_alignment"]["native_query"]["end_time"], **details,
                "operation_counts": {key: state[key] for key in PERMISSIONS[phase]},
                "new_source_HTTP": 0, "new_fits": 0, "new_accounts": 0,
                "historical_PIT_qualified": False, "independent_forward_qualified": False,
                "promotion": "not_assessed_source_only", "qlib_docs": plan["qlib_authority"],
                "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "../workflow.yaml",
                    "recorder_only_artifacts": ["opaque_selected_members", "native_training_prices", "original_train_index_clocks_and_targets",
                                                "H1_cross_vintage_paired", "cached_common_log_plane", "fixed_H5_components_S_Z"]}}
            # Direct prepare is the supported self-contained readback, never setup_data.
            if phase == "B":
                pd.testing.assert_frame_equal(prepare_all_groups(rec.load_object("source/native_training_prices")),
                                              prepare_all_groups(artifacts["source"]["native_training_prices"]))
            elif phase == "C":
                pd.testing.assert_frame_equal(prepare_all_groups(rec.load_object("labels/fixed_H5_components_S_Z")),
                                              prepare_all_groups(artifacts["labels"]["fixed_H5_components_S_Z"]))
            rec.save_objects(artifact_path="analysis", summary=summary)
            local = Path(summary["recorder_local_dir"]) / "artifacts"
            rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in local.rglob("*")),
                            new_fits=0, new_accounts=0, source_HTTP_requests=0)
        _seal(root, card, out, rec, summary, plan)
        state["status"] = "completed"; atomic_json(state, path)
        (out / "conclusion.md").write_text(_conclusion(summary))
        # Root entry points to current phase, without changing sealed preceding phases.
        atomic_json({"experiment": NAME, "latest_phase": phase, "phase_summary": "phase_" + phase + "/summary.json",
                     "status": summary["status"], "decision": summary["decision"], "recorder_id": rec.id,
                     "recorder_uri": summary["recorder_uri"], "recorder_local_dir": summary["recorder_local_dir"],
                     "provider_uri": summary["provider_uri"], "plan_sha256": summary["plan_sha256"],
                     "as_of": summary["source_snapshot_as_of"], "source_snapshot_as_of": summary["source_snapshot_as_of"],
                     "training_observation_end": summary["training_observation_end"],
                     "artifact_policy": {**summary["artifact_policy"], "workflow_config": "workflow.yaml"},
                     "new_fits": 0, "new_accounts": 0, "historical_PIT_qualified": False}, card / "summary.json")
        (card / "conclusion.md").write_text(_conclusion(summary))
        return summary
    except BaseException as error:
        state.update(status="failed", exception_type=type(error).__name__, exception=str(error))
        atomic_json(state, path)
        raise


def _conclusion(summary):
    return (f"# H5 日夜标签来源：阶段 {summary['phase']}\n\n"
        f"本阶段 `{summary['decision']}`；Recorder `{summary['recorder_id']}`，仍待独立结果验收。"
        "0 新行情 HTTP、0 模型拟合、0 账户。\n\n"
        "原 3014 行 H1/H5 目标不更改；新来源为独立的 2026-07-24 冻结版本，早段 close 仅按元数据映射到 open 平面。"
        "S/Z 属于分配调整后的研究价格平面，日内调整因子抵消，跨日因子变化进入 N；并非原始现金股利账本。\n\n"
        "本轮没有预测能力、市场分窗或经济晋升结论，历史 PIT/独立前向资格仍为 false。"
        "各阶段价格、原目标对照和标签保存在 Recorder；根卡只保留身份与资格汇总。"
        "下一阶段须绑定本阶段独立验收与 owner GO；不换源、不调系数或代数容差。\n")


def launch_phase(*, root, card, phase):
    """Notebook uses a new interpreter; no source object exists in its parent."""
    require(phase in PERMISSIONS, "phase must be A, B or C")
    env = os.environ.copy(); env["DAY_NIGHT_SOURCE_PARENT_PID"] = str(os.getpid())
    process = subprocess.Popen([sys.executable, "-m", "src.workflow.day_night_label_source",
        "--root", str(Path(root).resolve()), "--card", str(Path(card).resolve()), "--phase", phase],
        cwd=str(root), env=env)
    print(json.dumps({"phase": phase, "worker_pid": process.pid, "notebook_pid": os.getpid()}), flush=True)
    code = process.wait()
    require(code == 0, "phase worker failed; preserve attempt and do not retry automatically")
    return json.loads((Path(card) / ("phase_" + phase) / "summary.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True); parser.add_argument("--card", required=True)
    parser.add_argument("--phase", required=True, choices=tuple(PERMISSIONS))
    args = parser.parse_args()
    output = run_phase(root=args.root, card=args.card, phase=args.phase)
    print(json.dumps({"phase": args.phase, "recorder_id": output["recorder_id"], "status": output["status"]}), flush=True)

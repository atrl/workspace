"""Hash-bound frozen inputs for risk-aligned training; no forecast refitting.

Call before opening a writable Recorder. Only the pinned parent's data and
forecast objects are loaded; allocation models/checkpoints are never requested.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.benchmark_anchor_utility import (
    build_anchor_dataset, fit_feature_scaler, prepare_training_rows, transform_features,
)
from src.data.conditional_variance import primary_frame_from_qualified_qlib
from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES, FORECAST_FEATURES, frame_sha256
from src.data.price_basis import FIELDS, INSTRUMENTS, _sha as canonical_hash, price_frame_sha256
from src.data.qlib_native import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame
from src.workflow.anchor_checkpoint_feasibility import _same_evidence, checked_parent_load


PARENT_ARTIFACTS = {
    "parent_source_hashes": "inputs/source_hashes.pkl",
    "parent_source_audit": "inputs/source_audit.pkl",
    "raw": "inputs/primary_frame.pkl",
    "prices": "inputs/execution_prices.pkl",
    "forecasts": "inputs/forecasts.pkl",
    "forecast_models": "models/forecast_models.pkl",
    "forecast_audit": "inputs/forecast_fit_audit.pkl",
    "parent_dev_folds": "inputs/development_fold_inputs.pkl",
}
REQUIRED_PARENT_HELPERS = {
    "src/data/conditional_variance.py", "src/data/benchmark_anchor_utility.py",
    "src/data/integrated_allocation.py", "src/data/qlib_native.py", "src/data/price_basis.py",
    "src/model/benchmark_anchor_utility.py", "src/workflow/conditional_variance.py",
}


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _require(condition, message):
    if not bool(condition):
        raise ValueError("blocked: " + message)


def _same_frame(left, right, label):
    _require(isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame), label + " is not a DataFrame")
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True, check_freq=False)
    except AssertionError as error:
        raise ValueError("blocked: " + label + " differs from the pinned parent") from error
    _require(frame_sha256(left) == frame_sha256(right), label + " frame hash differs")


def _parent_header(root, plan):
    parent = plan["parent_input_source"]
    card = root / "research" / parent["experiment"]
    manifest_path, summary_path = card / "artifact_manifest.json", card / "summary.json"
    _require(_file_hash(manifest_path) == parent["artifact_manifest_sha256"], "parent artifact manifest identity differs")
    manifest = json.loads(manifest_path.read_text())
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    expected_suffix = f"/{parent['recorder_id']}/artifacts/card/summary.json"
    snapshots = [entry for entry in manifest["entries"] if entry["path"].endswith(expected_suffix)]
    _require(len(snapshots) == 1 and hashlib.sha256(summary_bytes).hexdigest() == snapshots[0]["sha256"],
             "local parent summary differs from its archived card snapshot")
    _require(summary["recorder_id"] == parent["recorder_id"] and summary["experiment"] == parent["experiment"],
             "parent Recorder identity mismatch")
    for key in ("experiment", "recorder_id", "provider_uri", "as_of", "model_signal_start", "basis"):
        _require(summary["source"][key] == plan["source"][key], f"qualified source declaration differs: {key}")
    return card, manifest, summary, summary_bytes


def _check_parent_code(root, hashes, summary):
    _require(isinstance(hashes, dict) and REQUIRED_PARENT_HELPERS.issubset(hashes), "parent source hashes omit required helpers")
    _require(hashes == summary["source_sha256"], "parent source hash object differs from archived summary")
    for name, digest in hashes.items():
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts, "parent code path escapes project")
        _require(_file_hash(root / relative) == digest, f"immutable parent helper changed: {name}")


def _native_readback(root, plan, parent_source_audit):
    """Native D values and canonical/byte hashes retain their distinct meanings."""
    from qlib.data import D

    source = plan["source"]
    provider = root / source["provider_uri"]
    manifest_path = provider / "price_basis_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    marker_path = provider / "price_basis_status.json"
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    _require(hashlib.sha256(manifest_bytes).hexdigest() == parent_source_audit["provider_manifest_sha256"],
             "native provider manifest file-byte hash differs")
    _require(marker.get("status") == "qualified" and marker.get("manifest_sha256") == canonical_hash(manifest),
             "native provider lacks its matching qualified marker")
    _require(manifest["qualification"]["qualification_pass"] is True
             and manifest["qualification"]["as_of"] == source["as_of"], "native qualification/as_of differs")
    canonical_path = root / "research" / source["experiment"] / "summary.json"
    canonical_bytes = canonical_path.read_bytes()
    canonical = json.loads(canonical_bytes)
    _require(hashlib.sha256(canonical_bytes).hexdigest() == parent_source_audit["parent_summary_sha256"],
             "canonical price card differs from the parent's verified source")
    _require(canonical["recorder_id"] == source["recorder_id"] == parent_source_audit["parent_recorder_id"]
             and canonical["source_qualification_pass"] is True and canonical["basis"] == source["basis"],
             "canonical qualification/source identity differs")
    raw_ohlcv = load_qlib_feature_frame(QlibProviderConfig(provider_uri=provider), instruments=list(INSTRUMENTS),
        fields=["$" + field for field in FIELDS], start=source["history_start"], end=source["as_of"]).sort_index()
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    _require(calendar.strftime("%Y-%m-%d").tolist() == manifest["reference_calendar"]["dates"]
             and str(calendar[0].date()) == source["history_start"]
             and str(calendar[-1].date()) == source["as_of"], "complete native calendar changed")
    _require(sorted(D.list_instruments(D.instruments("all"), as_list=True)) == sorted(INSTRUMENTS),
             "native instrument universe changed")
    native_hash = price_frame_sha256(raw_ohlcv)
    _require(native_hash == parent_source_audit["native_frame_sha256"]
             == manifest["native_readback"]["frame_sha256"] == canonical["native_readback"]["frame_sha256"],
             "native OHLCV content differs from frozen source")
    _require(manifest_path.read_bytes() == manifest_bytes and marker_path.read_bytes() == marker_bytes
             and canonical_path.read_bytes() == canonical_bytes, "qualified metadata changed during native readback")
    return raw_ohlcv, calendar, {"native_frame_sha256": native_hash, "native_rows": len(raw_ohlcv),
        "calendar_sessions": len(calendar), "provider_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "provider_marker_canonical_sha256": marker["manifest_sha256"], "provider_uri": str(provider.resolve())}


def _check_forecasts(raw, forecasts, models, audit):
    _require(isinstance(forecasts, pd.DataFrame) and forecasts.index.equals(raw.index), "frozen forecasts have different dates")
    _require(set(forecasts.columns) == set(raw.columns) | {"forecast_model_available_at"}, "frozen forecast schema differs")
    common = [column for column in raw.columns if column not in FORECAST_FEATURES]
    _same_frame(raw.loc[:, common], forecasts.loc[:, common], "non-forecast primary columns")
    _require(isinstance(audit, pd.DataFrame) and not audit.model_year.duplicated().any(), "forecast audit has duplicate/missing years")
    _require(set(audit.model_year) == set(raw.index.year), "forecast audit year coverage differs")
    _require(audit.available.map(lambda x: isinstance(x, (bool, np.bool_))).all(), "forecast availability audit must be boolean")
    ready = set(audit.loc[audit.available, "model_year"].astype(int))
    _require(isinstance(models, dict) and set(models) == ready, "frozen forecast model coverage differs")
    for row in audit.itertuples():
        cutoff = pd.Timestamp(f"{int(row.model_year)}-01-01", tz="America/New_York")
        _require(pd.Timestamp(row.decision_at) == cutoff, "forecast audit cutoff differs")
        _require(bool(row.available) == (row.rows >= row.minimum_training_rows), "forecast minimum-row availability differs")
        _require(row.minimum_training_rows == 252 and row.ridge_alpha == 10., "frozen forecast protocol differs")
        if row.rows:
            available_at = pd.Timestamp(row.label_available_at_max)
            _require(available_at.tzinfo is not None and available_at < cutoff, "forecast training outcome not strictly prior")
        part = forecasts.loc[forecasts.index.year == int(row.model_year)]
        flags = pd.to_numeric(part.forecast_available, errors="coerce")
        _require(flags.eq(float(row.available)).all(), "OOF forecast availability differs from saved audit")
        if row.available:
            stamps = pd.to_datetime(part.forecast_model_available_at, utc=True)
            _require(stamps.eq(cutoff.tz_convert("UTC")).all(), "OOF model availability time differs")
            model = models[int(row.model_year)]
            _require(callable(getattr(model, "predict", None))
                     and getattr(model, "n_features_in_", None) == len(COMMON_PRIMARY_FEATURES), "saved forecast model interface differs")
        else:
            _require(part.forecast_model_available_at.isna().all(), "unavailable OOF model must not have a fit timestamp")
    # The frozen fit audit includes list-valued feature names.  Its artifact
    # bytes are separately manifest-verified; this JSON digest is descriptive
    # and must not use pandas' scalar-only hash_pandas_object path.
    audit_json = audit.to_json(orient="split", date_format="iso", date_unit="ns", double_precision=15)
    return {"forecast_frame_sha256": frame_sha256(forecasts),
            "forecast_audit_json_sha256": hashlib.sha256(audit_json.encode()).hexdigest(),
            "forecast_years": sorted(int(year) for year in ready), "forecast_refit_performed": False}


def _check_development_folds(dataset, parent_folds, plan):
    expected = {(variant["name"], int(year)) for variant in plan["variants"] for year in plan["training"]["development_signal_years"]}
    _require(set(parent_folds) == expected, "parent development fold coverage differs")
    details = []
    for year in plan["training"]["development_signal_years"]:
        rows, maturity = prepare_training_rows(dataset, year=year, min_rows=plan["training"]["minimum_rows"],
                                               train_start=plan["source"]["model_signal_start"])
        for variant in (item["name"] for item in plan["variants"]):
            parent = parent_folds[(variant, int(year))]
            _same_frame(rows, parent["rows"], "development mature rows")
            _require(_same_evidence(maturity, parent["audit"]), "development maturity audit differs")
            scaler = fit_feature_scaler(rows, maturity, variant)
            _require(_same_evidence(scaler, parent["scaler"]), "prior-only reconstructed scaler differs")
            transformed = transform_features(rows, scaler, variant)
            returns = rows[["soxl_return", "soxx_return"]].to_numpy(float)
            _require(np.array_equal(transformed["features"], parent["features"]), "standardized development matrix differs")
            _require(np.array_equal(returns, parent["returns"]), "development return matrix differs")
            invalid = variant == "price_vol" and (transformed["invalid_forecast_mask"].any() or not scaler["eligible"])
            _require(bool(parent["invalid_training"]) == bool(invalid)
                     and parent["invalid_training_dates"] == scaler["invalid_training_forecast_dates"], "development forecast failure classification differs")
            _require(_same_evidence(transformed["audit"], parent["transform_audit"]), "development transform audit differs")
            details.append({"variant": variant, "model_year": int(year), "rows": len(rows),
                "dates_sha256": maturity["eligible_dates_sha256"], "scaler_sha256": scaler["scaler_sha256"],
                "training_feature_sha256": _array_hash(transformed["features"]),
                "training_return_sha256": _array_hash(returns), "invalid_training": bool(invalid)})
    return details


def load_constrained_inputs(root, plan):
    """Read only frozen parent inputs and reconstruct their exact native contracts."""
    from qlib.workflow import R

    root = Path(root).resolve()
    card, manifest, parent_summary, summary_bytes = _parent_header(root, plan)
    source = plan["parent_input_source"]
    init_qlib_provider(QlibProviderConfig(provider_uri=root / plan["source"]["provider_uri"]))
    objects, artifacts, artifact_paths = {}, [], []
    with R.uri_context(source["recorder_uri"]):
        recorder = R.get_recorder(experiment_name=source["experiment"], recorder_id=source["recorder_id"])
        _require(recorder.client.get_run(recorder.id).info.status == "FINISHED", "parent Recorder is not FINISHED")
        for key, artifact in PARENT_ARTIFACTS.items():
            value, identity = checked_parent_load(recorder, card, manifest, artifact)
            objects[key] = value
            artifacts.append(identity)
            artifact_paths.append((Path(recorder.get_local_dir()) / "artifacts" / artifact, identity["sha256"]))
            if key == "parent_source_hashes":
                _check_parent_code(root, value, parent_summary)
    _require(_same_evidence(objects["parent_source_audit"], parent_summary["source_audit"]), "parent source audit differs from archived summary")
    native, calendar, native_audit = _native_readback(root, plan, objects["parent_source_audit"])
    raw, prices, feature_audit = primary_frame_from_qualified_qlib(native, calendar, train_start=plan["source"]["model_signal_start"])
    _same_frame(raw, objects["raw"], "native primary frame")
    _same_frame(prices, objects["prices"], "native execution prices")
    _require(feature_audit["primary_frame_sha256"] == objects["parent_source_audit"]["features"]["primary_frame_sha256"],
             "native primary feature identity changed")
    forecast_audit = _check_forecasts(raw, objects["forecasts"], objects["forecast_models"], objects["forecast_audit"])
    dataset = build_anchor_dataset(objects["forecasts"], as_of=plan["source"]["as_of"])
    folds = _check_development_folds(dataset, objects["parent_dev_folds"], plan)
    _check_parent_code(root, objects["parent_source_hashes"], parent_summary)
    _require((card / "summary.json").read_bytes() == summary_bytes
             and _file_hash(card / "artifact_manifest.json") == source["artifact_manifest_sha256"], "parent card changed during read")
    _require(all(_file_hash(path) == digest for path, digest in artifact_paths), "parent artifact changed during reconstruction")
    details = {"source_artifacts": artifacts, "parent_source_hashes": objects["parent_source_hashes"],
        "native": native_audit, "forecasts": forecast_audit, "development_folds": folds,
        "raw_frame_sha256": frame_sha256(raw), "prices_frame_sha256": frame_sha256(prices)}
    audit = {"parent_experiment": source["experiment"], "parent_recorder_id": source["recorder_id"],
        "parent_artifact_manifest_sha256": source["artifact_manifest_sha256"],
        "qualified_source_recorder_id": plan["source"]["recorder_id"], "as_of": plan["source"]["as_of"],
        "provider_uri": plan["source"]["provider_uri"], "native_frame_sha256": native_audit["native_frame_sha256"],
        "raw_frame_sha256": frame_sha256(raw), "prices_frame_sha256": frame_sha256(prices),
        "forecasts_frame_sha256": forecast_audit["forecast_frame_sha256"],
        "rows": len(raw), "calendar_sessions": len(calendar), "forecast_model_count": len(objects["forecast_models"]),
        "verified_development_folds": len(folds), "verified_parent_artifacts": len(artifacts),
        "details_sha256": canonical_hash(details), "forecast_refit_performed": False,
        "allocation_models_loaded": False, "data_refreshed": False,
        "source_api": ["qlib.workflow.R.get_recorder", "Recorder.load_object", "qlib.data.D.calendar",
                       "qlib.data.D.instruments", "qlib.data.D.features", "qlib.data.dataset.DatasetH.prepare"]}
    return {key: objects[key] for key in ("raw", "prices", "forecasts", "forecast_models", "forecast_audit", "parent_dev_folds")} | {
        "dataset": dataset, "calendar": calendar, "source_audit": audit, "source_details": details}

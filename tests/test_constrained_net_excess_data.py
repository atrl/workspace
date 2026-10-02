"""Synthetic native-QLib source contracts; no market/model fitting."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src.data.benchmark_anchor_utility import (
    build_anchor_dataset, prepare_training_rows, fit_feature_scaler, transform_features,
)
from src.data.conditional_variance import primary_frame_from_qualified_qlib
from src.data.constrained_net_excess import (
    PARENT_ARTIFACTS, REQUIRED_PARENT_HELPERS, _check_development_folds, _check_forecasts,
    _native_readback, load_constrained_inputs,
)
from src.data.price_basis import FIELDS, PUBLISHED_BASIS, _sha, price_frame_sha256
from src.data.qlib_native import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame

ROOT = Path(__file__).resolve().parents[1]
ASOF = "2016-01-15"


class NeverRefitForecaster:
    n_features_in_ = 16

    def fit(self, *args, **kwargs):
        raise AssertionError("frozen data loader must never fit a forecaster")

    def predict(self, *args, **kwargs):
        raise AssertionError("loading frozen data must not regenerate predictions")


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def synthetic_source():
    dates = pd.bdate_range("2010-01-04", ASOF, name="datetime")
    t = np.arange(len(dates))
    parts = []
    for instrument, vol in (("SOXL", .025), ("SOXX", .009)):
        op = 25 * np.exp(np.cumsum(.0001 + vol * np.sin(t * .17)))
        close = op * (1 + .004 * np.cos(t * .1))
        parts.append(pd.DataFrame({"instrument": instrument, "datetime": dates, "open": op,
            "high": np.maximum(op, close) * 1.01, "low": np.minimum(op, close) * .99, "close": close,
            "volume": 100000 * np.exp(.3 * np.sin(t * .21))}))
    return pd.concat(parts, ignore_index=True), dates


@pytest.fixture
def frozen_parent(tmp_path, monkeypatch, synthetic_source):
    from qlib.workflow import R

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    source_frame, calendar = synthetic_source
    provider_relative = "data/qlib/qualified_test"
    provider = tmp_path / provider_relative
    build_qlib_store_from_frame(source_frame, provider_uri=provider, as_of=ASOF, field_map={field: field for field in FIELDS})
    native = load_qlib_feature_frame(QlibProviderConfig(provider_uri=provider), instruments=["SOXL", "SOXX"],
        fields=["$" + field for field in FIELDS], start="2010-01-04", end=ASOF).sort_index()
    native_hash = price_frame_sha256(native)
    provider_manifest = {"qualification": {"qualification_pass": True, "as_of": ASOF},
        "reference_calendar": {"dates": calendar.strftime("%Y-%m-%d").tolist()},
        "native_readback": {"frame_sha256": native_hash}}
    (provider / "price_basis_manifest.json").write_text(json.dumps(provider_manifest, indent=2))
    (provider / "price_basis_status.json").write_text(json.dumps({"status": "qualified", "manifest_sha256": _sha(provider_manifest)}))
    raw, prices, features_audit = primary_frame_from_qualified_qlib(native, calendar)
    canonical_card = tmp_path / "research/synthetic_qualified"
    canonical_card.mkdir(parents=True)
    canonical_summary = {"recorder_id": "synthetic_qualified_record", "source_qualification_pass": True,
        "basis": PUBLISHED_BASIS, "native_readback": {"frame_sha256": native_hash}}
    (canonical_card / "summary.json").write_text(json.dumps(canonical_summary))
    source_audit = {"parent_recorder_id": "synthetic_qualified_record",
        "parent_summary_sha256": _sha_file(canonical_card / "summary.json"),
        "provider_manifest_sha256": _sha_file(provider / "price_basis_manifest.json"),
        "native_frame_sha256": native_hash, "features": features_audit}
    forecasts = raw.copy()
    forecasts["forecast_model_available_at"] = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    audit_rows, forecast_models = [], {}
    for year in sorted(raw.index.year.unique()):
        cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
        mature = raw.feature_available_at.lt(cutoff) & raw.forecast_available_at.lt(cutoff) & raw.return_available_at.lt(cutoff)
        count = int(mature.sum()); ready = count >= 252
        audit_rows.append({"model_year": int(year), "available": ready, "rows": count,
            "minimum_training_rows": 252, "ridge_alpha": 10., "decision_at": cutoff.isoformat(),
            "label_available_at_max": raw.loc[mature, "forecast_available_at"].max().isoformat() if count else None})
        if ready:
            mask = raw.index.year == year
            forecasts.loc[mask, "forecast_available"] = 1.
            forecasts.loc[mask, "future_vol_prediction"] = .02 + .003 * np.sin(np.arange(int(mask.sum())) * .13)
            forecasts.loc[mask, "cash_advantage_prediction"] = .0001
            forecasts.loc[mask, "forecast_model_available_at"] = cutoff.tz_convert("UTC")
            forecast_models[int(year)] = NeverRefitForecaster()
    forecast_audit = pd.DataFrame(audit_rows)
    forecast_audit["features"] = [["feature_a", "feature_b"] for _ in audit_rows]
    dataset = build_anchor_dataset(forecasts, as_of=ASOF)
    rows, maturity = prepare_training_rows(dataset, 2016)
    folds = {}
    for variant in ("constant", "price", "price_vol"):
        scaler = fit_feature_scaler(rows, maturity, variant)
        transformed = transform_features(rows, scaler, variant)
        folds[(variant, 2016)] = {"rows": rows.copy(), "features": transformed["features"],
            "returns": rows[["soxl_return", "soxx_return"]].to_numpy(float), "scaler": scaler,
            "invalid_training": False, "invalid_training_dates": [], "audit": maturity,
            "transform_audit": transformed["audit"]}
    source_hashes = {}
    for filename in REQUIRED_PARENT_HELPERS:
        path = tmp_path / filename; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, path)
        source_hashes[filename] = _sha_file(path)
    source = {"experiment": "synthetic_qualified", "recorder_id": "synthetic_qualified_record",
        "provider_uri": provider_relative, "as_of": ASOF, "model_signal_start": "2011-01-03",
        "history_start": "2010-01-04", "basis": PUBLISHED_BASIS}
    objects = {"parent_source_hashes": source_hashes, "parent_source_audit": source_audit,
        "raw": raw, "prices": prices, "forecasts": forecasts, "forecast_models": forecast_models,
        "forecast_audit": forecast_audit, "parent_dev_folds": folds}
    card = tmp_path / "research/synthetic_parent"; card.mkdir(parents=True)
    uri = tmp_path / "recorder"
    with R.start(experiment_name="synthetic_parent", uri=str(uri)):
        recorder = R.get_recorder()
        for key, artifact in PARENT_ARTIFACTS.items():
            path = Path(artifact)
            recorder.save_objects(artifact_path=str(path.parent), **{path.name: objects[key]})
        summary = {"experiment": "synthetic_parent", "recorder_id": recorder.id,
            "source": source, "source_sha256": source_hashes, "source_audit": source_audit}
        (card / "summary.json").write_text(json.dumps(summary))
        recorder.log_artifact(str(card / "summary.json"), artifact_path="card")
        recdir = Path(recorder.get_local_dir())
    entries = []
    for file in sorted((recdir / "artifacts").rglob("*")):
        if file.is_file():
            entries.append({"path": str(Path("qlib_recorder") / recdir.relative_to(uri) / file.relative_to(recdir)),
                "sha256": _sha_file(file), "bytes": file.stat().st_size})
    (card / "artifact_manifest.json").write_text(json.dumps({"entries": entries}))
    plan = {"source": source, "parent_input_source": {"experiment": "synthetic_parent",
            "recorder_id": recorder.id, "recorder_uri": str(uri), "artifact_manifest_sha256": _sha_file(card / "artifact_manifest.json")},
        "variants": [{"name": name} for name in ("constant", "price", "price_vol")],
        "training": {"development_signal_years": [2016], "minimum_rows": 252}}
    return tmp_path, plan, objects, recdir


def test_loads_native_and_frozen_forecasts_without_fit_or_allocation_models(frozen_parent):
    root, plan, objects, _ = frozen_parent
    result = load_constrained_inputs(root, plan)
    pd.testing.assert_frame_equal(result["raw"], objects["raw"])
    pd.testing.assert_frame_equal(result["prices"], objects["prices"])
    assert len(result["dataset"].prepare("all")) == len(result["raw"])
    assert result["source_audit"]["verified_development_folds"] == 3
    assert result["source_audit"]["verified_parent_artifacts"] == 8
    assert not result["source_audit"]["forecast_refit_performed"]
    assert not result["source_audit"]["allocation_models_loaded"]
    assert all(isinstance(model, NeverRefitForecaster) for model in result["forecast_models"].values())
    assert {x["artifact"] for x in result["source_details"]["source_artifacts"]} == set(PARENT_ARTIFACTS.values())


def test_corrupted_parent_bytes_are_rejected_before_deserialization(frozen_parent):
    root, plan, _, recdir = frozen_parent
    (recdir / "artifacts" / PARENT_ARTIFACTS["forecast_models"]).write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="before deserialization"):
        load_constrained_inputs(root, plan)


def test_changed_parent_summary_or_code_is_rejected(frozen_parent):
    root, plan, _, _ = frozen_parent
    code = root / "src/model/benchmark_anchor_utility.py"
    code.write_text(code.read_text() + "\n# unexpected change\n")
    with pytest.raises(ValueError, match="immutable parent helper changed"):
        load_constrained_inputs(root, plan)
    summary = root / "research/synthetic_parent/summary.json"
    summary.write_text(summary.read_text() + "\n")
    with pytest.raises(ValueError, match="archived card snapshot"):
        load_constrained_inputs(root, plan)


def test_native_file_value_drift_is_not_hidden_by_matching_metadata(frozen_parent):
    root, plan, objects, _ = frozen_parent
    provider = root / plan["source"]["provider_uri"]
    file = provider / "features/soxl/open.day.bin"
    values = np.frombuffer(file.read_bytes(), dtype="<f4").copy(); values[150] += 1
    file.write_bytes(values.tobytes())
    with pytest.raises(ValueError, match="native OHLCV content"):
        _native_readback(root, plan, objects["parent_source_audit"])


def test_reconstructed_fold_rejects_missing_rows_scaler_and_matrix_tampering(frozen_parent):
    _, plan, objects, _ = frozen_parent
    dataset = build_anchor_dataset(objects["forecasts"], as_of=ASOF)
    for mutation, message in (("missing", "coverage"), ("scaler", "scaler differs"), ("matrix", "matrix differs")):
        folds = copy.deepcopy(objects["parent_dev_folds"])
        if mutation == "missing": del folds[("constant", 2016)]
        elif mutation == "scaler": folds[("price", 2016)]["scaler"]["price_mean"][0] += .01
        else: folds[("price", 2016)]["features"][0, 0] += .01
        with pytest.raises(ValueError, match=message):
            _check_development_folds(dataset, folds, plan)


def test_forecast_calendar_and_audit_maturity_are_immutable(frozen_parent):
    _, _, obj, _ = frozen_parent
    changed = obj["forecasts"].copy()
    changed.loc[changed.index[50], "soxx_return"] += .01
    with pytest.raises(ValueError, match="non-forecast primary columns"):
        _check_forecasts(obj["raw"], changed, obj["forecast_models"], obj["forecast_audit"])
    audit = obj["forecast_audit"].copy()
    selected = audit.index[audit.available][0]
    audit.loc[selected, "label_available_at_max"] = audit.loc[selected, "decision_at"]
    with pytest.raises(ValueError, match="outcome not strictly prior"):
        _check_forecasts(obj["raw"], obj["forecasts"], obj["forecast_models"], audit)

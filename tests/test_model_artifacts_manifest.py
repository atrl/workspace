from __future__ import annotations

import json

import pandas as pd

from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model import fit_model_artifact, load_model_artifact, save_model_artifact
from src.model.matrix import run_factor_feature_set_matrix


def test_model_artifact_save_load_round_trips_predictions(tmp_path):
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-04-30")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-03-31"]
    test = dataset.frame.loc[dataset.frame["date"] > "2026-03-31"]

    artifact = fit_model_artifact(
        "ridge_mse",
        train=train,
        feature_columns=dataset.feature_columns[:5],
    )
    before = artifact.predict(test)

    artifact_path = save_model_artifact(tmp_path / "ridge_mse.joblib", artifact)
    loaded = load_model_artifact(artifact_path)
    after = loaded.predict(test)

    assert artifact_path.exists()
    assert loaded.model_name == "ridge_mse"
    assert loaded.feature_columns == dataset.feature_columns[:5]
    assert loaded.metadata["schema_version"] == "model_artifact.v1"
    pd.testing.assert_series_equal(before["prediction"], after["prediction"])


def test_factor_feature_set_matrix_writes_manifest_with_dataset_and_sidecar_paths(tmp_path):
    run_factor_feature_set_matrix(
        card_dir=tmp_path,
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        top_n_factors=3,
    )

    manifest_path = tmp_path / "artifact_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "artifact.v1"
    assert manifest["artifact_type"] == "model_matrix"
    assert manifest["dataset"]["name"] == "soxl_support_daily"
    assert manifest["dataset"]["calendar_path"].endswith("data/qlib/soxl/calendars/day.txt")
    assert manifest["metadata"]["train_end"] == "2026-04-30"
    assert manifest["metadata"]["adapter_status"] == "smoke_adapter"
    assert len(manifest["manifest_hash"]) == 64
    assert {
        "analysis_summary",
        "feature_set_specs",
        "model_comparison",
        "qlib_risk_metrics",
        "model_effect_deltas",
    }.issubset(set(manifest["paths"]))

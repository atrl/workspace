from __future__ import annotations

from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model import fit_model_artifact
from src.model.registry import QUANTBENCH_MODEL_NAMES
from src.model.training import fit_predict_model, model_adapter_name, model_adapter_status


def _split_frame():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-04-30")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-03-31"].copy()
    test = dataset.frame.loc[dataset.frame["date"] > "2026-03-31"].copy()
    return dataset, train, test


def test_mlp_mse_uses_named_sklearn_mlp_adapter():
    dataset, train, test = _split_frame()

    predictions = fit_predict_model("mlp_mse", train=train, test=test, feature_columns=dataset.feature_columns[:8])

    assert predictions["adapter_status"].unique().tolist() == ["paper_strict_adapter"]
    assert predictions["model_adapter"].unique().tolist() == ["mlp_sklearn_mse_v1"]
    assert predictions["prediction"].notna().all()
    assert model_adapter_status("mlp_mse") == "paper_strict_adapter"
    assert model_adapter_name("mlp_mse") == "mlp_sklearn_mse_v1"


def test_all_quantbench_models_have_explicit_non_smoke_adapter_status():
    fallback_allowed = {"xgboost_mse", "catboost_rank"}

    statuses = {name: model_adapter_status(name) for name in QUANTBENCH_MODEL_NAMES}
    adapters = {name: model_adapter_name(name) for name in QUANTBENCH_MODEL_NAMES}

    assert {name for name, status in statuses.items() if status == "smoke_adapter"} == set()
    assert {name for name, adapter in adapters.items() if adapter == "smoke_fallback_v1"} == set()
    assert {name for name, status in statuses.items() if status == "dependency_fallback_adapter"} == fallback_allowed
    assert {name for name, status in statuses.items() if status == "paper_strict_adapter"} == (
        set(QUANTBENCH_MODEL_NAMES) - fallback_allowed
    )


def test_mlp_artifact_metadata_records_named_adapter():
    dataset, train, test = _split_frame()

    artifact = fit_model_artifact("mlp_mse", train=train, feature_columns=dataset.feature_columns[:8])
    predictions = artifact.predict(test)
    metadata = artifact.metadata_payload()

    assert artifact.adapter_status == "paper_strict_adapter"
    assert metadata["model_adapter"] == "mlp_sklearn_mse_v1"
    assert predictions["model_adapter"].unique().tolist() == ["mlp_sklearn_mse_v1"]

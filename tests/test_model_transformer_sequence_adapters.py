from __future__ import annotations

from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model import fit_model_artifact, load_model_artifact, save_model_artifact
from src.model.sequence import AutoformerDecompositionRegressor, DLinearDecompositionRegressor, PatchTSTTransformerRegressor
from src.model.training import fit_estimator_for_model, fit_predict_model, model_adapter_name, model_adapter_status


def _split_frame():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-06-09")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-04-30"].copy()
    test = dataset.frame.loc[dataset.frame["date"] > "2026-04-30"].copy()
    return dataset, train, test


def test_transformer_sequence_models_use_named_paper_strict_adapters():
    dataset, train, test = _split_frame()

    expected = {
        "hawkes_gru_mse": "hawkes_gru_event_decay_mse_v1",
        "mlp_mixer_mse": "mlp_mixer_sequence_mse_v1",
        "dlinear_mse": "dlinear_decomposition_mse_v1",
        "patch_tst_mse": "patch_tst_transformer_mse_v1",
        "autoformer_mse": "autoformer_decomposition_attention_mse_v1",
        "fedformer_mse": "fedformer_frequency_attention_mse_v1",
    }
    for model_name, adapter_name in expected.items():
        predictions = fit_predict_model(
            model_name,
            train=train,
            test=test,
            feature_columns=dataset.feature_columns[:5],
        )

        assert predictions["adapter_status"].unique().tolist() == ["paper_strict_adapter"]
        assert predictions["model_adapter"].unique().tolist() == [adapter_name]
        assert predictions["prediction"].notna().all()
        assert model_adapter_status(model_name) == "paper_strict_adapter"
        assert model_adapter_name(model_name) == adapter_name


def test_mixer_decomposition_patch_and_frequency_paths_are_distinct():
    dataset, train, test = _split_frame()

    mixer = fit_predict_model("mlp_mixer_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])
    dlinear = fit_predict_model("dlinear_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])
    patch = fit_predict_model("patch_tst_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])
    autoformer = fit_predict_model("autoformer_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])
    fedformer = fit_predict_model("fedformer_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])

    assert not mixer["prediction"].round(10).equals(dlinear["prediction"].round(10))
    assert not patch["prediction"].round(10).equals(autoformer["prediction"].round(10))
    assert not fedformer["prediction"].round(10).equals(dlinear["prediction"].round(10))


def test_dlinear_mse_uses_explicit_decomposition_linear_estimator():
    dataset, train, _ = _split_frame()

    estimator, constant_prediction = fit_estimator_for_model(
        "dlinear_mse",
        train=train,
        feature_columns=dataset.feature_columns[:5],
    )

    assert constant_prediction is None
    assert isinstance(estimator, DLinearDecompositionRegressor)
    assert hasattr(estimator, "trend_model_")
    assert hasattr(estimator, "residual_model_")
    assert estimator.feature_count_ == 5
    assert estimator.lookback_ == 5


def test_patch_tst_mse_uses_patch_transformer_estimator():
    dataset, train, _ = _split_frame()

    estimator, constant_prediction = fit_estimator_for_model(
        "patch_tst_mse",
        train=train,
        feature_columns=dataset.feature_columns[:5],
    )

    assert constant_prediction is None
    assert isinstance(estimator, PatchTSTTransformerRegressor)
    assert hasattr(estimator, "model_")
    assert estimator.feature_count_ == 5
    assert estimator.lookback_ == 5
    assert estimator.patch_size == 2


def test_autoformer_mse_uses_decomposition_autocorrelation_estimator():
    dataset, train, _ = _split_frame()

    estimator, constant_prediction = fit_estimator_for_model(
        "autoformer_mse",
        train=train,
        feature_columns=dataset.feature_columns[:5],
    )

    assert constant_prediction is None
    assert isinstance(estimator, AutoformerDecompositionRegressor)
    assert hasattr(estimator, "trend_model_")
    assert hasattr(estimator, "seasonal_model_")
    assert estimator.feature_count_ == 5
    assert estimator.lookback_ == 5
    assert estimator.autocorrelation_lags_ == (1, 2, 3, 4)
    assert estimator.top_k == 2


def test_hawkes_gru_event_decay_path_differs_from_plain_gru():
    dataset, train, test = _split_frame()

    hawkes = fit_predict_model("hawkes_gru_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])
    gru = fit_predict_model("gru_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])

    assert not hawkes["prediction"].round(10).equals(gru["prediction"].round(10))


def test_patch_tst_artifact_metadata_records_transformer_adapter_after_load(tmp_path):
    dataset, train, test = _split_frame()

    artifact = fit_model_artifact("patch_tst_mse", train=train, feature_columns=dataset.feature_columns[:5])
    path = save_model_artifact(tmp_path / "patch_tst_mse.joblib", artifact)
    loaded = load_model_artifact(path)
    predictions = loaded.predict(test)
    metadata = loaded.metadata_payload()

    assert metadata["model_adapter"] == "patch_tst_transformer_mse_v1"
    assert loaded.adapter_status == "paper_strict_adapter"
    assert predictions["model_adapter"].unique().tolist() == ["patch_tst_transformer_mse_v1"]
    assert predictions["prediction"].notna().all()

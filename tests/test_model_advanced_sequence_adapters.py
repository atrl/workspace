from __future__ import annotations

from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model import fit_model_artifact, load_model_artifact, save_model_artifact
from src.model.sequence import SFMFrequencyMemoryRegressor
from src.model.training import fit_predict_model, model_adapter_name, model_adapter_status


def _split_frame():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-06-09")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-04-30"].copy()
    test = dataset.frame.loc[dataset.frame["date"] > "2026-04-30"].copy()
    return dataset, train, test


def test_advanced_sequence_models_use_named_paper_strict_adapters():
    dataset, train, test = _split_frame()

    expected = {
        "alstm_mse": "alstm_attention_lstm_mse_v1",
        "sfm_mse": "sfm_state_frequency_memory_mse_v1",
        "darnn_mse": "darnn_dual_attention_gru_mse_v1",
        "tcn_mse": "tcn_temporal_conv_mse_v1",
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


def test_attention_frequency_and_temporal_conv_paths_are_distinct():
    dataset, train, test = _split_frame()

    alstm = fit_predict_model("alstm_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])
    sfm = fit_predict_model("sfm_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])
    darnn = fit_predict_model("darnn_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])
    tcn = fit_predict_model("tcn_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])

    assert not alstm["prediction"].round(10).equals(sfm["prediction"].round(10))
    assert not darnn["prediction"].round(10).equals(alstm["prediction"].round(10))
    assert not tcn["prediction"].round(10).equals(sfm["prediction"].round(10))


def test_darnn_artifact_metadata_records_attention_adapter_after_load(tmp_path):
    dataset, train, test = _split_frame()

    artifact = fit_model_artifact("darnn_mse", train=train, feature_columns=dataset.feature_columns[:5])
    path = save_model_artifact(tmp_path / "darnn_mse.joblib", artifact)
    loaded = load_model_artifact(path)
    predictions = loaded.predict(test)
    metadata = loaded.metadata_payload()

    assert metadata["model_adapter"] == "darnn_dual_attention_gru_mse_v1"
    assert loaded.adapter_status == "paper_strict_adapter"
    assert predictions["model_adapter"].unique().tolist() == ["darnn_dual_attention_gru_mse_v1"]
    assert predictions["prediction"].notna().all()


def test_sfm_artifact_uses_state_frequency_memory_estimator_after_load(tmp_path):
    dataset, train, test = _split_frame()

    artifact = fit_model_artifact("sfm_mse", train=train, feature_columns=dataset.feature_columns[:5])
    assert isinstance(artifact.estimator, SFMFrequencyMemoryRegressor)
    assert artifact.estimator.frequency_count == 3
    assert artifact.estimator.hidden_size == 8
    assert hasattr(artifact.estimator, "model_")

    path = save_model_artifact(tmp_path / "sfm_mse.joblib", artifact)
    loaded = load_model_artifact(path)
    predictions = loaded.predict(test)
    metadata = loaded.metadata_payload()

    assert isinstance(loaded.estimator, SFMFrequencyMemoryRegressor)
    assert metadata["model_adapter"] == "sfm_state_frequency_memory_mse_v1"
    assert loaded.adapter_status == "paper_strict_adapter"
    assert predictions["model_adapter"].unique().tolist() == ["sfm_state_frequency_memory_mse_v1"]
    assert predictions["prediction"].notna().all()

from __future__ import annotations

from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model import fit_model_artifact, load_model_artifact, save_model_artifact
from src.model.sequence import LaggedRidgeRegressor
from src.model.training import fit_estimator_for_model, fit_predict_model, model_adapter_name, model_adapter_status


def _split_frame():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-06-09")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-04-30"].copy()
    test = dataset.frame.loc[dataset.frame["date"] > "2026-04-30"].copy()
    return dataset, train, test


def test_sequence_models_use_named_paper_strict_adapters():
    dataset, train, test = _split_frame()

    expected = {
        "sequence_lag_ridge_mse": "sequence_lag_ridge_mse_v1",
        "lstm_mse": "lstm_pytorch_sequence_mse_v1",
        "gru_mse": "gru_pytorch_sequence_mse_v1",
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


def test_sequence_lag_ridge_mse_uses_native_lagged_sequence_estimator():
    dataset, train, _ = _split_frame()

    estimator, constant_prediction = fit_estimator_for_model(
        "sequence_lag_ridge_mse",
        train=train,
        feature_columns=dataset.feature_columns[:5],
    )

    assert constant_prediction is None
    assert isinstance(estimator, LaggedRidgeRegressor)
    assert hasattr(estimator, "model_")
    assert estimator.lookback_ == 5
    assert estimator.feature_count_ == 5
    assert estimator.flattened_feature_count_ == 25


def test_sequence_lag_ridge_predictions_are_not_plain_current_row_ridge():
    dataset, train, test = _split_frame()

    plain = fit_predict_model("ridge_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])
    lagged = fit_predict_model(
        "sequence_lag_ridge_mse",
        train=train,
        test=test,
        feature_columns=dataset.feature_columns[:5],
    )

    assert lagged["objective"].unique().tolist() == ["mse"]
    assert not lagged["prediction"].round(10).equals(plain["prediction"].round(10))


def test_lstm_and_gru_sequence_adapters_produce_distinct_predictions():
    dataset, train, test = _split_frame()

    lstm = fit_predict_model("lstm_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])
    gru = fit_predict_model("gru_mse", train=train, test=test, feature_columns=dataset.feature_columns[:5])

    assert not lstm["prediction"].round(10).equals(gru["prediction"].round(10))


def test_gru_artifact_metadata_records_sequence_adapter_after_load(tmp_path):
    dataset, train, test = _split_frame()

    artifact = fit_model_artifact("gru_mse", train=train, feature_columns=dataset.feature_columns[:5])
    path = save_model_artifact(tmp_path / "gru_mse.joblib", artifact)
    loaded = load_model_artifact(path)
    predictions = loaded.predict(test)
    metadata = loaded.metadata_payload()

    assert metadata["model_adapter"] == "gru_pytorch_sequence_mse_v1"
    assert loaded.adapter_status == "paper_strict_adapter"
    assert predictions["model_adapter"].unique().tolist() == ["gru_pytorch_sequence_mse_v1"]
    assert predictions["prediction"].notna().all()

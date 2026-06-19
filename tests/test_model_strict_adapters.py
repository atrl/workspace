from __future__ import annotations

from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model import fit_model_artifact
from src.model.training import RidgeICRegressor, fit_estimator_for_model, fit_predict_model


def _split_frame():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-04-30")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-03-31"].copy()
    test = dataset.frame.loc[dataset.frame["date"] > "2026-03-31"].copy()
    return dataset, train, test


def test_core_linear_models_use_named_paper_strict_adapters():
    dataset, train, test = _split_frame()

    for model_name, adapter_name in {
        "logistic_clf": "logistic_direction_v1",
        "ridge_mse": "ridge_mse_v1",
        "ridge_ic": "ridge_ic_rank_correlation_v1",
    }.items():
        predictions = fit_predict_model(
            model_name,
            train=train,
            test=test,
            feature_columns=dataset.feature_columns[:6],
        )

        assert predictions["adapter_status"].unique().tolist() == ["paper_strict_adapter"]
        assert predictions["model_adapter"].unique().tolist() == [adapter_name]
        assert predictions["prediction"].notna().all()


def test_ridge_ic_predictions_are_not_the_same_as_ridge_mse_proxy():
    dataset, train, test = _split_frame()

    ridge_mse = fit_predict_model("ridge_mse", train=train, test=test, feature_columns=dataset.feature_columns[:6])
    ridge_ic = fit_predict_model("ridge_ic", train=train, test=test, feature_columns=dataset.feature_columns[:6])

    assert ridge_ic["objective"].unique().tolist() == ["ic"]
    assert not ridge_ic["prediction"].round(10).equals(ridge_mse["prediction"].round(10))


def test_ridge_ic_uses_native_ic_estimator_with_rank_target_metadata():
    dataset, train, _ = _split_frame()

    estimator, constant_prediction = fit_estimator_for_model(
        "ridge_ic",
        train=train,
        feature_columns=dataset.feature_columns[:6],
    )

    assert constant_prediction is None
    assert isinstance(estimator, RidgeICRegressor)
    assert hasattr(estimator, "model_")
    assert estimator.objective_name_ == "ic"
    assert estimator.target_kind_ == "forward_return_rank"
    assert estimator.rank_target_std_ > 0.0
    assert abs(estimator.rank_target_mean_) < 1e-12


def test_model_artifact_metadata_records_strict_adapter_name_and_status():
    dataset, train, test = _split_frame()

    artifact = fit_model_artifact("ridge_ic", train=train, feature_columns=dataset.feature_columns[:6])
    predictions = artifact.predict(test)
    metadata = artifact.metadata_payload()

    assert artifact.adapter_status == "paper_strict_adapter"
    assert metadata["model_adapter"] == "ridge_ic_rank_correlation_v1"
    assert predictions["adapter_status"].unique().tolist() == ["paper_strict_adapter"]
    assert predictions["model_adapter"].unique().tolist() == ["ridge_ic_rank_correlation_v1"]

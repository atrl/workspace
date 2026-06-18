from __future__ import annotations

from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model import fit_model_artifact
from src.model.training import fit_predict_model, model_adapter_name, model_adapter_status


def _split_frame():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-04-30")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-03-31"].copy()
    test = dataset.frame.loc[dataset.frame["date"] > "2026-03-31"].copy()
    return dataset, train, test


def test_tree_and_installed_boosting_models_use_named_adapters():
    dataset, train, test = _split_frame()

    expected = {
        "random_forest_clf": ("paper_strict_adapter", "random_forest_direction_v1"),
        "hist_gradient_mse": ("paper_strict_adapter", "hist_gradient_mse_v1"),
        "extra_trees_rank": ("paper_strict_adapter", "extra_trees_rank_proxy_v1"),
        "lightgbm_mse": ("paper_strict_adapter", "lightgbm_mse_v1"),
    }
    for model_name, (status, adapter_name) in expected.items():
        predictions = fit_predict_model(
            model_name,
            train=train,
            test=test,
            feature_columns=dataset.feature_columns[:6],
        )

        assert predictions["adapter_status"].unique().tolist() == [status]
        assert predictions["model_adapter"].unique().tolist() == [adapter_name]
        assert predictions["prediction"].notna().all()
        assert model_adapter_status(model_name) == status
        assert model_adapter_name(model_name) == adapter_name


def test_xgboost_and_catboost_use_native_named_adapters():
    dataset, train, test = _split_frame()

    expected = {
        "xgboost_mse": "xgboost_mse_v1",
        "catboost_rank": "catboost_rank_v1",
    }
    for model_name, adapter_name in expected.items():
        predictions = fit_predict_model(
            model_name,
            train=train,
            test=test,
            feature_columns=dataset.feature_columns[:6],
        )

        assert predictions["adapter_status"].unique().tolist() == ["paper_strict_adapter"]
        assert predictions["model_adapter"].unique().tolist() == [adapter_name]
        assert predictions["prediction"].notna().all()
        assert model_adapter_status(model_name) == "paper_strict_adapter"
        assert model_adapter_name(model_name) == adapter_name


def test_ranking_adapter_predictions_are_distinct_from_mse_boosting_path():
    dataset, train, test = _split_frame()

    rank_predictions = fit_predict_model(
        "extra_trees_rank",
        train=train,
        test=test,
        feature_columns=dataset.feature_columns[:6],
    )
    mse_predictions = fit_predict_model(
        "hist_gradient_mse",
        train=train,
        test=test,
        feature_columns=dataset.feature_columns[:6],
    )

    assert rank_predictions["objective"].unique().tolist() == ["ranking"]
    assert not rank_predictions["prediction"].round(10).equals(mse_predictions["prediction"].round(10))


def test_artifact_metadata_records_lightgbm_adapter_name_and_status():
    dataset, train, test = _split_frame()

    artifact = fit_model_artifact("lightgbm_mse", train=train, feature_columns=dataset.feature_columns[:6])
    predictions = artifact.predict(test)
    metadata = artifact.metadata_payload()

    assert artifact.adapter_status == "paper_strict_adapter"
    assert metadata["model_adapter"] == "lightgbm_mse_v1"
    assert predictions["adapter_status"].unique().tolist() == ["paper_strict_adapter"]
    assert predictions["model_adapter"].unique().tolist() == ["lightgbm_mse_v1"]

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .registry import get_model_spec


STRICT_ADAPTER_NAMES = {
    "logistic_clf": "logistic_direction_v1",
    "ridge_mse": "ridge_mse_v1",
    "ridge_ic": "ridge_rank_ic_proxy_v1",
}


def model_adapter_name(model_name: str) -> str:
    return STRICT_ADAPTER_NAMES.get(model_name, "smoke_fallback_v1")


def model_adapter_status(model_name: str) -> str:
    return "paper_strict_adapter" if model_name in STRICT_ADAPTER_NAMES else "smoke_adapter"


def _feature_matrix(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, list(feature_columns)].astype(float).to_numpy()


def _regression_target(frame: pd.DataFrame) -> np.ndarray:
    return frame["target_return"].astype(float).to_numpy()


def _classification_target(frame: pd.DataFrame) -> np.ndarray:
    return (frame["target_return"].astype(float).to_numpy() > 0.0).astype(int)


def _ic_rank_target(frame: pd.DataFrame) -> np.ndarray:
    ranked = frame["target_return"].astype(float).rank(method="average", pct=True).to_numpy() - 0.5
    std = float(np.std(ranked))
    if not np.isfinite(std) or std <= 1e-12:
        return ranked
    return ranked / std


def _target_for_model(model_name: str, frame: pd.DataFrame) -> np.ndarray:
    spec = get_model_spec(model_name)
    if model_name == "logistic_clf" or spec.objective == "classification":
        return _classification_target(frame)
    if model_name == "ridge_ic":
        return _ic_rank_target(frame)
    return _regression_target(frame)


def _estimator_for_model(model_name: str):
    spec = get_model_spec(model_name)
    if model_name == "logistic_clf":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, random_state=7),
        )
    if model_name in {"ridge_mse", "ridge_ic"}:
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if spec.family == "tree" and spec.objective == "classification":
        return RandomForestClassifier(n_estimators=24, max_depth=4, random_state=7)
    if spec.objective == "classification":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=7))
    if spec.family == "tree" and spec.objective == "ranking":
        return ExtraTreesRegressor(n_estimators=24, max_depth=4, random_state=7)
    if spec.family == "tree":
        return RandomForestRegressor(n_estimators=24, max_depth=4, random_state=7)
    if spec.family == "boosting":
        return HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=15, random_state=7)
    if spec.family == "neural_tabular":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(16,), max_iter=250, random_state=7, early_stopping=True),
        )
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def fit_estimator_for_model(
    model_name: str,
    *,
    train: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[object, float | None]:
    estimator = _estimator_for_model(model_name)
    x_train = _feature_matrix(train, feature_columns)
    y_train = _target_for_model(model_name, train)
    constant_prediction: float | None = None
    spec = get_model_spec(model_name)

    if spec.objective == "classification":
        if len(set(y_train.astype(int).tolist())) < 2:
            constant_prediction = float(y_train[0]) - 0.5
        else:
            estimator.fit(x_train, y_train.astype(int))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(x_train, y_train)
    return estimator, constant_prediction


def predict_with_estimator(
    model_name: str,
    *,
    estimator: object,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    constant_prediction: float | None = None,
) -> np.ndarray:
    x_test = _feature_matrix(frame, feature_columns)
    spec = get_model_spec(model_name)
    if constant_prediction is not None:
        return np.full(len(frame), float(constant_prediction))
    if spec.objective == "classification":
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(x_test)[:, 1] - 0.5
        return estimator.predict(x_test).astype(float) - 0.5
    return estimator.predict(x_test)


def fit_predict_model(
    model_name: str,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    if train.empty or test.empty:
        raise ValueError("train and test frames must be non-empty")
    spec = get_model_spec(model_name)
    estimator, constant_prediction = fit_estimator_for_model(
        model_name,
        train=train,
        feature_columns=feature_columns,
    )
    raw_prediction = predict_with_estimator(
        model_name,
        estimator=estimator,
        frame=test,
        feature_columns=feature_columns,
        constant_prediction=constant_prediction,
    )

    return pd.DataFrame(
        {
            "date": test["date"].tolist(),
            "model": model_name,
            "prediction": np.asarray(raw_prediction, dtype=float),
            "target_return": test["target_return"].astype(float).tolist(),
            "objective": spec.objective,
            "family": spec.family,
            "adapter_status": model_adapter_status(model_name),
            "model_adapter": model_adapter_name(model_name),
        }
    )

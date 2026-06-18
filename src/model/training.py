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


def _feature_matrix(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, list(feature_columns)].astype(float).to_numpy()


def _regression_target(frame: pd.DataFrame) -> np.ndarray:
    return frame["target_return"].astype(float).to_numpy()


def _classification_target(frame: pd.DataFrame) -> np.ndarray:
    return (frame["target_return"].astype(float).to_numpy() > 0.0).astype(int)


def _estimator_for_model(model_name: str):
    spec = get_model_spec(model_name)
    if spec.objective == "classification":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, random_state=7),
        )
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
    estimator = _estimator_for_model(model_name)
    x_train = _feature_matrix(train, feature_columns)
    x_test = _feature_matrix(test, feature_columns)

    if spec.objective == "classification":
        y_train = _classification_target(train)
        if len(set(y_train.tolist())) < 2:
            raw_prediction = np.full(len(test), float(y_train[0]) - 0.5)
        else:
            estimator.fit(x_train, y_train)
            if hasattr(estimator, "predict_proba"):
                raw_prediction = estimator.predict_proba(x_test)[:, 1] - 0.5
            else:
                raw_prediction = estimator.predict(x_test).astype(float) - 0.5
    else:
        y_train = _regression_target(train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(x_train, y_train)
        raw_prediction = estimator.predict(x_test)

    return pd.DataFrame(
        {
            "date": test["date"].tolist(),
            "model": model_name,
            "prediction": np.asarray(raw_prediction, dtype=float),
            "target_return": test["target_return"].astype(float).tolist(),
            "objective": spec.objective,
            "family": spec.family,
            "adapter_status": "smoke_adapter",
        }
    )

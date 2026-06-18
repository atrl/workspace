from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .registry import get_model_spec
from .training import _classification_target, _estimator_for_model, _feature_matrix, _regression_target


@dataclass(frozen=True)
class ModelArtifact:
    model_name: str
    feature_columns: tuple[str, ...]
    estimator: Any
    objective: str
    family: str
    adapter_status: str = "smoke_adapter"
    target_column: str = "target_return"
    metadata: dict[str, Any] = field(default_factory=dict)
    constant_prediction: float | None = None

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("prediction frame must be non-empty")
        x_test = _feature_matrix(frame, self.feature_columns)
        if self.constant_prediction is not None:
            raw_prediction = np.full(len(frame), float(self.constant_prediction))
        elif self.objective == "classification":
            if hasattr(self.estimator, "predict_proba"):
                raw_prediction = self.estimator.predict_proba(x_test)[:, 1] - 0.5
            else:
                raw_prediction = self.estimator.predict(x_test).astype(float) - 0.5
        else:
            raw_prediction = self.estimator.predict(x_test)
        return pd.DataFrame(
            {
                "date": frame["date"].tolist(),
                "model": self.model_name,
                "prediction": np.asarray(raw_prediction, dtype=float),
                "target_return": frame[self.target_column].astype(float).tolist(),
                "objective": self.objective,
                "family": self.family,
                "adapter_status": self.adapter_status,
            }
        )

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "model_artifact.v1",
            "model_name": self.model_name,
            "feature_columns": list(self.feature_columns),
            "objective": self.objective,
            "family": self.family,
            "adapter_status": self.adapter_status,
            "target_column": self.target_column,
            **dict(self.metadata),
        }


def fit_model_artifact(
    model_name: str,
    *,
    train: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str = "target_return",
) -> ModelArtifact:
    if train.empty:
        raise ValueError("train frame must be non-empty")
    spec = get_model_spec(model_name)
    estimator = _estimator_for_model(model_name)
    x_train = _feature_matrix(train, feature_columns)
    constant_prediction: float | None = None

    if spec.objective == "classification":
        y_train = _classification_target(train)
        if len(set(y_train.tolist())) < 2:
            constant_prediction = float(y_train[0]) - 0.5
        else:
            estimator.fit(x_train, y_train)
    else:
        y_train = _regression_target(train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(x_train, y_train)

    return ModelArtifact(
        model_name=model_name,
        feature_columns=tuple(feature_columns),
        estimator=estimator,
        objective=spec.objective,
        family=spec.family,
        target_column=target_column,
        metadata={
            "schema_version": "model_artifact.v1",
            "train_rows": int(len(train)),
        },
        constant_prediction=constant_prediction,
    )


def save_model_artifact(path: str | Path, artifact: ModelArtifact) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return output


def load_model_artifact(path: str | Path) -> ModelArtifact:
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, ModelArtifact):
        raise TypeError("loaded object is not a ModelArtifact")
    return artifact

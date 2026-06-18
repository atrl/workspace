from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .registry import get_model_spec
from .training import (
    fit_estimator_for_model,
    model_adapter_name,
    model_adapter_status,
    predict_with_estimator,
)


@dataclass(frozen=True)
class ModelArtifact:
    model_name: str
    feature_columns: tuple[str, ...]
    estimator: Any
    objective: str
    family: str
    adapter_status: str = "smoke_adapter"
    model_adapter: str = "smoke_fallback_v1"
    target_column: str = "target_return"
    metadata: dict[str, Any] = field(default_factory=dict)
    constant_prediction: float | None = None

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("prediction frame must be non-empty")
        raw_prediction = predict_with_estimator(
            self.model_name,
            estimator=self.estimator,
            frame=frame,
            feature_columns=self.feature_columns,
            constant_prediction=self.constant_prediction,
        )
        return pd.DataFrame(
            {
                "date": frame["date"].tolist(),
                "model": self.model_name,
                "prediction": np.asarray(raw_prediction, dtype=float),
                "target_return": frame[self.target_column].astype(float).tolist(),
                "objective": self.objective,
                "family": self.family,
                "adapter_status": self.adapter_status,
                "model_adapter": self.model_adapter,
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
            "model_adapter": self.model_adapter,
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
    estimator, constant_prediction = fit_estimator_for_model(
        model_name,
        train=train,
        feature_columns=tuple(feature_columns),
    )

    return ModelArtifact(
        model_name=model_name,
        feature_columns=tuple(feature_columns),
        estimator=estimator,
        objective=spec.objective,
        family=spec.family,
        adapter_status=model_adapter_status(model_name),
        model_adapter=model_adapter_name(model_name),
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

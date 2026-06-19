from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    objective: str
    features: list[str]
    description: str


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate model spec: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ModelSpec:
        return self._specs[name]

    def list(self) -> list[ModelSpec]:
        return list(self._specs.values())


class MeanReturnModel:
    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, frame: pd.DataFrame, *, target_col: str) -> "MeanReturnModel":
        self._mean = float(frame[target_col].mean())
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self._mean is None:
            raise RuntimeError("model is not fitted")
        return pd.Series([self._mean] * len(frame), index=frame.index, dtype="float64")

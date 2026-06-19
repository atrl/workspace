from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    objective: str
    features: list[str]
    description: str
    capabilities: dict[str, bool] | None = None


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate model spec: {spec.name}")
        self._specs[spec.name] = spec

    @classmethod
    def from_specs(cls, specs: list[ModelSpec]) -> "ModelRegistry":
        registry = cls()
        for spec in specs:
            registry.register(spec)
        return registry

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

    def save(self, path: Path) -> None:
        if self._mean is None:
            raise RuntimeError("model is not fitted")
        Path(path).write_text(json.dumps({"model": "MeanReturnModel", "mean": self._mean}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MeanReturnModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model") != "MeanReturnModel":
            raise ValueError(f"unsupported model payload: {payload.get('model')}")
        model = cls()
        model._mean = float(payload["mean"])
        return model


def quantbench_supervised_model_specs() -> list[ModelSpec]:
    base_features = ["return_5d", "vix", "volume_zscore"]
    return [
        ModelSpec(
            name="linear_mse",
            family="linear",
            objective="mse",
            features=base_features,
            description="Linear regression baseline for continuous forward return.",
            capabilities={"supports_feature_importance": True, "supports_window_sensitivity": False},
        ),
        ModelSpec(
            name="linear_ic",
            family="linear",
            objective="ic",
            features=base_features,
            description="Linear model scored by information coefficient.",
            capabilities={"supports_feature_importance": True, "supports_window_sensitivity": False},
        ),
        ModelSpec(
            name="logistic_classification",
            family="linear",
            objective="classification",
            features=base_features,
            description="Directional classifier baseline.",
            capabilities={"supports_feature_importance": True, "supports_window_sensitivity": False},
        ),
        ModelSpec(
            name="fast_tree_rank",
            family="tree",
            objective="ranking",
            features=base_features + ["drawdown_20d"],
            description="Tree ranking candidate with feature importance.",
            capabilities={"supports_feature_importance": True, "supports_window_sensitivity": False},
        ),
        ModelSpec(
            name="gbdt_utility",
            family="tree",
            objective="utility",
            features=base_features + ["turnover_proxy"],
            description="Tree candidate scored by return and risk utility.",
            capabilities={"supports_feature_importance": True, "supports_window_sensitivity": False},
        ),
        ModelSpec(
            name="dlstm_mse",
            family="sequence",
            objective="mse",
            features=base_features + ["sequence_window"],
            description="Sequence model placeholder for rolling window return forecasting.",
            capabilities={"supports_feature_importance": False, "supports_window_sensitivity": True},
        ),
        ModelSpec(
            name="temporal_cnn_ic",
            family="sequence",
            objective="ic",
            features=base_features + ["sequence_window"],
            description="Temporal convolution candidate scored by IC.",
            capabilities={"supports_feature_importance": False, "supports_window_sensitivity": True},
        ),
        ModelSpec(
            name="crossformer_mse",
            family="transformer",
            objective="mse",
            features=base_features + ["multi_scale_window"],
            description="Transformer-style candidate for multi-scale windows.",
            capabilities={"supports_feature_importance": False, "supports_window_sensitivity": True},
        ),
    ]


def model_specs_to_dicts(specs: list[ModelSpec]) -> list[dict]:
    return [asdict(spec) for spec in specs]

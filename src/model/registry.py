from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.core import SCHEMA_VERSION


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

    def families(self) -> list[str]:
        return list(dict.fromkeys(spec.family for spec in self._specs.values()))

    def objectives(self) -> list[str]:
        return list(dict.fromkeys(spec.objective for spec in self._specs.values()))

    def by_family(self, family: str) -> list[ModelSpec]:
        return [spec for spec in self._specs.values() if spec.family == family]

    def by_objective(self, objective: str) -> list[ModelSpec]:
        return [spec for spec in self._specs.values() if spec.objective == objective]


class FitPredictModel(Protocol):
    def fit(self, frame: pd.DataFrame, *, target_col: str) -> "FitPredictModel":
        ...

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        ...


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


class QuantBenchBaselineModel:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._mean: float | None = None

    def fit(self, frame: pd.DataFrame, *, target_col: str) -> "QuantBenchBaselineModel":
        missing_features = [feature for feature in self.spec.features if feature not in frame.columns]
        if missing_features:
            raise ValueError(f"missing model features for {self.spec.name}: {missing_features}")
        if target_col not in frame.columns:
            raise ValueError(f"missing target column: {target_col}")
        self._mean = float(frame[target_col].mean())
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self._mean is None:
            raise RuntimeError("model is not fitted")
        missing_features = [feature for feature in self.spec.features if feature not in frame.columns]
        if missing_features:
            raise ValueError(f"missing model features for {self.spec.name}: {missing_features}")
        return pd.Series([self._mean] * len(frame), index=frame.index, dtype="float64", name=self.spec.name)

    def save(self, path: Path) -> None:
        if self._mean is None:
            raise RuntimeError("model is not fitted")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "model_class": "QuantBenchBaselineModel",
            "spec": asdict(self.spec),
            "mean": self._mean,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_model(spec: ModelSpec) -> FitPredictModel:
    return QuantBenchBaselineModel(spec)


def load_model(path: Path) -> QuantBenchBaselineModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("model_class") != "QuantBenchBaselineModel":
        raise ValueError(f"unsupported model payload: {payload.get('model_class')}")
    spec = ModelSpec(**payload["spec"])
    model = QuantBenchBaselineModel(spec)
    model._mean = float(payload["mean"])
    return model


def _spec(
    name: str,
    family: str,
    objective: str,
    features: list[str],
    description: str,
    **capabilities: bool,
) -> ModelSpec:
    default_capabilities = {
        "supports_feature_importance": family in {"linear", "tree", "kernel", "boosting", "ensemble"},
        "supports_window_sensitivity": family in {"sequence", "transformer", "graph", "ensemble"},
        "supports_graph_inputs": family == "graph",
    }
    default_capabilities.update(capabilities)
    return ModelSpec(
        name=name,
        family=family,
        objective=objective,
        features=features,
        description=description,
        capabilities=default_capabilities,
    )


def quantbench_supervised_model_specs() -> list[ModelSpec]:
    base_features = ["return_5d", "vix", "volume_zscore"]
    drawdown_features = base_features + ["drawdown_20d"]
    utility_features = base_features + ["turnover_proxy"]
    sequence_features = base_features + ["sequence_window"]
    transformer_features = base_features + ["multi_scale_window"]
    graph_features = base_features + ["graph_neighbor_return"]
    return [
        _spec("linear_mse", "linear", "mse", base_features, "Linear regression baseline for continuous forward return."),
        _spec("linear_ic", "linear", "ic", base_features, "Linear model scored by information coefficient."),
        _spec("linear_rank", "linear", "ranking", drawdown_features, "Linear ranker over forward-return labels."),
        _spec("logistic_classification", "linear", "classification", base_features, "Directional classifier baseline."),
        _spec("ridge_utility", "linear", "utility", utility_features, "Regularized linear utility candidate."),
        _spec("elasticnet_mse", "linear", "mse", base_features, "Elastic-net style continuous-return baseline."),
        _spec("fast_tree_rank", "tree", "ranking", drawdown_features, "Tree ranking candidate with feature importance."),
        _spec("random_forest_mse", "tree", "mse", drawdown_features, "Bagged tree return forecaster."),
        _spec("extra_trees_ic", "tree", "ic", drawdown_features, "Extra-trees candidate scored by IC."),
        _spec("decision_tree_classification", "tree", "classification", base_features, "Shallow tree directional classifier."),
        _spec("tree_utility", "tree", "utility", utility_features, "Tree candidate scored by return and risk utility."),
        _spec("monotonic_tree_rank", "tree", "ranking", drawdown_features, "Monotonic tree ranker for drawdown-aware labels."),
        _spec("svr_mse", "kernel", "mse", base_features, "Support-vector regression baseline."),
        _spec("kernel_ridge_ic", "kernel", "ic", base_features, "Kernel ridge candidate scored by IC."),
        _spec("svm_classification", "kernel", "classification", base_features, "Support-vector directional classifier."),
        _spec("gaussian_process_mse", "kernel", "mse", base_features, "Gaussian-process return smoother."),
        _spec("kernel_rank", "kernel", "ranking", drawdown_features, "Kernel ranker for cross-window ordering."),
        _spec("gbdt_utility", "boosting", "utility", utility_features, "GBDT candidate scored by return and risk utility."),
        _spec("lightgbm_mse", "boosting", "mse", drawdown_features, "LightGBM-style return forecaster."),
        _spec("lightgbm_rank", "boosting", "ranking", drawdown_features, "LightGBM-style ranker."),
        _spec("xgboost_ic", "boosting", "ic", drawdown_features, "XGBoost-style candidate scored by IC."),
        _spec("catboost_classification", "boosting", "classification", base_features, "CatBoost-style directional classifier."),
        _spec("hist_gbdt_utility", "boosting", "utility", utility_features, "Histogram GBDT utility candidate."),
        _spec("dart_rank", "boosting", "ranking", drawdown_features, "Dropout boosted-tree ranker."),
        _spec("dlstm_mse", "sequence", "mse", sequence_features, "Sequence model for rolling-window return forecasting."),
        _spec("gru_ic", "sequence", "ic", sequence_features, "GRU-style sequence model scored by IC."),
        _spec("temporal_cnn_ic", "sequence", "ic", sequence_features, "Temporal convolution candidate scored by IC."),
        _spec("tcn_classification", "sequence", "classification", sequence_features, "Temporal convolution directional classifier."),
        _spec("seq2seq_rank", "sequence", "ranking", sequence_features, "Sequence-to-sequence ranker."),
        _spec("nhits_utility", "sequence", "utility", sequence_features + ["turnover_proxy"], "N-HiTS-style utility candidate."),
        _spec("crossformer_mse", "transformer", "mse", transformer_features, "Transformer-style candidate for multi-scale windows."),
        _spec("temporal_fusion_transformer_ic", "transformer", "ic", transformer_features, "Temporal fusion candidate scored by IC."),
        _spec("informer_rank", "transformer", "ranking", transformer_features, "Informer-style ranker."),
        _spec("patchtst_utility", "transformer", "utility", transformer_features + ["turnover_proxy"], "PatchTST utility candidate."),
        _spec("timesnet_classification", "transformer", "classification", transformer_features, "TimesNet-style directional classifier."),
        _spec("graphsage_ic", "graph", "ic", graph_features, "GraphSAGE-style candidate using related-instrument context."),
        _spec("gat_rank", "graph", "ranking", graph_features + ["drawdown_20d"], "Graph-attention ranker."),
        _spec("graph_transformer_mse", "graph", "mse", graph_features, "Graph transformer return forecaster."),
        _spec("temporal_graph_utility", "graph", "utility", graph_features + ["turnover_proxy"], "Temporal graph utility candidate."),
        _spec("graph_classifier", "graph", "classification", graph_features, "Graph directional classifier."),
        _spec("stacking_utility", "ensemble", "utility", utility_features, "Stacked ensemble scored by promotion utility."),
        _spec("voting_classification", "ensemble", "classification", base_features, "Voting ensemble for directional labels."),
        _spec("blending_mse", "ensemble", "mse", base_features, "Blended return forecaster."),
        _spec("rank_ensemble_ic", "ensemble", "ic", drawdown_features, "Rank ensemble scored by IC."),
    ]


def model_specs_to_dicts(specs: list[ModelSpec]) -> list[dict]:
    return [asdict(spec) for spec in specs]

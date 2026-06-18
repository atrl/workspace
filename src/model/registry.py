from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    objective: str
    input_kind: str
    estimator: str
    trade_instrument: str = "SOXL"
    support_universe: tuple[str, ...] = ("SOXL", "SOXX", "VIX", "QQQ", "SPY")
    paper_fidelity: str = "paper_oriented"


def _spec(name: str, family: str, objective: str, input_kind: str, estimator: str) -> ModelSpec:
    return ModelSpec(name=name, family=family, objective=objective, input_kind=input_kind, estimator=estimator)


MODEL_SPECS: tuple[ModelSpec, ...] = (
    _spec("logistic_clf", "linear", "classification", "tabular", "LogisticRegression"),
    _spec("ridge_ic", "linear", "ic", "tabular", "Ridge"),
    _spec("ridge_mse", "linear", "mse", "tabular", "Ridge"),
    _spec("random_forest_clf", "tree", "classification", "tabular", "RandomForestClassifier"),
    _spec("hist_gradient_mse", "tree", "mse", "tabular", "HistGradientBoostingRegressor"),
    _spec("xgboost_mse", "boosting", "mse", "tabular", "XGBoostRegressor"),
    _spec("lightgbm_mse", "boosting", "mse", "tabular", "LightGBMRegressor"),
    _spec("catboost_rank", "boosting", "ranking", "tabular", "CatBoostRanker"),
    _spec("extra_trees_rank", "tree", "ranking", "tabular", "ExtraTreesRegressor"),
    _spec("mlp_mse", "neural_tabular", "mse", "tabular", "MLPRegressor"),
    _spec("sequence_lag_ridge_mse", "sequence", "mse", "lagged_sequence", "LaggedRidge"),
    _spec("lstm_mse", "sequence", "mse", "sequence", "LSTM"),
    _spec("gru_mse", "sequence", "mse", "sequence", "GRU"),
    _spec("alstm_mse", "sequence", "mse", "sequence", "ALSTM"),
    _spec("sfm_mse", "sequence", "mse", "sequence_frequency", "SFM"),
    _spec("darnn_mse", "sequence", "mse", "sequence_attention", "DA-RNN"),
    _spec("hawkes_gru_mse", "sequence", "mse", "event_sequence", "HawkesGRU"),
    _spec("tcn_mse", "sequence", "mse", "sequence", "TCN"),
    _spec("mlp_mixer_mse", "sequence", "mse", "sequence_patch", "MLPMixer"),
    _spec("dlinear_mse", "sequence", "mse", "sequence_decomposition", "DLinear"),
    _spec("patch_tst_mse", "sequence", "mse", "sequence_patch", "PatchTST"),
    _spec("autoformer_mse", "sequence", "mse", "sequence_decomposition", "Autoformer"),
    _spec("fedformer_mse", "sequence", "mse", "sequence_frequency", "FEDformer"),
    _spec("market_graph_mse", "graph", "mse", "support_graph", "MarketGraph"),
    _spec("gcn_mse", "graph", "mse", "support_graph", "GCN"),
    _spec("gat_mse", "graph", "mse", "support_graph", "GAT"),
    _spec("rgcn_mse", "graph", "mse", "support_graph", "RGCN"),
    _spec("rsr_mse", "graph", "mse", "support_graph", "RSR"),
    _spec("hypergraph_mse", "graph", "mse", "support_hypergraph", "HypergraphNN"),
    _spec("thgnn_mse", "graph", "mse", "temporal_hypergraph", "THGNN"),
    _spec("sthan_mse", "graph", "mse", "spatio_temporal_graph", "STHAN"),
    _spec("sthgcn_mse", "graph", "mse", "spatio_temporal_hypergraph", "STHGCN"),
)

QUANTBENCH_MODEL_NAMES: tuple[str, ...] = tuple(spec.name for spec in MODEL_SPECS)


def iter_model_specs() -> Iterable[ModelSpec]:
    return iter(MODEL_SPECS)


def get_model_spec(name: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown model spec: {name}")

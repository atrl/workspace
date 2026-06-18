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
    paper_fidelity: str
    implementation_notes: str
    promotion_blocker: str = ""
    trade_instrument: str = "SOXL"
    support_universe: tuple[str, ...] = ("SOXL", "SOXX", "VIX", "QQQ", "SPY")


def _spec(
    name: str,
    family: str,
    objective: str,
    input_kind: str,
    estimator: str,
    paper_fidelity: str,
    implementation_notes: str,
    promotion_blocker: str = "",
) -> ModelSpec:
    return ModelSpec(
        name=name,
        family=family,
        objective=objective,
        input_kind=input_kind,
        estimator=estimator,
        paper_fidelity=paper_fidelity,
        implementation_notes=implementation_notes,
        promotion_blocker=promotion_blocker,
    )


def _native(name: str, family: str, objective: str, input_kind: str, estimator: str) -> ModelSpec:
    return _spec(
        name,
        family,
        objective,
        input_kind,
        estimator,
        "native_library",
        f"Uses the native {estimator} training path behind the QuantBench-style registry contract.",
    )


def _proxy(
    name: str,
    family: str,
    objective: str,
    input_kind: str,
    estimator: str,
    notes: str,
) -> ModelSpec:
    return _spec(
        name,
        family,
        objective,
        input_kind,
        estimator,
        "proxy_feature_adapter",
        notes,
        "Requires full paper-fidelity architecture replacement before production promotion.",
    )


def _torch_native(name: str, family: str, objective: str, input_kind: str, estimator: str, notes: str) -> ModelSpec:
    return _spec(
        name,
        family,
        objective,
        input_kind,
        estimator,
        "native_torch",
        notes,
        "Needs longer-window robustness review before production promotion.",
    )


def _native_sequence(name: str, family: str, objective: str, input_kind: str, estimator: str, notes: str) -> ModelSpec:
    return _spec(
        name,
        family,
        objective,
        input_kind,
        estimator,
        "native_sequence",
        notes,
        "Needs longer-window robustness review before production promotion.",
    )


MODEL_SPECS: tuple[ModelSpec, ...] = (
    _native("logistic_clf", "linear", "classification", "tabular", "LogisticRegression"),
    _proxy("ridge_ic", "linear", "ic", "tabular", "Ridge", "Ridge is trained on an IC-style rank target proxy."),
    _native("ridge_mse", "linear", "mse", "tabular", "Ridge"),
    _native("random_forest_clf", "tree", "classification", "tabular", "RandomForestClassifier"),
    _native("hist_gradient_mse", "tree", "mse", "tabular", "HistGradientBoostingRegressor"),
    _native("xgboost_mse", "boosting", "mse", "tabular", "XGBoostRegressor"),
    _native("lightgbm_mse", "boosting", "mse", "tabular", "LightGBMRegressor"),
    _native("catboost_rank", "boosting", "ranking", "tabular", "CatBoostRanker"),
    _proxy(
        "extra_trees_rank",
        "tree",
        "ranking",
        "tabular",
        "ExtraTreesRegressor",
        "ExtraTrees is trained on an IC-style rank target proxy.",
    ),
    _native("mlp_mse", "neural_tabular", "mse", "tabular", "MLPRegressor"),
    _proxy(
        "sequence_lag_ridge_mse",
        "sequence",
        "mse",
        "lagged_sequence",
        "LaggedRidge",
        "Lagged sequence ridge keeps the sequence contract while acting as a linear baseline.",
    ),
    _torch_native("lstm_mse", "sequence", "mse", "sequence", "LSTM", "Uses a compact PyTorch LSTM regressor."),
    _torch_native("gru_mse", "sequence", "mse", "sequence", "GRU", "Uses a compact PyTorch GRU regressor."),
    _torch_native(
        "alstm_mse",
        "sequence",
        "mse",
        "sequence",
        "ALSTM",
        "Uses a compact PyTorch LSTM with temporal attention.",
    ),
    _proxy("sfm_mse", "sequence", "mse", "sequence_frequency", "SFM", "Uses frequency-domain ridge features."),
    _torch_native(
        "darnn_mse",
        "sequence",
        "mse",
        "sequence_attention",
        "DA-RNN",
        "Uses a compact attention GRU with an input gate.",
    ),
    _proxy("hawkes_gru_mse", "sequence", "mse", "event_sequence", "HawkesGRU", "Uses event-decay ridge features."),
    _torch_native("tcn_mse", "sequence", "mse", "sequence", "TCN", "Uses a compact PyTorch temporal convolution net."),
    _proxy("mlp_mixer_mse", "sequence", "mse", "sequence_patch", "MLPMixer", "Uses mixer-style engineered features."),
    _native_sequence(
        "dlinear_mse",
        "sequence",
        "mse",
        "sequence_decomposition",
        "DLinear",
        "Uses an explicit moving-average decomposition with separate linear trend and residual heads.",
    ),
    _native_sequence(
        "patch_tst_mse",
        "sequence",
        "mse",
        "sequence_patch",
        "PatchTST",
        "Uses a compact patch Transformer encoder over channel-wise sequence patches.",
    ),
    _native_sequence(
        "autoformer_mse",
        "sequence",
        "mse",
        "sequence_decomposition",
        "Autoformer",
        "Uses moving-average decomposition with autocorrelation lag aggregation and separate trend/seasonal heads.",
    ),
    _native_sequence(
        "fedformer_mse",
        "sequence",
        "mse",
        "sequence_frequency",
        "FEDformer",
        "Uses moving-average decomposition with sparse Fourier mode features and separate trend/frequency heads.",
    ),
    _proxy("market_graph_mse", "graph", "mse", "support_graph", "MarketGraph", "Uses market-context graph features."),
    _proxy("gcn_mse", "graph", "mse", "support_graph", "GCN", "Uses feature propagation over a fixed feature graph."),
    _proxy("gat_mse", "graph", "mse", "support_graph", "GAT", "Uses attention-weighted feature graph statistics."),
    _proxy("rgcn_mse", "graph", "mse", "support_graph", "RGCN", "Uses relation-group feature aggregation."),
    _proxy("rsr_mse", "graph", "mse", "support_graph", "RSR", "Uses relation-strength ridge features."),
    _proxy("hypergraph_mse", "graph", "mse", "support_hypergraph", "HypergraphNN", "Uses hypergraph-style grouped feature statistics."),
    _proxy("thgnn_mse", "graph", "mse", "temporal_hypergraph", "THGNN", "Uses hypergraph features with temporal lags."),
    _proxy("sthan_mse", "graph", "mse", "spatio_temporal_graph", "STHAN", "Uses spatio-temporal attention proxy features."),
    _proxy(
        "sthgcn_mse",
        "graph",
        "mse",
        "spatio_temporal_hypergraph",
        "STHGCN",
        "Uses spatio-temporal hypergraph proxy features.",
    ),
)

QUANTBENCH_MODEL_NAMES: tuple[str, ...] = tuple(spec.name for spec in MODEL_SPECS)


def iter_model_specs() -> Iterable[ModelSpec]:
    return iter(MODEL_SPECS)


def get_model_spec(name: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown model spec: {name}")

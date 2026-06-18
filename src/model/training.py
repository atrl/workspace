from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Import before sequence/torch modules; torch-first initialization can segfault xgboost 3.2.0 on this arm64 stack.
from xgboost import XGBRegressor

from .graph import GraphFeatureRidgeRegressor
from .registry import get_model_spec
from .sequence import (
    AutoformerDecompositionRegressor,
    DLinearDecompositionRegressor,
    FEDformerFrequencyRegressor,
    FeatureRidgeRegressor,
    FrequencyRidgeRegressor,
    LaggedRidgeRegressor,
    MLPMixerSequenceRegressor,
    PatchTSTTransformerRegressor,
    TorchSequenceRegressor,
    build_sequence_tensor,
)


STRICT_ADAPTER_NAMES = {
    "logistic_clf": "logistic_direction_v1",
    "ridge_mse": "ridge_mse_v1",
    "ridge_ic": "ridge_rank_ic_proxy_v1",
    "random_forest_clf": "random_forest_direction_v1",
    "hist_gradient_mse": "hist_gradient_mse_v1",
    "xgboost_mse": "xgboost_mse_v1",
    "extra_trees_rank": "extra_trees_rank_proxy_v1",
    "lightgbm_mse": "lightgbm_mse_v1",
    "catboost_rank": "catboost_rank_v1",
    "mlp_mse": "mlp_sklearn_mse_v1",
    "sequence_lag_ridge_mse": "sequence_lag_ridge_mse_v1",
    "lstm_mse": "lstm_pytorch_sequence_mse_v1",
    "gru_mse": "gru_pytorch_sequence_mse_v1",
    "alstm_mse": "alstm_attention_lstm_mse_v1",
    "sfm_mse": "sfm_frequency_ridge_mse_v1",
    "darnn_mse": "darnn_dual_attention_gru_mse_v1",
    "tcn_mse": "tcn_temporal_conv_mse_v1",
    "hawkes_gru_mse": "hawkes_gru_event_decay_mse_v1",
    "mlp_mixer_mse": "mlp_mixer_sequence_mse_v1",
    "dlinear_mse": "dlinear_decomposition_mse_v1",
    "patch_tst_mse": "patch_tst_transformer_mse_v1",
    "autoformer_mse": "autoformer_decomposition_attention_mse_v1",
    "fedformer_mse": "fedformer_frequency_attention_mse_v1",
    "market_graph_mse": "market_graph_feature_mse_v1",
    "gcn_mse": "gcn_feature_propagation_mse_v1",
    "gat_mse": "gat_feature_attention_mse_v1",
    "rgcn_mse": "rgcn_relation_feature_mse_v1",
    "rsr_mse": "rsr_relation_strength_mse_v1",
    "hypergraph_mse": "hypergraph_feature_mse_v1",
    "thgnn_mse": "thgnn_temporal_hypergraph_mse_v1",
    "sthan_mse": "sthan_spatio_temporal_attention_mse_v1",
    "sthgcn_mse": "sthgcn_spatio_temporal_hypergraph_mse_v1",
}

DEPENDENCY_FALLBACK_ADAPTER_NAMES: dict[str, str] = {}


def model_adapter_name(model_name: str) -> str:
    if model_name in STRICT_ADAPTER_NAMES:
        return STRICT_ADAPTER_NAMES[model_name]
    if model_name in DEPENDENCY_FALLBACK_ADAPTER_NAMES:
        return DEPENDENCY_FALLBACK_ADAPTER_NAMES[model_name]
    return "smoke_fallback_v1"


def model_adapter_status(model_name: str) -> str:
    if model_name in STRICT_ADAPTER_NAMES:
        return "paper_strict_adapter"
    if model_name in DEPENDENCY_FALLBACK_ADAPTER_NAMES:
        return "dependency_fallback_adapter"
    return "smoke_adapter"


def _feature_matrix(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    return np.ascontiguousarray(frame.loc[:, list(feature_columns)].astype(float).to_numpy(), dtype=float)


def _feature_frame(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[:, list(feature_columns)].astype(float)


def _sequence_feature_tensor(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    return build_sequence_tensor(_feature_matrix(frame, feature_columns), lookback=5)


def _uses_sequence_tensor(model_name: str) -> bool:
    return model_name in {
        "sequence_lag_ridge_mse",
        "lstm_mse",
        "gru_mse",
        "alstm_mse",
        "sfm_mse",
        "darnn_mse",
        "tcn_mse",
        "hawkes_gru_mse",
        "mlp_mixer_mse",
        "dlinear_mse",
        "patch_tst_mse",
        "autoformer_mse",
        "fedformer_mse",
    }


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
    if model_name == "ridge_ic" or spec.objective == "ranking":
        return _ic_rank_target(frame)
    return _regression_target(frame)


def _lightgbm_regressor():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=7,
        n_jobs=1,
        verbosity=-1,
    )


def _xgboost_regressor():
    return XGBRegressor(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=7,
        n_jobs=1,
        verbosity=0,
    )


class CatBoostRankerAdapter:
    def __init__(
        self,
        *,
        iterations: int = 60,
        depth: int = 3,
        learning_rate: float = 0.05,
        random_seed: int = 7,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.model = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        from catboost import CatBoostRanker, Pool

        group_id = np.zeros(len(y_train), dtype=np.int64)
        self.model = CatBoostRanker(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="YetiRank",
            random_seed=self.random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        self.model.fit(Pool(x_train, y_train, group_id=group_id))
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CatBoostRankerAdapter must be fitted before predict")
        return np.asarray(self.model.predict(x_test), dtype=float)


def _estimator_for_model(model_name: str):
    spec = get_model_spec(model_name)
    if model_name == "logistic_clf":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, random_state=7),
        )
    if model_name in {"ridge_mse", "ridge_ic"}:
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_name == "random_forest_clf":
        return RandomForestClassifier(n_estimators=48, max_depth=5, min_samples_leaf=3, random_state=7)
    if model_name == "hist_gradient_mse":
        return HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, random_state=7)
    if model_name == "xgboost_mse":
        return _xgboost_regressor()
    if model_name == "lightgbm_mse":
        return _lightgbm_regressor()
    if model_name == "catboost_rank":
        return CatBoostRankerAdapter()
    if model_name == "mlp_mse":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(24, 12),
                activation="relu",
                alpha=1e-4,
                max_iter=300,
                random_state=7,
                early_stopping=True,
            ),
        )
    if model_name == "sequence_lag_ridge_mse":
        return LaggedRidgeRegressor(alpha=1.0)
    if model_name == "lstm_mse":
        return TorchSequenceRegressor(cell_type="lstm", hidden_size=8, epochs=24, learning_rate=0.01, seed=7)
    if model_name == "gru_mse":
        return TorchSequenceRegressor(cell_type="gru", hidden_size=8, epochs=24, learning_rate=0.01, seed=11)
    if model_name == "alstm_mse":
        return TorchSequenceRegressor(
            cell_type="lstm",
            hidden_size=8,
            epochs=24,
            learning_rate=0.01,
            seed=13,
            architecture="attention",
        )
    if model_name == "sfm_mse":
        return FrequencyRidgeRegressor(alpha=1.0)
    if model_name == "darnn_mse":
        return TorchSequenceRegressor(
            cell_type="gru",
            hidden_size=8,
            epochs=24,
            learning_rate=0.01,
            seed=17,
            architecture="attention",
            use_input_gate=True,
        )
    if model_name == "tcn_mse":
        return TorchSequenceRegressor(
            cell_type="gru",
            hidden_size=8,
            epochs=24,
            learning_rate=0.01,
            seed=19,
            architecture="tcn",
        )
    if model_name == "hawkes_gru_mse":
        return FeatureRidgeRegressor(feature_kind="event_decay", alpha=1.0)
    if model_name == "mlp_mixer_mse":
        return MLPMixerSequenceRegressor(
            token_hidden_size=8,
            channel_hidden_size=16,
            num_blocks=1,
            epochs=24,
            learning_rate=0.01,
            seed=29,
        )
    if model_name == "dlinear_mse":
        return DLinearDecompositionRegressor(alpha=1.0, kernel_size=3)
    if model_name == "patch_tst_mse":
        return PatchTSTTransformerRegressor(patch_size=2, stride=1, d_model=8, nhead=2, num_layers=1, epochs=24)
    if model_name == "autoformer_mse":
        return AutoformerDecompositionRegressor(alpha=1.0, kernel_size=3, max_lag=4, top_k=2)
    if model_name == "fedformer_mse":
        return FEDformerFrequencyRegressor(alpha=1.0, kernel_size=3, top_k_modes=2)
    graph_kinds = {
        "market_graph_mse": "market",
        "gcn_mse": "gcn",
        "gat_mse": "gat",
        "rgcn_mse": "rgcn",
        "rsr_mse": "rsr",
        "hypergraph_mse": "hypergraph",
        "thgnn_mse": "thgnn",
        "sthan_mse": "sthan",
        "sthgcn_mse": "sthgcn",
    }
    if model_name in graph_kinds:
        return GraphFeatureRidgeRegressor(graph_kind=graph_kinds[model_name], alpha=1.0)
    if model_name == "extra_trees_rank":
        return ExtraTreesRegressor(n_estimators=48, max_depth=5, min_samples_leaf=3, random_state=7)
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
    if _uses_sequence_tensor(model_name):
        x_train = _sequence_feature_tensor(train, feature_columns)
    elif model_name == "lightgbm_mse":
        x_train = _feature_frame(train, feature_columns)
    else:
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
    if _uses_sequence_tensor(model_name):
        x_test = _sequence_feature_tensor(frame, feature_columns)
    elif model_name == "lightgbm_mse":
        x_test = _feature_frame(frame, feature_columns)
    else:
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

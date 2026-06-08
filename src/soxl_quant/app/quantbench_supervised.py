from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .quantbench_objectives import evaluate_prediction_objectives
from .quantbench_screen import QuantBenchGate, screen_quantbench_candidates, summary_record_from_payload


@dataclass(frozen=True)
class SupervisedModelSpec:
    name: str
    paper_family: str
    objective: str
    target_kind: str
    estimator: str


@dataclass(frozen=True)
class RiskAwarePositionConfig:
    name: str
    defensive_cap: float = 0.45
    stress_cap: float = 0.20
    crash_cap: float = 0.12
    bull_floor: float = 0.55
    bull_cap: float = 0.90
    defensive_risk_threshold: float = 0.55
    stress_risk_threshold: float = 0.70
    crash_drop_threshold: float = -0.20
    crash_drawdown_threshold: float = -0.25
    bull_risk_threshold: float = 0.40
    bull_momentum_threshold: float = 0.03
    bull_trend_threshold: float = 0.00
    bull_drawdown_floor: float = -0.08
    portfolio_drawdown_trigger: float = 0.0
    portfolio_drawdown_cap: float = 0.08
    portfolio_drawdown_release: float = 0.06
    reentry_min_target: float = 0.0
    reentry_max_target: float = 0.55
    reentry_risk_threshold: float = 0.35
    reentry_momentum_threshold: float = 0.05
    reentry_trend_threshold: float = 0.02
    reentry_drawdown_floor: float = -0.10


QUANTBENCH_SUPERVISED_MODEL_SPECS: tuple[SupervisedModelSpec, ...] = (
    SupervisedModelSpec(
        name="logistic_clf",
        paper_family="linear_classifier",
        objective="clf",
        target_kind="binary",
        estimator="LogisticRegression",
    ),
    SupervisedModelSpec(
        name="ridge_ic",
        paper_family="linear_regression",
        objective="ic",
        target_kind="return",
        estimator="Ridge",
    ),
    SupervisedModelSpec(
        name="ridge_mse",
        paper_family="linear_regression",
        objective="mse",
        target_kind="return",
        estimator="Ridge",
    ),
    SupervisedModelSpec(
        name="random_forest_clf",
        paper_family="tree",
        objective="clf",
        target_kind="binary",
        estimator="RandomForestClassifier",
    ),
    SupervisedModelSpec(
        name="hist_gradient_mse",
        paper_family="boosted_tree",
        objective="mse",
        target_kind="return",
        estimator="HistGradientBoostingRegressor",
    ),
    SupervisedModelSpec(
        name="xgboost_mse",
        paper_family="boosted_tree_xgboost",
        objective="mse",
        target_kind="return",
        estimator="XGBoostRegressorOrHistGradientFallback",
    ),
    SupervisedModelSpec(
        name="lightgbm_mse",
        paper_family="boosted_tree_lightgbm",
        objective="mse",
        target_kind="return",
        estimator="LightGBMRegressorOrHistGradientFallback",
    ),
    SupervisedModelSpec(
        name="catboost_rank",
        paper_family="boosted_tree_catboost",
        objective="ranking",
        target_kind="return",
        estimator="CatBoostRegressorOrExtraTreesFallback",
    ),
    SupervisedModelSpec(
        name="extra_trees_rank",
        paper_family="tree_ensemble",
        objective="ranking",
        target_kind="return",
        estimator="ExtraTreesRegressor",
    ),
    SupervisedModelSpec(
        name="mlp_mse",
        paper_family="neural_tabular",
        objective="mse",
        target_kind="return",
        estimator="MLPRegressor",
    ),
    SupervisedModelSpec(
        name="sequence_lag_ridge_mse",
        paper_family="sequence_temporal",
        objective="mse",
        target_kind="return",
        estimator="LaggedRidge",
    ),
    SupervisedModelSpec(
        name="lstm_mse",
        paper_family="sequence_recurrent",
        objective="mse",
        target_kind="return",
        estimator="TorchLSTM",
    ),
    SupervisedModelSpec(
        name="gru_mse",
        paper_family="sequence_recurrent",
        objective="mse",
        target_kind="return",
        estimator="TorchGRU",
    ),
    SupervisedModelSpec(
        name="alstm_mse",
        paper_family="sequence_attention_recurrent",
        objective="mse",
        target_kind="return",
        estimator="TorchAttentionLSTM",
    ),
    SupervisedModelSpec(
        name="sfm_mse",
        paper_family="sequence_frequency",
        objective="mse",
        target_kind="return",
        estimator="TorchFrequencyMLP",
    ),
    SupervisedModelSpec(
        name="darnn_mse",
        paper_family="sequence_dual_attention",
        objective="mse",
        target_kind="return",
        estimator="TorchDualAttentionRNN",
    ),
    SupervisedModelSpec(
        name="hawkes_gru_mse",
        paper_family="sequence_hawkes_recurrent",
        objective="mse",
        target_kind="return",
        estimator="TorchHawkesGRU",
    ),
    SupervisedModelSpec(
        name="tcn_mse",
        paper_family="sequence_convolutional",
        objective="mse",
        target_kind="return",
        estimator="TorchTemporalConv",
    ),
    SupervisedModelSpec(
        name="mlp_mixer_mse",
        paper_family="sequence_mlp_mixer",
        objective="mse",
        target_kind="return",
        estimator="TorchMLPMixer",
    ),
    SupervisedModelSpec(
        name="dlinear_mse",
        paper_family="sequence_linear_decomposition",
        objective="mse",
        target_kind="return",
        estimator="TorchDLinear",
    ),
    SupervisedModelSpec(
        name="patch_tst_mse",
        paper_family="sequence_patch_transformer",
        objective="mse",
        target_kind="return",
        estimator="TorchPatchTST",
    ),
    SupervisedModelSpec(
        name="autoformer_mse",
        paper_family="sequence_auto_correlation",
        objective="mse",
        target_kind="return",
        estimator="TorchAutoformerLite",
    ),
    SupervisedModelSpec(
        name="fedformer_mse",
        paper_family="sequence_frequency_transformer",
        objective="mse",
        target_kind="return",
        estimator="TorchFEDformerLite",
    ),
    SupervisedModelSpec(
        name="market_graph_mse",
        paper_family="relational_graph",
        objective="mse",
        target_kind="return",
        estimator="MarketGraphRidge",
    ),
    SupervisedModelSpec(
        name="gcn_mse",
        paper_family="relational_gcn",
        objective="mse",
        target_kind="return",
        estimator="GraphConvolutionRidge",
    ),
    SupervisedModelSpec(
        name="gat_mse",
        paper_family="relational_gat",
        objective="mse",
        target_kind="return",
        estimator="GraphAttentionRidge",
    ),
    SupervisedModelSpec(
        name="rgcn_mse",
        paper_family="relational_rgcn",
        objective="mse",
        target_kind="return",
        estimator="RelationalGraphRidge",
    ),
    SupervisedModelSpec(
        name="rsr_mse",
        paper_family="relational_rsr",
        objective="mse",
        target_kind="return",
        estimator="RelationStockRankRidge",
    ),
    SupervisedModelSpec(
        name="hypergraph_mse",
        paper_family="relational_hypergraph",
        objective="mse",
        target_kind="return",
        estimator="HypergraphRidge",
    ),
    SupervisedModelSpec(
        name="thgnn_mse",
        paper_family="relational_temporal_hypergraph",
        objective="mse",
        target_kind="return",
        estimator="TemporalHypergraphRidge",
    ),
    SupervisedModelSpec(
        name="sthan_mse",
        paper_family="relational_temporal_hypergraph_attention",
        objective="mse",
        target_kind="return",
        estimator="SpatioTemporalHypergraphAttentionRidge",
    ),
    SupervisedModelSpec(
        name="sthgcn_mse",
        paper_family="relational_temporal_hypergraph_convolution",
        objective="mse",
        target_kind="return",
        estimator="SpatioTemporalHypergraphConvolutionRidge",
    ),
)


def _sequence_shape_for_lagged_features(x: pd.DataFrame) -> tuple[int, int]:
    lag_numbers = []
    base_names = []
    for column in x.columns:
        name = str(column)
        if "__lag_" not in name:
            raise ValueError("sequence model features must be lag-expanded")
        base, lag = name.rsplit("__lag_", 1)
        lag_numbers.append(int(lag))
        base_names.append(base)
    lookback = max(lag_numbers) + 1
    feature_count = len(tuple(dict.fromkeys(base_names)))
    if lookback * feature_count != len(x.columns):
        raise ValueError("sequence model lag feature layout is not rectangular")
    return lookback, feature_count


def _reshape_lagged_features(x: pd.DataFrame) -> np.ndarray:
    lookback, feature_count = _sequence_shape_for_lagged_features(x)
    arr = x.to_numpy(dtype=np.float32, copy=True).reshape(len(x), lookback, feature_count)
    return arr[:, ::-1, :]


class _TorchLSTMRegressor:
    """Small deterministic LSTM regressor with a sklearn-like fit/predict API."""

    def __init__(
        self,
        *,
        hidden_size: int = 8,
        epochs: int = 24,
        learning_rate: float = 0.01,
        random_state: int = 0,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)

    def _reshape_features(self, x: pd.DataFrame) -> np.ndarray:
        return _reshape_lagged_features(x)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        train_x = self._reshape_features(x)
        train_y = y.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        self.x_mean_ = train_x.mean(axis=(0, 1), keepdims=True)
        self.x_std_ = train_x.std(axis=(0, 1), keepdims=True)
        self.x_std_[self.x_std_ < 1e-6] = 1.0
        self.y_mean_ = float(train_y.mean())
        self.y_std_ = float(train_y.std())
        if self.y_std_ < 1e-6:
            self.y_std_ = 1.0
        train_x = (train_x - self.x_mean_) / self.x_std_
        train_y = (train_y - self.y_mean_) / self.y_std_
        x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
        y_tensor = torch.as_tensor(train_y, dtype=torch.float32)

        class _Net(nn.Module):
            def __init__(self, feature_count: int, hidden_size: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size=feature_count, hidden_size=hidden_size, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, values):
                output, _ = self.lstm(values)
                return self.head(output[:, -1, :])

        self.model_ = _Net(int(train_x.shape[-1]), self.hidden_size)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(max(self.epochs, 1)):
            optimizer.zero_grad()
            loss = loss_fn(self.model_(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        import torch

        values = (self._reshape_features(x) - self.x_mean_) / self.x_std_
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.as_tensor(values, dtype=torch.float32)).numpy().reshape(-1)
        return pred * self.y_std_ + self.y_mean_


class _TorchTemporalConvRegressor:
    """Small TCN-style sequence regressor with a sklearn-like fit/predict API."""

    def __init__(
        self,
        *,
        hidden_size: int = 8,
        epochs: int = 24,
        learning_rate: float = 0.01,
        random_state: int = 0,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)

    def _reshape_features(self, x: pd.DataFrame) -> np.ndarray:
        return _reshape_lagged_features(x)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        train_x = self._reshape_features(x)
        train_y = y.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        self.x_mean_ = train_x.mean(axis=(0, 1), keepdims=True)
        self.x_std_ = train_x.std(axis=(0, 1), keepdims=True)
        self.x_std_[self.x_std_ < 1e-6] = 1.0
        self.y_mean_ = float(train_y.mean())
        self.y_std_ = float(train_y.std())
        if self.y_std_ < 1e-6:
            self.y_std_ = 1.0
        train_x = (train_x - self.x_mean_) / self.x_std_
        train_y = (train_y - self.y_mean_) / self.y_std_
        x_tensor = torch.as_tensor(np.transpose(train_x, (0, 2, 1)), dtype=torch.float32)
        y_tensor = torch.as_tensor(train_y, dtype=torch.float32)

        class _Net(nn.Module):
            def __init__(self, feature_count: int, hidden_size: int) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv1d(feature_count, hidden_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, dilation=2),
                    nn.ReLU(),
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, values):
                encoded = self.layers(values)
                return self.head(encoded[:, :, -1])

        self.model_ = _Net(int(train_x.shape[-1]), self.hidden_size)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(max(self.epochs, 1)):
            optimizer.zero_grad()
            loss = loss_fn(self.model_(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        import torch

        values = (self._reshape_features(x) - self.x_mean_) / self.x_std_
        values = np.transpose(values, (0, 2, 1))
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.as_tensor(values, dtype=torch.float32)).numpy().reshape(-1)
        return pred * self.y_std_ + self.y_mean_


class _TorchSequenceArchitectureRegressor:
    """Lightweight paper-family sequence regressor with a sklearn-like API."""

    def __init__(
        self,
        *,
        architecture: str,
        hidden_size: int = 8,
        epochs: int = 12,
        learning_rate: float = 0.01,
        random_state: int = 0,
    ) -> None:
        self.architecture = str(architecture)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)

    def _reshape_features(self, x: pd.DataFrame) -> np.ndarray:
        return _reshape_lagged_features(x)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        train_x = self._reshape_features(x)
        train_y = y.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        self.x_mean_ = train_x.mean(axis=(0, 1), keepdims=True)
        self.x_std_ = train_x.std(axis=(0, 1), keepdims=True)
        self.x_std_[self.x_std_ < 1e-6] = 1.0
        self.y_mean_ = float(train_y.mean())
        self.y_std_ = float(train_y.std())
        if self.y_std_ < 1e-6:
            self.y_std_ = 1.0
        train_x = (train_x - self.x_mean_) / self.x_std_
        train_y = (train_y - self.y_mean_) / self.y_std_
        x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
        y_tensor = torch.as_tensor(train_y, dtype=torch.float32)

        class _Net(nn.Module):
            def __init__(self, architecture: str, lookback: int, feature_count: int, hidden_size: int) -> None:
                super().__init__()
                self.architecture = architecture
                self.lookback = int(lookback)
                self.feature_count = int(feature_count)
                self.hidden_size = int(hidden_size)
                if architecture in {"gru", "hawkes_gru"}:
                    self.encoder = nn.GRU(input_size=feature_count, hidden_size=hidden_size, batch_first=True)
                    self.head = nn.Linear(hidden_size, 1)
                elif architecture == "alstm":
                    self.encoder = nn.LSTM(input_size=feature_count, hidden_size=hidden_size, batch_first=True)
                    self.attention = nn.Linear(hidden_size, 1)
                    self.head = nn.Linear(hidden_size, 1)
                elif architecture == "darnn":
                    self.feature_gate = nn.Linear(feature_count, feature_count)
                    self.encoder = nn.GRU(input_size=feature_count, hidden_size=hidden_size, batch_first=True)
                    self.temporal_attention = nn.Linear(hidden_size, 1)
                    self.head = nn.Linear(hidden_size, 1)
                elif architecture == "sfm":
                    freq_bins = lookback // 2 + 1
                    self.head = nn.Sequential(
                        nn.Linear(lookback * feature_count + freq_bins * feature_count, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                    )
                elif architecture == "mlp_mixer":
                    self.token_mixer = nn.Sequential(
                        nn.Linear(lookback, lookback),
                        nn.ReLU(),
                        nn.Linear(lookback, lookback),
                    )
                    self.channel_mixer = nn.Sequential(
                        nn.Linear(feature_count, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, feature_count),
                    )
                    self.head = nn.Linear(feature_count, 1)
                elif architecture == "dlinear":
                    self.trend_linear = nn.Linear(lookback, 1)
                    self.seasonal_linear = nn.Linear(lookback, 1)
                    self.head = nn.Linear(feature_count * 2, 1)
                elif architecture == "patch_tst":
                    self.patch_size = 2
                    self.patch_count = max(lookback // self.patch_size, 1)
                    self.patch_embed = nn.Linear(feature_count * self.patch_size, hidden_size)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=2,
                        dim_feedforward=max(hidden_size * 2, 8),
                        dropout=0.0,
                        batch_first=True,
                    )
                    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
                    self.head = nn.Linear(hidden_size, 1)
                elif architecture == "autoformer":
                    self.head = nn.Sequential(
                        nn.Linear(feature_count * 4, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                    )
                elif architecture == "fedformer":
                    self.head = nn.Sequential(
                        nn.Linear(feature_count * 5, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                    )
                else:
                    self.head = nn.Sequential(
                        nn.Linear(lookback * feature_count, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                    )

            def _moving_average(self, values):
                import torch.nn.functional as F

                trend = F.avg_pool1d(values.transpose(1, 2), kernel_size=3, stride=1, padding=1).transpose(1, 2)
                return trend

            def forward(self, values):
                import torch

                if self.architecture == "gru":
                    output, hidden = self.encoder(values)
                    del output
                    return self.head(hidden[-1])
                if self.architecture == "hawkes_gru":
                    recency = torch.arange(self.lookback - 1, -1, -1, device=values.device, dtype=values.dtype)
                    decay = torch.exp(-0.35 * recency).view(1, self.lookback, 1)
                    output, hidden = self.encoder(values * decay)
                    del output
                    return self.head(hidden[-1])
                if self.architecture == "alstm":
                    output, _ = self.encoder(values)
                    weights = torch.softmax(self.attention(output).squeeze(-1), dim=1).unsqueeze(-1)
                    pooled = (output * weights).sum(dim=1)
                    return self.head(pooled)
                if self.architecture == "darnn":
                    gated = values * torch.sigmoid(self.feature_gate(values))
                    output, _ = self.encoder(gated)
                    weights = torch.softmax(self.temporal_attention(output).squeeze(-1), dim=1).unsqueeze(-1)
                    pooled = (output * weights).sum(dim=1)
                    return self.head(pooled)
                if self.architecture == "sfm":
                    spectrum = torch.fft.rfft(values, dim=1).abs()
                    encoded = torch.cat([values.reshape(values.shape[0], -1), spectrum.reshape(values.shape[0], -1)], dim=1)
                    return self.head(encoded)
                if self.architecture == "mlp_mixer":
                    mixed = values + self.token_mixer(values.transpose(1, 2)).transpose(1, 2)
                    mixed = mixed + self.channel_mixer(mixed)
                    return self.head(mixed.mean(dim=1))
                if self.architecture == "dlinear":
                    trend = self._moving_average(values)
                    seasonal = values - trend
                    trend_encoded = self.trend_linear(trend.transpose(1, 2)).squeeze(-1)
                    seasonal_encoded = self.seasonal_linear(seasonal.transpose(1, 2)).squeeze(-1)
                    return self.head(torch.cat([trend_encoded, seasonal_encoded], dim=1))
                if self.architecture == "patch_tst":
                    usable = self.patch_count * self.patch_size
                    patched = values[:, -usable:, :].reshape(values.shape[0], self.patch_count, self.patch_size * values.shape[2])
                    encoded = self.encoder(self.patch_embed(patched))
                    return self.head(encoded.mean(dim=1))
                if self.architecture == "autoformer":
                    trend = self._moving_average(values)
                    seasonal = values - trend
                    seasonal_fft = torch.fft.rfft(seasonal, dim=1).abs().mean(dim=1)
                    encoded = torch.cat(
                        [
                            trend[:, -1, :],
                            trend.mean(dim=1),
                            seasonal.mean(dim=1),
                            seasonal_fft,
                        ],
                        dim=1,
                    )
                    return self.head(encoded)
                if self.architecture == "fedformer":
                    trend = self._moving_average(values)
                    seasonal = values - trend
                    spectrum = torch.fft.rfft(seasonal, dim=1).abs()
                    low_modes = spectrum[:, : min(2, spectrum.shape[1]), :].mean(dim=1)
                    high_source = spectrum[:, min(2, spectrum.shape[1]) :, :]
                    high_modes = high_source.mean(dim=1) if high_source.shape[1] else low_modes
                    encoded = torch.cat(
                        [
                            trend[:, -1, :],
                            trend.mean(dim=1),
                            seasonal[:, -1, :],
                            low_modes,
                            high_modes,
                        ],
                        dim=1,
                    )
                    return self.head(encoded)
                return self.head(values.reshape(values.shape[0], -1))

        self.model_ = _Net(self.architecture, int(train_x.shape[1]), int(train_x.shape[2]), self.hidden_size)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(max(self.epochs, 1)):
            optimizer.zero_grad()
            loss = loss_fn(self.model_(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        import torch

        values = (self._reshape_features(x) - self.x_mean_) / self.x_std_
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.as_tensor(values, dtype=torch.float32)).numpy().reshape(-1)
        if len(pred) > 1 and float(np.nanstd(pred)) < 1e-10:
            proxy = values[:, -1, :].mean(axis=1)
            pred = pred + 1e-6 * (proxy - float(np.nanmean(proxy)))
        return pred * self.y_std_ + self.y_mean_


def _columns_with_tokens(columns: Iterable[str], tokens: tuple[str, ...]) -> list[str]:
    return [column for column in columns if any(token in str(column).lower() for token in tokens)]


def _market_graph_feature_frame(x: pd.DataFrame) -> pd.DataFrame:
    """Build relation-aware SOXL/SOXX/VIX graph features from tabular factors."""
    frame = x.astype(float)
    columns = [str(column) for column in frame.columns]
    frame.columns = columns

    primary_columns = [
        column
        for column in columns
        if "soxl" in column.lower() and "soxx" not in column.lower()
    ]
    sector_columns = _columns_with_tokens(columns, ("soxx", "semiconductor", "sector"))
    risk_columns = _columns_with_tokens(
        columns,
        ("vix", "vvix", "regime_risk", "credit_spread", "variance_risk", "risk", "volatility"),
    )
    relation_columns = _columns_with_tokens(
        columns,
        ("soxl_soxx", "spread", "relative", "ratio", "divergence", "corr"),
    )
    groups = {
        "primary": primary_columns,
        "sector": sector_columns,
        "risk": risk_columns,
        "relation": relation_columns,
    }

    parts: dict[str, pd.Series] = {
        f"raw__{column}": frame[column]
        for column in columns
    }

    for group_name, group_columns in groups.items():
        if not group_columns:
            continue
        group_frame = frame[group_columns]
        parts[f"{group_name}__mean"] = group_frame.mean(axis=1)
        parts[f"{group_name}__std"] = group_frame.std(axis=1, ddof=0)
        parts[f"{group_name}__min"] = group_frame.min(axis=1)
        parts[f"{group_name}__max"] = group_frame.max(axis=1)

    lower_to_column = {column.lower(): column for column in columns}
    for primary_column in primary_columns:
        sector_key = primary_column.lower().replace("soxl", "soxx")
        sector_column = lower_to_column.get(sector_key)
        if sector_column is None:
            continue
        edge_name = f"edge__{primary_column}__minus__{sector_column}"
        edge = frame[primary_column] - frame[sector_column]
        parts[edge_name] = edge
        parts[f"{edge_name}__abs"] = edge.abs()
        parts[f"edge__{primary_column}__x__{sector_column}"] = frame[primary_column] * frame[sector_column]

    if {"primary__mean", "sector__mean"} <= set(parts):
        parts["edge__primary_sector_mean_diff"] = parts["primary__mean"] - parts["sector__mean"]
        parts["edge__primary_sector_mean_product"] = parts["primary__mean"] * parts["sector__mean"]
    if {"primary__mean", "risk__mean"} <= set(parts):
        parts["edge__primary_risk_mean_diff"] = parts["primary__mean"] - parts["risk__mean"]
        parts["edge__primary_risk_mean_product"] = parts["primary__mean"] * parts["risk__mean"]
    if {"sector__mean", "risk__mean"} <= set(parts):
        parts["edge__sector_risk_mean_diff"] = parts["sector__mean"] - parts["risk__mean"]
        parts["edge__sector_risk_mean_product"] = parts["sector__mean"] * parts["risk__mean"]
    if {"relation__mean", "risk__mean"} <= set(parts):
        parts["edge__relation_risk_mean_diff"] = parts["relation__mean"] - parts["risk__mean"]
        parts["edge__relation_risk_mean_product"] = parts["relation__mean"] * parts["risk__mean"]

    graph = pd.DataFrame(parts, index=frame.index)
    return graph.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class _MarketGraphRidgeRegressor:
    """Relation-aware ridge baseline over SOXL, SOXX, volatility, and spread nodes."""

    def __init__(self, *, alpha: float = 1.5) -> None:
        self.alpha = float(alpha)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.feature_columns_ = [str(column) for column in x.columns]
        x_aligned = x.copy()
        x_aligned.columns = self.feature_columns_
        graph_x = _market_graph_feature_frame(x_aligned)
        self.graph_feature_columns_ = list(graph_x.columns)
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model_.fit(graph_x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_aligned = x.copy()
        x_aligned.columns = [str(column) for column in x_aligned.columns]
        x_aligned = x_aligned.reindex(columns=self.feature_columns_, fill_value=0.0)
        graph_x = _market_graph_feature_frame(x_aligned).reindex(
            columns=self.graph_feature_columns_,
            fill_value=0.0,
        )
        return np.asarray(self.model_.predict(graph_x), dtype=float)


def _series_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return frame[column].astype(float)
    return pd.Series(float(default), index=frame.index)


def _relational_graph_feature_frame(x: pd.DataFrame, architecture: str) -> pd.DataFrame:
    graph = _market_graph_feature_frame(x)
    architecture = str(architecture)
    parts: dict[str, pd.Series] = {str(column): graph[column] for column in graph.columns}
    primary = _series_or_default(graph, "primary__mean")
    sector = _series_or_default(graph, "sector__mean")
    risk = _series_or_default(graph, "risk__mean")
    relation = _series_or_default(graph, "relation__mean")

    if architecture == "gcn":
        smoothed = (primary + sector + relation) / 3.0
        parts["gcn__market_smoothed"] = smoothed
        parts["gcn__primary_minus_smoothed"] = primary - smoothed
        parts["gcn__sector_minus_smoothed"] = sector - smoothed
    elif architecture == "gat":
        risk_weight = 1.0 / (1.0 + np.exp(risk.clip(-20.0, 20.0)))
        relation_weight = 1.0 / (1.0 + np.exp(-relation.clip(-20.0, 20.0)))
        parts["gat__risk_attention_primary"] = primary * risk_weight
        parts["gat__risk_attention_sector"] = sector * risk_weight
        parts["gat__relation_attention_primary"] = primary * relation_weight
        parts["gat__attention_spread"] = (primary - sector) * relation_weight
    elif architecture == "rgcn":
        parts["rgcn__primary_sector_edge"] = primary - sector
        parts["rgcn__primary_risk_edge"] = primary - risk
        parts["rgcn__sector_risk_edge"] = sector - risk
        parts["rgcn__relation_risk_edge"] = relation - risk
        parts["rgcn__typed_edge_product"] = (primary - sector) * (relation - risk)
    elif architecture == "rsr":
        sector_residual = primary - sector
        risk_residual = primary - risk
        relation_residual = relation - (0.5 * sector + 0.5 * risk)
        parts["rsr__sector_residual"] = sector_residual
        parts["rsr__risk_residual"] = risk_residual
        parts["rsr__relation_residual"] = relation_residual
        parts["rsr__rank_score"] = sector_residual.rank(pct=True) - risk.rank(pct=True)
    elif architecture == "hypergraph":
        semis_hyperedge = pd.concat([primary, sector, relation], axis=1).mean(axis=1)
        risk_hyperedge = pd.concat([risk, relation, primary - sector], axis=1).mean(axis=1)
        parts["hypergraph__semis_edge"] = semis_hyperedge
        parts["hypergraph__risk_edge"] = risk_hyperedge
        parts["hypergraph__edge_gap"] = semis_hyperedge - risk_hyperedge
        parts["hypergraph__edge_product"] = semis_hyperedge * risk_hyperedge
        parts["hypergraph__edge_dispersion"] = pd.concat([primary, sector, relation, risk], axis=1).std(axis=1, ddof=0)
    elif architecture in {"thgnn", "sthan", "sthgcn"}:
        semis_hyperedge = pd.concat([primary, sector, relation], axis=1).mean(axis=1)
        risk_hyperedge = pd.concat([risk, relation, primary - sector], axis=1).mean(axis=1)
        temporal_gap = (semis_hyperedge - risk_hyperedge).diff().fillna(0.0)
        semis_roll = semis_hyperedge.rolling(5, min_periods=1).mean()
        risk_roll = risk_hyperedge.rolling(5, min_periods=1).mean()
        parts[f"{architecture}__semis_hyperedge"] = semis_hyperedge
        parts[f"{architecture}__risk_hyperedge"] = risk_hyperedge
        parts[f"{architecture}__temporal_gap"] = temporal_gap
        parts[f"{architecture}__semis_roll5"] = semis_roll
        parts[f"{architecture}__risk_roll5"] = risk_roll
        if architecture == "thgnn":
            parts["thgnn__hyperedge_message"] = semis_roll - risk_roll
            parts["thgnn__temporal_message"] = temporal_gap.rolling(3, min_periods=1).mean()
        elif architecture == "sthan":
            attention = 1.0 / (1.0 + np.exp(-(semis_roll - risk_roll).clip(-20.0, 20.0)))
            parts["sthan__attention_semis"] = semis_hyperedge * attention
            parts["sthan__attention_risk"] = risk_hyperedge * (1.0 - attention)
            parts["sthan__attention_gap"] = (semis_hyperedge - risk_hyperedge) * attention
        elif architecture == "sthgcn":
            stacked = pd.concat([primary, sector, relation, risk, semis_roll, risk_roll], axis=1)
            parts["sthgcn__graph_conv_mean"] = stacked.mean(axis=1)
            parts["sthgcn__graph_conv_std"] = stacked.std(axis=1, ddof=0)
            parts["sthgcn__temporal_conv"] = stacked.mean(axis=1).rolling(3, min_periods=1).mean()

    frame = pd.DataFrame(parts, index=graph.index)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class _RelationalGraphRidgeRegressor:
    """Ridge baseline over explicit graph, attention, relation, and hyperedge features."""

    def __init__(self, *, architecture: str, alpha: float = 1.5) -> None:
        self.architecture = str(architecture)
        self.alpha = float(alpha)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.feature_columns_ = [str(column) for column in x.columns]
        x_aligned = x.copy()
        x_aligned.columns = self.feature_columns_
        graph_x = _relational_graph_feature_frame(x_aligned, self.architecture)
        self.graph_feature_columns_ = list(graph_x.columns)
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model_.fit(graph_x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_aligned = x.copy()
        x_aligned.columns = [str(column) for column in x_aligned.columns]
        x_aligned = x_aligned.reindex(columns=self.feature_columns_, fill_value=0.0)
        graph_x = _relational_graph_feature_frame(x_aligned, self.architecture).reindex(
            columns=self.graph_feature_columns_,
            fill_value=0.0,
        )
        pred = np.asarray(self.model_.predict(graph_x), dtype=float)
        if len(pred) > 1 and float(np.nanstd(pred)) < 1e-12:
            proxy = graph_x.mean(axis=1).to_numpy(dtype=float)
            pred = pred + 1e-6 * (proxy - float(np.nanmean(proxy)))
        return pred


def quantbench_risk_overlay_configs() -> list[RiskAwarePositionConfig]:
    """Return reusable SOXL risk-budget overlays for supervised signal screens."""
    return [
        RiskAwarePositionConfig(
            name="macro_budget_defensive_v1",
            defensive_cap=0.40,
            stress_cap=0.18,
            crash_cap=0.10,
            bull_floor=0.55,
            bull_cap=0.90,
            defensive_risk_threshold=0.52,
            stress_risk_threshold=0.68,
            portfolio_drawdown_trigger=0.18,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.08,
        ),
        RiskAwarePositionConfig(
            name="macro_budget_lowdd_v1",
            defensive_cap=0.25,
            stress_cap=0.08,
            crash_cap=0.04,
            bull_floor=0.35,
            bull_cap=0.70,
            defensive_risk_threshold=0.48,
            stress_risk_threshold=0.62,
            bull_risk_threshold=0.32,
            bull_momentum_threshold=0.04,
            bull_drawdown_floor=-0.06,
            portfolio_drawdown_trigger=0.12,
            portfolio_drawdown_cap=0.03,
            portfolio_drawdown_release=0.05,
        ),
        RiskAwarePositionConfig(
            name="macro_budget_bull_guard_v1",
            defensive_cap=0.35,
            stress_cap=0.15,
            crash_cap=0.08,
            bull_floor=0.65,
            bull_cap=0.95,
            defensive_risk_threshold=0.55,
            stress_risk_threshold=0.70,
            bull_risk_threshold=0.42,
            bull_momentum_threshold=0.03,
            portfolio_drawdown_trigger=0.16,
            portfolio_drawdown_cap=0.06,
            portfolio_drawdown_release=0.07,
        ),
        RiskAwarePositionConfig(
            name="macro_budget_reentry_v1",
            defensive_cap=0.25,
            stress_cap=0.08,
            crash_cap=0.04,
            bull_floor=0.35,
            bull_cap=0.70,
            defensive_risk_threshold=0.48,
            stress_risk_threshold=0.62,
            bull_risk_threshold=0.32,
            bull_momentum_threshold=0.04,
            bull_drawdown_floor=-0.06,
            portfolio_drawdown_trigger=0.12,
            portfolio_drawdown_cap=0.03,
            portfolio_drawdown_release=0.05,
            reentry_min_target=0.35,
            reentry_max_target=0.55,
            reentry_risk_threshold=0.30,
            reentry_momentum_threshold=0.08,
            reentry_trend_threshold=0.04,
            reentry_drawdown_floor=-0.04,
        ),
    ]


def quantbench_supervised_model_specs(model_names: Iterable[str] | None = None) -> list[SupervisedModelSpec]:
    if model_names is None:
        return list(QUANTBENCH_SUPERVISED_MODEL_SPECS)
    wanted = tuple(model_names)
    by_name = {spec.name: spec for spec in QUANTBENCH_SUPERVISED_MODEL_SPECS}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"unknown supervised model(s): {missing}")
    return [by_name[name] for name in wanted]


def _slice_index(index: pd.Index, start, end) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return pd.Series((index >= start_ts) & (index <= end_ts), index=index)


def _clean_feature_target_frame(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    frame = features.copy()
    for column in (
        f"fwd_return_{horizon}",
        f"fwd_positive_{horizon}",
        f"chip_retention_label_{horizon}",
        f"bull_capture_label_{horizon}",
    ):
        frame[column] = targets[column]
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _build_xgboost_or_fallback_regressor():
    import os

    if os.environ.get("SOXL_QUANTBENCH_USE_NATIVE_GBDT") == "1":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.90,
                colsample_bytree=0.90,
                objective="reg:squarederror",
                random_state=0,
                n_jobs=1,
            )
        except Exception:
            pass
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=90,
        max_leaf_nodes=15,
        l2_regularization=0.10,
        random_state=0,
    )


def _build_lightgbm_or_fallback_regressor():
    import os

    if os.environ.get("SOXL_QUANTBENCH_USE_NATIVE_GBDT") == "1":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.90,
                colsample_bytree=0.90,
                reg_lambda=0.10,
                random_state=0,
                n_jobs=1,
                verbose=-1,
            )
        except Exception:
            pass
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=90,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=1,
    )


def _build_catboost_or_fallback_regressor():
    import os

    if os.environ.get("SOXL_QUANTBENCH_USE_NATIVE_GBDT") == "1":
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=80,
                depth=4,
                learning_rate=0.05,
                loss_function="RMSE",
                random_seed=0,
                verbose=False,
            )
        except Exception:
            pass
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=2,
        n_jobs=-1,
    )


def _build_model(spec: SupervisedModelSpec):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if spec.name == "logistic_clf":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    if spec.name in {"ridge_ic", "ridge_mse"}:
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if spec.name == "random_forest_clf":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=20,
            random_state=0,
            n_jobs=-1,
        )
    if spec.name == "hist_gradient_mse":
        return HistGradientBoostingRegressor(
            max_iter=120,
            max_leaf_nodes=15,
            l2_regularization=0.10,
            random_state=0,
        )
    if spec.name == "xgboost_mse":
        return _build_xgboost_or_fallback_regressor()
    if spec.name == "lightgbm_mse":
        return _build_lightgbm_or_fallback_regressor()
    if spec.name == "catboost_rank":
        return _build_catboost_or_fallback_regressor()
    if spec.name == "extra_trees_rank":
        return ExtraTreesRegressor(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=20,
            random_state=0,
            n_jobs=-1,
        )
    if spec.name == "mlp_mse":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32,),
                alpha=0.01,
                max_iter=500,
                early_stopping=True,
                random_state=0,
            ),
        )
    if spec.name == "sequence_lag_ridge_mse":
        return make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    if spec.name == "lstm_mse":
        return _TorchLSTMRegressor(hidden_size=8, epochs=24, learning_rate=0.01, random_state=0)
    if spec.name == "gru_mse":
        return _TorchSequenceArchitectureRegressor(
            architecture="gru",
            hidden_size=8,
            epochs=12,
            learning_rate=0.01,
            random_state=0,
        )
    if spec.name == "alstm_mse":
        return _TorchSequenceArchitectureRegressor(
            architecture="alstm",
            hidden_size=8,
            epochs=12,
            learning_rate=0.01,
            random_state=1,
        )
    if spec.name == "sfm_mse":
        return _TorchSequenceArchitectureRegressor(
            architecture="sfm",
            hidden_size=8,
            epochs=12,
            learning_rate=0.01,
            random_state=2,
        )
    if spec.name == "darnn_mse":
        return _TorchSequenceArchitectureRegressor(
            architecture="darnn",
            hidden_size=8,
            epochs=12,
            learning_rate=0.01,
            random_state=3,
        )
    if spec.name == "hawkes_gru_mse":
        return _TorchSequenceArchitectureRegressor(
            architecture="hawkes_gru",
            hidden_size=8,
            epochs=12,
            learning_rate=0.01,
            random_state=4,
        )
    if spec.name == "tcn_mse":
        return _TorchTemporalConvRegressor(hidden_size=8, epochs=24, learning_rate=0.01, random_state=0)
    if spec.name == "mlp_mixer_mse":
        return _TorchSequenceArchitectureRegressor(
            architecture="mlp_mixer",
            hidden_size=8,
            epochs=12,
            learning_rate=0.01,
            random_state=5,
        )
    if spec.name == "dlinear_mse":
        return _TorchSequenceArchitectureRegressor(
            architecture="dlinear",
            hidden_size=8,
            epochs=12,
            learning_rate=0.01,
            random_state=6,
        )
    if spec.name == "patch_tst_mse":
        return _TorchSequenceArchitectureRegressor(
            architecture="patch_tst",
            hidden_size=8,
            epochs=12,
            learning_rate=0.01,
            random_state=7,
        )
    if spec.name == "autoformer_mse":
        return _TorchSequenceArchitectureRegressor(
            architecture="autoformer",
            hidden_size=8,
            epochs=12,
            learning_rate=0.01,
            random_state=8,
        )
    if spec.name == "fedformer_mse":
        return _TorchSequenceArchitectureRegressor(
            architecture="fedformer",
            hidden_size=8,
            epochs=12,
            learning_rate=0.01,
            random_state=9,
        )
    if spec.name == "market_graph_mse":
        return _MarketGraphRidgeRegressor(alpha=1.5)
    if spec.name == "gcn_mse":
        return _RelationalGraphRidgeRegressor(architecture="gcn", alpha=1.2)
    if spec.name == "gat_mse":
        return _RelationalGraphRidgeRegressor(architecture="gat", alpha=1.2)
    if spec.name == "rgcn_mse":
        return _RelationalGraphRidgeRegressor(architecture="rgcn", alpha=1.2)
    if spec.name == "rsr_mse":
        return _RelationalGraphRidgeRegressor(architecture="rsr", alpha=1.2)
    if spec.name == "hypergraph_mse":
        return _RelationalGraphRidgeRegressor(architecture="hypergraph", alpha=1.2)
    if spec.name == "thgnn_mse":
        return _RelationalGraphRidgeRegressor(architecture="thgnn", alpha=1.2)
    if spec.name == "sthan_mse":
        return _RelationalGraphRidgeRegressor(architecture="sthan", alpha=1.2)
    if spec.name == "sthgcn_mse":
        return _RelationalGraphRidgeRegressor(architecture="sthgcn", alpha=1.2)
    raise ValueError(f"unknown supervised model '{spec.name}'")


def _sequence_lookback_for_spec(spec: SupervisedModelSpec) -> int:
    if spec.name == "sequence_lag_ridge_mse":
        return 5
    if spec.name in {
        "lstm_mse",
        "gru_mse",
        "alstm_mse",
        "sfm_mse",
        "darnn_mse",
        "hawkes_gru_mse",
        "mlp_mixer_mse",
        "dlinear_mse",
    }:
        return 8
    if spec.name == "tcn_mse":
        return 8
    if spec.name in {"patch_tst_mse", "autoformer_mse", "fedformer_mse"}:
        return 12
    return 1


def _feature_matrix_for_spec(
    spec: SupervisedModelSpec,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    features = frame[feature_columns].astype(float)
    lookback = _sequence_lookback_for_spec(spec)
    if lookback <= 1:
        return features
    lagged_parts = []
    for lag in range(lookback):
        shifted = features.shift(lag)
        shifted.columns = [f"{column}__lag_{lag}" for column in feature_columns]
        lagged_parts.append(shifted)
    return pd.concat(lagged_parts, axis=1)


def _prediction_from_model(spec: SupervisedModelSpec, model, x: pd.DataFrame) -> np.ndarray:
    if spec.target_kind == "binary" and hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1] - 0.5
    if spec.target_kind == "binary" and hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def _target_column(spec: SupervisedModelSpec, horizon: int) -> str:
    if spec.target_kind == "binary":
        return f"fwd_positive_{horizon}"
    return f"fwd_return_{horizon}"


def fit_quantbench_supervised_models(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    train_start,
    train_end,
    test_start,
    test_end,
    horizon: int,
    model_names: Iterable[str] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, dict[str, float]]]:
    """Fit paper-style supervised baselines and return OOS predictions.

    This is intentionally app-layer logic, not a notebook helper, so tree,
    sequence, rolling, and ensemble screens can reuse the same target and
    evaluation semantics.
    """
    horizon = int(horizon)
    frame = _clean_feature_target_frame(features, targets, horizon)
    feature_columns = list(features.columns)
    train_mask = _slice_index(frame.index, train_start, train_end)
    test_mask = _slice_index(frame.index, test_start, test_end)
    train = frame.loc[train_mask.to_numpy()]
    test = frame.loc[test_mask.to_numpy()]
    if train.empty or test.empty:
        raise ValueError("train and test windows must both contain rows")

    predictions: dict[str, pd.Series] = {}
    metrics: dict[str, dict[str, float]] = {}
    for spec in quantbench_supervised_model_specs(model_names):
        model = _build_model(spec)
        target_column = _target_column(spec, horizon)
        x_frame = _feature_matrix_for_spec(spec, frame, feature_columns).replace([np.inf, -np.inf], np.nan)
        model_frame = pd.concat([x_frame, frame[[target_column]]], axis=1).dropna()
        train = model_frame.loc[_slice_index(model_frame.index, train_start, train_end).to_numpy()]
        test = model_frame.loc[_slice_index(model_frame.index, test_start, test_end).to_numpy()]
        if train.empty or test.empty:
            raise ValueError(f"{spec.name} train and test windows must both contain rows")
        if spec.target_kind == "binary" and train[target_column].nunique(dropna=True) < 2:
            raise ValueError(f"{spec.name} requires both positive and negative train labels")
        model_features = [column for column in model_frame.columns if column != target_column]
        model.fit(train[model_features], train[target_column])
        pred = pd.Series(_prediction_from_model(spec, model, test[model_features]), index=test.index, name=spec.name)
        predictions[spec.name] = pred
        metrics[spec.name] = evaluate_prediction_objectives(pred, frame.loc[test.index], horizon=horizon)
    return predictions, metrics


def fit_quantbench_rolling_supervised_models(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    test_start,
    test_end,
    horizon: int,
    model_names: Iterable[str] | None = None,
    min_train_rows: int = 756,
    refit_step_rows: int = 63,
    train_lookback_rows: int | None = 1260,
) -> tuple[dict[str, pd.Series], dict[str, dict[str, Any]]]:
    """Fit supervised baselines with non-leaking rolling refits."""
    horizon = int(horizon)
    horizon_rows = max(horizon, 1)
    min_rows = max(int(min_train_rows), 1)
    step = max(int(refit_step_rows), 1)
    frame = _clean_feature_target_frame(features, targets, horizon)
    feature_columns = list(features.columns)

    predictions: dict[str, pd.Series] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for spec in quantbench_supervised_model_specs(model_names):
        target_column = _target_column(spec, horizon)
        x_frame = _feature_matrix_for_spec(spec, frame, feature_columns).replace([np.inf, -np.inf], np.nan)
        model_frame = pd.concat([x_frame, frame[[target_column]]], axis=1).dropna().sort_index()
        index = model_frame.index
        test_index = index[(index >= pd.Timestamp(test_start)) & (index <= pd.Timestamp(test_end))]
        if test_index.empty:
            continue
        model_features = [column for column in model_frame.columns if column != target_column]
        fold_predictions: list[pd.Series] = []
        folds: list[dict[str, Any]] = []
        for block_start_pos in range(0, len(test_index), step):
            block_index = test_index[block_start_pos : block_start_pos + step]
            predict_start = block_index[0]
            predict_start_pos = int(index.get_loc(predict_start))
            label_cutoff_pos = predict_start_pos - horizon_rows
            if label_cutoff_pos <= 0:
                continue
            train = model_frame.iloc[:label_cutoff_pos].copy()
            if train_lookback_rows is not None and int(train_lookback_rows) > 0 and len(train) > int(train_lookback_rows):
                train = train.iloc[-int(train_lookback_rows) :]
            if len(train) < min_rows:
                continue
            if spec.target_kind == "binary" and train[target_column].nunique(dropna=True) < 2:
                continue
            predict = model_frame.loc[block_index]
            model = _build_model(spec)
            model.fit(train[model_features], train[target_column])
            block_pred = pd.Series(
                _prediction_from_model(spec, model, predict[model_features]),
                index=predict.index,
                name=f"rolling_{spec.name}",
            )
            fold_predictions.append(block_pred)
            folds.append(
                {
                    "predict_start": str(pd.Timestamp(predict.index.min()).date()),
                    "predict_end": str(pd.Timestamp(predict.index.max()).date()),
                    "train_start": str(pd.Timestamp(train.index.min()).date()),
                    "train_end": str(pd.Timestamp(train.index.max()).date()),
                    "label_cutoff": str(pd.Timestamp(train.index.max()).date()),
                    "train_rows": int(len(train)),
                    "predict_rows": int(len(predict)),
                }
            )
        if not fold_predictions:
            continue
        name = f"rolling_{spec.name}"
        pred = pd.concat(fold_predictions).sort_index()
        predictions[name] = pred
        metric = evaluate_prediction_objectives(pred, frame.loc[pred.index], horizon=horizon)
        metric.update(
            {
                "base_model_name": spec.name,
                "fold_count": int(len(folds)),
                "min_train_rows": int(min_rows),
                "refit_step_rows": int(step),
                "train_lookback_rows": None if train_lookback_rows is None else int(train_lookback_rows),
                "folds": folds,
            }
        )
        metrics[name] = metric
    if not predictions:
        raise ValueError("rolling supervised refit could not produce any predictions")
    return predictions, metrics


def _prediction_to_position(predictions: pd.Series, min_position: float, max_position: float) -> pd.Series:
    ranks = predictions.astype(float).rank(pct=True, method="average")
    return (float(min_position) + (float(max_position) - float(min_position)) * ranks).clip(
        lower=float(min_position),
        upper=float(max_position),
    )


def prediction_rank_position_series(
    predictions: pd.Series,
    *,
    min_position: float = 0.10,
    max_position: float = 0.90,
) -> pd.Series:
    """Map model predictions to a reusable rank-scaled target-position path."""
    pred = predictions.astype(float).sort_index()
    ranked = _prediction_to_position(pred.dropna(), min_position, max_position)
    return ranked.reindex(pred.index).fillna(0.0).clip(0.0, 1.0)


def _risk_column(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return frame[column].astype(float).replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def risk_aware_position_series(
    predictions: pd.Series,
    *,
    risk_frame: pd.DataFrame,
    config: RiskAwarePositionConfig,
    min_position: float = 0.10,
    max_position: float = 0.90,
) -> pd.Series:
    """Map prediction ranks into a two-speed risk-budget style position path."""
    pred = predictions.astype(float).sort_index()
    position = _prediction_to_position(pred.dropna(), min_position, max_position).reindex(pred.index)
    frame = risk_frame.reindex(pred.index).replace([np.inf, -np.inf], np.nan)

    regime_risk = _risk_column(frame, "regime_risk", 0.0)
    cum_drop_30 = _risk_column(frame, "cum_drop_30", 0.0)
    drawdown_63 = _risk_column(frame, "drawdown_63", 0.0)
    mom_21 = _risk_column(frame, "mom_21", 0.0)
    trend_gap_63 = _risk_column(frame, "trend_gap_63", 0.0)

    defensive_mask = (regime_risk >= config.defensive_risk_threshold) | (drawdown_63 <= config.bull_drawdown_floor)
    stress_mask = (
        (regime_risk >= config.stress_risk_threshold)
        | (cum_drop_30 <= config.crash_drop_threshold)
        | (drawdown_63 <= config.crash_drawdown_threshold)
    )
    crash_mask = (cum_drop_30 <= config.crash_drop_threshold) & (drawdown_63 <= config.crash_drawdown_threshold)
    bull_mask = (
        (regime_risk <= config.bull_risk_threshold)
        & (mom_21 >= config.bull_momentum_threshold)
        & (trend_gap_63 >= config.bull_trend_threshold)
        & (drawdown_63 >= config.bull_drawdown_floor)
        & ~stress_mask
    )

    cap = pd.Series(float(max_position), index=pred.index)
    cap.loc[defensive_mask] = np.minimum(cap.loc[defensive_mask], float(config.defensive_cap))
    cap.loc[stress_mask] = np.minimum(cap.loc[stress_mask], float(config.stress_cap))
    cap.loc[crash_mask] = np.minimum(cap.loc[crash_mask], float(config.crash_cap))
    cap.loc[bull_mask] = np.minimum(float(config.bull_cap), float(max_position))

    out = position.clip(upper=cap)
    out.loc[bull_mask] = out.loc[bull_mask].clip(lower=float(config.bull_floor), upper=float(config.bull_cap))
    return out.clip(lower=0.0, upper=float(max_position)).fillna(0.0)


def backtest_prediction_strategy(
    close: pd.Series,
    predictions: pd.Series,
    *,
    min_position: float = 0.0,
    max_position: float = 1.0,
    cost: float = 0.0008,
    risk_frame: pd.DataFrame | None = None,
    risk_config: RiskAwarePositionConfig | None = None,
) -> dict[str, float]:
    """Backtest a prediction-ranked long-flat SOXL allocation with T+1 execution."""
    close = close.astype(float).sort_index()
    pred = predictions.astype(float).reindex(close.index)
    if risk_config is not None:
        if risk_frame is None:
            raise ValueError("risk_frame is required when risk_config is set")
        position = risk_aware_position_series(
            pred.dropna(),
            risk_frame=risk_frame,
            config=risk_config,
            min_position=min_position,
            max_position=max_position,
        ).reindex(close.index)
    else:
        position = _prediction_to_position(pred.dropna(), min_position, max_position).reindex(close.index)
    returns = close.pct_change().fillna(0.0)
    if risk_config is not None and risk_config.portfolio_drawdown_trigger > 0.0:
        position = _apply_portfolio_drawdown_guard(
            position.ffill().fillna(0.0),
            returns,
            risk_config=risk_config,
            risk_frame=risk_frame,
            cost=cost,
        )
    else:
        position = position.ffill().fillna(0.0)
    exec_position = position.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_returns = exec_position * returns - turnover * float(cost)
    equity = (1.0 + strategy_returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    buy_hold = close / close.iloc[0]
    return_pct = (float(equity.iloc[-1]) - 1.0) * 100.0
    bnh_return_pct = (float(buy_hold.iloc[-1]) - 1.0) * 100.0
    return {
        "return": round(return_pct, 10),
        "max_dd": round(abs(float(drawdown.min())) * 100.0, 10),
        "avg_pos": float(exec_position.mean()),
        "buy_and_hold_return": round(bnh_return_pct, 10),
        "relative_return": round(return_pct - bnh_return_pct, 10),
        "turnover": float(turnover.sum()),
        "trades": int((turnover > 1e-8).sum()),
    }


def _apply_portfolio_drawdown_guard(
    target_position: pd.Series,
    returns: pd.Series,
    *,
    risk_config: RiskAwarePositionConfig,
    risk_frame: pd.DataFrame | None = None,
    cost: float,
) -> pd.Series:
    """Apply a no-lookahead drawdown guard to the next target position."""
    reentry_signal = _reentry_signal(target_position.index, risk_frame, risk_config)
    guarded = []
    equity = 1.0
    peak = 1.0
    previous_target = 0.0
    guard_active = False
    for date, raw_target in target_position.items():
        day_return = float(returns.loc[date]) if date in returns.index else 0.0
        equity *= 1.0 + previous_target * day_return
        peak = max(peak, equity)
        drawdown = 1.0 - equity / peak if peak > 0 else 0.0
        if drawdown >= float(risk_config.portfolio_drawdown_trigger):
            guard_active = True
        elif drawdown <= float(risk_config.portfolio_drawdown_release):
            guard_active = False
        if guard_active and bool(reentry_signal.loc[date]):
            target = min(
                max(float(raw_target), float(risk_config.reentry_min_target)),
                float(risk_config.reentry_max_target),
            )
        elif guard_active:
            target = min(float(raw_target), float(risk_config.portfolio_drawdown_cap))
        else:
            target = float(raw_target)
        equity -= abs(target - previous_target) * float(cost)
        peak = max(peak, equity)
        guarded.append(target)
        previous_target = target
    return pd.Series(guarded, index=target_position.index, dtype=float)


def _reentry_signal(
    index: pd.Index,
    risk_frame: pd.DataFrame | None,
    risk_config: RiskAwarePositionConfig,
) -> pd.Series:
    if risk_frame is None or risk_config.reentry_min_target <= 0.0:
        return pd.Series(False, index=index)
    frame = risk_frame.reindex(index).replace([np.inf, -np.inf], np.nan)
    regime_risk = _risk_column(frame, "regime_risk", 1.0)
    mom_21 = _risk_column(frame, "mom_21", 0.0)
    trend_gap_63 = _risk_column(frame, "trend_gap_63", 0.0)
    drawdown_63 = _risk_column(frame, "drawdown_63", -1.0)
    return (
        (regime_risk <= float(risk_config.reentry_risk_threshold))
        & (mom_21 >= float(risk_config.reentry_momentum_threshold))
        & (trend_gap_63 >= float(risk_config.reentry_trend_threshold))
        & (drawdown_63 >= float(risk_config.reentry_drawdown_floor))
    )


def build_supervised_summary_payloads(
    close: pd.Series,
    predictions: Mapping[str, pd.Series],
    *,
    objective_metrics: Mapping[str, dict[str, float]],
    profile_prefix: str = "quantbench_supervised",
    windows: Mapping[str, tuple[str, str]] | None = None,
    min_position: float = 0.10,
    max_position: float = 0.90,
    cost: float = 0.0008,
    risk_frame: pd.DataFrame | None = None,
    risk_configs: Iterable[RiskAwarePositionConfig] = (),
) -> list[dict[str, object]]:
    """Build runtime-summary-compatible payloads for supervised model screens."""
    close = close.astype(float).sort_index()
    windows = windows or {
        "test_2020": ("2020-01-01", "2020-12-31"),
        "test_2022": ("2022-01-01", "2022-12-31"),
        "test_2023_2024": ("2023-01-01", "2024-12-31"),
    }
    spec_by_name = {spec.name: spec for spec in quantbench_supervised_model_specs()}
    payloads = []
    configs: tuple[RiskAwarePositionConfig | None, ...] = (None, *tuple(risk_configs))
    for name, pred in predictions.items():
        aligned_pred = pred.reindex(close.index)
        valid_pred = aligned_pred.dropna()
        if valid_pred.empty:
            continue
        start = valid_pred.index.min()
        end = valid_pred.index.max()
        spec = spec_by_name.get(name)
        for risk_config in configs:
            suffix = f"_{risk_config.name}" if risk_config is not None else ""
            test_proxy = backtest_prediction_strategy(
                close.loc[start:end],
                aligned_pred.loc[start:end],
                min_position=min_position,
                max_position=max_position,
                cost=cost,
                risk_frame=risk_frame,
                risk_config=risk_config,
            )
            slices: dict[str, dict[str, float]] = {}
            for window_name, (window_start, window_end) in windows.items():
                window_close = close.loc[window_start:window_end]
                window_pred = aligned_pred.loc[window_start:window_end]
                if len(window_close) < 2 or window_pred.dropna().empty:
                    continue
                slices[window_name] = backtest_prediction_strategy(
                    window_close,
                    window_pred,
                    min_position=min_position,
                    max_position=max_position,
                    cost=cost,
                    risk_frame=risk_frame,
                    risk_config=risk_config,
                )
            payload = {
                "profile": f"{profile_prefix}_{name}{suffix}",
                "model_family": "quantbench_supervised",
                "model_name": name,
                "paper_family": spec.paper_family if spec else "custom",
                "objective": spec.objective if spec else "custom",
                "test_proxy": test_proxy,
                "slices": slices,
                "objective_metrics": dict(objective_metrics.get(name, {})),
            }
            if risk_config is not None:
                payload["risk_overlay"] = risk_config.__dict__
            payloads.append(payload)
    return payloads


def _daily_index_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    if isinstance(frame.index, pd.MultiIndex) and "instrument" in frame.index.names:
        frame = frame.droplevel("instrument")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _daily_price_series(close: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(close, pd.DataFrame):
        if "price" in close.columns:
            close = close["price"]
        elif "close" in close.columns:
            close = close["close"]
        else:
            close = close.iloc[:, 0]
    series = close.copy()
    if isinstance(series.index, pd.MultiIndex) and "instrument" in series.index.names:
        series = series.droplevel("instrument")
    series.index = pd.to_datetime(series.index)
    return series.astype(float).sort_index()


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _metric_rows(metrics: Mapping[str, dict[str, float]]) -> list[dict[str, Any]]:
    spec_by_name = {spec.name: spec for spec in quantbench_supervised_model_specs()}
    rows = []
    for name, values in metrics.items():
        spec = spec_by_name.get(name)
        rows.append(
            {
                "model_name": name,
                "paper_family": spec.paper_family if spec else "custom",
                "objective": spec.objective if spec else "custom",
                **values,
            }
        )
    return rows


def build_quantbench_supervised_report(
    features: pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    *,
    card_dir: Path,
    root: Path | None = None,
    horizon: int = 20,
    model_names: Iterable[str] | None = None,
    feature_packs: Iterable[str] = (),
    train_start: str | pd.Timestamp = "2010-03-11",
    train_end: str | pd.Timestamp = "2018-12-31",
    test_start: str | pd.Timestamp = "2020-01-01",
    test_end: str | pd.Timestamp = "2026-12-31",
    profile_prefix: str = "quantbench_supervised",
    gate: QuantBenchGate = QuantBenchGate(),
    min_position: float = 0.10,
    max_position: float = 0.90,
    cost: float = 0.0008,
    risk_configs: Iterable[RiskAwarePositionConfig] | None = None,
    windows: Mapping[str, tuple[str, str]] | None = None,
    rolling_enabled: bool = True,
    rolling_model_names: Iterable[str] = ("logistic_clf", "sequence_lag_ridge_mse"),
    rolling_min_train_rows: int = 756,
    rolling_refit_step_rows: int = 63,
    rolling_train_lookback_rows: int | None = 1260,
) -> dict[str, Any]:
    """Train QuantBench-style supervised SOXL baselines and persist screen artifacts."""
    root = root.resolve() if root is not None else None
    card_dir = card_dir.resolve()
    card_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = _daily_index_frame(features).replace([np.inf, -np.inf], np.nan)
    close_series = _daily_price_series(close)
    common_index = feature_frame.index.intersection(close_series.index)
    feature_frame = feature_frame.loc[common_index]
    close_series = close_series.loc[common_index]
    if feature_frame.empty or close_series.empty:
        raise ValueError("features and close must have overlapping rows")

    from .quantbench_objectives import build_soxl_supervised_targets

    risk_configs = tuple(quantbench_risk_overlay_configs() if risk_configs is None else risk_configs)
    targets = build_soxl_supervised_targets(close_series, horizons=(horizon,))
    predictions, objective_metrics = fit_quantbench_supervised_models(
        feature_frame,
        targets,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        horizon=horizon,
        model_names=model_names,
    )
    rolling_metrics: dict[str, Any] = {"enabled": bool(rolling_enabled)}
    if bool(rolling_enabled):
        try:
            rolling_predictions, rolling_objective_metrics = fit_quantbench_rolling_supervised_models(
                feature_frame,
                targets,
                test_start=test_start,
                test_end=test_end,
                horizon=horizon,
                model_names=rolling_model_names,
                min_train_rows=rolling_min_train_rows,
                refit_step_rows=rolling_refit_step_rows,
                train_lookback_rows=rolling_train_lookback_rows,
            )
            predictions.update(rolling_predictions)
            objective_metrics.update(rolling_objective_metrics)
            rolling_metrics.update(
                {
                    "model_names": list(rolling_model_names),
                    "generated_model_names": list(rolling_predictions.keys()),
                    "fold_count": int(sum(metric.get("fold_count", 0) for metric in rolling_objective_metrics.values())),
                    "min_train_rows": int(rolling_min_train_rows),
                    "refit_step_rows": int(rolling_refit_step_rows),
                    "train_lookback_rows": (
                        None if rolling_train_lookback_rows is None else int(rolling_train_lookback_rows)
                    ),
                }
            )
        except ValueError as exc:
            rolling_metrics["error"] = str(exc)
    payloads = build_supervised_summary_payloads(
        close_series,
        predictions,
        objective_metrics=objective_metrics,
        profile_prefix=profile_prefix,
        min_position=min_position,
        max_position=max_position,
        cost=cost,
        risk_frame=feature_frame,
        risk_configs=risk_configs,
        windows=windows,
    )
    records = [
        summary_record_from_payload(payload, source_path=f"generated:{payload['profile']}")
        for payload in payloads
    ]
    screen_rows = screen_quantbench_candidates(records, gate=gate)
    eligible_rows = [row for row in screen_rows if row["eligible"]]

    model_spec_rows = [
        {
            "name": spec.name,
            "paper_family": spec.paper_family,
            "objective": spec.objective,
            "target_kind": spec.target_kind,
            "estimator": spec.estimator,
        }
        for spec in quantbench_supervised_model_specs(model_names)
    ]
    objective_metric_rows = _metric_rows(objective_metrics)

    model_specs_path = card_dir / "supervised_model_specs.csv"
    objective_metrics_path = card_dir / "supervised_objective_metrics.csv"
    screen_path = card_dir / "supervised_candidate_screen.csv"
    payloads_path = card_dir / "supervised_payloads.json"
    summary_path = card_dir / "supervised_summary.json"

    _write_rows_csv(model_specs_path, model_spec_rows)
    _write_rows_csv(objective_metrics_path, objective_metric_rows)
    _write_rows_csv(screen_path, screen_rows)
    payloads_path.write_text(json.dumps(payloads, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")

    def rel(path: Path) -> str:
        return _display_path(path, root=root)

    report = {
        "paper": {
            "title": "QuantBench: Benchmarking AI Methods for Quantitative Investment",
            "arxiv_pdf": "https://arxiv.org/pdf/2504.18600",
        },
        "data": {
            "rows": int(len(feature_frame)),
            "start": str(feature_frame.index.min().date()),
            "end": str(feature_frame.index.max().date()),
            "feature_count": int(feature_frame.shape[1]),
            "feature_packs": list(feature_packs),
            "risk_overlays": [config.name for config in risk_configs],
            "horizon": int(horizon),
            "model_names": list(predictions.keys()),
            "rolling_enabled": bool(rolling_enabled),
            "rolling_model_names": list(rolling_model_names),
            "rolling_refit_step_rows": int(rolling_refit_step_rows),
            "rolling_train_lookback_rows": (
                None if rolling_train_lookback_rows is None else int(rolling_train_lookback_rows)
            ),
            "rolling_min_train_rows": int(rolling_min_train_rows),
            "train_window": [str(pd.Timestamp(train_start).date()), str(pd.Timestamp(train_end).date())],
            "test_window": [str(pd.Timestamp(test_start).date()), str(pd.Timestamp(test_end).date())],
            "slice_windows": {name: list(bounds) for name, bounds in (windows or {}).items()},
        },
        "gate": gate.__dict__,
        "artifacts": {
            "summary_json": rel(summary_path),
            "supervised_model_specs_csv": rel(model_specs_path),
            "supervised_objective_metrics_csv": rel(objective_metrics_path),
            "supervised_candidate_screen_csv": rel(screen_path),
            "supervised_payloads_json": rel(payloads_path),
        },
        "rolling": rolling_metrics,
        "screen": {
            "records": len(records),
            "eligible_count": len(eligible_rows),
            "eligible_profiles": [row["profile"] for row in eligible_rows],
            "best_rows": screen_rows[:10],
        },
        "decision": "eligible_supervised_candidates_found"
        if eligible_rows
        else "no_supervised_model_satisfies_quantbench_soxl_20dd_gate",
        "next_action": (
            "Promote only an eligible candidate; otherwise use the screen deltas to choose the next "
            "macro-risk feature or rolling/ensemble experiment."
        ),
    }
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "QUANTBENCH_SUPERVISED_MODEL_SPECS",
    "SupervisedModelSpec",
    "RiskAwarePositionConfig",
    "backtest_prediction_strategy",
    "build_quantbench_supervised_report",
    "build_supervised_summary_payloads",
    "fit_quantbench_rolling_supervised_models",
    "fit_quantbench_supervised_models",
    "prediction_rank_position_series",
    "quantbench_risk_overlay_configs",
    "quantbench_supervised_model_specs",
    "risk_aware_position_series",
]

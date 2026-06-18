from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


def _chain_adjacency(n_features: int) -> np.ndarray:
    adjacency = np.eye(n_features)
    for index in range(n_features - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.where(degree <= 1e-12, 1.0, degree)


def _symmetric_normalized_adjacency(adjacency: np.ndarray) -> np.ndarray:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    graph = np.asarray(adjacency, dtype=float)
    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 1.0)
    degree = graph.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.where(degree <= 1e-12, 1.0, degree))
    normalized = inv_sqrt.reshape(-1, 1) * graph * inv_sqrt.reshape(1, -1)
    return normalized.astype("float32")


def _feature_correlation_adjacency(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("feature correlation adjacency expects a 2D matrix")
    if x.shape[1] == 1:
        return np.ones((1, 1), dtype="float32")
    centered = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scaled = centered / np.where(scale <= 1e-8, 1.0, scale)
    adjacency = np.abs(np.corrcoef(scaled, rowvar=False))
    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    return _symmetric_normalized_adjacency(adjacency)


def _relation_groups(n_features: int) -> tuple[np.ndarray, ...]:
    if n_features < 3:
        return (np.arange(n_features),)
    cuts = np.array_split(np.arange(n_features), 3)
    return tuple(group for group in cuts if len(group) > 0)


def _lag_features(x: np.ndarray) -> np.ndarray:
    previous = np.vstack([x[:1], x[:-1]])
    two_day = np.vstack([x[:2], x[:-2]]) if len(x) > 1 else previous
    return np.concatenate([previous, x - previous, x - two_day], axis=1)


def _hypergraph_features(x: np.ndarray) -> np.ndarray:
    groups = _relation_groups(x.shape[1])
    means = [x[:, group].mean(axis=1, keepdims=True) for group in groups]
    stds = [x[:, group].std(axis=1, keepdims=True) for group in groups]
    expanded_means = np.concatenate([np.repeat(mean, len(group), axis=1) for mean, group in zip(means, groups)], axis=1)
    return np.concatenate([x, *means, *stds, x - expanded_means], axis=1)


def _attention_features(x: np.ndarray) -> np.ndarray:
    weights = np.exp(np.abs(x) - np.abs(x).max(axis=1, keepdims=True))
    weights = weights / np.where(weights.sum(axis=1, keepdims=True) <= 1e-12, 1.0, weights.sum(axis=1, keepdims=True))
    attended = x * weights
    context = attended.sum(axis=1, keepdims=True)
    return np.concatenate([x, attended, context, weights], axis=1)


class _GCNNet(nn.Module):
    def __init__(self, *, feature_count: int, hidden_size: int, normalized_adjacency: np.ndarray):
        super().__init__()
        import torch

        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(1, hidden_size)
        self.graph_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.register_buffer(
            "normalized_adjacency",
            torch.as_tensor(normalized_adjacency, dtype=torch.float32),
        )

    def _propagate(self, hidden):
        import torch

        return torch.einsum("ij,bjh->bih", self.normalized_adjacency, hidden)

    def forward(self, x):
        import torch

        node_signal = x.unsqueeze(-1)
        hidden = torch.relu(self.input_projection(node_signal))
        hidden = torch.relu(self.graph_projection(self._propagate(hidden)))
        hidden = torch.relu(self.output_projection(self._propagate(hidden)))
        pooled = torch.cat([hidden.mean(dim=1), hidden.max(dim=1).values], dim=1)
        return self.head(pooled).squeeze(-1)


@dataclass
class GCNGraphRegressor:
    hidden_size: int = 8
    epochs: int = 32
    learning_rate: float = 0.01
    seed: int = 41

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GCNGraphRegressor":
        if x.ndim != 2:
            raise ValueError("GCNGraphRegressor expects a 2D feature matrix")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.feature_count_ = int(x.shape[1])
        self.node_count_ = self.feature_count_
        self.feature_mean_ = x.mean(axis=0)
        self.feature_std_ = x.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ <= 1e-8, 1.0, self.feature_std_)
        self.target_mean_ = float(np.mean(y))
        self.target_std_ = float(np.std(y))
        if self.target_std_ <= 1e-8:
            self.target_std_ = 1.0

        x_scaled = (x - self.feature_mean_) / self.feature_std_
        y_scaled = (y.astype("float32") - self.target_mean_) / self.target_std_
        self.normalized_adjacency_ = _feature_correlation_adjacency(x_scaled)
        self.adjacency_ = self.normalized_adjacency_.copy()
        x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
        self.model_ = _GCNNet(
            feature_count=self.feature_count_,
            hidden_size=int(self.hidden_size),
            normalized_adjacency=self.normalized_adjacency_,
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=float(self.learning_rate))
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(int(self.epochs)):
            optimizer.zero_grad()
            loss = loss_fn(self.model_(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        self.model_.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("GCNGraphRegressor expects a 2D feature matrix")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        with torch.no_grad():
            output = self.model_(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_


@dataclass
class GraphFeatureRidgeRegressor:
    graph_kind: str
    alpha: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GraphFeatureRidgeRegressor":
        if x.ndim != 2:
            raise ValueError("GraphFeatureRidgeRegressor expects a 2D feature matrix")
        centered_y = y.astype(float) - float(np.mean(y))
        numerator = (x - x.mean(axis=0)) * centered_y.reshape(-1, 1)
        denominator = x.std(axis=0) * (np.std(centered_y) + 1e-12)
        self.relation_strength_ = numerator.mean(axis=0) / np.where(denominator <= 1e-12, 1.0, denominator)
        self.relation_strength_ = np.nan_to_num(self.relation_strength_, nan=0.0, posinf=0.0, neginf=0.0)
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model_.fit(self._features(x), y.astype(float))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("GraphFeatureRidgeRegressor expects a 2D feature matrix")
        return self.model_.predict(self._features(x))

    def _features(self, x: np.ndarray) -> np.ndarray:
        adjacency = _chain_adjacency(x.shape[1])
        propagated = x @ adjacency
        market_context = np.repeat(x.mean(axis=1, keepdims=True), x.shape[1], axis=1)

        if self.graph_kind == "market":
            return np.concatenate([x, propagated, market_context, x - market_context], axis=1)
        if self.graph_kind == "gcn":
            return np.concatenate([x, propagated, propagated @ adjacency], axis=1)
        if self.graph_kind == "gat":
            return _attention_features(x)
        if self.graph_kind == "rgcn":
            relation_parts = []
            for group in _relation_groups(x.shape[1]):
                part = np.zeros_like(x)
                part[:, group] = x[:, group].mean(axis=1, keepdims=True)
                relation_parts.append(part)
            return np.concatenate([x, *relation_parts], axis=1)
        if self.graph_kind == "rsr":
            strength = getattr(self, "relation_strength_", np.ones(x.shape[1]))
            relation_score = (x * strength).sum(axis=1, keepdims=True)
            return np.concatenate([x, x * strength, relation_score], axis=1)
        if self.graph_kind == "hypergraph":
            return _hypergraph_features(x)
        if self.graph_kind == "thgnn":
            return np.concatenate([_hypergraph_features(x), _lag_features(x)], axis=1)
        if self.graph_kind == "sthan":
            return np.concatenate([_attention_features(x), _lag_features(x), market_context], axis=1)
        if self.graph_kind == "sthgcn":
            return np.concatenate([_hypergraph_features(x), propagated, _lag_features(x)], axis=1)
        raise ValueError(f"unsupported graph feature kind: {self.graph_kind}")

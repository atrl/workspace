from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _chain_adjacency(n_features: int) -> np.ndarray:
    adjacency = np.eye(n_features)
    for index in range(n_features - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.where(degree <= 1e-12, 1.0, degree)


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

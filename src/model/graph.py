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


def _feature_market_strength(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("market strength expects a 2D feature matrix")
    market = x.mean(axis=1)
    centered_market = market - float(np.mean(market))
    numerator = (x - x.mean(axis=0)) * centered_market.reshape(-1, 1)
    denominator = x.std(axis=0) * (np.std(centered_market) + 1e-12)
    strength = numerator.mean(axis=0) / np.where(denominator <= 1e-12, 1.0, denominator)
    return np.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _relation_groups(n_features: int) -> tuple[np.ndarray, ...]:
    if n_features < 3:
        return (np.arange(n_features),)
    cuts = np.array_split(np.arange(n_features), 3)
    return tuple(group for group in cuts if len(group) > 0)


def _relation_adjacencies(n_features: int) -> np.ndarray:
    if n_features < 1:
        raise ValueError("n_features must be positive")
    groups = _relation_groups(n_features)
    identity = np.eye(n_features, dtype=float)
    intra_group = np.zeros((n_features, n_features), dtype=float)
    for group in groups:
        intra_group[np.ix_(group, group)] = 1.0
    inter_group = np.ones((n_features, n_features), dtype=float) - intra_group
    np.fill_diagonal(inter_group, 0.0)
    return np.stack(
        [
            _symmetric_normalized_adjacency(identity),
            _symmetric_normalized_adjacency(intra_group),
            _symmetric_normalized_adjacency(inter_group),
        ],
        axis=0,
    ).astype("float32")


def _lag_features(x: np.ndarray) -> np.ndarray:
    previous = np.vstack([x[:1], x[:-1]])
    two_day = np.vstack([x[:2], x[:-2]]) if len(x) > 1 else previous
    return np.concatenate([previous, x - previous, x - two_day], axis=1)


def _temporal_node_channels(x: np.ndarray) -> np.ndarray:
    previous = np.vstack([x[:1], x[:-1]])
    delta = x - previous
    return np.stack([x, previous, delta], axis=-1).astype("float32")


def _spatio_temporal_node_channels(x: np.ndarray) -> np.ndarray:
    previous = np.vstack([x[:1], x[:-1]])
    two_day = np.vstack([x[:2], x[:-2]]) if len(x) > 1 else previous
    delta = x - previous
    two_day_delta = x - two_day
    return np.stack([x, previous, delta, two_day_delta], axis=-1).astype("float32")


def _target_relation_strength(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_y = y.astype(float) - float(np.mean(y))
    numerator = (x - x.mean(axis=0)) * centered_y.reshape(-1, 1)
    denominator = x.std(axis=0) * (np.std(centered_y) + 1e-12)
    strength = numerator.mean(axis=0) / np.where(denominator <= 1e-12, 1.0, denominator)
    return np.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _hypergraph_features(x: np.ndarray) -> np.ndarray:
    groups = _relation_groups(x.shape[1])
    means = [x[:, group].mean(axis=1, keepdims=True) for group in groups]
    stds = [x[:, group].std(axis=1, keepdims=True) for group in groups]
    expanded_means = np.concatenate([np.repeat(mean, len(group), axis=1) for mean, group in zip(means, groups)], axis=1)
    return np.concatenate([x, *means, *stds, x - expanded_means], axis=1)


def _hypergraph_incidence(n_features: int) -> np.ndarray:
    if n_features < 1:
        raise ValueError("n_features must be positive")
    hyperedges: list[np.ndarray] = [np.arange(n_features)]
    hyperedges.extend(_relation_groups(n_features))
    if n_features > 1:
        hyperedges.extend(np.asarray([index, index + 1]) for index in range(n_features - 1))

    incidence = np.zeros((n_features, len(hyperedges)), dtype="float32")
    for edge_index, edge_nodes in enumerate(hyperedges):
        incidence[edge_nodes.astype(int), edge_index] = 1.0
    return incidence


def _hypergraph_propagation(incidence: np.ndarray) -> np.ndarray:
    if incidence.ndim != 2:
        raise ValueError("incidence must be a 2D matrix")
    hypergraph = np.asarray(incidence, dtype=float)
    edge_degree = hypergraph.sum(axis=0)
    node_degree = hypergraph.sum(axis=1)
    edge_inv = 1.0 / np.where(edge_degree <= 1e-12, 1.0, edge_degree)
    node_inv_sqrt = 1.0 / np.sqrt(np.where(node_degree <= 1e-12, 1.0, node_degree))
    propagation = node_inv_sqrt.reshape(-1, 1) * (hypergraph * edge_inv.reshape(1, -1)) @ hypergraph.T
    propagation = propagation * node_inv_sqrt.reshape(1, -1)
    return np.nan_to_num(propagation, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _market_structure_adjacency(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("market structure adjacency expects a 2D matrix")
    correlation = _feature_correlation_adjacency(x)
    chain = _chain_adjacency(x.shape[1])
    market = np.ones((x.shape[1], x.shape[1]), dtype=float) / max(1, x.shape[1])
    strength = np.abs(_feature_market_strength(x))
    adjacency = _symmetric_normalized_adjacency(correlation + chain + market + np.outer(strength, strength))
    return adjacency, strength.astype("float32")


def _temporal_heterogeneous_adjacencies(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("temporal heterogeneous adjacencies expect a 2D matrix")
    previous = np.vstack([x[:1], x[:-1]])
    incidence = _hypergraph_incidence(x.shape[1])
    relations = [
        np.eye(x.shape[1], dtype=float),
        _chain_adjacency(x.shape[1]),
        _hypergraph_propagation(incidence),
        _feature_correlation_adjacency(x),
        _feature_correlation_adjacency(previous),
    ]
    return np.stack([_symmetric_normalized_adjacency(relation) for relation in relations], axis=0).astype("float32")


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


class _GATNet(nn.Module):
    def __init__(self, *, feature_count: int, hidden_size: int, attention_prior: np.ndarray):
        super().__init__()
        import torch

        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(1, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.attention_source = nn.Linear(hidden_size, 1, bias=False)
        self.attention_target = nn.Linear(hidden_size, 1, bias=False)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.register_buffer("attention_prior", torch.as_tensor(attention_prior, dtype=torch.float32))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def _attention(self, hidden):
        import torch

        source_score = self.attention_source(hidden)
        target_score = self.attention_target(hidden).transpose(1, 2)
        prior = torch.log(self.attention_prior.clamp_min(1e-6)).reshape(1, self.feature_count, self.feature_count)
        scores = self.leaky_relu(source_score + target_score + prior)
        return torch.softmax(scores, dim=-1)

    def forward(self, x, *, return_attention: bool = False):
        import torch

        hidden = torch.relu(self.input_projection(x.unsqueeze(-1)))
        attention = self._attention(hidden)
        context = torch.einsum("bij,bjh->bih", attention, hidden)
        output = torch.relu(self.output_projection(context))
        pooled = torch.cat([output.mean(dim=1), output.max(dim=1).values], dim=1)
        prediction = self.head(pooled).squeeze(-1)
        if return_attention:
            return prediction, attention
        return prediction


class _MarketGraphNet(nn.Module):
    def __init__(
        self,
        *,
        feature_count: int,
        hidden_size: int,
        market_adjacency: np.ndarray,
        market_strength: np.ndarray,
    ):
        super().__init__()
        import torch

        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(1, hidden_size)
        self.market_projection = nn.Linear(1, hidden_size)
        self.graph_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.market_gate = nn.Parameter(torch.as_tensor(market_strength, dtype=torch.float32))
        self.register_buffer("market_adjacency", torch.as_tensor(market_adjacency, dtype=torch.float32))

    def forward(self, x):
        import torch

        hidden = torch.relu(self.input_projection(x.unsqueeze(-1)))
        market_context = x.mean(dim=1, keepdim=True).repeat(1, self.feature_count)
        gate = torch.sigmoid(self.market_gate).reshape(1, self.feature_count, 1)
        market_hidden = torch.relu(self.market_projection(market_context.unsqueeze(-1))) * gate
        propagated = torch.einsum("ij,bjh->bih", self.market_adjacency, hidden)
        hidden = torch.relu(self.graph_projection(propagated + market_hidden))
        hidden = torch.relu(self.output_projection(hidden))
        pooled = torch.cat([hidden.mean(dim=1), hidden.max(dim=1).values], dim=1)
        return self.head(pooled).squeeze(-1)


class _RGCNNet(nn.Module):
    def __init__(self, *, feature_count: int, hidden_size: int, relation_adjacencies: np.ndarray):
        super().__init__()
        import torch

        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.relation_count = int(relation_adjacencies.shape[0])
        self.input_projection = nn.Linear(1, hidden_size)
        self.root_projection = nn.Linear(hidden_size, hidden_size)
        self.relation_projections = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(self.relation_count))
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.register_buffer("relation_adjacencies", torch.as_tensor(relation_adjacencies, dtype=torch.float32))

    def forward(self, x):
        import torch

        hidden = torch.relu(self.input_projection(x.unsqueeze(-1)))
        messages = []
        for relation_index, projection in enumerate(self.relation_projections):
            propagated = torch.einsum("ij,bjh->bih", self.relation_adjacencies[relation_index], hidden)
            messages.append(projection(propagated))
        combined = self.root_projection(hidden) + torch.stack(messages, dim=0).mean(dim=0)
        output = torch.relu(self.output_projection(torch.relu(combined)))
        pooled = torch.cat([output.mean(dim=1), output.max(dim=1).values], dim=1)
        return self.head(pooled).squeeze(-1)


class _RSRNet(nn.Module):
    def __init__(
        self,
        *,
        feature_count: int,
        temporal_channel_count: int,
        hidden_size: int,
        relation_strength: np.ndarray,
    ):
        super().__init__()
        import torch

        self.feature_count = feature_count
        self.temporal_channel_count = temporal_channel_count
        self.hidden_size = hidden_size
        strength = np.asarray(relation_strength, dtype="float32")
        adjacency = _symmetric_normalized_adjacency(np.outer(np.abs(strength), np.abs(strength)))
        self.input_projection = nn.Linear(temporal_channel_count, hidden_size)
        self.relation_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.node_gate = nn.Parameter(torch.as_tensor(strength, dtype=torch.float32))
        self.register_buffer("relation_adjacency", torch.as_tensor(adjacency, dtype=torch.float32))

    def forward(self, x):
        import torch

        hidden = torch.relu(self.input_projection(x))
        gate = torch.sigmoid(self.node_gate).reshape(1, self.feature_count, 1)
        gated = hidden * gate
        propagated = torch.einsum("ij,bjh->bih", self.relation_adjacency, gated)
        hidden = torch.relu(self.relation_projection(propagated) + gated)
        hidden = torch.relu(self.output_projection(hidden))
        pooled = torch.cat([hidden.mean(dim=1), hidden.max(dim=1).values], dim=1)
        return self.head(pooled).squeeze(-1)


class _HypergraphNet(nn.Module):
    def __init__(
        self,
        *,
        feature_count: int,
        hidden_size: int,
        incidence_matrix: np.ndarray,
        hypergraph_propagation: np.ndarray,
    ):
        super().__init__()
        import torch

        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(1, hidden_size)
        self.hyperedge_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.register_buffer("incidence_matrix", torch.as_tensor(incidence_matrix, dtype=torch.float32))
        self.register_buffer(
            "hypergraph_propagation",
            torch.as_tensor(hypergraph_propagation, dtype=torch.float32),
        )

    def forward(self, x):
        import torch

        hidden = torch.relu(self.input_projection(x.unsqueeze(-1)))
        hypergraph_context = torch.einsum("ij,bjh->bih", self.hypergraph_propagation, hidden)
        hidden = torch.relu(self.hyperedge_projection(hypergraph_context) + hidden)
        hidden = torch.relu(self.output_projection(torch.einsum("ij,bjh->bih", self.hypergraph_propagation, hidden)))
        pooled = torch.cat([hidden.mean(dim=1), hidden.max(dim=1).values], dim=1)
        return self.head(pooled).squeeze(-1)


class _STHANNet(nn.Module):
    def __init__(
        self,
        *,
        feature_count: int,
        temporal_channel_count: int,
        hidden_size: int,
        hypergraph_propagation: np.ndarray,
    ):
        super().__init__()
        import torch

        self.feature_count = feature_count
        self.temporal_channel_count = temporal_channel_count
        self.hidden_size = hidden_size
        self.temporal_attention_logits = nn.Parameter(torch.zeros(temporal_channel_count, dtype=torch.float32))
        decay = np.exp(-np.arange(temporal_channel_count, dtype=float) / max(1.0, temporal_channel_count - 1.0))
        self.register_buffer("temporal_decay", torch.as_tensor(decay, dtype=torch.float32))
        self.input_projection = nn.Linear(1, hidden_size)
        self.hypergraph_projection = nn.Linear(hidden_size, hidden_size)
        self.spatial_source = nn.Linear(hidden_size, 1, bias=False)
        self.spatial_target = nn.Linear(hidden_size, 1, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.register_buffer(
            "hypergraph_propagation",
            torch.as_tensor(hypergraph_propagation, dtype=torch.float32),
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def _temporal_attention(self):
        import torch

        prior = torch.log(self.temporal_decay.clamp_min(1e-6))
        return torch.softmax(self.temporal_attention_logits + prior, dim=0)

    def _spatial_attention(self, hidden):
        import torch

        source_score = self.spatial_source(hidden)
        target_score = self.spatial_target(hidden).transpose(1, 2)
        prior = torch.log(self.hypergraph_propagation.clamp_min(1e-6)).reshape(
            1,
            self.feature_count,
            self.feature_count,
        )
        return torch.softmax(self.leaky_relu(source_score + target_score + prior), dim=-1)

    def forward(self, x, *, return_attention: bool = False):
        import torch

        temporal_attention = self._temporal_attention()
        node_signal = torch.einsum("bnt,t->bn", x, temporal_attention)
        hidden = torch.relu(self.input_projection(node_signal.unsqueeze(-1)))
        hypergraph_context = torch.einsum("ij,bjh->bih", self.hypergraph_propagation, hidden)
        hidden = torch.relu(self.hypergraph_projection(hypergraph_context) + hidden)
        spatial_attention = self._spatial_attention(hidden)
        spatial_context = torch.einsum("bij,bjh->bih", spatial_attention, hidden)
        output = torch.relu(self.output_projection(spatial_context))
        pooled = torch.cat([output.mean(dim=1), output.max(dim=1).values], dim=1)
        prediction = self.head(pooled).squeeze(-1)
        if return_attention:
            return prediction, temporal_attention, spatial_attention
        return prediction


class _STHGCNNet(nn.Module):
    def __init__(
        self,
        *,
        feature_count: int,
        temporal_channel_count: int,
        hidden_size: int,
        hypergraph_propagation: np.ndarray,
    ):
        super().__init__()
        import torch

        self.feature_count = feature_count
        self.temporal_channel_count = temporal_channel_count
        self.hidden_size = hidden_size
        self.graph_convolution_count = 2
        self.temporal_convolution = nn.Conv1d(1, hidden_size, kernel_size=2)
        self.spatial_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.register_buffer(
            "hypergraph_propagation",
            torch.as_tensor(hypergraph_propagation, dtype=torch.float32),
        )

    def forward(self, x):
        import torch

        batch_size = int(x.shape[0])
        temporal_signal = x.reshape(batch_size * self.feature_count, 1, self.temporal_channel_count)
        temporal_hidden = torch.relu(self.temporal_convolution(temporal_signal)).mean(dim=-1)
        hidden = temporal_hidden.reshape(batch_size, self.feature_count, self.hidden_size)
        propagated = torch.einsum("ij,bjh->bih", self.hypergraph_propagation, hidden)
        hidden = torch.relu(self.spatial_projection(propagated))
        propagated = torch.einsum("ij,bjh->bih", self.hypergraph_propagation, hidden)
        hidden = torch.relu(self.output_projection(propagated) + hidden)
        pooled = torch.cat([hidden.mean(dim=1), hidden.max(dim=1).values], dim=1)
        return self.head(pooled).squeeze(-1)


class _THGNNNet(nn.Module):
    def __init__(
        self,
        *,
        feature_count: int,
        temporal_channel_count: int,
        hidden_size: int,
        relation_adjacencies: np.ndarray,
    ):
        super().__init__()
        import torch

        self.feature_count = feature_count
        self.temporal_channel_count = temporal_channel_count
        self.hidden_size = hidden_size
        self.relation_count = int(relation_adjacencies.shape[0])
        self.temporal_attention_logits = nn.Parameter(torch.zeros(temporal_channel_count, dtype=torch.float32))
        self.relation_attention_logits = nn.Parameter(torch.zeros(self.relation_count, dtype=torch.float32))
        decay = np.exp(-np.arange(temporal_channel_count, dtype=float) / max(1.0, temporal_channel_count - 1.0))
        self.register_buffer("temporal_decay", torch.as_tensor(decay, dtype=torch.float32))
        self.input_projection = nn.Linear(1, hidden_size)
        self.root_projection = nn.Linear(hidden_size, hidden_size)
        self.relation_projections = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(self.relation_count))
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.register_buffer("relation_adjacencies", torch.as_tensor(relation_adjacencies, dtype=torch.float32))

    def _temporal_attention(self):
        import torch

        prior = torch.log(self.temporal_decay.clamp_min(1e-6))
        return torch.softmax(self.temporal_attention_logits + prior, dim=0)

    def _relation_attention(self):
        import torch

        return torch.softmax(self.relation_attention_logits, dim=0)

    def forward(self, x, *, return_attention: bool = False):
        import torch

        temporal_attention = self._temporal_attention()
        relation_attention = self._relation_attention()
        node_signal = torch.einsum("bnt,t->bn", x, temporal_attention)
        hidden = torch.relu(self.input_projection(node_signal.unsqueeze(-1)))
        messages = []
        for relation_index, projection in enumerate(self.relation_projections):
            propagated = torch.einsum("ij,bjh->bih", self.relation_adjacencies[relation_index], hidden)
            messages.append(projection(propagated) * relation_attention[relation_index])
        combined = self.root_projection(hidden) + torch.stack(messages, dim=0).sum(dim=0)
        output = torch.relu(self.output_projection(torch.relu(combined)))
        pooled = torch.cat([output.mean(dim=1), output.max(dim=1).values], dim=1)
        prediction = self.head(pooled).squeeze(-1)
        if return_attention:
            return prediction, temporal_attention, relation_attention
        return prediction


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
class GATGraphRegressor:
    hidden_size: int = 8
    epochs: int = 32
    learning_rate: float = 0.01
    seed: int = 43

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GATGraphRegressor":
        if x.ndim != 2:
            raise ValueError("GATGraphRegressor expects a 2D feature matrix")
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
        self.attention_prior_ = _feature_correlation_adjacency(x_scaled)
        x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
        self.model_ = _GATNet(
            feature_count=self.feature_count_,
            hidden_size=int(self.hidden_size),
            attention_prior=self.attention_prior_,
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
        with torch.no_grad():
            _, attention = self.model_(x_tensor, return_attention=True)
        self.attention_weights_ = attention.mean(dim=0).detach().cpu().numpy().astype("float32")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("GATGraphRegressor expects a 2D feature matrix")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        with torch.no_grad():
            output = self.model_(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_


@dataclass
class MarketGraphRegressor:
    hidden_size: int = 8
    epochs: int = 32
    learning_rate: float = 0.01
    seed: int = 39

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MarketGraphRegressor":
        if x.ndim != 2:
            raise ValueError("MarketGraphRegressor expects a 2D feature matrix")
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
        self.market_adjacency_, self.market_strength_ = _market_structure_adjacency(x_scaled)
        x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
        self.model_ = _MarketGraphNet(
            feature_count=self.feature_count_,
            hidden_size=int(self.hidden_size),
            market_adjacency=self.market_adjacency_,
            market_strength=self.market_strength_,
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
        self.market_context_gate_ = torch.sigmoid(self.model_.market_gate).detach().cpu().numpy().astype("float32")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("MarketGraphRegressor expects a 2D feature matrix")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        with torch.no_grad():
            output = self.model_(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_


@dataclass
class RGCNGraphRegressor:
    hidden_size: int = 8
    epochs: int = 32
    learning_rate: float = 0.01
    seed: int = 47

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RGCNGraphRegressor":
        if x.ndim != 2:
            raise ValueError("RGCNGraphRegressor expects a 2D feature matrix")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.feature_count_ = int(x.shape[1])
        self.node_count_ = self.feature_count_
        self.relation_adjacencies_ = _relation_adjacencies(self.feature_count_)
        self.relation_count_ = int(self.relation_adjacencies_.shape[0])
        self.feature_mean_ = x.mean(axis=0)
        self.feature_std_ = x.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ <= 1e-8, 1.0, self.feature_std_)
        self.target_mean_ = float(np.mean(y))
        self.target_std_ = float(np.std(y))
        if self.target_std_ <= 1e-8:
            self.target_std_ = 1.0

        x_scaled = (x - self.feature_mean_) / self.feature_std_
        y_scaled = (y.astype("float32") - self.target_mean_) / self.target_std_
        x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
        self.model_ = _RGCNNet(
            feature_count=self.feature_count_,
            hidden_size=int(self.hidden_size),
            relation_adjacencies=self.relation_adjacencies_,
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
            raise ValueError("RGCNGraphRegressor expects a 2D feature matrix")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        with torch.no_grad():
            output = self.model_(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_


@dataclass
class RSRGraphRegressor:
    hidden_size: int = 8
    epochs: int = 32
    learning_rate: float = 0.01
    seed: int = 53

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RSRGraphRegressor":
        if x.ndim != 2:
            raise ValueError("RSRGraphRegressor expects a 2D feature matrix")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.feature_count_ = int(x.shape[1])
        self.node_count_ = self.feature_count_
        self.temporal_channel_count_ = 3
        self.feature_mean_ = x.mean(axis=0)
        self.feature_std_ = x.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ <= 1e-8, 1.0, self.feature_std_)
        self.target_mean_ = float(np.mean(y))
        self.target_std_ = float(np.std(y))
        if self.target_std_ <= 1e-8:
            self.target_std_ = 1.0

        x_scaled = (x - self.feature_mean_) / self.feature_std_
        y_scaled = (y.astype("float32") - self.target_mean_) / self.target_std_
        self.relation_strength_ = _target_relation_strength(x_scaled, y_scaled)
        x_tensor = torch.as_tensor(_temporal_node_channels(x_scaled), dtype=torch.float32)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
        self.model_ = _RSRNet(
            feature_count=self.feature_count_,
            temporal_channel_count=self.temporal_channel_count_,
            hidden_size=int(self.hidden_size),
            relation_strength=self.relation_strength_,
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
        self.relation_gate_ = torch.sigmoid(self.model_.node_gate).detach().cpu().numpy().astype("float32")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("RSRGraphRegressor expects a 2D feature matrix")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        x_tensor = torch.as_tensor(_temporal_node_channels(x_scaled), dtype=torch.float32)
        with torch.no_grad():
            output = self.model_(x_tensor).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_


@dataclass
class HypergraphGraphRegressor:
    hidden_size: int = 8
    epochs: int = 32
    learning_rate: float = 0.01
    seed: int = 59

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HypergraphGraphRegressor":
        if x.ndim != 2:
            raise ValueError("HypergraphGraphRegressor expects a 2D feature matrix")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.feature_count_ = int(x.shape[1])
        self.node_count_ = self.feature_count_
        self.incidence_matrix_ = _hypergraph_incidence(self.feature_count_)
        self.hyperedge_count_ = int(self.incidence_matrix_.shape[1])
        self.hypergraph_propagation_ = _hypergraph_propagation(self.incidence_matrix_)
        self.feature_mean_ = x.mean(axis=0)
        self.feature_std_ = x.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ <= 1e-8, 1.0, self.feature_std_)
        self.target_mean_ = float(np.mean(y))
        self.target_std_ = float(np.std(y))
        if self.target_std_ <= 1e-8:
            self.target_std_ = 1.0

        x_scaled = (x - self.feature_mean_) / self.feature_std_
        y_scaled = (y.astype("float32") - self.target_mean_) / self.target_std_
        x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
        self.model_ = _HypergraphNet(
            feature_count=self.feature_count_,
            hidden_size=int(self.hidden_size),
            incidence_matrix=self.incidence_matrix_,
            hypergraph_propagation=self.hypergraph_propagation_,
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
            raise ValueError("HypergraphGraphRegressor expects a 2D feature matrix")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        with torch.no_grad():
            output = self.model_(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_


@dataclass
class STHANGraphRegressor:
    hidden_size: int = 8
    epochs: int = 32
    learning_rate: float = 0.01
    seed: int = 61

    def fit(self, x: np.ndarray, y: np.ndarray) -> "STHANGraphRegressor":
        if x.ndim != 2:
            raise ValueError("STHANGraphRegressor expects a 2D feature matrix")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.feature_count_ = int(x.shape[1])
        self.node_count_ = self.feature_count_
        self.temporal_channel_count_ = 4
        self.incidence_matrix_ = _hypergraph_incidence(self.feature_count_)
        self.hyperedge_count_ = int(self.incidence_matrix_.shape[1])
        self.hypergraph_propagation_ = _hypergraph_propagation(self.incidence_matrix_)
        self.feature_mean_ = x.mean(axis=0)
        self.feature_std_ = x.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ <= 1e-8, 1.0, self.feature_std_)
        self.target_mean_ = float(np.mean(y))
        self.target_std_ = float(np.std(y))
        if self.target_std_ <= 1e-8:
            self.target_std_ = 1.0

        x_scaled = (x - self.feature_mean_) / self.feature_std_
        y_scaled = (y.astype("float32") - self.target_mean_) / self.target_std_
        x_tensor = torch.as_tensor(_spatio_temporal_node_channels(x_scaled), dtype=torch.float32)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
        self.model_ = _STHANNet(
            feature_count=self.feature_count_,
            temporal_channel_count=self.temporal_channel_count_,
            hidden_size=int(self.hidden_size),
            hypergraph_propagation=self.hypergraph_propagation_,
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
        with torch.no_grad():
            _, temporal_attention, spatial_attention = self.model_(x_tensor, return_attention=True)
        self.temporal_attention_weights_ = temporal_attention.detach().cpu().numpy().astype("float32")
        self.spatial_attention_weights_ = spatial_attention.mean(dim=0).detach().cpu().numpy().astype("float32")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("STHANGraphRegressor expects a 2D feature matrix")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        x_tensor = torch.as_tensor(_spatio_temporal_node_channels(x_scaled), dtype=torch.float32)
        with torch.no_grad():
            output = self.model_(x_tensor).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_


@dataclass
class THGNNGraphRegressor:
    hidden_size: int = 8
    epochs: int = 32
    learning_rate: float = 0.01
    seed: int = 67

    def fit(self, x: np.ndarray, y: np.ndarray) -> "THGNNGraphRegressor":
        if x.ndim != 2:
            raise ValueError("THGNNGraphRegressor expects a 2D feature matrix")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.feature_count_ = int(x.shape[1])
        self.node_count_ = self.feature_count_
        self.temporal_channel_count_ = 4
        self.feature_mean_ = x.mean(axis=0)
        self.feature_std_ = x.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ <= 1e-8, 1.0, self.feature_std_)
        self.target_mean_ = float(np.mean(y))
        self.target_std_ = float(np.std(y))
        if self.target_std_ <= 1e-8:
            self.target_std_ = 1.0

        x_scaled = (x - self.feature_mean_) / self.feature_std_
        y_scaled = (y.astype("float32") - self.target_mean_) / self.target_std_
        self.relation_adjacencies_ = _temporal_heterogeneous_adjacencies(x_scaled)
        self.relation_count_ = int(self.relation_adjacencies_.shape[0])
        x_tensor = torch.as_tensor(_spatio_temporal_node_channels(x_scaled), dtype=torch.float32)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
        self.model_ = _THGNNNet(
            feature_count=self.feature_count_,
            temporal_channel_count=self.temporal_channel_count_,
            hidden_size=int(self.hidden_size),
            relation_adjacencies=self.relation_adjacencies_,
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
        with torch.no_grad():
            _, temporal_attention, relation_attention = self.model_(x_tensor, return_attention=True)
        self.temporal_attention_weights_ = temporal_attention.detach().cpu().numpy().astype("float32")
        self.relation_attention_weights_ = relation_attention.detach().cpu().numpy().astype("float32")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("THGNNGraphRegressor expects a 2D feature matrix")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        x_tensor = torch.as_tensor(_spatio_temporal_node_channels(x_scaled), dtype=torch.float32)
        with torch.no_grad():
            output = self.model_(x_tensor).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_


@dataclass
class STHGCNGraphRegressor:
    hidden_size: int = 8
    epochs: int = 32
    learning_rate: float = 0.01
    seed: int = 71

    def fit(self, x: np.ndarray, y: np.ndarray) -> "STHGCNGraphRegressor":
        if x.ndim != 2:
            raise ValueError("STHGCNGraphRegressor expects a 2D feature matrix")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.feature_count_ = int(x.shape[1])
        self.node_count_ = self.feature_count_
        self.temporal_channel_count_ = 4
        self.incidence_matrix_ = _hypergraph_incidence(self.feature_count_)
        self.hyperedge_count_ = int(self.incidence_matrix_.shape[1])
        self.hypergraph_propagation_ = _hypergraph_propagation(self.incidence_matrix_)
        self.feature_mean_ = x.mean(axis=0)
        self.feature_std_ = x.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ <= 1e-8, 1.0, self.feature_std_)
        self.target_mean_ = float(np.mean(y))
        self.target_std_ = float(np.std(y))
        if self.target_std_ <= 1e-8:
            self.target_std_ = 1.0

        x_scaled = (x - self.feature_mean_) / self.feature_std_
        y_scaled = (y.astype("float32") - self.target_mean_) / self.target_std_
        x_tensor = torch.as_tensor(_spatio_temporal_node_channels(x_scaled), dtype=torch.float32)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
        self.model_ = _STHGCNNet(
            feature_count=self.feature_count_,
            temporal_channel_count=self.temporal_channel_count_,
            hidden_size=int(self.hidden_size),
            hypergraph_propagation=self.hypergraph_propagation_,
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
        self.graph_convolution_count_ = int(self.model_.graph_convolution_count)
        self.temporal_convolution_weights_ = (
            self.model_.temporal_convolution.weight.detach().cpu().numpy().astype("float32")
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("STHGCNGraphRegressor expects a 2D feature matrix")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        x_tensor = torch.as_tensor(_spatio_temporal_node_channels(x_scaled), dtype=torch.float32)
        with torch.no_grad():
            output = self.model_(x_tensor).detach().cpu().numpy()
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

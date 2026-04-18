from __future__ import annotations

import torch


class TiDENet(torch.nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_size: int = 64,
        feature_dim: int = 48,
        observation_window: int = 1,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = obs_dim
        self.output_dim = hidden_size
        self.device = device
        self.feature_dim = feature_dim
        self.window = observation_window
        self.temporal_dim = feature_dim * observation_window
        self.static_dim = max(obs_dim - self.temporal_dim, 0)
        self.proj_dim = max(feature_dim // 2, 8)
        self.feature_projection = torch.nn.Linear(feature_dim, self.proj_dim)
        temporal_flat_dim = self.proj_dim * observation_window
        self.temporal_encoder = torch.nn.Sequential(
            torch.nn.Linear(temporal_flat_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        bottleneck_dim = max(hidden_size // 2, 16)
        self.dense_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, bottleneck_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.dense_decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        if self.static_dim > 0:
            self.static_encoder = torch.nn.Sequential(
                torch.nn.Linear(self.static_dim, hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            self.output = torch.nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        else:
            self.static_encoder = None
            self.output = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, obs, state=None, info=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        orig_leading = obs.shape[:-1]
        if obs.ndim != 2:
            obs = obs.reshape(-1, obs.shape[-1])
        batch = obs.shape[0]
        temporal = obs[:, :self.temporal_dim]
        static = obs[:, self.temporal_dim:] if self.static_dim > 0 else None
        temporal = temporal.reshape(batch, self.window, self.feature_dim)
        projected = self.feature_projection(temporal)
        projected_flat = projected.reshape(batch, -1)
        temporal_encoded = self.temporal_encoder(projected_flat)
        bottleneck = self.dense_encoder(temporal_encoded)
        temporal_decoded = self.dense_decoder(bottleneck)
        temporal_out = temporal_encoded + temporal_decoded
        if self.static_encoder is not None and static is not None:
            static_out = self.static_encoder(static)
            combined = torch.cat([temporal_out, static_out], dim=-1)
        else:
            combined = temporal_out
        out = self.output(combined)
        if orig_leading:
            out = out.reshape(*orig_leading, -1)
        return out, state


__all__ = ["TiDENet"]

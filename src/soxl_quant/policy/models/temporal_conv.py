from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConvNet(nn.Module):
    """Compact 1D temporal convolution encoder for short observation windows."""

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int,
        observation_window: int,
        channels: int,
        kernel_size: int,
        mlp_hidden_sizes: list[int],
        device: str = "cpu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.observation_window = max(int(observation_window), 1)
        self.channels = max(int(channels), 1)
        self.kernel_size = max(int(kernel_size), 1)
        self.device = device
        self.input_dim = int(obs_dim)

        self.temporal_dim = self.feature_dim * self.observation_window
        self.extra_dim = max(self.input_dim - self.temporal_dim, 0)
        padding = self.kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(self.feature_dim, self.channels, self.kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.extend(
            [
                nn.Conv1d(self.channels, self.channels, self.kernel_size, padding=padding),
                nn.ReLU(inplace=True),
            ]
        )
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self.temporal_encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        mlp_input_dim = self.channels + self.extra_dim
        mlp_layers: list[nn.Module] = []
        prev_dim = mlp_input_dim
        for h in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(prev_dim, h))
            mlp_layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                mlp_layers.append(nn.Dropout(p=dropout))
            prev_dim = h
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_dim = prev_dim

        self.to(device)

    def forward(self, obs, state=None, info=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        elif obs.dim() > 2:
            obs = obs.view(-1, obs.shape[-1])

        batch_size = obs.shape[0]
        temporal = obs[:, : self.temporal_dim].reshape(
            batch_size,
            self.observation_window,
            self.feature_dim,
        )
        static = obs[:, self.temporal_dim :] if self.extra_dim > 0 else None

        x = temporal.transpose(1, 2)
        x = self.temporal_encoder(x)
        pooled = self.pool(x).squeeze(-1)
        if static is not None:
            pooled = torch.cat([pooled, static], dim=-1)
        return self.mlp(pooled), None


__all__ = ["TemporalConvNet"]

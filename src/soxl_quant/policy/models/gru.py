from __future__ import annotations

import torch
import torch.nn as nn


class GRUNet(nn.Module):
    """GRU + MLP network for temporal credit assignment.

    Processes a window of observations through GRU, then feeds the final hidden
    state plus any extra features through an MLP head.
    """

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int,
        observation_window: int,
        gru_hidden_size: int,
        mlp_hidden_sizes: list[int],
        device: str = "cpu",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.observation_window = observation_window
        self.gru_hidden_size = gru_hidden_size
        self.device = device

        extra_dim = obs_dim - feature_dim * observation_window
        self.extra_dim = max(extra_dim, 0)

        # Expose input_dim so infer_actor_obs_dim finds the correct observation size
        self.input_dim = obs_dim
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_hidden_size,
            batch_first=True,
        )

        mlp_input_dim = gru_hidden_size + self.extra_dim
        layers = []
        prev_dim = mlp_input_dim
        for h in mlp_hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim

        self.to(device)

    def forward(self, obs, state=None, info=None):
        info = info or {}
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        batch_size = obs.shape[0]
        window_size = self.feature_dim * self.observation_window

        # Extract windowed features and extra features
        if self.observation_window > 1:
            window_features = obs[:, :window_size]
            extra_features = obs[:, window_size:] if self.extra_dim > 0 else None
            # Reshape to (batch, window, feature_dim)
            window_features = window_features.view(
                batch_size, self.observation_window, self.feature_dim
            )
        else:
            # observation_window == 1: treat whole obs as single timestep
            window_features = obs.view(batch_size, 1, obs.shape[1])
            extra_features = None

        # GRU
        gru_out, hidden = self.gru(window_features, state)
        gru_last = gru_out[:, -1, :]  # (batch, gru_hidden_size)

        # Concatenate with extra features
        if extra_features is not None:
            combined = torch.cat([gru_last, extra_features], dim=-1)
        else:
            combined = gru_last

        # MLP head
        out = self.mlp(combined)
        return out, hidden

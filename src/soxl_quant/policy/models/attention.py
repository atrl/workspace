"""Window-level self-attention model for temporal credit assignment.

Intentionally small (no multi-head, no positional encoding beyond a learned
per-step embedding) because the observation window we use is 5-10 steps. A
full Transformer would be overkill and slow down per-profile training.

This is the Tier 3 reserved model path per the next-step plan. Profiles opt
in by setting `model.use_attention: true`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class WindowSelfAttentionNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        feature_dim: int,
        observation_window: int,
        attention_dim: int,
        attention_heads: int,
        mlp_hidden_sizes: list[int],
        device: str = "cpu",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.observation_window = max(int(observation_window), 1)
        self.attention_dim = int(attention_dim)
        self.attention_heads = max(int(attention_heads), 1)
        self.device = device

        extra_dim = obs_dim - feature_dim * self.observation_window
        self.extra_dim = max(extra_dim, 0)
        self.input_dim = obs_dim

        self.input_proj = nn.Linear(feature_dim, self.attention_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.observation_window, self.attention_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.attention_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.attention_dim)

        mlp_input_dim = self.attention_dim + self.extra_dim
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
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        elif obs.dim() > 2:
            obs = obs.view(-1, obs.shape[-1])

        batch_size = obs.shape[0]
        window_size = self.feature_dim * self.observation_window

        if self.observation_window > 1:
            window_features = obs[:, :window_size].view(
                batch_size, self.observation_window, self.feature_dim
            )
            extra_features = (
                obs[:, window_size:] if self.extra_dim > 0 else None
            )
        else:
            window_features = obs.view(batch_size, 1, obs.shape[1])
            extra_features = None

        x = self.input_proj(window_features)
        x = x + self.pos_embed[:, : x.shape[1], :]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        pooled = x.mean(dim=1)

        if extra_features is not None:
            combined = torch.cat([pooled, extra_features], dim=-1)
        else:
            combined = pooled

        out = self.mlp(combined)
        return out, None


__all__ = ["WindowSelfAttentionNet"]

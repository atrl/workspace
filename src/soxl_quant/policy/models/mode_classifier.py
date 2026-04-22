from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class ModeClassifierNet(nn.Module):
    """Small MLP that classifies regime mode + predicts forward drawdown."""

    def __init__(
        self,
        n_features: int = 11,
        n_modes: int = 4,
        hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mode_head = nn.Linear(hidden // 2, n_modes)
        self.dd_head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        mode_logits = self.mode_head(h)
        dd_pred = torch.sigmoid(self.dd_head(h))
        return mode_logits, dd_pred


# Feature order must match training script
DEFAULT_FEATURE_COLS = [
    "regime_risk",
    "realized_vol_10",
    "trend_gap_21",
    "anchor_gap_180",
    "cum_drop_30",
    "relative_strength_21",
    "vix_zscore",
    "downside_ratio_10",
    "drawdown_streak_63",
    "below_anchor_ratio_63",
    "regime_risk_velocity_5",
]

MODE_NAMES = ["flat_wait", "crash_rebuild", "trend_hold", "defensive_hold"]


def load_mode_classifier(path: str, device: str = "cpu") -> ModeClassifierNet:
    checkpoint = torch.load(path, map_location=device)
    feature_cols = checkpoint.get("feature_cols", DEFAULT_FEATURE_COLS)
    hidden = checkpoint.get("hidden", 32)
    model = ModeClassifierNet(
        n_features=len(feature_cols),
        n_modes=len(MODE_NAMES),
        hidden=hidden,
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


class LearnedModeRouter:
    """Runtime wrapper that runs the learned classifier over raw observations."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = load_mode_classifier(model_path, device=device)
        self.feature_cols = DEFAULT_FEATURE_COLS
        self.mode_names = MODE_NAMES

    def classify(self, obs: np.ndarray, static_index: Dict[str, int]) -> str:
        feats = np.array(
            [float(np.nan_to_num(obs[static_index.get(c, 0)])) for c in self.feature_cols],
            dtype=np.float32,
        )
        with torch.no_grad():
            logits, _ = self.model(torch.from_numpy(feats).unsqueeze(0))
            idx = int(logits.argmax(dim=1).item())
        return self.mode_names[idx]


__all__ = ["ModeClassifierNet", "LearnedModeRouter", "MODE_NAMES", "DEFAULT_FEATURE_COLS"]

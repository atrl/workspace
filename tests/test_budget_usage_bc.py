from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dataclasses import replace

from src.soxl_quant.config import load_profile
from src.soxl_quant.data import FEATURE_COLUMNS
from src.soxl_quant.policy.policy_builder import (
    budget_usage_action_label,
    build_budget_usage_warm_start_dataset_from_frame,
)
from src.soxl_quant.settings import LOOKBACK


class FixedActionActor:
    output_dim = 5

    def __init__(self, action: int):
        self.action = int(action)

    def __call__(self, obs: torch.Tensor):
        logits = torch.full((obs.shape[0], self.output_dim), -10.0)
        logits[:, self.action] = 10.0
        return logits, None


def _sample_frame(rows: int = LOOKBACK + 20) -> pd.DataFrame:
    data = {col: np.zeros(rows, dtype=np.float32) for col in FEATURE_COLUMNS}
    data["datetime"] = pd.date_range("2016-01-01", periods=rows, freq="B")
    data["close"] = np.linspace(10.0, 12.0, rows, dtype=np.float32)
    data["return"] = pd.Series(data["close"]).pct_change().fillna(0.0).to_numpy(dtype=np.float32)
    data["regime_risk"] = np.linspace(0.10, 0.30, rows, dtype=np.float32)
    data["anchor_gap_180"] = 0.06
    data["anchor_slope_180"] = 0.01
    data["trend_gap_21"] = 0.04
    data["realized_vol_10"] = 0.30
    return pd.DataFrame(data)


def test_budget_usage_action_label_maps_usage_to_nearest_discrete_action():
    assert budget_usage_action_label(-0.2, 5) == 0
    assert budget_usage_action_label(0.49, 5) == 2
    assert budget_usage_action_label(1.7, 5) == 4


def test_budget_usage_bc_dataset_uses_expert_fast_position_as_usage_label():
    profile = load_profile("risk_budget_usage_v1")
    base_expert_profile = load_profile("balanced_rule_residual_imitation_v4")
    expert_profile = replace(
        base_expert_profile,
        model=replace(
            base_expert_profile.model,
            use_hybrid_overlay=False,
            action_levels=(0.0, 0.25, 0.50, 0.75, 1.0),
            hard_floor=None,
        ),
    )
    frame = _sample_frame()
    obs, high_actions = build_budget_usage_warm_start_dataset_from_frame(
        frame,
        profile=profile,
        expert_actor=FixedActionActor(action=4),
        expert_profile=expert_profile,
        sample_size=8,
    )
    _low_obs, low_actions = build_budget_usage_warm_start_dataset_from_frame(
        frame,
        profile=profile,
        expert_actor=FixedActionActor(action=0),
        expert_profile=expert_profile,
        sample_size=8,
    )

    assert obs.shape[0] == 8
    assert high_actions.min() > low_actions.max()


if __name__ == "__main__":
    test_budget_usage_action_label_maps_usage_to_nearest_discrete_action()
    test_budget_usage_bc_dataset_uses_expert_fast_position_as_usage_label()
    print("test_budget_usage_bc: OK")

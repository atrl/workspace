from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dataclasses import replace

from src.soxl_quant.config import load_profile
from src.soxl_quant.data import FEATURE_COLUMNS
from src.soxl_quant.policy.policy_builder import (
    budget_usage_action_label,
    budget_usage_action_label_from_position,
    build_budget_usage_oracle_dataset_from_frame,
    build_mode_conditioned_budget_usage_oracle_dataset_from_frame,
    build_budget_usage_warm_start_dataset_from_frame,
)
from src.soxl_quant.policy.mode_conditioning import MODE_ONE_HOT_ORDER, append_mode_one_hot
from src.soxl_quant.policy.observation import observation_dim_for_profile
from src.soxl_quant.policy.usage_regression import (
    build_budget_usage_regression_dataset_from_frame,
    usage_target_from_position,
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


def test_budget_usage_action_label_from_position_uses_mode_band():
    profile = load_profile("risk_budget_usage_v1")

    assert budget_usage_action_label_from_position(0.25, "flat_wait", profile, 5) == 4
    assert budget_usage_action_label_from_position(0.45, "trend_hold", profile, 5) == 0
    assert budget_usage_action_label_from_position(0.95, "trend_hold", profile, 5) == 4


def test_budget_usage_oracle_dataset_uses_daily_replay_labels():
    profile = load_profile("risk_budget_usage_v1")
    frame = _sample_frame()
    daily = pd.DataFrame(
        {
            "datetime": frame["datetime"].iloc[LOOKBACK : LOOKBACK + 8].to_numpy(),
            "position": [0.25, 0.25, 0.45, 0.95, 0.10, 0.45, 0.65, 0.00],
            "mode": [
                "flat_wait",
                "flat_wait",
                "trend_hold",
                "trend_hold",
                "defensive_hold",
                "defensive_hold",
                "crash_rebuild",
                "flat_wait",
            ],
        }
    )

    obs, actions = build_budget_usage_oracle_dataset_from_frame(
        frame,
        replay_daily=daily,
        profile=profile,
        sample_size=0,
    )

    assert obs.shape[0] == len(daily)
    assert actions.tolist() == [4, 4, 0, 4, 0, 4, 4, 0]


def test_budget_usage_oracle_dataset_accepts_datetime_index():
    profile = load_profile("risk_budget_usage_v1")
    frame = _sample_frame()
    daily = pd.DataFrame(
        {
            "datetime": frame["datetime"].iloc[LOOKBACK : LOOKBACK + 2].to_numpy(),
            "position": [0.25, 0.95],
            "mode": ["flat_wait", "trend_hold"],
        }
    )
    indexed = frame.set_index("datetime")

    _obs, actions = build_budget_usage_oracle_dataset_from_frame(
        indexed,
        replay_daily=daily,
        profile=profile,
        sample_size=0,
    )

    assert actions.tolist() == [4, 4]


def test_append_mode_one_hot_marks_requested_mode():
    obs = np.array([1.0, 2.0], dtype=np.float32)
    conditioned = append_mode_one_hot(obs, "crash_rebuild")

    assert conditioned.shape[0] == obs.shape[0] + len(MODE_ONE_HOT_ORDER)
    assert conditioned[-4:].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_usage_target_from_position_uses_mode_band():
    profile = load_profile("risk_budget_usage_v1")

    assert usage_target_from_position(0.25, "flat_wait", profile) == 1.0
    assert usage_target_from_position(0.45, "trend_hold", profile) == 0.0
    assert usage_target_from_position(0.95, "trend_hold", profile) == 1.0


def test_budget_usage_bc_dataset_uses_expert_fast_position_as_usage_label():
    profile = load_profile("risk_budget_usage_v1")
    base_expert_profile = load_profile("balanced_rule_residual_v19")
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
    test_budget_usage_action_label_from_position_uses_mode_band()
    test_budget_usage_oracle_dataset_uses_daily_replay_labels()
    test_budget_usage_oracle_dataset_accepts_datetime_index()
    test_append_mode_one_hot_marks_requested_mode()
    test_usage_target_from_position_uses_mode_band()
    test_budget_usage_bc_dataset_uses_expert_fast_position_as_usage_label()
    print("test_budget_usage_bc: OK")

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.soxl_quant.app.state_objective_bc import (
    StateObjectiveBCDataset,
    StateObjectiveBCSampleWeightConfig,
    StateObjectiveBCTrainingConfig,
    action_distribution_frame,
    build_state_objective_bc_dataset_from_frame,
    evaluate_bc_actor,
    fit_bc_actor,
    fit_state_objective_bc_actor_from_frame,
    state_objective_bc_sample_weights,
)
from src.soxl_quant.config import load_profile
from src.soxl_quant.data import FEATURE_COLUMNS
from src.soxl_quant.settings import LOOKBACK


def _separable_actor() -> torch.nn.Linear:
    actor = torch.nn.Linear(2, 2)
    torch.nn.init.zeros_(actor.weight)
    torch.nn.init.zeros_(actor.bias)
    return actor


def test_fit_bc_actor_reports_before_and_after_metrics_on_separable_actions():
    obs = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    actions = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    actor = _separable_actor()

    before = evaluate_bc_actor(actor, obs, actions)
    result = fit_bc_actor(
        actor,
        obs,
        actions,
        StateObjectiveBCTrainingConfig(steps=120, lr=0.05, batch_size=6, seed=7),
    )

    assert before["accuracy"] == 0.5
    assert result["samples"] == 6
    assert result["before_accuracy"] == 0.5
    assert result["after_accuracy"] >= 0.95
    assert result["after_loss"] < result["before_loss"]


def test_action_distribution_frame_keeps_missing_actions_visible():
    dist = action_distribution_frame(np.asarray([0, 2, 2], dtype=np.int64), action_count=4)

    assert dist["action"].tolist() == [0, 1, 2, 3]
    assert dist["count"].tolist() == [1, 0, 2, 0]
    assert dist["share"].round(6).tolist() == [0.333333, 0.0, 0.666667, 0.0]


def test_weighted_bc_actor_can_prioritize_high_weight_minority_label():
    obs = np.zeros((10, 2), dtype=np.float32)
    actions = np.asarray([0] * 9 + [1], dtype=np.int64)
    weights = np.asarray([1.0] * 9 + [25.0], dtype=np.float32)
    actor = _separable_actor()

    result = fit_bc_actor(
        actor,
        obs,
        actions,
        StateObjectiveBCTrainingConfig(steps=80, lr=0.05, batch_size=10, seed=17),
        sample_weight_array=weights,
    )
    logits = actor(torch.as_tensor([[0.0, 0.0]], dtype=torch.float32))

    assert result["weight_mean"] == weights.mean()
    assert int(logits.argmax(-1).item()) == 1


def test_state_objective_bc_sample_weights_prioritize_rebound_and_action_tail():
    metadata = pd.DataFrame(
        {
            "objective_weight": [1.0] * 6,
            "is_rebound_cluster": [False, False, False, False, False, True],
            "is_downtrend_state": [False, False, False, True, True, False],
        }
    )
    actions = np.asarray([0, 0, 0, 0, 1, 2], dtype=np.int64)

    weights = state_objective_bc_sample_weights(
        metadata,
        actions=actions,
        config=StateObjectiveBCSampleWeightConfig(
            rebound_multiplier=6.0,
            downtrend_multiplier=2.0,
            class_balanced_beta=0.99,
        ),
    )

    assert weights.shape == (6,)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[-1] > weights[4] > weights[0]


def test_fit_state_objective_bc_actor_from_frame_accepts_sample_weight_config():
    profile = load_profile("risk_budget_usage_v1")
    frame = _sample_frame()
    dates = frame["datetime"].iloc[LOOKBACK : LOOKBACK + 4].to_numpy()
    targets = pd.DataFrame(
        {
            "datetime": dates,
            "teacher_position": [0.05, 0.25, 0.45, 0.65],
            "rebound_target_position": [0.05, 0.35, 0.45, 0.65],
            "risk_target_position": [0.05, 0.25, 0.25, 0.25],
            "objective_weight": [1.0, 2.0, 3.0, 4.0],
            "is_rebound_cluster": [False, True, False, False],
            "is_downtrend_state": [False, False, True, False],
        }
    )

    result = fit_state_objective_bc_actor_from_frame(
        frame,
        targets=targets,
        profile=profile,
        train_config=StateObjectiveBCTrainingConfig(steps=5, lr=0.01, batch_size=4, seed=11),
        sample_weight_config=StateObjectiveBCSampleWeightConfig(rebound_multiplier=5.0),
    )

    assert result.metrics["weight_max"] > result.metrics["weight_min"]


def _sample_frame(rows: int = LOOKBACK + 8) -> pd.DataFrame:
    data = {col: np.zeros(rows, dtype=np.float32) for col in FEATURE_COLUMNS}
    data["datetime"] = pd.date_range("2020-01-01", periods=rows, freq="B")
    data["close"] = np.linspace(10.0, 12.0, rows, dtype=np.float32)
    data["return"] = pd.Series(data["close"]).pct_change().fillna(0.0).to_numpy(dtype=np.float32)
    data["regime_risk"] = np.linspace(0.10, 0.30, rows, dtype=np.float32)
    data["anchor_gap_180"] = 0.06
    data["anchor_slope_180"] = 0.01
    data["trend_gap_21"] = 0.04
    data["realized_vol_10"] = 0.30
    return pd.DataFrame(data)


def test_fit_state_objective_bc_actor_from_frame_returns_auditable_training_result():
    profile = load_profile("risk_budget_usage_v1")
    frame = _sample_frame()
    dates = frame["datetime"].iloc[LOOKBACK : LOOKBACK + 4].to_numpy()
    targets = pd.DataFrame(
        {
            "datetime": dates,
            "teacher_position": [0.05, 0.25, 0.45, 0.65],
            "rebound_target_position": [0.05, 0.35, 0.45, 0.65],
            "risk_target_position": [0.05, 0.25, 0.25, 0.25],
            "objective_weight": [1.0, 2.0, 3.0, 4.0],
            "is_rebound_cluster": [False, True, False, False],
            "is_downtrend_state": [False, False, True, False],
        }
    )

    result = fit_state_objective_bc_actor_from_frame(
        frame,
        targets=targets,
        profile=profile,
        train_config=StateObjectiveBCTrainingConfig(steps=5, lr=0.01, batch_size=4, seed=11),
        sample_size=0,
    )

    assert result.obs.shape[0] == 4
    assert result.actions.shape == (4,)
    assert result.actor is not None
    assert result.metrics["samples"] == 4
    assert result.action_distribution["count"].sum() == 4


def test_build_state_objective_bc_dataset_from_frame_keeps_target_metadata():
    profile = load_profile("risk_budget_usage_v1")
    frame = _sample_frame()
    dates = frame["datetime"].iloc[LOOKBACK : LOOKBACK + 4].to_numpy()
    targets = pd.DataFrame(
        {
            "datetime": [dates[0], dates[0], dates[1], dates[2]],
            "teacher_position": [0.05, 0.05, 0.45, 0.65],
            "rebound_target_position": [0.05, 0.35, 0.45, 0.65],
            "risk_target_position": [0.05, 0.35, 0.25, 0.25],
            "objective_weight": [1.0, 10.0, 3.0, 4.0],
            "is_rebound_cluster": [False, True, False, False],
            "is_downtrend_state": [False, False, True, False],
            "fold": ["wf_a", "wf_a", "wf_a", "wf_b"],
        }
    )

    dataset = build_state_objective_bc_dataset_from_frame(
        frame,
        targets=targets,
        profile=profile,
        sample_size=0,
    )

    assert isinstance(dataset, StateObjectiveBCDataset)
    assert dataset.obs.shape[0] == 3
    assert dataset.actions.shape == (3,)
    assert dataset.metadata["datetime"].dt.normalize().tolist() == [
        pd.Timestamp(dates[0]).normalize(),
        pd.Timestamp(dates[1]).normalize(),
        pd.Timestamp(dates[2]).normalize(),
    ]
    assert dataset.metadata["is_rebound_cluster"].tolist() == [True, False, False]
    assert dataset.metadata["is_downtrend_state"].tolist() == [False, True, False]
    assert dataset.metadata["fold"].tolist() == ["wf_a", "wf_a", "wf_b"]

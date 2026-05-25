from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.soxl_quant.config import StrategyConfig
from src.soxl_quant.policy.observation import observation_dim_for_profile
from src.soxl_quant.policy.policy_builder import (
    build_policy,
    build_state_objective_warm_start_dataset_from_frame,
    supervised_pretrain,
)


@dataclass(frozen=True)
class StateObjectiveBCTrainingConfig:
    steps: int = 300
    lr: float = 3e-4
    batch_size: int = 64
    seed: int = 0


@dataclass
class StateObjectiveBCFitResult:
    actor: Any
    obs: np.ndarray
    actions: np.ndarray
    metrics: dict[str, float | int]
    action_distribution: pd.DataFrame


def _actor_logits(actor: Any, obs_tensor: torch.Tensor) -> torch.Tensor:
    out = actor(obs_tensor)
    return out[0] if isinstance(out, tuple) else out


def evaluate_bc_actor(
    actor: Any,
    obs_array: np.ndarray,
    action_array: np.ndarray,
    *,
    batch_size: int = 4096,
) -> dict[str, float | int]:
    """Evaluate cross-entropy loss and label accuracy for BC actor logits."""

    if len(obs_array) == 0:
        return {"samples": 0, "loss": float("nan"), "accuracy": float("nan")}

    device = next(actor.parameters()).device
    obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(action_array, dtype=torch.long, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_samples = int(obs_tensor.shape[0])
    with torch.no_grad():
        for start in range(0, total_samples, int(batch_size)):
            stop = min(start + int(batch_size), total_samples)
            logits = _actor_logits(actor, obs_tensor[start:stop])
            labels = action_tensor[start:stop]
            total_loss += float(loss_fn(logits, labels).detach().cpu().item())
            total_correct += int((logits.argmax(-1) == labels).sum().detach().cpu().item())
    return {
        "samples": total_samples,
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def action_distribution_frame(action_array: np.ndarray, *, action_count: int | None = None) -> pd.DataFrame:
    """Return an action-count table with zero-count actions retained."""

    actions = np.asarray(action_array, dtype=np.int64)
    if action_count is None:
        action_count = int(actions.max()) + 1 if len(actions) else 0
    counts = pd.Series(actions, name="action").value_counts().to_dict()
    rows = [
        {"action": int(action), "count": int(counts.get(action, 0))}
        for action in range(int(action_count))
    ]
    result = pd.DataFrame(rows)
    total = int(result["count"].sum()) if not result.empty else 0
    result["share"] = result["count"] / total if total else 0.0
    return result


def fit_bc_actor(
    actor: Any,
    obs_array: np.ndarray,
    action_array: np.ndarray,
    train_config: StateObjectiveBCTrainingConfig = StateObjectiveBCTrainingConfig(),
) -> dict[str, float | int]:
    """Run supervised BC pretraining and return before/after audit metrics."""

    torch.manual_seed(int(train_config.seed))
    before = evaluate_bc_actor(actor, obs_array, action_array)
    train = supervised_pretrain(
        actor,
        obs_array,
        action_array,
        steps=int(train_config.steps),
        lr=float(train_config.lr),
        batch_size=int(train_config.batch_size),
    )
    after = evaluate_bc_actor(actor, obs_array, action_array)
    return {
        "samples": int(after["samples"]),
        "steps": int(train.get("steps", train_config.steps)),
        "lr": float(train_config.lr),
        "batch_size": int(train_config.batch_size),
        "before_loss": float(before["loss"]),
        "before_accuracy": float(before["accuracy"]),
        "train_last_loss": float(train["loss"]),
        "train_last_batch_accuracy": float(train["accuracy"]),
        "after_loss": float(after["loss"]),
        "after_accuracy": float(after["accuracy"]),
    }


def fit_state_objective_bc_actor_from_frame(
    df: pd.DataFrame,
    *,
    targets: pd.DataFrame,
    profile: StrategyConfig,
    train_config: StateObjectiveBCTrainingConfig = StateObjectiveBCTrainingConfig(),
    sample_size: int = 0,
    rng_seed: int = 0,
) -> StateObjectiveBCFitResult:
    """Build state-objective BC labels, instantiate the profile actor, and train it."""

    torch.manual_seed(int(train_config.seed))
    obs, actions = build_state_objective_warm_start_dataset_from_frame(
        df,
        targets=targets,
        profile=profile,
        sample_size=int(sample_size),
        rng_seed=int(rng_seed),
    )
    _policy, actor = build_policy(observation_dim_for_profile(profile), profile)
    metrics = fit_bc_actor(actor, obs, actions, train_config)
    action_count = int(getattr(actor, "output_dim", 0) or (int(actions.max()) + 1 if len(actions) else 0))
    return StateObjectiveBCFitResult(
        actor=actor,
        obs=obs,
        actions=actions,
        metrics=metrics,
        action_distribution=action_distribution_frame(actions, action_count=action_count),
    )


__all__ = [
    "StateObjectiveBCFitResult",
    "StateObjectiveBCTrainingConfig",
    "action_distribution_frame",
    "evaluate_bc_actor",
    "fit_bc_actor",
    "fit_state_objective_bc_actor_from_frame",
]

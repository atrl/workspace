from __future__ import annotations

import random
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
    state_objective_target_position_from_row,
    supervised_pretrain,
)
from src.soxl_quant.settings import LOOKBACK


@dataclass(frozen=True)
class StateObjectiveBCTrainingConfig:
    steps: int = 300
    lr: float = 3e-4
    batch_size: int = 64
    seed: int = 0


@dataclass(frozen=True)
class StateObjectiveBCSampleWeightConfig:
    rebound_multiplier: float = 1.0
    downtrend_multiplier: float = 1.0
    ordinary_multiplier: float = 1.0
    objective_weight_power: float = 0.0
    class_balanced_beta: float = 0.0
    weight_floor: float = 0.05
    weight_cap: float = 50.0
    normalize_mean: bool = True


@dataclass
class StateObjectiveBCFitResult:
    actor: Any
    obs: np.ndarray
    actions: np.ndarray
    metrics: dict[str, float | int]
    action_distribution: pd.DataFrame


@dataclass(frozen=True)
class StateObjectiveBCDataset:
    obs: np.ndarray
    actions: np.ndarray
    metadata: pd.DataFrame


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


def _row_date(df: pd.DataFrame, idx: int) -> pd.Timestamp:
    if "datetime" in df.columns:
        return pd.Timestamp(df.iloc[idx]["datetime"]).normalize()
    return pd.Timestamp(df.index[idx]).normalize()


def _selected_target_metadata(targets: pd.DataFrame) -> pd.DataFrame:
    required = {
        "datetime",
        "teacher_position",
        "rebound_target_position",
        "risk_target_position",
        "objective_weight",
        "is_rebound_cluster",
        "is_downtrend_state",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"state-objective targets missing required columns: {sorted(missing)}")

    frame = targets.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["objective_weight"] = pd.to_numeric(frame["objective_weight"], errors="coerce").fillna(0.0)
    frame["state_objective_target_position"] = [
        state_objective_target_position_from_row(row)
        for row in frame.to_dict("records")
    ]
    return (
        frame.sort_values(["datetime", "objective_weight"], ascending=[True, False])
        .drop_duplicates("datetime", keep="first")
        .reset_index(drop=True)
    )


def build_state_objective_bc_dataset_from_frame(
    df: pd.DataFrame,
    *,
    targets: pd.DataFrame,
    profile: StrategyConfig,
    sample_size: int = 0,
    rng_seed: int = 0,
) -> StateObjectiveBCDataset:
    """Build state-objective BC arrays plus per-sample target metadata."""

    obs, actions = build_state_objective_warm_start_dataset_from_frame(
        df,
        targets=targets,
        profile=profile,
        sample_size=int(sample_size),
        rng_seed=int(rng_seed),
    )
    selected = _selected_target_metadata(targets)
    selected_by_date = {
        pd.Timestamp(row["datetime"]).normalize(): row
        for row in selected.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for idx in range(LOOKBACK, len(df) - 1):
        date = _row_date(df, idx)
        row = selected_by_date.get(date)
        if row is None:
            continue
        rows.append(dict(row))

    if sample_size > 0 and len(rows) > sample_size:
        rng = random.Random(rng_seed)
        indices = sorted(rng.sample(range(len(rows)), int(sample_size)))
        rows = [rows[i] for i in indices]

    metadata = pd.DataFrame(rows).reset_index(drop=True)
    if len(metadata) != len(actions):
        raise RuntimeError(
            "state-objective BC metadata length does not match labels: "
            f"{len(metadata)} != {len(actions)}"
        )
    for col in ("is_rebound_cluster", "is_downtrend_state"):
        if col in metadata.columns:
            metadata[col] = metadata[col].fillna(False).astype(bool)
    return StateObjectiveBCDataset(obs=obs, actions=actions, metadata=metadata)


def _validate_sample_weights(sample_weight_array: np.ndarray, n_samples: int) -> np.ndarray:
    weights = np.asarray(sample_weight_array, dtype=np.float32)
    if weights.shape != (n_samples,):
        raise ValueError(f"sample weights shape {weights.shape} does not match samples {(n_samples,)}")
    if np.any(~np.isfinite(weights)):
        raise ValueError("sample weights must be finite")
    if np.any(weights < 0):
        raise ValueError("sample weights must be non-negative")
    if float(weights.sum()) <= 0.0:
        raise ValueError("sample weights must contain positive mass")
    return weights


def _class_balanced_action_weights(actions: np.ndarray, beta: float) -> np.ndarray:
    beta = float(beta)
    if beta <= 0.0:
        return np.ones(len(actions), dtype=np.float32)
    if beta >= 1.0:
        raise ValueError("class_balanced_beta must be < 1.0")

    action_series = pd.Series(np.asarray(actions, dtype=np.int64), name="action")
    counts = action_series.value_counts().to_dict()
    weights_by_action: dict[int, float] = {}
    for action, count in counts.items():
        n = max(int(count), 1)
        effective_num = (1.0 - beta**n) / (1.0 - beta)
        weights_by_action[int(action)] = 1.0 / max(effective_num, 1e-12)
    raw = np.asarray([weights_by_action[int(action)] for action in actions], dtype=np.float32)
    mean = float(raw.mean()) if len(raw) else 0.0
    return raw / mean if mean > 0.0 else np.ones(len(actions), dtype=np.float32)


def state_objective_bc_sample_weights(
    metadata: pd.DataFrame,
    *,
    actions: np.ndarray | None = None,
    config: StateObjectiveBCSampleWeightConfig = StateObjectiveBCSampleWeightConfig(),
) -> np.ndarray:
    """Build static BC sample weights from state labels and optional action imbalance."""

    n_samples = len(metadata)
    weights = np.full(n_samples, float(config.ordinary_multiplier), dtype=np.float32)
    if n_samples == 0:
        return weights
    if float(config.weight_floor) < 0.0 or float(config.weight_cap) <= 0.0:
        raise ValueError("sample weight floor/cap must be non-negative and positive")
    if float(config.weight_floor) > float(config.weight_cap):
        raise ValueError("sample weight floor must not exceed cap")

    rebound = (
        metadata.get("is_rebound_cluster", pd.Series(False, index=metadata.index))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    downtrend = (
        metadata.get("is_downtrend_state", pd.Series(False, index=metadata.index))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    weights[downtrend] = float(config.downtrend_multiplier)
    weights[rebound] = float(config.rebound_multiplier)

    objective_power = float(config.objective_weight_power)
    if objective_power:
        objective_weight = pd.to_numeric(
            metadata.get("objective_weight", pd.Series(1.0, index=metadata.index)),
            errors="coerce",
        ).fillna(1.0)
        objective_weight = objective_weight.clip(lower=0.0).to_numpy(dtype=np.float32)
        weights *= np.power(objective_weight, objective_power, dtype=np.float32)

    if actions is not None:
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (n_samples,):
            raise ValueError(f"actions shape {action_array.shape} does not match samples {(n_samples,)}")
        weights *= _class_balanced_action_weights(action_array, float(config.class_balanced_beta))

    weights = np.clip(weights, float(config.weight_floor), float(config.weight_cap))
    if bool(config.normalize_mean):
        mean = float(weights.mean())
        if mean > 0.0:
            weights = weights / mean
    return _validate_sample_weights(weights, n_samples)


def _supervised_pretrain_weighted(
    actor: Any,
    obs_array: np.ndarray,
    action_array: np.ndarray,
    sample_weight_array: np.ndarray,
    steps: int,
    lr: float,
    batch_size: int,
) -> dict[str, float]:
    if steps <= 0 or len(obs_array) == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "steps": 0}

    device = next(actor.parameters()).device
    obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(action_array, dtype=torch.long, device=device)
    weight_tensor = torch.as_tensor(sample_weight_array, dtype=torch.float32, device=device)
    optim = torch.optim.Adam(actor.parameters(), lr=float(lr))
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    n = obs_tensor.shape[0]

    last_loss = float("nan")
    last_acc = float("nan")
    for _step in range(int(steps)):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        logits = _actor_logits(actor, obs_tensor[idx])
        labels = action_tensor[idx]
        weights = weight_tensor[idx]
        losses = loss_fn(logits, labels)
        loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
        optim.zero_grad()
        loss.backward()
        optim.step()
        last_loss = float(loss.detach().cpu().item())
        with torch.no_grad():
            preds = logits.argmax(-1)
            last_acc = float((preds == labels).float().mean().item())
    return {"loss": last_loss, "accuracy": last_acc, "steps": int(steps)}


def fit_bc_actor(
    actor: Any,
    obs_array: np.ndarray,
    action_array: np.ndarray,
    train_config: StateObjectiveBCTrainingConfig = StateObjectiveBCTrainingConfig(),
    *,
    sample_weight_array: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Run supervised BC pretraining and return before/after audit metrics."""

    torch.manual_seed(int(train_config.seed))
    before = evaluate_bc_actor(actor, obs_array, action_array)
    weights = None
    if sample_weight_array is None:
        train = supervised_pretrain(
            actor,
            obs_array,
            action_array,
            steps=int(train_config.steps),
            lr=float(train_config.lr),
            batch_size=int(train_config.batch_size),
        )
    else:
        weights = _validate_sample_weights(sample_weight_array, len(obs_array))
        train = _supervised_pretrain_weighted(
            actor,
            obs_array,
            action_array,
            weights,
            steps=int(train_config.steps),
            lr=float(train_config.lr),
            batch_size=int(train_config.batch_size),
        )
    after = evaluate_bc_actor(actor, obs_array, action_array)
    result = {
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
    if weights is not None:
        result.update(
            {
                "weight_mean": float(weights.mean()),
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
            }
        )
    return result


def fit_state_objective_bc_actor_from_frame(
    df: pd.DataFrame,
    *,
    targets: pd.DataFrame,
    profile: StrategyConfig,
    train_config: StateObjectiveBCTrainingConfig = StateObjectiveBCTrainingConfig(),
    sample_weight_config: StateObjectiveBCSampleWeightConfig | None = None,
    sample_size: int = 0,
    rng_seed: int = 0,
) -> StateObjectiveBCFitResult:
    """Build state-objective BC labels, instantiate the profile actor, and train it."""

    torch.manual_seed(int(train_config.seed))
    dataset = build_state_objective_bc_dataset_from_frame(
        df,
        targets=targets,
        profile=profile,
        sample_size=int(sample_size),
        rng_seed=int(rng_seed),
    )
    _policy, actor = build_policy(observation_dim_for_profile(profile), profile)
    sample_weights = (
        state_objective_bc_sample_weights(
            dataset.metadata,
            actions=dataset.actions,
            config=sample_weight_config,
        )
        if sample_weight_config is not None
        else None
    )
    metrics = fit_bc_actor(actor, dataset.obs, dataset.actions, train_config, sample_weight_array=sample_weights)
    action_count = int(
        getattr(actor, "output_dim", 0)
        or (int(dataset.actions.max()) + 1 if len(dataset.actions) else 0)
    )
    return StateObjectiveBCFitResult(
        actor=actor,
        obs=dataset.obs,
        actions=dataset.actions,
        metrics=metrics,
        action_distribution=action_distribution_frame(dataset.actions, action_count=action_count),
    )


__all__ = [
    "StateObjectiveBCFitResult",
    "StateObjectiveBCDataset",
    "StateObjectiveBCSampleWeightConfig",
    "StateObjectiveBCTrainingConfig",
    "action_distribution_frame",
    "build_state_objective_bc_dataset_from_frame",
    "evaluate_bc_actor",
    "fit_bc_actor",
    "fit_state_objective_bc_actor_from_frame",
    "state_objective_bc_sample_weights",
]

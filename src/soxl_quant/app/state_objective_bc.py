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


def _bc_actor_predictions_and_losses(
    actor: Any,
    obs_array: np.ndarray,
    action_array: np.ndarray,
    *,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    if len(obs_array) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
    device = next(actor.parameters()).device
    obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(action_array, dtype=torch.long, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    predictions: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(obs_tensor.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(obs_tensor.shape[0]))
            logits = _actor_logits(actor, obs_tensor[start:stop])
            labels = action_tensor[start:stop]
            predictions.append(logits.argmax(-1).detach().cpu().numpy().astype(np.int64))
            losses.append(loss_fn(logits, labels).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(predictions), np.concatenate(losses)


def evaluate_bc_actor_slice_fit(
    actor: Any,
    obs_array: np.ndarray,
    action_array: np.ndarray,
    metadata: pd.DataFrame,
    *,
    slice_cols: tuple[str, ...] = ("is_rebound_cluster", "is_downtrend_state", "release_gate_state"),
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Evaluate BC label fit on sparse state slices, including release gates."""

    actions = np.asarray(action_array, dtype=np.int64)
    if len(metadata) != len(actions):
        raise ValueError(f"metadata length {len(metadata)} does not match samples {len(actions)}")
    predictions, losses = _bc_actor_predictions_and_losses(
        actor,
        obs_array,
        actions,
        batch_size=int(batch_size),
    )
    work = metadata.copy().reset_index(drop=True)
    work["_label_action"] = actions
    work["_predicted_action"] = predictions
    work["_loss"] = losses

    def _row(label: str, mask: pd.Series) -> dict[str, float | int | str | bool]:
        subset = work[mask.to_numpy(dtype=bool)]
        samples = int(len(subset))
        if samples == 0:
            return {
                "slice": label,
                "samples": 0,
                "accuracy": float("nan"),
                "loss": float("nan"),
                "label_unique_actions": 0,
                "predicted_unique_actions": 0,
                "label_dominant_action": -1,
                "label_dominant_share": 1.0,
                "predicted_dominant_action": -1,
                "predicted_dominant_share": 1.0,
            }
        correct = subset["_label_action"].to_numpy(dtype=np.int64) == subset["_predicted_action"].to_numpy(
            dtype=np.int64
        )
        label_counts = subset["_label_action"].value_counts()
        pred_counts = subset["_predicted_action"].value_counts()
        return {
            "slice": label,
            "samples": samples,
            "accuracy": float(correct.mean()),
            "loss": float(pd.to_numeric(subset["_loss"], errors="coerce").mean()),
            "label_unique_actions": int(label_counts.shape[0]),
            "predicted_unique_actions": int(pred_counts.shape[0]),
            "label_dominant_action": int(label_counts.index[0]),
            "label_dominant_share": float(label_counts.iloc[0] / samples),
            "predicted_dominant_action": int(pred_counts.index[0]),
            "predicted_dominant_share": float(pred_counts.iloc[0] / samples),
        }

    rows = [_row("all", pd.Series(True, index=work.index))]
    for col in slice_cols:
        if col not in work.columns:
            continue
        rows.append(_row(str(col), work[col].fillna(False).astype(bool)))
    return pd.DataFrame(rows).sort_values(["slice"]).reset_index(drop=True)


def summarize_bc_slice_fit(
    slice_metrics: pd.DataFrame,
    *,
    required_slices: tuple[str, ...] = ("release_gate_state",),
    min_accuracy: float = 0.60,
    min_samples: int = 1,
) -> dict[str, object]:
    """Summarize whether sparse BC state slices are fit well enough for PPO handoff."""

    if slice_metrics.empty:
        return {
            "decision": "bc_slice_fit_no_rows",
            "slice_fit_gate": False,
            "failed_slices": list(required_slices),
            "next_mechanism": "train_bc_before_ppo_handoff",
        }
    metrics = slice_metrics.set_index("slice")
    failed: list[str] = []
    slice_details: dict[str, dict[str, float | int]] = {}
    for name in required_slices:
        if name not in metrics.index:
            failed.append(str(name))
            slice_details[str(name)] = {"samples": 0, "accuracy": float("nan")}
            continue
        row = metrics.loc[name]
        samples = int(row.get("samples", 0))
        accuracy = float(row.get("accuracy", float("nan")))
        slice_details[str(name)] = {"samples": samples, "accuracy": accuracy}
        if samples < int(min_samples) or not np.isfinite(accuracy) or accuracy < float(min_accuracy):
            failed.append(str(name))
    gate = len(failed) == 0
    return {
        "decision": "bc_slice_fit_ready_for_ppo_screen" if gate else "bc_slice_fit_failed_no_ppo",
        "slice_fit_gate": gate,
        "failed_slices": failed,
        "required_slices": list(required_slices),
        "min_accuracy": float(min_accuracy),
        "min_samples": int(min_samples),
        "slice_details": slice_details,
        "next_mechanism": "run_walkforward_ppo_candidate" if gate else "retune_bc_slice_weighting_or_state_head",
    }


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def score_bc_walkforward_drift(
    fold: str,
    *,
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    slice_gate: bool,
    target_relative_return_pp: float = 30.0,
    target_max_dd_pct: float = 15.0,
    relative_drift_floor_pp: float = -5.0,
    max_dd_drift_ceiling_pp: float = 2.0,
) -> dict[str, Any]:
    """Score a BC walk-forward candidate against target gates and baseline drift."""

    candidate_relative = _metric_value(candidate_metrics, "relative_return")
    candidate_max_dd = _metric_value(candidate_metrics, "max_dd")
    candidate_avg_pos = _metric_value(candidate_metrics, "avg_pos")
    baseline_relative = _metric_value(baseline_metrics, "relative_return")
    baseline_max_dd = _metric_value(baseline_metrics, "max_dd")
    baseline_avg_pos = _metric_value(baseline_metrics, "avg_pos")

    candidate_gate = bool(
        np.isfinite(candidate_relative)
        and np.isfinite(candidate_max_dd)
        and candidate_relative >= float(target_relative_return_pp)
        and candidate_max_dd <= float(target_max_dd_pct)
    )
    baseline_gate = bool(
        np.isfinite(baseline_relative)
        and np.isfinite(baseline_max_dd)
        and baseline_relative >= float(target_relative_return_pp)
        and baseline_max_dd <= float(target_max_dd_pct)
    )
    relative_delta = float(candidate_relative - baseline_relative) if (
        np.isfinite(candidate_relative) and np.isfinite(baseline_relative)
    ) else float("nan")
    max_dd_delta = float(candidate_max_dd - baseline_max_dd) if (
        np.isfinite(candidate_max_dd) and np.isfinite(baseline_max_dd)
    ) else float("nan")
    drift_gate = bool(
        np.isfinite(relative_delta)
        and np.isfinite(max_dd_delta)
        and relative_delta >= float(relative_drift_floor_pp)
        and max_dd_delta <= float(max_dd_drift_ceiling_pp)
    )
    failed_checks: list[str] = []
    if not np.isfinite(candidate_relative) or candidate_relative < float(target_relative_return_pp):
        failed_checks.append("candidate_relative_return")
    if not np.isfinite(candidate_max_dd) or candidate_max_dd > float(target_max_dd_pct):
        failed_checks.append("candidate_max_dd")
    if not bool(slice_gate):
        failed_checks.append("slice_gate")
    if not np.isfinite(relative_delta) or relative_delta < float(relative_drift_floor_pp):
        failed_checks.append("relative_return_drift")
    if not np.isfinite(max_dd_delta) or max_dd_delta > float(max_dd_drift_ceiling_pp):
        failed_checks.append("max_dd_drift")

    return {
        "fold": str(fold),
        "candidate_relative_return": candidate_relative,
        "candidate_max_dd": candidate_max_dd,
        "candidate_avg_pos": candidate_avg_pos,
        "baseline_relative_return": baseline_relative,
        "baseline_max_dd": baseline_max_dd,
        "baseline_avg_pos": baseline_avg_pos,
        "relative_delta_vs_baseline": relative_delta,
        "max_dd_delta_vs_baseline": max_dd_delta,
        "candidate_gate": candidate_gate,
        "baseline_gate": baseline_gate,
        "slice_gate": bool(slice_gate),
        "drift_gate": drift_gate,
        "fold_gate": bool(candidate_gate and bool(slice_gate) and drift_gate),
        "target_relative_return_pp": float(target_relative_return_pp),
        "target_max_dd_pct": float(target_max_dd_pct),
        "relative_drift_floor_pp": float(relative_drift_floor_pp),
        "max_dd_drift_ceiling_pp": float(max_dd_drift_ceiling_pp),
        "failed_checks": failed_checks,
    }


def _bool_sum(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def summarize_bc_walkforward_drift_screen(
    fold_scores: pd.DataFrame,
    *,
    required_folds: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Summarize BC walk-forward target and drift gates without permitting promotion."""

    if fold_scores.empty:
        return {
            "decision": "bc_walkforward_drift_no_rows",
            "promotion_allowed": False,
            "fold_gate_pass_count": 0,
            "candidate_gate_pass_count": 0,
            "drift_gate_pass_count": 0,
            "failed_folds": list(required_folds),
            "next_mechanism": "train_bc_before_walkforward_drift_screen",
        }

    frame = fold_scores.copy()
    frame["fold"] = frame["fold"].astype(str)
    required = list(required_folds) if required_folds else sorted(frame["fold"].unique().tolist())
    required_set = set(required)
    scored = frame[frame["fold"].isin(required_set)].copy()
    missing = [fold for fold in required if fold not in set(scored["fold"])]
    failed = scored.loc[~scored.get("fold_gate", False).fillna(False).astype(bool), "fold"].tolist()
    failed_folds = [*failed, *missing]
    fold_gate_pass_count = _bool_sum(scored, "fold_gate")
    candidate_gate_pass_count = _bool_sum(scored, "candidate_gate")
    drift_gate_pass_count = _bool_sum(scored, "drift_gate")
    all_required_pass = bool(required and not missing and fold_gate_pass_count == len(required))

    if all_required_pass:
        decision = "bc_walkforward_drift_gate_passed_no_promotion"
        next_mechanism = "run_full_fixed_split_and_walkforward_replay_before_promotion"
    elif fold_gate_pass_count > 0:
        decision = "bc_walkforward_drift_mixed_no_promotion"
        next_mechanism = "inspect_failed_fold_slices_or_checkpoint_selection_before_ppo"
    elif candidate_gate_pass_count > 0:
        decision = "bc_walkforward_drift_failed_no_promotion"
        next_mechanism = "block_ppo_until_drawdown_drift_or_slice_fit_is_fixed"
    else:
        decision = "bc_walkforward_candidate_gate_failed_no_promotion"
        next_mechanism = "revise_training_targets_before_ppo"

    return {
        "decision": decision,
        "promotion_allowed": False,
        "required_folds": required,
        "scored_folds": scored["fold"].tolist(),
        "failed_folds": failed_folds,
        "fold_gate_pass_count": int(fold_gate_pass_count),
        "candidate_gate_pass_count": int(candidate_gate_pass_count),
        "baseline_gate_pass_count": _bool_sum(scored, "baseline_gate"),
        "drift_gate_pass_count": int(drift_gate_pass_count),
        "slice_gate_pass_count": _bool_sum(scored, "slice_gate"),
        "next_mechanism": next_mechanism,
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
    "evaluate_bc_actor_slice_fit",
    "fit_bc_actor",
    "fit_state_objective_bc_actor_from_frame",
    "score_bc_walkforward_drift",
    "state_objective_bc_sample_weights",
    "summarize_bc_walkforward_drift_screen",
    "summarize_bc_slice_fit",
]

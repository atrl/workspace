from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RepresentationDiagnosticsResult:
    feature_set_scores: pd.DataFrame
    fold_metrics: pd.DataFrame
    feature_effects: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True)
class StateHeadTargetRecipeConfig:
    min_recall: float = 0.60
    min_precision: float = 0.40
    max_false_positive_rate: float = 0.10
    min_effective_lift: float = 0.01
    max_effective_false_positive_rate: float = 0.05
    max_false_positive_lift_mean: float = 0.01
    max_false_positive_lift_max: float = 0.20
    max_target_mae: float = 0.08
    max_rebound_mae: float = 0.08
    min_covered_folds: int = 1


@dataclass(frozen=True)
class StateHeadTargetRecipeResult:
    recipe_scores: pd.DataFrame
    row_scores: pd.DataFrame
    summary: dict[str, object]


def rank_auc_score(labels: np.ndarray | Sequence[bool], scores: np.ndarray | Sequence[float]) -> float:
    """Compute binary rank AUC without sklearn dependency."""

    label_array = np.asarray(labels, dtype=bool)
    score_array = np.asarray(scores, dtype=float)
    if label_array.shape != score_array.shape:
        raise ValueError(f"labels shape {label_array.shape} does not match scores {score_array.shape}")
    finite = np.isfinite(score_array)
    label_array = label_array[finite]
    score_array = score_array[finite]
    n_pos = int(label_array.sum())
    n_neg = int((~label_array).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(score_array).rank(method="average").to_numpy(dtype=float)
    pos_rank_sum = float(ranks[label_array].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc)


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _standardized_matrix(train: pd.DataFrame, eval_frame: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    train_values = train.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    eval_values = eval_frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    train_z = ((train_values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eval_z = ((eval_values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train_z.to_numpy(dtype=float), eval_z.to_numpy(dtype=float)


def _centroid_scores(train_z: np.ndarray, train_labels: np.ndarray, eval_z: np.ndarray) -> np.ndarray:
    pos = train_z[train_labels]
    neg = train_z[~train_labels]
    if len(pos) == 0 or len(neg) == 0:
        return np.full(eval_z.shape[0], np.nan, dtype=float)
    pos_center = pos.mean(axis=0)
    neg_center = neg.mean(axis=0)
    dist_pos = np.linalg.norm(eval_z - pos_center, axis=1)
    dist_neg = np.linalg.norm(eval_z - neg_center, axis=1)
    return dist_neg - dist_pos


def _feature_effect_rows(
    frame: pd.DataFrame,
    *,
    label_col: str,
    feature_set: str,
    features: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = frame[label_col].fillna(False).astype(bool)
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        pos = values[labels]
        neg = values[~labels]
        pooled = values.std(ddof=0)
        effect = (pos.mean() - neg.mean()) / pooled if pooled and np.isfinite(pooled) else np.nan
        rows.append(
            {
                "feature_set": feature_set,
                "feature": feature,
                "positive_mean": float(pos.mean()) if len(pos) else np.nan,
                "negative_mean": float(neg.mean()) if len(neg) else np.nan,
                "effect_size": float(effect) if np.isfinite(effect) else np.nan,
                "positive_count": int(len(pos)),
                "negative_count": int(len(neg)),
            }
        )
    return rows


def score_binary_representation_feature_sets(
    frame: pd.DataFrame,
    *,
    label_col: str,
    fold_col: str,
    feature_sets: Mapping[str, Sequence[str]],
    min_positive_train: int = 1,
    min_positive_eval: int = 1,
    min_holdout_auc: float = 0.60,
    min_holdout_accuracy: float = 0.55,
) -> RepresentationDiagnosticsResult:
    """Score whether feature groups separate a binary state under fold-heldout evaluation."""

    all_features = {feature for features in feature_sets.values() for feature in features}
    _require_columns(frame, {label_col, fold_col, *all_features}, "representation frame")
    work = frame.copy()
    work[label_col] = work[label_col].fillna(False).astype(bool)
    work[fold_col] = work[fold_col].astype(str)
    folds = sorted(work[fold_col].dropna().unique().tolist())

    score_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []

    for feature_set, features in feature_sets.items():
        if not features:
            raise ValueError(f"feature set {feature_set!r} has no features")
        effect_rows.extend(
            _feature_effect_rows(
                work,
                label_col=label_col,
                feature_set=str(feature_set),
                features=features,
            )
        )
        for fold in folds:
            train = work[work[fold_col] != fold]
            eval_frame = work[work[fold_col] == fold]
            train_labels = train[label_col].to_numpy(dtype=bool)
            eval_labels = eval_frame[label_col].to_numpy(dtype=bool)
            train_pos = int(train_labels.sum())
            eval_pos = int(eval_labels.sum())
            eval_neg = int((~eval_labels).sum())
            coverage_gate = bool(train_pos >= int(min_positive_train) and eval_pos >= int(min_positive_eval) and eval_neg > 0)
            if coverage_gate:
                train_z, eval_z = _standardized_matrix(train, eval_frame, features)
                scores = _centroid_scores(train_z, train_labels, eval_z)
                auc = rank_auc_score(eval_labels, scores)
                predictions = scores > 0.0
                accuracy = float((predictions == eval_labels).mean()) if len(eval_labels) else np.nan
            else:
                auc = float("nan")
                accuracy = float("nan")
            fold_rows.append(
                {
                    "feature_set": str(feature_set),
                    "eval_fold": str(fold),
                    "features": ",".join(features),
                    "train_samples": int(len(train)),
                    "eval_samples": int(len(eval_frame)),
                    "train_positive": train_pos,
                    "eval_positive": eval_pos,
                    "eval_negative": eval_neg,
                    "coverage_gate": coverage_gate,
                    "holdout_auc": float(auc) if np.isfinite(auc) else np.nan,
                    "holdout_accuracy": float(accuracy) if np.isfinite(accuracy) else np.nan,
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    feature_effects = pd.DataFrame(effect_rows)
    for feature_set, group in fold_metrics.groupby("feature_set", sort=False):
        covered = group[group["coverage_gate"]]
        coverage_gate = bool(len(covered) == len(folds) and len(covered) > 0)
        holdout_auc_min = float(covered["holdout_auc"].min()) if not covered.empty else np.nan
        holdout_accuracy_min = float(covered["holdout_accuracy"].min()) if not covered.empty else np.nan
        representation_gate = bool(
            coverage_gate
            and holdout_auc_min >= float(min_holdout_auc)
            and holdout_accuracy_min >= float(min_holdout_accuracy)
        )
        score_rows.append(
            {
                "feature_set": str(feature_set),
                "features": group["features"].iloc[0],
                "feature_count": int(len(str(group["features"].iloc[0]).split(","))),
                "folds": int(len(group)),
                "covered_folds": int(group["coverage_gate"].sum()),
                "coverage_gate": coverage_gate,
                "holdout_auc_min": holdout_auc_min,
                "holdout_auc_mean": float(covered["holdout_auc"].mean()) if not covered.empty else np.nan,
                "holdout_accuracy_min": holdout_accuracy_min,
                "holdout_accuracy_mean": float(covered["holdout_accuracy"].mean()) if not covered.empty else np.nan,
                "representation_gate": representation_gate,
            }
        )

    feature_set_scores = pd.DataFrame(score_rows).sort_values(
        ["representation_gate", "holdout_auc_min", "holdout_accuracy_min", "feature_count", "feature_set"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    best = feature_set_scores.iloc[0].to_dict() if not feature_set_scores.empty else {}
    summary = {
        "best_feature_set": best.get("feature_set", ""),
        "best_holdout_auc_min": float(best.get("holdout_auc_min", np.nan)),
        "best_holdout_accuracy_min": float(best.get("holdout_accuracy_min", np.nan)),
        "representation_gate": bool(best.get("representation_gate", False)),
        "feature_sets": int(len(feature_set_scores)),
        "folds": int(len(folds)),
    }
    return RepresentationDiagnosticsResult(
        feature_set_scores=feature_set_scores,
        fold_metrics=fold_metrics.sort_values(["feature_set", "eval_fold"]).reset_index(drop=True),
        feature_effects=feature_effects.sort_values(["feature_set", "feature"]).reset_index(drop=True),
        summary=summary,
    )


def _fold_heldout_state_scores(
    frame: pd.DataFrame,
    *,
    label_col: str,
    fold_col: str,
    feature_set: str,
    features: Sequence[str],
    min_positive_train: int,
    min_positive_eval: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    folds = sorted(frame[fold_col].dropna().astype(str).unique().tolist())
    for fold in folds:
        train = frame[frame[fold_col].astype(str) != fold]
        eval_frame = frame[frame[fold_col].astype(str) == fold]
        train_labels = train[label_col].fillna(False).astype(bool).to_numpy()
        eval_labels = eval_frame[label_col].fillna(False).astype(bool).to_numpy()
        train_pos = int(train_labels.sum())
        eval_pos = int(eval_labels.sum())
        eval_neg = int((~eval_labels).sum())
        coverage_gate = bool(
            train_pos >= int(min_positive_train)
            and eval_pos >= int(min_positive_eval)
            and eval_neg > 0
        )
        if coverage_gate:
            train_z, eval_z = _standardized_matrix(train, eval_frame, features)
            scores = _centroid_scores(train_z, train_labels, eval_z)
        else:
            scores = np.full(len(eval_frame), np.nan, dtype=float)
        for row_idx, score in zip(eval_frame.index, scores):
            rows.append(
                {
                    "row_index": int(row_idx),
                    "feature_set": str(feature_set),
                    "eval_fold": str(fold),
                    "state_score": float(score) if np.isfinite(score) else np.nan,
                    "coverage_gate": coverage_gate,
                    "train_positive": train_pos,
                    "eval_positive": eval_pos,
                    "eval_negative": eval_neg,
                }
            )
    return rows


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def score_state_head_target_recipes(
    frame: pd.DataFrame,
    *,
    label_col: str,
    fold_col: str,
    true_target_col: str,
    default_target_col: str,
    state_target_col: str,
    feature_sets: Mapping[str, Sequence[str]],
    thresholds: Sequence[float] = (0.0,),
    min_positive_train: int = 1,
    min_positive_eval: int = 1,
    config: StateHeadTargetRecipeConfig = StateHeadTargetRecipeConfig(),
) -> StateHeadTargetRecipeResult:
    """Score fold-heldout binary state heads as target-position routing recipes."""

    all_features = {feature for features in feature_sets.values() for feature in features}
    _require_columns(
        frame,
        {label_col, fold_col, true_target_col, default_target_col, state_target_col, *all_features},
        "state-head recipe frame",
    )
    if not thresholds:
        raise ValueError("state-head target recipe requires at least one threshold")

    work = frame.copy()
    work[label_col] = work[label_col].fillna(False).astype(bool)
    work[fold_col] = work[fold_col].astype(str)
    for col in (true_target_col, default_target_col, state_target_col):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    score_rows: list[dict[str, object]] = []
    for feature_set, features in feature_sets.items():
        if not features:
            raise ValueError(f"feature set {feature_set!r} has no features")
        score_rows.extend(
            _fold_heldout_state_scores(
                work,
                label_col=label_col,
                fold_col=fold_col,
                feature_set=str(feature_set),
                features=features,
                min_positive_train=int(min_positive_train),
                min_positive_eval=int(min_positive_eval),
            )
        )
    score_frame = pd.DataFrame(score_rows)
    if score_frame.empty:
        empty_scores = pd.DataFrame()
        return StateHeadTargetRecipeResult(recipe_scores=empty_scores, row_scores=score_frame, summary={})

    score_frame = score_frame.join(
        work[[label_col, fold_col, true_target_col, default_target_col, state_target_col]],
        on="row_index",
    )
    recipe_rows: list[dict[str, object]] = []
    routed_rows: list[pd.DataFrame] = []
    for feature_set, group in score_frame.groupby("feature_set", sort=False):
        for threshold in thresholds:
            routed = group.copy()
            routed["threshold"] = float(threshold)
            routed["predicted_state"] = (
                routed["coverage_gate"].fillna(False).astype(bool)
                & (pd.to_numeric(routed["state_score"], errors="coerce") >= float(threshold))
            )
            routed["proposed_target"] = np.where(
                routed["predicted_state"],
                routed[state_target_col],
                routed[default_target_col],
            )
            routed["target_lift"] = routed["proposed_target"] - routed[default_target_col]
            routed["target_error"] = (routed["proposed_target"] - routed[true_target_col]).abs()
            labels = routed[label_col].fillna(False).astype(bool)
            predicted = routed["predicted_state"].fillna(False).astype(bool)
            tp = int((labels & predicted).sum())
            fp = int((~labels & predicted).sum())
            fn = int((labels & ~predicted).sum())
            tn = int((~labels & ~predicted).sum())
            positives = int(labels.sum())
            negatives = int((~labels).sum())
            covered_folds = int(routed.groupby(fold_col)["coverage_gate"].max().sum())
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            false_positive_rate = _safe_div(fp, fp + tn)
            true_positive_rate = recall
            all_mae = float(routed["target_error"].mean())
            rebound_mae = float(routed.loc[labels, "target_error"].mean()) if positives else float("nan")
            non_rebound_mae = float(routed.loc[~labels, "target_error"].mean()) if negatives else float("nan")
            fp_lift = routed.loc[(~labels) & predicted, "target_lift"].abs()
            false_positive_lift_mean = float(fp_lift.mean()) if len(fp_lift) else 0.0
            false_positive_lift_max = float(fp_lift.max()) if len(fp_lift) else 0.0
            effective_fp = int((fp_lift > float(config.min_effective_lift)).sum()) if len(fp_lift) else 0
            effective_false_positive_rate = _safe_div(effective_fp, negatives)
            recipe_gate = bool(
                covered_folds >= int(config.min_covered_folds)
                and np.isfinite(precision)
                and np.isfinite(recall)
                and np.isfinite(false_positive_rate)
                and precision >= float(config.min_precision)
                and recall >= float(config.min_recall)
                and false_positive_rate <= float(config.max_false_positive_rate)
                and np.isfinite(effective_false_positive_rate)
                and effective_false_positive_rate <= float(config.max_effective_false_positive_rate)
                and false_positive_lift_mean <= float(config.max_false_positive_lift_mean)
                and false_positive_lift_max <= float(config.max_false_positive_lift_max)
                and all_mae <= float(config.max_target_mae)
                and rebound_mae <= float(config.max_rebound_mae)
            )
            recipe_rows.append(
                {
                    "feature_set": str(feature_set),
                    "threshold": float(threshold),
                    "samples": int(len(routed)),
                    "positives": positives,
                    "negatives": negatives,
                    "covered_folds": covered_folds,
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_negative": tn,
                    "state_precision": precision,
                    "state_recall": recall,
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                    "effective_false_positive": effective_fp,
                    "effective_false_positive_rate": effective_false_positive_rate,
                    "false_positive_lift_mean": false_positive_lift_mean,
                    "false_positive_lift_max": false_positive_lift_max,
                    "target_mae": all_mae,
                    "rebound_target_mae": rebound_mae,
                    "non_rebound_target_mae": non_rebound_mae,
                    "recipe_gate": recipe_gate,
                    "recipe_loss": (
                        max(float(config.min_recall) - (recall if np.isfinite(recall) else 0.0), 0.0) * 4.0
                        + max(float(config.min_precision) - (precision if np.isfinite(precision) else 0.0), 0.0) * 2.0
                        + max((false_positive_rate if np.isfinite(false_positive_rate) else 1.0) - float(config.max_false_positive_rate), 0.0) * 2.0
                        + max((effective_false_positive_rate if np.isfinite(effective_false_positive_rate) else 1.0) - float(config.max_effective_false_positive_rate), 0.0) * 2.0
                        + max(false_positive_lift_mean - float(config.max_false_positive_lift_mean), 0.0)
                        + max(false_positive_lift_max - float(config.max_false_positive_lift_max), 0.0)
                        + max(all_mae - float(config.max_target_mae), 0.0)
                        + max((rebound_mae if np.isfinite(rebound_mae) else 1.0) - float(config.max_rebound_mae), 0.0)
                    ),
                }
            )
            routed_rows.append(routed)

    recipe_scores = pd.DataFrame(recipe_rows).sort_values(
        ["recipe_gate", "recipe_loss", "state_recall", "false_positive_rate", "feature_set"],
        ascending=[False, True, False, True, True],
    ).reset_index(drop=True)
    row_scores = pd.concat(routed_rows, ignore_index=True) if routed_rows else pd.DataFrame()
    best = recipe_scores.iloc[0].to_dict() if not recipe_scores.empty else {}
    summary = {
        "best_feature_set": best.get("feature_set", ""),
        "best_threshold": float(best.get("threshold", np.nan)),
        "best_state_recall": float(best.get("state_recall", np.nan)),
        "best_state_precision": float(best.get("state_precision", np.nan)),
        "best_false_positive_rate": float(best.get("false_positive_rate", np.nan)),
        "best_effective_false_positive_rate": float(best.get("effective_false_positive_rate", np.nan)),
        "best_false_positive_lift_mean": float(best.get("false_positive_lift_mean", np.nan)),
        "best_false_positive_lift_max": float(best.get("false_positive_lift_max", np.nan)),
        "best_target_mae": float(best.get("target_mae", np.nan)),
        "best_rebound_target_mae": float(best.get("rebound_target_mae", np.nan)),
        "recipe_gate": bool(best.get("recipe_gate", False)),
    }
    return StateHeadTargetRecipeResult(
        recipe_scores=recipe_scores,
        row_scores=row_scores.sort_values(["feature_set", "threshold", fold_col, "row_index"]).reset_index(drop=True),
        summary=summary,
    )


__all__ = [
    "RepresentationDiagnosticsResult",
    "StateHeadTargetRecipeConfig",
    "StateHeadTargetRecipeResult",
    "rank_auc_score",
    "score_binary_representation_feature_sets",
    "score_state_head_target_recipes",
]

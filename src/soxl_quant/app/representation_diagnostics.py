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


__all__ = [
    "RepresentationDiagnosticsResult",
    "rank_auc_score",
    "score_binary_representation_feature_sets",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinaryLabelDiscoveryConfig:
    min_positive_rows: int = 5
    min_positive_rate: float = 0.0
    max_positive_rate: float = 0.50
    min_fold_coverage: int = 1
    min_fold_positive_rows: int = 1
    min_forward_return_lift: float = 0.0
    min_target_lift: float = 0.0
    max_label_conflict_rate: float = 0.0


@dataclass(frozen=True)
class BinaryLabelDiscoveryResult:
    candidate_scores: pd.DataFrame
    fold_metrics: pd.DataFrame
    row_scores: pd.DataFrame
    summary: dict[str, object]


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _coerce_mask(mask: object, index: pd.Index, candidate: str) -> pd.Series:
    if isinstance(mask, pd.Series):
        if mask.index.equals(index):
            out = mask.copy()
        elif len(mask) == len(index):
            out = pd.Series(mask.to_numpy(), index=index)
        else:
            raise ValueError(f"candidate {candidate!r} mask length does not match frame")
    else:
        out = pd.Series(mask, index=index)
    if len(out) != len(index):
        raise ValueError(f"candidate {candidate!r} mask length does not match frame")
    return out.fillna(False).astype(bool)


def _mean_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _forward_lift(work: pd.DataFrame, mask: pd.Series, column: str) -> dict[str, float]:
    values = pd.to_numeric(work[column], errors="coerce")
    positive = values[mask].dropna()
    negative = values[~mask].dropna()
    positive_mean = _mean_or_nan(positive)
    negative_mean = _mean_or_nan(negative)
    lift = positive_mean - negative_mean if np.isfinite(positive_mean) and np.isfinite(negative_mean) else float("nan")
    return {
        f"{column}_positive_mean": float(positive_mean),
        f"{column}_negative_mean": float(negative_mean),
        f"{column}_lift": float(lift),
    }


def score_binary_label_candidates(
    frame: pd.DataFrame,
    candidate_masks: Mapping[str, object],
    *,
    group_col: str,
    fold_col: str | None = None,
    forward_return_cols: Sequence[str] = (),
    target_lift_col: str | None = None,
    config: BinaryLabelDiscoveryConfig = BinaryLabelDiscoveryConfig(),
) -> BinaryLabelDiscoveryResult:
    """Score candidate binary labels for duplicate-group stability and forward lift.

    The function is intentionally generic. Candidate construction, threshold
    grids, and economic interpretation should stay in notebooks or scripts.
    """

    required = {group_col, *forward_return_cols}
    if fold_col:
        required.add(fold_col)
    if target_lift_col:
        required.add(target_lift_col)
    _require_columns(frame, required, "binary label discovery frame")
    if not candidate_masks:
        raise ValueError("at least one candidate mask is required")

    work = frame.copy()
    if fold_col:
        work[fold_col] = work[fold_col].astype(str)
    total_rows = int(len(work))
    total_groups = int(work[group_col].nunique())
    folds = sorted(work[fold_col].dropna().unique().tolist()) if fold_col else []

    score_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    row_frames: list[pd.DataFrame] = []

    for candidate, raw_mask in candidate_masks.items():
        mask = _coerce_mask(raw_mask, work.index, str(candidate))
        positive_rows = int(mask.sum())
        positive_rate = _safe_div(positive_rows, total_rows)
        group_state = pd.DataFrame({group_col: work[group_col], "label": mask})
        grouped = group_state.groupby(group_col, sort=True)["label"].agg(["sum", "count"])
        grouped["positive_rows"] = grouped["sum"].astype(int)
        grouped["negative_rows"] = (grouped["count"] - grouped["sum"]).astype(int)
        grouped["label_conflict"] = (grouped["positive_rows"] > 0) & (grouped["negative_rows"] > 0)
        label_conflict_groups = int(grouped["label_conflict"].sum())
        label_conflict_rate = _safe_div(label_conflict_groups, total_groups)
        positive_groups = int((grouped["positive_rows"] > 0).sum())

        metric_row: dict[str, object] = {
            "candidate": str(candidate),
            "rows": total_rows,
            "groups": total_groups,
            "positive_rows": positive_rows,
            "positive_groups": positive_groups,
            "positive_rate": float(positive_rate) if np.isfinite(positive_rate) else float("nan"),
            "label_conflict_groups": label_conflict_groups,
            "label_conflict_rate": float(label_conflict_rate) if np.isfinite(label_conflict_rate) else float("nan"),
        }

        primary_forward_lift = float("nan")
        for idx, column in enumerate(forward_return_cols):
            lift_metrics = _forward_lift(work, mask, column)
            metric_row.update(lift_metrics)
            if idx == 0:
                primary_forward_lift = float(lift_metrics[f"{column}_lift"])
        metric_row["primary_forward_return_lift"] = primary_forward_lift

        if target_lift_col:
            target_values = pd.to_numeric(work[target_lift_col], errors="coerce")
            target_positive = target_values[mask].dropna()
            target_lift_mean = _mean_or_nan(target_positive)
            target_lift_median = float(target_positive.median()) if len(target_positive) else float("nan")
        else:
            target_lift_mean = float("nan")
            target_lift_median = float("nan")
        metric_row["target_lift_mean"] = target_lift_mean
        metric_row["target_lift_median"] = target_lift_median

        if fold_col:
            covered_folds = 0
            min_fold_positive_rows = np.inf
            for fold, sub in work.groupby(fold_col, sort=True):
                sub_mask = mask.loc[sub.index]
                fold_positive_rows = int(sub_mask.sum())
                covered_folds += int(fold_positive_rows >= int(config.min_fold_positive_rows))
                min_fold_positive_rows = min(min_fold_positive_rows, fold_positive_rows)
                fold_row: dict[str, object] = {
                    "candidate": str(candidate),
                    "fold": str(fold),
                    "rows": int(len(sub)),
                    "positive_rows": fold_positive_rows,
                    "positive_rate": _safe_div(fold_positive_rows, len(sub)),
                }
                for column in forward_return_cols:
                    fold_row.update(_forward_lift(sub, sub_mask, column))
                if target_lift_col:
                    fold_target = pd.to_numeric(sub[target_lift_col], errors="coerce")
                    fold_row["target_lift_mean"] = _mean_or_nan(fold_target[sub_mask].dropna())
                fold_rows.append(fold_row)
            metric_row["folds"] = int(len(folds))
            metric_row["covered_folds"] = int(covered_folds)
            metric_row["min_fold_positive_rows"] = (
                int(min_fold_positive_rows) if np.isfinite(min_fold_positive_rows) else 0
            )
        else:
            metric_row["folds"] = 0
            metric_row["covered_folds"] = 0
            metric_row["min_fold_positive_rows"] = 0

        forward_gate = (
            not forward_return_cols
            or (np.isfinite(primary_forward_lift) and primary_forward_lift >= float(config.min_forward_return_lift))
        )
        target_gate = (
            not target_lift_col
            or (np.isfinite(target_lift_mean) and target_lift_mean >= float(config.min_target_lift))
        )
        fold_gate = (
            not fold_col
            or (
                int(metric_row["covered_folds"]) >= int(config.min_fold_coverage)
                and int(metric_row["min_fold_positive_rows"]) >= int(config.min_fold_positive_rows)
            )
        )
        label_gate = bool(
            positive_rows >= int(config.min_positive_rows)
            and np.isfinite(positive_rate)
            and positive_rate >= float(config.min_positive_rate)
            and positive_rate <= float(config.max_positive_rate)
            and np.isfinite(label_conflict_rate)
            and label_conflict_rate <= float(config.max_label_conflict_rate)
            and forward_gate
            and target_gate
            and fold_gate
        )
        metric_row["forward_gate"] = bool(forward_gate)
        metric_row["target_gate"] = bool(target_gate)
        metric_row["fold_gate"] = bool(fold_gate)
        metric_row["label_gate"] = label_gate
        metric_row["label_loss"] = (
            max(int(config.min_positive_rows) - positive_rows, 0) / max(int(config.min_positive_rows), 1)
            + max(float(config.min_positive_rate) - (positive_rate if np.isfinite(positive_rate) else 0.0), 0.0)
            + max((positive_rate if np.isfinite(positive_rate) else 1.0) - float(config.max_positive_rate), 0.0)
            + max((label_conflict_rate if np.isfinite(label_conflict_rate) else 1.0) - float(config.max_label_conflict_rate), 0.0) * 2.0
            + max(float(config.min_forward_return_lift) - (primary_forward_lift if np.isfinite(primary_forward_lift) else 0.0), 0.0) * 4.0
            + max(float(config.min_target_lift) - (target_lift_mean if np.isfinite(target_lift_mean) else 0.0), 0.0)
        )
        score_rows.append(metric_row)

        rows = work.loc[:, [group_col, *([fold_col] if fold_col else [])]].copy()
        rows["candidate"] = str(candidate)
        rows["candidate_label"] = mask.to_numpy(dtype=bool)
        row_frames.append(rows)

    candidate_scores = pd.DataFrame(score_rows).sort_values(
        [
            "label_gate",
            "label_loss",
            "primary_forward_return_lift",
            "target_lift_mean",
            "label_conflict_rate",
            "candidate",
        ],
        ascending=[False, True, False, False, True, True],
    ).reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_rows)
    if not fold_metrics.empty:
        fold_metrics = fold_metrics.sort_values(["candidate", "fold"]).reset_index(drop=True)
    row_scores = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    best = candidate_scores.iloc[0].to_dict() if not candidate_scores.empty else {}
    summary = {
        "selected_candidate": str(best.get("candidate", "")),
        "label_gate": bool(best.get("label_gate", False)),
        "primary_forward_return_lift": float(best.get("primary_forward_return_lift", np.nan)),
        "target_lift_mean": float(best.get("target_lift_mean", np.nan)),
        "label_conflict_rate": float(best.get("label_conflict_rate", np.nan)),
        "candidates": int(len(candidate_scores)),
    }
    return BinaryLabelDiscoveryResult(
        candidate_scores=candidate_scores,
        fold_metrics=fold_metrics,
        row_scores=row_scores,
        summary=summary,
    )


__all__ = [
    "BinaryLabelDiscoveryConfig",
    "BinaryLabelDiscoveryResult",
    "score_binary_label_candidates",
]

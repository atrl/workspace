from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.core.contracts import stable_hash


OBJECTIVE_LABEL_MANIFEST_SCHEMA_VERSION = "objective.label_manifest.v1"
OBJECTIVE_LABEL_FRAME_SCHEMA_VERSION = "objective.label_frame.v1"


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    target_kind: str
    direction: str
    description: str


@dataclass(frozen=True)
class PromotionThresholds:
    min_return_pct: float | None = None
    min_relative_return_pct: float | None = None
    min_sharpe: float | None = None
    max_drawdown_pct: float | None = None
    max_turnover: float | None = None
    schema_version: str = "objective.promotion_thresholds.v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


OBJECTIVE_SPECS: tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec("classification", "binary_forward_return", "maximize", "预测未来收益方向。"),
    ObjectiveSpec("mse", "forward_return", "minimize", "最小化未来收益预测误差。"),
    ObjectiveSpec("ic", "forward_return_rank", "maximize", "最大化预测与未来收益的截面或时间序列相关。"),
    ObjectiveSpec("ranking", "relative_forward_return", "maximize", "优化候选信号或资产相对排序。"),
    ObjectiveSpec("utility", "portfolio_path", "maximize", "综合收益、回撤、换手和稳定性的晋升评分。"),
)
OBJECTIVE_NAMES: tuple[str, ...] = tuple(spec.name for spec in OBJECTIVE_SPECS)


def get_objective_spec(name: str) -> ObjectiveSpec:
    for spec in OBJECTIVE_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown objective spec: {name}")


def utility_score(*, return_pct: float, sharpe: float, max_drawdown_pct: float, turnover: float) -> float:
    return float(return_pct) + 10.0 * float(sharpe) - 1.25 * float(max_drawdown_pct) - 2.0 * float(turnover)


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing columns: {missing}")


def _frame_schema_hash(frame: pd.DataFrame) -> str:
    return stable_hash(
        {
            "schema_version": OBJECTIVE_LABEL_FRAME_SCHEMA_VERSION,
            "columns": [(str(column), str(dtype)) for column, dtype in zip(frame.columns, frame.dtypes, strict=True)],
        }
    )


def build_forward_return_labels(
    price_frame: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1,),
    date_col: str = "date",
    price_col: str = "close",
    group_col: str = "instrument",
    positive_threshold: float = 0.0,
) -> pd.DataFrame:
    """Build reusable classification, MSE and ranking targets from forward returns."""

    _require_columns(price_frame, {date_col, price_col}, context="price frame")
    if group_col and group_col not in price_frame.columns:
        raise ValueError(f"price frame missing columns: ['{group_col}']")
    if not horizons:
        raise ValueError("horizons must contain at least one positive horizon")
    if any(int(horizon) <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")

    result = price_frame.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    sort_columns = [group_col, date_col] if group_col else [date_col]
    result = result.sort_values(sort_columns).reset_index(drop=True)

    for horizon in tuple(int(value) for value in horizons):
        forward_col = f"forward_return_{horizon}d"
        class_col = f"classification_label_{horizon}d"
        rank_col = f"ranking_label_{horizon}d"
        available_col = f"label_available_{horizon}d"
        if group_col:
            future = result.groupby(group_col, sort=False)[price_col].shift(-horizon)
        else:
            future = result[price_col].shift(-horizon)
        result[forward_col] = future.astype(float) / result[price_col].astype(float) - 1.0
        result[available_col] = result[forward_col].notna()
        result[class_col] = (result[forward_col] > float(positive_threshold)).astype("Int64")
        result.loc[~result[available_col], class_col] = pd.NA
        result[rank_col] = result.groupby(date_col, sort=False)[forward_col].rank(
            method="first",
            ascending=False,
        )
        result.loc[~result[available_col], rank_col] = pd.NA

    return result


def _label_columns_for_horizon(horizon: int) -> dict[str, str]:
    suffix = f"{int(horizon)}d"
    return {
        "forward_return": f"forward_return_{suffix}",
        "classification": f"classification_label_{suffix}",
        "ranking": f"ranking_label_{suffix}",
        "available": f"label_available_{suffix}",
    }


def _label_manifest_metadata(labels: pd.DataFrame, *, horizons: Sequence[int]) -> dict[str, object]:
    if not horizons:
        raise ValueError("horizons must contain at least one positive horizon")
    horizon_values = [int(value) for value in horizons]
    if any(value <= 0 for value in horizon_values):
        raise ValueError("horizons must be positive")

    required = {"date"}
    label_columns = {f"{horizon}d": _label_columns_for_horizon(horizon) for horizon in horizon_values}
    for columns in label_columns.values():
        required.update(columns.values())
    _require_columns(labels, required, context="objective labels")

    dates = pd.to_datetime(labels["date"])
    available_counts = {
        suffix: int(pd.Series(labels[columns["available"]]).fillna(False).astype(bool).sum())
        for suffix, columns in label_columns.items()
    }
    return {
        "row_count": int(len(labels)),
        "column_count": int(len(labels.columns)),
        "date_start": dates.min().date().isoformat() if not dates.empty else "",
        "date_end": dates.max().date().isoformat() if not dates.empty else "",
        "instrument_count": int(labels["instrument"].nunique()) if "instrument" in labels.columns else 0,
        "horizons": horizon_values,
        "label_columns": label_columns,
        "label_available_counts": available_counts,
        "frame_schema_version": OBJECTIVE_LABEL_FRAME_SCHEMA_VERSION,
        "frame_schema_hash": _frame_schema_hash(labels),
    }


def write_objective_label_manifest(
    output_dir: str | Path,
    labels: pd.DataFrame,
    *,
    artifact_id: str,
    horizons: Sequence[int],
    labels_filename: str = "objective_labels.csv",
    manifest_filename: str = "objective_label_manifest.json",
) -> dict[str, object]:
    """Write objective labels and a stable manifest for notebook/dashboard review."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / labels_filename
    manifest_path = output / manifest_filename

    labels.to_csv(labels_path, index=False, lineterminator="\n")
    payload: dict[str, object] = {
        "schema_version": OBJECTIVE_LABEL_MANIFEST_SCHEMA_VERSION,
        "artifact_id": str(artifact_id),
        "artifact_type": "objective_labels",
        "producer": "src.objective.specs",
        "paths": {
            "labels": labels_filename,
            "manifest": manifest_filename,
        },
        "metadata": _label_manifest_metadata(labels, horizons=horizons),
    }
    payload["manifest_hash"] = stable_hash(payload)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _clean_numeric_pair(frame: pd.DataFrame, prediction_col: str, target_col: str) -> pd.DataFrame:
    values = frame.loc[:, [prediction_col, target_col]].copy()
    values[prediction_col] = pd.to_numeric(values[prediction_col], errors="coerce")
    values[target_col] = pd.to_numeric(values[target_col], errors="coerce")
    return values.replace([math.inf, -math.inf], pd.NA).dropna()


def _pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return 0.0
    value = x.astype(float).corr(y.astype(float), method="pearson")
    return 0.0 if pd.isna(value) else float(value)


def _rank_ic(frame: pd.DataFrame, prediction_col: str, target_col: str, group_col: str | None) -> tuple[float, int]:
    if group_col is None or group_col not in frame.columns:
        values = _clean_numeric_pair(frame, prediction_col, target_col)
        if len(values) < 2:
            return 0.0, 0
        score = values[prediction_col].rank(method="average").corr(
            values[target_col].rank(method="average"),
            method="pearson",
        )
        return (0.0 if pd.isna(score) else float(score), 1)

    scores: list[float] = []
    for _, group in frame.groupby(group_col, sort=True):
        values = _clean_numeric_pair(group, prediction_col, target_col)
        if len(values) < 2:
            continue
        score = values[prediction_col].rank(method="average").corr(
            values[target_col].rank(method="average"),
            method="pearson",
        )
        if not pd.isna(score):
            scores.append(float(score))
    if not scores:
        return 0.0, 0
    return float(sum(scores) / len(scores)), len(scores)


def evaluate_prediction_objectives(
    frame: pd.DataFrame,
    *,
    prediction_col: str = "prediction",
    target_col: str = "target_return",
    group_col: str | None = "date",
) -> dict[str, object]:
    """Evaluate common supervised objectives without introducing an RL reward surface."""

    required = {prediction_col, target_col}
    if group_col is not None:
        required.add(group_col)
    _require_columns(frame, required, context="prediction frame")
    values = frame.copy()
    clean = _clean_numeric_pair(values, prediction_col, target_col)
    if clean.empty:
        rank_ic, rank_group_count = _rank_ic(values, prediction_col, target_col, group_col)
        return {
            "schema_version": "objective.evaluation.v1",
            "row_count": 0,
            "mse": 0.0,
            "classification_accuracy": 0.0,
            "ic": 0.0,
            "rank_ic": rank_ic,
            "rank_group_count": rank_group_count,
        }

    error = clean[prediction_col].astype(float) - clean[target_col].astype(float)
    predicted_direction = clean[prediction_col].astype(float) > 0.0
    target_direction = clean[target_col].astype(float) > 0.0
    rank_ic, rank_group_count = _rank_ic(values, prediction_col, target_col, group_col)
    return {
        "schema_version": "objective.evaluation.v1",
        "row_count": int(len(clean)),
        "mse": float((error.pow(2)).mean()),
        "classification_accuracy": float((predicted_direction == target_direction).mean()),
        "ic": _pearson(clean[prediction_col], clean[target_col]),
        "rank_ic": rank_ic,
        "rank_group_count": int(rank_group_count),
    }


def _promotion_blockers(row: pd.Series, thresholds: PromotionThresholds) -> list[str]:
    blockers: list[str] = []
    checks = [
        ("return_pct", thresholds.min_return_pct, "min"),
        ("relative_return_pct", thresholds.min_relative_return_pct, "min"),
        ("sharpe", thresholds.min_sharpe, "min"),
        ("max_drawdown_pct", thresholds.max_drawdown_pct, "max"),
        ("turnover", thresholds.max_turnover, "max"),
    ]
    for column, threshold, kind in checks:
        if threshold is None or column not in row:
            continue
        value = float(row[column])
        if kind == "min" and value < float(threshold):
            blockers.append(column)
        if kind == "max" and value > float(threshold):
            blockers.append(column)
    return blockers


def build_promotion_scores(
    model_summary: pd.DataFrame,
    *,
    thresholds: PromotionThresholds | None = None,
) -> pd.DataFrame:
    """Rank model summaries with return-risk utility and explicit promotion gates."""

    _require_columns(
        model_summary,
        {"model", "return_pct", "sharpe", "max_drawdown_pct", "turnover"},
        context="model summary",
    )
    limits = thresholds or PromotionThresholds()
    rows: list[dict[str, object]] = []
    for _, row in model_summary.iterrows():
        score = utility_score(
            return_pct=float(row["return_pct"]),
            sharpe=float(row["sharpe"]),
            max_drawdown_pct=float(row["max_drawdown_pct"]),
            turnover=float(row["turnover"]),
        )
        blockers = _promotion_blockers(row, limits)
        rows.append(
            {
                "schema_version": "objective.promotion_score.v1",
                "rank": 0,
                "model": str(row["model"]),
                "promotion_score": score,
                "promotion_decision": "blocked" if blockers else "promote_candidate",
                "promotion_blockers": ",".join(blockers),
                "return_pct": float(row["return_pct"]),
                "relative_return_pct": float(row.get("relative_return_pct", 0.0)),
                "sharpe": float(row["sharpe"]),
                "max_drawdown_pct": float(row["max_drawdown_pct"]),
                "turnover": float(row["turnover"]),
                "thresholds": limits.to_dict(),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["promotion_decision", "promotion_score", "relative_return_pct"],
        ascending=[False, False, False],
    )
    if result.empty:
        return result
    result = result.reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result

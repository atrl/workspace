from __future__ import annotations

import pandas as pd


def _aligned(actual, predicted) -> pd.DataFrame:
    frame = pd.concat(
        [
            pd.Series(actual, dtype="float64", name="actual"),
            pd.Series(predicted, dtype="float64", name="predicted"),
        ],
        axis=1,
    ).dropna()
    if frame.empty:
        raise ValueError("objective metrics require at least one aligned observation")
    return frame


def classification_metrics(actual, predicted, *, threshold: float = 0.0) -> dict[str, float]:
    frame = _aligned(actual, predicted)
    actual_label = frame["actual"] > threshold
    predicted_label = frame["predicted"] > threshold
    accuracy = float((actual_label == predicted_label).mean())
    true_positive = int((actual_label & predicted_label).sum())
    false_positive = int((~actual_label & predicted_label).sum())
    false_negative = int((actual_label & ~predicted_label).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "source": "qlib_dataset_objective",
        "accuracy": round(accuracy, 10),
        "precision": round(float(precision), 10),
        "recall": round(float(recall), 10),
    }


def mse_metrics(actual, predicted) -> dict[str, float]:
    frame = _aligned(actual, predicted)
    error = frame["predicted"] - frame["actual"]
    return {
        "source": "qlib_dataset_objective",
        "mse": round(float((error**2).mean()), 10),
        "mae": round(float(error.abs().mean()), 10),
    }


def ic_metrics(actual, predicted) -> dict[str, float]:
    frame = _aligned(actual, predicted)
    if frame["actual"].nunique() <= 1 or frame["predicted"].nunique() <= 1:
        ic = 0.0
    else:
        ic = float(frame["actual"].corr(frame["predicted"], method="pearson"))
    return {"source": "qlib_dataset_objective", "ic": round(ic, 10)}


def ranking_metrics(actual, predicted) -> dict[str, float]:
    frame = _aligned(actual, predicted)
    if frame["actual"].nunique() <= 1 or frame["predicted"].nunique() <= 1:
        rank_ic = 0.0
    else:
        rank_ic = float(frame["actual"].corr(frame["predicted"], method="spearman"))
    return {"source": "qlib_dataset_objective", "rank_ic": round(rank_ic, 10)}


def utility_score(
    *,
    return_pct: float,
    max_drawdown_pct: float,
    turnover: float,
    ic: float,
    rank_ic: float,
) -> float:
    return round(
        float(return_pct + ic * 100.0 + rank_ic * 50.0 - abs(max_drawdown_pct) - turnover * 5.0),
        10,
    )


def promotion_scores(
    frame: pd.DataFrame,
    *,
    min_score: float,
    max_drawdown_floor: float,
) -> pd.DataFrame:
    required = {"model", "return_pct", "max_drawdown_pct", "turnover", "ic", "rank_ic", "recorder_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing qlib recorder metric columns: {missing}")

    result = frame.copy()
    result["promotion_score"] = [
        utility_score(
            return_pct=float(row["return_pct"]),
            max_drawdown_pct=float(row["max_drawdown_pct"]),
            turnover=float(row["turnover"]),
            ic=float(row["ic"]),
            rank_ic=float(row["rank_ic"]),
        )
        for _, row in result.iterrows()
    ]
    result["source"] = "qlib_recorder_metrics"
    result["promote"] = [
        bool(score >= min_score and drawdown >= max_drawdown_floor)
        for score, drawdown in zip(result["promotion_score"], result["max_drawdown_pct"], strict=True)
    ]
    result["promote"] = result["promote"].astype(object)
    return result.sort_values(["promotion_score", "return_pct"], ascending=[False, False]).reset_index(drop=True)

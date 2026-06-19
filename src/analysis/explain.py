from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def _mse(actual: pd.Series, predicted: pd.Series) -> float:
    diff = actual.astype(float) - predicted.astype(float)
    return float((diff**2).mean())


def feature_importance_table(importances: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in importances.items()
    ]
    result = pd.DataFrame(rows).sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def permutation_importance(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    predict_fn: Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    features = frame[feature_cols].copy()
    target = frame[target_col].astype(float)
    baseline_loss = _mse(target, predict_fn(features))
    importances: dict[str, float] = {}
    for feature in feature_cols:
        permuted = features.copy()
        permuted[feature] = list(reversed(permuted[feature].tolist()))
        importances[feature] = round(_mse(target, predict_fn(permuted)) - baseline_loss, 10)
    return feature_importance_table(importances)


def window_sensitivity(
    frame: pd.DataFrame,
    *,
    value_col: str,
    windows: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    rows: list[dict] = []
    for name, (start, end) in windows.items():
        window = data.loc[(data["date"] >= pd.Timestamp(start)) & (data["date"] <= pd.Timestamp(end))]
        contribution = abs(float(window[value_col].sum()))
        rows.append(
            {
                "window": name,
                "start": pd.Timestamp(start).date().isoformat(),
                "end": pd.Timestamp(end).date().isoformat(),
                "contribution": contribution,
            }
        )
    total = sum(row["contribution"] for row in rows)
    for row in rows:
        row["share"] = round(row["contribution"] / total, 10) if total else 0.0
    return pd.DataFrame(rows)

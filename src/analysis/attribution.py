from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.model.artifacts import ModelArtifact
from src.model.capabilities import PROXY_FIDELITIES
from src.model.registry import get_model_spec


@dataclass(frozen=True)
class ModelAttributionReport:
    feature_attribution: pd.DataFrame
    summary: dict[str, object]


def _predict_values(artifact: ModelArtifact, frame: pd.DataFrame) -> np.ndarray:
    return artifact.predict(frame)["prediction"].astype(float).to_numpy()


def _leaf_estimator(estimator: Any) -> Any:
    if hasattr(estimator, "steps"):
        return estimator.steps[-1][1]
    return estimator


def _native_importance_rows(artifact: ModelArtifact) -> list[dict[str, object]]:
    estimator = _leaf_estimator(artifact.estimator)
    features = list(artifact.feature_columns)
    rows: list[dict[str, object]] = []
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float).reshape(-1)
        for feature, value in zip(features, values):
            rows.append(
                {
                    "feature": feature,
                    "attribution_kind": "feature_importance",
                    "importance": abs(float(value)),
                    "direction": 0.0,
                }
            )
    elif hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        for feature, value in zip(features, values):
            rows.append(
                {
                    "feature": feature,
                    "attribution_kind": "coefficient",
                    "importance": abs(float(value)),
                    "direction": float(np.sign(value)),
                }
            )
    elif hasattr(estimator, "get_feature_importance"):
        values = np.asarray(estimator.get_feature_importance(), dtype=float).reshape(-1)
        for feature, value in zip(features, values):
            rows.append(
                {
                    "feature": feature,
                    "attribution_kind": "feature_importance",
                    "importance": abs(float(value)),
                    "direction": 0.0,
                }
            )
    return rows


def _baseline_values(background_frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> dict[str, float]:
    values: dict[str, float] = {}
    for feature in feature_columns:
        series = background_frame[feature].astype(float)
        value = float(series.median())
        if not np.isfinite(value):
            value = 0.0
        values[feature] = value
    return values


def _permutation_rows(
    artifact: ModelArtifact,
    frame: pd.DataFrame,
    *,
    baseline_prediction: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for feature in artifact.feature_columns:
        perturbed = frame.copy()
        perturbed[feature] = rng.permutation(perturbed[feature].astype(float).to_numpy())
        changed = _predict_values(artifact, perturbed) - baseline_prediction
        rows.append(
            {
                "feature": feature,
                "attribution_kind": "permutation",
                "importance": float(np.mean(np.abs(changed))),
                "direction": float(np.sign(np.mean(changed))),
            }
        )
    return rows


def _sensitivity_rows(
    artifact: ModelArtifact,
    frame: pd.DataFrame,
    *,
    baseline_prediction: np.ndarray,
    background_values: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in artifact.feature_columns:
        perturbed = frame.copy()
        perturbed[feature] = background_values[feature]
        changed = _predict_values(artifact, perturbed) - baseline_prediction
        rows.append(
            {
                "feature": feature,
                "attribution_kind": "sensitivity",
                "importance": float(np.mean(np.abs(changed))),
                "direction": float(np.sign(np.mean(changed))),
            }
        )
    return rows


def _rank_rows(rows: list[dict[str, object]], *, model_name: str, top_n: int | None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "feature",
                "attribution_kind",
                "importance",
                "normalized_importance",
                "rank",
                "direction",
            ]
        )
    frame = pd.DataFrame(rows)
    ranked: list[pd.DataFrame] = []
    for attribution_kind, group in frame.groupby("attribution_kind", sort=True):
        ordered = group.sort_values(["importance", "feature"], ascending=[False, True]).copy()
        if top_n is not None:
            ordered = ordered.head(top_n)
        total = float(ordered["importance"].sum())
        ordered["normalized_importance"] = ordered["importance"] / total if total > 1e-12 else 0.0
        ordered["rank"] = range(1, len(ordered) + 1)
        ordered["model"] = model_name
        ranked.append(ordered)
    result = pd.concat(ranked, ignore_index=True)
    columns = ["model", "feature", "attribution_kind", "importance", "normalized_importance", "rank", "direction"]
    return result.loc[:, columns].sort_values(["attribution_kind", "rank", "feature"]).reset_index(drop=True)


def explain_model_artifact(
    artifact: ModelArtifact,
    *,
    frame: pd.DataFrame,
    background_frame: pd.DataFrame | None = None,
    top_n: int | None = 10,
    seed: int = 7,
) -> ModelAttributionReport:
    if frame.empty:
        raise ValueError("frame must be non-empty")
    background = background_frame if background_frame is not None and not background_frame.empty else frame
    baseline_prediction = _predict_values(artifact, frame)
    rows = _native_importance_rows(artifact)
    rows.extend(
        _permutation_rows(
            artifact,
            frame,
            baseline_prediction=baseline_prediction,
            seed=seed,
        )
    )
    rows.extend(
        _sensitivity_rows(
            artifact,
            frame,
            baseline_prediction=baseline_prediction,
            background_values=_baseline_values(background, artifact.feature_columns),
        )
    )
    attribution = _rank_rows(rows, model_name=artifact.model_name, top_n=top_n)
    spec = get_model_spec(artifact.model_name)
    top_features = (
        attribution.groupby("feature")["normalized_importance"].sum().sort_values(ascending=False).head(top_n).index.tolist()
        if not attribution.empty
        else []
    )
    summary = {
        "schema_version": "analysis.model_attribution.v1",
        "model": artifact.model_name,
        "family": artifact.family,
        "objective": artifact.objective,
        "paper_fidelity": spec.paper_fidelity,
        "proxy_model": bool(spec.paper_fidelity in PROXY_FIDELITIES),
        "feature_count": int(len(artifact.feature_columns)),
        "row_count": int(len(frame)),
        "attribution_kinds": {
            str(kind): int(count)
            for kind, count in attribution["attribution_kind"].value_counts().sort_index().items()
        },
        "top_features": top_features,
        "notes": "Native coefficients/importances are included when available; permutation and sensitivity run for every artifact.",
    }
    return ModelAttributionReport(feature_attribution=attribution, summary=summary)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

import numpy as np
import pandas as pd

from src.model.artifacts import ModelArtifact, fit_model_artifact
from src.model.capabilities import PROXY_FIDELITIES, model_capability_table
from src.model.registry import get_model_spec


@dataclass(frozen=True)
class ModelAttributionReport:
    feature_attribution: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True)
class ModelAttributionCard:
    feature_attribution: pd.DataFrame
    top_features_by_model: pd.DataFrame
    model_capabilities: pd.DataFrame
    model_summaries: list[dict[str, object]]
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


def _top_features_by_model(feature_attribution: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if feature_attribution.empty:
        return pd.DataFrame(columns=["model", "feature", "normalized_importance", "rank"])
    grouped = (
        feature_attribution.groupby(["model", "feature"], as_index=False)["normalized_importance"]
        .sum()
        .sort_values(["model", "normalized_importance", "feature"], ascending=[True, False, True])
    )
    ranked: list[pd.DataFrame] = []
    for _, group in grouped.groupby("model", sort=True):
        ordered = group.head(top_n).copy()
        ordered["rank"] = range(1, len(ordered) + 1)
        ranked.append(ordered)
    return pd.concat(ranked, ignore_index=True).loc[:, ["model", "feature", "normalized_importance", "rank"]]


def build_model_attribution_card(
    *,
    model_names: Iterable[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Iterable[str],
    top_n: int = 6,
) -> ModelAttributionCard:
    if train.empty:
        raise ValueError("train frame must be non-empty")
    if test.empty:
        raise ValueError("test frame must be non-empty")
    selected_models = tuple(model_names)
    selected_features = tuple(feature_columns)
    if not selected_models:
        raise ValueError("model_names must be non-empty")
    if not selected_features:
        raise ValueError("feature_columns must be non-empty")

    reports: list[ModelAttributionReport] = []
    for model_name in selected_models:
        artifact = fit_model_artifact(model_name, train=train, feature_columns=selected_features)
        reports.append(
            explain_model_artifact(
                artifact,
                frame=test,
                background_frame=train,
                top_n=top_n,
            )
        )

    feature_attribution = pd.concat([report.feature_attribution for report in reports], ignore_index=True)
    top_features = _top_features_by_model(feature_attribution, top_n=top_n)
    capabilities = model_capability_table(selected_models)
    attribution_kinds = {
        str(kind): int(count)
        for kind, count in feature_attribution["attribution_kind"].value_counts().sort_index().items()
    }
    model_summaries = [report.summary for report in reports]
    proxy_models = capabilities.loc[capabilities["is_proxy"].astype(bool), "model"].astype(str).tolist()
    summary = {
        "status": "completed",
        "schema_version": "analysis.model_attribution.batch.v1",
        "models": list(selected_models),
        "model_count": int(len(selected_models)),
        "train_start": str(pd.to_datetime(train["date"]).min().date()),
        "train_end": str(pd.to_datetime(train["date"]).max().date()),
        "test_start": str(pd.to_datetime(test["date"]).min().date()),
        "test_end": str(pd.to_datetime(test["date"]).max().date()),
        "feature_count": int(len(selected_features)),
        "top_n": int(top_n),
        "attribution_kinds": attribution_kinds,
        "proxy_models": proxy_models,
        "artifacts": {
            "feature_attribution": "feature_attribution.csv",
            "model_attribution_summaries": "model_attribution_summaries.json",
            "model_capabilities": "model_capabilities.csv",
            "top_features_by_model": "top_features_by_model.csv",
        },
        "risk_notes": [
            "Permutation 和 sensitivity 是局部解释工具，不等价于因果解释。",
            "proxy_feature_adapter 模型的归因解释的是当前 proxy feature path，不等价于完整论文结构。",
            "短测试窗归因只适合 review，应继续在 walk-forward 和 regime 分段中复验。",
        ],
    }
    return ModelAttributionCard(
        feature_attribution=feature_attribution,
        top_features_by_model=top_features,
        model_capabilities=capabilities,
        model_summaries=model_summaries,
        summary=summary,
    )


def write_model_attribution_card(
    output_dir: str | Path,
    *,
    model_names: Iterable[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Iterable[str],
    top_n: int = 6,
) -> ModelAttributionCard:
    card = Path(output_dir)
    card.mkdir(parents=True, exist_ok=True)
    result = build_model_attribution_card(
        model_names=model_names,
        train=train,
        test=test,
        feature_columns=feature_columns,
        top_n=top_n,
    )
    result.feature_attribution.to_csv(card / "feature_attribution.csv", index=False)
    result.top_features_by_model.to_csv(card / "top_features_by_model.csv", index=False)
    result.model_capabilities.to_csv(card / "model_capabilities.csv", index=False)
    (card / "model_attribution_summaries.json").write_text(
        json.dumps(result.model_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (card / "analysis_summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result

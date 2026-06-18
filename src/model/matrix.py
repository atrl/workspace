from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis.exports import build_regime_window_analysis_report, write_analysis_report
from src.analysis.qlib_ext import analyze_model_effects_by_regime_and_window
from src.analysis.reports import compare_models, compute_regime_metrics, compute_window_metrics
from src.backtest import backtest_prediction_frame
from src.core import ArtifactManifest, DatasetSnapshot
from src.data import build_supervised_dataset, load_soxl_support_frame
from src.factor import prepare_factor_feature_sets
from src.model.capabilities import model_capability_table, model_fidelity_summary
from src.model.registry import QUANTBENCH_MODEL_NAMES
from src.model.training import fit_predict_model


@dataclass(frozen=True)
class QuantBenchMatrixOutputs:
    predictions: pd.DataFrame
    daily: pd.DataFrame
    regime_metrics: pd.DataFrame
    window_metrics: pd.DataFrame
    window_regime_metrics: pd.DataFrame
    model_comparison: pd.DataFrame
    qlib_risk_metrics: pd.DataFrame
    model_effect_deltas: pd.DataFrame
    summary: dict


def _enrich_comparison_with_variant_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    result = comparison.copy()
    if "base_model" not in result.columns:
        result["base_model"] = result["model"].astype(str).str.split("__", n=1).str[0]
    if "feature_set" not in result.columns:
        split = result["model"].astype(str).str.split("__", n=1, expand=True)
        result["feature_set"] = split[1] if split.shape[1] > 1 else "all_features"
    front = ["rank", "model", "base_model", "feature_set"]
    remaining = [column for column in result.columns if column not in front]
    return result.loc[:, [column for column in front if column in result.columns] + remaining]


def _adapter_summary(predictions: pd.DataFrame) -> dict[str, object]:
    statuses = sorted(str(value) for value in predictions["adapter_status"].dropna().unique().tolist())
    adapters = sorted(str(value) for value in predictions["model_adapter"].dropna().unique().tolist())
    return {
        "adapter_status": statuses[0] if len(statuses) == 1 else "mixed_adapters",
        "adapter_statuses": statuses,
        "model_adapters": adapters,
    }


def _dataset_snapshot(*, start: str, end: str) -> DatasetSnapshot:
    return DatasetSnapshot(
        name="soxl_support_daily",
        universe=("SOXL", "SOXX", "VIX"),
        start=start,
        end=end,
        calendar_path="data/qlib/soxl/calendars/day.txt",
        data_paths=("data/processed/panels/soxl_soxx_vix_full.csv",),
        metadata={"source": "load_soxl_support_frame"},
    )


def _write_matrix_manifest(
    card: Path,
    *,
    artifact_id: str,
    start: str,
    end: str,
    paths: dict[str, str],
    metadata: dict,
) -> dict:
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type="model_matrix",
        producer="src.model.matrix",
        schema_version="artifact.v1",
        dataset=_dataset_snapshot(start=start, end=end),
        paths=paths,
        metadata=metadata,
    )
    payload = manifest.to_dict()
    payload["manifest_hash"] = manifest.manifest_hash()
    (card / "artifact_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _write_analysis_sidecars(
    card: Path,
    *,
    artifact_id: str,
    start: str,
    end: str,
    summary: dict,
    effect_analysis,
    model_comparison: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    window_metrics: pd.DataFrame,
    manifest_metadata: dict,
) -> dict:
    report = build_regime_window_analysis_report(
        artifact_id=artifact_id,
        effect_analysis=effect_analysis,
        model_comparison=model_comparison,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
        summary=summary,
        metadata=manifest_metadata,
    )
    write_analysis_report(
        report,
        card,
        dataset=_dataset_snapshot(start=start, end=end),
        manifest_metadata=manifest_metadata,
    )
    return report.summary


def run_quantbench_matrix_smoke(
    *,
    card_dir: Path,
    start: str,
    end: str,
    train_end: str,
    model_names: Iterable[str] = QUANTBENCH_MODEL_NAMES,
) -> QuantBenchMatrixOutputs:
    card = Path(card_dir)
    card.mkdir(parents=True, exist_ok=True)
    selected_model_names = tuple(model_names)
    price_frame = load_soxl_support_frame(start=start, end=end)
    dataset = build_supervised_dataset(price_frame, horizon=1)
    train_end_ts = pd.Timestamp(train_end)
    train = dataset.frame.loc[dataset.frame["date"] <= train_end_ts].copy()
    test = dataset.frame.loc[dataset.frame["date"] > train_end_ts].copy()
    if train.empty or test.empty:
        raise ValueError("train/test split produced an empty side")

    prediction_frames = [
        fit_predict_model(name, train=train, test=test, feature_columns=dataset.feature_columns)
        for name in selected_model_names
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    adapter_summary = _adapter_summary(predictions)
    daily = backtest_prediction_frame(predictions, price_frame=price_frame)
    regime_metrics = compute_regime_metrics(daily)
    windows = {"test": (str(test["date"].min().date()), str(test["date"].max().date()))}
    window_metrics = compute_window_metrics(daily, windows=windows)
    model_comparison = compare_models(daily)
    model_capabilities = model_capability_table(selected_model_names)
    model_fidelity = model_fidelity_summary(selected_model_names)
    effect_analysis = analyze_model_effects_by_regime_and_window(
        daily,
        windows=windows,
        baseline_model=str(model_comparison.iloc[0]["model"]),
    )
    window_regime_metrics = effect_analysis.window_regime_metrics
    summary = {
        "status": "smoke_completed",
        "model_count": int(predictions["model"].nunique()),
        "start": start,
        "end": end,
        "train_end": train_end,
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "top_model": str(model_comparison.iloc[0]["model"]),
        **adapter_summary,
        "model_fidelity": model_fidelity,
        "analysis_layers": [
            "regime_metrics",
            "window_metrics",
            "window_regime_metrics",
            "qlib_risk_metrics",
            "model_effect_deltas",
        ],
        "analysis_summary": effect_analysis.summary,
        "note": "Matrix keeps one contract while model_fidelity distinguishes native implementations from proxy adapters.",
    }

    analysis_metadata = {
        "status": summary["status"],
        "train_end": train_end,
        "adapter_status": summary["adapter_status"],
        "adapter_statuses": summary["adapter_statuses"],
        "model_adapters": summary["model_adapters"],
        "model_fidelity": summary["model_fidelity"],
        "model_count": summary["model_count"],
    }
    summary = _write_analysis_sidecars(
        card,
        artifact_id=card.name,
        start=start,
        end=end,
        summary=summary,
        effect_analysis=effect_analysis,
        model_comparison=model_comparison,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
        manifest_metadata=analysis_metadata,
    )
    predictions.to_csv(card / "predictions.csv", index=False)
    daily.to_csv(card / "daily_backtest.csv", index=False)
    model_capabilities.to_csv(card / "model_capabilities.csv", index=False)
    _write_matrix_manifest(
        card,
        artifact_id=card.name,
        start=start,
        end=end,
        paths={
            "analysis_summary": "analysis_summary.json",
            "predictions": "predictions.csv",
            "daily_backtest": "daily_backtest.csv",
            "regime_metrics": "regime_metrics.csv",
            "window_metrics": "window_metrics.csv",
            "window_regime_metrics": "window_regime_metrics.csv",
            "model_comparison": "model_comparison.csv",
            "model_capabilities": "model_capabilities.csv",
            "qlib_risk_metrics": "qlib_risk_metrics.csv",
            "qlib_return_frame": "qlib_return_frame.csv",
            "segment_metrics": "segment_metrics.csv",
            "model_effect_deltas": "model_effect_deltas.csv",
            "analysis_manifest": "analysis_manifest.json",
        },
        metadata=analysis_metadata,
    )
    return QuantBenchMatrixOutputs(
        predictions=predictions,
        daily=daily,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
        window_regime_metrics=window_regime_metrics,
        model_comparison=model_comparison,
        qlib_risk_metrics=effect_analysis.qlib_risk_metrics,
        model_effect_deltas=effect_analysis.model_effect_deltas,
        summary=summary,
    )


def run_factor_feature_set_matrix(
    *,
    card_dir: Path,
    start: str,
    end: str,
    train_end: str,
    model_names: Iterable[str] = ("ridge_mse", "random_forest_clf"),
    top_n_factors: int = 5,
) -> QuantBenchMatrixOutputs:
    card = Path(card_dir)
    card.mkdir(parents=True, exist_ok=True)
    selected_model_names = tuple(model_names)
    price_frame = load_soxl_support_frame(start=start, end=end)
    dataset = build_supervised_dataset(price_frame, horizon=1)
    train_end_ts = pd.Timestamp(train_end)
    train = dataset.frame.loc[dataset.frame["date"] <= train_end_ts].copy()
    test = dataset.frame.loc[dataset.frame["date"] > train_end_ts].copy()
    if train.empty or test.empty:
        raise ValueError("train/test split produced an empty side")

    plan = prepare_factor_feature_sets(
        train=train,
        test=test,
        factor_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        top_n=top_n_factors,
    )
    prediction_frames: list[pd.DataFrame] = []
    for feature_set in plan.feature_sets:
        for model_name in selected_model_names:
            predictions = fit_predict_model(
                model_name,
                train=plan.train,
                test=plan.test,
                feature_columns=feature_set.feature_columns,
            )
            predictions["base_model"] = model_name
            predictions["feature_set"] = feature_set.name
            predictions["model"] = f"{model_name}__{feature_set.name}"
            predictions["feature_columns"] = ",".join(feature_set.feature_columns)
            prediction_frames.append(predictions)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    adapter_summary = _adapter_summary(predictions)
    daily = backtest_prediction_frame(predictions, price_frame=price_frame)
    regime_metrics = compute_regime_metrics(daily)
    windows = {"test": (str(test["date"].min().date()), str(test["date"].max().date()))}
    window_metrics = compute_window_metrics(daily, windows=windows)
    model_comparison = _enrich_comparison_with_variant_columns(compare_models(daily))
    model_capabilities = model_capability_table(selected_model_names)
    model_fidelity = model_fidelity_summary(selected_model_names)
    effect_analysis = analyze_model_effects_by_regime_and_window(
        daily,
        windows=windows,
        baseline_model=str(model_comparison.iloc[0]["model"]),
    )
    window_regime_metrics = effect_analysis.window_regime_metrics
    summary = {
        "status": "feature_set_matrix_completed",
        "base_model_count": int(len(tuple(model_names))),
        "feature_set_count": int(len(plan.feature_sets)),
        "variant_count": int(predictions["model"].nunique()),
        "start": start,
        "end": end,
        "train_end": train_end,
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "top_variant": str(model_comparison.iloc[0]["model"]),
        **adapter_summary,
        "model_fidelity": model_fidelity,
        "analysis_layers": [
            "regime_metrics",
            "window_metrics",
            "window_regime_metrics",
            "qlib_risk_metrics",
            "model_effect_deltas",
        ],
        **plan.summary,
        "feature_sets": [feature_set.to_dict() for feature_set in plan.feature_sets],
        "analysis_summary": effect_analysis.summary,
        "note": "Compares the same model adapters across raw, IC-selected, and composite factor sets.",
    }

    analysis_metadata = {
        "status": summary["status"],
        "train_end": train_end,
        "adapter_status": summary["adapter_status"],
        "adapter_statuses": summary["adapter_statuses"],
        "model_adapters": summary["model_adapters"],
        "model_fidelity": summary["model_fidelity"],
        "base_model_count": summary["base_model_count"],
        "feature_set_count": summary["feature_set_count"],
        "variant_count": summary["variant_count"],
        "selection_source": summary["selection_source"],
    }
    summary = _write_analysis_sidecars(
        card,
        artifact_id=card.name,
        start=start,
        end=end,
        summary=summary,
        effect_analysis=effect_analysis,
        model_comparison=model_comparison,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
        manifest_metadata=analysis_metadata,
    )
    predictions.to_csv(card / "predictions.csv", index=False)
    daily.to_csv(card / "daily_backtest.csv", index=False)
    model_capabilities.to_csv(card / "model_capabilities.csv", index=False)
    plan.diagnostics.ic.to_csv(card / "factor_ic.csv", index=False)
    (card / "feature_set_specs.json").write_text(
        json.dumps([feature_set.to_dict() for feature_set in plan.feature_sets], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_matrix_manifest(
        card,
        artifact_id=card.name,
        start=start,
        end=end,
        paths={
            "analysis_summary": "analysis_summary.json",
            "predictions": "predictions.csv",
            "daily_backtest": "daily_backtest.csv",
            "regime_metrics": "regime_metrics.csv",
            "window_metrics": "window_metrics.csv",
            "window_regime_metrics": "window_regime_metrics.csv",
            "model_comparison": "model_comparison.csv",
            "model_capabilities": "model_capabilities.csv",
            "qlib_risk_metrics": "qlib_risk_metrics.csv",
            "qlib_return_frame": "qlib_return_frame.csv",
            "segment_metrics": "segment_metrics.csv",
            "model_effect_deltas": "model_effect_deltas.csv",
            "analysis_manifest": "analysis_manifest.json",
            "factor_ic": "factor_ic.csv",
            "feature_set_specs": "feature_set_specs.json",
        },
        metadata=analysis_metadata,
    )
    return QuantBenchMatrixOutputs(
        predictions=predictions,
        daily=daily,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
        window_regime_metrics=window_regime_metrics,
        model_comparison=model_comparison,
        qlib_risk_metrics=effect_analysis.qlib_risk_metrics,
        model_effect_deltas=effect_analysis.model_effect_deltas,
        summary=summary,
    )

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis import analyze_model_effects_by_regime_and_window, compare_models, compute_regime_metrics, compute_window_metrics
from src.backtest import backtest_prediction_frame
from src.data import build_supervised_dataset, load_soxl_support_frame
from src.factor import prepare_factor_feature_sets
from src.model.registry import QUANTBENCH_MODEL_NAMES
from src.model.training import fit_predict_model


@dataclass(frozen=True)
class QuantBenchMatrixOutputs:
    predictions: pd.DataFrame
    daily: pd.DataFrame
    regime_metrics: pd.DataFrame
    window_metrics: pd.DataFrame
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
    price_frame = load_soxl_support_frame(start=start, end=end)
    dataset = build_supervised_dataset(price_frame, horizon=1)
    train_end_ts = pd.Timestamp(train_end)
    train = dataset.frame.loc[dataset.frame["date"] <= train_end_ts].copy()
    test = dataset.frame.loc[dataset.frame["date"] > train_end_ts].copy()
    if train.empty or test.empty:
        raise ValueError("train/test split produced an empty side")

    prediction_frames = [
        fit_predict_model(name, train=train, test=test, feature_columns=dataset.feature_columns)
        for name in tuple(model_names)
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    daily = backtest_prediction_frame(predictions, price_frame=price_frame)
    regime_metrics = compute_regime_metrics(daily)
    windows = {"test": (str(test["date"].min().date()), str(test["date"].max().date()))}
    window_metrics = compute_window_metrics(daily, windows=windows)
    model_comparison = compare_models(daily)
    effect_analysis = analyze_model_effects_by_regime_and_window(
        daily,
        windows=windows,
        baseline_model=str(model_comparison.iloc[0]["model"]),
    )
    summary = {
        "status": "smoke_completed",
        "model_count": int(predictions["model"].nunique()),
        "start": start,
        "end": end,
        "train_end": train_end,
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "top_model": str(model_comparison.iloc[0]["model"]),
        "adapter_status": "smoke_adapter",
        "analysis_layers": ["regime_metrics", "window_metrics", "qlib_risk_metrics", "model_effect_deltas"],
        "analysis_summary": effect_analysis.summary,
        "note": "Smoke adapters validate the full matrix contract; paper-strict replacements can be swapped per ModelSpec.",
    }

    predictions.to_csv(card / "predictions.csv", index=False)
    daily.to_csv(card / "daily_backtest.csv", index=False)
    regime_metrics.to_csv(card / "regime_metrics.csv", index=False)
    window_metrics.to_csv(card / "window_metrics.csv", index=False)
    model_comparison.to_csv(card / "model_comparison.csv", index=False)
    effect_analysis.qlib_risk_metrics.to_csv(card / "qlib_risk_metrics.csv", index=False)
    effect_analysis.model_effect_deltas.to_csv(card / "model_effect_deltas.csv", index=False)
    (card / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return QuantBenchMatrixOutputs(
        predictions=predictions,
        daily=daily,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
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
        for model_name in tuple(model_names):
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
    daily = backtest_prediction_frame(predictions, price_frame=price_frame)
    regime_metrics = compute_regime_metrics(daily)
    windows = {"test": (str(test["date"].min().date()), str(test["date"].max().date()))}
    window_metrics = compute_window_metrics(daily, windows=windows)
    model_comparison = _enrich_comparison_with_variant_columns(compare_models(daily))
    effect_analysis = analyze_model_effects_by_regime_and_window(
        daily,
        windows=windows,
        baseline_model=str(model_comparison.iloc[0]["model"]),
    )
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
        "adapter_status": "smoke_adapter",
        "analysis_layers": ["regime_metrics", "window_metrics", "qlib_risk_metrics", "model_effect_deltas"],
        **plan.summary,
        "feature_sets": [feature_set.to_dict() for feature_set in plan.feature_sets],
        "analysis_summary": effect_analysis.summary,
        "note": "Compares the same smoke model adapters across raw, IC-selected, and composite factor sets.",
    }

    predictions.to_csv(card / "predictions.csv", index=False)
    daily.to_csv(card / "daily_backtest.csv", index=False)
    regime_metrics.to_csv(card / "regime_metrics.csv", index=False)
    window_metrics.to_csv(card / "window_metrics.csv", index=False)
    model_comparison.to_csv(card / "model_comparison.csv", index=False)
    effect_analysis.qlib_risk_metrics.to_csv(card / "qlib_risk_metrics.csv", index=False)
    effect_analysis.model_effect_deltas.to_csv(card / "model_effect_deltas.csv", index=False)
    plan.diagnostics.ic.to_csv(card / "factor_ic.csv", index=False)
    (card / "feature_set_specs.json").write_text(
        json.dumps([feature_set.to_dict() for feature_set in plan.feature_sets], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (card / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return QuantBenchMatrixOutputs(
        predictions=predictions,
        daily=daily,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
        model_comparison=model_comparison,
        qlib_risk_metrics=effect_analysis.qlib_risk_metrics,
        model_effect_deltas=effect_analysis.model_effect_deltas,
        summary=summary,
    )

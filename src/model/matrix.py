from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis.exports import build_regime_window_analysis_report, write_analysis_report
from src.analysis.qlib_ext import analyze_model_effects_by_regime_and_window
from src.analysis.reports import compare_models, compute_regime_metrics, compute_window_metrics
from src.backtest import (
    BacktestConfig,
    BacktestResult,
    CostModel,
    PositionPolicy,
    build_walk_forward_splits,
    run_prediction_backtest,
)
from src.core import ArtifactManifest, DatasetSnapshot
from src.data import build_supervised_dataset, load_soxl_support_frame
from src.factor import prepare_factor_feature_sets
from src.model.capabilities import model_capability_table, model_fidelity_summary
from src.model.registry import QUANTBENCH_MODEL_NAMES
from src.model.training import fit_predict_model
from src.objective import build_promotion_scores, evaluate_prediction_objectives


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


def _objective_score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in predictions.groupby("model", sort=True):
        scores = evaluate_prediction_objectives(group, group_col="date")
        row: dict[str, object] = {
            "model": str(model),
            "objective": str(group["objective"].dropna().iloc[0]) if "objective" in group else "",
            "family": str(group["family"].dropna().iloc[0]) if "family" in group else "",
            **scores,
        }
        if "base_model" in group.columns:
            row["base_model"] = str(group["base_model"].dropna().iloc[0])
        if "feature_set" in group.columns:
            row["feature_set"] = str(group["feature_set"].dropna().iloc[0])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    front = ["schema_version", "model", "base_model", "feature_set", "objective", "family"]
    remaining = [column for column in result.columns if column not in front]
    return result.loc[:, [column for column in front if column in result.columns] + remaining].reset_index(drop=True)


def _objective_evaluation_summary(objective_scores: pd.DataFrame) -> dict[str, object]:
    if objective_scores.empty:
        return {
            "schema_version": "objective.matrix_evaluation.v1",
            "model_count": 0,
            "avg_mse": 0.0,
            "avg_ic": 0.0,
            "avg_rank_ic": 0.0,
        }
    return {
        "schema_version": "objective.matrix_evaluation.v1",
        "model_count": int(objective_scores["model"].nunique()),
        "avg_mse": float(objective_scores["mse"].astype(float).mean()),
        "avg_ic": float(objective_scores["ic"].astype(float).mean()),
        "avg_rank_ic": float(objective_scores["rank_ic"].astype(float).mean()),
    }


def _promotion_score_table(model_comparison: pd.DataFrame) -> pd.DataFrame:
    scores = build_promotion_scores(model_comparison)
    metadata_columns = [column for column in ["model", "base_model", "feature_set"] if column in model_comparison.columns]
    if scores.empty or not metadata_columns:
        return scores
    metadata = model_comparison.loc[:, metadata_columns].drop_duplicates(subset=["model"])
    result = scores.merge(metadata, on="model", how="left")
    front = ["schema_version", "rank", "model", "base_model", "feature_set"]
    remaining = [column for column in result.columns if column not in front]
    return result.loc[:, [column for column in front if column in result.columns] + remaining].reset_index(drop=True)


def _promotion_summary(promotion_scores: pd.DataFrame) -> dict[str, object]:
    if promotion_scores.empty:
        return {
            "schema_version": "objective.promotion_summary.v1",
            "model_count": 0,
            "candidate_count": 0,
            "blocked_count": 0,
            "top_model": "",
            "top_decision": "",
        }
    decisions = promotion_scores["promotion_decision"].astype(str)
    top = promotion_scores.sort_values(["rank", "promotion_score"], ascending=[True, False]).iloc[0]
    return {
        "schema_version": "objective.promotion_summary.v1",
        "model_count": int(promotion_scores["model"].nunique()),
        "candidate_count": int((decisions == "promote_candidate").sum()),
        "blocked_count": int((decisions == "blocked").sum()),
        "top_model": str(top["model"]),
        "top_decision": str(top["promotion_decision"]),
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
    extra_paths: dict[str, str] | None = None,
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
        extra_paths=extra_paths,
    )
    return report.summary


def _default_backtest_config(*, test_start: str, test_end: str) -> BacktestConfig:
    return BacktestConfig(
        start=test_start,
        end=test_end,
        benchmark="SOXL",
        position_policy=PositionPolicy(min_position=0.0, max_position=1.0, neutral_position=0.5, lag=1),
        cost_model=CostModel(open_cost_bps=1.0, close_cost_bps=1.0, slippage_bps=1.0),
    )


def _walk_forward_splits(frame: pd.DataFrame, *, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_size = min(max(1, int(len(train) // 2)), int(len(train)))
    test_size = min(max(1, int(len(test) // 2)), int(len(test)))
    splits = build_walk_forward_splits(
        frame["date"],
        train_size=train_size,
        test_size=test_size,
        step_size=test_size,
    )
    return pd.DataFrame([split.to_dict() for split in splits])


def _run_contract_backtest(
    *,
    predictions: pd.DataFrame,
    price_frame: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    full_frame: pd.DataFrame,
) -> tuple[BacktestResult, pd.DataFrame, dict[str, object]]:
    config = _default_backtest_config(
        test_start=str(test["date"].min().date()),
        test_end=str(test["date"].max().date()),
    )
    result = run_prediction_backtest(predictions, price_frame=price_frame, config=config)
    splits = _walk_forward_splits(full_frame, train=train, test=test)
    walk_forward_summary = {
        "schema_version": "backtest.walk_forward.v1",
        "fold_count": int(len(splits)),
        "train_size": int(min(max(1, int(len(train) // 2)), int(len(train)))),
        "test_size": int(min(max(1, int(len(test) // 2)), int(len(test)))),
    }
    return result, splits, walk_forward_summary


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
    backtest_result, walk_forward_splits, walk_forward_summary = _run_contract_backtest(
        predictions=predictions,
        price_frame=price_frame,
        train=train,
        test=test,
        full_frame=dataset.frame,
    )
    daily = backtest_result.daily
    regime_metrics = compute_regime_metrics(daily)
    windows = {"test": (str(test["date"].min().date()), str(test["date"].max().date()))}
    window_metrics = compute_window_metrics(daily, windows=windows)
    model_comparison = compare_models(daily)
    objective_scores = _objective_score_table(predictions)
    promotion_scores = _promotion_score_table(model_comparison)
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
        "backtest_contract": backtest_result.config.to_dict(),
        "walk_forward": walk_forward_summary,
        "objective_evaluation": _objective_evaluation_summary(objective_scores),
        "promotion_summary": _promotion_summary(promotion_scores),
        **adapter_summary,
        "model_fidelity": model_fidelity,
        "analysis_layers": [
            "regime_metrics",
            "window_metrics",
            "window_regime_metrics",
            "qlib_risk_metrics",
            "model_effect_deltas",
            "objective_scores",
            "promotion_scores",
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
        "backtest_contract": summary["backtest_contract"],
        "walk_forward": summary["walk_forward"],
        "objective_evaluation": summary["objective_evaluation"],
        "promotion_summary": summary["promotion_summary"],
    }
    (card / "backtest_config.json").write_text(
        json.dumps(backtest_result.config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    walk_forward_splits.to_csv(card / "walk_forward_splits.csv", index=False)
    objective_scores.to_csv(card / "objective_scores.csv", index=False)
    promotion_scores.to_csv(card / "promotion_scores.csv", index=False)
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
        extra_paths={
            "backtest_config": "backtest_config.json",
            "walk_forward_splits": "walk_forward_splits.csv",
            "objective_scores": "objective_scores.csv",
            "promotion_scores": "promotion_scores.csv",
        },
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
            "backtest_config": "backtest_config.json",
            "walk_forward_splits": "walk_forward_splits.csv",
            "objective_scores": "objective_scores.csv",
            "promotion_scores": "promotion_scores.csv",
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
    backtest_result, walk_forward_splits, walk_forward_summary = _run_contract_backtest(
        predictions=predictions,
        price_frame=price_frame,
        train=train,
        test=test,
        full_frame=dataset.frame,
    )
    daily = backtest_result.daily
    regime_metrics = compute_regime_metrics(daily)
    windows = {"test": (str(test["date"].min().date()), str(test["date"].max().date()))}
    window_metrics = compute_window_metrics(daily, windows=windows)
    model_comparison = _enrich_comparison_with_variant_columns(compare_models(daily))
    objective_scores = _objective_score_table(predictions)
    promotion_scores = _promotion_score_table(model_comparison)
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
        "backtest_contract": backtest_result.config.to_dict(),
        "walk_forward": walk_forward_summary,
        "objective_evaluation": _objective_evaluation_summary(objective_scores),
        "promotion_summary": _promotion_summary(promotion_scores),
        **adapter_summary,
        "model_fidelity": model_fidelity,
        "analysis_layers": [
            "regime_metrics",
            "window_metrics",
            "window_regime_metrics",
            "qlib_risk_metrics",
            "model_effect_deltas",
            "objective_scores",
            "promotion_scores",
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
        "backtest_contract": summary["backtest_contract"],
        "walk_forward": summary["walk_forward"],
        "objective_evaluation": summary["objective_evaluation"],
        "promotion_summary": summary["promotion_summary"],
    }
    (card / "backtest_config.json").write_text(
        json.dumps(backtest_result.config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    walk_forward_splits.to_csv(card / "walk_forward_splits.csv", index=False)
    objective_scores.to_csv(card / "objective_scores.csv", index=False)
    promotion_scores.to_csv(card / "promotion_scores.csv", index=False)
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
        extra_paths={
            "backtest_config": "backtest_config.json",
            "walk_forward_splits": "walk_forward_splits.csv",
            "objective_scores": "objective_scores.csv",
            "promotion_scores": "promotion_scores.csv",
        },
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
            "backtest_config": "backtest_config.json",
            "walk_forward_splits": "walk_forward_splits.csv",
            "objective_scores": "objective_scores.csv",
            "promotion_scores": "promotion_scores.csv",
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

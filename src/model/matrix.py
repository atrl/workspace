from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis.exports import build_regime_window_analysis_report, write_analysis_report
from src.analysis.qlib_ext import analyze_model_effects_by_regime_and_window
from src.analysis.reports import compare_models, compute_regime_metrics, compute_window_metrics
from src.analysis.windows import build_analysis_windows
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
from src.model.registry import QUANTBENCH_MODEL_NAMES, get_model_spec
from src.model.training import fit_predict_model, model_adapter_name, model_adapter_status
from src.objective import build_promotion_scores, evaluate_prediction_objectives


@dataclass(frozen=True)
class QuantBenchMatrixOutputs:
    predictions: pd.DataFrame
    daily: pd.DataFrame
    model_run_ledger: pd.DataFrame
    regime_metrics: pd.DataFrame
    window_metrics: pd.DataFrame
    window_regime_metrics: pd.DataFrame
    model_comparison: pd.DataFrame
    qlib_risk_metrics: pd.DataFrame
    model_effect_deltas: pd.DataFrame
    summary: dict


MODEL_RUN_LEDGER_SCHEMA_VERSION = "model.run_ledger.v1"
MODEL_RUN_LEDGER_SUMMARY_SCHEMA_VERSION = "model.run_ledger_summary.v1"


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


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _group_for_model(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    if frame.empty or "model" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["model"].astype(str) == model].copy()


def _row_for_model(frame: pd.DataFrame, model: str) -> pd.Series:
    group = _group_for_model(frame, model)
    if group.empty:
        return pd.Series(dtype=object)
    return group.iloc[0]


def _row_value(row: pd.Series, column: str, default: object = "") -> object:
    if row.empty or column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return value


def _float_value(row: pd.Series, column: str) -> float:
    value = _row_value(row, column, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_value(row: pd.Series, column: str) -> int:
    value = _row_value(row, column, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _manifest_has_paths(card: Path, manifest: dict[str, object], required_paths: Iterable[str]) -> bool:
    paths = manifest.get("paths", {})
    if not isinstance(paths, dict):
        return False
    for key in required_paths:
        value = paths.get(key)
        if not value or not (card / str(value)).exists():
            return False
    return True


def build_model_run_ledger(
    card_dir: Path,
    *,
    expected_model_names: Iterable[str] = QUANTBENCH_MODEL_NAMES,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the durable per-model proof that a matrix run trained and backtested each model."""

    card = Path(card_dir)
    expected = tuple(dict.fromkeys(str(name) for name in expected_model_names))
    predictions = _read_csv_if_exists(card / "predictions.csv")
    daily = _read_csv_if_exists(card / "daily_backtest.csv")
    regime_metrics = _read_csv_if_exists(card / "regime_metrics.csv")
    window_metrics = _read_csv_if_exists(card / "window_metrics.csv")
    model_comparison = _read_csv_if_exists(card / "model_comparison.csv")
    model_capabilities = _read_csv_if_exists(card / "model_capabilities.csv")
    objective_scores = _read_csv_if_exists(card / "objective_scores.csv")
    promotion_scores = _read_csv_if_exists(card / "promotion_scores.csv")
    artifact_manifest = _read_json_if_exists(card / "artifact_manifest.json")
    manifest_covers_core = _manifest_has_paths(
        card,
        artifact_manifest,
        (
            "analysis_summary",
            "analysis_manifest",
            "predictions",
            "daily_backtest",
            "model_comparison",
            "model_capabilities",
            "objective_scores",
            "promotion_scores",
        ),
    )

    rows: list[dict[str, object]] = []
    for model in expected:
        base_model = model.split("__", 1)[0]
        registered = True
        try:
            spec = get_model_spec(base_model)
        except KeyError:
            registered = False
            spec = None
        capability_row = _row_for_model(model_capabilities, model)
        if capability_row.empty and base_model != model:
            capability_row = _row_for_model(model_capabilities, base_model)
        if capability_row.empty and registered:
            capability_row = model_capability_table((base_model,)).iloc[0]
        prediction_group = _group_for_model(predictions, model)
        daily_group = _group_for_model(daily, model)
        regime_group = _group_for_model(regime_metrics, model)
        window_group = _group_for_model(window_metrics, model)
        comparison_row = _row_for_model(model_comparison, model)
        objective_row = _row_for_model(objective_scores, model)
        promotion_row = _row_for_model(promotion_scores, model)
        prediction_rows = int(len(prediction_group))
        backtest_rows = int(len(daily_group))
        regime_rows = int(len(regime_group))
        window_rows = int(len(window_group))
        predicted = prediction_rows > 0 and bool(prediction_group["prediction"].notna().all())
        backtested = backtest_rows > 0 and bool(daily_group["nav"].notna().all()) if "nav" in daily_group else False
        analysis_recorded = not comparison_row.empty and regime_rows > 0 and window_rows > 0
        adapter_status = (
            str(prediction_group["adapter_status"].dropna().iloc[0])
            if not prediction_group.empty and "adapter_status" in prediction_group
            else model_adapter_status(base_model)
            if registered
            else ""
        )
        adapter_name = (
            str(prediction_group["model_adapter"].dropna().iloc[0])
            if not prediction_group.empty and "model_adapter" in prediction_group
            else model_adapter_name(base_model)
            if registered
            else ""
        )
        rows.append(
            {
                "schema_version": MODEL_RUN_LEDGER_SCHEMA_VERSION,
                "model": model,
                "base_model": base_model,
                "family": str(_row_value(capability_row, "family", spec.family if spec else "")),
                "objective": str(_row_value(capability_row, "objective", spec.objective if spec else "")),
                "input_kind": str(_row_value(capability_row, "input_kind", spec.input_kind if spec else "")),
                "estimator": str(_row_value(capability_row, "estimator", spec.estimator if spec else "")),
                "paper_fidelity": str(
                    _row_value(capability_row, "paper_fidelity", spec.paper_fidelity if spec else "")
                ),
                "is_proxy": bool(_row_value(capability_row, "is_proxy", False)),
                "adapter_status": adapter_status,
                "model_adapter": adapter_name,
                "registered": registered,
                "capability_recorded": not _row_for_model(model_capabilities, model).empty,
                "trained": predicted,
                "predicted": predicted,
                "backtested": backtested,
                "objective_evaluated": not objective_row.empty,
                "promotion_evaluated": not promotion_row.empty,
                "analysis_recorded": analysis_recorded,
                "manifested": manifest_covers_core and predicted and backtested,
                "prediction_rows": prediction_rows,
                "backtest_rows": backtest_rows,
                "regime_rows": regime_rows,
                "window_rows": window_rows,
                "objective_row_count": _int_value(objective_row, "row_count"),
                "comparison_rank": _int_value(comparison_row, "rank"),
                "return_pct": _float_value(comparison_row, "return_pct"),
                "benchmark_return_pct": _float_value(comparison_row, "benchmark_return_pct"),
                "relative_return_pct": _float_value(comparison_row, "relative_return_pct"),
                "max_drawdown_pct": _float_value(comparison_row, "max_drawdown_pct"),
                "sharpe": _float_value(comparison_row, "sharpe"),
                "turnover": _float_value(comparison_row, "turnover"),
                "stability_score": _float_value(comparison_row, "stability_score"),
                "promotion_rank": _int_value(promotion_row, "rank"),
                "promotion_score": _float_value(promotion_row, "promotion_score"),
                "promotion_decision": str(_row_value(promotion_row, "promotion_decision", "")),
                "promotion_blockers": str(_row_value(promotion_row, "promotion_blockers", "")),
                "promotion_blocker": str(
                    _row_value(capability_row, "promotion_blocker", spec.promotion_blocker if spec else "")
                ),
                "implementation_notes": str(
                    _row_value(capability_row, "implementation_notes", spec.implementation_notes if spec else "")
                ),
            }
        )

    ledger = pd.DataFrame(rows)
    required_flags = [
        "registered",
        "trained",
        "predicted",
        "backtested",
        "objective_evaluated",
        "promotion_evaluated",
        "analysis_recorded",
        "manifested",
    ]
    incomplete_mask = ~ledger.loc[:, required_flags].astype(bool).all(axis=1) if not ledger.empty else pd.Series()
    missing_models = ledger.loc[incomplete_mask, "model"].astype(str).tolist() if not ledger.empty else []
    paper_fidelity_counts = (
        ledger["paper_fidelity"].astype(str).value_counts().sort_index().astype(int).to_dict() if not ledger.empty else {}
    )
    adapter_statuses = sorted(ledger["adapter_status"].astype(str).dropna().unique().tolist()) if not ledger.empty else []
    summary = {
        "schema_version": MODEL_RUN_LEDGER_SUMMARY_SCHEMA_VERSION,
        "expected_model_count": int(len(expected)),
        "ledger_model_count": int(len(ledger)),
        "registered_model_count": int(ledger["registered"].astype(bool).sum()) if not ledger.empty else 0,
        "trained_model_count": int(ledger["trained"].astype(bool).sum()) if not ledger.empty else 0,
        "predicted_model_count": int(ledger["predicted"].astype(bool).sum()) if not ledger.empty else 0,
        "backtested_model_count": int(ledger["backtested"].astype(bool).sum()) if not ledger.empty else 0,
        "objective_evaluated_model_count": int(ledger["objective_evaluated"].astype(bool).sum())
        if not ledger.empty
        else 0,
        "promotion_evaluated_model_count": int(ledger["promotion_evaluated"].astype(bool).sum())
        if not ledger.empty
        else 0,
        "analysis_recorded_model_count": int(ledger["analysis_recorded"].astype(bool).sum()) if not ledger.empty else 0,
        "manifested_model_count": int(ledger["manifested"].astype(bool).sum()) if not ledger.empty else 0,
        "proxy_model_count": int(ledger["is_proxy"].astype(bool).sum()) if not ledger.empty else 0,
        "native_model_count": int((~ledger["is_proxy"].astype(bool)).sum()) if not ledger.empty else 0,
        "paper_fidelity_counts": paper_fidelity_counts,
        "adapter_statuses": adapter_statuses,
        "missing_models": missing_models,
        "all_models_completed": not missing_models and int(len(ledger)) == int(len(expected)),
    }
    return ledger, summary


def write_model_run_ledger(
    card_dir: Path,
    *,
    expected_model_names: Iterable[str] = QUANTBENCH_MODEL_NAMES,
) -> tuple[pd.DataFrame, dict[str, object]]:
    card = Path(card_dir)
    ledger, summary = build_model_run_ledger(card, expected_model_names=expected_model_names)
    ledger.to_csv(card / "model_run_ledger.csv", index=False)
    (card / "model_run_ledger_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ledger, summary


def _adapter_summary(predictions: pd.DataFrame) -> dict[str, object]:
    statuses = sorted(str(value) for value in predictions["adapter_status"].dropna().unique().tolist())
    adapters = sorted(str(value) for value in predictions["model_adapter"].dropna().unique().tolist())
    return {
        "adapter_status": statuses[0] if len(statuses) == 1 else "mixed_adapters",
        "adapter_statuses": statuses,
        "model_adapters": adapters,
    }


def _analysis_windows_for_test(test: pd.DataFrame) -> dict[str, tuple[str, str]]:
    test_start = str(test["date"].min().date())
    test_end = str(test["date"].max().date())
    return build_analysis_windows(
        start=test_start,
        end=test_end,
        custom_windows={"test": (test_start, test_end)},
    )


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
    windows = _analysis_windows_for_test(test)
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
            "model_regime_window_diagnostics",
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
            "model_run_ledger": "model_run_ledger.csv",
            "model_run_ledger_summary": "model_run_ledger_summary.json",
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
            "model_regime_window_diagnostics": "model_regime_window_diagnostics.csv",
            "backtest_config": "backtest_config.json",
            "walk_forward_splits": "walk_forward_splits.csv",
            "objective_scores": "objective_scores.csv",
            "promotion_scores": "promotion_scores.csv",
            "model_run_ledger": "model_run_ledger.csv",
            "model_run_ledger_summary": "model_run_ledger_summary.json",
            "analysis_manifest": "analysis_manifest.json",
        },
        metadata=analysis_metadata,
    )
    model_run_ledger, _ = write_model_run_ledger(card, expected_model_names=selected_model_names)
    return QuantBenchMatrixOutputs(
        predictions=predictions,
        daily=daily,
        model_run_ledger=model_run_ledger,
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
    windows = _analysis_windows_for_test(test)
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
            "model_regime_window_diagnostics",
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
            "model_run_ledger": "model_run_ledger.csv",
            "model_run_ledger_summary": "model_run_ledger_summary.json",
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
            "model_regime_window_diagnostics": "model_regime_window_diagnostics.csv",
            "backtest_config": "backtest_config.json",
            "walk_forward_splits": "walk_forward_splits.csv",
            "objective_scores": "objective_scores.csv",
            "promotion_scores": "promotion_scores.csv",
            "model_run_ledger": "model_run_ledger.csv",
            "model_run_ledger_summary": "model_run_ledger_summary.json",
            "analysis_manifest": "analysis_manifest.json",
            "factor_ic": "factor_ic.csv",
            "feature_set_specs": "feature_set_specs.json",
        },
        metadata=analysis_metadata,
    )
    model_run_ledger, _ = write_model_run_ledger(
        card,
        expected_model_names=tuple(str(value) for value in sorted(predictions["model"].unique())),
    )
    return QuantBenchMatrixOutputs(
        predictions=predictions,
        daily=daily,
        model_run_ledger=model_run_ledger,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
        window_regime_metrics=window_regime_metrics,
        model_comparison=model_comparison,
        qlib_risk_metrics=effect_analysis.qlib_risk_metrics,
        model_effect_deltas=effect_analysis.model_effect_deltas,
        summary=summary,
    )

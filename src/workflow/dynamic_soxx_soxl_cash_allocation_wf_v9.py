from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import ETF_RETURN_COL, UNDERLYING_RETURN_COL
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v2 import _return_drawdown
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v6 import (
    DynamicSoxxSoxlCashAllocationWfV6Spec,
    _benchmark_metrics,
    _load_model_frame,
    _source_causal_feature_cols,
)
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v7 import _build_predictions_by_model
from src.workflow.multi_3x_etf_normalized_nested_wf import _json_ready, _resolve_root
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v9"
CARD_TYPE = "dynamic_soxx_soxl_cash_high_confidence_precision_frontier_card"
SOURCE_EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v1"
SOXX_EXCESS_FLOOR_PCT = 20.0
MAX_DRAWDOWN_FLOOR_PCT = -10.0
MODEL_NAMES = ("xgb", "lgbm", "avg_xgb_lgb")


@dataclass(frozen=True)
class DynamicSoxxSoxlCashAllocationWfV9Spec:
    root: Path
    experiment: str = EXPERIMENT
    prediction_years: tuple[int, ...] = tuple(range(2019, 2027))
    test_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    validation_window_years: int = 4
    cost_per_turnover: float = 0.0003

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / SOURCE_EXPERIMENT

    @property
    def v6_spec(self) -> DynamicSoxxSoxlCashAllocationWfV6Spec:
        return DynamicSoxxSoxlCashAllocationWfV6Spec(root=self.root)

    @property
    def daily_provider_uri(self) -> Path:
        return self.v6_spec.daily_provider_uri

    @property
    def intraday_15m_provider_uri(self) -> Path:
        return self.v6_spec.intraday_15m_provider_uri


def run_dynamic_soxx_soxl_cash_allocation_wf_v9_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfV9Spec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashAllocationWfV9Spec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    common_end = str(source_summary["common_data_end"])
    frame, feature_cols, feature_audit = _load_model_frame(
        spec.v6_spec,
        common_end=common_end,
        source_feature_cols=_source_causal_feature_cols(source_summary),
    )
    predictions_by_model = _build_predictions_by_model(frame, feature_cols=feature_cols, spec=spec)
    benchmark = _benchmark_metrics(predictions_by_model["xgb"].loc[predictions_by_model["xgb"]["year"].isin(spec.test_years)].copy())
    capacity = _high_confidence_capacity(predictions_by_model, spec=spec, benchmark=benchmark)
    nested = _nested_walk_forward(predictions_by_model, spec=spec, benchmark=benchmark)
    precision_frontier = _precision_recall_frontier(predictions_by_model["xgb"], spec=spec, benchmark=benchmark)
    label_learnability = _label_learnability(predictions_by_model, spec=spec)
    feature_family_ic = _feature_family_ic(frame, feature_cols=feature_cols, spec=spec)
    oracle_boundary = _oracle_boundary(
        predictions_by_model["xgb"],
        benchmark=benchmark,
        precision_frontier=precision_frontier,
        label_learnability=label_learnability,
    )
    capacity_summary = _capacity_summary(capacity)
    nested_summary = _nested_summary(nested, predictions_by_model=predictions_by_model, spec=spec, benchmark=benchmark)
    promotion_decision = "not_promoted_high_confidence_capacity_pass_failed_validation"

    sidecars = {
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
        "high_confidence_capacity": "high_confidence_capacity.csv",
        "model_comparison": "model_comparison.csv",
        "nested_walk_forward": "nested_walk_forward.csv",
        "precision_recall_frontier": "precision_recall_frontier.csv",
        "risk_analysis": "risk_analysis.csv",
        "selected_model": "selected_model.json",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        capacity_path = tmp_dir / "high_confidence_capacity.csv"
        nested_path = tmp_dir / "nested_walk_forward.csv"
        precision_path = tmp_dir / "precision_recall_frontier.csv"
        capacity.to_csv(capacity_path, index=False)
        nested.to_csv(nested_path, index=False)
        precision_frontier.to_csv(precision_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                daily_provider_uri=spec.daily_provider_uri.resolve().as_posix(),
                intraday_15m_provider_uri=spec.intraday_15m_provider_uri.resolve().as_posix(),
                benchmark="SOXX",
                benchmark_basis="open_to_open",
                allocation_assets="SOXX,SOXL,cash",
                frontier_policy="high_confidence_lgb_xgb_soxl_cash",
                model_families="xgb,lgbm,avg_xgb_lgb",
                objective="SOXX plus 20 percentage points with max drawdown above -10%",
            )
            R.log_metrics(
                soxx_return_pct=float(benchmark["return_pct"]),
                soxx_max_drawdown_pct=float(benchmark["max_drawdown_pct"]),
                capacity_passes_target_count=float(capacity_summary["passes_target_count"]),
                capacity_best_pass_return_pct=float(capacity_summary["best_pass_return_pct"]),
                nested_stitched_return_pct=float(nested_summary["stitched_return_pct"]),
                nested_stitched_max_drawdown_pct=float(nested_summary["stitched_max_drawdown_pct"]),
                oracle_best_daily_return_pct=float(oracle_boundary["oracle_best_daily_return_pct"]),
                walk_forward_best_label_auc=float(oracle_boundary["walk_forward_best_label_auc"]),
            )
            recorder.log_artifact(capacity_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(nested_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(precision_path.as_posix(), artifact_path=spec.experiment)
            summary = _summary_payload(
                spec=spec,
                common_end=common_end,
                recorder_id=recorder.id,
                recorder_uri=recorder_uri,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                sidecars=sidecars,
                source_summary=source_summary,
                feature_audit=feature_audit,
                benchmark=benchmark,
                capacity_summary=capacity_summary,
                nested_summary=nested_summary,
                oracle_boundary=oracle_boundary,
                feature_family_ic=feature_family_ic,
                label_learnability=label_learnability,
                promotion_decision=promotion_decision,
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                capacity=capacity,
                nested=nested,
                precision_frontier=precision_frontier,
            )
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.suffix != ".md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            recorder_dir = Path(recorder.get_local_dir()).resolve()
            _remove_qlib_code_snapshot_artifacts(recorder_dir)
            _remove_noisy_recorder_params(recorder_dir)
    _write_summary_files(card, summary)
    return summary


def _remove_noisy_recorder_params(recorder_dir: Path) -> None:
    # nbconvert injects a temporary kernel connection path into sys.argv.
    # It is not research evidence and makes the recorder non-portable.
    noisy = recorder_dir / "params" / "cmd-sys.argv"
    if noisy.exists():
        noisy.unlink()


def _candidate_params() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for score in ["pup", "p_soxl"]:
            for q in [0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]:
                rows.append({"model": model, "score": score, "q": float(q), "fallback": "cash", "soxl_w": 1.0})
    return rows


def _high_confidence_capacity(
    predictions_by_model: dict[str, pd.DataFrame],
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV9Spec,
    benchmark: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for params in _candidate_params():
        pred = predictions_by_model[str(params["model"])]
        test = pred.loc[pred["year"].isin(spec.test_years)].copy()
        result = _simulate_high_confidence_policy(test, params=params, threshold=None, cost_per_turnover=spec.cost_per_turnover)
        rows.append(
            {
                **params,
                "threshold_source": "outer_test_quantile_capacity_diagnostic_not_promotable",
                "threshold": result["threshold"],
                **_metric_payload(result, benchmark_return=float(benchmark["return_pct"])),
                "precision": result["precision"],
                "recall": result["recall"],
                "false_positive_count": result["false_positive_count"],
                "selected_count": result["selected_count"],
                "selection_uses_outer_test": True,
            }
        )
    return pd.DataFrame(rows).sort_values(["passes_target", "return_pct"], ascending=[False, False]).reset_index(drop=True)


def _nested_walk_forward(
    predictions_by_model: dict[str, pd.DataFrame],
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV9Spec,
    benchmark: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outer_year in spec.test_years:
        scored: list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for params in _candidate_params():
            pred = predictions_by_model[str(params["model"])]
            valid = pred.loc[
                pred["year"].ge(int(outer_year) - int(spec.validation_window_years))
                & pred["year"].lt(int(outer_year))
            ].copy()
            test = pred.loc[pred["year"].eq(int(outer_year))].copy()
            if valid.empty or test.empty:
                continue
            valid_result = _simulate_high_confidence_policy(
                valid,
                params=params,
                threshold=None,
                cost_per_turnover=spec.cost_per_turnover,
            )
            test_result = _simulate_high_confidence_policy(
                test,
                params=params,
                threshold=float(valid_result["threshold"]),
                cost_per_turnover=spec.cost_per_turnover,
            )
            score = float(valid_result["return_pct"]) - 100.0 * max(0.0, MAX_DRAWDOWN_FLOOR_PCT - float(valid_result["max_drawdown_pct"]))
            scored.append((score, params, valid_result, test_result))
        if not scored:
            continue
        score, selected, valid_result, test_result = sorted(scored, key=lambda item: item[0], reverse=True)[0]
        annual_benchmark, _ = _return_drawdown(
            pd.to_numeric(
                predictions_by_model[str(selected["model"])].loc[
                    predictions_by_model[str(selected["model"])]["year"].eq(int(outer_year)),
                    UNDERLYING_RETURN_COL,
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(dtype="float64")
        )
        rows.append(
            {
                "outer_year": int(outer_year),
                **selected,
                "validation_threshold": float(valid_result["threshold"]),
                "validation_return_pct": float(valid_result["return_pct"]),
                "validation_max_drawdown_pct": float(valid_result["max_drawdown_pct"]),
                "validation_precision": float(valid_result["precision"]),
                "validation_recall": float(valid_result["recall"]),
                "validation_false_positive_count": int(valid_result["false_positive_count"]),
                "test_return_pct": float(test_result["return_pct"]),
                "test_max_drawdown_pct": float(test_result["max_drawdown_pct"]),
                "test_precision": float(test_result["precision"]),
                "test_recall": float(test_result["recall"]),
                "test_false_positive_count": int(test_result["false_positive_count"]),
                "test_selected_count": int(test_result["selected_count"]),
                "test_soxx_return_pct": float(annual_benchmark),
                "test_excess_vs_annual_soxx_pct": float(test_result["return_pct"] - annual_benchmark),
                "selection_uses_outer_test": False,
                "selection_score": float(score),
            }
        )
    return pd.DataFrame(rows)


def _simulate_high_confidence_policy(
    frame: pd.DataFrame,
    *,
    params: dict[str, Any],
    threshold: float | None,
    cost_per_turnover: float,
) -> dict[str, Any]:
    work = frame.copy().sort_values("datetime").reset_index(drop=True)
    score = pd.to_numeric(work[str(params["score"])], errors="coerce").to_numpy(dtype="float64")
    selected_threshold = float(np.nanquantile(score, float(params["q"]))) if threshold is None else float(threshold)
    active = score >= selected_threshold
    soxl = pd.to_numeric(work[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxx = pd.to_numeric(work[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxl_weight = active.astype("float64") * float(params.get("soxl_w", 1.0))
    soxx_weight = np.zeros(len(work), dtype="float64")
    if params.get("fallback") == "soxx":
        soxx_weight = (~active).astype("float64")
    turnover = np.abs(soxx_weight - np.r_[0.0, soxx_weight[:-1]]) + np.abs(soxl_weight - np.r_[0.0, soxl_weight[:-1]])
    returns = soxx_weight * soxx + soxl_weight * soxl - turnover * float(cost_per_turnover)
    ret, maxdd = _return_drawdown(returns)
    positive = soxl > 0.0
    true_positive = int((active & positive).sum())
    false_positive = int((active & ~positive).sum())
    selected_count = int(active.sum())
    positive_count = int(positive.sum())
    return {
        "return_pct": ret,
        "max_drawdown_pct": maxdd,
        "returns": returns,
        "threshold": selected_threshold,
        "avg_soxx_weight": float(np.mean(soxx_weight)),
        "avg_soxl_weight": float(np.mean(soxl_weight)),
        "active_share": float(np.mean(active)),
        "precision": float(true_positive / selected_count) if selected_count else 0.0,
        "recall": float(true_positive / positive_count) if positive_count else 0.0,
        "false_positive_count": false_positive,
        "selected_count": selected_count,
    }


def _precision_recall_frontier(
    pred: pd.DataFrame,
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV9Spec,
    benchmark: dict[str, Any],
) -> pd.DataFrame:
    test = pred.loc[pred["year"].isin(spec.test_years)].copy().sort_values("datetime").reset_index(drop=True)
    soxl = pd.to_numeric(test[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    positive = soxl > 0.0
    target = float(benchmark["return_pct"]) + SOXX_EXCESS_FLOOR_PCT
    rng = np.random.default_rng(20260707)
    rows: list[dict[str, Any]] = []
    for tpr in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        for fpr in [0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]:
            metrics = []
            for _ in range(100):
                active = np.zeros(len(test), dtype=bool)
                active[positive] = rng.random(int(positive.sum())) < float(tpr)
                active[~positive] = rng.random(int((~positive).sum())) < float(fpr)
                soxl_weight = active.astype("float64")
                turnover = np.abs(soxl_weight - np.r_[0.0, soxl_weight[:-1]])
                returns = soxl_weight * soxl - turnover * float(spec.cost_per_turnover)
                ret, maxdd = _return_drawdown(returns)
                true_positive = int((active & positive).sum())
                false_positive = int((active & ~positive).sum())
                selected_count = int(active.sum())
                precision = true_positive / selected_count if selected_count else 0.0
                recall = true_positive / int(positive.sum()) if positive.sum() else 0.0
                metrics.append((ret, maxdd, precision, recall, selected_count, false_positive))
            arr = np.asarray(metrics, dtype="float64")
            rows.append(
                {
                    "label": "soxl_positive",
                    "fallback": "cash",
                    "tpr": float(tpr),
                    "fpr": float(fpr),
                    "pass_rate": float(((arr[:, 0] >= target) & (arr[:, 1] >= MAX_DRAWDOWN_FLOOR_PCT)).mean()),
                    "return_median_pct": float(np.median(arr[:, 0])),
                    "max_drawdown_median_pct": float(np.median(arr[:, 1])),
                    "precision_median": float(np.median(arr[:, 2])),
                    "recall_median": float(np.median(arr[:, 3])),
                    "selected_count_median": float(np.median(arr[:, 4])),
                    "false_positive_count_median": float(np.median(arr[:, 5])),
                }
            )
    return pd.DataFrame(rows)


def _label_learnability(
    predictions_by_model: dict[str, pd.DataFrame],
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV9Spec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, pred in predictions_by_model.items():
        test = pred.loc[pred["year"].isin(spec.test_years)].copy()
        labels = {
            "soxl_positive": pd.to_numeric(test[ETF_RETURN_COL], errors="coerce").fillna(0.0).gt(0.0).astype(int),
            "oracle_best_is_soxl": (
                pd.to_numeric(test[ETF_RETURN_COL], errors="coerce").fillna(0.0)
                .gt(pd.to_numeric(test[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0))
                & pd.to_numeric(test[ETF_RETURN_COL], errors="coerce").fillna(0.0).gt(0.0)
            ).astype(int),
        }
        scores = {"pup": test["pup"], "p_soxl": test["p_soxl"]} if "p_soxl" in test else {"pup": test["pup"]}
        for label_name, label in labels.items():
            for score_name, score in scores.items():
                rows.append(
                    {
                        "model": model,
                        "label": label_name,
                        "score": score_name,
                        "positive_rate": float(label.mean()),
                        "auc": _safe_auc(label, pd.to_numeric(score, errors="coerce").fillna(0.0)),
                    }
                )
    return pd.DataFrame(rows)


def _feature_family_ic(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    spec: DynamicSoxxSoxlCashAllocationWfV9Spec,
) -> dict[str, dict[str, float]]:
    test = frame.loc[frame["datetime"].dt.year.isin(spec.test_years)].copy().reset_index(drop=True)
    soxl = pd.to_numeric(test[ETF_RETURN_COL], errors="coerce").fillna(0.0).reset_index(drop=True)
    targets = [
        soxl,
        soxl.gt(0.0).astype(int),
        soxl.le(-0.03).astype(int),
    ]
    rows: list[dict[str, Any]] = []
    for column in feature_cols:
        if column not in test:
            continue
        values = pd.to_numeric(test[column], errors="coerce").reset_index(drop=True)
        if values.notna().sum() < 100 or values.nunique(dropna=True) < 3:
            continue
        family = _feature_family(column)
        correlations = []
        for target in targets:
            corr = values.corr(target, method="spearman")
            if np.isfinite(corr):
                correlations.append(abs(float(corr)))
        if correlations:
            rows.append(
                {
                    "family": family,
                    "feature": column,
                    "max_abs_ic": max(correlations),
                    "mean_abs_ic": float(np.mean(correlations)),
                }
            )
    if not rows:
        return {}
    ic = pd.DataFrame(rows)
    out: dict[str, dict[str, float]] = {}
    for family, group in ic.groupby("family"):
        out[str(family)] = {
            "feature_count": int(len(group)),
            "max_abs_ic": float(group["max_abs_ic"].max()),
            "mean_abs_ic": float(group["mean_abs_ic"].mean()),
            "top10_mean_abs_ic": float(group["max_abs_ic"].sort_values(ascending=False).head(10).mean()),
        }
    return out


def _feature_family(column: str) -> str:
    lower = column.lower()
    if "const" in lower or "breadth" in lower or "downside" in lower:
        return "constituent_breadth"
    if column.startswith("i15_same"):
        return "same_day_15m"
    if "vix" in lower:
        return "vix"
    if "tracking_error" in lower or "range_ratio" in lower:
        return "leveraged_etf_microstructure"
    if "ret_" in lower or "ma_ratio" in lower or "drawdown" in lower or "realized_vol" in lower:
        return "price_trend_vol"
    return "other"


def _oracle_boundary(
    pred: pd.DataFrame,
    *,
    benchmark: dict[str, Any],
    precision_frontier: pd.DataFrame,
    label_learnability: pd.DataFrame,
) -> dict[str, Any]:
    test = pred.loc[pred["year"].isin((2023, 2024, 2025, 2026))].copy().sort_values("datetime").reset_index(drop=True)
    soxl = pd.to_numeric(test[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxx = pd.to_numeric(test[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    best_action_soxl = (soxl > soxx) & (soxl > 0.0)
    best_action_soxx = (soxx > 0.0) & ~best_action_soxl
    soxl_weight = best_action_soxl.astype("float64")
    soxx_weight = best_action_soxx.astype("float64")
    turnover = np.abs(soxx_weight - np.r_[0.0, soxx_weight[:-1]]) + np.abs(soxl_weight - np.r_[0.0, soxl_weight[:-1]])
    oracle_return, oracle_dd = _return_drawdown(soxx_weight * soxx + soxl_weight * soxl - turnover * 0.0003)
    pass_frontier = precision_frontier.loc[precision_frontier["pass_rate"].ge(0.5)]
    min_fpr = float(pass_frontier["fpr"].min()) if not pass_frontier.empty else float("nan")
    best_auc = float(label_learnability["auc"].max()) if not label_learnability.empty else float("nan")
    return {
        "diagnostic_only": True,
        "oracle_policy": "hindsight_best_daily_cash_soxx_soxl",
        "oracle_best_daily_return_pct": float(oracle_return),
        "oracle_best_daily_max_drawdown_pct": float(oracle_dd),
        "benchmark_return_pct": float(benchmark["return_pct"]),
        "target_return_pct": float(benchmark["return_pct"] + SOXX_EXCESS_FLOOR_PCT),
        "minimum_random_false_positive_rate_for_pass": min_fpr,
        "walk_forward_best_label_auc": best_auc,
        "interpretation": "target is feasible only with near-zero false-positive SOXL entries; current LGB/XGB scores do not show enough label learnability",
    }


def _capacity_summary(capacity: pd.DataFrame) -> dict[str, Any]:
    passes = capacity.loc[capacity["passes_target"]].copy()
    best_pass = passes.sort_values("return_pct", ascending=False).head(1)
    best_dd_safe = capacity.loc[capacity["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)].sort_values(
        "return_pct",
        ascending=False,
    ).head(1)
    return {
        "selection_uses_outer_test": True,
        "frontier_policy": "high_confidence_lgb_xgb_soxl_cash",
        "threshold_policy": "outer_test_quantile_capacity_diagnostic_not_promotable",
        "row_count": int(len(capacity)),
        "passes_target_count": int(capacity["passes_target"].sum()),
        "best_pass_model": _frame_value(best_pass, "model"),
        "best_pass_score": _frame_value(best_pass, "score"),
        "best_pass_q": _frame_float(best_pass, "q"),
        "best_pass_return_pct": _frame_float(best_pass, "return_pct"),
        "best_pass_max_drawdown_pct": _frame_float(best_pass, "max_drawdown_pct"),
        "best_pass_excess_vs_soxx_pct": _frame_float(best_pass, "excess_vs_soxx_pct"),
        "best_pass_precision": _frame_float(best_pass, "precision"),
        "best_pass_recall": _frame_float(best_pass, "recall"),
        "best_pass_false_positive_count": int(_frame_float(best_pass, "false_positive_count"))
        if not best_pass.empty
        else 0,
        "best_drawdown_safe_return_pct": _frame_float(best_dd_safe, "return_pct"),
        "best_drawdown_safe_max_drawdown_pct": _frame_float(best_dd_safe, "max_drawdown_pct"),
    }


def _nested_summary(
    nested: pd.DataFrame,
    *,
    predictions_by_model: dict[str, pd.DataFrame],
    spec: DynamicSoxxSoxlCashAllocationWfV9Spec,
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    returns: list[np.ndarray] = []
    for row in nested.to_dict("records"):
        test = predictions_by_model[str(row["model"])].loc[predictions_by_model[str(row["model"])]["year"].eq(int(row["outer_year"]))].copy()
        result = _simulate_high_confidence_policy(
            test,
            params={key: row[key] for key in ["model", "score", "q", "fallback", "soxl_w"]},
            threshold=float(row["validation_threshold"]),
            cost_per_turnover=spec.cost_per_turnover,
        )
        returns.append(result["returns"])
    stitched = np.concatenate(returns) if returns else np.asarray([], dtype="float64")
    ret, maxdd = _return_drawdown(stitched)
    excess = ret - float(benchmark["return_pct"])
    required = bool(excess >= SOXX_EXCESS_FLOOR_PCT and maxdd >= MAX_DRAWDOWN_FLOOR_PCT)
    return {
        "selection_uses_outer_test": False,
        "threshold_policy": "validation_quantile_threshold_applied_to_outer_year",
        "validation_window_years": int(spec.validation_window_years),
        "stitched_return_pct": float(ret),
        "stitched_max_drawdown_pct": float(maxdd),
        "stitched_excess_vs_soxx_pct": float(excess),
        "required_policy_satisfied": required,
        "annual_rows": int(len(nested)),
        "blocking_reasons": []
        if required
        else ["high_confidence_capacity_pass_does_not_validate_out_of_sample"],
    }


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV9Spec,
    common_end: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    sidecars: dict[str, str],
    source_summary: dict[str, Any],
    feature_audit: dict[str, Any],
    benchmark: dict[str, Any],
    capacity_summary: dict[str, Any],
    nested_summary: dict[str, Any],
    oracle_boundary: dict[str, Any],
    feature_family_ic: dict[str, dict[str, float]],
    label_learnability: pd.DataFrame,
    promotion_decision: str,
) -> dict[str, Any]:
    return _json_ready(
        {
            "experiment": spec.experiment,
            "card_type": CARD_TYPE,
            "promotion_decision": promotion_decision,
            "common_data_end": common_end,
            "source_experiment": SOURCE_EXPERIMENT,
            "provider_uri": spec.daily_provider_uri.resolve().as_posix(),
            "intraday_15m_provider_uri": spec.intraday_15m_provider_uri.resolve().as_posix(),
            "benchmark_asset": "SOXX",
            "benchmark_basis": "open_to_open",
            "allocation_assets": ["SOXX", "SOXL", "cash"],
            "execution_timing": "same_day_15m_close_t_signal_next_open_to_following_open",
            "goal": {
                "excess_vs_soxx_floor_pct": SOXX_EXCESS_FLOOR_PCT,
                "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
            },
            "benchmark": benchmark,
            "feature_audit": feature_audit,
            "high_confidence_capacity": capacity_summary,
            "validation_only_nested": nested_summary,
            "oracle_boundary": oracle_boundary,
            "feature_family_ic": feature_family_ic,
            "label_learnability_preview": label_learnability.to_dict("records"),
            "source_v1": {
                "promotion_decision": source_summary.get("promotion_decision"),
                "recorder_id": source_summary.get("recorder_id"),
                "common_data_end": source_summary.get("common_data_end"),
            },
            "qlib_docs": {
                "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
                "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
                "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            },
            "qlib_native_api": [
                "qlib.init",
                "qlib.data.D.features",
                "qlib.workflow.R.start",
                "qlib.workflow.R.log_params",
                "qlib.workflow.R.log_metrics",
                "qlib.workflow.Recorder.log_artifact",
                "XGBRegressor",
                "XGBClassifier",
                "LGBMRegressor",
                "LGBMClassifier",
            ],
            "sidecars": sidecars,
            "artifact_policy": {
                "policy": "slim_card_recorder_first",
                "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": [
                    f"artifacts/{spec.experiment}/high_confidence_capacity.csv",
                    f"artifacts/{spec.experiment}/nested_walk_forward.csv",
                    f"artifacts/{spec.experiment}/precision_recall_frontier.csv",
                ],
            },
            "workflow_config": "workflow.yaml",
            "recorder_id": recorder_id,
            "recorder_uri": recorder_uri.resolve().as_posix(),
            "recorder_local_dir": recorder_local_dir.as_posix(),
        }
    )


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    summary: dict[str, Any],
    capacity: pd.DataFrame,
    nested: pd.DataFrame,
    precision_frontier: pd.DataFrame,
) -> None:
    capacity.to_csv(card / sidecars["high_confidence_capacity"], index=False)
    nested.to_csv(card / sidecars["nested_walk_forward"], index=False)
    precision_frontier.to_csv(card / sidecars["precision_recall_frontier"], index=False)
    model_comparison = (
        capacity.groupby("model")
        .agg(
            capacity_passes_target_count=("passes_target", "sum"),
            best_return_pct=("return_pct", "max"),
            best_drawdown_safe_return_pct=(
                "return_pct",
                lambda s: float(
                    capacity.loc[s.index]
                    .loc[capacity.loc[s.index, "max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT), "return_pct"]
                    .max()
                ),
            ),
        )
        .reset_index()
    )
    nested_model = nested.groupby("model").agg(nested_year_count=("outer_year", "count")).reset_index()
    model_comparison = model_comparison.merge(nested_model, on="model", how="left")
    model_comparison.to_csv(card / sidecars["model_comparison"], index=False)
    analysis = {
        "source": "qlib_recorder_high_confidence_precision_frontier",
        "recorder_id": summary["recorder_id"],
        "high_confidence_capacity": summary["high_confidence_capacity"],
        "validation_only_nested": summary["validation_only_nested"],
        "oracle_boundary": summary["oracle_boundary"],
        "promotion_decision": summary["promotion_decision"],
    }
    (card / sidecars["analysis_summary"]).write_text(json.dumps(_json_ready(analysis), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "check": "high_confidence_capacity",
                "status": "capacity_pass_not_promotable",
                "details": json.dumps(summary["high_confidence_capacity"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "validation_only_nested",
                "status": "fail" if not summary["validation_only_nested"]["required_policy_satisfied"] else "pass",
                "details": json.dumps(summary["validation_only_nested"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "oracle_boundary",
                "status": "diagnostic_only",
                "details": json.dumps(summary["oracle_boundary"], ensure_ascii=False, sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    selected = {
        "selected_model": None,
        "promotion_decision": summary["promotion_decision"],
        "reason": "high-confidence XGB capacity can pass on the outer test window, but validation-selected thresholds do not generate enough return",
        "capacity": summary["high_confidence_capacity"],
        "validation_only_nested": summary["validation_only_nested"],
    }
    (card / sidecars["selected_model"]).write_text(json.dumps(_json_ready(selected), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / sidecars["failure_modes"]).write_text(_failure_modes_text(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion_text(summary), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")


def _failure_modes_text(summary: dict[str, Any]) -> str:
    cap = summary["high_confidence_capacity"]
    nested = summary["validation_only_nested"]
    oracle = summary["oracle_boundary"]
    return (
        "# Failure Modes\n\n"
        "- The outer-test high confidence frontier contains a pass, but the threshold is selected from the test window and is not promotable.\n"
        "- Validation-only threshold transfer keeps drawdown controlled but collapses participation and return.\n"
        f"- Best capacity pass is {cap['best_pass_model']} {cap['best_pass_score']} q={cap['best_pass_q']} with "
        f"{_fmt_pct(cap['best_pass_return_pct'])}% return and {_fmt_pct(cap['best_pass_max_drawdown_pct'])}% max drawdown.\n"
        f"- Nested stitched return is {_fmt_pct(nested['stitched_return_pct'])}% with "
        f"{_fmt_pct(nested['stitched_max_drawdown_pct'])}% max drawdown.\n"
        f"- Oracle return is {_fmt_pct(oracle['oracle_best_daily_return_pct'])}%, but current label AUC is only "
        f"{oracle['walk_forward_best_label_auc']:.4f}.\n"
    )


def _conclusion_text(summary: dict[str, Any]) -> str:
    cap = summary["high_confidence_capacity"]
    nested = summary["validation_only_nested"]
    oracle = summary["oracle_boundary"]
    return (
        "# 中文结论\n\n"
        "- 做了什么：新增 v9 high confidence precision frontier；用 LGB/XGB/平均模型筛 SOXL 少数高置信交易日，其他时间 cash。\n"
        f"- 容量结果：outer-test capacity passes={cap['passes_target_count']}；best={cap['best_pass_model']} "
        f"{cap['best_pass_score']} q={cap['best_pass_q']}，return={_fmt_pct(cap['best_pass_return_pct'])}%，"
        f"maxDD={_fmt_pct(cap['best_pass_max_drawdown_pct'])}%，precision={cap['best_pass_precision']:.4f}。\n"
        f"- nested 结果：validation-only threshold transfer return={_fmt_pct(nested['stitched_return_pct'])}%，"
        f"maxDD={_fmt_pct(nested['stitched_max_drawdown_pct'])}%，excess={_fmt_pct(nested['stitched_excess_vs_soxx_pct'])}%。\n"
        f"- oracle 诊断：事后最优能到 {_fmt_pct(oracle['oracle_best_daily_return_pct'])}%，但当前 walk-forward label AUC "
        f"最高只有 {oracle['walk_forward_best_label_auc']:.4f}，说明目标需要近乎零误买下跌日的 precision。\n"
        "- 结论：high confidence 路径给出了明确容量靶子，但泛化失败；不能晋升到实盘或 production。\n"
        f"- 是否晋升：不晋升；decision={summary['promotion_decision']}。\n"
        "- 下一步：继续优化应围绕提高正日筛选纯度，重点扩展成分股/期权/订单流或更长跨资产样本；单靠现有日线+15m 特征不足。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
        "- 不构成买卖建议。\n"
    )


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            f"intraday_15m_provider_uri: {summary['intraday_15m_provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            "benchmark: SOXX",
            "benchmark_basis: open_to_open",
            "allocation_assets: [SOXX, SOXL, cash]",
            "model_selection:",
            "  models: [xgb, lgbm, avg_xgb_lgb]",
            "  capacity_selection_uses_outer_test: true",
            "  nested_selection_uses_outer_test: false",
            "frontier:",
            "  policy: high_confidence_lgb_xgb_soxl_cash",
            "  threshold_policy: validation_quantile_threshold_applied_to_outer_year",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            f"promotion_decision: {summary['promotion_decision']}",
            "",
        ]
    )


def _metric_payload(result: dict[str, Any], *, benchmark_return: float) -> dict[str, Any]:
    excess = float(result["return_pct"]) - float(benchmark_return)
    return {
        "return_pct": float(result["return_pct"]),
        "max_drawdown_pct": float(result["max_drawdown_pct"]),
        "excess_vs_soxx_pct": excess,
        "passes_target": bool(excess >= SOXX_EXCESS_FLOOR_PCT and float(result["max_drawdown_pct"]) >= MAX_DRAWDOWN_FLOOR_PCT),
        "avg_soxx_weight": float(result["avg_soxx_weight"]),
        "avg_soxl_weight": float(result["avg_soxl_weight"]),
        "active_share": float(result["active_share"]),
    }


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    try:
        return float(roc_auc_score(y_true.astype(int), score.astype("float64")))
    except ValueError:
        return float("nan")


def _frame_float(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(frame[column].iloc[0])


def _frame_value(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    return str(frame[column].iloc[0])


def _fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    if abs(numeric) >= 1_000_000.0:
        return f"{numeric:.4e}"
    return f"{numeric:.4f}"

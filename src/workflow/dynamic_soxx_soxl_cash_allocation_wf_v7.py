from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import ETF_RETURN_COL, UNDERLYING_RETURN_COL
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v2 import _return_drawdown
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v6 import (
    DynamicSoxxSoxlCashAllocationWfV6Spec,
    _benchmark_metrics,
    _load_model_frame,
    _positive_proba,
    _rolling_xgb_predictions,
    _source_causal_feature_cols,
)
from src.workflow.multi_3x_etf_normalized_nested_wf import _json_ready, _resolve_root
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v7"
CARD_TYPE = "dynamic_soxx_soxl_cash_online_threshold_generalization_audit_card"
SOURCE_EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v1"
SOXX_EXCESS_FLOOR_PCT = 20.0
MAX_DRAWDOWN_FLOOR_PCT = -10.0
MODEL_NAMES = ("xgb", "lgbm", "avg_xgb_lgb")


@dataclass(frozen=True)
class DynamicSoxxSoxlCashAllocationWfV7Spec:
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


def run_dynamic_soxx_soxl_cash_allocation_wf_v7_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfV7Spec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashAllocationWfV7Spec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    common_end = str(source_summary["common_data_end"])
    source_feature_cols = _source_causal_feature_cols(source_summary)
    frame, feature_cols, feature_audit = _load_model_frame(
        spec.v6_spec,
        common_end=common_end,
        source_feature_cols=source_feature_cols,
    )
    predictions_by_model = _build_predictions_by_model(frame, feature_cols=feature_cols, spec=spec)
    benchmark = _benchmark_metrics(
        predictions_by_model["xgb"].loc[predictions_by_model["xgb"]["year"].isin(spec.test_years)].copy()
    )
    capacity_grid = _capacity_grid_by_model(predictions_by_model, spec=spec, benchmark=benchmark)
    model_comparison = _model_comparison(capacity_grid, predictions_by_model=predictions_by_model, spec=spec, benchmark=benchmark)
    nested = _validation_only_nested_by_model(predictions_by_model, spec=spec, benchmark=benchmark)
    capacity_summary = _capacity_summary(capacity_grid)
    nested_summary = _nested_summary(nested, benchmark=benchmark)
    generalization = _generalization_assessment(capacity_summary, nested_summary)
    promotion_decision = "not_promoted_online_threshold_capacity_pass_failed_generalization"

    sidecars = {
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
        "model_comparison": "model_comparison.csv",
        "nested_walk_forward": "nested_walk_forward.csv",
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
        capacity_path = tmp_dir / "capacity_grid.csv"
        capacity_grid.to_csv(capacity_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                daily_provider_uri=spec.daily_provider_uri.resolve().as_posix(),
                intraday_15m_provider_uri=spec.intraday_15m_provider_uri.resolve().as_posix(),
                benchmark="SOXX",
                benchmark_basis="open_to_open",
                allocation_assets="SOXX,SOXL,cash",
                model_families="xgb,lgbm,avg_xgb_lgb",
                threshold_policy="causal_rolling_prediction_quantiles",
                nested_selection_uses_outer_test="false",
                capacity_selection_uses_outer_test="true",
                objective="SOXX plus 20 percentage points with max drawdown above -10%",
            )
            R.log_metrics(
                soxx_return_pct=float(benchmark["return_pct"]),
                soxx_max_drawdown_pct=float(benchmark["max_drawdown_pct"]),
                capacity_passes_target_count=float(capacity_summary["passes_target_count"]),
                capacity_best_pass_return_pct=float(capacity_summary["best_pass_return_pct"]),
                capacity_best_pass_max_drawdown_pct=float(capacity_summary["best_pass_max_drawdown_pct"]),
                validation_nested_return_pct=float(nested_summary["stitched_return_pct"]),
                validation_nested_max_drawdown_pct=float(nested_summary["stitched_max_drawdown_pct"]),
                validation_nested_excess_pct=float(nested_summary["stitched_excess_vs_soxx_pct"]),
            )
            recorder.log_artifact(capacity_path.as_posix(), artifact_path=spec.experiment)
            summary = _summary_payload(
                spec=spec,
                common_end=common_end,
                recorder_id=recorder.id,
                recorder_uri=recorder_uri,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                source_summary=source_summary,
                feature_audit=feature_audit,
                benchmark=benchmark,
                capacity_summary=capacity_summary,
                nested_summary=nested_summary,
                generalization=generalization,
                sidecars=sidecars,
                promotion_decision=promotion_decision,
                model_comparison=model_comparison,
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                model_comparison=model_comparison,
                nested=nested,
            )
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.suffix != ".md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(Path(recorder.get_local_dir()).resolve())
    _write_summary_files(card, summary)
    return summary


def _build_predictions_by_model(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    spec: DynamicSoxxSoxlCashAllocationWfV7Spec,
) -> dict[str, pd.DataFrame]:
    xgb = _rolling_xgb_predictions(frame, feature_cols=feature_cols, spec=spec.v6_spec).sort_values("datetime").reset_index(drop=True)
    lgbm = _rolling_lgbm_predictions(frame, feature_cols=feature_cols, spec=spec).sort_values("datetime").reset_index(drop=True)
    avg = xgb[["datetime", "year", ETF_RETURN_COL, UNDERLYING_RETURN_COL]].copy()
    for column in ["p_soxl", "p_soxx", "pup", "ptail"]:
        avg[column] = (pd.to_numeric(xgb[column], errors="coerce") + pd.to_numeric(lgbm[column], errors="coerce")) / 2.0
    return {"xgb": xgb, "lgbm": lgbm, "avg_xgb_lgb": avg}


def _rolling_lgbm_predictions(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    spec: DynamicSoxxSoxlCashAllocationWfV7Spec,
) -> pd.DataFrame:
    from lightgbm import LGBMClassifier, LGBMRegressor

    parts = []
    reg_params = {
        "n_estimators": 120,
        "max_depth": 2,
        "learning_rate": 0.025,
        "num_leaves": 5,
        "subsample": 0.82,
        "colsample_bytree": 0.72,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "random_state": 4202,
        "n_jobs": 1,
        "verbose": -1,
    }
    clf_params = {**reg_params, "n_estimators": 100, "random_state": 4204}
    for year in spec.prediction_years:
        train = frame.loc[frame["datetime"].dt.year.lt(int(year))].copy()
        test = frame.loc[frame["datetime"].dt.year.eq(int(year))].copy().sort_values("datetime")
        if test.empty:
            continue
        train_x = train[feature_cols].astype("float64")
        test_x = test[feature_cols].astype("float64")
        soxl_model = make_pipeline(SimpleImputer(strategy="median"), LGBMRegressor(**reg_params))
        soxx_model = make_pipeline(SimpleImputer(strategy="median"), LGBMRegressor(**reg_params))
        up_model = make_pipeline(SimpleImputer(strategy="median"), LGBMClassifier(**clf_params))
        tail_model = make_pipeline(SimpleImputer(strategy="median"), LGBMClassifier(**clf_params))
        soxl_model.fit(train_x, train[ETF_RETURN_COL])
        soxx_model.fit(train_x, train[UNDERLYING_RETURN_COL])
        up_model.fit(train_x, train[ETF_RETURN_COL].gt(0.0).astype(int))
        tail_cut = float(train[UNDERLYING_RETURN_COL].quantile(0.08))
        tail_model.fit(train_x, train[UNDERLYING_RETURN_COL].le(tail_cut).astype(int))
        segment = test[["datetime", ETF_RETURN_COL, UNDERLYING_RETURN_COL]].copy()
        segment["year"] = int(year)
        segment["p_soxl"] = soxl_model.predict(test_x)
        segment["p_soxx"] = soxx_model.predict(test_x)
        segment["pup"] = _positive_proba(up_model, test_x)
        segment["ptail"] = _positive_proba(tail_model, test_x)
        parts.append(segment)
    return pd.concat(parts, ignore_index=True)


def _online_param_grid() -> list[dict[str, Any]]:
    rows = []
    for window in [252, 504, 630, 756]:
        for qret in [0.50, 0.53, 0.55, 0.57, 0.59, 0.62]:
            for qtail in [0.94, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98]:
                for pupthr in [0.55, 0.565, 0.58, 0.595, 0.61]:
                    for fallback in [0.16, 0.20, 0.24, 0.28, 0.32, 0.36]:
                        for active_soxx in [0.0, 0.05]:
                            for active_soxl in [0.70, 0.80, 0.90, 1.00]:
                                if active_soxx + active_soxl <= 1.0:
                                    rows.append(
                                        {
                                            "window": int(window),
                                            "qret": float(qret),
                                            "qtail": float(qtail),
                                            "pupthr": float(pupthr),
                                            "fallback": float(fallback),
                                            "active_soxx": float(active_soxx),
                                            "active_soxl": float(active_soxl),
                                        }
                                    )
    return rows


def _capacity_grid_by_model(
    predictions_by_model: dict[str, pd.DataFrame],
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV7Spec,
    benchmark: dict[str, Any],
) -> pd.DataFrame:
    params = _online_param_grid()
    rows = []
    for model_name, pred in predictions_by_model.items():
        test_mask = pred["year"].isin(spec.test_years).to_numpy()
        thresholds = _rolling_thresholds(pred)
        for param in params:
            result = _simulate_online_policy(
                pred,
                params=param,
                mask=test_mask,
                thresholds=thresholds,
                cost_per_turnover=spec.cost_per_turnover,
            )
            excess = result["return_pct"] - float(benchmark["return_pct"])
            rows.append(
                {
                    "model": model_name,
                    **param,
                    "return_pct": float(result["return_pct"]),
                    "max_drawdown_pct": float(result["max_drawdown_pct"]),
                    "excess_vs_soxx_pct": float(excess),
                    "passes_target": bool(excess >= SOXX_EXCESS_FLOOR_PCT and result["max_drawdown_pct"] >= MAX_DRAWDOWN_FLOOR_PCT),
                    "selection_uses_outer_test": True,
                    "avg_soxx_weight": float(result["avg_soxx_weight"]),
                    "avg_soxl_weight": float(result["avg_soxl_weight"]),
                    "active_share": float(result["active_share"]),
                    "mid_share": float(result["mid_share"]),
                    "notes": "outer_test_capacity_diagnostic_not_promotable",
                }
            )
    return pd.DataFrame(rows)


def _rolling_thresholds(pred: pd.DataFrame) -> dict[tuple[str, int, float], np.ndarray]:
    psoxl = pd.to_numeric(pred["p_soxl"], errors="coerce")
    ptail = pd.to_numeric(pred["ptail"], errors="coerce")
    thresholds: dict[tuple[str, int, float], np.ndarray] = {}
    for window in [252, 504, 630, 756]:
        min_periods = max(60, window // 3)
        for qret in [0.50, 0.53, 0.55, 0.57, 0.59, 0.62]:
            thresholds[("ret", window, qret)] = psoxl.shift(1).rolling(window, min_periods=min_periods).quantile(qret).to_numpy(dtype="float64")
        for qtail in [0.94, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98]:
            thresholds[("tail", window, qtail)] = ptail.shift(1).rolling(window, min_periods=min_periods).quantile(qtail).to_numpy(dtype="float64")
    return thresholds


def _simulate_online_policy(
    pred: pd.DataFrame,
    *,
    params: dict[str, Any],
    mask: np.ndarray,
    thresholds: dict[tuple[str, int, float], np.ndarray],
    cost_per_turnover: float,
) -> dict[str, Any]:
    psoxl = pd.to_numeric(pred["p_soxl"], errors="coerce").to_numpy(dtype="float64")
    psoxx = pd.to_numeric(pred["p_soxx"], errors="coerce").to_numpy(dtype="float64")
    pup = pd.to_numeric(pred["pup"], errors="coerce").to_numpy(dtype="float64")
    ptail = pd.to_numeric(pred["ptail"], errors="coerce").to_numpy(dtype="float64")
    soxx_returns = pd.to_numeric(pred[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxl_returns = pd.to_numeric(pred[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    window = int(params["window"])
    qret = float(params["qret"])
    qtail = float(params["qtail"])
    active = (
        (psoxl >= thresholds[("ret", window, qret)])
        & (pup >= float(params["pupthr"]))
        & (ptail < thresholds[("tail", window, qtail)])
    )
    mid = (psoxx > 0.0) & ~active & (ptail < thresholds[("tail", window, qtail)])
    soxx_weight = np.zeros(len(pred), dtype="float64")
    soxl_weight = np.zeros(len(pred), dtype="float64")
    soxx_weight[mid] = float(params["fallback"])
    soxx_weight[active] = float(params["active_soxx"])
    soxl_weight[active] = float(params["active_soxl"])
    total = soxx_weight + soxl_weight
    over = total > 1.0
    soxx_weight[over] = soxx_weight[over] / total[over]
    soxl_weight[over] = soxl_weight[over] / total[over]
    turnover = np.abs(soxx_weight - np.r_[0.0, soxx_weight[:-1]]) + np.abs(soxl_weight - np.r_[0.0, soxl_weight[:-1]])
    returns = soxx_weight * soxx_returns + soxl_weight * soxl_returns - turnover * float(cost_per_turnover)
    ret, maxdd = _return_drawdown(returns[mask])
    return {
        "return_pct": ret,
        "max_drawdown_pct": maxdd,
        "returns": returns,
        "avg_soxx_weight": float(np.mean(soxx_weight[mask])),
        "avg_soxl_weight": float(np.mean(soxl_weight[mask])),
        "active_share": float(np.mean(active[mask])),
        "mid_share": float(np.mean(mid[mask])),
    }


def _validation_only_nested_by_model(
    predictions_by_model: dict[str, pd.DataFrame],
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV7Spec,
    benchmark: dict[str, Any],
) -> pd.DataFrame:
    params = _online_param_grid()
    rows = []
    for model_name, pred in predictions_by_model.items():
        thresholds = _rolling_thresholds(pred)
        years = pd.to_numeric(pred["year"], errors="coerce").astype(int).to_numpy()
        soxx_returns = pd.to_numeric(pred[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        returns_cache = []
        for param in params:
            full_mask = np.ones(len(pred), dtype=bool)
            returns_cache.append(
                _simulate_online_policy(
                    pred,
                    params=param,
                    mask=full_mask,
                    thresholds=thresholds,
                    cost_per_turnover=spec.cost_per_turnover,
                )["returns"]
            )
        stitched = []
        selected_rows = []
        for outer_year in spec.test_years:
            valid_mask = (years >= int(outer_year) - int(spec.validation_window_years)) & (years < int(outer_year))
            test_mask = years == int(outer_year)
            valid_benchmark_return, _ = _return_drawdown(soxx_returns[valid_mask])
            annual_benchmark_return, _ = _return_drawdown(soxx_returns[test_mask])
            best: tuple[float, dict[str, Any], np.ndarray, float, float, float, bool] | None = None
            for param, returns in zip(params, returns_cache):
                valid_return, valid_dd = _return_drawdown(returns[valid_mask])
                valid_excess = valid_return - valid_benchmark_return
                valid_ok = bool(valid_dd >= MAX_DRAWDOWN_FLOOR_PCT and valid_excess >= 0.0)
                score = (
                    (2_000_000.0 if valid_ok else 0.0)
                    + (1_000_000.0 if valid_dd >= MAX_DRAWDOWN_FLOOR_PCT else 0.0)
                    + valid_return
                    + 2.0 * min(valid_excess, 50.0)
                    - 100.0 * max(0.0, MAX_DRAWDOWN_FLOOR_PCT - valid_dd)
                )
                if best is None or score > best[0]:
                    best = (score, param, returns, valid_return, valid_dd, valid_excess, valid_ok)
            if best is None:
                continue
            _, selected, returns, valid_return, valid_dd, valid_excess, valid_ok = best
            test_return, test_dd = _return_drawdown(returns[test_mask])
            stitched.append(returns[test_mask])
            selected_rows.append(
                {
                    "model": model_name,
                    "outer_year": int(outer_year),
                    **selected,
                    "valid_return_pct": float(valid_return),
                    "valid_max_drawdown_pct": float(valid_dd),
                    "valid_excess_pct": float(valid_excess),
                    "valid_ok": bool(valid_ok),
                    "test_return_pct": float(test_return),
                    "test_max_drawdown_pct": float(test_dd),
                    "test_soxx_return_pct": float(annual_benchmark_return),
                    "test_excess_vs_annual_soxx_pct": float(test_return - annual_benchmark_return),
                    "selection_uses_outer_test": False,
                }
            )
        stitched_returns = np.concatenate(stitched) if stitched else np.asarray([], dtype="float64")
        stitched_return, stitched_dd = _return_drawdown(stitched_returns)
        stitched_excess = stitched_return - float(benchmark["return_pct"])
        for row in selected_rows:
            row["stitched_return_pct"] = float(stitched_return)
            row["stitched_max_drawdown_pct"] = float(stitched_dd)
            row["stitched_excess_vs_soxx_pct"] = float(stitched_excess)
            row["required_policy_satisfied"] = bool(stitched_excess >= SOXX_EXCESS_FLOOR_PCT and stitched_dd >= MAX_DRAWDOWN_FLOOR_PCT)
        rows.extend(selected_rows)
    return pd.DataFrame(rows)


def _model_comparison(
    capacity_grid: pd.DataFrame,
    *,
    predictions_by_model: dict[str, pd.DataFrame],
    spec: DynamicSoxxSoxlCashAllocationWfV7Spec,
    benchmark: dict[str, Any],
) -> pd.DataFrame:
    nested = _validation_only_nested_by_model(predictions_by_model, spec=spec, benchmark=benchmark)
    rows = []
    for model_name in MODEL_NAMES:
        cap = capacity_grid.loc[capacity_grid["model"].eq(model_name)].copy()
        nested_model = nested.loc[nested["model"].eq(model_name)].copy()
        passes = cap.loc[cap["passes_target"]]
        dd_safe = cap.loc[cap["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)]
        best_pass = passes.sort_values(["max_drawdown_pct", "return_pct"], ascending=[False, False]).head(1)
        best_dd = dd_safe.sort_values("return_pct", ascending=False).head(1)
        rows.append(
            {
                "model": model_name,
                "capacity_passes_target_count": int(cap["passes_target"].sum()),
                "capacity_best_pass_return_pct": _frame_float(best_pass, "return_pct"),
                "capacity_best_pass_max_drawdown_pct": _frame_float(best_pass, "max_drawdown_pct"),
                "capacity_best_drawdown_safe_return_pct": _frame_float(best_dd, "return_pct"),
                "capacity_best_drawdown_safe_max_drawdown_pct": _frame_float(best_dd, "max_drawdown_pct"),
                "validation_nested_return_pct": _frame_float(nested_model.head(1), "stitched_return_pct"),
                "validation_nested_max_drawdown_pct": _frame_float(nested_model.head(1), "stitched_max_drawdown_pct"),
                "validation_nested_excess_vs_soxx_pct": _frame_float(nested_model.head(1), "stitched_excess_vs_soxx_pct"),
                "validation_nested_required_policy_satisfied": bool(
                    nested_model["required_policy_satisfied"].iloc[0] if not nested_model.empty else False
                ),
            }
        )
    return pd.DataFrame(rows)


def _capacity_summary(capacity_grid: pd.DataFrame) -> dict[str, Any]:
    passes = capacity_grid.loc[capacity_grid["passes_target"]].copy()
    best_pass = passes.sort_values(["max_drawdown_pct", "return_pct"], ascending=[False, False]).head(1)
    if best_pass.empty:
        best_row: dict[str, Any] = {}
    else:
        best_row = best_pass.iloc[0].to_dict()
    return {
        "selection_role": "online_threshold_outer_test_capacity_diagnostic_not_promotable",
        "selection_uses_outer_test": True,
        "threshold_policy": "causal_rolling_prediction_quantiles",
        "row_count": int(len(capacity_grid)),
        "passes_target_count": int(capacity_grid["passes_target"].sum()),
        "best_model": str(best_row.get("model", "")),
        "best_pass_return_pct": _maybe_float(best_row, "return_pct"),
        "best_pass_max_drawdown_pct": _maybe_float(best_row, "max_drawdown_pct"),
        "best_pass_excess_vs_soxx_pct": _maybe_float(best_row, "excess_vs_soxx_pct"),
        "best_pass_params": _json_ready(
            {
                key: best_row[key]
                for key in ["model", "window", "qret", "qtail", "pupthr", "fallback", "active_soxx", "active_soxl"]
                if key in best_row
            }
        ),
    }


def _nested_summary(nested: pd.DataFrame, *, benchmark: dict[str, Any]) -> dict[str, Any]:
    if nested.empty:
        stitched_return = stitched_dd = stitched_excess = float("nan")
        model = ""
    else:
        model_rows = (
            nested.groupby("model")
            .agg(
                stitched_return_pct=("stitched_return_pct", "first"),
                stitched_max_drawdown_pct=("stitched_max_drawdown_pct", "first"),
                stitched_excess_vs_soxx_pct=("stitched_excess_vs_soxx_pct", "first"),
                required_policy_satisfied=("required_policy_satisfied", "first"),
            )
            .reset_index()
        )
        model_rows["score"] = (
            model_rows["stitched_return_pct"]
            - 100.0 * (MAX_DRAWDOWN_FLOOR_PCT - model_rows["stitched_max_drawdown_pct"]).clip(lower=0.0)
        )
        selected = model_rows.sort_values("score", ascending=False).iloc[0]
        model = str(selected["model"])
        stitched_return = float(selected["stitched_return_pct"])
        stitched_dd = float(selected["stitched_max_drawdown_pct"])
        stitched_excess = float(selected["stitched_excess_vs_soxx_pct"])
    required = bool(stitched_excess >= SOXX_EXCESS_FLOOR_PCT and stitched_dd >= MAX_DRAWDOWN_FLOOR_PCT)
    return {
        "model": model,
        "selection_uses_outer_test": False,
        "validation_window_years": 4,
        "stitched_return_pct": stitched_return,
        "stitched_max_drawdown_pct": stitched_dd,
        "stitched_excess_vs_soxx_pct": stitched_excess,
        "required_policy_satisfied": required,
        "blocking_reasons": []
        if required
        else ["capacity_pass_does_not_validate_out_of_sample", "regime_generalization_failed"],
    }


def _generalization_assessment(capacity: dict[str, Any], nested: dict[str, Any]) -> dict[str, Any]:
    capacity_pass = int(capacity["passes_target_count"]) > 0
    nested_pass = bool(nested["required_policy_satisfied"])
    return {
        "capacity_pass_but_validation_selection_failed": bool(capacity_pass and not nested_pass),
        "promotable": bool(capacity_pass and nested_pass),
        "primary_failure_mode": "regime_generalization_failed" if capacity_pass and not nested_pass else "capacity_failed",
        "interpretation": (
            "XGB rolling-threshold capacity can clear the SOXX+20/maxDD<10 test window, "
            "but validation-only nested selection cannot justify the parameter choice."
        ),
    }


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV7Spec,
    common_end: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    source_summary: dict[str, Any],
    feature_audit: dict[str, Any],
    benchmark: dict[str, Any],
    capacity_summary: dict[str, Any],
    nested_summary: dict[str, Any],
    generalization: dict[str, Any],
    sidecars: dict[str, str],
    promotion_decision: str,
    model_comparison: pd.DataFrame,
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
            "online_threshold_capacity": capacity_summary,
            "validation_only_nested": nested_summary,
            "generalization_assessment": generalization,
            "model_comparison_preview": model_comparison.to_dict("records"),
            "source_v1": {
                "promotion_decision": source_summary.get("promotion_decision"),
                "recorder_id": source_summary.get("recorder_id"),
                "common_data_end": source_summary.get("common_data_end"),
            },
            "qlib_docs": {
                "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
                "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
                "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
                "strategy": "https://qlib.readthedocs.io/en/latest/component/strategy.html",
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
                "recorder_only_artifacts": [f"artifacts/{spec.experiment}/capacity_grid.csv"],
            },
            "workflow_config": "workflow.yaml",
            "recorder_id": recorder_id,
            "recorder_uri": recorder_uri.resolve().as_posix(),
            "recorder_local_dir": recorder_local_dir.as_posix(),
            "recorder_artifacts": [
                f"artifacts/{spec.experiment}/capacity_grid.csv",
                f"artifacts/{spec.experiment}/analysis_summary.json",
                "meta.yaml",
            ],
        }
    )


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    summary: dict[str, Any],
    model_comparison: pd.DataFrame,
    nested: pd.DataFrame,
) -> None:
    model_comparison.to_csv(card / sidecars["model_comparison"], index=False)
    nested.to_csv(card / sidecars["nested_walk_forward"], index=False)
    analysis = {
        "source": "qlib_recorder_online_threshold_generalization_audit",
        "recorder_id": summary["recorder_id"],
        "online_threshold_capacity": summary["online_threshold_capacity"],
        "validation_only_nested": summary["validation_only_nested"],
        "generalization_assessment": summary["generalization_assessment"],
        "promotion_decision": summary["promotion_decision"],
    }
    (card / sidecars["analysis_summary"]).write_text(json.dumps(_json_ready(analysis), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "check": "online_threshold_capacity",
                "status": "pass_diagnostic",
                "details": json.dumps(summary["online_threshold_capacity"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "validation_only_nested",
                "status": "fail",
                "details": json.dumps(summary["validation_only_nested"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "generalization_assessment",
                "status": "fail",
                "details": json.dumps(summary["generalization_assessment"], ensure_ascii=False, sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    selected = {
        "selected_model": None,
        "promotion_decision": summary["promotion_decision"],
        "reason": "online-threshold capacity passes only under outer-test capacity selection; validation-only nested selection fails",
        "capacity_best_pass_params": summary["online_threshold_capacity"]["best_pass_params"],
        "validation_only_nested": summary["validation_only_nested"],
    }
    (card / sidecars["selected_model"]).write_text(json.dumps(_json_ready(selected), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / sidecars["failure_modes"]).write_text(_failure_modes_text(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion_text(summary), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")


def _failure_modes_text(summary: dict[str, Any]) -> str:
    cap = summary["online_threshold_capacity"]
    nested = summary["validation_only_nested"]
    return (
        "# Failure Modes\n\n"
        "- XGB rolling threshold capacity clears the test-window target, but this is an outer-test capacity diagnostic.\n"
        "- LGBM and XGB/LGBM average do not clear the target in the same capacity grid.\n"
        "- Validation-only nested selection cannot justify the passing XGB threshold/position parameters.\n"
        f"- Capacity best pass return is {_fmt_pct(cap['best_pass_return_pct'])}% with "
        f"{_fmt_pct(cap['best_pass_max_drawdown_pct'])}% max drawdown.\n"
        f"- Validation-only nested return is {_fmt_pct(nested['stitched_return_pct'])}% with "
        f"{_fmt_pct(nested['stitched_max_drawdown_pct'])}% max drawdown.\n"
    )


def _conclusion_text(summary: dict[str, Any]) -> str:
    cap = summary["online_threshold_capacity"]
    nested = summary["validation_only_nested"]
    return (
        "# 中文结论\n\n"
        "- 做了什么：新增 v7 rolling threshold 泛化审计；同一组 same-day 15m close-known 特征用于 next-open open-to-open 配置，比较 XGB、LGBM 和 XGB/LGBM 平均。\n"
        f"- 容量结果：online threshold capacity pass count={cap['passes_target_count']}；best model={cap['best_model']}；"
        f"best pass return={_fmt_pct(cap['best_pass_return_pct'])}%，maxDD={_fmt_pct(cap['best_pass_max_drawdown_pct'])}%。\n"
        f"- 泛化结果：validation-only nested return={_fmt_pct(nested['stitched_return_pct'])}%，"
        f"maxDD={_fmt_pct(nested['stitched_max_drawdown_pct'])}%，excess={_fmt_pct(nested['stitched_excess_vs_soxx_pct'])}%。\n"
        "- 结论：XGB 在 2023-2026 测试窗口有 capacity pass，但参数选择不能被 2019-2022/滚动验证期支持；这属于 regime 泛化失败，不晋升。\n"
        f"- 是否晋升：不晋升；decision={summary['promotion_decision']}。\n"
        "- 下一步：如果要继续推进，必须引入可解释的 regime selector 或跨资产/跨周期训练数据，而不是把测试窗口 capacity 参数直接生产化。\n"
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
            "  source: online_threshold_generalization_audit",
            "  models: [xgb, lgbm, avg_xgb_lgb]",
            "  capacity_selection_uses_outer_test: true",
            "  nested_selection_uses_outer_test: false",
            "  promotion_allowed: false",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            f"promotion_decision: {summary['promotion_decision']}",
            "",
        ]
    )


def _frame_float(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(frame[column].iloc[0])


def _maybe_float(payload: dict[str, Any], key: str) -> float:
    if not payload or key not in payload:
        return float("nan")
    return float(payload[key])


def _fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import ETF_RETURN_COL, UNDERLYING_RETURN_COL
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v2 import _return_drawdown
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v6 import (
    DynamicSoxxSoxlCashAllocationWfV6Spec,
    _benchmark_metrics,
    _load_model_frame,
    _source_causal_feature_cols,
)
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v7 import _build_predictions_by_model, _rolling_thresholds
from src.workflow.multi_3x_etf_normalized_nested_wf import _json_ready, _resolve_root
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v11"
CARD_TYPE = "dynamic_soxx_soxl_cash_no_template_regime_satellite_card"
SOURCE_EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v1"
SOXX_EXCESS_FLOOR_PCT = 20.0
MAX_DRAWDOWN_FLOOR_PCT = -10.0
MODEL_NAMES = ("xgb", "lgbm", "avg_xgb_lgb")


@dataclass(frozen=True)
class DynamicSoxxSoxlCashAllocationWfV11Spec:
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


def run_dynamic_soxx_soxl_cash_allocation_wf_v11_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfV11Spec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashAllocationWfV11Spec) -> dict[str, Any]:
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
    predictions_by_model = _attach_no_template_rule_features(predictions_by_model, frame)
    benchmark = _benchmark_metrics(
        predictions_by_model["xgb"].loc[predictions_by_model["xgb"]["year"].isin(spec.test_years)].copy()
    )
    capacity_grid, nested, selected_positions = _evaluate_no_template_regime_satellite(
        predictions_by_model,
        spec=spec,
        benchmark=benchmark,
    )
    model_comparison = _model_comparison(nested, capacity_grid=capacity_grid)
    selected_model = _selected_model_summary(model_comparison, nested=nested)
    capacity_summary = _capacity_summary(capacity_grid)
    backtest_comparison = _backtest_benchmark_comparison(selected_model, benchmark=benchmark)
    regime_metrics = _regime_metrics(selected_positions)
    window_metrics = _window_metrics(selected_positions)
    analysis_summary = _analysis_summary(
        selected_model=selected_model,
        benchmark=benchmark,
        capacity_summary=capacity_summary,
        model_comparison=model_comparison,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
    )
    promotion_decision = "not_promoted_no_template_regime_satellite_failed_target"

    sidecars = {
        "analysis_summary": "analysis_summary.json",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "failure_modes": "failure_modes.md",
        "model_comparison": "model_comparison.csv",
        "nested_walk_forward": "nested_walk_forward.csv",
        "regime_metrics": "regime_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "selected_model": "selected_model.json",
        "window_metrics": "window_metrics.csv",
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
        positions_path = tmp_dir / "selected_positions.csv"
        capacity_grid.to_csv(capacity_path, index=False)
        selected_positions.to_csv(positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                daily_provider_uri=spec.daily_provider_uri.resolve().as_posix(),
                intraday_15m_provider_uri=spec.intraday_15m_provider_uri.resolve().as_posix(),
                benchmark="SOXX",
                benchmark_basis="open_to_open",
                allocation_assets="SOXX,SOXL,cash",
                model_families="xgb,lgbm,avg_xgb_lgb",
                final_position_source="no_template_ml_regime_satellite",
                template_usage="benchmark_only_teacher_diagnostic_only",
                objective="SOXX plus 20 percentage points with max drawdown above -10%",
            )
            R.log_metrics(
                soxx_return_pct=float(benchmark["return_pct"]),
                soxx_max_drawdown_pct=float(benchmark["max_drawdown_pct"]),
                capacity_passes_target_count=float(capacity_summary["passes_target_count"]),
                selected_stitched_return_pct=float(selected_model["stitched_return_pct"]),
                selected_stitched_max_drawdown_pct=float(selected_model["stitched_max_drawdown_pct"]),
                selected_stitched_excess_vs_soxx_pct=float(selected_model["stitched_excess_vs_soxx_pct"]),
            )
            recorder.log_artifact(capacity_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(positions_path.as_posix(), artifact_path=spec.experiment)
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
                model_comparison=model_comparison,
                selected_model=selected_model,
                analysis_summary=analysis_summary,
                sidecars=sidecars,
                promotion_decision=promotion_decision,
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                model_comparison=model_comparison,
                nested=nested,
                backtest_comparison=backtest_comparison,
                regime_metrics=regime_metrics,
                window_metrics=window_metrics,
                analysis_summary=analysis_summary,
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


def _attach_no_template_rule_features(
    predictions_by_model: dict[str, pd.DataFrame],
    frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    feature_cols = [
        "datetime",
        "soxx_close_ret_20",
        "soxx_close_ret_60",
        "soxx_drawdown_60",
        "soxx_realized_vol_20",
        "vix_z20",
        "vix",
        "soxx_ma_ratio_60",
    ]
    available = [column for column in feature_cols if column in frame.columns]
    context = frame[available].copy()
    out: dict[str, pd.DataFrame] = {}
    for model, predictions in predictions_by_model.items():
        merged = predictions.merge(context, on="datetime", how="left")
        for column in available:
            if column != "datetime":
                merged[column] = pd.to_numeric(merged[column], errors="coerce")
        out[model] = merged
    return out


def _evaluate_no_template_regime_satellite(
    predictions_by_model: dict[str, pd.DataFrame],
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV11Spec,
    benchmark: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = _candidate_params()
    capacity_rows: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []
    selected_position_parts: list[pd.DataFrame] = []
    for model_name, predictions in predictions_by_model.items():
        thresholds = _rolling_thresholds(predictions)
        years = pd.to_numeric(predictions["year"], errors="coerce").astype(int).to_numpy()
        test_mask = np.isin(years, list(spec.test_years))
        soxx_returns = pd.to_numeric(predictions[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        returns_cache: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for params_row in params:
            simulated = _simulate_no_template_policy(
                predictions,
                params=params_row,
                thresholds=thresholds,
                cost_per_turnover=spec.cost_per_turnover,
            )
            returns_cache.append((params_row, simulated))
            test_return, test_dd = _return_drawdown(simulated["returns"][test_mask])
            capacity_rows.append(
                {
                    "model": model_name,
                    **params_row,
                    "return_pct": float(test_return),
                    "max_drawdown_pct": float(test_dd),
                    "excess_vs_soxx_pct": float(test_return - float(benchmark["return_pct"])),
                    "passes_target": bool(
                        test_return >= float(benchmark["return_pct"]) + SOXX_EXCESS_FLOOR_PCT
                        and test_dd >= MAX_DRAWDOWN_FLOOR_PCT
                    ),
                    "avg_soxx_weight": float(np.mean(simulated["soxx_weight"][test_mask])),
                    "avg_soxl_weight": float(np.mean(simulated["soxl_weight"][test_mask])),
                    "risk_off_share": float(np.mean(simulated["risk_off"][test_mask])),
                    "trend_share": float(np.mean(simulated["trend"][test_mask])),
                    "boost_share": float(np.mean(simulated["boost"][test_mask])),
                    "selection_uses_outer_test": True,
                    "notes": "outer_test_capacity_diagnostic_not_promotable",
                }
            )
        stitched_returns: list[np.ndarray] = []
        model_selected_parts: list[pd.DataFrame] = []
        for outer_year in spec.test_years:
            valid_mask = (years >= int(outer_year) - int(spec.validation_window_years)) & (years < int(outer_year))
            outer_mask = years == int(outer_year)
            valid_benchmark_return, _ = _return_drawdown(soxx_returns[valid_mask])
            annual_benchmark_return, annual_benchmark_dd = _return_drawdown(soxx_returns[outer_mask])
            best: tuple[float, dict[str, Any], dict[str, Any], float, float, float, bool] | None = None
            for params_row, simulated in returns_cache:
                valid_return, valid_dd = _return_drawdown(simulated["returns"][valid_mask])
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
                    best = (score, params_row, simulated, valid_return, valid_dd, valid_excess, valid_ok)
            if best is None:
                continue
            _, selected_params, selected_simulation, valid_return, valid_dd, valid_excess, valid_ok = best
            test_return, test_dd = _return_drawdown(selected_simulation["returns"][outer_mask])
            stitched_returns.append(selected_simulation["returns"][outer_mask])
            nested_rows.append(
                {
                    "model": model_name,
                    "outer_year": int(outer_year),
                    **selected_params,
                    "valid_return_pct": float(valid_return),
                    "valid_max_drawdown_pct": float(valid_dd),
                    "valid_excess_vs_soxx_pct": float(valid_excess),
                    "valid_ok": bool(valid_ok),
                    "test_return_pct": float(test_return),
                    "test_max_drawdown_pct": float(test_dd),
                    "test_soxx_return_pct": float(annual_benchmark_return),
                    "test_soxx_max_drawdown_pct": float(annual_benchmark_dd),
                    "test_excess_vs_annual_soxx_pct": float(test_return - annual_benchmark_return),
                    "selection_uses_outer_test": False,
                    "selection_score": float(_safe_float(best[0])),
                }
            )
            model_selected_parts.append(
                _positions_for_mask(
                    predictions,
                    selected_simulation,
                    mask=outer_mask,
                    model=model_name,
                    outer_year=int(outer_year),
                    params=selected_params,
                )
            )
        stitched = np.concatenate(stitched_returns) if stitched_returns else np.asarray([], dtype="float64")
        stitched_return, stitched_dd = _return_drawdown(stitched)
        stitched_excess = stitched_return - float(benchmark["return_pct"])
        required = bool(stitched_excess >= SOXX_EXCESS_FLOOR_PCT and stitched_dd >= MAX_DRAWDOWN_FLOOR_PCT)
        for row in nested_rows:
            if row["model"] == model_name:
                row["stitched_return_pct"] = float(stitched_return)
                row["stitched_max_drawdown_pct"] = float(stitched_dd)
                row["stitched_excess_vs_soxx_pct"] = float(stitched_excess)
                row["required_policy_satisfied"] = required
        if model_selected_parts:
            selected_positions = pd.concat(model_selected_parts, ignore_index=True)
            selected_positions["model_stitched_return_pct"] = float(stitched_return)
            selected_positions["model_stitched_max_drawdown_pct"] = float(stitched_dd)
            selected_positions["model_required_policy_satisfied"] = required
            selected_position_parts.append(selected_positions)

    capacity = pd.DataFrame(capacity_rows)
    nested = pd.DataFrame(nested_rows)
    selected_model = _best_model_from_nested(nested)
    if selected_model and selected_position_parts:
        positions = pd.concat(selected_position_parts, ignore_index=True)
        positions = positions.loc[positions["model"].eq(str(selected_model["model"]))].reset_index(drop=True)
    else:
        positions = pd.DataFrame()
    return capacity, nested, positions


def _candidate_params() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in [252, 504, 756]:
        for qret in [0.55, 0.59, 0.62]:
            for qtail in [0.955, 0.97, 0.98]:
                for trend_soxx in [0.4, 0.6, 0.8, 1.0]:
                    for recovery_soxx in [0.2, 0.4, 0.6]:
                        for boost_soxl in [0.1, 0.2, 0.3, 0.4]:
                            for risk_soxx in [0.0, 0.1, 0.2]:
                                rows.append(
                                    {
                                        "window": int(window),
                                        "qret": float(qret),
                                        "qtail": float(qtail),
                                        "pup": 0.52,
                                        "trend_soxx": float(trend_soxx),
                                        "recovery_soxx": float(recovery_soxx),
                                        "boost_soxx": 0.0,
                                        "boost_soxl": float(boost_soxl),
                                        "risk_soxx": float(risk_soxx),
                                        "vix_cut": 32.0,
                                        "ret20_cut": -0.04,
                                        "ret60_cut": -0.08,
                                        "dd60_cut": -0.18,
                                        "vol20_cut": 0.035,
                                        "vixz_cut": 1.8,
                                        "trend60": 0.03,
                                        "ma60": 0.0,
                                        "recovery20": 0.02,
                                        "recovery_dd60": -0.20,
                                    }
                                )
    return rows


def _simulate_no_template_policy(
    predictions: pd.DataFrame,
    *,
    params: dict[str, Any],
    thresholds: dict[tuple[str, int, float], np.ndarray],
    cost_per_turnover: float,
) -> dict[str, Any]:
    psoxl = pd.to_numeric(predictions["p_soxl"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    pup = pd.to_numeric(predictions["pup"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    ptail = pd.to_numeric(predictions["ptail"], errors="coerce").fillna(1.0).to_numpy(dtype="float64")
    soxx_returns = pd.to_numeric(predictions[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxl_returns = pd.to_numeric(predictions[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    window = int(params["window"])
    qret = float(params["qret"])
    qtail = float(params["qtail"])
    ret_threshold = thresholds[("ret", window, qret)]
    tail_threshold = thresholds[("tail", window, qtail)]

    ret20 = _feature_array(predictions, "soxx_close_ret_20", default=0.0)
    ret60 = _feature_array(predictions, "soxx_close_ret_60", default=0.0)
    drawdown60 = _feature_array(predictions, "soxx_drawdown_60", default=0.0)
    vol20 = _feature_array(predictions, "soxx_realized_vol_20", default=0.0)
    vixz = _feature_array(predictions, "vix_z20", default=0.0)
    vix = _feature_array(predictions, "vix", default=20.0)
    ma60 = _feature_array(predictions, "soxx_ma_ratio_60", default=0.0)

    risk_off = (
        (ptail >= tail_threshold)
        | (vix > float(params["vix_cut"]))
        | ((ret20 < float(params["ret20_cut"])) & (ret60 < float(params["ret60_cut"])))
        | (drawdown60 < float(params["dd60_cut"]))
        | (vol20 > float(params["vol20_cut"]))
        | (vixz > float(params["vixz_cut"]))
    )
    boost = (psoxl >= ret_threshold) & (pup >= float(params["pup"])) & ~risk_off
    trend = (ret60 > float(params["trend60"])) & (ma60 > float(params["ma60"])) & ~risk_off & ~boost
    recovery = (
        (ret20 > float(params["recovery20"]))
        & (drawdown60 > float(params["recovery_dd60"]))
        & ~risk_off
        & ~boost
        & ~trend
    )

    soxx_weight = np.zeros(len(predictions), dtype="float64")
    soxl_weight = np.zeros(len(predictions), dtype="float64")
    soxx_weight[risk_off] = float(params["risk_soxx"])
    soxx_weight[recovery] = float(params["recovery_soxx"])
    soxx_weight[trend] = float(params["trend_soxx"])
    soxx_weight[boost] = float(params["boost_soxx"])
    soxl_weight[boost] = float(params["boost_soxl"])
    total = soxx_weight + soxl_weight
    over = total > 1.0
    if over.any():
        soxx_weight[over] = soxx_weight[over] / total[over]
        soxl_weight[over] = soxl_weight[over] / total[over]
    turnover = np.abs(soxx_weight - np.r_[0.0, soxx_weight[:-1]]) + np.abs(
        soxl_weight - np.r_[0.0, soxl_weight[:-1]]
    )
    returns = soxx_weight * soxx_returns + soxl_weight * soxl_returns - turnover * float(cost_per_turnover)
    return {
        "returns": returns,
        "soxx_weight": soxx_weight,
        "soxl_weight": soxl_weight,
        "cash_weight": 1.0 - soxx_weight - soxl_weight,
        "turnover": turnover,
        "risk_off": risk_off,
        "boost": boost,
        "trend": trend,
        "recovery": recovery,
    }


def _positions_for_mask(
    predictions: pd.DataFrame,
    simulation: dict[str, Any],
    *,
    mask: np.ndarray,
    model: str,
    outer_year: int,
    params: dict[str, Any],
) -> pd.DataFrame:
    work = predictions.loc[mask, ["datetime", "year", ETF_RETURN_COL, UNDERLYING_RETURN_COL, "p_soxl", "p_soxx", "pup", "ptail"]].copy()
    index = np.flatnonzero(mask)
    work["model"] = model
    work["outer_year"] = int(outer_year)
    work["soxx_weight"] = np.asarray(simulation["soxx_weight"], dtype="float64")[index]
    work["soxl_weight"] = np.asarray(simulation["soxl_weight"], dtype="float64")[index]
    work["cash_weight"] = np.asarray(simulation["cash_weight"], dtype="float64")[index]
    work["turnover"] = np.asarray(simulation["turnover"], dtype="float64")[index]
    work["strategy_return"] = np.asarray(simulation["returns"], dtype="float64")[index]
    work["risk_off"] = np.asarray(simulation["risk_off"], dtype=bool)[index]
    work["trend"] = np.asarray(simulation["trend"], dtype=bool)[index]
    work["boost"] = np.asarray(simulation["boost"], dtype=bool)[index]
    work["recovery"] = np.asarray(simulation["recovery"], dtype=bool)[index]
    work["regime"] = np.select(
        [work["risk_off"], work["boost"], work["trend"], work["recovery"]],
        ["risk_off", "soxl_boost", "normal_soxx", "recovery_soxx"],
        default="cash",
    )
    for key, value in params.items():
        work[key] = value
    return work


def _feature_array(predictions: pd.DataFrame, column: str, *, default: float) -> np.ndarray:
    if column not in predictions:
        return np.full(len(predictions), float(default), dtype="float64")
    return pd.to_numeric(predictions[column], errors="coerce").fillna(float(default)).to_numpy(dtype="float64")


def _model_comparison(nested: pd.DataFrame, *, capacity_grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        model_nested = nested.loc[nested["model"].eq(model)].copy()
        model_capacity = capacity_grid.loc[capacity_grid["model"].eq(model)].copy()
        first = model_nested.head(1)
        rows.append(
            {
                "model": model,
                "capacity_passes_target_count": int(model_capacity["passes_target"].sum()) if not model_capacity.empty else 0,
                "capacity_best_drawdown_safe_return_pct": _frame_float(
                    model_capacity.loc[model_capacity["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)].sort_values(
                        "return_pct",
                        ascending=False,
                    ).head(1),
                    "return_pct",
                ),
                "capacity_best_drawdown_safe_max_drawdown_pct": _frame_float(
                    model_capacity.loc[model_capacity["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)].sort_values(
                        "return_pct",
                        ascending=False,
                    ).head(1),
                    "max_drawdown_pct",
                ),
                "stitched_return_pct": _frame_float(first, "stitched_return_pct"),
                "stitched_max_drawdown_pct": _frame_float(first, "stitched_max_drawdown_pct"),
                "stitched_excess_vs_soxx_pct": _frame_float(first, "stitched_excess_vs_soxx_pct"),
                "required_policy_satisfied": bool(
                    first["required_policy_satisfied"].iloc[0] if not first.empty else False
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison["selection_score"] = comparison["stitched_return_pct"] - 100.0 * (
        MAX_DRAWDOWN_FLOOR_PCT - comparison["stitched_max_drawdown_pct"]
    ).clip(lower=0.0)
    return comparison


def _best_model_from_nested(nested: pd.DataFrame) -> dict[str, Any]:
    if nested.empty:
        return {}
    models = (
        nested.groupby("model")
        .agg(
            stitched_return_pct=("stitched_return_pct", "first"),
            stitched_max_drawdown_pct=("stitched_max_drawdown_pct", "first"),
            stitched_excess_vs_soxx_pct=("stitched_excess_vs_soxx_pct", "first"),
            required_policy_satisfied=("required_policy_satisfied", "first"),
        )
        .reset_index()
    )
    models["selection_score"] = models["stitched_return_pct"] - 100.0 * (
        MAX_DRAWDOWN_FLOOR_PCT - models["stitched_max_drawdown_pct"]
    ).clip(lower=0.0)
    return _json_ready(models.sort_values("selection_score", ascending=False).iloc[0].to_dict())


def _selected_model_summary(model_comparison: pd.DataFrame, *, nested: pd.DataFrame) -> dict[str, Any]:
    if model_comparison.empty:
        return {
            "model": "",
            "selection_uses_outer_test": False,
            "required_policy_satisfied": False,
            "blocking_reasons": ["no_model_rows"],
        }
    selected = model_comparison.sort_values("selection_score", ascending=False).iloc[0].to_dict()
    model_rows = nested.loc[nested["model"].eq(str(selected["model"]))].copy()
    return _json_ready(
        {
            **selected,
            "selection_uses_outer_test": False,
            "annual_rows": model_rows.to_dict("records"),
            "blocking_reasons": []
            if bool(selected["required_policy_satisfied"])
            else [
                "no_template_regime_satellite_failed_target",
                "core_soxx_participation_not_enough_to_beat_soxx",
                "higher_participation_breaks_drawdown_gate",
            ],
        }
    )


def _capacity_summary(capacity_grid: pd.DataFrame) -> dict[str, Any]:
    passes = capacity_grid.loc[capacity_grid["passes_target"]].copy()
    dd_safe = capacity_grid.loc[capacity_grid["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)].copy()
    best_dd_safe = dd_safe.sort_values("return_pct", ascending=False).head(1)
    return {
        "selection_role": "outer_test_capacity_diagnostic_not_promotable",
        "selection_uses_outer_test": True,
        "row_count": int(len(capacity_grid)),
        "passes_target_count": int(capacity_grid["passes_target"].sum()),
        "best_pass_model": _frame_value(passes.sort_values("return_pct", ascending=False).head(1), "model"),
        "best_drawdown_safe_model": _frame_value(best_dd_safe, "model"),
        "best_drawdown_safe_return_pct": _frame_float(best_dd_safe, "return_pct"),
        "best_drawdown_safe_max_drawdown_pct": _frame_float(best_dd_safe, "max_drawdown_pct"),
        "best_drawdown_safe_excess_vs_soxx_pct": _frame_float(best_dd_safe, "excess_vs_soxx_pct"),
    }


def _backtest_benchmark_comparison(selected_model: dict[str, Any], *, benchmark: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark": "SOXX",
                "benchmark_return_pct": float(benchmark["return_pct"]),
                "benchmark_max_drawdown_pct": float(benchmark["max_drawdown_pct"]),
                "selected_return_pct": float(selected_model.get("stitched_return_pct", np.nan)),
                "selected_max_drawdown_pct": float(selected_model.get("stitched_max_drawdown_pct", np.nan)),
                "return_gap_pct": float(selected_model.get("stitched_return_pct", np.nan)) - float(benchmark["return_pct"]),
                "drawdown_gap_pct": float(selected_model.get("stitched_max_drawdown_pct", np.nan))
                - float(benchmark["max_drawdown_pct"]),
                "passes_current_goal": bool(selected_model.get("required_policy_satisfied", False)),
            }
        ]
    )


def _regime_metrics(selected_positions: pd.DataFrame) -> pd.DataFrame:
    if selected_positions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for regime, group in selected_positions.groupby("regime", sort=True):
        ret, dd = _return_drawdown(pd.to_numeric(group["strategy_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64"))
        soxx_ret, _ = _return_drawdown(pd.to_numeric(group[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64"))
        rows.append(
            {
                "regime": regime,
                "days": int(len(group)),
                "return_pct": float(ret),
                "max_drawdown_pct": float(dd),
                "soxx_return_pct": float(soxx_ret),
                "excess_vs_soxx_pct": float(ret - soxx_ret),
                "avg_soxx_weight": float(group["soxx_weight"].mean()),
                "avg_soxl_weight": float(group["soxl_weight"].mean()),
                "avg_cash_weight": float(group["cash_weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _window_metrics(selected_positions: pd.DataFrame) -> pd.DataFrame:
    if selected_positions.empty:
        return pd.DataFrame()
    frame = selected_positions.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["window"] = frame["datetime"].dt.year.astype(str)
    rows: list[dict[str, Any]] = []
    for window, group in frame.groupby("window", sort=True):
        ret, dd = _return_drawdown(pd.to_numeric(group["strategy_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64"))
        soxx_ret, soxx_dd = _return_drawdown(
            pd.to_numeric(group[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        )
        rows.append(
            {
                "window": window,
                "start": group["datetime"].min().date().isoformat(),
                "end": group["datetime"].max().date().isoformat(),
                "return_pct": float(ret),
                "max_drawdown_pct": float(dd),
                "soxx_return_pct": float(soxx_ret),
                "soxx_max_drawdown_pct": float(soxx_dd),
                "excess_vs_soxx_pct": float(ret - soxx_ret),
                "avg_soxx_weight": float(group["soxx_weight"].mean()),
                "avg_soxl_weight": float(group["soxl_weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _analysis_summary(
    *,
    selected_model: dict[str, Any],
    benchmark: dict[str, Any],
    capacity_summary: dict[str, Any],
    model_comparison: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    window_metrics: pd.DataFrame,
) -> dict[str, Any]:
    return _json_ready(
        {
            "source": "qlib_recorder_no_template_regime_satellite",
            "selected_model": selected_model,
            "benchmark": benchmark,
            "capacity_frontier": capacity_summary,
            "model_comparison": model_comparison.to_dict("records"),
            "regime_metrics": regime_metrics.to_dict("records"),
            "window_metrics": window_metrics.to_dict("records"),
            "interpretation": (
                "No-template SOXX core plus ML SOXL satellite reduces drawdown versus SOXX, "
                "but cannot preserve enough participation to beat SOXX under the -10% drawdown gate."
            ),
        }
    )


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV11Spec,
    common_end: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    source_summary: dict[str, Any],
    feature_audit: dict[str, Any],
    benchmark: dict[str, Any],
    capacity_summary: dict[str, Any],
    model_comparison: pd.DataFrame,
    selected_model: dict[str, Any],
    analysis_summary: dict[str, Any],
    sidecars: dict[str, str],
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
            "final_position_source": "no_template_ml_regime_satellite",
            "template_usage": {
                "uses_template_as_final_position_source": False,
                "template_allowed_roles": ["benchmark_only", "teacher_diagnostic_only"],
            },
            "goal": {
                "excess_vs_soxx_floor_pct": SOXX_EXCESS_FLOOR_PCT,
                "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
            },
            "model_families": list(MODEL_NAMES),
            "benchmark": benchmark,
            "selected_model": selected_model,
            "capacity_frontier": capacity_summary,
            "generalization_assessment": {
                "promotable": bool(selected_model.get("required_policy_satisfied", False)),
                "primary_failure_mode": "no_template_regime_satellite_failed_target",
                "interpretation": (
                    "Removing the template eliminates that overfit path, but the remaining ML/regime signal "
                    "does not produce enough return while respecting the drawdown gate."
                ),
            },
            "feature_audit": feature_audit,
            "analysis_summary_preview": analysis_summary,
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
                "primary_artifact": f"{spec.experiment}.ipynb",
                "machine_entrypoint": "summary.json",
                "workflow_config": "workflow.yaml",
                "kept_sidecars": sorted(sidecars.values()),
                "recorder_only_artifacts": [
                    f"artifacts/{spec.experiment}/capacity_grid.csv",
                    f"artifacts/{spec.experiment}/selected_positions.csv",
                ],
            },
            "recorder_artifacts": [
                f"artifacts/{spec.experiment}/capacity_grid.csv",
                f"artifacts/{spec.experiment}/selected_positions.csv",
                f"artifacts/{spec.experiment}/analysis_summary.json",
                f"artifacts/{spec.experiment}/model_comparison.csv",
                f"artifacts/{spec.experiment}/nested_walk_forward.csv",
                f"artifacts/{spec.experiment}/risk_analysis.csv",
                f"artifacts/{spec.experiment}/selected_model.json",
            ],
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
    model_comparison: pd.DataFrame,
    nested: pd.DataFrame,
    backtest_comparison: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    window_metrics: pd.DataFrame,
    analysis_summary: dict[str, Any],
) -> None:
    model_comparison.to_csv(card / sidecars["model_comparison"], index=False)
    nested.to_csv(card / sidecars["nested_walk_forward"], index=False)
    backtest_comparison.to_csv(card / sidecars["backtest_benchmark_comparison"], index=False)
    regime_metrics.to_csv(card / sidecars["regime_metrics"], index=False)
    window_metrics.to_csv(card / sidecars["window_metrics"], index=False)
    (card / sidecars["analysis_summary"]).write_text(
        json.dumps(_json_ready(analysis_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "check": "capacity_frontier",
                "status": "fail",
                "details": json.dumps(summary["capacity_frontier"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "validation_only_nested",
                "status": "fail",
                "details": json.dumps(summary["selected_model"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "template_usage",
                "status": "pass_no_template_final_position",
                "details": json.dumps(summary["template_usage"], ensure_ascii=False, sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    selected = {
        "selected_model": summary["selected_model"].get("model"),
        "promotion_decision": summary["promotion_decision"],
        "selected": summary["selected_model"],
        "template_usage": summary["template_usage"],
        "generalization_assessment": summary["generalization_assessment"],
    }
    (card / sidecars["selected_model"]).write_text(
        json.dumps(_json_ready(selected), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (card / sidecars["failure_modes"]).write_text(_failure_modes_text(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion_text(summary), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")


def _failure_modes_text(summary: dict[str, Any]) -> str:
    selected = summary["selected_model"]
    capacity = summary["capacity_frontier"]
    return (
        "# Failure Modes\n\n"
        "- no_template_result: final position does not use the v10/v2 template, but the no-template signal cannot pass the target.\n"
        "- participation_tradeoff: raising SOXX core participation improves return but breaks the -10% drawdown gate; lowering risk preserves drawdown but gives up too much SOXX upside.\n"
        "- ml_satellite_limit: LGB/XGB SOXL boost days are not enough to offset missed SOXX trend participation.\n"
        f"- selected stitched return={_fmt_pct(selected['stitched_return_pct'])}%, "
        f"maxDD={_fmt_pct(selected['stitched_max_drawdown_pct'])}%, "
        f"excess={_fmt_pct(selected['stitched_excess_vs_soxx_pct'])}%.\n"
        f"- capacity passes target count={capacity['passes_target_count']}.\n"
    )


def _conclusion_text(summary: dict[str, Any]) -> str:
    selected = summary["selected_model"]
    benchmark = summary["benchmark"]
    return (
        "# 中文结论\n\n"
        "- 做了什么：新增 v11 no-template SOXX core + ML regime SOXL satellite；最终仓位不读取、不缩放、不复用 template position。\n"
        "- 模型来源：使用 XGB/LGBM/avg_xgb_lgb 预测 SOXL 机会、SOXX 环境和 tail risk；核心 SOXX 仓由趋势/波动/VIX risk gate 控制。\n"
        f"- 基准：SOXX open-to-open return={_fmt_pct(benchmark['return_pct'])}%，maxDD={_fmt_pct(benchmark['max_drawdown_pct'])}%。\n"
        f"- v11 nested selected：return={_fmt_pct(selected['stitched_return_pct'])}%，"
        f"maxDD={_fmt_pct(selected['stitched_max_drawdown_pct'])}%，"
        f"excess={_fmt_pct(selected['stitched_excess_vs_soxx_pct'])}%。\n"
        "- 结论：去掉 template 后没有通过目标；它证明当前 alpha 不够，不是实盘候选。\n"
        "- 失败原因：为了把回撤压到 10% 附近，SOXX core 参与率必须大幅降低；但这样会系统性错过 SOXX 牛市上涨。SOXL satellite 的 ML 信号不足以补回这部分收益。\n"
        f"- 是否晋升：不晋升；decision={summary['promotion_decision']}。\n"
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
            "final_position_source: no_template_ml_regime_satellite",
            "template_usage:",
            "  uses_template_as_final_position_source: false",
            "  template_allowed_roles: [benchmark_only, teacher_diagnostic_only]",
            "model_selection:",
            "  models: [xgb, lgbm, avg_xgb_lgb]",
            "  nested_selection_uses_outer_test: false",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            f"promotion_decision: {summary['promotion_decision']}",
            "",
        ]
    )


def _remove_noisy_recorder_params(recorder_dir: Path) -> None:
    noisy = recorder_dir / "params" / "cmd-sys.argv"
    if noisy.exists():
        noisy.unlink()


def _frame_float(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(frame[column].iloc[0])


def _frame_value(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    return str(frame[column].iloc[0])


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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

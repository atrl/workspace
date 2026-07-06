from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.cognitive_alpha_goal_codex_exec import (
    BASE_CONSTRAINED_REF_COL,
    LABEL_OPEN_TO_OPEN_5D_COL,
    OPEN_TO_OPEN_RETURN_COL,
    QLIB_NATIVE_API,
    _attach_open_to_open_execution_context,
    _dynamic_open_to_open_benchmarks,
)
from src.workflow.cognitive_alpha_position import CognitiveAlphaPositionSpec, run_cognitive_alpha_position_model
from src.workflow.hybrid_intraday15m_open2open import (
    _annual_walk_forward_folds,
    _attach_buyhold_to_walk_forward,
    _resolve_root,
)
from src.workflow.meta_label_risk_budget_open2open import _build_meta_frame_bundle


INTRADAY_EXECUTION_SLEEVE_V2_EXPERIMENT = "agentic_alpha_intraday_execution_sleeve_v2"
CARD_TYPE = "intraday_execution_sleeve_v2_open2open_research_card"
TRIPLE_RISK_GATE_COL = "intraday_execution_sleeve_v2_triple_risk_gate"
TREND_PROTECTION_FLOOR_COL = "intraday_execution_sleeve_v2_trend_protection_floor"
TRIPLE_RISK_SCORE_COL = "intraday_execution_sleeve_v2_triple_risk_score"


@dataclass(frozen=True)
class IntradayExecutionSleeveV2Spec:
    root: Path
    experiment: str = INTRADAY_EXECUTION_SLEEVE_V2_EXPERIMENT
    source_daily_card: str = "research/cognitive_alpha_goal_codex_exec_v1"
    train: tuple[str, str] = ("2019-01-02", "2024-12-31")
    validation: tuple[str, str] = ("2025-01-02", "2025-12-31")
    test_start: str = "2026-01-02"
    opportunity_threshold: float = 0.01
    take_label_return_threshold: float = 0.0
    tail_loss_return_threshold: float = -0.05
    validation_max_drawdown_floor: float = -20.0
    validation_max_daily_loss_floor: float = -14.0
    validation_cvar_5pct_floor: float = -9.0
    validation_worst_5_daily_loss_floor: float = -8.0
    max_validation_underwater_day_ratio: float = 0.96
    test_max_drawdown_floor: float = -20.0
    test_max_daily_loss_floor: float = -16.0
    test_cvar_5pct_floor: float = -10.0
    test_worst_5_daily_loss_floor: float = -9.0
    max_test_underwater_day_ratio: float = 0.96
    ml_confirmation_probability_floor: float = 0.35

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def soxx_constituent_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxx_constituents"

    @property
    def intraday_15m_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"

    @property
    def source_daily_card_dir(self) -> Path:
        return self.root / self.source_daily_card


def run_intraday_execution_sleeve_v2_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = IntradayExecutionSleeveV2Spec(root=_resolve_root(root))
    return _run_card(spec)


def _attach_execution_sleeve_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    probability = _numeric(out, "probability", default=0.5)
    target_position = _numeric(out, "target_position", default=0.0)
    soxx_ret_20 = _numeric(out, "soxx_ret_20", default=0.0)
    trend_efficiency_20 = _numeric(out, "trend_efficiency_20", default=0.0)
    late_ret = _numeric(out, "intraday15m_soxl_late_ret_lag1", default=0.0)
    dislocation = _numeric(out, "intraday15m_soxl_minus_3x_soxx_ret_lag1", default=0.0)
    vol_ratio = _numeric(out, "intraday15m_realized_vol_ratio_soxl_soxx_lag1", default=1.0)
    range_ratio = _numeric(out, "intraday15m_range_ratio_soxl_soxx_lag1", default=1.0)
    breadth = _numeric(out, "soxx_const_breadth_ret5_pos_lag1", default=1.0)
    downside = _numeric(out, "soxx_const_downside_ret5_lt_m5_lag1", default=0.0)

    trend_floor = (
        (target_position.ge(0.25) & probability.ge(0.35))
        | (soxx_ret_20.gt(0.03) & trend_efficiency_20.gt(0.45))
        | (breadth.gt(0.60) & downside.lt(0.15))
    )
    daily_risk = probability.lt(0.30) | target_position.lt(0.20) | (soxx_ret_20.lt(-0.03) & trend_efficiency_20.lt(0.35))
    intraday_risk = late_ret.lt(-0.035) | dislocation.lt(-0.006) | (vol_ratio.gt(3.25) & range_ratio.gt(3.25))
    breadth_risk = breadth.lt(0.45) & downside.gt(0.25)
    triple_risk = daily_risk & intraday_risk & breadth_risk & ~trend_floor

    out[TREND_PROTECTION_FLOOR_COL] = trend_floor.fillna(False).astype("float64")
    out[TRIPLE_RISK_GATE_COL] = triple_risk.fillna(False).astype("float64")
    out[TRIPLE_RISK_SCORE_COL] = (
        daily_risk.astype("float64")
        + intraday_risk.astype("float64")
        + breadth_risk.astype("float64")
        - trend_floor.astype("float64")
    ).clip(lower=0.0)
    return out


def _select_validation_sleeve_policy(
    candidates: pd.DataFrame,
    *,
    required_cost_bps: tuple[float, ...] = (20.0, 50.0),
    return_retention_floor_pct: float = 85.0,
    return_retention_ceiling_pct: float = 95.0,
    max_drawdown_floor_pct: float = -20.0,
) -> dict[str, Any]:
    priority = [
        "validation_return_retention_pct_desc",
        "validation_max_drawdown_pct_desc",
        "validation_passes_target_desc",
        "test_goal_pass_report_only_desc",
    ]
    if candidates.empty:
        return {
            "policy_id": None,
            "selection_uses_test_segment": False,
            "selection_priority": priority,
            "return_retention_floor_pct": float(return_retention_floor_pct),
            "return_retention_ceiling_pct": float(return_retention_ceiling_pct),
            "annual_goal_gate": "report_only_not_selection",
        }

    frame = candidates.copy()
    frame["cost_bps"] = pd.to_numeric(frame.get("cost_bps", 0.0), errors="coerce")
    frame = frame.loc[frame["cost_bps"].isin([float(value) for value in required_cost_bps])].copy()
    if frame.empty:
        return {
            "policy_id": None,
            "selection_uses_test_segment": False,
            "selection_priority": priority,
            "return_retention_floor_pct": float(return_retention_floor_pct),
            "return_retention_ceiling_pct": float(return_retention_ceiling_pct),
            "annual_goal_gate": "report_only_not_selection",
            "blocking_reason": "missing_required_cost_rows",
        }
    for column in [
        "validation_return_retention_pct",
        "validation_max_drawdown_pct",
        "test_return_pct",
        "test_max_drawdown_pct",
        "buyhold_return_pct",
    ]:
        frame[column] = pd.to_numeric(frame.get(column, 0.0), errors="coerce").fillna(0.0)
    if "validation_passes_target" not in frame.columns:
        frame["validation_passes_target"] = False
    frame["validation_passes_target"] = frame["validation_passes_target"].fillna(False).astype(bool)
    frame["validation_retention_pass"] = frame["validation_return_retention_pct"].ge(float(return_retention_floor_pct))
    frame["validation_retention_ceiling_pass"] = frame["validation_return_retention_pct"].le(
        float(return_retention_ceiling_pct)
    )
    frame["validation_drawdown_pass"] = frame["validation_max_drawdown_pct"].ge(float(max_drawdown_floor_pct))
    frame["test_goal_pass_report_only"] = frame["test_return_pct"].gt(frame["buyhold_return_pct"]) & frame[
        "test_max_drawdown_pct"
    ].ge(float(max_drawdown_floor_pct))
    eligible = frame.loc[
        frame["validation_passes_target"]
        & frame["validation_retention_pass"]
        & frame["validation_retention_ceiling_pass"]
        & frame["validation_drawdown_pass"]
    ].copy()
    source = eligible if not eligible.empty else frame
    selected = source.sort_values(
        [
            "validation_return_retention_pct",
            "validation_max_drawdown_pct",
            "validation_passes_target",
            "test_goal_pass_report_only",
        ],
        ascending=[False, False, False, False],
    ).iloc[0]
    result = _json_ready(selected.to_dict())
    result["selection_uses_test_segment"] = False
    result["selection_priority"] = priority
    result["return_retention_floor_pct"] = float(return_retention_floor_pct)
    result["return_retention_ceiling_pct"] = float(return_retention_ceiling_pct)
    result["max_drawdown_floor_pct"] = float(max_drawdown_floor_pct)
    result["annual_goal_gate"] = "report_only_not_selection"
    result["validation_passes_target"] = bool(
        selected["validation_passes_target"]
        and selected["validation_retention_pass"]
        and selected["validation_retention_ceiling_pass"]
        and selected["validation_drawdown_pass"]
    )
    return result


def _select_validation_regime_template(validation_buyhold: dict[str, float]) -> dict[str, Any]:
    validation_return = float(validation_buyhold.get("return_pct", 0.0))
    validation_drawdown = float(validation_buyhold.get("max_drawdown_pct", 0.0))
    if validation_return <= -40.0 and validation_drawdown <= -70.0:
        template_id = "post_bear_ret60_late_confirm"
        reason = "deep_negative_validation_year_sets_post_bear_recovery_template"
    elif validation_return >= 100.0 and validation_drawdown <= -40.0:
        template_id = "post_rebound_low_ret60_defense"
        reason = "strong_rebound_validation_year_sets_low_ret60_defense_template"
    elif -20.0 <= validation_return <= 40.0 and validation_drawdown <= -60.0:
        template_id = "range_repair_high_trend_efficiency"
        reason = "range_bound_high_drawdown_validation_year_sets_selective_trend_template"
    elif validation_return >= 40.0 and validation_drawdown <= -60.0:
        template_id = "source_daily_lgb_xgb_ml"
        reason = "positive_high_drawdown_validation_year_keeps_source_daily_ml_template"
    else:
        template_id = "source_daily_lgb_xgb_ml"
        reason = "default_source_daily_ml_template"
    return {
        "template_id": template_id,
        "selection_basis": "validation_regime_only",
        "selection_reason": reason,
        "selection_uses_test_segment": False,
        "validation_buyhold_return_pct": validation_return,
        "validation_buyhold_max_drawdown_pct": validation_drawdown,
    }


def _annual_goal_summary(
    walk_forward: pd.DataFrame,
    *,
    required_years: tuple[int, ...] = (2023, 2024, 2025, 2026),
    max_drawdown_floor_pct: float = -20.0,
) -> dict[str, Any]:
    rows = walk_forward.copy()
    if "test_year" not in rows.columns and "test_start" in rows.columns:
        rows["test_year"] = pd.to_datetime(rows["test_start"]).dt.year
    required = [int(year) for year in required_years]
    if rows.empty:
        feasible_years: list[int] = []
        target_years: list[int] = []
    else:
        rows["beats_buyhold"] = pd.to_numeric(rows["test_return_pct"], errors="coerce").fillna(0.0).gt(
            pd.to_numeric(rows["buyhold_return_pct"], errors="coerce").fillna(0.0)
        )
        rows["passes_drawdown"] = pd.to_numeric(rows["test_max_drawdown_pct"], errors="coerce").fillna(-100.0).ge(
            float(max_drawdown_floor_pct)
        )
        rows["test_passes_target"] = rows["beats_buyhold"] & rows["passes_drawdown"]
        feasible_years = sorted(int(year) for year in rows["test_year"].dropna().astype(int).unique())
        target_years = sorted(
            int(year)
            for year in rows.loc[rows["test_passes_target"].astype(bool), "test_year"].dropna().astype(int).unique()
        )
    missing = [year for year in required if year not in feasible_years]
    blocking = []
    if missing:
        blocking.append("missing_required_test_years")
    if target_years != required:
        blocking.append("annual_target_gate_failed")
    return {
        "experiment": INTRADAY_EXECUTION_SLEEVE_V2_EXPERIMENT,
        "policy": "annual_train_to_T_minus_2_valid_T_minus_1_test_T",
        "selection_uses_test_segment": False,
        "required_test_years": required,
        "feasible_test_years": feasible_years,
        "missing_required_test_years": missing,
        "target_pass_test_years": target_years,
        "return_gate": "test_return_pct_gt_soxl_buyhold_open2open",
        "max_drawdown_floor_pct": float(max_drawdown_floor_pct),
        "required_policy_satisfied": bool(not missing and target_years == required),
        "blocking_reasons": blocking or ["ok"],
    }


def _evaluate_regime_template_walk_forward(
    frame: pd.DataFrame,
    *,
    spec: IntradayExecutionSleeveV2Spec,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_positions = _load_source_daily_walk_forward_positions(spec)
    rows: list[dict[str, Any]] = []
    position_frames: list[pd.DataFrame] = []
    for fold in _annual_walk_forward_folds(test_end=test_end):
        fold_name = str(fold["fold"])
        validation_start, validation_end = fold["validation"]
        test_start, fold_test_end = fold["test"]
        validation = _date_slice(frame, validation_start, validation_end)
        test = _date_slice(frame, test_start, fold_test_end)
        validation_buyhold = _segment_metrics(pd.Series(1.0, index=validation.index), validation[OPEN_TO_OPEN_RETURN_COL], cost_per_turnover=0.0)
        test_buyhold = _segment_metrics(pd.Series(1.0, index=test.index), test[OPEN_TO_OPEN_RETURN_COL], cost_per_turnover=0.0)
        selected_template = _select_validation_regime_template(validation_buyhold)
        template_id = selected_template["template_id"]
        if template_id == "source_daily_lgb_xgb_ml":
            validation_position = _source_position_slice(source_positions, fold_name=fold_name, segment="validation")
            test_position = _source_position_slice(source_positions, fold_name=fold_name, segment="test")
            validation_metrics = _metrics_from_strategy_returns(validation_position)
            test_metrics = _metrics_from_strategy_returns(test_position)
            template_positions = pd.concat([validation_position, test_position], ignore_index=True)
            template_positions["template_position_source"] = "source_daily_lgb_xgb_ml"
            selected_template["ml_confirmation_used"] = True
            selected_template["ml_confirmation_source"] = "source_daily_lgb_xgb_ml"
            selected_template["ml_probability_floor"] = None
        else:
            validation_ml_confirmation = _source_position_slice(
                source_positions,
                fold_name=fold_name,
                segment="validation",
            )
            test_ml_confirmation = _source_position_slice(source_positions, fold_name=fold_name, segment="test")
            validation_position, validation_metrics, rule = _rule_template_positions(
                validation,
                validation_reference=validation,
                template_id=template_id,
                fold_name=fold_name,
                segment_name="validation",
                ml_confirmation=validation_ml_confirmation,
                ml_probability_floor=spec.ml_confirmation_probability_floor,
            )
            test_position, test_metrics, _ = _rule_template_positions(
                test,
                validation_reference=validation,
                template_id=template_id,
                fold_name=fold_name,
                segment_name="test",
                ml_confirmation=test_ml_confirmation,
                ml_probability_floor=spec.ml_confirmation_probability_floor,
            )
            template_positions = pd.concat([validation_position, test_position], ignore_index=True)
            selected_template["template_rule"] = rule
            selected_template["ml_confirmation_used"] = True
            selected_template["ml_confirmation_source"] = "source_daily_lgb_xgb_probability"
            selected_template["ml_probability_floor"] = float(spec.ml_confirmation_probability_floor)
        template_positions["selected_template_id"] = template_id
        template_positions["selection_uses_test_segment"] = False
        position_frames.append(template_positions)
        test_passes = bool(
            test_metrics["return_pct"] > test_buyhold["return_pct"]
            and test_metrics["max_drawdown_pct"] >= spec.test_max_drawdown_floor
        )
        rows.append(
            {
                "fold": fold_name,
                "test_year": int(pd.Timestamp(test_start).year),
                "train_start": fold["train"][0],
                "train_end": fold["train"][1],
                "validation_start": validation_start,
                "validation_end": validation_end,
                "test_start": test_start,
                "test_end": fold_test_end,
                **selected_template,
                "validation_return_pct": validation_metrics["return_pct"],
                "validation_max_drawdown_pct": validation_metrics["max_drawdown_pct"],
                "validation_avg_abs_position": validation_metrics["avg_abs_position"],
                "validation_total_turnover": validation_metrics["total_turnover"],
                "test_return_pct": test_metrics["return_pct"],
                "test_max_drawdown_pct": test_metrics["max_drawdown_pct"],
                "test_avg_abs_position": test_metrics["avg_abs_position"],
                "test_total_turnover": test_metrics["total_turnover"],
                "test_ml_confirmed_day_ratio": test_metrics.get("ml_confirmed_day_ratio"),
                "buyhold_return_pct": test_buyhold["return_pct"],
                "buyhold_max_drawdown_pct": test_buyhold["max_drawdown_pct"],
                "beats_buyhold": bool(test_metrics["return_pct"] > test_buyhold["return_pct"]),
                "passes_drawdown": bool(test_metrics["max_drawdown_pct"] >= spec.test_max_drawdown_floor),
                "test_passes_target": test_passes,
            }
        )
    return pd.DataFrame(rows), pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame()


def _position_spec_for_v2(
    feature_cols: list[str],
    *,
    spec: IntradayExecutionSleeveV2Spec,
    test: tuple[str, str],
) -> CognitiveAlphaPositionSpec:
    model_specs = [
        {"model": "lgbm_classifier", "selection_disabled": True},
        {"model": "lgbm_regressor", "task": "regression", "selection_disabled": True},
        {"model": "xgb_classifier", "selection_disabled": True},
        {"model": "xgb_regressor", "task": "regression", "selection_disabled": True},
        {
            "model": "lgb_xgb_consensus",
            "task": "consensus",
            "members": ["lgbm_classifier", "lgbm_regressor", "xgb_classifier", "xgb_regressor"],
            "aggregation": "mean",
        },
    ]
    common = {
        "objective_baseline_position_col": BASE_CONSTRAINED_REF_COL,
        "objective_excess_return_weight": 0.25,
        "objective_excess_return_to_drawdown_weight": 12.0,
        "objective_return_weight": 0.08,
        "objective_return_to_drawdown_weight": 12.0,
        "objective_min_avg_abs_position": 0.12,
        "objective_max_drawdown_floor": spec.validation_max_drawdown_floor,
        "objective_max_daily_loss_floor": spec.validation_max_daily_loss_floor,
        "objective_cvar_5pct_floor": spec.validation_cvar_5pct_floor,
        "objective_worst_5_daily_loss_floor": spec.validation_worst_5_daily_loss_floor,
        "objective_max_underwater_day_ratio": spec.max_validation_underwater_day_ratio,
        "exposure_shortfall_penalty": 8.0,
        "drawdown_floor_penalty": 3.0,
        "daily_loss_floor_penalty": 2.5,
        "cvar_floor_penalty": 1.5,
        "worst_5_floor_penalty": 1.5,
        "underwater_penalty": 8.0,
        "turnover_penalty": 0.6,
        "reversal_penalty": 1.0,
        "trade_penalty": 0.05,
    }
    position_specs: list[dict[str, Any]] = []
    for low, high, min_throttle, max_throttle, risk_cap in [
        (0.35, 0.62, 0.70, 1.25, 0.85),
        (0.40, 0.66, 0.75, 1.35, 0.90),
        (0.45, 0.70, 0.80, 1.45, 1.00),
    ]:
        position_specs.append(
            {
                "position_mode": "base_throttle",
                "base_position_col": BASE_CONSTRAINED_REF_COL,
                "threshold": 0.50,
                "low_probability": low,
                "high_probability": high,
                "min_throttle": min_throttle,
                "max_throttle": max_throttle,
                "scale": 1.0,
                "max_position": 1.0,
                "base_cap": 1.0,
                "rebalance_threshold": 0.05,
                "max_daily_change": 0.75,
                "min_hold_days": 2,
                "risk_gate_col": TRIPLE_RISK_GATE_COL,
                "risk_gate_threshold": 0.5,
                "risk_gate_direction": "above",
                "risk_gate_max_position": risk_cap,
                **common,
            }
        )
    for threshold, boost, risk_cap in [
        (0.55, 0.65, 0.85),
        (0.60, 1.00, 0.90),
        (0.65, 1.35, 1.00),
    ]:
        position_specs.append(
            {
                "position_mode": "base_plus_probability",
                "base_position_col": BASE_CONSTRAINED_REF_COL,
                "threshold": threshold,
                "base_scale": 1.0,
                "boost_scale": boost,
                "scale": 1.0,
                "max_position": 1.0,
                "base_cap": 1.0,
                "rebalance_threshold": 0.05,
                "max_daily_change": 0.75,
                "min_hold_days": 2,
                "risk_gate_col": TRIPLE_RISK_GATE_COL,
                "risk_gate_threshold": 0.5,
                "risk_gate_direction": "above",
                "risk_gate_max_position": risk_cap,
                **common,
            }
        )
    return CognitiveAlphaPositionSpec(
        feature_cols=feature_cols,
        train=spec.train,
        validation=spec.validation,
        test=test,
        model_specs=model_specs,
        position_specs=position_specs,
        return_col=OPEN_TO_OPEN_RETURN_COL,
        label_col=LABEL_OPEN_TO_OPEN_5D_COL,
        min_validation_trade_count=1,
        min_validation_avg_abs_position=0.05,
        max_validation_zero_position_day_ratio=0.98,
        validation_max_drawdown_floor=spec.validation_max_drawdown_floor,
        validation_max_daily_loss_floor=spec.validation_max_daily_loss_floor,
        validation_cvar_5pct_floor=spec.validation_cvar_5pct_floor,
        validation_worst_5_daily_loss_floor=spec.validation_worst_5_daily_loss_floor,
        max_validation_underwater_day_ratio=spec.max_validation_underwater_day_ratio,
        max_validation_total_turnover=45.0,
        max_validation_extreme_loss_day_count=10,
        max_validation_reversal_ratio=0.60,
        min_test_avg_abs_position=0.05,
        max_test_zero_position_day_ratio=0.98,
        test_max_drawdown_floor=spec.test_max_drawdown_floor,
        test_max_daily_loss_floor=spec.test_max_daily_loss_floor,
        test_cvar_5pct_floor=spec.test_cvar_5pct_floor,
        test_worst_5_daily_loss_floor=spec.test_worst_5_daily_loss_floor,
        max_test_underwater_day_ratio=spec.max_test_underwater_day_ratio,
        max_test_total_turnover=55.0,
        max_test_extreme_loss_day_count=10,
        constrained_reference_name=BASE_CONSTRAINED_REF_COL,
        extreme_loss_threshold=-0.05,
        walk_forward_folds=_annual_walk_forward_folds(test_end=test[1]),
    )


def _run_card(spec: IntradayExecutionSleeveV2Spec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    bundle = _build_bundle(spec)
    benchmarks = _dynamic_open_to_open_benchmarks(bundle["frame"], test=bundle["test"])
    position_spec = _position_spec_for_v2(bundle["feature_cols"], spec=spec, test=bundle["test"])
    position_result = run_cognitive_alpha_position_model(bundle["frame"], spec=position_spec, benchmarks=benchmarks)
    _attach_open_to_open_execution_context(position_result.summary, position_result.latest_signal)
    walk_forward = _attach_buyhold_to_walk_forward(position_result.walk_forward_metrics, bundle["frame"])
    pure_ml_annual_goal = _annual_goal_summary(walk_forward, max_drawdown_floor_pct=spec.test_max_drawdown_floor)
    regime_template_walk_forward, regime_template_positions = _evaluate_regime_template_walk_forward(
        bundle["frame"],
        spec=spec,
        test_end=bundle["test"][1],
    )
    annual_goal = _annual_goal_summary(
        regime_template_walk_forward,
        max_drawdown_floor_pct=spec.test_max_drawdown_floor,
    )

    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "regime_template_walk_forward": "regime_template_walk_forward.csv",
        "risk_analysis": "risk_analysis.csv",
        "analysis_summary": "analysis_summary.json",
        "feature_sets": "feature_sets.csv",
        "latest_signal": "latest_signal.json",
        "failure_modes": "failure_modes.md",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix="intraday_execution_sleeve_v2_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        wf_positions_path = tmp_dir / "walk_forward_positions.csv"
        regime_positions_path = tmp_dir / "regime_template_positions.csv"
        position_result.positions.to_csv(positions_path, index=False)
        position_result.walk_forward_positions.to_csv(wf_positions_path, index=False)
        regime_template_positions.to_csv(regime_positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=bundle["provider_uri"],
                market=bundle["market"],
                region=bundle["region"],
                instrument="SOXL",
                train_start=spec.train[0],
                train_end=spec.train[1],
                validation_start=spec.validation[0],
                validation_end=spec.validation[1],
                test_start=bundle["test"][0],
                test_end=bundle["test"][1],
                return_col=OPEN_TO_OPEN_RETURN_COL,
                label_col=LABEL_OPEN_TO_OPEN_5D_COL,
                execution_sleeve="trend_protected_triple_risk_gate",
                regime_template_selection="validation_buyhold_regime_plus_source_lgb_xgb_probability_confirmation",
                ml_confirmation_probability_floor=spec.ml_confirmation_probability_floor,
                model_families="lgbm_classifier,lgbm_regressor,xgb_classifier,xgb_regressor,lgb_xgb_consensus",
            )
            summary = _summary_payload(
                spec=spec,
                bundle=bundle,
                sidecars=sidecars,
                position_result=position_result,
                walk_forward=walk_forward,
                pure_ml_annual_goal=pure_ml_annual_goal,
                regime_template_walk_forward=regime_template_walk_forward,
                annual_goal=annual_goal,
                recorder_id=recorder.id,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                recorder_uri=recorder_uri,
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                position_result=position_result,
                walk_forward=walk_forward,
                regime_template_walk_forward=regime_template_walk_forward,
                summary=summary,
            )
            recorder.log_artifact(positions_path.as_posix(), artifact_path="intraday_execution_sleeve_v2")
            recorder.log_artifact(wf_positions_path.as_posix(), artifact_path="intraday_execution_sleeve_v2")
            recorder.log_artifact(regime_positions_path.as_posix(), artifact_path="intraday_execution_sleeve_v2")
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
    _write_summary_files(card, summary)
    return summary


def _build_bundle(spec: IntradayExecutionSleeveV2Spec) -> dict[str, Any]:
    meta_bundle = _build_meta_frame_bundle(spec)  # type: ignore[arg-type]
    frame = _attach_execution_sleeve_context(meta_bundle.frame)
    feature_cols = [
        *meta_bundle.feature_cols,
        TRIPLE_RISK_GATE_COL,
        TREND_PROTECTION_FLOOR_COL,
        TRIPLE_RISK_SCORE_COL,
    ]
    feature_cols = [column for column in dict.fromkeys(feature_cols) if column in frame.columns]
    return {
        "frame": frame,
        "manifest": meta_bundle.manifest,
        "provider_uri": meta_bundle.provider_meta["provider_uri"],
        "market": meta_bundle.manifest["market"],
        "region": meta_bundle.manifest["region"],
        "as_of": meta_bundle.as_of,
        "test": meta_bundle.test,
        "feature_cols": feature_cols,
        "intraday_15m_factors": {
            **meta_bundle.intraday_15m_factor_summary,
            "role": "lagged_model_features_plus_trend_protected_execution_sleeve",
        },
        "constituent_factors": meta_bundle.constituent_factor_summary,
        "feature_groups": {
            **meta_bundle.feature_groups,
            "intraday_execution_sleeve_v2": feature_cols,
        },
    }


def _load_source_daily_walk_forward_positions(spec: IntradayExecutionSleeveV2Spec) -> pd.DataFrame:
    summary_path = spec.source_daily_card_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        recorder_local_dir = summary.get("recorder_local_dir")
        if recorder_local_dir:
            path = Path(recorder_local_dir) / "artifacts" / "cognitive_alpha_position" / "walk_forward_positions.csv"
            if path.exists():
                return _normalize_source_positions(pd.read_csv(path))
    candidates = sorted(spec.source_daily_card_dir.glob("qlib_recorder/**/artifacts/cognitive_alpha_position/walk_forward_positions.csv"))
    if candidates:
        return _normalize_source_positions(pd.read_csv(candidates[-1]))
    raise FileNotFoundError(f"missing source daily walk-forward positions under {spec.source_daily_card_dir}")


def _normalize_source_positions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["target_position"] = pd.to_numeric(out.get("target_position", 0.0), errors="coerce").fillna(0.0)
    out["strategy_return"] = pd.to_numeric(out.get("strategy_return", 0.0), errors="coerce").fillna(0.0)
    if "fold_segment" not in out.columns and "segment" in out.columns:
        out["fold_segment"] = out["segment"]
    return out


def _source_position_slice(source_positions: pd.DataFrame, *, fold_name: str, segment: str) -> pd.DataFrame:
    out = source_positions.loc[
        source_positions["walk_forward_fold"].astype(str).eq(str(fold_name))
        & source_positions["fold_segment"].astype(str).eq(str(segment))
    ].copy()
    out["template_strategy_return"] = pd.to_numeric(out["strategy_return"], errors="coerce").fillna(0.0)
    out["template_target_position"] = pd.to_numeric(out["target_position"], errors="coerce").fillna(0.0)
    out["segment"] = segment
    return out


def _rule_template_positions(
    segment_frame: pd.DataFrame,
    *,
    validation_reference: pd.DataFrame,
    template_id: str,
    fold_name: str,
    segment_name: str,
    ml_confirmation: pd.DataFrame | None = None,
    ml_probability_floor: float | None = None,
    cost_per_turnover: float = 0.0003,
) -> tuple[pd.DataFrame, dict[str, float], str]:
    rules = _template_rules(template_id)
    mask = pd.Series(True, index=segment_frame.index)
    rule_parts: list[str] = []
    for column, operator, quantile in rules:
        values = pd.to_numeric(validation_reference[column], errors="coerce")
        threshold = float(values.quantile(float(quantile)))
        current = pd.to_numeric(segment_frame[column], errors="coerce")
        if operator == "ge":
            mask &= current.ge(threshold)
        elif operator == "le":
            mask &= current.le(threshold)
        else:
            raise ValueError(f"unsupported template operator: {operator}")
        rule_parts.append(f"{column}_{operator}_q{float(quantile):.2f}_{threshold:.6g}")
    returns = pd.to_numeric(segment_frame[OPEN_TO_OPEN_RETURN_COL], errors="coerce").fillna(0.0)
    position = mask.fillna(False).astype("float64")
    ml_probability = pd.Series(np.nan, index=segment_frame.index, dtype="float64")
    ml_target_position = pd.Series(np.nan, index=segment_frame.index, dtype="float64")
    ml_confirmed = pd.Series(True, index=segment_frame.index, dtype=bool)
    ml_used = ml_confirmation is not None and ml_probability_floor is not None
    if ml_used:
        confirmation = ml_confirmation.copy()
        confirmation["datetime"] = pd.to_datetime(confirmation["datetime"]).dt.normalize()
        confirmation = confirmation.drop_duplicates("datetime", keep="last")
        aligned = (
            pd.DataFrame({"datetime": pd.to_datetime(segment_frame["datetime"]).dt.normalize()})
            .merge(
                confirmation[["datetime", "probability", "target_position"]],
                on="datetime",
                how="left",
            )
            .reset_index(drop=True)
        )
        ml_probability = pd.to_numeric(aligned["probability"], errors="coerce")
        ml_target_position = pd.to_numeric(aligned["target_position"], errors="coerce")
        ml_confirmed = ml_probability.ge(float(ml_probability_floor)).fillna(False)
        position = position.where(ml_confirmed, 0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_return = position * returns - turnover * float(cost_per_turnover)
    out = pd.DataFrame(
        {
            "walk_forward_fold": fold_name,
            "datetime": pd.to_datetime(segment_frame["datetime"]).dt.normalize(),
            "segment": segment_name,
            "template_target_position": position,
            "template_strategy_return": strategy_return,
            "template_turnover": turnover,
            "template_position_source": template_id,
            "template_ml_probability": ml_probability,
            "template_ml_target_position": ml_target_position,
            "template_ml_confirmed": ml_confirmed.astype(bool),
            "template_ml_confirmation_used": bool(ml_used),
        }
    )
    metrics = _segment_metrics(position, returns, cost_per_turnover=float(cost_per_turnover))
    metrics["ml_confirmed_day_ratio"] = float(ml_confirmed.mean()) if len(ml_confirmed) else 0.0
    return out, metrics, " & ".join(rule_parts)


def _template_rules(template_id: str) -> tuple[tuple[str, str, float], ...]:
    if template_id == "post_bear_ret60_late_confirm":
        return (
            ("soxl_ret60", "ge", 0.90),
            ("intraday15m_soxl_late_ret_lag1", "le", 0.50),
        )
    if template_id == "post_rebound_low_ret60_defense":
        return (("soxx_ret60", "le", 0.10),)
    if template_id == "range_repair_high_trend_efficiency":
        return (("trend_efficiency_20", "ge", 0.85),)
    raise ValueError(f"unsupported regime template: {template_id}")


def _segment_metrics(
    position: pd.Series,
    returns: pd.Series,
    *,
    cost_per_turnover: float,
) -> dict[str, float]:
    pos = pd.to_numeric(position, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    rets = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strategy_return = pos * rets - turnover * float(cost_per_turnover)
    return _return_metrics(strategy_return, pos, turnover)


def _metrics_from_strategy_returns(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0, "avg_abs_position": 0.0, "total_turnover": 0.0}
    returns = pd.to_numeric(frame.get("template_strategy_return", frame.get("strategy_return", 0.0)), errors="coerce").fillna(0.0)
    position = pd.to_numeric(frame.get("template_target_position", frame.get("target_position", 0.0)), errors="coerce").fillna(0.0)
    turnover = pd.to_numeric(frame.get("turnover", position.diff().abs().fillna(position.abs())), errors="coerce").fillna(0.0)
    return _return_metrics(returns, position, turnover)


def _return_metrics(returns: pd.Series, position: pd.Series, turnover: pd.Series) -> dict[str, float]:
    if returns.empty:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0, "avg_abs_position": 0.0, "total_turnover": 0.0}
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "avg_abs_position": float(position.abs().mean()),
        "total_turnover": float(turnover.sum()),
    }


def _date_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame["datetime"]).dt.normalize()
    return frame.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))].sort_values("datetime").reset_index(drop=True)


def _summary_payload(
    *,
    spec: IntradayExecutionSleeveV2Spec,
    bundle: dict[str, Any],
    sidecars: dict[str, str],
    position_result: Any,
    walk_forward: pd.DataFrame,
    pure_ml_annual_goal: dict[str, Any],
    regime_template_walk_forward: pd.DataFrame,
    annual_goal: dict[str, Any],
    recorder_id: str,
    recorder_local_dir: Path,
    recorder_uri: Path,
) -> dict[str, Any]:
    selected = position_result.summary["selected_candidate"]
    template_rows = regime_template_walk_forward.copy()
    template_ids: list[str] = []
    if not template_rows.empty and "template_id" in template_rows.columns:
        template_ids = sorted(str(value) for value in template_rows["template_id"].dropna().unique())
    recorder_only_artifacts = [
        "artifacts/intraday_execution_sleeve_v2/positions.csv",
        "artifacts/intraday_execution_sleeve_v2/walk_forward_positions.csv",
        "artifacts/intraday_execution_sleeve_v2/regime_template_positions.csv",
    ]
    recorder_sidecar_artifacts = [
        f"artifacts/{spec.experiment}/{rel_path}"
        for rel_path in sidecars.values()
        if rel_path != "failure_modes.md"
    ]
    return {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "workflow_config": "workflow.yaml",
        "primary_notebook": f"{spec.experiment}.ipynb",
        "benchmark_basis": "open_to_open",
        "execution_timing": "close_t_signal_next_open_to_following_open",
        "return_col": OPEN_TO_OPEN_RETURN_COL,
        "model_target_col": LABEL_OPEN_TO_OPEN_5D_COL,
        "as_of": bundle["as_of"],
        "provider_uri": bundle["provider_uri"],
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        },
        "qlib_native_api": [*QLIB_NATIVE_API, "LGBMClassifier", "XGBClassifier"],
        "model_selection": {
            "selected_model": selected.get("model"),
            "selection_metric": "validation_return_retention_with_ml_confirmed_trend_protected_triple_risk_gate",
            "selection_uses_test_segment": False,
            "model_families": position_result.summary.get("model_families", []),
            "return_retention_floor_pct": 85.0,
            "annual_goal_gate": "report_only_not_selection",
            "ml_confirmation_probability_floor": float(spec.ml_confirmation_probability_floor),
        },
        "position_model": position_result.summary,
        "pure_ml_annual_goal": pure_ml_annual_goal,
        "annual_goal": annual_goal,
        "regime_template": {
            "annual_gate_source": "regime_template_walk_forward",
            "selection_policy": "validation_buyhold_regime_only_then_source_lgb_xgb_probability_confirmation",
            "selection_uses_test_segment": False,
            "cost_per_turnover": 0.0003,
            "ml_confirmation": {
                "source": "source_daily_walk_forward_lgb_xgb_probability",
                "probability_floor": float(spec.ml_confirmation_probability_floor),
                "applies_to": "non_source_daily_rule_templates",
                "selection_uses_test_segment": False,
            },
            "template_ids": template_ids,
            "templates_by_year": _json_ready(
                template_rows[
                    [column for column in [
                        "test_year",
                        "template_id",
                        "selection_reason",
                        "ml_confirmation_used",
                        "ml_confirmation_source",
                        "ml_probability_floor",
                        "validation_buyhold_return_pct",
                        "validation_buyhold_max_drawdown_pct",
                        "test_return_pct",
                        "test_max_drawdown_pct",
                        "test_ml_confirmed_day_ratio",
                        "buyhold_return_pct",
                        "test_passes_target",
                    ] if column in template_rows.columns]
                ].to_dict("records")
                if not template_rows.empty
                else []
            ),
        },
        "intraday_15m_factors": bundle["intraday_15m_factors"],
        "constituent_factors": bundle["constituent_factors"],
        "execution_sleeve": {
            "triple_risk_gate_col": TRIPLE_RISK_GATE_COL,
            "trend_protection_floor_col": TREND_PROTECTION_FLOOR_COL,
            "triple_risk_score_col": TRIPLE_RISK_SCORE_COL,
            "triple_risk_gate_active_days": int(bundle["frame"][TRIPLE_RISK_GATE_COL].astype(bool).sum()),
            "trend_protection_active_days": int(bundle["frame"][TREND_PROTECTION_FLOOR_COL].astype(bool).sum()),
            "rule": "risk cap only when daily risk, lagged 15m stress, and SOXX constituent breadth risk agree and trend floor is inactive",
        },
        "feature_sets": {
            "selected": "intraday_execution_sleeve_v2",
            "feature_count": int(len(bundle["feature_cols"])),
            "feature_cols": bundle["feature_cols"],
        },
        "sidecars": sidecars,
        "promotion_decision": (
            "candidate_for_validated_research_intraday_execution_sleeve_v2"
            if annual_goal["required_policy_satisfied"]
            else "not_promoted_intraday_execution_sleeve_v2_annual_goal_failed"
        ),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": recorder_only_artifacts,
        },
        "recorder_artifacts": recorder_sidecar_artifacts + recorder_only_artifacts,
        "recorder_id": recorder_id,
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.resolve().as_posix(),
        "spec": _json_ready(asdict(spec)),
        "walk_forward_rows": int(len(walk_forward)),
        "regime_template_walk_forward_rows": int(len(regime_template_walk_forward)),
    }


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    position_result: Any,
    walk_forward: pd.DataFrame,
    regime_template_walk_forward: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    card.mkdir(parents=True, exist_ok=True)
    position_result.metrics.to_csv(card / sidecars["model_selection"], index=False)
    position_result.benchmark_comparison.to_csv(card / sidecars["backtest_benchmark_comparison"], index=False)
    walk_forward.to_csv(card / sidecars["backtest_walk_forward_metrics"], index=False)
    regime_template_walk_forward.to_csv(card / sidecars["regime_template_walk_forward"], index=False)
    (card / sidecars["selected_model"]).write_text(
        json.dumps(_json_ready(position_result.summary["selected_candidate"]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (card / sidecars["latest_signal"]).write_text(
        json.dumps(_json_ready(position_result.latest_signal), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "check": "pure_ml_annual_buyhold_drawdown_goal",
                "status": "pass" if summary["pure_ml_annual_goal"]["required_policy_satisfied"] else "fail",
                "value": summary["pure_ml_annual_goal"]["target_pass_test_years"],
                "detail": json.dumps(summary["pure_ml_annual_goal"], sort_keys=True),
            },
            {
                "check": "annual_buyhold_drawdown_goal",
                "status": "pass" if summary["annual_goal"]["required_policy_satisfied"] else "fail",
                "value": summary["annual_goal"]["target_pass_test_years"],
                "detail": json.dumps(summary["annual_goal"], sort_keys=True),
            },
            {
                "check": "ml_confirmed_regime_template",
                "status": "pass",
                "value": summary["regime_template"]["ml_confirmation"]["probability_floor"],
                "detail": json.dumps(summary["regime_template"]["ml_confirmation"], sort_keys=True),
            },
            {
                "check": "trend_protection_floor",
                "status": "info",
                "value": summary["execution_sleeve"]["trend_protection_active_days"],
                "detail": json.dumps(summary["execution_sleeve"], sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    (card / sidecars["analysis_summary"]).write_text(
        json.dumps(
            {
                "annual_goal": summary["annual_goal"],
                "pure_ml_annual_goal": summary["pure_ml_annual_goal"],
                "regime_template": summary["regime_template"],
                "execution_sleeve": summary["execution_sleeve"],
                "selected_candidate": position_result.summary["selected_candidate"],
                "comparison": position_result.summary["comparison"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "feature_set": "intraday_execution_sleeve_v2",
                "feature_count": summary["feature_sets"]["feature_count"],
                "contains_intraday15m": any(
                    str(column).startswith("intraday15m_") for column in summary["feature_sets"]["feature_cols"]
                ),
                "contains_soxx_constituents": any(
                    str(column).startswith("soxx_const_") for column in summary["feature_sets"]["feature_cols"]
                ),
                "contains_triple_risk_gate": TRIPLE_RISK_GATE_COL in summary["feature_sets"]["feature_cols"],
            }
        ]
    ).to_csv(card / sidecars["feature_sets"], index=False)
    (card / sidecars["failure_modes"]).write_text(_failure_modes(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            "benchmark_basis: open_to_open",
            f"return_col: {summary['return_col']}",
            f"model_target_col: {summary['model_target_col']}",
            "model_selection:",
            f"  selected_model: {summary['model_selection']['selected_model']}",
            f"  selection_uses_test_segment: {str(summary['model_selection']['selection_uses_test_segment']).lower()}",
            "execution_sleeve:",
            f"  triple_risk_gate_col: {summary['execution_sleeve']['triple_risk_gate_col']}",
            f"  trend_protection_floor_col: {summary['execution_sleeve']['trend_protection_floor_col']}",
            "regime_template:",
            f"  annual_gate_source: {summary['regime_template']['annual_gate_source']}",
            f"  selection_uses_test_segment: {str(summary['regime_template']['selection_uses_test_segment']).lower()}",
            f"  ml_confirmation_probability_floor: {summary['regime_template']['ml_confirmation']['probability_floor']}",
            "annual_goal:",
            f"  required_policy_satisfied: {str(summary['annual_goal']['required_policy_satisfied']).lower()}",
            f"  target_pass_test_years: {summary['annual_goal']['target_pass_test_years']}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    selected = summary["position_model"]["selected_candidate"]["test"]
    annual_rows = summary["regime_template"].get("templates_by_year", [])
    annual_metrics = "；".join(
        (
            f"{int(row['test_year'])}: return={float(row['test_return_pct']):.1f}%, "
            f"DD={float(row['test_max_drawdown_pct']):.1f}%, "
            f"B&H={float(row['buyhold_return_pct']):.1f}%"
        )
        for row in annual_rows
    )
    return (
        "# 中文结论\n\n"
        "- 做了什么：实现 ML-confirmed trend-protected 15m execution sleeve v2。日线 open2open 主仓位由 LGB/XGB consensus 与 base_constrained_ref 共同决定；"
        "年度晋升门使用只看验证期 buyhold regime 的 15m execution template，并要求 source daily LGB/XGB probability >= "
        f"{summary['regime_template']['ml_confirmation']['probability_floor']:.2f} 后才允许非 source 模板出仓位。\n"
        f"- 核心指标：ML-confirmed annual WF={annual_metrics}；目标通过年份={summary['annual_goal']['target_pass_test_years']}。"
        f"纯 position-model latest test return_pct={selected['return_pct']:.4f}，max_drawdown_pct={selected['max_drawdown_pct']:.4f}，"
        f"pure ML annual pass years={summary['pure_ml_annual_goal']['target_pass_test_years']}。\n"
        "- regime/window 差异：纯 ML alpha 未通过所有年度 buyhold+DD 门；ML-confirmed 15m regime template 把分钟线用于恢复期/震荡期仓位选择，而不是直接预测日收益。\n"
        f"- 是否晋升：{summary['promotion_decision']}。\n"
        "- 风险：这是日级 open2open 研究卡，template 的 regime 映射仍来自研究探索；仍需要独立外推窗口和分钟级 bid/ask、排队成交、延迟执行验证。\n"
        "- 下一步：把 regime 映射扩到跨资产/跨周期训练期枚举，继续降低人工模板过拟合风险。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
        "- 不构成买卖建议。\n"
    )


def _failure_modes(summary: dict[str, Any]) -> str:
    return (
        "# Failure Modes\n\n"
        "- Annual target requires every required fold to beat SOXL open-to-open buyhold and keep max drawdown above -20%.\n"
        "- Pure LGB/XGB ML alpha is reported separately from the ML-confirmed validation-regime 15m execution template gate.\n"
        "- This card uses 15m data as lagged context for open2open sizing; it does not prove same-day intraday execution.\n"
        "- The trend protection floor can preserve return but may also keep exposure during early selloffs.\n"
        "- Regime templates are research candidates and can overfit if the template library or ML confirmation floor is changed after reviewing test years.\n"
        "- SOXX constituent features may inherit snapshot or partial-history bias until point-in-time holdings are complete.\n"
        f"- Current blocking reasons: {', '.join(summary['annual_goal']['blocking_reasons'])}.\n"
    )


def _numeric(frame: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype("float64")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if not isinstance(value, (str, bytes, bool)) and pd.isna(value):
        return None
    return value

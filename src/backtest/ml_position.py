from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .position_sizing import apply_position_sizing_policy, volatility_scaled_target


CLASSIFIER_MODELS = {
    "extra_trees_classifier",
    "hist_gradient_boosting_classifier",
    "lgbm_classifier",
    "logistic_regression",
    "random_forest_classifier",
    "xgb_classifier",
}
REGRESSOR_MODELS = {
    "extra_trees_regressor",
    "hist_gradient_boosting_regressor",
    "lgbm_regressor",
    "random_forest_regressor",
    "ridge",
    "xgb_regressor",
}
CONSENSUS_MODELS = {
    "lgb_xgb_consensus",
}


def _selection_tiebreaker_sort(
    frame: pd.DataFrame,
    tiebreakers: list[str] | None,
) -> pd.DataFrame:
    if not tiebreakers:
        return frame.sort_values("selection_objective_score", ascending=False)

    sort_columns: list[str] = []
    ascending: list[bool] = []
    for item in tiebreakers:
        column, _, direction = str(item).partition(":")
        column = column.strip()
        direction = (direction or "asc").strip().lower()
        if column not in frame.columns:
            raise ValueError(f"selection tiebreaker requires missing column: {column}")
        if direction not in {"asc", "desc"}:
            raise ValueError("selection tiebreaker direction must be asc or desc")
        sort_columns.append(column)
        ascending.append(direction == "asc")

    if "candidate" not in sort_columns:
        sort_columns.append("candidate")
        ascending.append(True)
    return frame.sort_values(sort_columns, ascending=ascending)


def _build_model(model_name: str, *, random_state: int):
    if model_name == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000),
        )
    if model_name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=2.0))
    if model_name == "hist_gradient_boosting_classifier":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.03,
                random_state=random_state,
            ),
        )
    if model_name == "hist_gradient_boosting_regressor":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.03,
                random_state=random_state,
            ),
        )
    if model_name == "extra_trees_classifier":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=240,
                max_depth=6,
                min_samples_leaf=12,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    if model_name == "extra_trees_regressor":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=240,
                max_depth=6,
                min_samples_leaf=12,
                random_state=random_state,
                n_jobs=1,
            ),
        )
    if model_name == "random_forest_classifier":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=240,
                max_depth=5,
                min_samples_leaf=16,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    if model_name == "random_forest_regressor":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=240,
                max_depth=5,
                min_samples_leaf=16,
                random_state=random_state,
                n_jobs=1,
            ),
        )
    if model_name == "lgbm_classifier":
        from lightgbm import LGBMClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.2,
                random_state=random_state,
                verbose=-1,
            ),
        )
    if model_name == "lgbm_regressor":
        from lightgbm import LGBMRegressor

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.2,
                random_state=random_state,
                verbose=-1,
            ),
        )
    if model_name == "xgb_classifier":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=160,
                max_depth=3,
                learning_rate=0.035,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    if model_name == "xgb_regressor":
        from xgboost import XGBRegressor

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=160,
                max_depth=3,
                learning_rate=0.035,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    raise ValueError(f"unknown ML position model: {model_name}")


def _model_task(model_name: str, explicit_task: str | None) -> str:
    if explicit_task is not None:
        if explicit_task not in {"classification", "regression", "consensus"}:
            raise ValueError("model task must be classification, regression, or consensus")
        return explicit_task
    if model_name in CLASSIFIER_MODELS:
        return "classification"
    if model_name in REGRESSOR_MODELS:
        return "regression"
    if model_name in CONSENSUS_MODELS:
        return "consensus"
    raise ValueError(f"cannot infer ML position model task: {model_name}")


def _consensus_probability(
    probabilities: list[pd.Series],
    *,
    aggregation: str,
) -> pd.Series:
    if len(probabilities) < 2:
        raise ValueError("consensus probability requires at least two member probability series")
    frame = pd.concat([probability.rename(str(index)) for index, probability in enumerate(probabilities)], axis=1)
    if aggregation == "min":
        consensus = frame.min(axis=1)
    elif aggregation == "mean":
        consensus = frame.mean(axis=1)
    elif aggregation == "median":
        consensus = frame.median(axis=1)
    else:
        raise ValueError("consensus aggregation must be min, mean, or median")
    return consensus.clip(lower=0.0, upper=1.0).rename("probability")


def _max_drawdown_pct(strategy_returns: pd.Series) -> float:
    if strategy_returns.empty:
        return 0.0
    equity = (1.0 + strategy_returns.astype("float64")).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def _tail_risk_metrics(strategy_returns: pd.Series) -> dict[str, Any]:
    if strategy_returns.empty:
        return {
            "worst_5_daily_loss_pct": 0.0,
            "cvar_5pct_pct": 0.0,
            "underwater_day_count": 0,
            "underwater_day_ratio": 0.0,
        }
    daily = strategy_returns.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    worst_5_count = min(5, len(daily))
    cvar_count = max(int(np.ceil(len(daily) * 0.05)), 1)
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "worst_5_daily_loss_pct": float(daily.nsmallest(worst_5_count).mean() * 100.0),
        "cvar_5pct_pct": float(daily.nsmallest(cvar_count).mean() * 100.0),
        "underwater_day_count": int(drawdown.lt(-1e-12).sum()),
        "underwater_day_ratio": float(drawdown.lt(-1e-12).mean()) if len(drawdown) else 0.0,
    }


def _holding_period_stats(target_position: pd.Series) -> tuple[int, int, float]:
    deltas = target_position.astype("float64").diff().fillna(target_position.astype("float64")).to_numpy()
    trade_mask = np.abs(deltas) > 1e-9
    trade_count = int(trade_mask.sum())
    next_day_reversal_count = int(
        sum(
            1
            for previous, current in zip(deltas[:-1], deltas[1:])
            if abs(previous) > 1e-9 and abs(current) > 1e-9 and np.sign(previous) != np.sign(current)
        )
    )
    trade_indices = np.flatnonzero(trade_mask)
    avg_holding_days = (
        float(len(target_position))
        if len(trade_indices) < 2
        else float(np.diff(trade_indices).mean())
    )
    return trade_count, next_day_reversal_count, avg_holding_days


def _positive_probability(model, x: pd.DataFrame, *, train_score: pd.Series | None = None) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            positive_index = classes.index(1)
            return pd.Series(probabilities[:, positive_index], index=x.index, name="probability")
        return pd.Series(float(classes[0]) if classes else 0.0, index=x.index, name="probability")

    score = pd.Series(model.predict(x), index=x.index, name="score").astype("float64")
    if train_score is None or train_score.empty:
        center = float(score.median())
        scale = float(score.std(ddof=0) or 1.0)
    else:
        center = float(train_score.median())
        scale = float(train_score.std(ddof=0) or 1.0)
    zscore = ((score - center) / max(scale, 1e-12)).clip(-12.0, 12.0)
    return (1.0 / (1.0 + np.exp(-zscore))).rename("probability")


def _raw_target_from_probability(
    data: pd.DataFrame,
    probability: pd.Series,
    *,
    position_spec: dict[str, Any],
) -> pd.Series:
    threshold = float(position_spec.get("threshold", 0.55))
    scale = float(position_spec.get("scale", 1.0))
    mode = str(position_spec.get("position_mode", "probability"))
    probability = probability.astype("float64").reindex(data.index).fillna(0.0)
    gated_probability = ((probability - threshold) / max(1.0 - threshold, 1e-12)).clip(lower=0.0, upper=1.0)

    if mode == "probability":
        raw = gated_probability * scale
    elif mode == "binary":
        raw = (probability >= threshold).astype("float64") * scale
    elif mode == "base_gate":
        base_col = str(position_spec.get("base_position_col", "base_position"))
        if base_col not in data.columns:
            raise ValueError(f"position spec requires missing base_position_col: {base_col}")
        raw = data[base_col].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = raw * (probability >= threshold).astype("float64") * scale
    elif mode == "base_probability":
        base_col = str(position_spec.get("base_position_col", "base_position"))
        if base_col not in data.columns:
            raise ValueError(f"position spec requires missing base_position_col: {base_col}")
        raw = data[base_col].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = raw * gated_probability * scale
    elif mode == "base_plus_probability":
        base_col = str(position_spec.get("base_position_col", "base_position"))
        if base_col not in data.columns:
            raise ValueError(f"position spec requires missing base_position_col: {base_col}")
        base_cap = float(position_spec.get("base_cap", position_spec.get("max_position", 1.0)))
        base = (
            data[base_col]
            .astype("float64")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0, upper=base_cap)
        )
        base_scale = float(position_spec.get("base_scale", 1.0))
        boost_scale = float(position_spec.get("boost_scale", scale))
        raw = (base * base_scale + gated_probability * boost_scale).clip(lower=0.0)
    elif mode == "base_throttle":
        base_col = str(position_spec.get("base_position_col", "base_position"))
        if base_col not in data.columns:
            raise ValueError(f"position spec requires missing base_position_col: {base_col}")
        low_probability = float(position_spec.get("low_probability", threshold))
        high_probability = float(position_spec.get("high_probability", max(low_probability + 0.05, 0.70)))
        if high_probability <= low_probability:
            raise ValueError("base_throttle high_probability must exceed low_probability")
        min_throttle = float(position_spec.get("min_throttle", 0.25))
        max_throttle = float(position_spec.get("max_throttle", 1.0))
        if min_throttle < 0.0 or max_throttle <= 0.0 or min_throttle > max_throttle:
            raise ValueError("base_throttle requires 0 <= min_throttle <= max_throttle")

        base_cap = float(position_spec.get("base_cap", position_spec.get("max_position", 1.0)))
        base = (
            data[base_col]
            .astype("float64")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0, upper=base_cap)
        )
        unit = ((probability - low_probability) / max(high_probability - low_probability, 1e-12)).clip(
            lower=0.0,
            upper=1.0,
        )
        if str(position_spec.get("throttle_curve", "smoothstep")) == "smoothstep":
            unit = unit * unit * (3.0 - 2.0 * unit)
        throttle = min_throttle + (max_throttle - min_throttle) * unit
        off_threshold = position_spec.get("off_threshold")
        if off_threshold is not None:
            throttle = throttle.where(probability >= float(off_threshold), 0.0)

        raw = base * throttle * scale
        trend_floor_col = position_spec.get("trend_floor_col")
        if trend_floor_col is not None:
            trend_floor_col = str(trend_floor_col)
            if trend_floor_col not in data.columns:
                raise ValueError(f"position spec requires missing trend_floor_col: {trend_floor_col}")
            trend_floor_fraction = float(position_spec.get("trend_floor_fraction", min_throttle))
            trend_threshold = float(position_spec.get("trend_floor_threshold", 0.0))
            trend_active = (
                data[trend_floor_col]
                .astype("float64")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                > trend_threshold
            )
            raw = raw.where(~trend_active, np.maximum(raw, base * trend_floor_fraction))
    else:
        raise ValueError(
            "position_mode must be one of: probability, binary, base_gate, "
            "base_probability, base_plus_probability, base_throttle"
        )

    target_vol = position_spec.get("target_vol")
    if target_vol is not None:
        realized_vol_col = position_spec.get("realized_vol_col")
        if realized_vol_col is None:
            raise ValueError("position spec requires realized_vol_col when target_vol is set")
        realized_vol_col = str(realized_vol_col)
        if realized_vol_col not in data.columns:
            raise ValueError(f"position spec requires missing realized_vol_col: {realized_vol_col}")
        raw = volatility_scaled_target(
            pd.Series(raw.to_numpy(dtype="float64"), index=data.index, name="raw_target"),
            data[realized_vol_col],
            target_vol=float(target_vol),
            min_scale=float(position_spec.get("min_vol_scale", 0.0)),
            max_scale=float(position_spec.get("max_vol_scale", 1.0)),
        )

    risk_gate_col = position_spec.get("risk_gate_col")
    if risk_gate_col is not None:
        risk_gate_col = str(risk_gate_col)
        if risk_gate_col not in data.columns:
            raise ValueError(f"position spec requires missing risk_gate_col: {risk_gate_col}")
        threshold = float(position_spec.get("risk_gate_threshold", 0.0))
        direction = str(position_spec.get("risk_gate_direction", "above"))
        values = data[risk_gate_col].astype("float64").replace([np.inf, -np.inf], np.nan)
        if direction == "above":
            risk_active = values > threshold
        elif direction == "below":
            risk_active = values < threshold
        else:
            raise ValueError("risk_gate_direction must be above or below")
        cap = float(position_spec.get("risk_gate_max_position", position_spec.get("max_position", 1.0)))
        raw = pd.Series(raw.to_numpy(dtype="float64"), index=data.index, name="raw_target")
        raw = raw.where(~risk_active.fillna(False), raw.clip(upper=cap))
    return pd.Series(raw.to_numpy(dtype="float64"), index=data.index, name="raw_target")


def _strategy_metrics(
    *,
    candidate: str,
    segment: str,
    data: pd.DataFrame,
    probability: pd.Series,
    position_spec: dict[str, Any],
    return_col: str,
    cost_per_turnover: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw_target = _raw_target_from_probability(data, probability, position_spec=position_spec)
    target_position = apply_position_sizing_policy(
        raw_target,
        min_position=float(position_spec.get("min_position", 0.0)),
        max_position=float(position_spec.get("max_position", 1.0)),
        rebalance_threshold=float(position_spec.get("rebalance_threshold", 0.0)),
        max_daily_change=(
            None
            if position_spec.get("max_daily_change") is None
            else float(position_spec["max_daily_change"])
        ),
        min_hold_days=int(position_spec.get("min_hold_days", 1)),
    )
    turnover = target_position.diff().abs().fillna(target_position.abs())
    forward_return = data[return_col].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    strategy_returns = target_position * forward_return - turnover * float(cost_per_turnover)
    return_pct = float(((1.0 + strategy_returns).prod() - 1.0) * 100.0)
    max_drawdown_pct = _max_drawdown_pct(strategy_returns)
    max_daily_loss_pct = float(strategy_returns.min() * 100.0) if not strategy_returns.empty else 0.0
    tail_metrics = _tail_risk_metrics(strategy_returns)
    trade_count, next_day_reversal_count, avg_holding_days = _holding_period_stats(target_position)
    return_to_drawdown = return_pct / max(abs(max_drawdown_pct), 1.0)
    reversal_ratio = float(next_day_reversal_count / max(trade_count, 1))
    avg_abs_position = float(target_position.abs().mean())
    baseline_return_pct = np.nan
    baseline_max_drawdown_pct = np.nan
    excess_return_pct = np.nan
    excess_return_to_drawdown = np.nan
    baseline_col = position_spec.get("objective_baseline_position_col")
    if baseline_col is not None:
        baseline_col = str(baseline_col)
        if baseline_col not in data.columns:
            raise ValueError(f"position spec requires missing objective_baseline_position_col: {baseline_col}")
        baseline_target = (
            data[baseline_col]
            .astype("float64")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0)
        )
        baseline_turnover = baseline_target.diff().abs().fillna(baseline_target.abs())
        baseline_returns = baseline_target * forward_return - baseline_turnover * float(cost_per_turnover)
        baseline_return_pct = float(((1.0 + baseline_returns).prod() - 1.0) * 100.0)
        baseline_max_drawdown_pct = _max_drawdown_pct(baseline_returns)
        excess_return_pct = float(return_pct - baseline_return_pct)
        excess_return_to_drawdown = float(excess_return_pct / max(abs(max_drawdown_pct), 1.0))
    stability_penalty = (
        float(position_spec.get("turnover_penalty", 1.5)) * float(turnover.sum())
        + float(position_spec.get("reversal_penalty", 2.0)) * float(next_day_reversal_count)
        + float(position_spec.get("trade_penalty", 0.10)) * float(trade_count)
    )
    exposure_floor = float(position_spec.get("objective_min_avg_abs_position", 0.0))
    exposure_penalty = max(exposure_floor - avg_abs_position, 0.0) * float(
        position_spec.get("exposure_shortfall_penalty", 100.0)
    )
    drawdown_floor = position_spec.get("objective_max_drawdown_floor")
    drawdown_penalty = 0.0
    if drawdown_floor is not None and max_drawdown_pct < float(drawdown_floor):
        drawdown_penalty = abs(max_drawdown_pct - float(drawdown_floor)) * float(
            position_spec.get("drawdown_floor_penalty", 2.0)
        )
    daily_loss_penalty = 0.0
    daily_loss_floor = position_spec.get("objective_max_daily_loss_floor")
    if daily_loss_floor is not None and max_daily_loss_pct < float(daily_loss_floor):
        daily_loss_penalty = abs(max_daily_loss_pct - float(daily_loss_floor)) * float(
            position_spec.get("daily_loss_floor_penalty", 2.0)
        )
    cvar_penalty = 0.0
    cvar_floor = position_spec.get("objective_cvar_5pct_floor")
    if cvar_floor is not None and tail_metrics["cvar_5pct_pct"] < float(cvar_floor):
        cvar_penalty = abs(tail_metrics["cvar_5pct_pct"] - float(cvar_floor)) * float(
            position_spec.get("cvar_floor_penalty", 1.5)
        )
    worst_5_penalty = 0.0
    worst_5_floor = position_spec.get("objective_worst_5_daily_loss_floor")
    if worst_5_floor is not None and tail_metrics["worst_5_daily_loss_pct"] < float(worst_5_floor):
        worst_5_penalty = abs(tail_metrics["worst_5_daily_loss_pct"] - float(worst_5_floor)) * float(
            position_spec.get("worst_5_floor_penalty", 1.0)
        )
    underwater_penalty = 0.0
    max_underwater_ratio = position_spec.get("objective_max_underwater_day_ratio")
    if max_underwater_ratio is not None:
        underwater_penalty = max(tail_metrics["underwater_day_ratio"] - float(max_underwater_ratio), 0.0) * float(
            position_spec.get("underwater_penalty", 25.0)
        )
    selection_objective_score = float(
        return_to_drawdown * float(position_spec.get("objective_return_to_drawdown_weight", 10.0))
        + return_pct * float(position_spec.get("objective_return_weight", 0.02))
        + (0.0 if pd.isna(excess_return_pct) else excess_return_pct)
        * float(position_spec.get("objective_excess_return_weight", 0.0))
        + (0.0 if pd.isna(excess_return_to_drawdown) else excess_return_to_drawdown)
        * float(position_spec.get("objective_excess_return_to_drawdown_weight", 0.0))
        + avg_abs_position * float(position_spec.get("objective_exposure_weight", 0.0))
        - stability_penalty
        - exposure_penalty
        - drawdown_penalty
        - daily_loss_penalty
        - cvar_penalty
        - worst_5_penalty
        - underwater_penalty
    )

    metrics = {
        "candidate": candidate,
        "segment": segment,
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "max_daily_loss_pct": max_daily_loss_pct,
        "return_to_drawdown": float(return_to_drawdown),
        "selection_objective_score": selection_objective_score,
        "baseline_return_pct": baseline_return_pct,
        "baseline_max_drawdown_pct": baseline_max_drawdown_pct,
        "excess_return_pct": excess_return_pct,
        "excess_return_to_drawdown": excess_return_to_drawdown,
        **tail_metrics,
        "avg_abs_position": avg_abs_position,
        "max_position": float(target_position.max()),
        "min_position": float(target_position.min()),
        "days_above_100": int((target_position > 1.0 + 1e-9).sum()),
        "days_below_0": int((target_position < -1e-9).sum()),
        "total_turnover": float(turnover.sum()),
        "trade_count": int(trade_count),
        "next_day_reversal_count": int(next_day_reversal_count),
        "next_day_reversal_ratio": reversal_ratio,
        "avg_holding_days": float(avg_holding_days),
        "cost_per_turnover": float(cost_per_turnover),
    }
    positions = pd.DataFrame(
        {
            "datetime": pd.to_datetime(data["datetime"]).dt.normalize().to_numpy(),
            "candidate": candidate,
            "segment": segment,
            "probability": probability.reindex(data.index).to_numpy(dtype="float64"),
            "raw_target": raw_target.to_numpy(dtype="float64"),
            "target_position": target_position.to_numpy(dtype="float64"),
            "turnover": turnover.to_numpy(dtype="float64"),
            "strategy_return": strategy_returns.to_numpy(dtype="float64"),
        }
    )
    return metrics, positions


def _candidate_name(model_name: str, position_spec: dict[str, Any]) -> str:
    mode = str(position_spec.get("position_mode", "probability"))
    threshold = float(position_spec.get("threshold", 0.55))
    scale = float(position_spec.get("scale", 1.0))
    rebalance = float(position_spec.get("rebalance_threshold", 0.0))
    hold = int(position_spec.get("min_hold_days", 1))
    base = position_spec.get("base_position_col")
    base_suffix = "" if base is None else f"_{base}"
    risk_suffix = ""
    if position_spec.get("risk_gate_col") is not None:
        risk_col = str(position_spec["risk_gate_col"])
        risk_threshold = float(position_spec.get("risk_gate_threshold", 0.0))
        risk_cap = float(position_spec.get("risk_gate_max_position", position_spec.get("max_position", 1.0)))
        risk_direction = str(position_spec.get("risk_gate_direction", "above"))
        risk_suffix = f"_rg{risk_col}_{risk_direction}{risk_threshold:g}_cap{risk_cap:g}"
    if mode == "base_throttle":
        low = float(position_spec.get("low_probability", threshold))
        high = float(position_spec.get("high_probability", max(low + 0.05, 0.70)))
        min_throttle = float(position_spec.get("min_throttle", 0.25))
        max_throttle = float(position_spec.get("max_throttle", 1.0))
        return (
            f"{model_name}_{mode}{base_suffix}_p{low:g}-{high:g}_"
            f"t{min_throttle:g}-{max_throttle:g}_s{scale:g}_h{rebalance:g}_hold{hold}{risk_suffix}"
        )
    if mode == "base_plus_probability":
        base_scale = float(position_spec.get("base_scale", 1.0))
        boost_scale = float(position_spec.get("boost_scale", scale))
        return (
            f"{model_name}_{mode}{base_suffix}_p{threshold:g}_"
            f"b{base_scale:g}_boost{boost_scale:g}_h{rebalance:g}_hold{hold}{risk_suffix}"
        )
    return f"{model_name}_{mode}{base_suffix}_p{threshold:g}_s{scale:g}_h{rebalance:g}_hold{hold}{risk_suffix}"


def evaluate_ml_position_candidates(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    model_specs: list[dict[str, Any]],
    train_window: tuple[str, str],
    validation_window: tuple[str, str],
    test_window: tuple[str, str],
    position_specs: list[dict[str, Any]] | None = None,
    return_col: str = "forward_return",
    label_col: str | None = None,
    datetime_col: str = "datetime",
    cost_per_turnover: float = 0.0003,
    min_validation_trade_count: int = 0,
    min_validation_avg_abs_position: float = 0.0,
    validation_max_drawdown_floor: float | None = None,
    validation_max_daily_loss_floor: float | None = None,
    max_validation_total_turnover: float | None = None,
    max_validation_reversal_ratio: float | None = None,
    validation_cvar_5pct_floor: float | None = None,
    validation_worst_5_daily_loss_floor: float | None = None,
    max_validation_underwater_day_ratio: float | None = None,
    selection_score_tolerance: float = 0.0,
    selection_tiebreakers: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train ML scores on train data, select position candidates on validation, and report test stability."""

    if frame.empty:
        raise ValueError("ML position frame is empty")
    if not feature_cols:
        raise ValueError("at least one feature column is required")
    if not model_specs:
        raise ValueError("at least one model spec is required")
    position_specs = position_specs or [{"position_mode": "probability", "threshold": 0.55, "scale": 1.0}]
    model_target_col = label_col or return_col

    required = {datetime_col, return_col, model_target_col, *feature_cols}
    for spec in position_specs:
        if str(spec.get("position_mode", "probability")) in {
            "base_gate",
            "base_probability",
            "base_plus_probability",
            "base_throttle",
        }:
            required.add(str(spec.get("base_position_col", "base_position")))
        if spec.get("trend_floor_col") is not None:
            required.add(str(spec["trend_floor_col"]))
        if spec.get("realized_vol_col") is not None:
            required.add(str(spec["realized_vol_col"]))
        if spec.get("risk_gate_col") is not None:
            required.add(str(spec["risk_gate_col"]))
        if spec.get("objective_baseline_position_col") is not None:
            required.add(str(spec["objective_baseline_position_col"]))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ML position frame missing columns: {missing}")

    data = frame.copy()
    data["datetime"] = pd.to_datetime(data[datetime_col]).dt.normalize()
    data = data.sort_values("datetime").reset_index(drop=True)
    for column in [*feature_cols, return_col, model_target_col]:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[return_col]).reset_index(drop=True)

    def segment(window: tuple[str, str]) -> pd.DataFrame:
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        return data.loc[(data["datetime"] >= start) & (data["datetime"] <= end)].copy()

    train = segment(train_window)
    validation = segment(validation_window)
    test = segment(test_window)
    train = train.dropna(subset=[model_target_col]).reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise ValueError("train, validation, and test windows must all contain rows")

    x_train = train[feature_cols].astype("float64")
    metric_rows: list[dict[str, Any]] = []
    position_frames: list[pd.DataFrame] = []
    probability_cache: dict[tuple[str, str], pd.Series] = {}
    consensus_specs: list[dict[str, Any]] = []

    for index, spec in enumerate(model_specs):
        model_name = str(spec["model"])
        task = _model_task(model_name, spec.get("task"))
        if task == "consensus":
            consensus_specs.append(spec)
            continue
        positive_threshold = float(spec.get("positive_threshold", 0.0))
        random_state = int(spec.get("random_state", 42 + index))
        if task == "classification":
            y_train = train[model_target_col].gt(positive_threshold).astype(int)
            if y_train.nunique() < 2:
                model = DummyClassifier(strategy="constant", constant=int(y_train.iloc[0] if len(y_train) else 0))
            else:
                model = _build_model(model_name, random_state=random_state)
            model.fit(x_train, y_train)
            train_score = None
        else:
            model = _build_model(model_name, random_state=random_state)
            y_train = train[model_target_col].astype("float64")
            model.fit(x_train, y_train)
            train_score = pd.Series(model.predict(x_train), index=train.index, name="score")

        for segment_name, segment_frame in [("validation", validation), ("test", test)]:
            x_segment = segment_frame[feature_cols].astype("float64")
            probability = _positive_probability(model, x_segment, train_score=train_score)
            probability_cache[(model_name, segment_name)] = probability
            for position_spec in position_specs:
                candidate = _candidate_name(model_name, position_spec)
                metrics, positions = _strategy_metrics(
                    candidate=candidate,
                    segment=segment_name,
                    data=segment_frame,
                    probability=probability,
                    position_spec=position_spec,
                    return_col=return_col,
                    cost_per_turnover=cost_per_turnover,
                )
                metrics.update(
                    {
                        "model": model_name,
                        "model_task": task,
                        "model_target_col": model_target_col,
                        "return_col": return_col,
                        "position_mode": str(position_spec.get("position_mode", "probability")),
                        "selection_disabled": bool(position_spec.get("selection_disabled", False))
                        or bool(spec.get("selection_disabled", False)),
                        "threshold": float(position_spec.get("threshold", 0.55)),
                        "scale": float(position_spec.get("scale", 1.0)),
                        "selection_segment": "validation",
                        "report_segment": "test",
                        "train_start": pd.Timestamp(train["datetime"].min()).date().isoformat(),
                        "train_end": pd.Timestamp(train["datetime"].max()).date().isoformat(),
                        "validation_start": pd.Timestamp(validation["datetime"].min()).date().isoformat(),
                        "validation_end": pd.Timestamp(validation["datetime"].max()).date().isoformat(),
                        "test_start": pd.Timestamp(test["datetime"].min()).date().isoformat(),
                        "test_end": pd.Timestamp(test["datetime"].max()).date().isoformat(),
                        "params": json.dumps(position_spec, sort_keys=True),
                    }
                )
                metric_rows.append(metrics)
                position_frames.append(positions)

    for spec in consensus_specs:
        model_name = str(spec["model"])
        members = [str(member) for member in spec.get("members", [])]
        if len(members) < 2:
            raise ValueError("consensus model spec requires at least two members")
        aggregation = str(spec.get("aggregation", "min"))
        for segment_name, segment_frame in [("validation", validation), ("test", test)]:
            missing_members = [member for member in members if (member, segment_name) not in probability_cache]
            if missing_members:
                raise ValueError(
                    f"consensus model {model_name} missing member probabilities for {segment_name}: {missing_members}"
                )
            probability = _consensus_probability(
                [probability_cache[(member, segment_name)] for member in members],
                aggregation=aggregation,
            )
            for position_spec in position_specs:
                candidate = _candidate_name(model_name, position_spec)
                consensus_position_spec = {
                    **position_spec,
                    "consensus_members": members,
                    "consensus_aggregation": aggregation,
                }
                metrics, positions = _strategy_metrics(
                    candidate=candidate,
                    segment=segment_name,
                    data=segment_frame,
                    probability=probability,
                    position_spec=consensus_position_spec,
                    return_col=return_col,
                    cost_per_turnover=cost_per_turnover,
                )
                metrics.update(
                    {
                        "model": model_name,
                        "model_task": "consensus",
                        "model_target_col": model_target_col,
                        "return_col": return_col,
                        "position_mode": str(position_spec.get("position_mode", "probability")),
                        "selection_disabled": bool(position_spec.get("selection_disabled", False))
                        or bool(spec.get("selection_disabled", False)),
                        "threshold": float(position_spec.get("threshold", 0.55)),
                        "scale": float(position_spec.get("scale", 1.0)),
                        "selection_segment": "validation",
                        "report_segment": "test",
                        "train_start": pd.Timestamp(train["datetime"].min()).date().isoformat(),
                        "train_end": pd.Timestamp(train["datetime"].max()).date().isoformat(),
                        "validation_start": pd.Timestamp(validation["datetime"].min()).date().isoformat(),
                        "validation_end": pd.Timestamp(validation["datetime"].max()).date().isoformat(),
                        "test_start": pd.Timestamp(test["datetime"].min()).date().isoformat(),
                        "test_end": pd.Timestamp(test["datetime"].max()).date().isoformat(),
                        "params": json.dumps(consensus_position_spec, sort_keys=True),
                    }
                )
                metric_rows.append(metrics)
                position_frames.append(positions)

    metrics_frame = pd.DataFrame(metric_rows)
    if metrics_frame.empty:
        raise ValueError("ML position evaluation produced no metrics")
    validation_metrics = metrics_frame.loc[metrics_frame["segment"] == "validation"].copy()
    eligible_mask = (
        validation_metrics["trade_count"].astype(float) >= float(min_validation_trade_count)
    ) & (
        validation_metrics["avg_abs_position"].astype(float) >= float(min_validation_avg_abs_position)
    ) & (
        ~validation_metrics["selection_disabled"].astype(bool)
    )
    if validation_max_drawdown_floor is not None:
        eligible_mask = eligible_mask & (
            validation_metrics["max_drawdown_pct"].astype(float) >= float(validation_max_drawdown_floor)
        )
    if validation_max_daily_loss_floor is not None:
        eligible_mask = eligible_mask & (
            validation_metrics["max_daily_loss_pct"].astype(float) >= float(validation_max_daily_loss_floor)
        )
    if max_validation_total_turnover is not None:
        eligible_mask = eligible_mask & (
            validation_metrics["total_turnover"].astype(float) <= float(max_validation_total_turnover)
        )
    if max_validation_reversal_ratio is not None:
        eligible_mask = eligible_mask & (
            validation_metrics["next_day_reversal_ratio"].astype(float) <= float(max_validation_reversal_ratio)
        )
    if validation_cvar_5pct_floor is not None:
        eligible_mask = eligible_mask & (
            validation_metrics["cvar_5pct_pct"].astype(float) >= float(validation_cvar_5pct_floor)
        )
    if validation_worst_5_daily_loss_floor is not None:
        eligible_mask = eligible_mask & (
            validation_metrics["worst_5_daily_loss_pct"].astype(float) >= float(validation_worst_5_daily_loss_floor)
        )
    if max_validation_underwater_day_ratio is not None:
        eligible_mask = eligible_mask & (
            validation_metrics["underwater_day_ratio"].astype(float) <= float(max_validation_underwater_day_ratio)
        )
    eligible_validation = validation_metrics.loc[eligible_mask]
    selectable_validation = validation_metrics.loc[~validation_metrics["selection_disabled"].astype(bool)]
    if not eligible_validation.empty:
        selection_pool = eligible_validation
        selection_pool_reason = "eligible_validation"
    elif not selectable_validation.empty:
        selection_pool = selectable_validation
        selection_pool_reason = "selectable_validation_fallback"
    else:
        selection_pool = validation_metrics
        selection_pool_reason = "all_validation_fallback"
    best_selection_score = float(selection_pool["selection_objective_score"].max())
    selection_tolerance = max(float(selection_score_tolerance), 0.0)
    near_tie_pool = selection_pool.loc[
        selection_pool["selection_objective_score"].astype(float) >= best_selection_score - selection_tolerance
    ].copy()
    selected_candidate = str(_selection_tiebreaker_sort(near_tie_pool, selection_tiebreakers).iloc[0]["candidate"])
    eligible_candidates = set(eligible_validation["candidate"].astype(str))
    metrics_frame["selection_eligible"] = metrics_frame["candidate"].astype(str).isin(eligible_candidates)
    metrics_frame["selection_eligible"] = metrics_frame["selection_eligible"].map(bool).astype(object)
    metrics_frame["selection_tiebreaker_pool_size"] = int(len(near_tie_pool))
    metrics_frame["selection_gate"] = json.dumps(
        {
            "min_validation_trade_count": int(min_validation_trade_count),
            "min_validation_avg_abs_position": float(min_validation_avg_abs_position),
            "validation_max_drawdown_floor": validation_max_drawdown_floor,
            "validation_max_daily_loss_floor": validation_max_daily_loss_floor,
            "max_validation_total_turnover": max_validation_total_turnover,
            "max_validation_reversal_ratio": max_validation_reversal_ratio,
            "validation_cvar_5pct_floor": validation_cvar_5pct_floor,
            "validation_worst_5_daily_loss_floor": validation_worst_5_daily_loss_floor,
            "max_validation_underwater_day_ratio": max_validation_underwater_day_ratio,
            "selection_score_tolerance": float(selection_tolerance),
            "selection_tiebreakers": list(selection_tiebreakers or []),
            "selection_tiebreaker_pool_size": int(len(near_tie_pool)),
            "selection_disabled_candidate_count": int(validation_metrics["selection_disabled"].astype(bool).sum()),
            "selection_pool_reason": selection_pool_reason,
            "fallback_to_selectable_candidates": bool(eligible_validation.empty and not selectable_validation.empty),
            "fallback_to_all_candidates": bool(eligible_validation.empty and selectable_validation.empty),
        },
        sort_keys=True,
    )
    metrics_frame["selected"] = metrics_frame["candidate"].eq(selected_candidate)
    metrics_frame["selected"] = metrics_frame["selected"].map(bool).astype(object)
    sort_cols = ["selected", "segment", "selection_objective_score", "return_pct"]
    metrics_frame = metrics_frame.sort_values(sort_cols, ascending=[False, True, False, False]).reset_index(drop=True)
    positions_frame = pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame()
    positions_frame["selected"] = positions_frame["candidate"].eq(selected_candidate)
    return metrics_frame, positions_frame


def predict_ml_position_candidate(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    model_spec: dict[str, Any],
    train_window: tuple[str, str],
    score_window: tuple[str, str],
    position_spec: dict[str, Any],
    return_col: str = "forward_return",
    label_col: str | None = None,
    datetime_col: str = "datetime",
    cost_per_turnover: float = 0.0003,
) -> pd.DataFrame:
    """Fit one ML position model and emit target positions for a score window."""

    if frame.empty:
        raise ValueError("ML position frame is empty")
    if not feature_cols:
        raise ValueError("at least one feature column is required")
    model_target_col = label_col or return_col
    required = {datetime_col, return_col, model_target_col, *feature_cols}
    if str(position_spec.get("position_mode", "probability")) in {
        "base_gate",
        "base_probability",
        "base_plus_probability",
        "base_throttle",
    }:
        required.add(str(position_spec.get("base_position_col", "base_position")))
    if position_spec.get("trend_floor_col") is not None:
        required.add(str(position_spec["trend_floor_col"]))
    if position_spec.get("realized_vol_col") is not None:
        required.add(str(position_spec["realized_vol_col"]))
    if position_spec.get("risk_gate_col") is not None:
        required.add(str(position_spec["risk_gate_col"]))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ML position frame missing columns: {missing}")

    data = frame.copy()
    data["datetime"] = pd.to_datetime(data[datetime_col]).dt.normalize()
    data = data.sort_values("datetime").reset_index(drop=True)
    for column in [*feature_cols, return_col, model_target_col]:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[return_col]).reset_index(drop=True)

    def segment(window: tuple[str, str]) -> pd.DataFrame:
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        return data.loc[(data["datetime"] >= start) & (data["datetime"] <= end)].copy()

    train = segment(train_window).dropna(subset=[model_target_col]).reset_index(drop=True)
    score = segment(score_window).reset_index(drop=True)
    if train.empty or score.empty:
        raise ValueError("train and score windows must both contain rows")

    model_name = str(model_spec["model"])
    task = _model_task(model_name, model_spec.get("task"))
    random_state = int(model_spec.get("random_state", 42))
    x_train = train[feature_cols].astype("float64")
    if task == "classification":
        positive_threshold = float(model_spec.get("positive_threshold", 0.0))
        y_train = train[model_target_col].gt(positive_threshold).astype(int)
        if y_train.nunique() < 2:
            model = DummyClassifier(strategy="constant", constant=int(y_train.iloc[0] if len(y_train) else 0))
        else:
            model = _build_model(model_name, random_state=random_state)
        model.fit(x_train, y_train)
        train_score = None
    else:
        model = _build_model(model_name, random_state=random_state)
        y_train = train[model_target_col].astype("float64")
        model.fit(x_train, y_train)
        train_score = pd.Series(model.predict(x_train), index=train.index, name="score")

    probability = _positive_probability(model, score[feature_cols].astype("float64"), train_score=train_score)
    candidate = _candidate_name(model_name, position_spec)
    _, positions = _strategy_metrics(
        candidate=candidate,
        segment="score",
        data=score,
        probability=probability,
        position_spec=position_spec,
        return_col=return_col,
        cost_per_turnover=cost_per_turnover,
    )
    positions["model"] = model_name
    positions["model_task"] = task
    positions["model_target_col"] = model_target_col
    positions["return_col"] = return_col
    return positions

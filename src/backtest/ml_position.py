from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .position_sizing import apply_position_sizing_policy


CLASSIFIER_MODELS = {"logistic_regression", "hist_gradient_boosting_classifier", "lgbm_classifier"}
REGRESSOR_MODELS = {"ridge", "hist_gradient_boosting_regressor", "lgbm_regressor"}


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
    raise ValueError(f"unknown ML position model: {model_name}")


def _model_task(model_name: str, explicit_task: str | None) -> str:
    if explicit_task is not None:
        if explicit_task not in {"classification", "regression"}:
            raise ValueError("model task must be classification or regression")
        return explicit_task
    if model_name in CLASSIFIER_MODELS:
        return "classification"
    if model_name in REGRESSOR_MODELS:
        return "regression"
    raise ValueError(f"cannot infer ML position model task: {model_name}")


def _max_drawdown_pct(strategy_returns: pd.Series) -> float:
    if strategy_returns.empty:
        return 0.0
    equity = (1.0 + strategy_returns.astype("float64")).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


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
    else:
        raise ValueError("position_mode must be one of: probability, binary, base_gate, base_probability")
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
    trade_count, next_day_reversal_count, avg_holding_days = _holding_period_stats(target_position)
    return_to_drawdown = return_pct / max(abs(max_drawdown_pct), 1.0)
    stability_penalty = (
        1.5 * float(turnover.sum())
        + 2.0 * float(next_day_reversal_count)
        + 0.10 * float(trade_count)
    )
    selection_objective_score = float(return_to_drawdown * 10.0 + return_pct * 0.02 - stability_penalty)

    metrics = {
        "candidate": candidate,
        "segment": segment,
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "return_to_drawdown": float(return_to_drawdown),
        "selection_objective_score": selection_objective_score,
        "avg_abs_position": float(target_position.abs().mean()),
        "max_position": float(target_position.max()),
        "min_position": float(target_position.min()),
        "days_above_100": int((target_position > 1.0 + 1e-9).sum()),
        "days_below_0": int((target_position < -1e-9).sum()),
        "total_turnover": float(turnover.sum()),
        "trade_count": int(trade_count),
        "next_day_reversal_count": int(next_day_reversal_count),
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
    return f"{model_name}_{mode}{base_suffix}_p{threshold:g}_s{scale:g}_h{rebalance:g}_hold{hold}"


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
    datetime_col: str = "datetime",
    cost_per_turnover: float = 0.0003,
    min_validation_trade_count: int = 0,
    min_validation_avg_abs_position: float = 0.0,
    validation_max_drawdown_floor: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train ML scores on train data, select position candidates on validation, and report test stability."""

    if frame.empty:
        raise ValueError("ML position frame is empty")
    if not feature_cols:
        raise ValueError("at least one feature column is required")
    if not model_specs:
        raise ValueError("at least one model spec is required")
    position_specs = position_specs or [{"position_mode": "probability", "threshold": 0.55, "scale": 1.0}]

    required = {datetime_col, return_col, *feature_cols}
    for spec in position_specs:
        if str(spec.get("position_mode", "probability")) in {"base_gate", "base_probability"}:
            required.add(str(spec.get("base_position_col", "base_position")))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ML position frame missing columns: {missing}")

    data = frame.copy()
    data["datetime"] = pd.to_datetime(data[datetime_col]).dt.normalize()
    data = data.sort_values("datetime").reset_index(drop=True)
    for column in [*feature_cols, return_col]:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[return_col]).reset_index(drop=True)

    def segment(window: tuple[str, str]) -> pd.DataFrame:
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        return data.loc[(data["datetime"] >= start) & (data["datetime"] <= end)].copy()

    train = segment(train_window)
    validation = segment(validation_window)
    test = segment(test_window)
    if train.empty or validation.empty or test.empty:
        raise ValueError("train, validation, and test windows must all contain rows")

    x_train = train[feature_cols].astype("float64")
    metric_rows: list[dict[str, Any]] = []
    position_frames: list[pd.DataFrame] = []

    for index, spec in enumerate(model_specs):
        model_name = str(spec["model"])
        task = _model_task(model_name, spec.get("task"))
        positive_threshold = float(spec.get("positive_threshold", 0.0))
        random_state = int(spec.get("random_state", 42 + index))
        if task == "classification":
            y_train = train[return_col].gt(positive_threshold).astype(int)
            if y_train.nunique() < 2:
                model = DummyClassifier(strategy="constant", constant=int(y_train.iloc[0] if len(y_train) else 0))
            else:
                model = _build_model(model_name, random_state=random_state)
            model.fit(x_train, y_train)
            train_score = None
        else:
            model = _build_model(model_name, random_state=random_state)
            y_train = train[return_col].astype("float64")
            model.fit(x_train, y_train)
            train_score = pd.Series(model.predict(x_train), index=train.index, name="score")

        for segment_name, segment_frame in [("validation", validation), ("test", test)]:
            x_segment = segment_frame[feature_cols].astype("float64")
            probability = _positive_probability(model, x_segment, train_score=train_score)
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
                        "position_mode": str(position_spec.get("position_mode", "probability")),
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

    metrics_frame = pd.DataFrame(metric_rows)
    if metrics_frame.empty:
        raise ValueError("ML position evaluation produced no metrics")
    validation_metrics = metrics_frame.loc[metrics_frame["segment"] == "validation"].copy()
    eligible_mask = (
        validation_metrics["trade_count"].astype(float) >= float(min_validation_trade_count)
    ) & (
        validation_metrics["avg_abs_position"].astype(float) >= float(min_validation_avg_abs_position)
    )
    if validation_max_drawdown_floor is not None:
        eligible_mask = eligible_mask & (
            validation_metrics["max_drawdown_pct"].astype(float) >= float(validation_max_drawdown_floor)
        )
    eligible_validation = validation_metrics.loc[eligible_mask]
    selection_pool = eligible_validation if not eligible_validation.empty else validation_metrics
    selected_candidate = str(selection_pool.sort_values("selection_objective_score", ascending=False).iloc[0]["candidate"])
    eligible_candidates = set(eligible_validation["candidate"].astype(str))
    metrics_frame["selection_eligible"] = metrics_frame["candidate"].astype(str).isin(eligible_candidates)
    metrics_frame["selection_eligible"] = metrics_frame["selection_eligible"].map(bool).astype(object)
    metrics_frame["selection_gate"] = json.dumps(
        {
            "min_validation_trade_count": int(min_validation_trade_count),
            "min_validation_avg_abs_position": float(min_validation_avg_abs_position),
            "validation_max_drawdown_floor": validation_max_drawdown_floor,
            "fallback_to_all_candidates": bool(eligible_validation.empty),
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

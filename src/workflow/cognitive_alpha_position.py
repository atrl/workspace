from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest import evaluate_ml_position_candidates


@dataclass(frozen=True)
class CognitiveAlphaPositionSpec:
    feature_cols: list[str]
    train: tuple[str, str]
    validation: tuple[str, str]
    test: tuple[str, str]
    model_specs: list[dict[str, Any]]
    position_specs: list[dict[str, Any]] | None = None
    return_col: str = "forward_return"
    label_col: str = "label_return_5d"
    datetime_col: str = "datetime"
    cost_per_turnover: float = 0.0003
    min_validation_trade_count: int = 0
    min_validation_avg_abs_position: float = 0.0
    validation_max_drawdown_floor: float | None = None
    validation_max_daily_loss_floor: float | None = None
    max_validation_total_turnover: float | None = None
    max_validation_extreme_loss_day_count: int | None = None
    max_validation_reversal_ratio: float | None = None
    test_max_drawdown_floor: float | None = None
    test_max_daily_loss_floor: float | None = None
    max_test_total_turnover: float | None = None
    max_test_extreme_loss_day_count: int | None = None
    extreme_loss_threshold: float = -0.05
    constrained_reference_name: str = "constrained_weight_reference"
    walk_forward_folds: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class CognitiveAlphaPositionResult:
    metrics: pd.DataFrame
    positions: pd.DataFrame
    walk_forward_metrics: pd.DataFrame
    benchmark_comparison: pd.DataFrame
    summary: dict[str, Any]
    latest_signal: dict[str, Any]

    def write_sidecars(
        self,
        directory: Path,
        *,
        prefix: str = "cognitive_alpha_position",
        include_positions: bool = False,
    ) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "model_selection": directory / f"{prefix}_model_selection.csv",
            "benchmark_comparison": directory / f"{prefix}_benchmark_comparison.csv",
            "summary": directory / f"{prefix}_summary.json",
            "latest_signal": directory / f"{prefix}_latest_signal.json",
        }
        self.metrics.to_csv(paths["model_selection"], index=False)
        self.benchmark_comparison.to_csv(paths["benchmark_comparison"], index=False)
        if not self.walk_forward_metrics.empty:
            paths["walk_forward_metrics"] = directory / f"{prefix}_walk_forward_metrics.csv"
            self.walk_forward_metrics.to_csv(paths["walk_forward_metrics"], index=False)
        paths["summary"].write_text(json.dumps(self.summary, indent=2, sort_keys=True), encoding="utf-8")
        paths["latest_signal"].write_text(json.dumps(self.latest_signal, indent=2, sort_keys=True), encoding="utf-8")
        if include_positions:
            paths["positions"] = directory / f"{prefix}_positions.csv"
            self.positions.to_csv(paths["positions"], index=False)
        return paths


def run_cognitive_alpha_position_model(
    frame: pd.DataFrame,
    *,
    spec: CognitiveAlphaPositionSpec,
    benchmarks: list[dict[str, Any]],
) -> CognitiveAlphaPositionResult:
    _validate_position_frame(frame, spec)
    metrics, positions = evaluate_ml_position_candidates(
        frame,
        feature_cols=spec.feature_cols,
        model_specs=spec.model_specs,
        train_window=spec.train,
        validation_window=spec.validation,
        test_window=spec.test,
        position_specs=spec.position_specs,
        return_col=spec.return_col,
        label_col=spec.label_col,
        datetime_col=spec.datetime_col,
        cost_per_turnover=spec.cost_per_turnover,
        min_validation_trade_count=spec.min_validation_trade_count,
        min_validation_avg_abs_position=spec.min_validation_avg_abs_position,
        validation_max_drawdown_floor=spec.validation_max_drawdown_floor,
        validation_max_daily_loss_floor=spec.validation_max_daily_loss_floor,
        max_validation_total_turnover=spec.max_validation_total_turnover,
        max_validation_reversal_ratio=spec.max_validation_reversal_ratio,
    )
    selected = _selected_payload(metrics)
    _attach_selected_tail_metrics(selected, positions, extreme_loss_threshold=spec.extreme_loss_threshold)
    comparison = _benchmark_comparison(selected["test"], benchmarks=benchmarks)
    comparison_summary = _comparison_summary(selected["test"], comparison, constrained_name=spec.constrained_reference_name)
    risk_gates = _risk_gate_summary(selected, spec)
    walk_forward_metrics = _run_walk_forward_metrics(frame, spec)
    walk_forward_summary = _walk_forward_summary(walk_forward_metrics, extreme_loss_threshold=spec.extreme_loss_threshold)
    latest_signal = _latest_signal(positions)
    summary = {
        "workflow": "cognitive_alpha_position_model",
        "model_families": sorted({str(item["model"]) for item in spec.model_specs}),
        "feature_count": int(len(spec.feature_cols)),
        "feature_cols": list(spec.feature_cols),
        "segments": {
            "train": list(spec.train),
            "validation": list(spec.validation),
            "test": list(spec.test),
        },
        "selected_candidate": selected,
        "comparison": comparison_summary,
        "risk_gates": risk_gates,
        "walk_forward": walk_forward_summary,
        "latest_signal": latest_signal,
        "promotion_decision": _promotion_decision(
            comparison_summary,
            constrained_name=spec.constrained_reference_name,
            risk_gates=risk_gates,
            walk_forward=walk_forward_summary,
        ),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "recorder_only_artifacts": ["positions"],
        },
        "spec": asdict(spec),
    }
    return CognitiveAlphaPositionResult(
        metrics=metrics,
        positions=positions,
        walk_forward_metrics=walk_forward_metrics,
        benchmark_comparison=comparison,
        summary=summary,
        latest_signal=latest_signal,
    )


def _validate_position_frame(frame: pd.DataFrame, spec: CognitiveAlphaPositionSpec) -> None:
    if frame.empty:
        raise ValueError("cognitive alpha position frame is empty")
    required = {spec.datetime_col, spec.return_col, spec.label_col, *spec.feature_cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing cognitive alpha position columns: {missing}")
    if not spec.feature_cols:
        raise ValueError("at least one cognitive alpha feature is required")
    if not spec.model_specs:
        raise ValueError("at least one cognitive alpha position model is required")


def _selected_payload(metrics: pd.DataFrame) -> dict[str, Any]:
    selected_rows = metrics.loc[metrics["selected"].astype(bool)].copy()
    if selected_rows.empty:
        raise ValueError("ML position model did not select a candidate")
    payload: dict[str, Any] = {"selected_by": "validation_selection_objective_score"}
    for segment in ["validation", "test"]:
        rows = selected_rows.loc[selected_rows["segment"] == segment]
        if rows.empty:
            continue
        row = rows.iloc[0].to_dict()
        payload[segment] = _jsonable(row)
    payload["candidate"] = str(selected_rows.iloc[0]["candidate"])
    payload["model"] = str(selected_rows.iloc[0]["model"])
    return payload


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            out[key] = None
        elif isinstance(value, (bool, str, int, float)):
            out[key] = value
        elif hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = str(value)
    return out


def _benchmark_comparison(test_row: dict[str, Any], *, benchmarks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "benchmark": "cognitive_alpha_selected",
            "return_pct": float(test_row["return_pct"]),
            "max_drawdown_pct": float(test_row["max_drawdown_pct"]),
            "trade_count": int(test_row.get("trade_count", 0) or 0),
            "source": "cognitive_alpha_position_model",
        }
    ]
    for benchmark in benchmarks:
        name = str(benchmark["name"])
        rows.append(
            {
                "benchmark": name,
                "return_pct": float(benchmark.get("return_pct", 0.0)),
                "max_drawdown_pct": float(benchmark.get("max_drawdown_pct", 0.0)),
                "trade_count": int(benchmark.get("trade_count", 0) or 0),
                "source": str(benchmark.get("source", name)),
            }
        )
    selected_return = rows[0]["return_pct"]
    selected_drawdown = rows[0]["max_drawdown_pct"]
    for row in rows:
        row["selected_return_gap_pct"] = float(selected_return - row["return_pct"])
        row["selected_max_drawdown_gap_pct"] = float(selected_drawdown - row["max_drawdown_pct"])
        row["selected_beats_return"] = bool(selected_return > row["return_pct"])
    return pd.DataFrame(rows)


def _comparison_summary(
    test_row: dict[str, Any],
    comparison: pd.DataFrame,
    *,
    constrained_name: str,
) -> dict[str, Any]:
    summary = {
        "selected_return_pct": float(test_row["return_pct"]),
        "selected_max_drawdown_pct": float(test_row["max_drawdown_pct"]),
    }
    benchmark_rows = comparison.loc[comparison["benchmark"] != "cognitive_alpha_selected"]
    for _, row in benchmark_rows.iterrows():
        name = str(row["benchmark"])
        summary[f"beats_{name}"] = bool(float(test_row["return_pct"]) > float(row["return_pct"]))
        summary[f"selected_vs_{name}_return_gap_pct"] = float(test_row["return_pct"]) - float(row["return_pct"])
    if f"beats_{constrained_name}" not in summary:
        summary[f"beats_{constrained_name}"] = False
    return summary


def _segment_passes_risk_gates(
    row: dict[str, Any],
    *,
    max_drawdown_floor: float | None,
    max_daily_loss_floor: float | None,
    max_total_turnover: float | None,
    max_extreme_loss_day_count: int | None,
) -> bool:
    if max_drawdown_floor is not None and float(row.get("max_drawdown_pct", 0.0)) < float(max_drawdown_floor):
        return False
    if max_daily_loss_floor is not None and float(row.get("max_daily_loss_pct", 0.0)) < float(max_daily_loss_floor):
        return False
    if max_total_turnover is not None and float(row.get("total_turnover", 0.0)) > float(max_total_turnover):
        return False
    if (
        max_extreme_loss_day_count is not None
        and int(row.get("extreme_loss_day_count", 0) or 0) > int(max_extreme_loss_day_count)
    ):
        return False
    return True


def _risk_gate_summary(selected: dict[str, Any], spec: CognitiveAlphaPositionSpec) -> dict[str, Any]:
    validation_passes = _segment_passes_risk_gates(
        selected.get("validation", {}),
        max_drawdown_floor=spec.validation_max_drawdown_floor,
        max_daily_loss_floor=spec.validation_max_daily_loss_floor,
        max_total_turnover=spec.max_validation_total_turnover,
        max_extreme_loss_day_count=spec.max_validation_extreme_loss_day_count,
    )
    test_passes = _segment_passes_risk_gates(
        selected.get("test", {}),
        max_drawdown_floor=spec.test_max_drawdown_floor,
        max_daily_loss_floor=spec.test_max_daily_loss_floor,
        max_total_turnover=spec.max_test_total_turnover,
        max_extreme_loss_day_count=spec.max_test_extreme_loss_day_count,
    )
    return {
        "validation_max_drawdown_floor": spec.validation_max_drawdown_floor,
        "validation_max_daily_loss_floor": spec.validation_max_daily_loss_floor,
        "max_validation_total_turnover": spec.max_validation_total_turnover,
        "max_validation_extreme_loss_day_count": spec.max_validation_extreme_loss_day_count,
        "test_max_drawdown_floor": spec.test_max_drawdown_floor,
        "test_max_daily_loss_floor": spec.test_max_daily_loss_floor,
        "max_test_total_turnover": spec.max_test_total_turnover,
        "max_test_extreme_loss_day_count": spec.max_test_extreme_loss_day_count,
        "extreme_loss_threshold": spec.extreme_loss_threshold,
        "selected_validation_passes": bool(validation_passes),
        "selected_test_passes": bool(test_passes),
        "selected_passes_all": bool(validation_passes and test_passes),
    }


def _position_tail_metrics(positions: pd.DataFrame, *, extreme_loss_threshold: float) -> dict[str, Any]:
    if positions.empty:
        return {
            "zero_position_day_count": 0,
            "zero_position_day_ratio": 0.0,
            "month_count": 0,
            "positive_month_count": 0,
            "positive_month_ratio": 0.0,
            "extreme_loss_threshold": float(extreme_loss_threshold),
            "extreme_loss_day_count": 0,
            "worst_strategy_return_pct": 0.0,
        }
    data = positions.copy()
    data["datetime"] = pd.to_datetime(data["datetime"]).dt.normalize()
    strategy_returns = data["strategy_return"].astype("float64")
    target_position = data["target_position"].astype("float64")
    monthly = strategy_returns.groupby(data["datetime"].dt.to_period("M")).apply(lambda values: (1.0 + values).prod() - 1.0)
    month_count = int(len(monthly))
    positive_month_count = int(monthly.gt(0.0).sum())
    return {
        "zero_position_day_count": int(target_position.abs().le(1e-9).sum()),
        "zero_position_day_ratio": float(target_position.abs().le(1e-9).mean()) if len(target_position) else 0.0,
        "month_count": month_count,
        "positive_month_count": positive_month_count,
        "positive_month_ratio": float(positive_month_count / max(month_count, 1)),
        "extreme_loss_threshold": float(extreme_loss_threshold),
        "extreme_loss_day_count": int(strategy_returns.le(float(extreme_loss_threshold)).sum()),
        "worst_strategy_return_pct": float(strategy_returns.min() * 100.0) if not strategy_returns.empty else 0.0,
    }


def _selected_segment_positions(positions: pd.DataFrame, segment: str) -> pd.DataFrame:
    if positions.empty:
        return positions.copy()
    selected = positions.loc[
        positions["selected"].astype(bool) & positions["segment"].astype(str).eq(segment)
    ].copy()
    return selected.sort_values("datetime").reset_index(drop=True)


def _attach_selected_tail_metrics(
    selected: dict[str, Any],
    positions: pd.DataFrame,
    *,
    extreme_loss_threshold: float,
) -> None:
    for segment in ["validation", "test"]:
        if segment in selected:
            selected[segment].update(
                _position_tail_metrics(
                    _selected_segment_positions(positions, segment),
                    extreme_loss_threshold=extreme_loss_threshold,
                )
            )


def _fold_window(fold: dict[str, Any], key: str) -> tuple[str, str]:
    value = fold.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"walk-forward fold {fold.get('fold', '<unknown>')} missing {key} window")
    return str(value[0]), str(value[1])


def _run_walk_forward_metrics(frame: pd.DataFrame, spec: CognitiveAlphaPositionSpec) -> pd.DataFrame:
    folds = spec.walk_forward_folds or []
    if not folds:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for index, fold in enumerate(folds, start=1):
        fold_name = str(fold.get("fold") or f"walk_forward_{index}")
        train = _fold_window(fold, "train")
        validation = _fold_window(fold, "validation")
        test = _fold_window(fold, "test")
        metrics, positions = evaluate_ml_position_candidates(
            frame,
            feature_cols=spec.feature_cols,
            model_specs=spec.model_specs,
            train_window=train,
            validation_window=validation,
            test_window=test,
            position_specs=spec.position_specs,
            return_col=spec.return_col,
            label_col=spec.label_col,
            datetime_col=spec.datetime_col,
            cost_per_turnover=spec.cost_per_turnover,
            min_validation_trade_count=spec.min_validation_trade_count,
            min_validation_avg_abs_position=spec.min_validation_avg_abs_position,
            validation_max_drawdown_floor=spec.validation_max_drawdown_floor,
            validation_max_daily_loss_floor=spec.validation_max_daily_loss_floor,
            max_validation_total_turnover=spec.max_validation_total_turnover,
            max_validation_reversal_ratio=spec.max_validation_reversal_ratio,
        )
        selected = _selected_payload(metrics)
        _attach_selected_tail_metrics(selected, positions, extreme_loss_threshold=spec.extreme_loss_threshold)
        validation_row = selected.get("validation", {})
        test_row = selected.get("test", {})
        validation_passes = _segment_passes_risk_gates(
            validation_row,
            max_drawdown_floor=spec.validation_max_drawdown_floor,
            max_daily_loss_floor=spec.validation_max_daily_loss_floor,
            max_total_turnover=spec.max_validation_total_turnover,
            max_extreme_loss_day_count=spec.max_validation_extreme_loss_day_count,
        )
        test_passes = _segment_passes_risk_gates(
            test_row,
            max_drawdown_floor=spec.test_max_drawdown_floor,
            max_daily_loss_floor=spec.test_max_daily_loss_floor,
            max_total_turnover=spec.max_test_total_turnover,
            max_extreme_loss_day_count=spec.max_test_extreme_loss_day_count,
        )
        rows.append(
            {
                "fold": fold_name,
                "segment": "test",
                "train_start": train[0],
                "train_end": train[1],
                "validation_start": validation[0],
                "validation_end": validation[1],
                "test_start": test[0],
                "test_end": test[1],
                "selected_candidate": str(selected.get("candidate", "")),
                "selected_model": str(selected.get("model", "")),
                "validation_return_pct": float(validation_row.get("return_pct", 0.0)),
                "validation_max_drawdown_pct": float(validation_row.get("max_drawdown_pct", 0.0)),
                "validation_max_daily_loss_pct": float(validation_row.get("max_daily_loss_pct", 0.0)),
                "validation_total_turnover": float(validation_row.get("total_turnover", 0.0)),
                "validation_trade_count": int(validation_row.get("trade_count", 0) or 0),
                "validation_positive_month_ratio": float(validation_row.get("positive_month_ratio", 0.0)),
                "validation_zero_position_day_ratio": float(validation_row.get("zero_position_day_ratio", 0.0)),
                "validation_extreme_loss_day_count": int(validation_row.get("extreme_loss_day_count", 0) or 0),
                "test_return_pct": float(test_row.get("return_pct", 0.0)),
                "test_max_drawdown_pct": float(test_row.get("max_drawdown_pct", 0.0)),
                "test_max_daily_loss_pct": float(test_row.get("max_daily_loss_pct", 0.0)),
                "test_total_turnover": float(test_row.get("total_turnover", 0.0)),
                "test_trade_count": int(test_row.get("trade_count", 0) or 0),
                "test_avg_abs_position": float(test_row.get("avg_abs_position", 0.0)),
                "test_max_position": float(test_row.get("max_position", 0.0)),
                "test_days_above_100": int(test_row.get("days_above_100", 0) or 0),
                "test_month_count": int(test_row.get("month_count", 0) or 0),
                "test_positive_month_count": int(test_row.get("positive_month_count", 0) or 0),
                "test_positive_month_ratio": float(test_row.get("positive_month_ratio", 0.0)),
                "test_zero_position_day_count": int(test_row.get("zero_position_day_count", 0) or 0),
                "test_zero_position_day_ratio": float(test_row.get("zero_position_day_ratio", 0.0)),
                "test_extreme_loss_threshold": float(test_row.get("extreme_loss_threshold", spec.extreme_loss_threshold)),
                "test_extreme_loss_day_count": int(test_row.get("extreme_loss_day_count", 0) or 0),
                "test_worst_strategy_return_pct": float(test_row.get("worst_strategy_return_pct", 0.0)),
                "validation_risk_gate_pass": bool(validation_passes),
                "test_risk_gate_pass": bool(test_passes),
                "risk_gate_pass": bool(validation_passes and test_passes),
            }
        )
    return pd.DataFrame(rows)


def _walk_forward_summary(metrics: pd.DataFrame, *, extreme_loss_threshold: float) -> dict[str, Any]:
    if metrics.empty:
        return {
            "fold_count": 0,
            "positive_test_fold_count": 0,
            "positive_test_fold_ratio": None,
            "mean_test_positive_month_ratio": None,
            "mean_test_zero_position_day_ratio": None,
            "max_test_extreme_loss_day_count": None,
            "extreme_loss_threshold": float(extreme_loss_threshold),
            "mean_test_return_pct": None,
            "median_test_return_pct": None,
            "worst_test_max_drawdown_pct": None,
            "worst_test_max_daily_loss_pct": None,
            "max_test_total_turnover": None,
            "total_test_trade_count": 0,
            "mean_test_avg_abs_position": None,
            "all_folds_pass_risk_gates": None,
        }
    returns = metrics["test_return_pct"].astype(float)
    fold_count = int(len(metrics))
    positive_count = int(returns.gt(0.0).sum())
    return {
        "fold_count": fold_count,
        "positive_test_fold_count": positive_count,
        "positive_test_fold_ratio": float(positive_count / fold_count),
        "mean_test_positive_month_ratio": float(metrics["test_positive_month_ratio"].astype(float).mean()),
        "mean_test_zero_position_day_ratio": float(metrics["test_zero_position_day_ratio"].astype(float).mean()),
        "max_test_extreme_loss_day_count": int(metrics["test_extreme_loss_day_count"].astype(int).max()),
        "extreme_loss_threshold": float(extreme_loss_threshold),
        "mean_test_return_pct": float(returns.mean()),
        "median_test_return_pct": float(returns.median()),
        "worst_test_max_drawdown_pct": float(metrics["test_max_drawdown_pct"].astype(float).min()),
        "worst_test_max_daily_loss_pct": float(metrics["test_max_daily_loss_pct"].astype(float).min()),
        "max_test_total_turnover": float(metrics["test_total_turnover"].astype(float).max()),
        "total_test_trade_count": int(metrics["test_trade_count"].astype(int).sum()),
        "mean_test_avg_abs_position": float(metrics["test_avg_abs_position"].astype(float).mean()),
        "all_folds_pass_risk_gates": bool(metrics["risk_gate_pass"].astype(bool).all()),
    }


def _promotion_decision(
    summary: dict[str, Any],
    *,
    constrained_name: str,
    risk_gates: dict[str, Any],
    walk_forward: dict[str, Any],
) -> str:
    if risk_gates.get("selected_passes_all") is False:
        return "not_promoted_failed_position_risk_gates"
    if walk_forward.get("all_folds_pass_risk_gates") is False:
        return "not_promoted_failed_walk_forward_risk_gates"
    if bool(summary.get(f"beats_{constrained_name}", False)):
        return "candidate_for_validated_research"
    return f"not_promoted_under_{constrained_name}"


def _latest_signal(positions: pd.DataFrame) -> dict[str, Any]:
    selected = positions.loc[positions["selected"].astype(bool) & (positions["segment"] == "test")].copy()
    if selected.empty:
        raise ValueError("selected test positions are required for latest signal")
    selected["datetime"] = pd.to_datetime(selected["datetime"]).dt.normalize()
    selected = selected.sort_values("datetime").reset_index(drop=True)
    latest = selected.iloc[-1]
    previous_position = float(selected.iloc[-2]["target_position"]) if len(selected) > 1 else 0.0
    target_position = float(latest["target_position"])
    delta = target_position - previous_position
    if abs(delta) <= 1e-9:
        action = "HOLD"
    elif delta > 0:
        action = "BUY"
    else:
        action = "SELL"
    changes = selected.loc[selected["target_position"].astype("float64").diff().abs().fillna(selected["target_position"].abs()) > 1e-9]
    last_change = changes.iloc[-1] if not changes.empty else latest
    return {
        "as_of": pd.Timestamp(latest["datetime"]).date().isoformat(),
        "candidate": str(latest["candidate"]),
        "action": action,
        "target_position": target_position,
        "target_percent": target_position * 100.0,
        "previous_target_position": previous_position,
        "target_delta": delta,
        "target_delta_percent": delta * 100.0,
        "probability": float(latest.get("probability", 0.0)),
        "last_change_date": pd.Timestamp(last_change["datetime"]).date().isoformat(),
        "last_change_target_position": float(last_change["target_position"]),
    }

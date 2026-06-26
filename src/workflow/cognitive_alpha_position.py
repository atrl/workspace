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
    max_validation_reversal_ratio: float | None = None
    test_max_drawdown_floor: float | None = None
    test_max_daily_loss_floor: float | None = None
    max_test_total_turnover: float | None = None
    constrained_reference_name: str = "constrained_weight_reference"


@dataclass(frozen=True)
class CognitiveAlphaPositionResult:
    metrics: pd.DataFrame
    positions: pd.DataFrame
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
    comparison = _benchmark_comparison(selected["test"], benchmarks=benchmarks)
    comparison_summary = _comparison_summary(selected["test"], comparison, constrained_name=spec.constrained_reference_name)
    risk_gates = _risk_gate_summary(selected, spec)
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
        "latest_signal": latest_signal,
        "promotion_decision": _promotion_decision(
            comparison_summary,
            constrained_name=spec.constrained_reference_name,
            risk_gates=risk_gates,
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
) -> bool:
    if max_drawdown_floor is not None and float(row.get("max_drawdown_pct", 0.0)) < float(max_drawdown_floor):
        return False
    if max_daily_loss_floor is not None and float(row.get("max_daily_loss_pct", 0.0)) < float(max_daily_loss_floor):
        return False
    if max_total_turnover is not None and float(row.get("total_turnover", 0.0)) > float(max_total_turnover):
        return False
    return True


def _risk_gate_summary(selected: dict[str, Any], spec: CognitiveAlphaPositionSpec) -> dict[str, Any]:
    validation_passes = _segment_passes_risk_gates(
        selected.get("validation", {}),
        max_drawdown_floor=spec.validation_max_drawdown_floor,
        max_daily_loss_floor=spec.validation_max_daily_loss_floor,
        max_total_turnover=spec.max_validation_total_turnover,
    )
    test_passes = _segment_passes_risk_gates(
        selected.get("test", {}),
        max_drawdown_floor=spec.test_max_drawdown_floor,
        max_daily_loss_floor=spec.test_max_daily_loss_floor,
        max_total_turnover=spec.max_test_total_turnover,
    )
    return {
        "validation_max_drawdown_floor": spec.validation_max_drawdown_floor,
        "validation_max_daily_loss_floor": spec.validation_max_daily_loss_floor,
        "max_validation_total_turnover": spec.max_validation_total_turnover,
        "test_max_drawdown_floor": spec.test_max_drawdown_floor,
        "test_max_daily_loss_floor": spec.test_max_daily_loss_floor,
        "max_test_total_turnover": spec.max_test_total_turnover,
        "selected_validation_passes": bool(validation_passes),
        "selected_test_passes": bool(test_passes),
        "selected_passes_all": bool(validation_passes and test_passes),
    }


def _promotion_decision(
    summary: dict[str, Any],
    *,
    constrained_name: str,
    risk_gates: dict[str, Any],
) -> str:
    if risk_gates.get("selected_passes_all") is False:
        return "not_promoted_failed_position_risk_gates"
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

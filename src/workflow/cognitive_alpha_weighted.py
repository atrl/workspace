from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest import evaluate_weighted_position_candidates


@dataclass(frozen=True)
class CognitiveAlphaWeightedFactorSearchSpec:
    component_cols: list[str]
    candidates: list[dict[str, Any]]
    return_col: str = "forward_return"
    datetime_col: str = "datetime"
    cost_per_turnover: float = 0.0003
    min_effective_weight: float = 0.05
    min_effective_component_count: int = 2
    max_position: float = 1.0
    min_position: float = 0.0
    max_drawdown_floor: float | None = None
    constrained_reference_name: str = "constrained_weight_reference"


@dataclass(frozen=True)
class CognitiveAlphaWeightedFactorSearchResult:
    metrics: pd.DataFrame
    positions: pd.DataFrame
    benchmark_comparison: pd.DataFrame
    summary: dict[str, Any]

    def write_sidecars(self, directory: Path, *, prefix: str = "cognitive_alpha_weighted_factor") -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "candidate_metrics": directory / f"{prefix}_candidate_metrics.csv",
            "benchmark_comparison": directory / f"{prefix}_benchmark_comparison.csv",
            "summary": directory / f"{prefix}_summary.json",
        }
        self.metrics.to_csv(paths["candidate_metrics"], index=False)
        self.benchmark_comparison.to_csv(paths["benchmark_comparison"], index=False)
        paths["summary"].write_text(json.dumps(self.summary, indent=2, sort_keys=True), encoding="utf-8")
        return paths


def run_cognitive_alpha_weighted_factor_search(
    frame: pd.DataFrame,
    *,
    spec: CognitiveAlphaWeightedFactorSearchSpec,
    benchmarks: list[dict[str, Any]],
) -> CognitiveAlphaWeightedFactorSearchResult:
    _validate_weighted_frame(frame, spec)
    metrics, positions = evaluate_weighted_position_candidates(
        frame,
        candidates=spec.candidates,
        component_cols=spec.component_cols,
        return_col=spec.return_col,
        datetime_col=spec.datetime_col,
        cost_per_turnover=spec.cost_per_turnover,
        min_effective_weight=spec.min_effective_weight,
    )
    metrics = _add_weighted_search_gates(metrics, spec=spec, benchmarks=benchmarks)
    selected = _select_weighted_candidate(metrics)
    comparison = _benchmark_comparison(selected, benchmarks=benchmarks)
    comparison_summary = _comparison_summary(selected, comparison, constrained_name=spec.constrained_reference_name)
    summary = {
        "workflow": "cognitive_alpha_weighted_factor_search",
        "selection": "return_sorted_long_only_multifactor_gate",
        "component_cols": list(spec.component_cols),
        "candidate_count": int(len(metrics)),
        "selected_candidate": _jsonable(selected),
        "comparison": comparison_summary,
        "promotion_decision": _promotion_decision(comparison_summary, constrained_name=spec.constrained_reference_name),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "recorder_only_artifacts": ["positions"],
        },
        "spec": asdict(spec),
    }
    return CognitiveAlphaWeightedFactorSearchResult(
        metrics=metrics,
        positions=positions,
        benchmark_comparison=comparison,
        summary=summary,
    )


def _validate_weighted_frame(frame: pd.DataFrame, spec: CognitiveAlphaWeightedFactorSearchSpec) -> None:
    if frame.empty:
        raise ValueError("cognitive alpha weighted factor frame is empty")
    if not spec.component_cols:
        raise ValueError("at least one cognitive alpha weighted component is required")
    if not spec.candidates:
        raise ValueError("at least one cognitive alpha weighted candidate is required")
    required = {spec.return_col, *spec.component_cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing cognitive alpha weighted factor columns: {missing}")


def _add_weighted_search_gates(
    metrics: pd.DataFrame,
    *,
    spec: CognitiveAlphaWeightedFactorSearchSpec,
    benchmarks: list[dict[str, Any]],
) -> pd.DataFrame:
    out = metrics.copy()
    constrained = next((item for item in benchmarks if item["name"] == spec.constrained_reference_name), None)
    constrained_return = float(constrained["return_pct"]) if constrained is not None else 0.0
    out["passes_position_bounds"] = (out["max_position"] <= float(spec.max_position) + 1e-9) & (
        out["min_position"] >= float(spec.min_position) - 1e-9
    )
    out["passes_effective_component_gate"] = out["effective_component_count"] >= int(spec.min_effective_component_count)
    out["beats_constrained_weight_reference"] = out["return_pct"] > constrained_return
    if spec.max_drawdown_floor is None:
        out["passes_drawdown_gate"] = True
    else:
        out["passes_drawdown_gate"] = out["max_drawdown_pct"] >= float(spec.max_drawdown_floor)
    out["selection_eligible"] = (
        out["passes_position_bounds"]
        & out["passes_effective_component_gate"]
        & out["beats_constrained_weight_reference"]
        & out["passes_drawdown_gate"]
    )
    return out.sort_values(["selection_eligible", "return_pct"], ascending=[False, False]).reset_index(drop=True)


def _select_weighted_candidate(metrics: pd.DataFrame) -> dict[str, Any]:
    eligible = metrics.loc[metrics["selection_eligible"].astype(bool)].copy()
    selected = eligible.iloc[0] if not eligible.empty else metrics.sort_values("return_pct", ascending=False).iloc[0]
    return selected.to_dict()


def _benchmark_comparison(selected: dict[str, Any], *, benchmarks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "benchmark": "cognitive_alpha_weighted_selected",
            "return_pct": float(selected["return_pct"]),
            "max_drawdown_pct": float(selected["max_drawdown_pct"]),
            "trade_count": int(selected.get("trade_count", 0) or 0),
            "source": "cognitive_alpha_weighted_factor_search",
        }
    ]
    for benchmark in benchmarks:
        rows.append(
            {
                "benchmark": str(benchmark["name"]),
                "return_pct": float(benchmark.get("return_pct", 0.0)),
                "max_drawdown_pct": float(benchmark.get("max_drawdown_pct", 0.0)),
                "trade_count": int(benchmark.get("trade_count", 0) or 0),
                "source": str(benchmark.get("source", benchmark["name"])),
            }
        )
    selected_return = rows[0]["return_pct"]
    for row in rows:
        row["selected_return_gap_pct"] = float(selected_return - row["return_pct"])
        row["selected_beats_return"] = bool(selected_return > row["return_pct"])
    return pd.DataFrame(rows)


def _comparison_summary(
    selected: dict[str, Any],
    comparison: pd.DataFrame,
    *,
    constrained_name: str,
) -> dict[str, Any]:
    summary = {
        "selected_return_pct": float(selected["return_pct"]),
        "selected_max_drawdown_pct": float(selected["max_drawdown_pct"]),
    }
    for _, row in comparison.loc[comparison["benchmark"] != "cognitive_alpha_weighted_selected"].iterrows():
        name = str(row["benchmark"])
        summary[f"beats_{name}"] = bool(float(selected["return_pct"]) > float(row["return_pct"]))
        summary[f"selected_vs_{name}_return_gap_pct"] = float(selected["return_pct"]) - float(row["return_pct"])
    if f"beats_{constrained_name}" not in summary:
        summary[f"beats_{constrained_name}"] = False
    return summary


def _promotion_decision(summary: dict[str, Any], *, constrained_name: str) -> str:
    if bool(summary.get(f"beats_{constrained_name}", False)):
        return "diagnostic_direct_factor_candidate"
    return f"not_promoted_under_{constrained_name}"


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

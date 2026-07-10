from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .cognitive_alpha import CognitiveAlphaCandidate, CognitiveAlphaReport

if TYPE_CHECKING:
    from src.workflow.cognitive_alpha_evolution import (
        CognitiveAlphaCandidateProvider,
        CognitiveAlphaEvolutionResult,
        CognitiveAlphaEvolutionSpec,
    )


COGNITIVE_ALPHA_FACTOR_MINING_WORKFLOW = "cognitive_alpha_factor_mining"
COGNITIVE_ALPHA_FACTOR_MINING_STAGE_ORDER = [
    "candidate_generation",
    "quality_checker",
    "fitness_evaluation",
    "evolution_loop",
    "factor_library",
]


@dataclass(frozen=True)
class CognitiveAlphaFactorMiningSpec:
    evolution_spec: "CognitiveAlphaEvolutionSpec"
    label_col: str
    alpha158_columns: tuple[str, ...] = ()
    max_factor_library_size: int | None = None
    paper_reference: str = "arxiv:2511.18850"
    source_layer: str = "src.factor"


@dataclass(frozen=True)
class CognitiveAlphaFactorMiningResult:
    run_dir: Path
    summary: dict[str, Any]
    factor_library: list[dict[str, Any]]
    selected_feature_names: list[str]
    selected_candidates: list[CognitiveAlphaCandidate]
    final_report: CognitiveAlphaReport
    generation_metrics: pd.DataFrame
    evolution_result: "CognitiveAlphaEvolutionResult"

    def write_manifest(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary, indent=2, sort_keys=True), encoding="utf-8")
        return path


def run_cognitive_alpha_factor_mining(
    frame: pd.DataFrame,
    *,
    spec: CognitiveAlphaFactorMiningSpec,
    initial_candidates: list[CognitiveAlphaCandidate] | None = None,
    candidate_provider: "CognitiveAlphaCandidateProvider | None" = None,
) -> CognitiveAlphaFactorMiningResult:
    from src.workflow.cognitive_alpha_evolution import run_cognitive_alpha_evolution

    evolution = run_cognitive_alpha_evolution(
        frame,
        spec=spec.evolution_spec,
        label_col=spec.label_col,
        initial_candidates=initial_candidates,
        alpha158_columns=list(spec.alpha158_columns),
        candidate_provider=candidate_provider,
    )
    selected_candidates, selected_metrics = _select_factor_library_archive(evolution, spec=spec)
    factor_library = _factor_library_from_candidates(
        selected_candidates,
        metrics=selected_metrics,
        source_layer=spec.source_layer,
    )
    summary = _factor_mining_summary(spec=spec, evolution=evolution, factor_library=factor_library)
    return CognitiveAlphaFactorMiningResult(
        run_dir=evolution.run_dir,
        summary=summary,
        factor_library=factor_library,
        selected_feature_names=[candidate.name for candidate in selected_candidates],
        selected_candidates=list(selected_candidates),
        final_report=evolution.final_report,
        generation_metrics=evolution.generation_metrics,
        evolution_result=evolution,
    )


def _factor_mining_summary(
    *,
    spec: CognitiveAlphaFactorMiningSpec,
    evolution: "CognitiveAlphaEvolutionResult",
    factor_library: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "workflow": COGNITIVE_ALPHA_FACTOR_MINING_WORKFLOW,
        "source_layer": spec.source_layer,
        "paper_reference": spec.paper_reference,
        "stage_order": list(COGNITIVE_ALPHA_FACTOR_MINING_STAGE_ORDER),
        "position_model_included": False,
        "position_signal_included": False,
        "label_col": spec.label_col,
        "alpha158_columns": list(spec.alpha158_columns),
        "evolution": evolution.summary,
        "factor_library": {
            "selected_count": len(factor_library),
            "selected_feature_names": [item["name"] for item in factor_library],
            "artifact_role": "alpha_factor_library",
            "archive_policy": {
                "candidate_scope": "all_generations",
                "prefer_non_fallback": True,
                "dedupe_key": "candidate_code_or_name",
                "max_factor_library_size": _max_factor_library_size(spec),
            },
            "fallback_selected_count": int(
                sum(item.get("metrics", {}).get("redundancy_status") == "selected_fallback" for item in factor_library)
            ),
            "selected_generation_counts": _value_counts(factor_library, metrics_key="evolution_generation"),
            "selected_tier_counts": _value_counts(factor_library, metrics_key="selection_tier"),
            "selected_redundancy_counts": _value_counts(factor_library, metrics_key="redundancy_status"),
        },
    }


def _select_factor_library_archive(
    evolution: "CognitiveAlphaEvolutionResult",
    *,
    spec: CognitiveAlphaFactorMiningSpec,
) -> tuple[list[CognitiveAlphaCandidate], pd.DataFrame]:
    metrics = evolution.generation_metrics.copy()
    if metrics.empty:
        return list(evolution.selected_candidates), evolution.final_report.metrics.copy()
    if "factor" not in metrics.columns:
        raise ValueError("cognitive alpha generation metrics missing factor column")

    for column in ["ic", "rank_ic", "icir", "rank_icir"]:
        if column in metrics.columns:
            metrics[f"abs_{column}"] = pd.to_numeric(metrics[column], errors="coerce").abs().fillna(0.0)

    pass_metrics = metrics.loc[metrics["quality_status"] == "pass"].copy()
    if pass_metrics.empty:
        return list(evolution.selected_candidates), evolution.final_report.metrics.copy()

    selected_non_fallback = pass_metrics.loc[pass_metrics["redundancy_status"] == "selected"].copy()
    tiered = selected_non_fallback.loc[
        selected_non_fallback["selection_tier"].isin(["elite", "qualified"])
    ].copy()
    selected_rows = tiered if not tiered.empty else selected_non_fallback
    if selected_rows.empty:
        selected_rows = pass_metrics.loc[pass_metrics["redundancy_status"] == "selected_fallback"].copy()

    sort_cols = [
        column
        for column in [
            "selection_score",
            "min_metric_percentile",
            "abs_ic",
            "abs_rank_ic",
            "abs_icir",
            "abs_rank_icir",
            "mutual_information",
        ]
        if column in selected_rows.columns
    ]
    selected_rows = selected_rows.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    selected: list[CognitiveAlphaCandidate] = []
    metric_rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_signatures: set[str] = set()
    max_size = _max_factor_library_size(spec)
    for row in selected_rows.to_dict(orient="records"):
        name = str(row["factor"])
        if name in seen_names:
            continue
        candidate = evolution.candidate_map.get(name)
        if candidate is None:
            continue
        signature = _candidate_dedupe_signature(candidate)
        if signature in seen_signatures:
            continue
        selected.append(candidate)
        metric_rows.append(row)
        seen_names.add(name)
        seen_signatures.add(signature)
        if len(selected) >= max_size:
            break

    if not selected:
        selected = list(evolution.selected_candidates)
        metric_rows = [
            row
            for row in pass_metrics.to_dict(orient="records")
            if str(row.get("factor", "")) in {candidate.name for candidate in selected}
        ]

    return selected, pd.DataFrame(metric_rows)


def _factor_library_from_candidates(
    candidates: list[CognitiveAlphaCandidate],
    *,
    metrics: pd.DataFrame,
    source_layer: str,
) -> list[dict[str, Any]]:
    metrics_by_factor = _metrics_by_factor(metrics)
    library = []
    for candidate in candidates:
        library.append(
            {
                "name": candidate.name,
                "level": candidate.level,
                "agent": candidate.agent,
                "theme": candidate.theme,
                "required_columns": list(candidate.required_columns),
                "lookback": int(candidate.lookback),
                "code": candidate.code,
                "economic_rationale": candidate.economic_rationale,
                "declared_lookahead": int(candidate.declared_lookahead),
                "metadata": dict(candidate.metadata),
                "metrics": metrics_by_factor.get(candidate.name, {}),
                "source_layer": source_layer,
                "artifact_role": "alpha_factor",
            }
        )
    return library


def _max_factor_library_size(spec: CognitiveAlphaFactorMiningSpec) -> int:
    if spec.max_factor_library_size is not None:
        return max(int(spec.max_factor_library_size), 1)
    return max(int(spec.evolution_spec.fallback_top_n or spec.evolution_spec.parent_pool_size), 1)


def _candidate_dedupe_signature(candidate: CognitiveAlphaCandidate) -> str:
    code = str(candidate.code or "").strip()
    if code:
        return code
    return "::".join(
        [
            candidate.level,
            candidate.agent,
            candidate.theme,
            ",".join(candidate.required_columns),
            str(candidate.lookback),
            candidate.name,
        ]
    )


def _value_counts(factor_library: list[dict[str, Any]], *, metrics_key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in factor_library:
        value = item.get("metrics", {}).get(metrics_key)
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _metrics_by_factor(metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if metrics.empty or "factor" not in metrics.columns:
        return {}
    metric_rows = {}
    for row in metrics.to_dict(orient="records"):
        factor = str(row.get("factor", ""))
        if factor:
            metric_rows[factor] = {key: _jsonable_metric(value) for key, value in row.items() if key != "factor"}
    return metric_rows


def _jsonable_metric(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value

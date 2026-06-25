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
    factor_library = _factor_library_from_evolution(evolution, source_layer=spec.source_layer)
    summary = _factor_mining_summary(spec=spec, evolution=evolution, factor_library=factor_library)
    return CognitiveAlphaFactorMiningResult(
        run_dir=evolution.run_dir,
        summary=summary,
        factor_library=factor_library,
        selected_feature_names=list(evolution.selected_feature_names),
        selected_candidates=list(evolution.selected_candidates),
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
        },
    }


def _factor_library_from_evolution(
    evolution: "CognitiveAlphaEvolutionResult",
    *,
    source_layer: str,
) -> list[dict[str, Any]]:
    metrics_by_factor = _metrics_by_factor(evolution.final_report.metrics)
    library = []
    for candidate in evolution.selected_candidates:
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

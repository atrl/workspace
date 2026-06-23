from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core import stable_hash


AGENTIC_ALPHA_STAGE_ORDER = ["research", "retrieve", "generate", "evaluate", "validate", "distill"]
AGENTIC_ALPHA_DECISION_LABELS = ["Promote", "Iterate", "Drop"]
AGENTIC_ALPHA_SOURCE_URL = "https://github.com/JacobDu/agentic-alpha"
AGENTIC_ALPHA_SOURCE_TYPES = {"github", "paper", "twitter", "arxiv"}


@dataclass(frozen=True)
class AgenticAlphaWorkflowSpec:
    name: str
    route: str
    market: str
    provider_uri: Path
    recorder_uri: Path
    source_url: str = AGENTIC_ALPHA_SOURCE_URL


@dataclass(frozen=True)
class AgenticAlphaKnowledgeSource:
    source_type: str
    title: str
    url: str
    extracted_signal: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source_type not in AGENTIC_ALPHA_SOURCE_TYPES:
            raise ValueError(f"source_type must be one of: {sorted(AGENTIC_ALPHA_SOURCE_TYPES)}")
        if not self.url:
            raise ValueError("knowledge source requires url")


@dataclass(frozen=True)
class AgenticAlphaHypothesis:
    hypothesis_id: str
    stage: str
    theme: str
    candidate_logic: str
    source_types: tuple[str, ...]
    source_urls: tuple[str, ...]


@dataclass(frozen=True)
class AgenticAlphaBacktestMetric:
    name: str
    return_pct: float
    max_drawdown_pct: float
    benchmark_return_pct: float
    recorder_id: str


@dataclass(frozen=True)
class AgenticAlphaSearchCandidate:
    name: str
    hypothesis_id: str
    source_urls: tuple[str, ...]
    metric: AgenticAlphaBacktestMetric


@dataclass(frozen=True)
class AgenticAlphaSearchReport:
    workflow: str
    candidate_metrics: pd.DataFrame
    summary: dict[str, Any]

    def write_sidecars(self, directory: Path) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "candidate_metrics": directory / "agentic_alpha_candidate_metrics.csv",
            "summary": directory / "agentic_alpha_search_summary.json",
        }
        self.candidate_metrics.to_csv(paths["candidate_metrics"], index=False)
        paths["summary"].write_text(json.dumps(self.summary, indent=2, sort_keys=True), encoding="utf-8")
        return paths


def build_agentic_alpha_workflow_manifest(spec: AgenticAlphaWorkflowSpec) -> dict[str, Any]:
    route = spec.route.lower()
    if route not in {"research", "sfa", "mfa"}:
        raise ValueError("route must be one of: research, sfa, mfa")

    return {
        "workflow": spec.name,
        "route": route,
        "source_url": spec.source_url,
        "market": spec.market,
        "provider_uri": Path(spec.provider_uri).as_posix(),
        "recorder_uri": Path(spec.recorder_uri).as_posix(),
        "stage_order": AGENTIC_ALPHA_STAGE_ORDER,
        "decision_labels": AGENTIC_ALPHA_DECISION_LABELS,
        "qlib_native_required": True,
        "qlib_contract": {
            "data": ["qlib.init", "qlib.data.D.features", "DatasetH", "DataHandlerLP"],
            "workflow": ["qlib.workflow.R.start", "SignalRecord", "SigAnaRecord", "PortAnaRecord"],
        },
        "src_boundary": {
            "business_logic_surface": "src",
            "skill_scripts_imported": False,
            "notebook_surface": "research_or_production_card",
            "external_repo_role": "workflow_reference",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "recorder_only_artifacts": [
                "predictions",
                "signals",
                "portfolio_report",
                "positions",
                "all_model_predictions",
            ],
        },
    }


def _source_types(sources: list[AgenticAlphaKnowledgeSource]) -> tuple[str, ...]:
    return tuple(sorted({source.source_type for source in sources}))


def _source_urls(sources: list[AgenticAlphaKnowledgeSource]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(source.url for source in sources))


def _hypothesis_id(theme: str, sources: list[AgenticAlphaKnowledgeSource]) -> str:
    return stable_hash({"theme": theme, "source_urls": _source_urls(sources)})


def build_agentic_alpha_hypotheses(sources: list[AgenticAlphaKnowledgeSource]) -> list[AgenticAlphaHypothesis]:
    if not sources:
        raise ValueError("at least one external knowledge source is required")

    source_types = _source_types(sources)
    source_urls = _source_urls(sources)
    return [
        AgenticAlphaHypothesis(
            hypothesis_id=_hypothesis_id("volatility_targeted_trend", sources),
            stage="generate",
            theme="volatility_targeted_trend",
            candidate_logic=(
                "Use external LETF evidence to combine target volatility, trend persistence, "
                "and high-volatility de-risking before holding SOXL exposure."
            ),
            source_types=source_types,
            source_urls=source_urls,
        ),
        AgenticAlphaHypothesis(
            hypothesis_id=_hypothesis_id("autocorrelation_rebalance_gate", sources),
            stage="generate",
            theme="autocorrelation_rebalance_gate",
            candidate_logic=(
                "Use return autocorrelation and volatility clustering to choose when daily leveraged "
                "exposure is acceptable and when lower exposure or cash is required."
            ),
            source_types=source_types,
            source_urls=source_urls,
        ),
    ]


def _source_counts(sources: list[AgenticAlphaKnowledgeSource]) -> dict[str, int]:
    return {
        source_type: sum(1 for source in sources if source.source_type == source_type)
        for source_type in sorted(AGENTIC_ALPHA_SOURCE_TYPES)
        if any(source.source_type == source_type for source in sources)
    }


def _candidate_row(
    candidate: AgenticAlphaSearchCandidate,
    *,
    baseline: AgenticAlphaBacktestMetric,
    max_drawdown_floor: float,
    require_return_above_baseline: bool,
) -> dict[str, Any]:
    beats_baseline = float(candidate.metric.return_pct) > float(baseline.return_pct)
    passes_drawdown = float(candidate.metric.max_drawdown_pct) >= float(max_drawdown_floor)
    if not passes_drawdown:
        decision = "Drop"
        decision_basis = "drawdown_gate"
    elif require_return_above_baseline and not beats_baseline:
        decision = "Iterate"
        decision_basis = "return_gate"
    else:
        decision = "Promote"
        decision_basis = "beats_baseline_and_drawdown_gate"

    return {
        "candidate": candidate.name,
        "hypothesis_id": candidate.hypothesis_id,
        "return_pct": float(candidate.metric.return_pct),
        "baseline_return_pct": float(baseline.return_pct),
        "return_improvement_pct": round(float(candidate.metric.return_pct - baseline.return_pct), 10),
        "benchmark_return_pct": float(candidate.metric.benchmark_return_pct),
        "max_drawdown_pct": float(candidate.metric.max_drawdown_pct),
        "max_drawdown_floor": float(max_drawdown_floor),
        "beats_baseline_return": bool(beats_baseline),
        "passes_drawdown_gate": bool(passes_drawdown),
        "decision": decision,
        "decision_basis": decision_basis,
        "recorder_id": candidate.metric.recorder_id,
        "source_urls": ";".join(candidate.source_urls),
    }


def distill_agentic_alpha_search(
    *,
    sources: list[AgenticAlphaKnowledgeSource],
    hypotheses: list[AgenticAlphaHypothesis],
    candidates: list[AgenticAlphaSearchCandidate],
    baseline: AgenticAlphaBacktestMetric,
    max_drawdown_floor: float,
    require_return_above_baseline: bool = True,
) -> AgenticAlphaSearchReport:
    known_hypotheses = {hypothesis.hypothesis_id for hypothesis in hypotheses}
    unknown = [candidate.hypothesis_id for candidate in candidates if candidate.hypothesis_id not in known_hypotheses]
    if unknown:
        raise ValueError(f"candidate references unknown hypothesis_id: {unknown}")

    rows = [
        _candidate_row(
            candidate,
            baseline=baseline,
            max_drawdown_floor=max_drawdown_floor,
            require_return_above_baseline=require_return_above_baseline,
        )
        for candidate in candidates
    ]
    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics["beats_baseline_return"] = metrics["beats_baseline_return"].astype(object)
        metrics["passes_drawdown_gate"] = metrics["passes_drawdown_gate"].astype(object)
    promoted = metrics.loc[metrics["decision"] == "Promote"] if not metrics.empty else metrics
    best_candidate = None
    if promoted is not None and not promoted.empty:
        best_candidate = str(promoted.sort_values(["return_pct"], ascending=False).iloc[0]["candidate"])

    summary = {
        "workflow": "agentic_alpha_rgevd_search",
        "stage_order": AGENTIC_ALPHA_STAGE_ORDER,
        "external_source_counts": _source_counts(sources),
        "hypothesis_count": int(len(hypotheses)),
        "candidate_count": int(len(candidates)),
        "baseline": {
            "name": baseline.name,
            "return_pct": float(baseline.return_pct),
            "max_drawdown_pct": float(baseline.max_drawdown_pct),
            "benchmark_return_pct": float(baseline.benchmark_return_pct),
            "recorder_id": baseline.recorder_id,
        },
        "gates": {
            "max_drawdown_floor": float(max_drawdown_floor),
            "require_return_above_baseline": bool(require_return_above_baseline),
        },
        "goal_achieved": best_candidate is not None,
        "best_candidate": best_candidate,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "metrics_source": "qlib_recorder_derived_sidecars",
            "recorder_only_artifacts": ["predictions", "signals", "portfolio_report", "positions"],
        },
    }
    return AgenticAlphaSearchReport(
        workflow="agentic_alpha_rgevd_search",
        candidate_metrics=metrics,
        summary=summary,
    )


def load_agentic_alpha_backtest_metric(card_dir: Path, *, window: str = "test") -> AgenticAlphaBacktestMetric:
    card_dir = Path(card_dir)
    summary_path = card_dir / "summary.json"
    metrics_path = card_dir / "backtest_window_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing qlib card summary: {summary_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing qlib card backtest window metrics: {metrics_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    required = {"window", "strategy_return_pct", "benchmark_return_pct", "max_drawdown_pct"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"missing qlib backtest metric columns: {missing}")
    rows = metrics.loc[metrics["window"] == window]
    if rows.empty:
        raise ValueError(f"window {window!r} not found in {metrics_path}")
    row = rows.iloc[0]
    return AgenticAlphaBacktestMetric(
        name=str(summary.get("experiment") or card_dir.name),
        return_pct=float(row["strategy_return_pct"]),
        max_drawdown_pct=float(row["max_drawdown_pct"]),
        benchmark_return_pct=float(row["benchmark_return_pct"]),
        recorder_id=str(summary.get("recorder_id") or ""),
    )

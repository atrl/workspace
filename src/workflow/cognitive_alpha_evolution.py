from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.factor import (
    CognitiveAlphaCandidate,
    CognitiveAlphaReport,
    build_cognitive_alpha_seed_candidates,
    build_column_alpha_candidates,
    evaluate_cognitive_alpha_candidates,
)


COGNITIVE_ALPHA_PAPER_REFERENCE = "arxiv:2511.18850"
COGNITIVE_ALPHA_EVOLUTION_OPS = ("mutation", "crossover", "crossover_mutation")


@dataclass(frozen=True)
class CognitiveAlphaEvolutionProfile:
    profile: str
    agent_count: int
    candidates_per_agent: int
    generations: int
    initial_pool_size: int
    parent_pool_size: int
    children_per_generation: int


@dataclass(frozen=True)
class CognitiveAlphaEvolutionSpec:
    run_id: str
    profile: str
    runtime_dir: Path
    generations: int
    parent_pool_size: int
    children_per_generation: int
    initial_pool_size: int | None = None
    qualified_percentile: float = 0.65
    elite_percentile: float = 0.80
    max_abs_correlation: float = 0.95
    min_observations: int = 60
    max_missing_ratio: float = 0.45
    min_distinct_values: int = 5
    fallback_top_n: int = 3
    resume: bool = False


@dataclass(frozen=True)
class CognitiveAlphaEvolutionResult:
    run_dir: Path
    summary: dict[str, Any]
    generation_metrics: pd.DataFrame
    selected_feature_names: list[str]
    selected_candidates: list[CognitiveAlphaCandidate]
    final_report: CognitiveAlphaReport


def build_cognitive_alpha_evolution_profile(profile: str) -> CognitiveAlphaEvolutionProfile:
    normalized = profile.lower()
    if normalized == "soxl_smoke":
        return CognitiveAlphaEvolutionProfile(
            profile="soxl_smoke",
            agent_count=3,
            candidates_per_agent=3,
            generations=2,
            initial_pool_size=9,
            parent_pool_size=4,
            children_per_generation=6,
        )
    if normalized == "soxl_full":
        return CognitiveAlphaEvolutionProfile(
            profile="soxl_full",
            agent_count=21,
            candidates_per_agent=4,
            generations=24,
            initial_pool_size=80,
            parent_pool_size=32,
            children_per_generation=96,
        )
    raise ValueError("profile must be one of: soxl_smoke, soxl_full")


def run_cognitive_alpha_evolution(
    frame: pd.DataFrame,
    *,
    spec: CognitiveAlphaEvolutionSpec,
    label_col: str,
    initial_candidates: list[CognitiveAlphaCandidate] | None = None,
    alpha158_columns: list[str] | None = None,
) -> CognitiveAlphaEvolutionResult:
    if frame.empty:
        raise ValueError("cognitive alpha evolution frame is empty")
    if spec.generations <= 0:
        raise ValueError("generations must be positive")
    if spec.parent_pool_size <= 0:
        raise ValueError("parent_pool_size must be positive")
    if spec.children_per_generation <= 0:
        raise ValueError("children_per_generation must be positive")

    run_dir = Path(spec.runtime_dir) / spec.run_id
    _ensure_run_dirs(run_dir)
    candidate_map = _candidate_map(_initial_candidate_pool(initial_candidates, alpha158_columns))
    generation_metrics: list[pd.DataFrame] = []
    start_generation, previous_selected, resumed = _resume_state(run_dir, spec)
    if resumed:
        replayed_selected = _replay_candidate_map(
            run_dir=run_dir,
            completed_generation=start_generation - 1,
            candidate_map=candidate_map,
            parent_pool_size=spec.parent_pool_size,
            children_per_generation=spec.children_per_generation,
        )
        if replayed_selected:
            previous_selected = replayed_selected

    if start_generation == 0:
        generation_candidates = list(candidate_map.values())
    else:
        generation_candidates = _build_next_generation(
            generation=start_generation,
            parent_names=previous_selected,
            candidate_map=candidate_map,
            children_per_generation=spec.children_per_generation,
        )
        candidate_map.update(_candidate_map(generation_candidates))

    final_report: CognitiveAlphaReport | None = None
    selected_names: list[str] = previous_selected
    for generation in range(start_generation, spec.generations):
        if generation > start_generation:
            generation_candidates = _build_next_generation(
                generation=generation,
                parent_names=selected_names,
                candidate_map=candidate_map,
                children_per_generation=spec.children_per_generation,
            )
            candidate_map.update(_candidate_map(generation_candidates))

        prepared_candidates = _with_repair_candidates(generation_candidates)
        candidate_map.update(_candidate_map(prepared_candidates))
        _write_candidate_pool(run_dir, generation, prepared_candidates)
        final_report = evaluate_cognitive_alpha_candidates(
            frame,
            candidates=prepared_candidates,
            label_col=label_col,
            generation=generation,
            min_observations=spec.min_observations,
            max_missing_ratio=spec.max_missing_ratio,
            min_distinct_values=spec.min_distinct_values,
            qualified_percentile=spec.qualified_percentile,
            elite_percentile=spec.elite_percentile,
            max_abs_correlation=spec.max_abs_correlation,
            fallback_top_n=spec.fallback_top_n,
        )
        selected_names = _parent_names(final_report.metrics, limit=spec.parent_pool_size)
        _write_generation_outputs(run_dir, generation, final_report)
        _write_state(
            run_dir,
            spec=spec,
            completed_generation=generation,
            selected_names=selected_names,
            resumed=resumed,
            resume_start_generation=start_generation,
            reports=generation_metrics + [final_report.metrics],
        )
        generation_metrics.append(final_report.metrics.assign(evolution_generation=generation))

    if final_report is None:
        final_generation = spec.generations - 1
        final_report = _load_final_report(run_dir, final_generation)
        generation_metrics.append(final_report.metrics.assign(evolution_generation=final_generation))
        selected_names = final_report.selected_feature_names

    combined_metrics = pd.concat(generation_metrics, ignore_index=True) if generation_metrics else pd.DataFrame()
    summary = _summary(
        run_dir,
        spec=spec,
        selected_names=selected_names,
        generation_metrics=combined_metrics,
        resumed=resumed,
        resume_start_generation=start_generation,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return CognitiveAlphaEvolutionResult(
        run_dir=run_dir,
        summary=summary,
        generation_metrics=combined_metrics,
        selected_feature_names=selected_names,
        selected_candidates=[candidate_map[name] for name in selected_names if name in candidate_map],
        final_report=final_report,
    )


def _ensure_run_dirs(run_dir: Path) -> None:
    for child in ["candidate_pool", "quality", "fitness", "rejected", "lineage"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def _initial_candidate_pool(
    initial_candidates: list[CognitiveAlphaCandidate] | None,
    alpha158_columns: list[str] | None,
) -> list[CognitiveAlphaCandidate]:
    candidates = list(initial_candidates) if initial_candidates is not None else build_cognitive_alpha_seed_candidates()
    if alpha158_columns:
        candidates.extend(
            build_column_alpha_candidates(
                alpha158_columns,
                level="alpha158_baseline",
                agent="Alpha158",
            )
        )
    if not candidates:
        raise ValueError("at least one initial cognitive alpha candidate is required")
    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: Iterable[CognitiveAlphaCandidate]) -> list[CognitiveAlphaCandidate]:
    out: dict[str, CognitiveAlphaCandidate] = {}
    for candidate in candidates:
        out.setdefault(candidate.name, candidate)
    return list(out.values())


def _candidate_map(candidates: Iterable[CognitiveAlphaCandidate]) -> dict[str, CognitiveAlphaCandidate]:
    return {candidate.name: candidate for candidate in candidates}


def _resume_state(run_dir: Path, spec: CognitiveAlphaEvolutionSpec) -> tuple[int, list[str], bool]:
    state_path = run_dir / "state.json"
    if not spec.resume or not state_path.exists():
        return 0, [], False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed = int(state.get("completed_generation", -1))
    start_generation = min(completed + 1, spec.generations)
    selected = [str(name) for name in state.get("selected_feature_names", [])]
    return start_generation, selected, True


def _load_generation_metrics(run_dir: Path, generation: int) -> pd.DataFrame:
    path = run_dir / "fitness" / f"generation_{generation:03d}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _replay_candidate_map(
    *,
    run_dir: Path,
    completed_generation: int,
    candidate_map: dict[str, CognitiveAlphaCandidate],
    parent_pool_size: int,
    children_per_generation: int,
) -> list[str]:
    candidate_map.update(_candidate_map(_with_repair_candidates(list(candidate_map.values()))))
    selected: list[str] = []
    for generation in range(0, completed_generation + 1):
        if generation > 0:
            children = _build_next_generation(
                generation=generation,
                parent_names=selected,
                candidate_map=candidate_map,
                children_per_generation=children_per_generation,
            )
            candidate_map.update(_candidate_map(_with_repair_candidates(children)))
        metrics = _load_generation_metrics(run_dir, generation)
        if not metrics.empty:
            selected = _parent_names(metrics, limit=parent_pool_size)
    return selected


def _with_repair_candidates(candidates: list[CognitiveAlphaCandidate]) -> list[CognitiveAlphaCandidate]:
    repaired: list[CognitiveAlphaCandidate] = []
    for candidate in candidates:
        repaired.append(candidate)
        repair = _repair_candidate(candidate)
        if repair is not None:
            repaired.append(repair)
    return _dedupe_candidates(repaired)


def _repair_candidate(candidate: CognitiveAlphaCandidate) -> CognitiveAlphaCandidate | None:
    code = candidate.code or ""
    has_future_pattern = ".shift(-" in code or "Ref(" in code and ", -" in code
    if candidate.declared_lookahead <= 0 and not has_future_pattern:
        return None
    if not candidate.required_columns:
        return None
    source_col = candidate.required_columns[0]
    repaired_name = f"{candidate.name}_repaired_lag1"

    def _safe_lagged_change(frame: pd.DataFrame, col: str = source_col) -> pd.Series:
        values = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        return values.pct_change().shift(1).rolling(3, min_periods=1).mean().rename(repaired_name)

    return CognitiveAlphaCandidate(
        name=repaired_name,
        level=candidate.level,
        agent=f"{candidate.agent}Repair",
        theme=f"repaired {candidate.theme}",
        required_columns=(source_col,),
        lookback=max(int(candidate.lookback), 3),
        function=_safe_lagged_change,
        economic_rationale=f"Repair of {candidate.name} using only lagged changes in {source_col}.",
        code=f"safe_lagged_pct_change({source_col}).rolling(3).mean()",
        declared_lookahead=0,
    )


def _rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    mean = values.rolling(window, min_periods=max(3, min(window, 10))).mean()
    std = values.rolling(window, min_periods=max(3, min(window, 10))).std(ddof=0).replace(0.0, np.nan)
    return (values - mean) / std


def _factor_values(candidate: CognitiveAlphaCandidate, frame: pd.DataFrame) -> pd.Series:
    cache = frame.attrs.setdefault("_cognitive_alpha_factor_cache", {})
    if candidate.name in cache:
        return cache[candidate.name]
    values = pd.to_numeric(candidate.function(frame), errors="coerce").replace([np.inf, -np.inf], np.nan)
    cache[candidate.name] = values
    return values


def _build_next_generation(
    *,
    generation: int,
    parent_names: list[str],
    candidate_map: dict[str, CognitiveAlphaCandidate],
    children_per_generation: int,
) -> list[CognitiveAlphaCandidate]:
    parents = [candidate_map[name] for name in parent_names if name in candidate_map]
    if not parents:
        parents = list(candidate_map.values())[: max(1, min(len(candidate_map), children_per_generation))]
    if not parents:
        raise ValueError("cannot evolve without parent candidates")

    children: list[CognitiveAlphaCandidate] = []
    for child_index in range(children_per_generation):
        operation = COGNITIVE_ALPHA_EVOLUTION_OPS[child_index % len(COGNITIVE_ALPHA_EVOLUTION_OPS)]
        left = parents[child_index % len(parents)]
        right = parents[(child_index + 1) % len(parents)]
        if operation == "mutation":
            children.append(_mutation_candidate(left, generation=generation, child_index=child_index))
        elif operation == "crossover":
            children.append(_crossover_candidate(left, right, generation=generation, child_index=child_index))
        else:
            children.append(
                _crossover_mutation_candidate(left, right, generation=generation, child_index=child_index)
            )
    return children


def _mutation_candidate(
    parent: CognitiveAlphaCandidate,
    *,
    generation: int,
    child_index: int,
) -> CognitiveAlphaCandidate:
    name = f"g{generation:03d}_{child_index:03d}_mutation_{_candidate_slug(parent.name)}"

    def _func(frame: pd.DataFrame, candidate: CognitiveAlphaCandidate = parent) -> pd.Series:
        return _rolling_zscore(_factor_values(candidate, frame), 10).shift(1).rename(name)

    return CognitiveAlphaCandidate(
        name=name,
        level=parent.level,
        agent=parent.agent,
        theme=f"mutation of {_candidate_slug(parent.name)}",
        required_columns=parent.required_columns,
        lookback=max(int(parent.lookback), 10),
        function=_func,
        economic_rationale=f"Smooths and lags parent {_candidate_slug(parent.name)} to reduce overreaction.",
        code=f"mutation:zscore({parent.name},10).shift(1)",
    )


def _crossover_candidate(
    left: CognitiveAlphaCandidate,
    right: CognitiveAlphaCandidate,
    *,
    generation: int,
    child_index: int,
) -> CognitiveAlphaCandidate:
    name = (
        f"g{generation:03d}_{child_index:03d}_crossover_"
        f"{_candidate_slug(left.name)}__{_candidate_slug(right.name)}"
    )

    def _func(
        frame: pd.DataFrame,
        left_candidate: CognitiveAlphaCandidate = left,
        right_candidate: CognitiveAlphaCandidate = right,
    ) -> pd.Series:
        left_values = _rolling_zscore(_factor_values(left_candidate, frame), 20)
        right_values = _rolling_zscore(_factor_values(right_candidate, frame), 20)
        return ((left_values + right_values) / 2.0).rename(name)

    return CognitiveAlphaCandidate(
        name=name,
        level="geometric_fusion",
        agent="AgentComposite",
        theme=f"crossover of {_candidate_slug(left.name)} and {_candidate_slug(right.name)}",
        required_columns=tuple(dict.fromkeys((*left.required_columns, *right.required_columns))),
        lookback=max(int(left.lookback), int(right.lookback), 20),
        function=_func,
        economic_rationale=(
            f"Combines parents {_candidate_slug(left.name)} and {_candidate_slug(right.name)} after scale normalization."
        ),
        code=f"crossover:mean(zscore({left.name},20), zscore({right.name},20))",
    )


def _crossover_mutation_candidate(
    left: CognitiveAlphaCandidate,
    right: CognitiveAlphaCandidate,
    *,
    generation: int,
    child_index: int,
) -> CognitiveAlphaCandidate:
    name = (
        f"g{generation:03d}_{child_index:03d}_crossover_mutation_"
        f"{_candidate_slug(left.name)}__{_candidate_slug(right.name)}"
    )

    def _func(
        frame: pd.DataFrame,
        left_candidate: CognitiveAlphaCandidate = left,
        right_candidate: CognitiveAlphaCandidate = right,
    ) -> pd.Series:
        spread = _rolling_zscore(_factor_values(left_candidate, frame), 10) - _rolling_zscore(
            _factor_values(right_candidate, frame),
            10,
        )
        return spread.ewm(span=5, adjust=False).mean().shift(1).rename(name)

    return CognitiveAlphaCandidate(
        name=name,
        level="geometric_fusion",
        agent="AgentCreative",
        theme=f"mutated crossover of {_candidate_slug(left.name)} and {_candidate_slug(right.name)}",
        required_columns=tuple(dict.fromkeys((*left.required_columns, *right.required_columns))),
        lookback=max(int(left.lookback), int(right.lookback), 10),
        function=_func,
        economic_rationale=(
            f"Uses lagged spread between parents {_candidate_slug(left.name)} and {_candidate_slug(right.name)}."
        ),
        code=f"crossover_mutation:ewm(zscore({left.name},10)-zscore({right.name},10),5).shift(1)",
    )


def _candidate_slug(name: str, *, prefix_chars: int = 18, digest_chars: int = 10) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_") or "candidate"
    if len(slug) <= prefix_chars + digest_chars + 1:
        return slug
    digest = sha1(str(name).encode("utf-8")).hexdigest()[:digest_chars]
    return f"{slug[:prefix_chars]}_{digest}"


def _parent_names(metrics: pd.DataFrame, *, limit: int) -> list[str]:
    selectable = metrics.loc[metrics["redundancy_status"].isin(["selected", "selected_fallback"])].copy()
    if selectable.empty:
        selectable = metrics.loc[metrics["quality_status"] == "pass"].copy()
    if selectable.empty:
        return []
    selectable = selectable.sort_values(["selection_score", "min_metric_percentile"], ascending=[False, False])
    return selectable["factor"].astype(str).head(limit).tolist()


def _candidate_manifest(candidate: CognitiveAlphaCandidate) -> dict[str, Any]:
    operation = _operation_from_name(candidate.name)
    return {
        "name": candidate.name,
        "level": candidate.level,
        "agent": candidate.agent,
        "theme": candidate.theme,
        "required_columns": list(candidate.required_columns),
        "lookback": int(candidate.lookback),
        "economic_rationale": candidate.economic_rationale,
        "code": candidate.code,
        "declared_lookahead": int(candidate.declared_lookahead),
        "operation": operation,
        "parents": _parents_from_code(candidate.code),
    }


def _operation_from_name(name: str) -> str:
    if "_crossover_mutation_" in name:
        return "crossover_mutation"
    if "_crossover_" in name:
        return "crossover"
    if "_mutation_" in name:
        return "mutation"
    if name.endswith("_repaired_lag1"):
        return "repair"
    return "initial"


def _parents_from_code(code: str) -> list[str]:
    if ":" not in code:
        return []
    payload = code.split(":", 1)[1]
    tokens = []
    for marker in ["zscore(", "mean("]:
        payload = payload.replace(marker, "(")
    for part in payload.replace(")", "(").split("("):
        if part and "," not in part and " " not in part and part not in {"mutation", "crossover_mutation"}:
            tokens.append(part)
    return list(dict.fromkeys(tokens))


def _write_candidate_pool(run_dir: Path, generation: int, candidates: list[CognitiveAlphaCandidate]) -> None:
    path = run_dir / "candidate_pool" / f"generation_{generation:03d}.jsonl"
    rows = [_candidate_manifest(candidate) | {"generation": generation} for candidate in candidates]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    lineage_path = run_dir / "lineage" / f"generation_{generation:03d}.jsonl"
    lineage_path.write_text(
        "".join(
            json.dumps(
                {
                    "generation": generation,
                    "candidate": row["name"],
                    "operation": row["operation"],
                    "parents": row["parents"],
                },
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_generation_outputs(run_dir: Path, generation: int, report: CognitiveAlphaReport) -> None:
    metrics = report.metrics.copy()
    metrics.to_csv(run_dir / "fitness" / f"generation_{generation:03d}.csv", index=False)
    metrics[
        [
            "factor",
            "generation",
            "level",
            "agent",
            "quality_status",
            "quality_reason",
            "missing_ratio",
            "distinct_values",
            "selection_tier",
            "redundancy_status",
        ]
    ].to_csv(run_dir / "quality" / f"generation_{generation:03d}.csv", index=False)
    rejected = metrics.loc[metrics["quality_status"] == "reject"].to_dict(orient="records")
    (run_dir / "rejected" / f"generation_{generation:03d}.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected),
        encoding="utf-8",
    )


def _write_state(
    run_dir: Path,
    *,
    spec: CognitiveAlphaEvolutionSpec,
    completed_generation: int,
    selected_names: list[str],
    resumed: bool,
    resume_start_generation: int,
    reports: list[pd.DataFrame],
) -> None:
    combined = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    state = _summary(
        run_dir,
        spec=spec,
        selected_names=selected_names,
        generation_metrics=combined,
        resumed=resumed,
        resume_start_generation=resume_start_generation,
    ) | {
        "completed_generation": int(completed_generation),
        "status": "running" if completed_generation < spec.generations - 1 else "complete",
    }
    (run_dir / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _summary(
    run_dir: Path,
    *,
    spec: CognitiveAlphaEvolutionSpec,
    selected_names: list[str],
    generation_metrics: pd.DataFrame,
    resumed: bool,
    resume_start_generation: int,
) -> dict[str, Any]:
    repair_attempt_count = 0
    if not generation_metrics.empty:
        repair_attempt_count = int(generation_metrics["factor"].astype(str).str.endswith("_repaired_lag1").sum())
    return {
        "workflow": "cognitive_alpha_evolution",
        "paper_reference": COGNITIVE_ALPHA_PAPER_REFERENCE,
        "run_id": spec.run_id,
        "profile": spec.profile,
        "run_dir": run_dir.as_posix(),
        "completed_generations": int(spec.generations),
        "parent_pool_size": int(spec.parent_pool_size),
        "children_per_generation": int(spec.children_per_generation),
        "selected_feature_names": selected_names,
        "selected_count": int(len(selected_names)),
        "candidate_metric_rows": int(len(generation_metrics)),
        "repair_attempt_count": repair_attempt_count,
        "evolution_ops": list(COGNITIVE_ALPHA_EVOLUTION_OPS),
        "resumed": bool(resumed),
        "resume_start_generation": int(resume_start_generation),
        "thresholds": {
            "qualified_percentile": float(spec.qualified_percentile),
            "elite_percentile": float(spec.elite_percentile),
            "max_abs_correlation": float(spec.max_abs_correlation),
            "min_observations": int(spec.min_observations),
            "max_missing_ratio": float(spec.max_missing_ratio),
            "min_distinct_values": int(spec.min_distinct_values),
            "fallback_top_n": int(spec.fallback_top_n),
        },
        "checkpoints": {
            "candidate_pool": "candidate_pool/generation_*.jsonl",
            "quality": "quality/generation_*.csv",
            "fitness": "fitness/generation_*.csv",
            "rejected": "rejected/generation_*.jsonl",
            "lineage": "lineage/generation_*.jsonl",
            "state": "state.json",
        },
        "spec": {
            **asdict(spec),
            "runtime_dir": Path(spec.runtime_dir).as_posix(),
        },
    }


def _load_final_report(run_dir: Path, generation: int) -> CognitiveAlphaReport:
    metrics = pd.read_csv(run_dir / "fitness" / f"generation_{generation:03d}.csv")
    selected = _parent_names(metrics, limit=len(metrics))
    return CognitiveAlphaReport(
        metrics=metrics,
        correlation=pd.DataFrame(),
        feature_frame=pd.DataFrame(),
        selected_feature_names=selected,
        summary={
            "workflow": "cognitive_alpha_evolution",
            "paper_reference": COGNITIVE_ALPHA_PAPER_REFERENCE,
            "generation": int(generation),
        },
    )

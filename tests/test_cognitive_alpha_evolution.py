from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _evolution_frame(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    trend = np.linspace(0.0, 1.0, rows)
    cycle = np.sin(np.arange(rows) / 4.0)
    close = 30.0 + trend * 10.0 + cycle
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "open": close * (1.0 - 0.002),
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000 + np.arange(rows) * 10_000 + cycle * 15_000,
        }
    )
    frame["label_return_5d"] = pd.Series(close).pct_change(5).shift(-5).ffill().fillna(0.0)
    frame["alpha158_good"] = frame["label_return_5d"].shift(1).fillna(0.0) + trend * 0.01
    frame["alpha158_noise"] = np.cos(np.arange(rows) * 1.9)
    return frame


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cognitive_alpha_evolution_profiles_match_smoke_and_full_contracts():
    from src.workflow import build_cognitive_alpha_evolution_profile

    smoke = build_cognitive_alpha_evolution_profile("soxl_smoke")
    full = build_cognitive_alpha_evolution_profile("soxl_full")

    assert smoke.profile == "soxl_smoke"
    assert smoke.agent_count == 3
    assert smoke.candidates_per_agent == 3
    assert smoke.generations == 2
    assert smoke.parent_pool_size == 4
    assert smoke.children_per_generation == 6

    assert full.profile == "soxl_full"
    assert full.agent_count == 21
    assert full.initial_pool_size == 80
    assert full.parent_pool_size == 32
    assert full.children_per_generation == 96
    assert full.generations == 24


def test_cognitive_alpha_evolution_runner_checkpoints_repairs_and_evolves(tmp_path):
    from src.factor import CognitiveAlphaCandidate, build_cognitive_alpha_seed_candidates
    from src.workflow import CognitiveAlphaEvolutionSpec, run_cognitive_alpha_evolution

    frame = _evolution_frame()
    leaky = CognitiveAlphaCandidate(
        name="factor_future_leak",
        level="geometric_fusion",
        agent="AgentCreative",
        theme="future leakage repair path",
        required_columns=("close",),
        lookback=1,
        declared_lookahead=1,
        code="df['close'].shift(-1) / df['close'] - 1",
        economic_rationale="Invalid future leak used to prove reject and repair bookkeeping.",
        function=lambda data: data["close"].shift(-1) / data["close"] - 1.0,
    )
    spec = CognitiveAlphaEvolutionSpec(
        run_id="unit_smoke",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=2,
        parent_pool_size=3,
        children_per_generation=5,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        max_abs_correlation=0.98,
        fallback_top_n=2,
    )

    result = run_cognitive_alpha_evolution(
        frame,
        spec=spec,
        label_col="label_return_5d",
        initial_candidates=[leaky, *build_cognitive_alpha_seed_candidates()[:3]],
        alpha158_columns=["alpha158_good", "alpha158_noise"],
    )

    run_dir = tmp_path / "unit_smoke"
    assert result.run_dir == run_dir
    assert result.summary["workflow"] == "cognitive_alpha_evolution"
    assert result.summary["paper_reference"] == "arxiv:2511.18850"
    assert result.summary["completed_generations"] == 2
    assert result.summary["repair_attempt_count"] >= 1
    assert result.summary["evolution_ops"] == ["mutation", "crossover", "crossover_mutation"]

    assert (run_dir / "candidate_pool" / "generation_000.jsonl").exists()
    assert (run_dir / "candidate_pool" / "generation_001.jsonl").exists()
    assert (run_dir / "fitness" / "generation_000.csv").exists()
    assert (run_dir / "fitness" / "generation_001.csv").exists()
    assert (run_dir / "rejected" / "generation_000.jsonl").exists()
    assert (run_dir / "state.json").exists()

    first_pool = _read_jsonl(run_dir / "candidate_pool" / "generation_000.jsonl")
    second_pool = _read_jsonl(run_dir / "candidate_pool" / "generation_001.jsonl")
    rejected = _read_jsonl(run_dir / "rejected" / "generation_000.jsonl")
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "fitness" / "generation_001.csv")

    assert any(row["name"] == "factor_future_leak" for row in first_pool)
    assert any(row["name"] == "factor_future_leak_repaired_lag1" for row in first_pool)
    assert any(row["factor"] == "factor_future_leak" and row["quality_reason"] == "future_lookahead_declared" for row in rejected)
    assert {"mutation", "crossover", "crossover_mutation"}.issubset({row["operation"] for row in second_pool})
    assert {"ic", "icir", "rank_ic", "rank_icir", "mutual_information"}.issubset(metrics.columns)
    assert state["run_id"] == "unit_smoke"
    assert state["completed_generation"] == 1
    assert state["selected_feature_names"] == result.selected_feature_names
    assert result.selected_feature_names


def test_cognitive_alpha_evolution_resume_preserves_completed_checkpoints(tmp_path):
    from src.factor import build_cognitive_alpha_seed_candidates
    from src.workflow import CognitiveAlphaEvolutionSpec, run_cognitive_alpha_evolution

    frame = _evolution_frame()
    first_spec = CognitiveAlphaEvolutionSpec(
        run_id="resume_smoke",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=1,
        parent_pool_size=3,
        children_per_generation=4,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=2,
    )
    run_cognitive_alpha_evolution(
        frame,
        spec=first_spec,
        label_col="label_return_5d",
        initial_candidates=build_cognitive_alpha_seed_candidates()[:4],
        alpha158_columns=["alpha158_good"],
    )
    generation_zero = tmp_path / "resume_smoke" / "candidate_pool" / "generation_000.jsonl"
    original_generation_zero = generation_zero.read_text(encoding="utf-8")

    resumed_spec = CognitiveAlphaEvolutionSpec(
        run_id="resume_smoke",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=2,
        parent_pool_size=3,
        children_per_generation=4,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=2,
        resume=True,
    )
    result = run_cognitive_alpha_evolution(
        frame,
        spec=resumed_spec,
        label_col="label_return_5d",
        initial_candidates=build_cognitive_alpha_seed_candidates()[:4],
        alpha158_columns=["alpha158_good"],
    )

    assert generation_zero.read_text(encoding="utf-8") == original_generation_zero
    assert (tmp_path / "resume_smoke" / "candidate_pool" / "generation_001.jsonl").exists()
    assert result.summary["resumed"] is True
    assert result.summary["resume_start_generation"] == 1


def test_cognitive_alpha_evolution_resume_replays_evolved_parent_candidates(tmp_path):
    from src.factor import build_cognitive_alpha_seed_candidates
    from src.workflow import CognitiveAlphaEvolutionSpec, run_cognitive_alpha_evolution

    frame = _evolution_frame()
    initial = build_cognitive_alpha_seed_candidates()[:4]
    first_spec = CognitiveAlphaEvolutionSpec(
        run_id="deep_resume_smoke",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=2,
        parent_pool_size=3,
        children_per_generation=6,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=2,
    )
    first_result = run_cognitive_alpha_evolution(
        frame,
        spec=first_spec,
        label_col="label_return_5d",
        initial_candidates=initial,
        alpha158_columns=["alpha158_good"],
    )
    evolved_parents = [name for name in first_result.selected_feature_names if name.startswith("g001_")]
    assert evolved_parents

    resumed_spec = CognitiveAlphaEvolutionSpec(
        run_id="deep_resume_smoke",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=3,
        parent_pool_size=3,
        children_per_generation=6,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=2,
        resume=True,
    )
    result = run_cognitive_alpha_evolution(
        frame,
        spec=resumed_spec,
        label_col="label_return_5d",
        initial_candidates=initial,
        alpha158_columns=["alpha158_good"],
    )

    generation_two = _read_jsonl(tmp_path / "deep_resume_smoke" / "candidate_pool" / "generation_002.jsonl")
    generation_two_payload = json.dumps(generation_two)
    assert result.summary["resume_start_generation"] == 2
    assert any(parent in generation_two_payload for parent in evolved_parents)

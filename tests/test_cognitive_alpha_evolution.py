from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
    from src.factor import CognitiveAlphaCandidate, build_cognitive_alpha_seed_candidates, materialize_cognitive_alpha_features
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
    assert [candidate.name for candidate in result.selected_candidates] == result.selected_feature_names

    full_features = materialize_cognitive_alpha_features(
        frame.tail(20).copy(),
        result.selected_candidates,
        selected_names=result.selected_feature_names,
    )
    assert full_features.columns.tolist() == result.selected_feature_names
    assert len(full_features) == 20


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


def test_cognitive_alpha_evolution_uses_bounded_candidate_names_for_full_profile_scale(tmp_path):
    from src.factor import build_cognitive_alpha_seed_candidates
    from src.workflow import CognitiveAlphaEvolutionSpec, run_cognitive_alpha_evolution

    frame = _evolution_frame(rows=180)
    spec = CognitiveAlphaEvolutionSpec(
        run_id="bounded_names",
        profile="soxl_full",
        runtime_dir=tmp_path,
        generations=5,
        parent_pool_size=6,
        children_per_generation=12,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=4,
    )

    run_cognitive_alpha_evolution(
        frame,
        spec=spec,
        label_col="label_return_5d",
        initial_candidates=build_cognitive_alpha_seed_candidates(),
        alpha158_columns=["alpha158_good", "alpha158_noise"],
    )

    names = []
    themes = []
    for path in sorted((tmp_path / "bounded_names" / "candidate_pool").glob("generation_*.jsonl")):
        rows = _read_jsonl(path)
        names.extend(row["name"] for row in rows)
        themes.extend(row["theme"] for row in rows)

    assert names
    assert max(len(name) for name in names) <= 96
    assert max(len(theme) for theme in themes) <= 160


def test_cognitive_alpha_evolution_caches_parent_factor_values_within_generation(tmp_path):
    from src.factor import CognitiveAlphaCandidate
    from src.workflow import CognitiveAlphaEvolutionSpec, run_cognitive_alpha_evolution

    frame = _evolution_frame(rows=160)
    calls = {"parent": 0}

    def parent_factor(data):
        calls["parent"] += 1
        return data["alpha158_good"].rename("counted_parent")

    counted_parent = CognitiveAlphaCandidate(
        name="counted_parent",
        level="stability_regime_gating",
        agent="AgentStability",
        theme="call count parent",
        required_columns=("alpha158_good",),
        lookback=3,
        function=parent_factor,
        economic_rationale="Used to prove evolved children reuse materialized parent values.",
        code="df['alpha158_good']",
    )
    spec = CognitiveAlphaEvolutionSpec(
        run_id="cached_parent_values",
        profile="soxl_full",
        runtime_dir=tmp_path,
        generations=2,
        parent_pool_size=1,
        children_per_generation=9,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=1,
    )

    run_cognitive_alpha_evolution(
        frame,
        spec=spec,
        label_col="label_return_5d",
        initial_candidates=[counted_parent],
    )

    assert calls["parent"] <= 2


def test_cognitive_alpha_code_candidate_provider_generates_full_profile_agent_pool(tmp_path):
    from src.workflow import (
        CognitiveAlphaEvolutionSpec,
        build_cognitive_alpha_code_candidate_provider,
        run_cognitive_alpha_evolution,
    )

    frame = _evolution_frame(rows=180)
    spec = CognitiveAlphaEvolutionSpec(
        run_id="provider_full_pool",
        profile="soxl_full",
        runtime_dir=tmp_path,
        generations=1,
        initial_pool_size=80,
        parent_pool_size=8,
        children_per_generation=12,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=4,
    )

    result = run_cognitive_alpha_evolution(
        frame,
        spec=spec,
        label_col="label_return_5d",
        candidate_provider=build_cognitive_alpha_code_candidate_provider(),
    )

    generation_zero = _read_jsonl(tmp_path / "provider_full_pool" / "candidate_pool" / "generation_000.jsonl")
    agents = {row["agent"] for row in generation_zero}
    guidance_styles = {row["metadata"]["guidance_style"] for row in generation_zero}

    assert len(generation_zero) == 80
    assert len(agents) == 21
    assert guidance_styles == {
        "trend_continuation",
        "mean_reversion",
        "volatility_regime",
        "liquidity_pressure",
        "fragility_tail",
    }
    assert result.summary["candidate_provider"] == {
        "provider_id": "deterministic_code_agent_pool",
        "provider_kind": "repo_owned_code",
        "profile_agent_count": 21,
        "guidance_style_count": 5,
        "requested_initial_pool_size": 80,
        "generated_candidate_count": 80,
    }
    assert result.summary["initial_candidate_count"] == 80


def test_cognitive_alpha_external_llm_provider_records_prompt_response_and_repair_provenance(tmp_path):
    from src.workflow import (
        CognitiveAlphaEvolutionSpec,
        build_cognitive_alpha_external_llm_candidate_provider,
        run_cognitive_alpha_evolution,
    )

    requests: list[dict] = []

    def fake_completion(request: dict) -> dict:
        requests.append(request)
        index = int(request["request_index"])
        code = "close.shift(-1) / close - 1.0" if index == 0 else "close.pct_change(5).shift(1)"
        return {
            "candidates": [
                {
                    "name": f"external_llm_candidate_{index:03d}",
                    "level": request["level"],
                    "agent": request["agent"],
                    "theme": f"{request['guidance_style']} external code alpha",
                    "required_columns": ["close"],
                    "lookback": 5,
                    "code": code,
                    "economic_rationale": "External LLM generated lag-aware SOXL OHLCV code.",
                }
            ]
        }

    provider = build_cognitive_alpha_external_llm_candidate_provider(
        completion=fake_completion,
        provider_id="unit_external_llm",
        model="unit-llm",
    )
    spec = CognitiveAlphaEvolutionSpec(
        run_id="external_llm_provider",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=1,
        initial_pool_size=4,
        parent_pool_size=2,
        children_per_generation=3,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=2,
    )

    result = run_cognitive_alpha_evolution(
        _evolution_frame(rows=160),
        spec=spec,
        label_col="label_return_5d",
        candidate_provider=provider,
    )

    generation_zero = _read_jsonl(tmp_path / "external_llm_provider" / "candidate_pool" / "generation_000.jsonl")
    by_name = {row["name"]: row for row in generation_zero}

    assert len(requests) == 4
    assert requests[0]["paper_reference"] == "arxiv:2511.18850"
    assert requests[0]["model"] == "unit-llm"
    assert requests[0]["request_index"] == 0
    assert requests[0]["target_candidate_count"] == 1
    assert result.summary["candidate_provider"]["provider_id"] == "unit_external_llm"
    assert result.summary["candidate_provider"]["provider_kind"] == "external_llm_code"
    assert result.summary["candidate_provider"]["llm_model"] == "unit-llm"
    assert result.summary["candidate_provider"]["prompt_count"] == 4
    assert result.summary["candidate_provider"]["response_count"] == 4
    assert result.summary["candidate_provider"]["provider_failure_count"] == 0
    assert result.summary["initial_candidate_count"] == 4

    original = by_name["external_llm_candidate_000"]
    repaired = by_name["external_llm_candidate_000_repaired_lag1"]
    assert original["metadata"]["candidate_source"] == "external_llm_code"
    assert original["metadata"]["llm_model"] == "unit-llm"
    assert original["metadata"]["prompt_hash"]
    assert original["metadata"]["response_hash"]
    assert repaired["operation"] == "repair"
    assert repaired["metadata"]["repair_source_candidate"] == "external_llm_candidate_000"
    assert repaired["metadata"]["repair_source_provider_kind"] == "external_llm_code"
    assert repaired["metadata"]["repair_reason"] == "future_lookahead_code_pattern"


def test_cognitive_alpha_command_llm_provider_generates_candidates_from_external_process(tmp_path):
    from src.workflow import (
        CognitiveAlphaEvolutionSpec,
        build_cognitive_alpha_command_llm_candidate_provider,
        run_cognitive_alpha_evolution,
    )

    provider_script = tmp_path / "provider.py"
    provider_script.write_text(
        "\n".join(
            [
                "import json, sys",
                "request = json.load(sys.stdin)",
                "idx = request['request_index']",
                "json.dump({'candidates': [{",
                "    'name': f'command_llm_candidate_{idx:03d}',",
                "    'level': request['level'],",
                "    'agent': request['agent'],",
                "    'theme': request['guidance_style'] + ' command provider alpha',",
                "    'required_columns': ['close'],",
                "    'lookback': 5,",
                "    'code': 'close.pct_change(5).shift(1)',",
                "    'economic_rationale': 'External command generated lag-aware SOXL code.'",
                "}]}, sys.stdout)",
            ]
        ),
        encoding="utf-8",
    )
    provider = build_cognitive_alpha_command_llm_candidate_provider(
        command=(sys.executable, provider_script.as_posix()),
        provider_id="unit_command_llm",
        model="unit-command",
    )
    spec = CognitiveAlphaEvolutionSpec(
        run_id="command_llm_provider",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=1,
        initial_pool_size=2,
        parent_pool_size=1,
        children_per_generation=2,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=1,
    )

    result = run_cognitive_alpha_evolution(
        _evolution_frame(rows=120),
        spec=spec,
        label_col="label_return_5d",
        candidate_provider=provider,
    )

    generation_zero = _read_jsonl(tmp_path / "command_llm_provider" / "candidate_pool" / "generation_000.jsonl")

    assert [row["name"] for row in generation_zero] == ["command_llm_candidate_000", "command_llm_candidate_001"]
    assert result.summary["candidate_provider"]["provider_id"] == "unit_command_llm"
    assert result.summary["candidate_provider"]["provider_kind"] == "external_llm_code"
    assert result.summary["candidate_provider"]["llm_model"] == "unit-command"
    assert result.summary["candidate_provider"]["prompt_count"] == 2
    assert result.summary["candidate_provider"]["response_count"] == 2


def test_cognitive_alpha_fixture_provider_exercises_external_command_and_repair_loop(tmp_path):
    from src.workflow import (
        CognitiveAlphaEvolutionSpec,
        build_cognitive_alpha_command_llm_candidate_provider,
        run_cognitive_alpha_evolution,
    )

    provider = build_cognitive_alpha_command_llm_candidate_provider(
        command=(sys.executable, "-m", "src.workflow.cognitive_alpha_fixture_provider"),
        provider_id="fixture_external_command_provider",
        model="fixture-command",
    )
    spec = CognitiveAlphaEvolutionSpec(
        run_id="fixture_command_provider",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=1,
        initial_pool_size=3,
        parent_pool_size=2,
        children_per_generation=3,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=2,
    )

    result = run_cognitive_alpha_evolution(
        _evolution_frame(rows=140),
        spec=spec,
        label_col="label_return_5d",
        candidate_provider=provider,
    )

    generation_zero = _read_jsonl(tmp_path / "fixture_command_provider" / "candidate_pool" / "generation_000.jsonl")
    by_name = {row["name"]: row for row in generation_zero}

    assert result.summary["candidate_provider"]["provider_id"] == "fixture_external_command_provider"
    assert result.summary["candidate_provider"]["provider_kind"] == "external_llm_code"
    assert result.summary["candidate_provider"]["llm_model"] == "fixture-command"
    assert result.summary["candidate_provider"]["prompt_count"] == 3
    assert result.summary["candidate_provider"]["response_count"] == 3
    assert result.summary["repair_attempt_count"] >= 1
    assert "fixture_external_alpha_000" in by_name
    assert "fixture_external_alpha_000_repaired_lag1" in by_name
    assert by_name["fixture_external_alpha_000"]["metadata"]["provider_id"] == "fixture_external_command_provider"
    assert by_name["fixture_external_alpha_000"]["metadata"]["prompt_hash"]
    assert by_name["fixture_external_alpha_000"]["metadata"]["response_hash"]
    assert by_name["fixture_external_alpha_000_repaired_lag1"]["metadata"]["repair_source_provider_kind"] == (
        "external_llm_code"
    )


def test_cognitive_alpha_external_llm_provider_can_use_alpha158_columns_and_numpy_functions(tmp_path):
    from src.workflow import (
        CognitiveAlphaEvolutionSpec,
        build_cognitive_alpha_external_llm_candidate_provider,
        run_cognitive_alpha_evolution,
    )

    def fake_completion(request: dict) -> dict:
        assert "alpha158_good" in request["alpha158_columns"]
        return {
            "candidates": [
                {
                    "name": f"external_alpha158_{request['request_index']:03d}",
                    "level": request["level"],
                    "agent": request["agent"],
                    "theme": "alpha158 plus volume external code alpha",
                    "required_columns": ["alpha158_good", "volume"],
                    "lookback": 10,
                    "code": "alpha158_good.shift(1) + np.log1p(volume).pct_change().shift(1)",
                    "economic_rationale": "External LLM combines a qlib Alpha158 signal with lagged volume impulse.",
                }
            ]
        }

    provider = build_cognitive_alpha_external_llm_candidate_provider(
        completion=fake_completion,
        provider_id="unit_external_alpha158",
        model="unit-llm",
    )
    spec = CognitiveAlphaEvolutionSpec(
        run_id="external_alpha158_provider",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=1,
        initial_pool_size=1,
        parent_pool_size=1,
        children_per_generation=1,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=1,
    )

    result = run_cognitive_alpha_evolution(
        _evolution_frame(rows=120),
        spec=spec,
        label_col="label_return_5d",
        alpha158_columns=["alpha158_good"],
        candidate_provider=provider,
    )

    assert result.summary["candidate_provider"]["provider_id"] == "unit_external_alpha158"
    assert result.generation_metrics.loc[0, "quality_status"] == "pass"


def test_cognitive_alpha_candidate_provider_fails_closed_when_empty(tmp_path):
    from src.workflow import CognitiveAlphaEvolutionSpec, run_cognitive_alpha_evolution

    class EmptyProvider:
        provider_id = "empty_provider"
        provider_kind = "test"

        def generate_candidates(self, **_kwargs):
            return []

    spec = CognitiveAlphaEvolutionSpec(
        run_id="empty_provider",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=1,
        parent_pool_size=3,
        children_per_generation=4,
    )

    with pytest.raises(ValueError, match="candidate provider returned no candidates"):
        run_cognitive_alpha_evolution(
            _evolution_frame(),
            spec=spec,
            label_col="label_return_5d",
            candidate_provider=EmptyProvider(),
        )


def test_cognitive_alpha_candidate_provider_preserves_explicit_repair_candidates(tmp_path):
    from src.factor import CognitiveAlphaCandidate
    from src.workflow import (
        CognitiveAlphaEvolutionSpec,
        build_cognitive_alpha_code_candidate_provider,
        run_cognitive_alpha_evolution,
    )

    frame = _evolution_frame(rows=160)
    leaky = CognitiveAlphaCandidate(
        name="provider_future_leak",
        level="geometric_fusion",
        agent="AgentCreative",
        theme="provider repair proof",
        required_columns=("close",),
        lookback=1,
        declared_lookahead=1,
        code="df['close'].shift(-1)",
        economic_rationale="Invalid explicit candidate included to prove provider runs still keep repair evidence.",
        function=lambda data: data["close"].shift(-1),
    )
    spec = CognitiveAlphaEvolutionSpec(
        run_id="provider_plus_repair",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=1,
        initial_pool_size=9,
        parent_pool_size=4,
        children_per_generation=6,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=4,
    )

    result = run_cognitive_alpha_evolution(
        frame,
        spec=spec,
        label_col="label_return_5d",
        initial_candidates=[leaky],
        candidate_provider=build_cognitive_alpha_code_candidate_provider(),
    )

    generation_zero = _read_jsonl(tmp_path / "provider_plus_repair" / "candidate_pool" / "generation_000.jsonl")
    names = {row["name"] for row in generation_zero}

    assert "provider_future_leak" in names
    assert "provider_future_leak_repaired_lag1" in names
    assert result.summary["initial_candidate_count"] == 9
    assert result.summary["candidate_provider"]["generated_candidate_count"] == 9

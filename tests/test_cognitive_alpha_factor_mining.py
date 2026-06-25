from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _mining_frame(rows: int = 140) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    trend = np.linspace(0.0, 1.0, rows)
    close = 30.0 + trend * 8.0 + np.sin(np.arange(rows) / 5.0)
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "open": close * 0.998,
            "high": close * 1.015,
            "low": close * 0.985,
            "close": close,
            "volume": 1_000_000 + np.arange(rows) * 5_000,
        }
    )
    frame["label_return_5d"] = pd.Series(close).pct_change(5).shift(-5).ffill().fillna(0.0)
    frame["alpha158_good"] = frame["label_return_5d"].shift(1).fillna(0.0) + trend * 0.01
    return frame


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cognitive_alpha_factor_mining_lives_in_src_factor_and_has_no_position_signal(tmp_path):
    from src.factor import (
        CognitiveAlphaCandidate,
        CognitiveAlphaFactorMiningSpec,
        run_cognitive_alpha_factor_mining,
    )
    from src.workflow import CognitiveAlphaEvolutionSpec

    leaky = CognitiveAlphaCandidate(
        name="factor_future_leak",
        level="geometric_fusion",
        agent="AgentCreative",
        theme="future leakage repair path",
        required_columns=("close",),
        lookback=1,
        declared_lookahead=1,
        code="close.shift(-1) / close - 1.0",
        economic_rationale="Invalid future leak used to prove factor-mining quality gates.",
        function=lambda data: data["close"].shift(-1) / data["close"] - 1.0,
    )
    spec = CognitiveAlphaFactorMiningSpec(
        evolution_spec=CognitiveAlphaEvolutionSpec(
            run_id="factor_mining_unit",
            profile="soxl_smoke",
            runtime_dir=tmp_path,
            generations=2,
            initial_pool_size=4,
            parent_pool_size=2,
            children_per_generation=4,
            qualified_percentile=0.50,
            elite_percentile=0.75,
            fallback_top_n=2,
        ),
        label_col="label_return_5d",
        alpha158_columns=("alpha158_good",),
    )

    result = run_cognitive_alpha_factor_mining(
        _mining_frame(),
        spec=spec,
        initial_candidates=[leaky],
    )

    assert result.summary["workflow"] == "cognitive_alpha_factor_mining"
    assert result.summary["source_layer"] == "src.factor"
    assert result.summary["paper_reference"] == "arxiv:2511.18850"
    assert result.summary["position_model_included"] is False
    assert result.summary["position_signal_included"] is False
    assert "latest_signal" not in result.summary
    assert "target_position" not in result.summary
    assert result.summary["stage_order"] == [
        "candidate_generation",
        "quality_checker",
        "fitness_evaluation",
        "evolution_loop",
        "factor_library",
    ]
    assert result.summary["evolution"]["workflow"] == "cognitive_alpha_evolution"
    assert result.summary["evolution"]["completed_generations"] == 2
    assert result.summary["factor_library"]["selected_count"] == len(result.factor_library)
    assert result.selected_feature_names
    assert [item["name"] for item in result.factor_library] == result.selected_feature_names
    assert all(item["source_layer"] == "src.factor" for item in result.factor_library)
    assert all(item["artifact_role"] == "alpha_factor" for item in result.factor_library)

    run_dir = tmp_path / "factor_mining_unit"
    assert result.run_dir == run_dir
    assert (run_dir / "candidate_pool" / "generation_001.jsonl").exists()
    assert (run_dir / "fitness" / "generation_001.csv").exists()
    rejected = _read_jsonl(run_dir / "rejected" / "generation_000.jsonl")
    assert any(row["factor"] == "factor_future_leak" for row in rejected)

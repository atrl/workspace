from __future__ import annotations

import numpy as np
import pandas as pd


def _sample_ohlcv_frame(rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    trend = np.linspace(0.0, 1.0, rows)
    cycle = np.sin(np.arange(rows) / 5.0)
    close = 20.0 + trend * 8.0 + cycle
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "open": close * (1.0 - 0.003 * np.cos(np.arange(rows) / 3.0)),
            "high": close * 1.018,
            "low": close * 0.982,
            "close": close,
            "volume": 1_000_000 + np.arange(rows) * 15_000 + cycle * 20_000,
        }
    )
    forward = pd.Series(close).pct_change(5).shift(-5)
    frame["label_return_5d"] = forward.ffill().fillna(0.0)
    frame["alpha158_good"] = frame["label_return_5d"].shift(1).fillna(0.0) + trend * 0.01
    frame["alpha158_duplicate"] = frame["alpha158_good"] * 2.0
    frame["alpha158_noise"] = np.cos(np.arange(rows) * 1.7)
    return frame


def test_cognitive_alpha_seed_hierarchy_matches_paper_layers():
    from src.factor import COGALPHA_AGENT_LEVELS, build_cognitive_alpha_seed_candidates

    assert len(COGALPHA_AGENT_LEVELS) == 7
    assert COGALPHA_AGENT_LEVELS[0]["level"] == "market_structure_cycle"
    assert "AgentVolatilityRegime" in COGALPHA_AGENT_LEVELS[0]["agents"]
    assert "AgentComposite" in COGALPHA_AGENT_LEVELS[-1]["agents"]

    candidates = build_cognitive_alpha_seed_candidates()
    candidate_names = {candidate.name for candidate in candidates}
    assert "factor_pressure_drawdown_fisher_10d" in candidate_names
    assert "factor_volume_confirmed_trend_20d" in candidate_names
    assert all(candidate.declared_lookahead == 0 for candidate in candidates)
    assert all(candidate.economic_rationale for candidate in candidates)


def test_cognitive_alpha_quality_checker_rejects_future_leakage():
    from src.factor import CognitiveAlphaCandidate, evaluate_cognitive_alpha_candidates

    frame = _sample_ohlcv_frame()
    leaky = CognitiveAlphaCandidate(
        name="factor_future_return_leak",
        level="geometric_fusion",
        agent="AgentCreative",
        theme="future leakage control",
        required_columns=("close",),
        lookback=1,
        declared_lookahead=1,
        code="df['close'].shift(-1) / df['close'] - 1",
        economic_rationale="Intentionally invalid future return leak used by the test.",
        function=lambda data: data["close"].shift(-1) / data["close"] - 1.0,
    )

    report = evaluate_cognitive_alpha_candidates(
        frame,
        candidates=[leaky],
        label_col="label_return_5d",
        min_observations=30,
    )
    row = report.metrics.set_index("factor").loc["factor_future_return_leak"]

    assert row["quality_status"] == "reject"
    assert row["quality_reason"] == "future_lookahead_declared"
    assert row["selection_tier"] == "drop"
    assert report.selected_feature_names == []


def test_cognitive_alpha_evaluates_codex_and_alpha158_candidates_with_percentile_gates():
    from src.factor import (
        CognitiveAlphaCandidate,
        build_column_alpha_candidates,
        evaluate_cognitive_alpha_candidates,
        materialize_cognitive_alpha_features,
    )

    frame = _sample_ohlcv_frame()
    codex_candidate = CognitiveAlphaCandidate(
        name="factor_codex_known_predictor",
        level="stability_regime_gating",
        agent="AgentStability",
        theme="known predictive control",
        required_columns=("alpha158_good",),
        lookback=5,
        code="df['alpha158_good']",
        economic_rationale="Synthetic predictor used to prove the IC and MI gates select useful factors.",
        function=lambda data: data["alpha158_good"],
    )
    candidates = [
        codex_candidate,
        *build_column_alpha_candidates(
            ["alpha158_good", "alpha158_duplicate", "alpha158_noise"],
            level="alpha158_baseline",
            agent="Alpha158",
        ),
    ]

    report = evaluate_cognitive_alpha_candidates(
        frame,
        candidates=candidates,
        label_col="label_return_5d",
        min_observations=30,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        max_abs_correlation=0.98,
    )
    metrics = report.metrics.set_index("factor")

    assert {"ic", "rank_ic", "icir", "rank_icir", "mutual_information"}.issubset(metrics.columns)
    assert metrics.loc["factor_codex_known_predictor", "selection_tier"] == "elite"
    assert metrics.loc["alpha158_good", "selection_tier"] in {"elite", "qualified"}
    assert metrics.loc["alpha158_duplicate", "redundancy_status"] == "drop_high_correlation"
    assert "factor_codex_known_predictor" in report.selected_feature_names
    assert "alpha158_duplicate" not in report.selected_feature_names
    assert report.summary["workflow"] == "cognitive_alpha_mining"
    assert report.summary["paper_reference"] == "arxiv:2511.18850"

    selected_frame = materialize_cognitive_alpha_features(
        frame,
        candidates,
        selected_names=report.selected_feature_names,
    )
    assert selected_frame.index.equals(frame.index)
    assert selected_frame.columns.tolist() == report.selected_feature_names

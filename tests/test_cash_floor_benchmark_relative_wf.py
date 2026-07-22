from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


EXPERIMENT = "cash_floor_benchmark_relative_walk_forward_v1"


def _card() -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[1]
    card = root / "research" / EXPERIMENT
    return card, json.loads((card / "summary.json").read_text(encoding="utf-8"))


def test_committed_cash_floor_card_is_qlib_native_outer_safe_and_slim() -> None:
    card, summary = _card()
    notebook = json.loads((card / f"{EXPERIMENT}.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_test_use"] == "diagnostic_only_previously_observed"
    assert summary["qlib_dataset"]["api"] == ["DatasetH", "DataHandlerLP", "StaticDataLoader"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["portfolio_contract"]["tradable_assets"] == ["SOXL", "SOXX", "cash"]
    assert summary["portfolio_contract"]["soxs_feature_only"] is True
    assert summary["model"]["targets"] == ["cash_advantage_5d", "future_vol_5d"]
    for name in ("predictions.csv", "positions.csv", "strategy_returns.csv"):
        assert not (card / name).exists()
    for token in ("qlib.init", "D.calendar", "DatasetH", "DataHandlerLP", "R.get_recorder"):
        assert token in source
    assert "最终中文结论" in source
    assert notebook["cells"][1]["execution_count"] is not None
    assert summary["recorder_id"] in json.dumps(notebook)


def test_cash_floor_weights_obey_frozen_portfolio_contract() -> None:
    _, summary = _card()
    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts" / EXPERIMENT
    positions = pd.read_csv(artifact_root / "positions.csv")
    dynamic = positions.loc[
        positions["segment"].eq("development_walk_forward")
        & positions["strategy"].eq("dynamic_cash_floor")
        & positions["cost_bps"].eq(10)
    ].copy()

    weights = dynamic[["soxl_weight", "soxx_weight", "soxs_weight", "cash_weight"]]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert dynamic["cash_weight"].min() >= 0.10 - 1e-12
    assert dynamic["cash_weight"].max() <= 0.60 + 1e-12
    assert dynamic["cash_weight"].mean() >= 0.15
    assert dynamic["soxl_weight"].max() <= 0.27 + 1e-12
    assert dynamic["soxs_weight"].eq(0.0).all()
    risky = dynamic["soxl_weight"] + dynamic["soxx_weight"]
    assert np.allclose(dynamic["soxl_weight"] / risky, 0.30)
    assert (pd.to_datetime(dynamic["signal_date"]) < pd.to_datetime(dynamic["trade_date"])).all()


def test_backtest_records_honest_relative_wealth_failure_and_risk_improvement() -> None:
    card, summary = _card()
    comparison = pd.read_csv(card / "backtest_benchmark_comparison.csv")
    development = comparison.query("segment == 'development_walk_forward'")
    dynamic10 = development.query("strategy == 'dynamic_cash_floor' and cost_bps == 10").iloc[0]
    dynamic25 = development.query("strategy == 'dynamic_cash_floor' and cost_bps == 25").iloc[0]
    soxx10 = development.query("strategy == 'soxx_buyhold' and cost_bps == 10").iloc[0]
    no_cash10 = development.query(
        "strategy == 'no_cash_30soxl_70soxx' and cost_bps == 10"
    ).iloc[0]

    assert dynamic10["terminal_wealth_ratio_vs_soxx"] == pytest.approx(0.8619345, rel=1e-5)
    assert dynamic25["terminal_wealth_ratio_vs_soxx"] == pytest.approx(0.7335657, rel=1e-5)
    assert dynamic10["avg_cash_weight_pct"] == pytest.approx(38.522826, rel=1e-5)
    assert dynamic10["max_drawdown_pct"] > soxx10["max_drawdown_pct"]
    assert dynamic10["max_drawdown_pct"] - no_cash10["max_drawdown_pct"] >= 5.0
    assert summary["promotion_decision"] == "needs_iteration_relative_wealth"
    assert summary["gate_evaluation"]["passed_gate_count"] == 9
    assert summary["gate_evaluation"]["total_gate_count"] == 15
    assert summary["gate_evaluation"]["development_positive_fold_count_10bps"] == 3
    assert summary["bootstrap"]["outperformance_probability"] < 0.60

    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts" / EXPERIMENT
    assert (artifact_root / "predictions.csv").exists()
    assert (artifact_root / "strategy_returns.csv").exists()
    assert (artifact_root / "bootstrap_terminal_wealth.csv").exists()
    assert len(list((artifact_root / "models").glob("*.pkl"))) == 16

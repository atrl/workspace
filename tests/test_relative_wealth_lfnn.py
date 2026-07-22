from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.model.relative_wealth_lfnn import (
    RelativeWealthPolicy,
    RelativeWealthTrainingSpec,
    simulate_policy_paths,
    train_relative_wealth_policy,
)
from src.objective.relative_wealth import (
    aggregate_bootstrap_paths,
    stationary_block_bootstrap_indices,
    target_relative_wealth_loss,
    terminal_relative_wealth_metrics,
)


def test_stationary_bootstrap_is_reproducible_and_bounded() -> None:
    first = stationary_block_bootstrap_indices(
        observation_count=20,
        path_length=12,
        path_count=4,
        expected_block_length=3,
        seed=7,
    )
    second = stationary_block_bootstrap_indices(
        observation_count=20,
        path_length=12,
        path_count=4,
        expected_block_length=3,
        seed=7,
    )
    assert np.array_equal(first, second)
    assert first.shape == (4, 12)
    assert first.min() >= 0
    assert first.max() < 20


def test_bootstrap_aggregation_compounds_returns_and_samples_features_at_rebalance() -> None:
    daily_returns = np.array([[0.10, 0.05], [0.20, -0.10], [0.0, 0.10], [-0.10, 0.0]])
    daily_features = np.arange(8, dtype=float).reshape(4, 2)
    indices = np.array([[0, 1, 2, 3]])
    returns, features = aggregate_bootstrap_paths(
        daily_returns,
        daily_features,
        indices,
        rebalance_days=2,
    )
    assert returns.shape == (1, 2, 2)
    assert returns[0, 0, 0] == pytest.approx(1.10 * 1.20 - 1.0)
    assert returns[0, 0, 1] == pytest.approx(1.05 * 0.90 - 1.0)
    assert np.array_equal(features[0], daily_features[[0, 2]])


def test_relative_wealth_metrics_and_target_loss_use_terminal_ratio() -> None:
    metrics = terminal_relative_wealth_metrics(np.array([1.2, 0.9, 1.1]), np.ones(3))
    assert metrics["outperformance_probability"] == pytest.approx(2.0 / 3.0)
    assert metrics["median_terminal_wealth_ratio"] == pytest.approx(1.1)
    loss = target_relative_wealth_loss(
        torch.tensor([1.21, 1.21]),
        torch.ones(2),
        annual_excess_target=0.10,
        horizon_years=2.0,
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-10)


def test_policy_outputs_feasible_weights_and_tiny_training_runs() -> None:
    returns = np.zeros((16, 3, 2), dtype=np.float32)
    returns[:, :, 0] = 0.04
    returns[:, :, 1] = 0.01
    features = np.zeros((16, 3, 2), dtype=np.float32)
    policy = RelativeWealthPolicy(market_feature_count=2, hidden_size=8)
    simulated = simulate_policy_paths(
        policy,
        torch.tensor(returns),
        torch.tensor(features),
        cost_bps=10,
    )
    weights = simulated["weights"].detach().numpy()
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=-1), 1.0)

    fitted, metrics = train_relative_wealth_policy(
        training_batches=[(returns, features)] * 2,
        validation_returns=returns,
        validation_features=features,
        spec=RelativeWealthTrainingSpec(
            annual_excess_target=0.02,
            hidden_size=8,
            epochs=2,
            paths_per_epoch=16,
            patience=2,
        ),
        seed=3,
        device="cpu",
    )
    assert isinstance(fitted, RelativeWealthPolicy)
    assert metrics["path_count"] == 16
    assert metrics["avg_soxl_weight"] >= 0.0


def test_committed_research_card_is_qlib_native_slim_and_rejected() -> None:
    root = Path(__file__).resolve().parents[1]
    card = root / "research" / "relative_wealth_lfnn_soxx_v1"
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    notebook = json.loads(
        (card / "relative_wealth_lfnn_soxx_v1.ipynb").read_text(encoding="utf-8")
    )
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert summary["promotion_decision"] == (
        "research_reject_relative_wealth_validation_distribution_gates_failed"
    )
    assert summary["selection_uses_outer_test"] is False
    assert summary["objective"]["max_drawdown_in_training_loss"] is False
    assert summary["model"]["seeds"] == [11, 23, 37, 53, 71]
    assert summary["qlib_dataset"]["api"] == ["DatasetH", "DataHandlerLP", "StaticDataLoader"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["workflow_config"] == "workflow.yaml"
    assert "positions.csv" not in summary["sidecars"].values()
    assert "strategy_returns.csv" not in summary["sidecars"].values()
    assert not (card / "positions.csv").exists()
    assert not (card / "strategy_returns.csv").exists()
    recorder_root = Path(summary["recorder_local_dir"])
    artifact_root = recorder_root / "artifacts" / summary["experiment"]
    assert (artifact_root / "positions.csv").is_file()
    assert (artifact_root / "strategy_returns.csv").is_file()
    assert (artifact_root / "bootstrap_terminal_wealth.csv").is_file()
    assert len(list((artifact_root / "models").glob("*.pt"))) == 5
    positions = pd.read_csv(artifact_root / "positions.csv", nrows=1)
    assert positions.loc[0, "signal_date"] < positions.loc[0, "trade_date"]

    for token in ("qlib.init", "D.calendar", "DatasetH", "DataHandlerLP", "R.get_recorder"):
        assert token in notebook_source
    assert "最终中文结论" in notebook_source
    assert notebook["cells"][1]["execution_count"] is not None
    assert summary["recorder_id"] in json.dumps(notebook)

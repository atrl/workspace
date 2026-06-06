from __future__ import annotations

import numpy as np
import pandas as pd

from src.soxl_quant.app.quantbench_objectives import (
    build_soxl_supervised_targets,
    evaluate_prediction_objectives,
    quantbench_objective_families,
)


def test_quantbench_objective_families_cover_paper_losses_and_soxl_utility():
    rows = quantbench_objective_families()
    keys = {row["key"] for row in rows}

    assert {"clf", "ic", "mse", "ranking", "utility", "rl"}.issubset(keys)
    assert all(row["soxl_target"] and row["paper_basis"] for row in rows)


def test_build_soxl_supervised_targets_uses_future_returns_and_drawdowns():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    close = pd.Series([100.0, 110.0, 99.0, 120.0, 132.0, 118.8], index=index)

    targets = build_soxl_supervised_targets(close, horizons=(2,), chip_drawdown_floor=-0.15, bull_capture_return=0.10)

    assert np.isclose(targets.loc[index[0], "fwd_return_2"], -0.01)
    assert np.isclose(targets.loc[index[1], "fwd_return_2"], 120.0 / 110.0 - 1.0)
    assert np.isclose(targets.loc[index[1], "fwd_drawdown_2"], 99.0 / 110.0 - 1.0)
    assert targets.loc[index[1], "chip_retention_label_2"] == 1
    assert targets.loc[index[0], "bull_capture_label_2"] == 0
    assert targets.loc[index[1], "bull_capture_label_2"] == 0
    assert targets.loc[index[2], "bull_capture_label_2"] == 1
    assert pd.isna(targets.loc[index[-1], "fwd_return_2"])


def test_evaluate_prediction_objectives_scores_clf_ic_mse_ranking_and_utility():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    predictions = pd.Series([0.30, -0.20, 0.10, 0.05, 0.10], index=index)
    targets = pd.DataFrame(
        {
            "fwd_return_1": [0.20, -0.10, 0.05, -0.02, 0.01],
            "fwd_positive_1": [1, 0, 1, 0, 1],
            "chip_retention_label_1": [1, 1, 1, 0, 1],
            "bull_capture_label_1": [1, 0, 0, 0, 0],
        },
        index=index,
    )

    metrics = evaluate_prediction_objectives(predictions, targets, horizon=1)

    assert metrics["clf_accuracy"] == 0.8
    assert metrics["mse"] < 0.02
    assert metrics["ic"] > 0.5
    assert metrics["ranking_spread"] > 0
    assert metrics["utility_return"] > 0

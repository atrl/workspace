from __future__ import annotations

import numpy as np
import pandas as pd

from src.soxl_quant.app.quantbench_objectives import build_soxl_supervised_targets
from src.soxl_quant.app.quantbench_supervised import (
    backtest_prediction_strategy,
    build_quantbench_supervised_report,
    build_supervised_summary_payloads,
    fit_quantbench_supervised_models,
    quantbench_supervised_model_specs,
)


def test_backtest_prediction_strategy_uses_lagged_rank_position_and_reports_metrics():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    close = pd.Series([100.0, 102.0, 104.0, 100.0, 105.0, 110.0], index=index)
    predictions = pd.Series([0.10, 0.90, 0.80, 0.20, 0.70, 0.30], index=index)

    metrics = backtest_prediction_strategy(close, predictions, min_position=0.10, max_position=0.90, cost=0.0)

    assert metrics["return"] > 0.0
    assert metrics["buy_and_hold_return"] == 10.0
    assert metrics["max_dd"] >= 0.0
    assert 0.10 <= metrics["avg_pos"] <= 0.90
    assert metrics["trades"] > 0


def test_fit_quantbench_supervised_models_trains_clf_and_mse_specs():
    index = pd.date_range("2024-01-01", periods=80, freq="D")
    signal = np.linspace(-1.0, 1.0, len(index))
    close = pd.Series(100.0 * np.cumprod(1.0 + 0.002 * np.sign(signal + 0.1)), index=index)
    features = pd.DataFrame({"signal": signal, "anti_signal": -signal}, index=index)
    targets = build_soxl_supervised_targets(close, horizons=(1,))

    predictions, metrics = fit_quantbench_supervised_models(
        features,
        targets,
        train_start=index[0],
        train_end=index[49],
        test_start=index[50],
        test_end=index[70],
        horizon=1,
        model_names=("logistic_clf", "ridge_mse"),
    )

    assert set(predictions) == {"logistic_clf", "ridge_mse"}
    assert all(pred.index.min() >= index[50] for pred in predictions.values())
    assert metrics["logistic_clf"]["rows"] > 0
    assert "clf_accuracy" in metrics["logistic_clf"]
    assert "mse" in metrics["ridge_mse"]


def test_quantbench_supervised_model_specs_cover_paper_objectives():
    specs = quantbench_supervised_model_specs()

    assert {"linear_classifier", "linear_regression", "tree", "boosted_tree", "tree_ensemble", "neural_tabular"} <= {
        spec.paper_family for spec in specs
    }
    assert {"clf", "ic", "mse", "ranking"} <= {spec.objective for spec in specs}


def test_build_supervised_summary_payloads_matches_quantbench_screen_schema():
    index = pd.date_range("2020-01-01", periods=900, freq="D")
    close = pd.Series(100.0 * np.cumprod(1.0 + np.sin(np.arange(len(index)) / 30.0) * 0.002), index=index)
    prediction = pd.Series(np.cos(np.arange(len(index)) / 25.0), index=index)

    payloads = build_supervised_summary_payloads(
        close,
        {"ridge_mse": prediction},
        objective_metrics={"ridge_mse": {"ic": 0.1, "rows": 100.0}},
        profile_prefix="quantbench_supervised",
        windows={
            "test_2020": ("2020-01-01", "2020-12-31"),
            "test_2022": ("2021-01-01", "2021-12-31"),
            "test_2023_2024": ("2022-01-01", "2022-06-01"),
        },
    )

    payload = payloads[0]
    assert payload["profile"] == "quantbench_supervised_ridge_mse"
    assert set(payload["slices"]) == {"test_2020", "test_2022", "test_2023_2024"}
    assert {"return", "max_dd", "avg_pos", "buy_and_hold_return", "relative_return"}.issubset(payload["test_proxy"])
    assert payload["objective_metrics"]["ic"] == 0.1


def test_build_quantbench_supervised_report_writes_screen_artifacts(tmp_path):
    index = pd.date_range("2024-01-01", periods=100, freq="D")
    signal = np.sin(np.arange(len(index)) / 8.0)
    close = pd.Series(100.0 * np.cumprod(1.0 + 0.003 * np.sign(signal + 0.2)), index=index)
    features = pd.DataFrame(
        {
            "signal": signal,
            "signal_lag": pd.Series(signal, index=index).shift(1).fillna(0.0),
            "risk_proxy": np.cos(np.arange(len(index)) / 11.0),
        },
        index=index,
    )

    report = build_quantbench_supervised_report(
        features,
        close,
        card_dir=tmp_path,
        root=tmp_path,
        horizon=1,
        model_names=("logistic_clf", "ridge_mse"),
        train_start=index[0],
        train_end=index[59],
        test_start=index[60],
        test_end=index[90],
    )

    assert report["screen"]["records"] == 2
    assert report["data"]["feature_count"] == 3
    assert (tmp_path / "supervised_summary.json").exists()
    assert (tmp_path / "supervised_candidate_screen.csv").read_text(encoding="utf-8").startswith("profile,")
    assert "supervised_payloads_json" in report["artifacts"]

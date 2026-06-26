from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _position_frame(rows: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    cycle = np.sin(np.arange(rows) / 5.0)
    trend = np.linspace(-0.4, 0.6, rows)
    forward_return = 0.012 * np.sign(cycle + trend) + 0.003 * np.cos(np.arange(rows) / 3.0)
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "alpha_good": pd.Series(forward_return).shift(1).fillna(0.0) + trend * 0.01,
            "alpha_gate": cycle,
            "alpha_noise": np.cos(np.arange(rows) * 1.7),
            "label_return_5d": pd.Series(forward_return).rolling(5, min_periods=1).sum().shift(-1).ffill(),
            "forward_return": forward_return,
        }
    )
    return frame


def test_cognitive_alpha_position_model_trains_lightgbm_and_compares_benchmarks(tmp_path):
    from src.workflow import CognitiveAlphaPositionSpec, run_cognitive_alpha_position_model

    frame = _position_frame()
    spec = CognitiveAlphaPositionSpec(
        feature_cols=["alpha_good", "alpha_gate", "alpha_noise"],
        train=("2024-01-02", "2024-05-31"),
        validation=("2024-06-03", "2024-07-31"),
        test=("2024-08-01", "2024-09-06"),
        model_specs=[
            {"model": "lgbm_classifier", "positive_threshold": 0.0},
            {"model": "logistic_regression", "positive_threshold": 0.0},
            {"model": "ridge", "task": "regression"},
            {"model": "hist_gradient_boosting_classifier", "positive_threshold": 0.0},
        ],
        position_specs=[
            {
                "position_mode": "probability",
                "threshold": 0.45,
                "scale": 1.0,
                "max_position": 1.0,
                "rebalance_threshold": 0.15,
                "min_hold_days": 2,
                "target_vol": 0.25,
                "realized_vol_col": "realized_vol_20",
                "min_vol_scale": 0.0,
                "max_vol_scale": 1.0,
                "risk_gate_col": "alpha_gate",
                "risk_gate_threshold": -0.80,
                "risk_gate_max_position": 0.50,
            },
            {
                "position_mode": "binary",
                "threshold": 0.55,
                "scale": 1.0,
                "max_position": 1.0,
                "rebalance_threshold": 0.20,
                "min_hold_days": 2,
            },
        ],
        min_validation_trade_count=1,
        max_validation_reversal_ratio=0.75,
        validation_max_drawdown_floor=-80.0,
        validation_max_daily_loss_floor=-10.0,
        max_validation_total_turnover=100.0,
        test_max_drawdown_floor=-80.0,
        test_max_daily_loss_floor=-10.0,
        max_test_total_turnover=100.0,
    )
    frame["realized_vol_20"] = frame["forward_return"].rolling(20, min_periods=5).std().fillna(0.02) * np.sqrt(252)

    result = run_cognitive_alpha_position_model(
        frame,
        spec=spec,
        benchmarks=[
            {"name": "soxl_backtest", "return_pct": 5.0, "max_drawdown_pct": -8.0},
            {"name": "constrained_weight_reference", "return_pct": 80.0, "max_drawdown_pct": -12.0},
            {"name": "soxl_buyhold", "return_pct": 15.0, "max_drawdown_pct": -18.0},
            {"name": "previous_factor_enhanced", "return_pct": 7.5, "max_drawdown_pct": -9.0},
        ],
    )

    assert result.summary["workflow"] == "cognitive_alpha_position_model"
    assert result.summary["model_families"] == [
        "hist_gradient_boosting_classifier",
        "lgbm_classifier",
        "logistic_regression",
        "ridge",
    ]
    assert result.summary["feature_count"] == 3
    assert result.summary["risk_gates"] == {
        "validation_max_drawdown_floor": -80.0,
        "validation_max_daily_loss_floor": -10.0,
        "max_validation_total_turnover": 100.0,
        "test_max_drawdown_floor": -80.0,
        "test_max_daily_loss_floor": -10.0,
        "max_test_total_turnover": 100.0,
        "selected_validation_passes": True,
        "selected_test_passes": True,
        "selected_passes_all": True,
    }
    assert result.summary["selected_candidate"]["selected_by"] == "validation_selection_objective_score"
    assert result.summary["selected_candidate"]["test"]["max_position"] <= 1.0
    assert result.summary["selected_candidate"]["test"]["days_above_100"] == 0
    assert result.summary["selected_candidate"]["test"]["max_daily_loss_pct"] >= -10.0
    assert result.metrics["max_daily_loss_pct"].notna().all()
    assert result.summary["comparison"]["beats_soxl_backtest"] in {True, False}
    assert result.summary["comparison"]["beats_constrained_weight_reference"] is False
    assert result.summary["promotion_decision"] == "not_promoted_under_constrained_weight_reference"
    assert set(result.benchmark_comparison["benchmark"]) == {
        "cognitive_alpha_selected",
        "soxl_backtest",
        "constrained_weight_reference",
        "soxl_buyhold",
        "previous_factor_enhanced",
    }
    assert result.latest_signal["as_of"] == "2024-09-06"
    assert result.latest_signal["action"] in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= result.latest_signal["target_position"] <= 1.0

    paths = result.write_sidecars(tmp_path)
    assert set(paths) == {"model_selection", "benchmark_comparison", "summary", "latest_signal"}
    assert json.loads(paths["latest_signal"].read_text(encoding="utf-8"))["as_of"] == "2024-09-06"
    assert not (tmp_path / "cognitive_alpha_position_positions.csv").exists()

    all_paths = result.write_sidecars(tmp_path / "recorder_upload", include_positions=True)
    assert "positions" in all_paths
    assert all_paths["positions"].exists()


def test_cognitive_alpha_position_model_fails_closed_on_missing_features():
    from src.workflow import CognitiveAlphaPositionSpec, run_cognitive_alpha_position_model

    spec = CognitiveAlphaPositionSpec(
        feature_cols=["missing_alpha"],
        train=("2024-01-02", "2024-05-31"),
        validation=("2024-06-03", "2024-07-31"),
        test=("2024-08-01", "2024-09-06"),
        model_specs=[{"model": "lgbm_classifier"}],
    )

    try:
        run_cognitive_alpha_position_model(_position_frame(), spec=spec, benchmarks=[])
    except ValueError as exc:
        assert "missing cognitive alpha position columns" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing feature should fail closed")


def test_cognitive_alpha_weighted_factor_search_selects_direct_backtest_above_reference(tmp_path):
    from src.workflow import (
        CognitiveAlphaWeightedFactorSearchSpec,
        run_cognitive_alpha_weighted_factor_search,
    )

    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-02", periods=8, freq="B"),
            "canary_alpha": [1, 1, 0, 1, 1, 0, 1, 1],
            "tsmom_alpha": [0, 1, 1, 0, 1, 0, 1, 0],
            "forward_return": [0.20, 0.10, -0.20, 0.18, 0.12, -0.10, 0.16, 0.08],
        }
    )
    spec = CognitiveAlphaWeightedFactorSearchSpec(
        component_cols=["canary_alpha", "tsmom_alpha"],
        candidates=[
            {
                "candidate": "canary_tsmom_direct",
                "weights": {"canary_alpha": 1.0, "tsmom_alpha": 0.0501},
                "policy": {"min_position": 0.0, "max_position": 1.0, "rebalance_threshold": 0.0},
            },
            {
                "candidate": "weak_single_factor",
                "weights": {"tsmom_alpha": 0.0501},
                "policy": {"min_position": 0.0, "max_position": 1.0, "rebalance_threshold": 0.0},
            },
        ],
        min_effective_component_count=2,
        constrained_reference_name="constrained_weight_reference",
    )

    result = run_cognitive_alpha_weighted_factor_search(
        frame,
        spec=spec,
        benchmarks=[
            {"name": "constrained_weight_reference", "return_pct": 20.0, "max_drawdown_pct": -12.0},
            {"name": "soxl_backtest", "return_pct": 10.0, "max_drawdown_pct": -8.0},
        ],
    )

    assert result.summary["workflow"] == "cognitive_alpha_weighted_factor_search"
    assert result.summary["selected_candidate"]["candidate"] == "canary_tsmom_direct"
    assert result.summary["selected_candidate"]["max_position"] <= 1.0
    assert result.summary["selected_candidate"]["min_position"] >= 0.0
    assert result.summary["selected_candidate"]["effective_component_count"] == 2
    assert result.summary["comparison"]["beats_constrained_weight_reference"] is True
    assert result.summary["promotion_decision"] == "diagnostic_direct_factor_candidate"

    paths = result.write_sidecars(tmp_path)
    assert set(paths) == {"candidate_metrics", "benchmark_comparison", "summary"}
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["selected_candidate"]["candidate"] == "canary_tsmom_direct"

from __future__ import annotations

import numpy as np
import pandas as pd

from src.soxl_quant.app.quantbench_ensemble import (
    QuantBenchPathSelectorConfig,
    apply_quantbench_ensemble_drawdown_guard,
    apply_quantbench_hazard_guard,
    apply_quantbench_volatility_budget_guard,
    build_quantbench_path_ensemble_report,
    build_quantbench_supervised_ensemble_report,
    default_quantbench_path_selector_configs,
    fit_quantbench_drawdown_hazard_model,
    fit_quantbench_rolling_drawdown_hazard_model,
    select_quantbench_path_ensemble,
)
from src.soxl_quant.app.quantbench_screen import QuantBenchGate


def test_select_quantbench_path_ensemble_routes_stress_and_bull_regimes_with_audit_columns():
    index = pd.date_range("2024-01-01", periods=7, freq="D")
    paths = {
        "lowdd": pd.Series([0.04] * len(index), index=index),
        "neutral": pd.Series([0.30] * len(index), index=index),
        "attack": pd.Series([0.80] * len(index), index=index),
    }
    risk_frame = pd.DataFrame(
        {
            "regime_risk": [0.85, 0.78, 0.45, 0.38, 0.30, 0.25, 0.20],
            "cum_drop_30": [-0.22, -0.18, -0.08, -0.03, -0.01, 0.00, 0.00],
            "drawdown_63": [-0.28, -0.22, -0.10, -0.07, -0.04, -0.02, -0.01],
            "mom_21": [-0.10, -0.04, 0.00, 0.02, 0.05, 0.07, 0.09],
            "trend_gap_63": [-0.12, -0.06, -0.01, 0.01, 0.03, 0.05, 0.06],
        },
        index=index,
    )
    config = QuantBenchPathSelectorConfig(
        name="unit_selector",
        defensive_path="lowdd",
        neutral_path="neutral",
        offensive_path="attack",
        stress_floor=0.10,
        stress_cap=0.18,
        bull_floor=0.55,
        bull_cap=0.90,
        stress_risk_threshold=0.70,
        stress_drawdown_threshold=-0.20,
        bull_risk_threshold=0.35,
        bull_momentum_threshold=0.04,
        bull_trend_threshold=0.02,
        bull_drawdown_floor=-0.08,
    )

    daily = select_quantbench_path_ensemble(paths, risk_frame, config)

    assert daily["macro_regime"].tolist() == ["stress", "stress", "neutral", "neutral", "bull", "bull", "bull"]
    assert daily["selected_path"].tolist() == ["lowdd", "lowdd", "neutral", "neutral", "attack", "attack", "attack"]
    assert daily["position"].round(4).tolist() == [0.10, 0.10, 0.30, 0.30, 0.80, 0.80, 0.80]
    assert {"lowdd_position", "neutral_position", "attack_position", "regime_risk", "drawdown_63"} <= set(
        daily.columns
    )


def test_apply_quantbench_ensemble_drawdown_guard_caps_after_account_loss_and_reenters_on_bull():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    daily = pd.DataFrame(
        {
            "datetime": index,
            "position": [0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
            "macro_regime": ["bull", "bull", "stress", "stress", "bull", "bull"],
        }
    )
    close = pd.Series([100.0, 120.0, 80.0, 80.0, 120.0, 150.0], index=index)
    config = QuantBenchPathSelectorConfig(
        name="guarded",
        defensive_path="lowdd",
        offensive_path="attack",
        portfolio_drawdown_trigger=0.10,
        portfolio_drawdown_cap=0.05,
        portfolio_drawdown_release=0.04,
        portfolio_reentry_floor=0.40,
        portfolio_reentry_cap=0.55,
    )

    guarded = apply_quantbench_ensemble_drawdown_guard(daily, close, config, cost=0.0)

    assert guarded["portfolio_drawdown_guard"].tolist() == [False, False, True, True, True, True]
    assert guarded["position"].round(4).tolist() == [0.80, 0.80, 0.05, 0.05, 0.55, 0.55]


def test_apply_quantbench_ensemble_drawdown_guard_blocks_reentry_during_deep_drawdown():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    daily = pd.DataFrame(
        {
            "datetime": index,
            "position": [0.80, 0.80, 0.80, 0.80],
            "macro_regime": ["bull", "stress", "bull", "bull"],
        }
    )
    close = pd.Series([100.0, 80.0, 80.0, 120.0], index=index)
    config = QuantBenchPathSelectorConfig(
        name="deep_guard",
        defensive_path="lowdd",
        offensive_path="attack",
        portfolio_drawdown_trigger=0.10,
        portfolio_drawdown_cap=0.10,
        portfolio_drawdown_release=0.04,
        portfolio_reentry_floor=0.40,
        portfolio_reentry_cap=0.50,
        portfolio_reentry_max_drawdown=0.15,
    )

    guarded = apply_quantbench_ensemble_drawdown_guard(daily, close, config, cost=0.0)

    assert guarded["portfolio_drawdown_guard"].tolist() == [False, True, True, True]
    assert guarded["position"].round(4).tolist() == [0.80, 0.10, 0.10, 0.50]
    assert guarded.loc[2, "portfolio_drawdown"] > guarded.loc[3, "portfolio_drawdown"]


def test_apply_quantbench_ensemble_drawdown_guard_reenters_only_after_lagged_recovery_confirmation():
    index = pd.date_range("2024-01-01", periods=7, freq="D")
    daily = pd.DataFrame(
        {
            "datetime": index,
            "position": [0.80] * len(index),
            "macro_regime": ["bull"] * len(index),
        }
    )
    close = pd.Series([100.0, 80.0, 80.0, 81.0, 100.0, 110.0, 120.0], index=index)
    config = QuantBenchPathSelectorConfig(
        name="confirmed_reentry",
        defensive_path="lowdd",
        offensive_path="attack",
        portfolio_drawdown_trigger=0.10,
        portfolio_drawdown_cap=0.10,
        portfolio_drawdown_release=0.04,
        portfolio_reentry_floor=0.35,
        portfolio_reentry_cap=0.50,
        portfolio_reentry_max_drawdown=0.12,
        confirmed_reentry_floor=0.35,
        confirmed_reentry_cap=0.45,
        confirmed_reentry_ma_window=3,
        confirmed_reentry_momentum_window=2,
        confirmed_reentry_high_window=3,
        confirmed_reentry_momentum_threshold=0.05,
        confirmed_reentry_drawdown_floor=-0.08,
    )

    guarded = apply_quantbench_ensemble_drawdown_guard(daily, close, config, cost=0.0)

    assert guarded["position"].round(4).tolist() == [0.80, 0.10, 0.10, 0.10, 0.10, 0.45, 0.45]
    assert guarded["confirmed_reentry"].tolist() == [False, False, False, False, False, True, True]
    assert guarded.loc[4, "confirmed_reentry_momentum"] > guarded.loc[3, "confirmed_reentry_momentum"]


def test_apply_quantbench_ensemble_drawdown_guard_applies_lagged_drawdown_budget_cap():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    daily = pd.DataFrame(
        {
            "datetime": index,
            "position": [0.80] * len(index),
            "macro_regime": ["bull"] * len(index),
        }
    )
    close = pd.Series([100.0, 110.0, 90.0, 90.0, 120.0], index=index)
    config = QuantBenchPathSelectorConfig(
        name="budget_guard",
        defensive_path="lowdd",
        offensive_path="attack",
        portfolio_drawdown_budget_start=0.05,
        portfolio_drawdown_budget_full=0.20,
        portfolio_drawdown_budget_start_cap=0.60,
        portfolio_drawdown_budget_min_cap=0.10,
    )

    guarded = apply_quantbench_ensemble_drawdown_guard(daily, close, config, cost=0.0)

    assert guarded.loc[1, "position"] == 0.80
    assert not bool(guarded.loc[1, "portfolio_drawdown_budget_guard"])
    assert guarded.loc[2, "position"] < 0.60
    assert guarded.loc[2, "position"] >= 0.10
    assert bool(guarded.loc[2, "portfolio_drawdown_budget_guard"])
    assert guarded.loc[2, "portfolio_drawdown_budget_cap"] < guarded.loc[1, "portfolio_drawdown_budget_cap"]


def test_apply_quantbench_ensemble_drawdown_guard_preserves_stress_chip_floor_inside_budget_cap():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    daily = pd.DataFrame(
        {
            "datetime": index,
            "position": [0.80] * len(index),
            "macro_regime": ["bull", "bull", "stress", "stress"],
        }
    )
    close = pd.Series([100.0, 80.0, 70.0, 70.0], index=index)
    config = QuantBenchPathSelectorConfig(
        name="budget_chip_floor",
        defensive_path="lowdd",
        offensive_path="attack",
        portfolio_drawdown_budget_start=0.05,
        portfolio_drawdown_budget_full=0.12,
        portfolio_drawdown_budget_start_cap=0.20,
        portfolio_drawdown_budget_min_cap=0.02,
        portfolio_drawdown_budget_stress_floor=0.10,
    )

    guarded = apply_quantbench_ensemble_drawdown_guard(daily, close, config, cost=0.0)

    assert guarded.loc[2, "portfolio_drawdown_budget_cap"] == 0.02
    assert guarded.loc[2, "position"] == 0.10
    assert bool(guarded.loc[2, "portfolio_drawdown_budget_guard"])


def test_apply_quantbench_hazard_guard_caps_before_account_drawdown_with_audit_columns():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    daily = pd.DataFrame(
        {
            "datetime": index,
            "position": [0.80, 0.80, 0.80, 0.80, 0.80],
            "macro_regime": ["bull"] * 5,
        }
    )
    hazard = pd.Series([0.10, 0.85, 0.90, 0.20, 0.10], index=index)
    config = QuantBenchPathSelectorConfig(
        name="hazard_guarded",
        defensive_path="lowdd",
        offensive_path="attack",
        hazard_threshold=0.80,
        hazard_cap=0.12,
    )

    guarded = apply_quantbench_hazard_guard(daily, hazard, config)

    assert guarded["hazard_guard"].tolist() == [False, True, True, False, False]
    assert guarded["position"].round(4).tolist() == [0.80, 0.12, 0.12, 0.80, 0.80]
    assert guarded["hazard_score"].round(2).tolist() == [0.10, 0.85, 0.90, 0.20, 0.10]


def test_apply_quantbench_volatility_budget_guard_uses_lagged_realized_volatility():
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    daily = pd.DataFrame(
        {
            "datetime": index,
            "position": [0.80] * len(index),
            "macro_regime": ["bull"] * len(index),
        }
    )
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 60.0, 61.0, 62.0, 63.0], index=index)
    config = QuantBenchPathSelectorConfig(
        name="vol_budget",
        defensive_path="lowdd",
        offensive_path="attack",
        volatility_target=0.20,
        volatility_bull_target=0.20,
        volatility_lookback=2,
        volatility_min_periods=2,
        volatility_floor=0.05,
    )

    guarded = apply_quantbench_volatility_budget_guard(daily, close, config)

    assert guarded.loc[4, "position"] == 0.80
    assert not bool(guarded.loc[4, "volatility_budget_guard"])
    assert guarded.loc[5, "position"] < 0.20
    assert guarded.loc[5, "position"] >= 0.05
    assert bool(guarded.loc[5, "volatility_budget_guard"])
    assert guarded.loc[5, "realized_volatility"] > guarded.loc[4, "realized_volatility"]


def test_apply_quantbench_volatility_budget_guard_uses_regime_specific_budget():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    daily = pd.DataFrame(
        {
            "datetime": index,
            "position": [0.80] * len(index),
            "macro_regime": ["stress", "bull", "stress", "bull", "stress", "bull"],
        }
    )
    close = pd.Series([100.0, 112.0, 96.0, 108.0, 92.0, 103.0], index=index)
    config = QuantBenchPathSelectorConfig(
        name="mode_budget",
        defensive_path="lowdd",
        offensive_path="attack",
        volatility_target=0.40,
        volatility_stress_target=0.12,
        volatility_bull_target=0.60,
        volatility_lookback=2,
        volatility_min_periods=2,
        volatility_floor=0.03,
    )

    guarded = apply_quantbench_volatility_budget_guard(daily, close, config)

    assert guarded.loc[4, "macro_regime"] == "stress"
    assert guarded.loc[5, "macro_regime"] == "bull"
    assert guarded.loc[5, "realized_volatility"] > 0.0
    assert guarded.loc[5, "volatility_budget_target"] > guarded.loc[4, "volatility_budget_target"]
    assert guarded.loc[5, "position"] > guarded.loc[4, "position"]


def test_default_quantbench_path_selector_configs_include_volatility_budget_candidate():
    configs = default_quantbench_path_selector_configs()

    budget_configs = [
        config for config in configs if config.volatility_target > 0.0 or config.portfolio_drawdown_budget_start > 0.0
    ]
    names = {config.name for config in budget_configs}
    assert "quantbench_ensemble_vol_budget_rf_logistic_v1" in names
    assert "quantbench_ensemble_confirmed_reentry_rf_logistic_v1" in names
    assert "quantbench_ensemble_drawdown_budget_guarded_rf_logistic_v1" in names
    candidate = next(config for config in budget_configs if config.name == "quantbench_ensemble_vol_budget_rf_logistic_v1")
    assert candidate.volatility_stress_target < candidate.volatility_bull_target
    assert candidate.portfolio_drawdown_trigger > 0.0
    assert candidate.portfolio_reentry_max_drawdown < 1.0
    assert candidate.confirmed_reentry_floor > 0.0
    reentry_candidate = next(
        config for config in budget_configs if config.name == "quantbench_ensemble_confirmed_reentry_rf_logistic_v1"
    )
    assert reentry_candidate.volatility_floor >= 0.10
    assert reentry_candidate.volatility_stress_target >= 0.30
    assert reentry_candidate.portfolio_drawdown_cap >= 0.12
    budget_candidate = next(
        config for config in budget_configs if config.name == "quantbench_ensemble_drawdown_budget_guarded_rf_logistic_v1"
    )
    assert budget_candidate.portfolio_drawdown_budget_full <= 0.18
    assert budget_candidate.portfolio_drawdown_budget_min_cap <= 0.03


def test_fit_quantbench_drawdown_hazard_model_predicts_oos_future_drawdown_risk():
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    stress_signal = np.r_[np.zeros(30), np.ones(30), np.zeros(30), np.ones(30)]
    returns = np.where(stress_signal > 0.5, -0.02, 0.003)
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index)
    features = pd.DataFrame(
        {
            "stress_signal": stress_signal,
            "safe_signal": 1.0 - stress_signal,
        },
        index=index,
    )

    hazard, metrics = fit_quantbench_drawdown_hazard_model(
        features,
        close,
        train_start=index[0],
        train_end=index[79],
        test_start=index[80],
        test_end=index[109],
        horizon=5,
        drawdown_threshold=-0.05,
    )

    assert hazard.index.min() >= index[80]
    assert metrics["train_rows"] > 0
    assert metrics["test_rows"] > 0
    assert hazard.loc[index[90]:index[109]].mean() > hazard.loc[index[80]:index[89]].mean()


def test_fit_quantbench_rolling_drawdown_hazard_model_refits_without_future_label_leakage():
    index = pd.date_range("2024-01-01", periods=180, freq="D")
    stress_signal = np.r_[np.zeros(60), np.ones(40), np.zeros(40), np.ones(40)]
    returns = np.where(stress_signal > 0.5, -0.018, 0.004)
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index)
    features = pd.DataFrame(
        {
            "stress_signal": stress_signal,
            "safe_signal": 1.0 - stress_signal,
        },
        index=index,
    )

    hazard, metrics = fit_quantbench_rolling_drawdown_hazard_model(
        features,
        close,
        test_start=index[90],
        test_end=index[169],
        horizon=5,
        drawdown_threshold=-0.05,
        min_train_rows=40,
        refit_step_rows=20,
        train_lookback_rows=90,
    )

    assert hazard.index.min() >= index[90]
    assert hazard.index.max() <= index[169]
    assert metrics["fold_count"] >= 3
    assert all(pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["predict_start"]) for fold in metrics["folds"])
    assert all(pd.Timestamp(fold["label_cutoff"]) <= pd.Timestamp(fold["train_end"]) for fold in metrics["folds"])
    assert hazard.loc[index[140]:index[169]].mean() > hazard.loc[index[100]:index[119]].mean()


def test_build_quantbench_path_ensemble_report_writes_runtime_screen_artifacts(tmp_path):
    index = pd.date_range("2020-01-01", periods=90, freq="D")
    close = pd.Series(100.0 * np.cumprod(1.0 + np.sin(np.arange(len(index)) / 8.0) * 0.01), index=index)
    paths = {
        "lowdd": pd.Series(0.12, index=index),
        "neutral": pd.Series(0.30, index=index),
        "attack": pd.Series(0.75, index=index),
    }
    risk_frame = pd.DataFrame(
        {
            "regime_risk": np.r_[np.full(30, 0.80), np.full(30, 0.45), np.full(30, 0.20)],
            "cum_drop_30": np.r_[np.full(30, -0.22), np.zeros(60)],
            "drawdown_63": np.r_[np.full(30, -0.25), np.full(30, -0.05), np.zeros(30)],
            "mom_21": np.r_[np.full(60, -0.02), np.full(30, 0.08)],
            "trend_gap_63": np.r_[np.full(60, -0.01), np.full(30, 0.05)],
        },
        index=index,
    )
    config = QuantBenchPathSelectorConfig(
        name="unit_selector",
        defensive_path="lowdd",
        neutral_path="neutral",
        offensive_path="attack",
        stress_floor=0.10,
        stress_cap=0.18,
    )
    gate = QuantBenchGate(
        max_drawdown_pct=99.0,
        min_stress_relative_return=-999.0,
        min_stress_avg_pos=0.0,
        max_stress_avg_pos=1.0,
        min_bull_capture=-999.0,
        min_proxy_return=-999.0,
    )

    report = build_quantbench_path_ensemble_report(
        close,
        paths,
        risk_frame,
        card_dir=tmp_path,
        root=tmp_path,
        configs=(config,),
        gate=gate,
        windows={
            "test_2020": (index[0], index[29]),
            "test_2022": (index[30], index[59]),
            "test_2023_2024": (index[60], index[-1]),
        },
    )

    assert report["screen"]["records"] == 1
    assert report["data"]["path_count"] == 3
    assert (tmp_path / "ensemble_summary.json").exists()
    assert (tmp_path / "ensemble_payloads.json").exists()
    assert (tmp_path / "ensemble_candidate_screen.csv").read_text(encoding="utf-8").startswith("profile,")
    assert (tmp_path / "unit_selector_soxl_daily.csv").exists()


def test_build_quantbench_supervised_ensemble_report_trains_paths_inside_src(tmp_path):
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    signal = np.sin(np.arange(len(index)) / 8.0)
    close = pd.Series(100.0 * np.cumprod(1.0 + 0.003 * np.sign(signal + 0.1)), index=index)
    features = pd.DataFrame(
        {
            "signal": signal,
            "signal_lag": pd.Series(signal, index=index).shift(1).fillna(0.0),
            "regime_risk": np.where(np.arange(len(index)) < 80, 0.70, 0.20),
            "cum_drop_30": np.where(np.arange(len(index)) < 80, -0.20, 0.0),
            "drawdown_63": np.where(np.arange(len(index)) < 80, -0.22, -0.02),
            "mom_21": np.where(np.arange(len(index)) < 80, -0.02, 0.08),
            "trend_gap_63": np.where(np.arange(len(index)) < 80, -0.01, 0.05),
        },
        index=index,
    )
    config = QuantBenchPathSelectorConfig(
        name="unit_supervised_selector",
        defensive_path="quantbench_supervised_ridge_mse",
        neutral_path="quantbench_supervised_ridge_mse",
        offensive_path="quantbench_supervised_logistic_clf",
        stress_floor=0.10,
        stress_cap=0.20,
    )
    gate = QuantBenchGate(
        max_drawdown_pct=99.0,
        min_stress_relative_return=-999.0,
        min_stress_avg_pos=0.0,
        max_stress_avg_pos=1.0,
        min_bull_capture=-999.0,
        min_proxy_return=-999.0,
    )

    report = build_quantbench_supervised_ensemble_report(
        features,
        close,
        card_dir=tmp_path,
        root=tmp_path,
        horizon=1,
        model_names=("logistic_clf", "ridge_mse"),
        train_start=index[0],
        train_end=index[69],
        test_start=index[70],
        test_end=index[105],
        risk_configs=(),
        ensemble_configs=(config,),
        gate=gate,
        hazard_enabled=False,
        windows={
            "test_2020": (index[70], index[80]),
            "test_2022": (index[81], index[93]),
            "test_2023_2024": (index[94], index[105]),
        },
    )

    assert report["screen"]["records"] == 1
    assert report["data"]["path_count"] == 2
    assert report["data"]["model_names"] == ["logistic_clf", "ridge_mse"]
    assert (tmp_path / "ensemble_summary.json").exists()
    assert (tmp_path / "unit_supervised_selector_soxl_daily.csv").exists()


def test_build_quantbench_supervised_ensemble_report_writes_hazard_metrics(tmp_path):
    index = pd.date_range("2024-01-01", periods=140, freq="D")
    stress_signal = np.r_[np.zeros(35), np.ones(35), np.zeros(35), np.ones(35)]
    returns = np.where(stress_signal > 0.5, -0.012, 0.004)
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index)
    features = pd.DataFrame(
        {
            "signal": np.sin(np.arange(len(index)) / 6.0),
            "stress_signal": stress_signal,
            "regime_risk": np.where(stress_signal > 0.5, 0.75, 0.20),
            "cum_drop_30": np.where(stress_signal > 0.5, -0.22, 0.0),
            "drawdown_63": np.where(stress_signal > 0.5, -0.20, -0.02),
            "mom_21": np.where(stress_signal > 0.5, -0.03, 0.08),
            "trend_gap_63": np.where(stress_signal > 0.5, -0.02, 0.05),
        },
        index=index,
    )
    config = QuantBenchPathSelectorConfig(
        name="unit_hazard_selector",
        defensive_path="quantbench_supervised_ridge_mse",
        neutral_path="quantbench_supervised_ridge_mse",
        offensive_path="quantbench_supervised_logistic_clf",
        hazard_threshold=0.50,
        hazard_cap=0.10,
    )
    gate = QuantBenchGate(
        max_drawdown_pct=99.0,
        min_stress_relative_return=-999.0,
        min_stress_avg_pos=0.0,
        max_stress_avg_pos=1.0,
        min_bull_capture=-999.0,
        min_proxy_return=-999.0,
    )

    report = build_quantbench_supervised_ensemble_report(
        features,
        close,
        card_dir=tmp_path,
        root=tmp_path,
        horizon=5,
        model_names=("logistic_clf", "ridge_mse"),
        train_start=index[0],
        train_end=index[89],
        test_start=index[90],
        test_end=index[124],
        risk_configs=(),
        ensemble_configs=(config,),
        gate=gate,
        hazard_enabled=True,
        hazard_mode="rolling",
        hazard_drawdown_threshold=-0.04,
        hazard_min_train_rows=40,
        hazard_refit_step_rows=10,
        hazard_train_lookback_rows=80,
        windows={
            "test_2020": (index[90], index[99]),
            "test_2022": (index[100], index[112]),
            "test_2023_2024": (index[113], index[124]),
        },
    )

    assert report["data"]["hazard_enabled"] is True
    assert report["data"]["hazard_mode"] == "rolling"
    assert "ensemble_hazard_metrics_json" in report["artifacts"]
    assert (tmp_path / "ensemble_hazard_metrics.json").exists()
    metrics = pd.read_json(tmp_path / "ensemble_hazard_metrics.json", typ="series")
    assert metrics["mode"] == "rolling"
    assert metrics["fold_count"] >= 1
    daily = pd.read_csv(tmp_path / "unit_hazard_selector_soxl_daily.csv")
    assert "hazard_score" in daily.columns
    assert "hazard_guard" in daily.columns

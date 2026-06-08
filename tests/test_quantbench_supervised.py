from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.soxl_quant.app.quantbench_objectives import build_soxl_supervised_targets
from src.soxl_quant.app.quantbench_supervised import (
    RiskAwarePositionConfig,
    backtest_prediction_strategy,
    build_quantbench_supervised_report,
    build_supervised_summary_payloads,
    fit_quantbench_rolling_supervised_models,
    fit_quantbench_supervised_models,
    quantbench_supervised_model_specs,
    risk_aware_position_series,
)


FULL_PAPER_ARCHITECTURE_MODEL_NAMES = (
    "logistic_clf",
    "ridge_ic",
    "ridge_mse",
    "random_forest_clf",
    "hist_gradient_mse",
    "xgboost_mse",
    "lightgbm_mse",
    "catboost_rank",
    "extra_trees_rank",
    "mlp_mse",
    "sequence_lag_ridge_mse",
    "lstm_mse",
    "gru_mse",
    "alstm_mse",
    "sfm_mse",
    "darnn_mse",
    "multi_scale_rnn_mse",
    "dlstm_mse",
    "hawkes_gru_mse",
    "mstr_i_mse",
    "tcn_mse",
    "mlp_mixer_mse",
    "dlinear_mse",
    "informer_mse",
    "patch_tst_mse",
    "autoformer_mse",
    "fedformer_mse",
    "pyraformer_mse",
    "tft_mse",
    "market_graph_mse",
    "gcn_mse",
    "gat_mse",
    "rgcn_mse",
    "hats_mse",
    "rsr_mse",
    "hatr_i_mse",
    "hypergraph_mse",
    "estimate_mse",
    "thgnn_mse",
    "sthan_mse",
    "sthcn_mse",
    "sthgcn_mse",
    "dtml_mse",
    "crossformer_mse",
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


def test_risk_aware_position_series_caps_stress_and_restores_bull_participation():
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    predictions = pd.Series(np.linspace(0.1, 0.8, len(index)), index=index)
    risk_frame = pd.DataFrame(
        {
            "regime_risk": [0.90, 0.85, 0.30, 0.25, 0.20, 0.20, 0.18, 0.16],
            "cum_drop_30": [-0.24, -0.18, -0.05, -0.02, -0.01, -0.01, -0.01, -0.01],
            "drawdown_63": [-0.30, -0.22, -0.06, -0.04, -0.03, -0.02, -0.02, -0.01],
            "mom_21": [-0.10, -0.06, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
            "trend_gap_63": [-0.12, -0.08, 0.00, 0.01, 0.03, 0.04, 0.05, 0.06],
        },
        index=index,
    )
    config = RiskAwarePositionConfig(
        name="test_overlay",
        stress_cap=0.18,
        defensive_cap=0.35,
        bull_floor=0.55,
        bull_cap=0.90,
        stress_risk_threshold=0.70,
        bull_risk_threshold=0.35,
        bull_momentum_threshold=0.03,
        bull_drawdown_floor=-0.08,
    )

    positions = risk_aware_position_series(
        predictions,
        risk_frame=risk_frame,
        config=config,
        min_position=0.10,
        max_position=0.90,
    )

    assert positions.loc[index[:2]].max() <= 0.18
    assert positions.loc[index[-3:]].min() >= 0.55
    assert positions.max() <= 0.90


def test_backtest_prediction_strategy_portfolio_drawdown_guard_reduces_crash_damage():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    close = pd.Series([100.0, 112.0, 92.0, 70.0, 50.0, 52.0, 58.0, 70.0, 88.0, 110.0], index=index)
    predictions = pd.Series(np.linspace(0.8, 1.0, len(index)), index=index)
    risk_frame = pd.DataFrame(
        {
            "regime_risk": [0.10, 0.15, 0.72, 0.85, 0.80, 0.55, 0.30, 0.25, 0.20, 0.18],
            "cum_drop_30": [0.00, 0.00, -0.18, -0.32, -0.28, -0.20, -0.10, -0.05, -0.02, 0.00],
            "drawdown_63": [0.00, 0.00, -0.18, -0.35, -0.30, -0.22, -0.12, -0.07, -0.03, 0.00],
            "mom_21": [0.02, 0.04, -0.08, -0.20, -0.10, -0.04, 0.02, 0.04, 0.06, 0.08],
            "trend_gap_63": [0.01, 0.02, -0.05, -0.12, -0.10, -0.05, 0.00, 0.02, 0.04, 0.06],
        },
        index=index,
    )
    no_guard_config = RiskAwarePositionConfig(
        name="no_guard_test",
        defensive_cap=0.80,
        stress_cap=0.70,
        crash_cap=0.60,
    )
    risk_config = RiskAwarePositionConfig(
        name="guard_test",
        defensive_cap=0.80,
        stress_cap=0.70,
        crash_cap=0.60,
        portfolio_drawdown_trigger=0.10,
        portfolio_drawdown_cap=0.02,
        portfolio_drawdown_release=0.04,
    )

    plain = backtest_prediction_strategy(
        close,
        predictions,
        min_position=0.20,
        max_position=0.95,
        cost=0.0,
        risk_frame=risk_frame,
        risk_config=no_guard_config,
    )
    guarded = backtest_prediction_strategy(
        close,
        predictions,
        min_position=0.20,
        max_position=0.95,
        cost=0.0,
        risk_frame=risk_frame,
        risk_config=risk_config,
    )

    assert guarded["max_dd"] < plain["max_dd"]
    assert guarded["avg_pos"] < plain["avg_pos"]


def test_backtest_prediction_strategy_reentry_guard_captures_confirmed_rebound():
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    close = pd.Series([100.0, 112.0, 92.0, 70.0, 50.0, 52.0, 58.0, 70.0, 88.0, 110.0, 132.0, 150.0], index=index)
    predictions = pd.Series(np.linspace(0.8, 1.0, len(index)), index=index)
    risk_frame = pd.DataFrame(
        {
            "regime_risk": [0.10, 0.15, 0.72, 0.85, 0.80, 0.55, 0.35, 0.28, 0.22, 0.20, 0.18, 0.16],
            "cum_drop_30": [0.00, 0.00, -0.18, -0.32, -0.28, -0.20, -0.10, -0.05, -0.02, 0.00, 0.00, 0.00],
            "drawdown_63": [0.00, 0.00, -0.18, -0.35, -0.30, -0.22, -0.12, -0.07, -0.03, 0.00, 0.00, 0.00],
            "mom_21": [0.02, 0.04, -0.08, -0.20, -0.10, -0.04, 0.02, 0.06, 0.09, 0.12, 0.14, 0.16],
            "trend_gap_63": [0.01, 0.02, -0.05, -0.12, -0.10, -0.05, 0.00, 0.03, 0.05, 0.07, 0.08, 0.09],
        },
        index=index,
    )
    capped_config = RiskAwarePositionConfig(
        name="capped_test",
        defensive_cap=0.80,
        stress_cap=0.70,
        crash_cap=0.60,
        portfolio_drawdown_trigger=0.10,
        portfolio_drawdown_cap=0.02,
        portfolio_drawdown_release=0.04,
    )
    reentry_config = RiskAwarePositionConfig(
        name="reentry_test",
        defensive_cap=0.80,
        stress_cap=0.70,
        crash_cap=0.60,
        portfolio_drawdown_trigger=0.10,
        portfolio_drawdown_cap=0.02,
        portfolio_drawdown_release=0.04,
        reentry_min_target=0.45,
        reentry_max_target=0.65,
        reentry_risk_threshold=0.35,
        reentry_momentum_threshold=0.05,
        reentry_trend_threshold=0.02,
        reentry_drawdown_floor=-0.10,
    )

    capped = backtest_prediction_strategy(
        close,
        predictions,
        min_position=0.20,
        max_position=0.95,
        cost=0.0,
        risk_frame=risk_frame,
        risk_config=capped_config,
    )
    reentry = backtest_prediction_strategy(
        close,
        predictions,
        min_position=0.20,
        max_position=0.95,
        cost=0.0,
        risk_frame=risk_frame,
        risk_config=reentry_config,
    )

    assert reentry["return"] > capped["return"]
    assert reentry["avg_pos"] > capped["avg_pos"]
    assert reentry["max_dd"] <= capped["max_dd"] + 1.0


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


def test_fit_quantbench_supervised_models_trains_sequence_temporal_spec():
    index = pd.date_range("2024-01-01", periods=90, freq="D")
    signal = np.sin(np.arange(len(index)) / 4.0)
    returns = 0.003 * np.sign(pd.Series(signal).shift(2).fillna(0.0).to_numpy())
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index)
    features = pd.DataFrame({"signal": signal, "anti_signal": -signal}, index=index)
    targets = build_soxl_supervised_targets(close, horizons=(1,))

    predictions, metrics = fit_quantbench_supervised_models(
        features,
        targets,
        train_start=index[0],
        train_end=index[59],
        test_start=index[60],
        test_end=index[80],
        horizon=1,
        model_names=("sequence_lag_ridge_mse",),
    )

    pred = predictions["sequence_lag_ridge_mse"]
    assert pred.index.min() >= index[60]
    assert pred.index.max() <= index[80]
    assert pred.nunique() > 1
    assert metrics["sequence_lag_ridge_mse"]["rows"] > 0
    assert "mse" in metrics["sequence_lag_ridge_mse"]


def test_fit_quantbench_supervised_models_trains_recurrent_sequence_spec():
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    signal = np.sin(np.arange(len(index)) / 5.0)
    lagged_signal = pd.Series(signal, index=index).shift(3).fillna(0.0).to_numpy()
    close = pd.Series(100.0 * np.cumprod(1.0 + 0.004 * np.sign(lagged_signal)), index=index)
    features = pd.DataFrame({"signal": signal, "anti_signal": -signal}, index=index)
    targets = build_soxl_supervised_targets(close, horizons=(1,))

    predictions, metrics = fit_quantbench_supervised_models(
        features,
        targets,
        train_start=index[0],
        train_end=index[79],
        test_start=index[80],
        test_end=index[110],
        horizon=1,
        model_names=("lstm_mse",),
    )

    pred = predictions["lstm_mse"]
    assert pred.index.min() >= index[80]
    assert pred.index.max() <= index[110]
    assert pred.nunique() > 1
    assert metrics["lstm_mse"]["rows"] > 0
    assert "mse" in metrics["lstm_mse"]


def test_fit_quantbench_supervised_models_trains_temporal_convolution_spec():
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    signal = np.sin(np.arange(len(index)) / 4.0) + 0.5 * np.sin(np.arange(len(index)) / 9.0)
    lagged_signal = pd.Series(signal, index=index).shift(4).fillna(0.0).to_numpy()
    close = pd.Series(100.0 * np.cumprod(1.0 + 0.004 * np.sign(lagged_signal)), index=index)
    features = pd.DataFrame({"signal": signal, "anti_signal": -signal}, index=index)
    targets = build_soxl_supervised_targets(close, horizons=(1,))

    predictions, metrics = fit_quantbench_supervised_models(
        features,
        targets,
        train_start=index[0],
        train_end=index[79],
        test_start=index[80],
        test_end=index[110],
        horizon=1,
        model_names=("tcn_mse",),
    )

    pred = predictions["tcn_mse"]
    assert pred.index.min() >= index[80]
    assert pred.index.max() <= index[110]
    assert pred.nunique() > 1
    assert metrics["tcn_mse"]["rows"] > 0
    assert "mse" in metrics["tcn_mse"]


def test_fit_quantbench_supervised_models_trains_relational_market_graph_spec():
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    base_signal = np.sin(np.arange(len(index)) / 5.0)
    sector_signal = np.cos(np.arange(len(index)) / 7.0)
    vix_signal = np.where(np.arange(len(index)) % 17 < 4, 1.0, -0.3)
    spread_signal = base_signal - 0.6 * sector_signal
    relation_signal = spread_signal - 0.4 * vix_signal
    lagged_relation = pd.Series(relation_signal, index=index).shift(1).fillna(0.0).to_numpy()
    close = pd.Series(100.0 * np.cumprod(1.0 + 0.004 * np.sign(lagged_relation)), index=index)
    features = pd.DataFrame(
        {
            "SOXL_momentum_21": base_signal,
            "SOXX_momentum_21": sector_signal,
            "SOXL_SOXX_RATIO_Z_252": spread_signal,
            "SOXL_SOXX_SPREAD_VOL_Z_21": spread_signal * vix_signal,
            "vix_zscore": vix_signal,
            "regime_risk": np.clip((vix_signal + 1.0) / 2.0, 0.0, 1.0),
        },
        index=index,
    )
    targets = build_soxl_supervised_targets(close, horizons=(1,))

    predictions, metrics = fit_quantbench_supervised_models(
        features,
        targets,
        train_start=index[0],
        train_end=index[79],
        test_start=index[80],
        test_end=index[110],
        horizon=1,
        model_names=("market_graph_mse",),
    )

    pred = predictions["market_graph_mse"]
    assert pred.index.min() >= index[80]
    assert pred.index.max() <= index[110]
    assert pred.nunique() > 1
    assert metrics["market_graph_mse"]["rows"] > 0
    assert "mse" in metrics["market_graph_mse"]


@pytest.mark.parametrize("model_name", FULL_PAPER_ARCHITECTURE_MODEL_NAMES)
def test_fit_quantbench_supervised_models_trains_full_paper_architecture_specs(model_name):
    index = pd.date_range("2024-01-01", periods=150, freq="D")
    step = np.arange(len(index), dtype=float)
    base_signal = np.sin(step / 5.0) + 0.30 * np.cos(step / 13.0)
    sector_signal = np.cos(step / 7.0) - 0.20 * np.sin(step / 17.0)
    risk_signal = np.where((step.astype(int) % 23) < 6, 0.90, -0.20)
    spread_signal = base_signal - 0.55 * sector_signal
    relation_signal = spread_signal - 0.35 * risk_signal
    lagged_relation = pd.Series(relation_signal, index=index).shift(2).fillna(0.0).to_numpy()
    close = pd.Series(100.0 * np.cumprod(1.0 + 0.004 * np.sign(lagged_relation)), index=index)
    features = pd.DataFrame(
        {
            "signal": base_signal,
            "anti_signal": -base_signal,
            "SOXL_momentum_21": base_signal,
            "SOXX_momentum_21": sector_signal,
            "SOXL_SOXX_RATIO_Z_252": spread_signal,
            "SOXL_SOXX_SPREAD_VOL_Z_21": spread_signal * risk_signal,
            "vix_zscore": risk_signal,
            "regime_risk": np.clip((risk_signal + 1.0) / 2.0, 0.0, 1.0),
        },
        index=index,
    )
    targets = build_soxl_supervised_targets(close, horizons=(1,))

    predictions, metrics = fit_quantbench_supervised_models(
        features,
        targets,
        train_start=index[0],
        train_end=index[94],
        test_start=index[95],
        test_end=index[135],
        horizon=1,
        model_names=(model_name,),
    )

    pred = predictions[model_name]
    assert pred.index.min() >= index[95]
    assert pred.index.max() <= index[135]
    assert np.isfinite(pred.to_numpy()).all()
    assert pred.nunique() > 1
    assert metrics[model_name]["rows"] > 0
    assert {"mse", "ic", "ranking_spread"} <= set(metrics[model_name])


def test_fit_quantbench_rolling_supervised_models_refits_without_future_label_leakage():
    index = pd.date_range("2024-01-01", periods=180, freq="D")
    signal = np.sin(np.arange(len(index)) / 5.0)
    returns = 0.004 * np.sign(pd.Series(signal).shift(1).fillna(0.0).to_numpy())
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index)
    features = pd.DataFrame({"signal": signal, "anti_signal": -signal}, index=index)
    targets = build_soxl_supervised_targets(close, horizons=(3,))

    predictions, metrics = fit_quantbench_rolling_supervised_models(
        features,
        targets,
        test_start=index[90],
        test_end=index[160],
        horizon=3,
        model_names=("ridge_mse",),
        min_train_rows=40,
        refit_step_rows=15,
        train_lookback_rows=80,
    )

    pred = predictions["rolling_ridge_mse"]
    assert pred.index.min() >= index[90]
    assert pred.index.max() <= index[160]
    assert pred.nunique() > 1
    assert metrics["rolling_ridge_mse"]["fold_count"] >= 3
    assert metrics["rolling_ridge_mse"]["train_lookback_rows"] == 80
    assert all(
        pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["predict_start"])
        for fold in metrics["rolling_ridge_mse"]["folds"]
    )
    assert all(
        pd.Timestamp(fold["label_cutoff"]) <= pd.Timestamp(fold["train_end"])
        for fold in metrics["rolling_ridge_mse"]["folds"]
    )


def test_quantbench_supervised_model_specs_cover_paper_objectives():
    specs = quantbench_supervised_model_specs()

    assert {
        "linear_classifier",
        "linear_regression",
        "tree",
        "boosted_tree",
        "tree_ensemble",
        "boosted_tree_xgboost",
        "boosted_tree_lightgbm",
        "boosted_tree_catboost",
        "neural_tabular",
        "sequence_temporal",
        "sequence_recurrent",
        "sequence_convolutional",
        "sequence_attention_recurrent",
        "sequence_frequency",
        "sequence_dual_attention",
        "sequence_multi_scale_recurrent",
        "sequence_dilated_recurrent",
        "sequence_hawkes_recurrent",
        "sequence_multi_scale_temporal_routing",
        "sequence_mlp_mixer",
        "sequence_linear_decomposition",
        "sequence_sparse_attention_transformer",
        "sequence_patch_transformer",
        "sequence_auto_correlation",
        "sequence_frequency_transformer",
        "sequence_pyramid_transformer",
        "sequence_temporal_fusion_transformer",
        "relational_graph",
        "relational_gcn",
        "relational_gat",
        "relational_rgcn",
        "relational_hats",
        "relational_rsr",
        "relational_hatr",
        "relational_hypergraph",
        "relational_estimate_hypergraph",
        "relational_temporal_hypergraph",
        "relational_temporal_hypergraph_attention",
        "relational_temporal_hypergraph_convolution_sthcn",
        "relational_temporal_hypergraph_convolution",
        "relational_dynamic_temporal",
        "sequence_cross_dimension_transformer",
    } <= {spec.paper_family for spec in specs}
    assert set(FULL_PAPER_ARCHITECTURE_MODEL_NAMES) <= {spec.name for spec in specs}
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


def test_build_supervised_summary_payloads_can_emit_risk_overlay_profiles():
    index = pd.date_range("2020-01-01", periods=900, freq="D")
    close = pd.Series(100.0 * np.cumprod(1.0 + np.sin(np.arange(len(index)) / 30.0) * 0.002), index=index)
    prediction = pd.Series(np.cos(np.arange(len(index)) / 25.0), index=index)
    risk_frame = pd.DataFrame(
        {
            "regime_risk": np.where(index.year == 2021, 0.85, 0.25),
            "cum_drop_30": np.where(index.year == 2021, -0.22, -0.02),
            "drawdown_63": np.where(index.year == 2021, -0.25, -0.03),
            "mom_21": np.where(index.year == 2022, 0.06, 0.01),
            "trend_gap_63": np.where(index.year == 2022, 0.04, 0.0),
        },
        index=index,
    )

    payloads = build_supervised_summary_payloads(
        close,
        {"ridge_mse": prediction},
        objective_metrics={"ridge_mse": {"ic": 0.1, "rows": 100.0}},
        profile_prefix="quantbench_supervised",
        risk_frame=risk_frame,
        risk_configs=(
            RiskAwarePositionConfig(name="macro_budget_test", stress_cap=0.15, defensive_cap=0.35),
        ),
    )

    profiles = {payload["profile"] for payload in payloads}
    assert "quantbench_supervised_ridge_mse" in profiles
    assert "quantbench_supervised_ridge_mse_macro_budget_test" in profiles
    overlay_payload = [payload for payload in payloads if payload["profile"].endswith("macro_budget_test")][0]
    assert overlay_payload["risk_overlay"]["name"] == "macro_budget_test"


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

    assert report["screen"]["records"] == 10
    assert report["data"]["feature_count"] == 3
    assert report["data"]["risk_overlays"] == [
        "macro_budget_defensive_v1",
        "macro_budget_lowdd_v1",
        "macro_budget_bull_guard_v1",
        "macro_budget_reentry_v1",
    ]
    assert (tmp_path / "supervised_summary.json").exists()
    assert (tmp_path / "supervised_candidate_screen.csv").read_text(encoding="utf-8").startswith("profile,")
    assert "supervised_payloads_json" in report["artifacts"]


def test_build_quantbench_supervised_report_accepts_custom_latest_windows(tmp_path):
    index = pd.date_range("2024-01-01", periods=140, freq="D")
    signal = np.sin(np.arange(len(index)) / 8.0)
    close = pd.Series(100.0 * np.cumprod(1.0 + 0.003 * np.sign(signal + 0.2)), index=index)
    features = pd.DataFrame({"signal": signal, "anti_signal": -signal}, index=index)

    build_quantbench_supervised_report(
        features,
        close,
        card_dir=tmp_path,
        root=tmp_path,
        horizon=1,
        model_names=("ridge_mse",),
        rolling_enabled=False,
        train_start=index[0],
        train_end=index[79],
        test_start=index[80],
        test_end=index[120],
        windows={"latest_test": (str(index[90].date()), str(index[120].date()))},
    )

    import json

    payloads = json.loads((tmp_path / "supervised_payloads.json").read_text(encoding="utf-8"))
    assert set(payloads[0]["slices"]) == {"latest_test"}

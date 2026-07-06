from __future__ import annotations

import pandas as pd

from src.workflow.trainable_regime_policy_nested_wf import (
    CRASH_TARGET_COL,
    MAX_ADVERSE_TARGET_COL,
    POSITION_TARGET_COL,
    REGIME_TARGET_COL,
    RISK_TARGET_COL,
    VOL_ADJUSTED_TARGET_COL,
    _append_soxx_constituent_normalized_features,
    _attach_soxx_constituent_model_features,
    _attach_trainable_policy_targets,
    _intraday_execution_rows_from_wide,
    _policy_feature_set_configs,
    _policy_feature_columns,
    _percentile_against_train,
    _select_policy_from_inner_metrics,
    _simulate_alpha_risk_policy,
)


def test_attach_trainable_policy_targets_replaces_manual_templates_with_supervised_labels():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
            ),
            "instrument": ["SOXL"] * 6,
            "open_to_open_return": [0.10, -0.04, 0.03, -0.12, 0.08, 0.02],
            "label_open_to_open_5d": [0.18, -0.16, 0.07, -0.09, 0.12, 0.01],
            "ret_20": [-0.30, -0.22, 0.02, 0.05, 0.18, 0.25],
            "ret_60": [-0.55, -0.48, -0.12, 0.04, 0.30, 0.42],
            "drawdown_60": [-0.70, -0.68, -0.40, -0.22, -0.12, -0.08],
            "realized_vol_20": [0.12, 0.16, 0.08, 0.20, 0.10, 0.09],
        }
    )

    out = _attach_trainable_policy_targets(frame)

    assert REGIME_TARGET_COL in out.columns
    assert CRASH_TARGET_COL in out.columns
    assert MAX_ADVERSE_TARGET_COL in out.columns
    assert RISK_TARGET_COL in out.columns
    assert VOL_ADJUSTED_TARGET_COL in out.columns
    assert POSITION_TARGET_COL in out.columns
    assert "template_id" not in out.columns
    assert out[REGIME_TARGET_COL].notna().all()
    assert set(out[REGIME_TARGET_COL].unique()).issubset({"defensive", "selective", "rebound", "trend"})
    assert out[CRASH_TARGET_COL].isin([0, 1]).all()
    assert out[RISK_TARGET_COL].isin([0, 1]).all()
    assert out[POSITION_TARGET_COL].between(0.0, 1.0).all()


def test_position_target_is_risk_budget_exposure_not_low_return_score():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "instrument": ["SOXL", "SOXL"],
            "open_to_open_return": [0.01, -0.01],
            "label_open_to_open_5d": [0.18, -0.14],
            "ret_20": [0.10, -0.20],
            "ret_60": [0.20, -0.30],
            "drawdown_60": [-0.05, -0.35],
            "realized_vol_20": [0.12, 0.18],
        }
    )

    out = _attach_trainable_policy_targets(frame)

    assert out[POSITION_TARGET_COL].iloc[0] >= 0.80
    assert out[POSITION_TARGET_COL].iloc[1] <= 0.05


def test_trainable_policy_targets_include_20d_adverse_risk_when_5d_return_is_positive():
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    open_prices = [100.0] * len(dates)
    open_prices[1] = 100.0
    open_prices[2:7] = [104.0, 105.0, 106.0, 107.0, 108.0]
    open_prices[10] = 70.0
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "instrument": ["SOXL"] * len(dates),
            "open": open_prices,
            "open_to_open_return": [0.0] * len(dates),
            "label_open_to_open_5d": [0.08] + [0.0] * (len(dates) - 1),
            "ret_20": [0.10] * len(dates),
            "ret_60": [0.20] * len(dates),
            "drawdown_60": [-0.05] * len(dates),
            "realized_vol_20": [0.12] * len(dates),
        }
    )

    out = _attach_trainable_policy_targets(frame)

    assert round(out["max_adverse_return_20d_target"].iloc[0], 6) == -0.30
    assert out["crash_probability_20d_target"].iloc[0] == 1


def test_learned_position_policy_uses_long_horizon_risk_head_as_independent_cap():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02"]),
            "instrument": ["SOXL"],
            "open_to_open_return": [0.01],
        }
    )
    predictions = pd.DataFrame(
        {
            "alpha_probability": [0.90],
            "risk_probability": [0.0],
            "risk_percentile": [0.0],
            "crash_probability": [0.0],
            "crash_risk_percentile": [0.0],
            "max_adverse_score": [-0.02],
            "max_adverse_risk_percentile": [0.0],
            "crash_20d_probability": [0.90],
            "max_adverse_20d_score": [-0.25],
            "max_adverse_20d_risk_percentile": [0.95],
            "regime": ["trend"],
            "vol_adjusted_score": [0.0],
            "learned_position_score": [0.80],
            "intraday_risk_state": [0.0],
        }
    )

    _, positions = _simulate_alpha_risk_policy(
        frame,
        predictions,
        policy={
            "policy_id": "unit_learned_position_long_horizon_risk",
            "position_mode": "learned_position",
            "alpha_floor": 0.0,
            "risk_cut": 2.0,
            "risk_cap": 1.0,
            "trend_scale": 1.0,
            "rebound_scale": 1.0,
            "selective_scale": 1.0,
            "defensive_scale": 1.0,
            "intraday_risk_cap": 1.0,
        },
        cost_per_turnover=0.0,
    )

    assert positions["target_position"].iloc[0] <= 0.10


def test_learned_position_policy_does_not_blanket_discount_budgeted_20d_risk():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02"]),
            "instrument": ["SOXL"],
            "open_to_open_return": [0.01],
        }
    )
    predictions = pd.DataFrame(
        {
            "alpha_probability": [0.90],
            "risk_probability": [0.0],
            "risk_percentile": [0.0],
            "crash_probability": [0.0],
            "crash_risk_percentile": [0.0],
            "max_adverse_score": [-0.02],
            "max_adverse_risk_percentile": [0.0],
            "crash_20d_probability": [0.15],
            "crash_20d_risk_percentile": [0.40],
            "max_adverse_20d_score": [-0.08],
            "max_adverse_20d_risk_percentile": [0.45],
            "regime": ["trend"],
            "vol_adjusted_score": [0.0],
            "learned_position_score": [0.80],
            "intraday_risk_state": [0.0],
        }
    )

    _, positions = _simulate_alpha_risk_policy(
        frame,
        predictions,
        policy={
            "policy_id": "unit_learned_position_budgeted_long_horizon_risk",
            "position_mode": "learned_position",
            "alpha_floor": 0.0,
            "risk_cut": 2.0,
            "risk_cap": 1.0,
            "trend_scale": 1.0,
            "rebound_scale": 1.0,
            "selective_scale": 1.0,
            "defensive_scale": 1.0,
            "intraday_risk_cap": 1.0,
        },
        cost_per_turnover=0.0,
    )

    assert positions["target_position"].iloc[0] >= 0.60


def test_learned_position_policy_uses_model_position_score_without_manual_thresholds():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "instrument": ["SOXL", "SOXL"],
            "open_to_open_return": [0.01, 0.01],
        }
    )
    predictions = pd.DataFrame(
        {
            "alpha_probability": [0.20, 0.90],
            "risk_probability": [0.0, 0.80],
            "risk_percentile": [0.0, 1.0],
            "crash_probability": [0.0, 0.50],
            "max_adverse_score": [0.0, -0.10],
            "regime": ["defensive", "trend"],
            "vol_adjusted_score": [0.0, 0.0],
            "learned_position_score": [0.80, 0.80],
            "intraday_risk_state": [0.0, 0.0],
        }
    )

    _, positions = _simulate_alpha_risk_policy(
        frame,
        predictions,
        policy={
            "policy_id": "unit_learned_position",
            "position_mode": "learned_position",
            "alpha_floor": 0.99,
            "risk_cut": 2.0,
            "risk_cap": 1.0,
            "trend_scale": 1.0,
            "rebound_scale": 1.0,
            "selective_scale": 1.0,
            "defensive_scale": 1.0,
            "intraday_risk_cap": 1.0,
        },
        cost_per_turnover=0.0,
    )

    assert round(positions["target_position"].iloc[0], 6) == 0.80
    assert 0.0 < positions["target_position"].iloc[1] < 0.80
    assert "learned_position_score" in positions.columns


def test_simulate_alpha_risk_policy_caps_position_independently_from_alpha():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "instrument": ["SOXL", "SOXL", "SOXL"],
            "open_to_open_return": [0.05, 0.05, 0.05],
        }
    )
    predictions = pd.DataFrame(
        {
            "alpha_probability": [0.80, 0.80, 0.80],
            "risk_probability": [0.10, 0.90, 0.20],
            "regime": ["trend", "trend", "rebound"],
            "vol_adjusted_score": [0.50, 0.50, 0.20],
            "intraday_risk_state": [0.0, 1.0, 0.0],
        }
    )

    metrics, positions = _simulate_alpha_risk_policy(
        frame,
        predictions,
        policy={
            "policy_id": "unit_trainable_policy",
            "alpha_floor": 0.55,
            "risk_cut": 0.75,
            "risk_cap": 0.25,
            "trend_scale": 1.0,
            "rebound_scale": 0.75,
            "selective_scale": 0.5,
            "defensive_scale": 0.0,
            "intraday_risk_cap": 0.10,
        },
        cost_per_turnover=0.0,
    )

    assert positions["target_position"].iloc[0] > positions["target_position"].iloc[2]
    assert positions["target_position"].iloc[1] == 0.10
    assert metrics["selection_avg_abs_position"] > 0.0


def test_intraday_execution_rows_capture_first_trigger_path_returns():
    wide = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02 09:35", "2026-01-02 09:40", "2026-01-02 16:00"]),
            "date": pd.to_datetime(["2026-01-02"] * 3),
            "open_soxl": [100.0, 100.0, 100.0],
            "close_soxl": [99.0, 90.0, 85.0],
            "open_soxx": [50.0, 50.0, 50.0],
            "close_soxx": [49.9, 49.0, 48.8],
        }
    )

    rows = _intraday_execution_rows_from_wide(wide, freq="5min")

    assert len(rows) == 1
    row = rows[0]
    assert row["intraday_5min_execution_triggered"] == 1.0
    assert row["intraday_5min_trigger_bar_index"] == 1
    assert round(row["intraday_5min_open_to_trigger_return"], 6) == -0.10
    assert round(row["intraday_5min_trigger_to_close_return"], 6) == round(85.0 / 90.0 - 1.0, 6)


def test_intraday_execution_replay_caps_after_trigger_not_for_full_day():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02"]),
            "instrument": ["SOXL"],
            "open_to_open_return": [-0.10],
            "intraday_execution_triggered": [1.0],
            "intraday_open_to_trigger_return": [-0.02],
            "intraday_trigger_to_close_return": [-0.08],
        }
    )
    predictions = pd.DataFrame(
        {
            "alpha_probability": [0.90],
            "risk_probability": [0.0],
            "regime": ["trend"],
            "vol_adjusted_score": [0.0],
            "intraday_risk_state": [1.0],
        }
    )

    metrics, positions = _simulate_alpha_risk_policy(
        frame,
        predictions,
        policy={
            "policy_id": "unit_intraday_execution_replay",
            "position_mode": "binary_full",
            "alpha_floor": 0.50,
            "risk_cut": 0.99,
            "risk_cap": 1.0,
            "trend_scale": 1.0,
            "rebound_scale": 1.0,
            "selective_scale": 1.0,
            "defensive_scale": 0.0,
            "intraday_risk_cap": 0.25,
            "intraday_execution_replay": True,
        },
        cost_per_turnover=0.0,
    )

    assert positions["opening_position"].iloc[0] == 1.0
    assert positions["target_position"].iloc[0] == 0.25
    assert round(positions["strategy_return"].iloc[0], 6) == -0.04
    assert metrics["intraday_execution_replay_day_ratio"] == 1.0


def test_crash_and_max_adverse_predictions_cap_position_independently_from_alpha():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "instrument": ["SOXL", "SOXL", "SOXL"],
            "open_to_open_return": [0.01, 0.01, 0.01],
        }
    )
    predictions = pd.DataFrame(
        {
            "alpha_probability": [0.80, 0.80, 0.80],
            "risk_probability": [0.05, 0.05, 0.05],
            "crash_probability": [0.75, 0.10, 0.10],
            "max_adverse_score": [-0.02, -0.18, -0.02],
            "regime": ["trend", "trend", "trend"],
            "vol_adjusted_score": [0.50, 0.50, 0.50],
            "intraday_risk_state": [0.0, 0.0, 0.0],
        }
    )

    _, positions = _simulate_alpha_risk_policy(
        frame,
        predictions,
        policy={
            "policy_id": "unit_multi_target_risk_cap",
            "position_mode": "binary_full",
            "alpha_floor": 0.55,
            "risk_cut": 0.99,
            "risk_cap": 1.0,
            "crash_cut": 0.60,
            "max_adverse_cut": -0.12,
            "adverse_risk_cap": 0.15,
            "trend_scale": 1.0,
            "rebound_scale": 1.0,
            "selective_scale": 1.0,
            "defensive_scale": 0.0,
            "intraday_risk_cap": 1.0,
        },
        cost_per_turnover=0.0,
    )

    assert list(positions["target_position"]) == [0.15, 0.15, 1.0]


def test_percentile_risk_predictions_use_training_distribution_without_test_ranking():
    train_scores = pd.Series([0.10, 0.20, 0.30, 0.40])
    predict_scores = pd.Series([0.05, 0.25, 0.50])

    high_risk = _percentile_against_train(train_scores, predict_scores, higher_is_risk=True)
    low_value_risk = _percentile_against_train(train_scores, predict_scores, higher_is_risk=False)

    assert list(high_risk.round(2)) == [0.0, 0.5, 1.0]
    assert list(low_value_risk.round(2)) == [1.0, 0.5, 0.0]


def test_risk_percentile_predictions_cap_position_independently_from_raw_probability():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "instrument": ["SOXL", "SOXL", "SOXL"],
            "open_to_open_return": [0.01, 0.01, 0.01],
        }
    )
    predictions = pd.DataFrame(
        {
            "alpha_probability": [0.80, 0.80, 0.80],
            "risk_probability": [0.05, 0.05, 0.05],
            "risk_percentile": [0.20, 0.92, 0.20],
            "crash_probability": [0.05, 0.05, 0.05],
            "crash_risk_percentile": [0.91, 0.10, 0.10],
            "max_adverse_score": [-0.02, -0.02, -0.02],
            "max_adverse_risk_percentile": [0.10, 0.10, 0.93],
            "regime": ["trend", "trend", "trend"],
            "vol_adjusted_score": [0.50, 0.50, 0.50],
            "intraday_risk_state": [0.0, 0.0, 0.0],
        }
    )

    _, positions = _simulate_alpha_risk_policy(
        frame,
        predictions,
        policy={
            "policy_id": "unit_percentile_multi_target_risk_cap",
            "position_mode": "binary_full",
            "alpha_floor": 0.55,
            "risk_cut": 0.99,
            "risk_cap": 1.0,
            "risk_percentile_cut": 0.90,
            "crash_percentile_cut": 0.90,
            "adverse_percentile_cut": 0.90,
            "percentile_risk_cap": 0.20,
            "trend_scale": 1.0,
            "rebound_scale": 1.0,
            "selective_scale": 1.0,
            "defensive_scale": 0.0,
            "intraday_risk_cap": 1.0,
        },
        cost_per_turnover=0.0,
    )

    assert list(positions["target_position"]) == [0.20, 0.20, 0.20]


def test_binary_full_policy_uses_alpha_as_entry_permission_and_risk_as_cap():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "instrument": ["SOXL", "SOXL", "SOXL"],
            "open_to_open_return": [0.01, 0.01, 0.01],
        }
    )
    predictions = pd.DataFrame(
        {
            "alpha_probability": [0.59, 0.59, 0.54],
            "risk_probability": [0.08, 0.14, 0.08],
            "regime": ["trend", "trend", "trend"],
            "vol_adjusted_score": [0.0, 0.0, 0.0],
            "intraday_risk_state": [0.0, 0.0, 0.0],
        }
    )

    _, positions = _simulate_alpha_risk_policy(
        frame,
        predictions,
        policy={
            "policy_id": "unit_binary_full",
            "position_mode": "binary_full",
            "alpha_floor": 0.58,
            "risk_cut": 0.12,
            "risk_cap": 0.20,
            "trend_scale": 1.0,
            "rebound_scale": 1.0,
            "selective_scale": 1.0,
            "defensive_scale": 0.0,
            "intraday_risk_cap": 1.0,
        },
        cost_per_turnover=0.0,
    )

    assert list(positions["target_position"]) == [1.0, 0.20, 0.0]


def test_drawdown_cooldown_cap_is_path_dependent_without_future_returns():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
            "instrument": ["SOXL", "SOXL", "SOXL", "SOXL"],
            "open_to_open_return": [-0.20, 0.10, 0.10, 0.10],
        }
    )
    predictions = pd.DataFrame(
        {
            "alpha_probability": [0.90, 0.90, 0.90, 0.90],
            "risk_probability": [0.0, 0.0, 0.0, 0.0],
            "regime": ["trend", "trend", "trend", "trend"],
            "vol_adjusted_score": [0.0, 0.0, 0.0, 0.0],
            "intraday_risk_state": [0.0, 0.0, 0.0, 0.0],
        }
    )

    _, positions = _simulate_alpha_risk_policy(
        frame,
        predictions,
        policy={
            "policy_id": "unit_cooldown",
            "position_mode": "binary_full",
            "alpha_floor": 0.50,
            "risk_cut": 0.99,
            "risk_cap": 1.0,
            "trend_scale": 1.0,
            "rebound_scale": 1.0,
            "selective_scale": 1.0,
            "defensive_scale": 0.0,
            "intraday_risk_cap": 1.0,
            "drawdown_cooldown_trigger": -0.10,
            "drawdown_cooldown_days": 1,
            "drawdown_cooldown_cap": 0.25,
        },
        cost_per_turnover=0.0,
    )

    assert list(positions["target_position"]) == [1.0, 0.25, 0.25, 0.25]
    assert list(positions["drawdown_cooldown_active"]) == [0, 1, 1, 1]


def test_fixed_window_drawdown_cooldown_releases_without_retrigger_until_rearmed():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
            "instrument": ["SOXL", "SOXL", "SOXL", "SOXL"],
            "open_to_open_return": [-0.20, 0.01, 0.01, 0.01],
        }
    )
    predictions = pd.DataFrame(
        {
            "alpha_probability": [0.90, 0.90, 0.90, 0.90],
            "risk_probability": [0.0, 0.0, 0.0, 0.0],
            "regime": ["trend", "trend", "trend", "trend"],
            "vol_adjusted_score": [0.0, 0.0, 0.0, 0.0],
            "intraday_risk_state": [0.0, 0.0, 0.0, 0.0],
        }
    )

    _, positions = _simulate_alpha_risk_policy(
        frame,
        predictions,
        policy={
            "policy_id": "unit_fixed_window_cooldown",
            "position_mode": "binary_full",
            "alpha_floor": 0.50,
            "risk_cut": 0.99,
            "risk_cap": 1.0,
            "trend_scale": 1.0,
            "rebound_scale": 1.0,
            "selective_scale": 1.0,
            "defensive_scale": 0.0,
            "intraday_risk_cap": 1.0,
            "drawdown_cooldown_trigger": -0.10,
            "drawdown_cooldown_days": 1,
            "drawdown_cooldown_cap": 0.25,
            "drawdown_cooldown_retrigger": False,
            "drawdown_cooldown_rearm_level": -0.05,
        },
        cost_per_turnover=0.0,
    )

    assert list(positions["target_position"]) == [1.0, 0.25, 1.0, 1.0]
    assert list(positions["drawdown_cooldown_active"]) == [0, 1, 0, 0]


def test_attach_soxx_constituent_features_enters_model_feature_columns_without_template_gate():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "instrument": ["SOXL", "SOXL"],
            "ret_5_z": [0.1, -0.2],
            "intraday_risk_state": [0.0, 1.0],
            "intraday_risk_score": [0.0, 1.2],
        }
    )
    factors = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "soxx_const_breadth_ret5_pos_lag1": [0.7, 0.3],
            "soxx_const_breadth_ret5_pos_lag1_z60": [1.1, -0.8],
            "soxx_const_dispersion_ret20_lag1_pctile252": [0.6, 0.9],
        }
    )
    factors.attrs["feature_cols"] = [
        "soxx_const_breadth_ret5_pos_lag1_z60",
        "soxx_const_dispersion_ret20_lag1_pctile252",
    ]

    out, constituent_cols = _attach_soxx_constituent_model_features(frame, factors)
    feature_cols = _policy_feature_columns(out)

    assert constituent_cols == [
        "soxx_const_breadth_ret5_pos_lag1_z60",
        "soxx_const_dispersion_ret20_lag1_pctile252",
    ]
    assert "ret_5_z" in feature_cols
    assert "soxx_const_breadth_ret5_pos_lag1" not in feature_cols
    assert "soxx_const_breadth_ret5_pos_lag1_z60" in feature_cols
    assert "soxx_const_dispersion_ret20_lag1_pctile252" in feature_cols
    assert "intraday_risk_state" not in feature_cols
    assert "intraday_risk_score" not in feature_cols


def test_soxx_constituent_normalization_creates_zscore_and_percentile_features():
    values = list(range(1, 70))
    factors = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-02", periods=len(values), freq="B"),
            "soxx_const_weighted_ret5_lag1": values,
        }
    )

    out, feature_cols = _append_soxx_constituent_normalized_features(
        factors,
        raw_feature_cols=["soxx_const_weighted_ret5_lag1"],
    )

    assert feature_cols == [
        "soxx_const_weighted_ret5_lag1_z60",
        "soxx_const_weighted_ret5_lag1_pctile252",
    ]
    assert out["soxx_const_weighted_ret5_lag1_z60"].notna().sum() > 0
    assert out["soxx_const_weighted_ret5_lag1_pctile252"].notna().sum() > 0


def test_policy_feature_sets_make_soxx_diffusion_nested_selectable_not_forced():
    frame = pd.DataFrame(
        {
            "ret_5_z": [0.1, -0.2],
            "ret_20_pctile": [0.7, 0.4],
            "soxx_const_breadth_ret5_pos_lag1_z60": [1.1, -0.8],
            "soxx_const_dispersion_ret20_lag1_pctile252": [0.6, 0.9],
            "intraday_risk_state": [0.0, 1.0],
            "intraday_risk_score": [0.0, 1.2],
        }
    )

    configs = _policy_feature_set_configs(frame)
    by_name = {config["feature_set"]: config for config in configs}

    assert set(by_name) == {"base", "base_soxx_diffusion"}
    assert by_name["base"]["contains_soxx_constituents"] is False
    assert by_name["base_soxx_diffusion"]["contains_soxx_constituents"] is True
    assert "ret_5_z" in by_name["base"]["feature_cols"]
    assert "soxx_const_breadth_ret5_pos_lag1_z60" not in by_name["base"]["feature_cols"]
    assert "soxx_const_breadth_ret5_pos_lag1_z60" in by_name["base_soxx_diffusion"]["feature_cols"]
    assert "intraday_risk_state" not in by_name["base_soxx_diffusion"]["feature_cols"]


def test_selector_treats_feature_set_as_nested_choice_without_manual_template():
    rows = []
    for feature_set in ["base", "base_soxx_diffusion"]:
        for heldout_asset in ["SOXL", "TQQQ", "UPRO", "SPXL", "TECL"]:
            base_winner = feature_set == "base"
            rows.append(
                {
                    "candidate": "same_policy",
                    "feature_set": feature_set,
                    "contains_soxx_constituents": not base_winner,
                    "feature_count": 42 if base_winner else 54,
                    "soxx_constituent_feature_count": 0 if base_winner else 12,
                    "selection_return_pct": 25.0 if base_winner and heldout_asset == "SOXL" else 8.0,
                    "selection_max_drawdown_pct": -12.0,
                    "selection_avg_abs_position": 0.30,
                    "selection_total_turnover": 4.0,
                    "excess_return_pct": 18.0 if base_winner and heldout_asset == "SOXL" else -4.0,
                    "beats_buyhold": bool(base_winner and heldout_asset == "SOXL"),
                    "heldout_asset": heldout_asset,
                    "inner_year": 2024,
                }
            )

    selected = _select_policy_from_inner_metrics(
        pd.DataFrame(rows),
        outer_test_year=2025,
        required_heldout_assets=("SOXL", "TQQQ", "UPRO", "SPXL", "TECL"),
    )

    assert selected["candidate"] == "same_policy"
    assert selected["feature_set"] == "base"
    assert selected["contains_soxx_constituents"] is False
    assert selected["soxx_constituent_feature_count"] == 0
    assert selected["selection_policy"] == "nested_inner_soxl_calibrated_learned_position_no_template_grid"


def test_selector_uses_soxl_inner_calibration_before_cross_asset_stress_average():
    rows = []
    for candidate in ["cross_average_winner", "soxl_calibrated_winner"]:
        for heldout_asset in ["SOXL", "TQQQ", "UPRO", "SPXL", "TECL"]:
            if candidate == "cross_average_winner":
                selection_return = 18.0 if heldout_asset != "SOXL" else 3.0
                excess_return = 12.0 if heldout_asset != "SOXL" else -5.0
                max_drawdown = -10.0
            else:
                selection_return = 22.0 if heldout_asset == "SOXL" else 1.0
                excess_return = 18.0 if heldout_asset == "SOXL" else -2.0
                max_drawdown = -12.0
            rows.append(
                {
                    "candidate": candidate,
                    "selection_return_pct": selection_return,
                    "selection_max_drawdown_pct": max_drawdown,
                    "selection_avg_abs_position": 0.35,
                    "selection_total_turnover": 4.0,
                    "excess_return_pct": excess_return,
                    "beats_buyhold": excess_return > 0,
                    "heldout_asset": heldout_asset,
                    "inner_year": 2024,
                }
            )
    metrics = pd.DataFrame(rows)

    selected = _select_policy_from_inner_metrics(
        metrics,
        outer_test_year=2025,
        required_heldout_assets=("SOXL", "TQQQ", "UPRO", "SPXL", "TECL"),
    )

    assert selected["selection_policy"] == "nested_inner_soxl_calibrated_learned_position_no_template_grid"
    assert selected["calibration_asset"] == "SOXL"
    assert selected["candidate"] == "soxl_calibrated_winner"
    assert selected["soxl_inner_excess_return_pct"] == 18.0
    assert selected["cross_asset_stress_role"] == "overfit_audit_not_primary_selection_target"
    assert selected["selection_uses_outer_test"] is False


def test_selector_prefers_soxl_drawdown_qualified_candidate_over_risky_soxl_return():
    rows = []
    for candidate in ["risky_soxl_return", "drawdown_qualified_soxl"]:
        for heldout_asset in ["SOXL", "TQQQ", "UPRO", "SPXL", "TECL"]:
            if candidate == "risky_soxl_return" and heldout_asset == "SOXL":
                selection_return, excess_return, max_drawdown = 85.0, 60.0, -55.0
            elif candidate == "drawdown_qualified_soxl" and heldout_asset == "SOXL":
                selection_return, excess_return, max_drawdown = 24.0, 16.0, -12.0
            else:
                selection_return, excess_return, max_drawdown = 6.0, 1.0, -14.0
            rows.append(
                {
                    "candidate": candidate,
                    "selection_return_pct": selection_return,
                    "selection_max_drawdown_pct": max_drawdown,
                    "selection_avg_abs_position": 0.30,
                    "selection_total_turnover": 4.0,
                    "excess_return_pct": excess_return,
                    "beats_buyhold": excess_return > 0,
                    "heldout_asset": heldout_asset,
                    "inner_year": 2024,
                }
            )
    selected = _select_policy_from_inner_metrics(
        pd.DataFrame(rows),
        outer_test_year=2025,
        required_heldout_assets=("SOXL", "TQQQ", "UPRO", "SPXL", "TECL"),
    )

    assert selected["candidate"] == "drawdown_qualified_soxl"
    assert selected["soxl_drawdown_gate_applied"] is True
    assert selected["soxl_inner_pass_rate"] == 1.0


def test_selector_does_not_expose_manual_regime_template_filter_fields():
    rows = []
    for candidate in ["soxl_calibrated_winner", "manual_regime_shape_winner"]:
        for heldout_asset in ["SOXL", "TQQQ", "UPRO", "SPXL", "TECL"]:
            is_manual_shape = candidate == "manual_regime_shape_winner"
            rows.append(
                {
                    "candidate": candidate,
                    "selection_return_pct": 40.0 if is_manual_shape else 24.0,
                    "selection_max_drawdown_pct": -12.0,
                    "selection_avg_abs_position": 0.30,
                    "selection_total_turnover": 4.0,
                    "excess_return_pct": -20.0 if is_manual_shape and heldout_asset != "SOXL" else 12.0,
                    "beats_buyhold": not (is_manual_shape and heldout_asset != "SOXL"),
                    "heldout_asset": heldout_asset,
                    "inner_year": 2024,
                    "selector_regime": "post_bear_repair_setup" if is_manual_shape else "balanced_setup",
                }
            )

    selected = _select_policy_from_inner_metrics(
        pd.DataFrame(rows),
        outer_test_year=2025,
        required_heldout_assets=("SOXL", "TQQQ", "UPRO", "SPXL", "TECL"),
    )

    assert selected["selection_policy"] == "nested_inner_soxl_calibrated_learned_position_no_template_grid"
    assert "outer_selector_regime" not in selected
    assert "selector_regime_filter_applied" not in selected
    assert "regime_soxl_inner_excess_return_pct" not in selected

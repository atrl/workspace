from __future__ import annotations

import pandas as pd

from src.workflow.trainable_regime_policy_nested_wf import (
    CRASH_TARGET_COL,
    MAX_ADVERSE_TARGET_COL,
    REGIME_TARGET_COL,
    RISK_TARGET_COL,
    VOL_ADJUSTED_TARGET_COL,
    _append_soxx_constituent_normalized_features,
    _attach_soxx_constituent_model_features,
    _attach_trainable_policy_targets,
    _policy_feature_columns,
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
    assert "template_id" not in out.columns
    assert out[REGIME_TARGET_COL].notna().all()
    assert set(out[REGIME_TARGET_COL].unique()).issubset({"defensive", "selective", "rebound", "trend"})
    assert out[CRASH_TARGET_COL].isin([0, 1]).all()
    assert out[RISK_TARGET_COL].isin([0, 1]).all()


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

    assert selected["selection_policy"] == "nested_inner_soxl_calibrated_no_template_policy_grid"
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

    assert selected["selection_policy"] == "nested_inner_soxl_calibrated_no_template_policy_grid"
    assert "outer_selector_regime" not in selected
    assert "selector_regime_filter_applied" not in selected
    assert "regime_soxl_inner_excess_return_pct" not in selected

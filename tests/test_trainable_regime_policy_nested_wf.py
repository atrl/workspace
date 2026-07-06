from __future__ import annotations

import pandas as pd

from src.workflow.trainable_regime_policy_nested_wf import (
    CRASH_TARGET_COL,
    MAX_ADVERSE_TARGET_COL,
    REGIME_TARGET_COL,
    RISK_TARGET_COL,
    VOL_ADJUSTED_TARGET_COL,
    _attach_trainable_policy_targets,
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

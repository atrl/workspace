from __future__ import annotations

import numpy as np
import pandas as pd

from src.workflow.cognitive_alpha_goal_codex_exec import (
    BASE_CONSTRAINED_REF_COL,
    LABEL_OPEN_TO_OPEN_5D_COL,
    OPEN_TO_OPEN_RETURN_COL,
)
from src.workflow.meta_label_risk_budget_open2open import (
    _risk_budget_components,
    evaluate_meta_label_risk_budget_candidates,
)


def _candidate_spec() -> dict:
    return {
        "max_position": 0.65,
        "take_threshold": 0.55,
        "target_daily_vol": 0.04,
        "min_vol_scale": 0.0,
        "max_vol_scale": 1.0,
        "tail_soft_threshold": 0.25,
        "tail_hard_threshold": 0.50,
        "tail_soft_cap": 0.50,
        "tail_hard_cap": 0.10,
        "drawdown_soft_floor": -0.08,
        "drawdown_hard_floor": -0.12,
        "drawdown_soft_cap": 0.55,
        "drawdown_hard_cap": 0.25,
        "intraday_vol_ratio_threshold": 3.25,
        "intraday_range_ratio_threshold": 3.35,
        "intraday_late_ret_floor": -0.03,
        "intraday_dislocation_floor": -0.006,
        "intraday_soft_cap": 0.65,
        "intraday_hard_cap": 0.35,
        "rebalance_threshold": 0.05,
        "max_daily_change": 0.35,
    }


def test_meta_label_risk_budget_uses_lgb_xgb_consensus_without_probability_boost():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-02", periods=900)
    trend = np.sin(np.linspace(0, 24, len(dates)))
    tail_pressure = (np.cos(np.linspace(0, 33, len(dates))) < -0.88).astype(float)
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "f_trend": trend + rng.normal(0.0, 0.10, len(dates)),
            "f_tail": tail_pressure + rng.normal(0.0, 0.05, len(dates)),
            "realized_vol_20": 0.035 + 0.020 * tail_pressure,
            BASE_CONSTRAINED_REF_COL: (trend > -0.25).astype(float) * 0.7,
            "intraday15m_realized_vol_ratio_soxl_soxx_lag1": 3.0 + 0.55 * tail_pressure,
            "intraday15m_range_ratio_soxl_soxx_lag1": 3.05 + 0.45 * tail_pressure,
            "intraday15m_soxl_late_ret_lag1": -0.005 - 0.04 * tail_pressure,
            "intraday15m_soxl_minus_3x_soxx_ret_lag1": -0.001 - 0.008 * tail_pressure,
        }
    )
    noise = rng.normal(0.0, 0.01, len(dates))
    frame[OPEN_TO_OPEN_RETURN_COL] = 0.006 * (trend > 0).astype(float) - 0.070 * tail_pressure + noise
    frame[LABEL_OPEN_TO_OPEN_5D_COL] = 0.025 * (trend > 0).astype(float) - 0.060 * tail_pressure + noise

    metrics, positions, labels, importance = evaluate_meta_label_risk_budget_candidates(
        frame,
        feature_cols=[
            "f_trend",
            "f_tail",
            "realized_vol_20",
            "intraday15m_realized_vol_ratio_soxl_soxx_lag1",
            "intraday15m_soxl_late_ret_lag1",
        ],
        train_window=("2020-01-02", "2021-12-31"),
        validation_window=("2022-01-03", "2022-12-30"),
        test_window=("2023-01-03", "2023-06-30"),
        candidate_specs=[_candidate_spec()],
        validation_gates={
            "min_validation_return_pct": -100.0,
            "min_avg_abs_position": 0.0,
            "max_zero_position_day_ratio": 1.0,
            "validation_max_drawdown_floor": -100.0,
            "validation_max_daily_loss_floor": -100.0,
            "validation_cvar_5pct_floor": -100.0,
            "validation_worst_5_daily_loss_floor": -100.0,
            "max_validation_underwater_day_ratio": 1.0,
            "max_validation_total_turnover": 1000.0,
            "max_extreme_loss_day_count": 1000,
        },
    )

    assert {"lgbm_classifier", "xgb_classifier", "lgb_xgb_consensus"}.issubset(set(metrics["model"]))
    selected = metrics.loc[metrics["selected"].map(bool)]
    assert set(selected["model"]) == {"lgb_xgb_consensus"}
    assert set(selected["position_architecture"]) == {"opportunity_side_times_edge_times_risk_budgets"}
    assert set(selected["position_mode"]) == {"meta_label_risk_budget"}
    assert not selected["params"].str.contains("base_plus_probability").any()
    assert {"take_trade_positive_rate", "tail_loss_positive_rate"}.issubset(labels.columns)
    assert {"tail_cap", "intraday_execution_cap", "drawdown_budget"}.issubset(positions.columns)
    assert not importance.empty


def test_risk_budget_components_cap_tail_and_intraday_pressure():
    data = pd.DataFrame(
        {
            BASE_CONSTRAINED_REF_COL: [0.7, 0.7],
            "realized_vol_20": [0.04, 0.04],
            "intraday15m_realized_vol_ratio_soxl_soxx_lag1": [3.0, 3.6],
            "intraday15m_range_ratio_soxl_soxx_lag1": [3.0, 3.6],
            "intraday15m_soxl_late_ret_lag1": [0.0, -0.05],
            "intraday15m_soxl_minus_3x_soxx_ret_lag1": [0.0, -0.010],
        }
    )
    components = _risk_budget_components(
        data,
        take_probability=pd.Series([0.80, 0.80]),
        tail_probability=pd.Series([0.10, 0.70]),
        position_spec=_candidate_spec(),
        base_position_col=BASE_CONSTRAINED_REF_COL,
        realized_vol_col="realized_vol_20",
    )

    assert components["tail_cap"].iloc[1] < components["tail_cap"].iloc[0]
    assert components["intraday_execution_cap"].iloc[1] < components["intraday_execution_cap"].iloc[0]
    assert components["raw_target"].iloc[1] < components["raw_target"].iloc[0]

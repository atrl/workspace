from __future__ import annotations

import pandas as pd

from src.workflow.cognitive_alpha_goal_codex_exec import BASE_CONSTRAINED_REF_COL
from src.workflow.hybrid_intraday15m_open2open import (
    HYBRID_INTRADAY_REGIME_GATE_COL,
    HybridIntraday15mOpen2OpenSpec,
    _attach_hybrid_intraday_regime_gate,
    _hybrid_goal_summary,
    _position_spec_for_hybrid,
)


def test_hybrid_intraday_regime_gate_uses_lagged_15m_features():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "intraday15m_soxl_late_ret_lag1": [0.01, -0.045, -0.010],
            "intraday15m_soxl_minus_3x_soxx_ret_lag1": [0.0, -0.002, -0.009],
            "intraday15m_realized_vol_ratio_soxl_soxx_lag1": [1.8, 2.0, 4.2],
            "intraday15m_range_ratio_soxl_soxx_lag1": [1.9, 2.2, 4.4],
            "soxx_const_breadth_ret5_pos_lag1": [0.8, 0.6, 0.3],
            "soxx_const_downside_ret5_lt_m5_lag1": [0.0, 0.1, 0.4],
        }
    )

    out = _attach_hybrid_intraday_regime_gate(frame)

    assert HYBRID_INTRADAY_REGIME_GATE_COL in out.columns
    assert out[HYBRID_INTRADAY_REGIME_GATE_COL].tolist() == [0.0, 1.0, 1.0]


def test_hybrid_position_spec_keeps_intraday_as_features_and_validation_selected():
    feature_cols = [
        "vix",
        "intraday15m_soxl_late_ret_lag1",
        "intraday15m_soxl_minus_3x_soxx_ret_lag1",
        HYBRID_INTRADAY_REGIME_GATE_COL,
    ]
    spec = HybridIntraday15mOpen2OpenSpec(root=pd.Timestamp("2026-01-01"))  # type: ignore[arg-type]

    position_spec = _position_spec_for_hybrid(feature_cols, spec=spec, test=("2026-01-02", "2026-07-02"))

    assert {item["model"] for item in position_spec.model_specs} >= {
        "lgbm_classifier",
        "lgbm_regressor",
        "xgb_classifier",
        "xgb_regressor",
        "lgb_xgb_consensus",
    }
    assert any(item["model"] == "lgb_xgb_consensus" and not item.get("selection_disabled", False) for item in position_spec.model_specs)
    assert {fold["fold"] for fold in position_spec.walk_forward_folds or []} == {
        "wf_2023",
        "wf_2024",
        "wf_2025",
        "wf_2026",
    }
    assert all(item.get("base_position_col") == BASE_CONSTRAINED_REF_COL for item in position_spec.position_specs or [])
    assert all(item.get("risk_gate_col") == HYBRID_INTRADAY_REGIME_GATE_COL for item in position_spec.position_specs or [])
    assert all(
        not str(item.get("base_position_col", "")).startswith("intraday15m_")
        for item in position_spec.position_specs or []
    )


def test_hybrid_goal_summary_requires_buyhold_outperformance_and_drawdown_floor():
    rows = pd.DataFrame(
        [
            {"fold": "wf_2023", "test_year": 2023, "test_return_pct": 175.0, "test_max_drawdown_pct": -19.0, "buyhold_return_pct": 169.0},
            {"fold": "wf_2024", "test_year": 2024, "test_return_pct": 30.0, "test_max_drawdown_pct": -18.0, "buyhold_return_pct": 6.5},
            {"fold": "wf_2025", "test_year": 2025, "test_return_pct": 70.0, "test_max_drawdown_pct": -21.0, "buyhold_return_pct": 77.8},
            {"fold": "wf_2026", "test_year": 2026, "test_return_pct": 360.0, "test_max_drawdown_pct": -18.0, "buyhold_return_pct": 335.4},
        ]
    )

    summary = _hybrid_goal_summary(rows, required_years=(2023, 2024, 2025, 2026))

    assert summary["selection_uses_test_segment"] is False
    assert summary["required_test_years"] == [2023, 2024, 2025, 2026]
    assert summary["target_pass_test_years"] == [2023, 2024, 2026]
    assert summary["required_policy_satisfied"] is False
    assert "annual_target_gate_failed" in summary["blocking_reasons"]

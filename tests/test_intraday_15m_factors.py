from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_intraday_15m_factor_frame_uses_lagged_qlib_daily_aggregates(tmp_path):
    from src.data import QlibProviderConfig, build_qlib_store_from_frame
    from src.factor.intraday_15m_factors import build_intraday_15m_factor_frame

    raw = _intraday_raw_frame()
    provider_uri = tmp_path / "qlib" / "soxl_intraday_15min"
    build_qlib_store_from_frame(
        raw,
        provider_uri=provider_uri,
        freq="15min",
        as_of="2026-01-06",
        source="unit_test_intraday_15m",
    )

    factors = build_intraday_15m_factor_frame(
        QlibProviderConfig(provider_uri=provider_uri, freq="15min"),
        instruments=["SOXL", "SOXX"],
        start="2026-01-02 09:45:00",
        end="2026-01-06 16:00:00",
        lag_days=1,
    )

    rows = factors.set_index("datetime")
    ret_col = "intraday15m_soxl_ret_lag1"
    dislocation_col = "intraday15m_soxl_minus_3x_soxx_ret_lag1"
    assert ret_col in factors.attrs["feature_cols"]
    assert dislocation_col in factors.attrs["feature_cols"]
    assert pd.isna(rows.loc[pd.Timestamp("2026-01-02"), ret_col])
    assert rows.loc[pd.Timestamp("2026-01-05"), ret_col] == pytest.approx(108.0 / 100.0 - 1.0)
    assert rows.loc[pd.Timestamp("2026-01-05"), "intraday15m_soxl_range_lag1"] == pytest.approx(
        110.0 / 98.0 - 1.0
    )
    assert rows.loc[pd.Timestamp("2026-01-05"), "intraday15m_soxl_close_location_lag1"] == pytest.approx(
        (108.0 - 98.0) / (110.0 - 98.0)
    )
    assert rows.loc[pd.Timestamp("2026-01-05"), dislocation_col] == pytest.approx(
        (108.0 / 100.0 - 1.0) - 3.0 * (52.0 / 50.0 - 1.0)
    )
    assert factors.attrs["source"] == "qlib.data.D.features"
    assert factors.attrs["freq"] == "15min"
    assert factors.attrs["lag_days"] == 1


def test_cognitive_alpha_goal_codex_exec_merges_intraday_15m_features_as_model_inputs_not_gates():
    from src.workflow.cognitive_alpha_goal_codex_exec import (
        BASE_CONSTRAINED_REF_COL,
        COGALPHA_OPEN2OPEN_RISK_GATE_COL,
        CognitiveAlphaGoalCodexExecCardSpec,
        _attach_intraday_15m_model_features,
        _model_feature_columns,
        _position_spec,
    )

    factor_col = "intraday15m_soxl_minus_3x_soxx_ret_lag1"
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            BASE_CONSTRAINED_REF_COL: [0.5, 0.4],
            COGALPHA_OPEN2OPEN_RISK_GATE_COL: [0.0, 1.0],
            "vix": [20.0, 25.0],
        }
    )
    intraday = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            factor_col: [0.01, -0.02],
        }
    )
    intraday.attrs["feature_cols"] = [factor_col]

    out, intraday_cols = _attach_intraday_15m_model_features(frame, intraday)
    model_features = _model_feature_columns([], [], out, extra_columns=intraday_cols)
    position_spec = _position_spec(
        model_features,
        spec=CognitiveAlphaGoalCodexExecCardSpec(root=Path("/tmp")),
        test=("2026-01-02", "2026-01-05"),
    )

    assert intraday_cols == [factor_col]
    assert out[factor_col].tolist() == [0.01, -0.02]
    assert factor_col in model_features
    assert all(item.get("base_position_col") != factor_col for item in position_spec.position_specs or [])
    assert all(item.get("risk_gate_col") != factor_col for item in position_spec.position_specs or [])


def test_cognitive_alpha_goal_codex_exec_builds_constituent_intraday_combo_model_inputs():
    from src.workflow.cognitive_alpha_goal_codex_exec import (
        BASE_CONSTRAINED_REF_COL,
        COGALPHA_OPEN2OPEN_RISK_GATE_COL,
        CognitiveAlphaGoalCodexExecCardSpec,
        _attach_constituent_intraday_combo_features,
        _model_feature_columns,
        _position_spec,
    )

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            BASE_CONSTRAINED_REF_COL: [0.5, 0.4],
            COGALPHA_OPEN2OPEN_RISK_GATE_COL: [0.0, 1.0],
            "vix": [20.0, 25.0],
            "soxx_const_breadth_ret5_pos_lag1": [0.8, 0.2],
            "soxx_const_downside_ret5_lt_m5_lag1": [0.1, 0.5],
            "soxx_const_dispersion_ret5_lag1": [0.03, 0.04],
            "soxx_const_top5_minus_rest_ret5_lag1": [0.02, -0.01],
            "intraday15m_soxx_ret_lag1": [0.01, -0.02],
            "intraday15m_soxx_late_ret_lag1": [0.004, -0.003],
            "intraday15m_soxx_realized_vol_lag1": [0.005, 0.01],
            "intraday15m_soxx_close_location_lag1": [0.8, 0.2],
            "intraday15m_soxx_volume_z20_lag1": [0.5, -0.2],
            "intraday15m_soxl_minus_3x_soxx_ret_lag1": [-0.02, 0.03],
        }
    )

    out, combo_cols = _attach_constituent_intraday_combo_features(frame)
    model_features = _model_feature_columns([], [], out, extra_columns=combo_cols)
    position_spec = _position_spec(
        model_features,
        spec=CognitiveAlphaGoalCodexExecCardSpec(root=Path("/tmp")),
        test=("2026-01-02", "2026-01-05"),
    )

    assert "combo_semi_breadth_thrust_15m_lag1" in combo_cols
    assert "combo_levered_dislocation_absorption_lag1" in combo_cols
    assert "combo_top_weight_lead_confirm_15m_lag1" in combo_cols
    assert "combo_panic_recovery_quality_15m_lag1" in combo_cols
    assert "combo_crowded_close_risk_15m_lag1" in combo_cols
    assert out.loc[0, "combo_semi_breadth_thrust_15m_lag1"] == pytest.approx(0.008)
    assert out.loc[0, "combo_top_weight_lead_confirm_15m_lag1"] == pytest.approx(0.00008)
    assert set(combo_cols).issubset(set(model_features))
    assert all(item.get("base_position_col") not in combo_cols for item in position_spec.position_specs or [])
    assert all(item.get("risk_gate_col") not in combo_cols for item in position_spec.position_specs or [])


def _intraday_raw_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fixtures = {
        "SOXL": {
            "2026-01-02": {
                "open": [100.0, 104.0, 106.0],
                "high": [102.0, 106.0, 110.0],
                "low": [98.0, 103.0, 105.0],
                "close": [104.0, 106.0, 108.0],
                "volume": [100.0, 200.0, 300.0],
            },
            "2026-01-05": {
                "open": [108.0, 107.0, 105.0],
                "high": [109.0, 108.0, 106.0],
                "low": [106.0, 104.0, 102.0],
                "close": [107.0, 105.0, 103.0],
                "volume": [120.0, 220.0, 320.0],
            },
            "2026-01-06": {
                "open": [103.0, 104.0, 106.0],
                "high": [105.0, 107.0, 108.0],
                "low": [102.0, 103.0, 105.0],
                "close": [104.0, 106.0, 107.0],
                "volume": [130.0, 230.0, 330.0],
            },
        },
        "SOXX": {
            "2026-01-02": {
                "open": [50.0, 51.0, 51.5],
                "high": [51.0, 52.0, 52.5],
                "low": [49.5, 50.5, 51.0],
                "close": [51.0, 51.5, 52.0],
                "volume": [80.0, 90.0, 100.0],
            },
            "2026-01-05": {
                "open": [52.0, 51.8, 51.5],
                "high": [52.2, 52.0, 51.8],
                "low": [51.7, 51.4, 51.0],
                "close": [51.8, 51.5, 51.2],
                "volume": [81.0, 91.0, 101.0],
            },
            "2026-01-06": {
                "open": [51.2, 51.6, 52.0],
                "high": [51.8, 52.2, 52.6],
                "low": [51.0, 51.5, 51.9],
                "close": [51.6, 52.0, 52.4],
                "volume": [82.0, 92.0, 102.0],
            },
        },
    }
    times = ["09:45:00", "10:00:00", "16:00:00"]
    for instrument, by_date in fixtures.items():
        for date, fields in by_date.items():
            for index, time in enumerate(times):
                rows.append(
                    {
                        "instrument": instrument,
                        "datetime": pd.Timestamp(f"{date} {time}"),
                        "open": fields["open"][index],
                        "high": fields["high"][index],
                        "low": fields["low"][index],
                        "close": fields["close"][index],
                        "volume": fields["volume"][index],
                        "turnover": fields["close"][index] * fields["volume"][index],
                        "bar_index": float(index),
                        "bar_count": float(len(times)),
                    }
                )
    return pd.DataFrame(rows)

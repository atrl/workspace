from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_soxx_constituent_factor_frame_uses_lagged_qlib_breadth_and_dispersion(tmp_path):
    from src.data import QlibProviderConfig, build_qlib_store_from_frame
    from src.factor.soxx_constituent_factors import build_soxx_constituent_factor_frame

    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    raw = pd.DataFrame(
        [
            {
                "instrument": instrument,
                "datetime": date,
                "adj_close": price,
                "close": price,
                "volume": 1000,
                "weight_pct": weight,
            }
            for instrument, weight, prices in [
                ("AAA", 60.0, [100.0, 110.0, 121.0, 108.9]),
                ("BBB", 30.0, [100.0, 90.0, 99.0, 108.9]),
                ("CCC", 10.0, [100.0, 120.0, 114.0, 119.7]),
            ]
            for date, price in zip(dates, prices)
        ]
    )
    provider_uri = tmp_path / "qlib" / "soxx_constituents"
    build_qlib_store_from_frame(raw, provider_uri=provider_uri, as_of="2026-01-07")

    factors = build_soxx_constituent_factor_frame(
        QlibProviderConfig(provider_uri=provider_uri),
        instruments=["AAA", "BBB", "CCC"],
        start="2026-01-02",
        end="2026-01-07",
        lookbacks=(1,),
        lag_days=1,
        top_n=1,
    )

    rows = factors.set_index("datetime")
    assert "soxx_const_breadth_ret1_pos_lag1" in factors.attrs["feature_cols"]
    assert pd.isna(rows.loc[pd.Timestamp("2026-01-05"), "soxx_const_weighted_ret1_lag1"])
    assert rows.loc[pd.Timestamp("2026-01-06"), "soxx_const_breadth_ret1_pos_lag1"] == pytest.approx(2 / 3)
    assert rows.loc[pd.Timestamp("2026-01-06"), "soxx_const_weighted_ret1_lag1"] == pytest.approx(0.05)
    assert rows.loc[pd.Timestamp("2026-01-06"), "soxx_const_equal_ret1_lag1"] == pytest.approx(0.0666666667)
    assert rows.loc[pd.Timestamp("2026-01-06"), "soxx_const_top1_minus_rest_ret1_lag1"] == pytest.approx(0.125)
    assert rows.loc[pd.Timestamp("2026-01-06"), "soxx_const_available_count_lag1"] == pytest.approx(3.0)
    assert factors.attrs["source"] == "qlib.data.D.features"
    assert factors.attrs["lag_days"] == 1


def test_cognitive_alpha_goal_codex_exec_merges_constituent_features_as_model_inputs_not_gates():
    from src.workflow.cognitive_alpha_goal_codex_exec import (
        BASE_CONSTRAINED_REF_COL,
        COGALPHA_OPEN2OPEN_RISK_GATE_COL,
        CognitiveAlphaGoalCodexExecCardSpec,
        _attach_soxx_constituent_model_features,
        _model_feature_columns,
        _position_spec,
    )

    factor_col = "soxx_const_weighted_ret5_lag1"
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            BASE_CONSTRAINED_REF_COL: [0.5, 0.4],
            COGALPHA_OPEN2OPEN_RISK_GATE_COL: [0.0, 1.0],
            "vix": [20.0, 25.0],
        }
    )
    constituent = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            factor_col: [0.01, -0.02],
        }
    )
    constituent.attrs["feature_cols"] = [factor_col]

    out, constituent_cols = _attach_soxx_constituent_model_features(frame, constituent)
    model_features = _model_feature_columns([], [], out, extra_columns=constituent_cols)
    position_spec = _position_spec(
        model_features,
        spec=CognitiveAlphaGoalCodexExecCardSpec(root=Path("/tmp")),
        test=("2026-01-02", "2026-01-05"),
    )

    assert constituent_cols == [factor_col]
    assert out[factor_col].tolist() == [0.01, -0.02]
    assert factor_col in model_features
    assert all(item.get("base_position_col") != factor_col for item in position_spec.position_specs or [])
    assert all(item.get("risk_gate_col") != factor_col for item in position_spec.position_specs or [])

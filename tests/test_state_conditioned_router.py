from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.state_conditioned_router import (
    StateConditionedRouterConfig,
    build_state_conditioned_router_frame,
    screen_state_conditioned_router_configs,
)


def test_router_frame_caps_risk_and_lifts_rebound_only_outside_risk():
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "price": [100.0, 120.0, 90.0],
            "position": [0.20, 0.20, 0.20],
            "mom_5": [0.01, 0.04, 0.05],
            "mom_21": [-0.10, 0.50, 0.60],
            "drawdown_63": [-0.30, -0.10, -0.30],
            "rv_21": [0.10, 0.90, 0.90],
            "vol_of_vol_21": [0.01, 0.01, 0.01],
            "soxl_soxx_ratio_widen_21": [0.0, 0.0, 0.0],
            "soxl_soxx_ratio_z_252": [0.0, 0.0, -1.0],
            "soxx_mom_21": [0.0, 0.0, -0.10],
        }
    )
    upside = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "position": [0.60, 0.60, 0.60],
        }
    )

    routed = build_state_conditioned_router_frame(
        base,
        upside,
        StateConditionedRouterConfig(
            upside_cap=0.45,
            rebound_mom21=0.40,
            rebound_volq=0.50,
            risk_cap=0.10,
            risk_dd63=-0.20,
            risk_mom21=0.0,
            rebound_mom5=0.02,
            rebound_min_dd63=-0.35,
            use_semis_stress=True,
        ),
    )

    assert routed["router_position"].round(2).tolist() == [0.10, 0.45, 0.10]
    assert routed["rebound_state"].tolist() == [False, True, True]
    assert routed["risk_state"].tolist() == [True, False, True]
    assert {"strategy_nav", "buyhold_nav", "vol_threshold"}.issubset(routed.columns)


def test_router_frame_can_apply_path_equity_drawdown_controller():
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
            "price": [100.0, 80.0, 80.0, 120.0],
            "position": [1.0, 1.0, 1.0, 1.0],
            "mom_5": [0.0, 0.0, 0.0, 0.0],
            "mom_21": [0.0, 0.0, 0.0, 0.0],
            "drawdown_63": [0.0, 0.0, 0.0, 0.0],
            "rv_21": [0.0, 0.0, 0.0, 0.0],
        }
    )
    upside = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
            "position": [1.0, 1.0, 1.0, 1.0],
        }
    )

    routed = build_state_conditioned_router_frame(
        base,
        upside,
        StateConditionedRouterConfig(
            rebound_mom21=99.0,
            risk_dd63=-1.0,
            use_semis_stress=False,
            equity_dd_start=0.10,
            equity_dd_full=0.20,
            equity_dd_min_cap=0.25,
        ),
    )

    assert routed["router_position"].round(2).tolist() == [1.0, 0.25, 0.25, 1.0]
    assert routed["equity_drawdown_cap"].round(2).tolist() == [1.0, 0.25, 0.25, 1.0]
    assert routed["equity_drawdown_control"].tolist() == [False, True, True, False]
    assert round(routed["strategy_nav"].iloc[-1], 2) == 900000.0


def test_screen_router_configs_scores_only_train_window():
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
            ),
            "price": [100.0, 120.0, 60.0, 180.0],
            "position": [0.20, 0.20, 0.20, 0.20],
            "mom_5": [0.03, 0.03, 0.03, 0.03],
            "mom_21": [0.50, 0.50, 0.50, 0.50],
            "drawdown_63": [-0.10, -0.10, -0.10, -0.10],
            "rv_21": [0.90, 0.90, 0.90, 0.90],
        }
    )
    upside = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
            ),
            "position": [0.80, 0.80, 0.80, 0.80],
        }
    )
    configs = [
        StateConditionedRouterConfig(upside_cap=0.20, rebound_mom21=0.40),
        StateConditionedRouterConfig(upside_cap=0.80, rebound_mom21=0.40),
    ]

    scored = screen_state_conditioned_router_configs(
        base,
        upside,
        configs,
        fold="wf_2022_train",
        train_start="2020-01-02",
        train_end="2020-01-03",
    )

    by_cap = scored.set_index("upside_cap")
    assert by_cap.loc[0.20, "train_rows"] == 2
    assert by_cap.loc[0.80, "train_rows"] == 2
    assert by_cap.loc[0.80, "train_return"] > by_cap.loc[0.20, "train_return"]
    assert by_cap.loc[0.80, "full_rows"] == 4

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt


def _signals(periods: int = 18) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.bdate_range("2026-01-02", periods=periods),
            "soxl_return": np.zeros(periods),
            "soxx_return": np.zeros(periods),
            "realized_vol_20": np.full(periods, 0.35),
            "p_benign_chop": np.full(periods, 0.55),
            "p_breakout": np.full(periods, 0.25),
            "p_breakdown": np.full(periods, 0.10),
            "p_whipsaw": np.full(periods, 0.10),
            "price_break": np.zeros(periods, dtype=bool),
            "breadth_break": np.zeros(periods, dtype=bool),
            "high_chop_intensity": np.full(periods, 0.25),
        }
    )


def test_breakdown_gate_needs_probability_and_confirming_market_break():
    from src.backtest.multi_asset_risk_control import apply_multi_asset_risk_control

    signals = _signals()
    signals.loc[5, "p_breakdown"] = 0.90
    probability_only = apply_multi_asset_risk_control(signals)
    signals.loc[5, "price_break"] = True
    confirmed = apply_multi_asset_risk_control(signals)

    assert probability_only.loc[5, "controller_state"] != "breakdown"
    assert confirmed.loc[5, "controller_state"] == "breakdown"
    assert confirmed.loc[5, "soxl_weight"] <= 0.15
    assert confirmed.loc[5, "risky_weight"] < probability_only.loc[5, "risky_weight"]


def test_controller_cuts_immediately_and_recovers_by_bounded_steps():
    from src.backtest.multi_asset_risk_control import RiskControlSpec, apply_multi_asset_risk_control

    signals = _signals()
    signals.loc[5, ["p_breakdown", "price_break"]] = [0.90, True]
    spec = RiskControlSpec(cooldown_days=2, risk_up_step=0.20, min_state_days=1)
    controlled = apply_multi_asset_risk_control(signals, spec=spec)

    assert controlled.loc[5, "risky_weight"] < controlled.loc[4, "risky_weight"]
    assert controlled.loc[5, "risky_weight"] <= 0.25
    increases = controlled["risky_weight"].diff().dropna()
    assert increases.max() <= spec.risk_up_step + 1e-12
    assert controlled.loc[6, "risky_weight"] <= controlled.loc[5, "risky_weight"] + 1e-12


def test_drawdown_override_uses_only_returns_available_before_new_target():
    from src.backtest.multi_asset_risk_control import RiskControlSpec, apply_multi_asset_risk_control

    signals = _signals()
    signals.loc[4:6, "soxl_return"] = [-0.12, -0.05, -0.05]
    signals.loc[4:6, "soxx_return"] = [-0.02, -0.02, -0.02]
    spec = RiskControlSpec(drawdown_hard_stop=-0.08, drawdown_soft_stop=-0.04, min_state_days=1)
    controlled = apply_multi_asset_risk_control(signals, spec=spec)

    hard_rows = controlled[controlled["controller_state"] == "drawdown_hard_stop"]
    assert not hard_rows.empty
    assert hard_rows["soxl_weight"].eq(0.0).all()
    assert hard_rows["soxx_weight"].le(0.10).all()

    mutated = signals.copy()
    mutated.loc[12:, ["soxl_return", "soxx_return"]] = 1.0
    rebuilt = apply_multi_asset_risk_control(mutated, spec=spec)
    columns = ["datetime", "soxl_weight", "soxx_weight", "cash_weight", "controller_state"]
    pdt.assert_frame_equal(controlled.loc[:11, columns], rebuilt.loc[:11, columns])


def test_high_chop_replaces_leverage_with_soxx_and_weights_are_valid():
    from src.backtest.multi_asset_risk_control import apply_multi_asset_risk_control

    signals = _signals()
    signals.loc[5:8, "p_whipsaw"] = 0.75
    signals.loc[5:8, "high_chop_intensity"] = 0.80
    controlled = apply_multi_asset_risk_control(signals)

    chop = controlled.loc[5:8]
    assert chop["controller_state"].eq("high_chop").all()
    assert chop["soxl_weight"].le(0.25).all()
    assert chop["soxx_weight"].gt(0.0).all()
    assert controlled[["soxl_weight", "soxx_weight", "cash_weight"]].ge(0.0).all().all()
    np.testing.assert_allclose(
        controlled[["soxl_weight", "soxx_weight", "cash_weight"]].sum(axis=1),
        1.0,
    )


def test_evaluation_charges_two_asset_turnover_at_next_session():
    from src.backtest.multi_asset_risk_control import evaluate_multi_asset_risk_control

    controlled = pd.DataFrame(
        {
            "datetime": pd.bdate_range("2026-01-02", periods=3),
            "soxl_return": [0.0, 0.10, 0.00],
            "soxx_return": [0.0, 0.02, 0.00],
            "soxl_weight": [0.5, 0.0, 0.0],
            "soxx_weight": [0.0, 0.5, 0.5],
            "cash_weight": [0.5, 0.5, 0.5],
        }
    )
    report, metrics = evaluate_multi_asset_risk_control(controlled, cost_per_turnover=0.001)

    assert report.loc[1, "gross_return"] == 0.05
    assert report.loc[1, "turnover"] == 0.5
    assert report.loc[1, "strategy_return"] == 0.0495
    assert metrics["total_turnover"] == 1.0
    assert {"annualized_return", "annualized_volatility", "max_drawdown", "sharpe"}.issubset(metrics)

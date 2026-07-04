from __future__ import annotations

import pandas as pd

from src.workflow.cognitive_alpha_intraday_overlay import (
    evaluate_conditional_overlay_candidates,
    evaluate_observable_delayed_signal_candidates,
    select_validation_robust_delayed_policy,
    simulate_observable_delayed_signal_overlay,
    simulate_execution_overlays,
)


def test_simulate_execution_overlays_delays_entry_and_records_intraday_variants():
    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "target_position": 1.0,
                "strategy_return": 0.10,
            }
        ]
    )
    daily_prices = pd.DataFrame(
        [
            {"datetime": "2026-01-02", "open": 90.0},
            {"datetime": "2026-01-05", "open": 100.0},
            {"datetime": "2026-01-06", "open": 110.0},
        ]
    )
    intraday = pd.DataFrame(
        [
            {
                "datetime": "2026-01-05 09:45:00",
                "date": "2026-01-05",
                "open": 101.0,
                "close": 102.0,
                "prob_lgb_xgb_consensus": 0.40,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2026-01-05 10:00:00",
                "date": "2026-01-05",
                "open": 102.0,
                "close": 103.0,
                "prob_lgb_xgb_consensus": 0.60,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2026-01-05 10:15:00",
                "date": "2026-01-05",
                "open": 104.0,
                "close": 105.0,
                "prob_lgb_xgb_consensus": 0.52,
                "regime_gate_v2_dislocation": False,
            },
            {
                "datetime": "2026-01-05 10:30:00",
                "date": "2026-01-05",
                "open": 106.0,
                "close": 107.0,
                "prob_lgb_xgb_consensus": 0.51,
                "regime_gate_v2_dislocation": False,
            },
        ]
    )

    rows = simulate_execution_overlays(
        daily,
        daily_prices=daily_prices,
        intraday=intraday,
        signal_threshold=0.55,
        risk_threshold=0.45,
        cost_bps=0.0,
    )

    assert set(rows["variant"]) == {
        "baseline_next_open",
        "overlay_a_signal_gate",
        "overlay_b_twap4",
        "overlay_c_risk_exit",
    }
    by_variant = rows.set_index("variant")
    assert by_variant.loc["baseline_next_open", "strategy_return"] == 0.10
    assert by_variant.loc["overlay_a_signal_gate", "entry_time"] == "2026-01-05 10:15:00"
    assert by_variant.loc["overlay_a_signal_gate", "strategy_return"] == (110.0 / 104.0 - 1.0)
    assert by_variant.loc["overlay_b_twap4", "entry_time"] == "twap4_first_4_bars"
    assert by_variant.loc["overlay_c_risk_exit", "exit_time"] == "2026-01-05 09:45:00"
    assert by_variant.loc["overlay_c_risk_exit", "strategy_return"] == (102.0 / 100.0 - 1.0)


def test_evaluate_conditional_overlay_candidates_uses_validation_quantiles_only():
    trades = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "validation",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "strategy_return": 0.10,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-03",
                "segment": "validation",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "strategy_return": -0.02,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-04",
                "segment": "test",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "strategy_return": 0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-05",
                "segment": "test",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "strategy_return": -0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-02",
                "segment": "validation",
                "variant": "overlay_b_twap4",
                "cost_bps": 20.0,
                "strategy_return": 0.08,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-03",
                "segment": "validation",
                "variant": "overlay_b_twap4",
                "cost_bps": 20.0,
                "strategy_return": -0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-04",
                "segment": "test",
                "variant": "overlay_b_twap4",
                "cost_bps": 20.0,
                "strategy_return": 0.50,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-05",
                "segment": "test",
                "variant": "overlay_b_twap4",
                "cost_bps": 20.0,
                "strategy_return": 0.50,
                "skipped_reason": None,
            },
        ]
    )
    regime_features = pd.DataFrame(
        [
            {"datetime": "2026-01-02", "score": 0.20},
            {"datetime": "2026-01-03", "score": 0.80},
            {"datetime": "2026-01-04", "score": 100.00},
            {"datetime": "2026-01-05", "score": 200.00},
        ]
    )

    candidates = evaluate_conditional_overlay_candidates(
        trades,
        regime_features,
        cost_bps=20.0,
        variants=("overlay_b_twap4",),
        condition_columns=("score",),
        quantiles=(0.50,),
    )

    assert not candidates.empty
    assert set(candidates["selection_uses_test_segment"]) == {False}
    assert set(candidates["condition_threshold_source"]) == {"validation_quantile"}
    assert set(candidates["condition_threshold"]) == {0.50}


def test_simulate_observable_delayed_signal_overlay_waits_for_observation_window():
    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "target_position": 1.0,
                "strategy_return": 0.10,
            }
        ]
    )
    daily_prices = pd.DataFrame(
        [
            {"datetime": "2026-01-02", "open": 90.0},
            {"datetime": "2026-01-05", "open": 100.0},
            {"datetime": "2026-01-06", "open": 110.0},
        ]
    )
    intraday = pd.DataFrame(
        [
            {
                "datetime": "2026-01-05 09:45:00",
                "date": "2026-01-05",
                "open": 100.0,
                "close": 101.0,
                "prob_lgb_xgb_consensus": 0.70,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2026-01-05 10:00:00",
                "date": "2026-01-05",
                "open": 102.0,
                "close": 103.0,
                "prob_lgb_xgb_consensus": 0.70,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2026-01-05 10:15:00",
                "date": "2026-01-05",
                "open": 104.0,
                "close": 105.0,
                "prob_lgb_xgb_consensus": 0.80,
                "regime_gate_v2_dislocation": True,
            },
        ]
    )

    rows = simulate_observable_delayed_signal_overlay(
        daily,
        daily_prices=daily_prices,
        intraday=intraday,
        observation_bars=2,
        condition_feature="observed_prob_min",
        condition_operator="ge",
        condition_threshold=0.70,
        signal_threshold=0.65,
        cost_bps=0.0,
    )

    assert set(rows["variant"]) == {"baseline_next_open", "delayed_signal_after_30m"}
    by_variant = rows.set_index("variant")
    assert by_variant.loc["baseline_next_open", "entry_price"] == 100.0
    assert by_variant.loc["delayed_signal_after_30m", "entry_time"] == "2026-01-05 10:15:00"
    assert by_variant.loc["delayed_signal_after_30m", "entry_price"] == 104.0
    assert by_variant.loc["delayed_signal_after_30m", "strategy_return"] == (110.0 / 104.0 - 1.0)


def test_simulate_observable_delayed_signal_overlay_can_exit_on_later_intraday_risk_signal():
    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "target_position": 1.0,
                "strategy_return": 0.10,
            }
        ]
    )
    daily_prices = pd.DataFrame(
        [
            {"datetime": "2026-01-02", "open": 90.0},
            {"datetime": "2026-01-05", "open": 100.0},
            {"datetime": "2026-01-06", "open": 110.0},
        ]
    )
    intraday = pd.DataFrame(
        [
            {
                "datetime": "2026-01-05 09:45:00",
                "date": "2026-01-05",
                "open": 100.0,
                "close": 101.0,
                "prob_lgb_xgb_consensus": 0.72,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2026-01-05 10:00:00",
                "date": "2026-01-05",
                "open": 102.0,
                "close": 103.0,
                "prob_lgb_xgb_consensus": 0.74,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2026-01-05 10:15:00",
                "date": "2026-01-05",
                "open": 104.0,
                "close": 105.0,
                "prob_lgb_xgb_consensus": 0.80,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2026-01-05 10:30:00",
                "date": "2026-01-05",
                "open": 106.0,
                "close": 98.0,
                "prob_lgb_xgb_consensus": 0.20,
                "regime_gate_v2_dislocation": True,
            },
        ]
    )

    rows = simulate_observable_delayed_signal_overlay(
        daily,
        daily_prices=daily_prices,
        intraday=intraday,
        observation_bars=2,
        condition_feature="observed_prob_min",
        condition_operator="ge",
        condition_threshold=0.70,
        signal_threshold=0.65,
        risk_threshold=0.50,
        cost_bps=0.0,
    )

    assert set(rows["variant"]) == {"baseline_next_open", "delayed_signal_after_30m_risk_exit"}
    by_variant = rows.set_index("variant")
    assert by_variant.loc["delayed_signal_after_30m_risk_exit", "entry_time"] == "2026-01-05 10:15:00"
    assert by_variant.loc["delayed_signal_after_30m_risk_exit", "exit_time"] == "2026-01-05 10:30:00"
    assert by_variant.loc["delayed_signal_after_30m_risk_exit", "exit_price"] == 98.0
    assert by_variant.loc["delayed_signal_after_30m_risk_exit", "strategy_return"] == (98.0 / 104.0 - 1.0)


def test_evaluate_observable_delayed_signal_candidates_includes_risk_exit_policy():
    daily = pd.DataFrame(
        [
            {"datetime": "2026-01-02", "segment": "validation", "target_position": 1.0, "strategy_return": 0.10},
            {"datetime": "2026-01-06", "segment": "validation", "target_position": 1.0, "strategy_return": 0.05},
            {"datetime": "2026-01-08", "segment": "test", "target_position": 1.0, "strategy_return": 0.03},
        ]
    )
    daily_prices = pd.DataFrame(
        [
            {"datetime": "2026-01-02", "open": 90.0},
            {"datetime": "2026-01-05", "open": 100.0},
            {"datetime": "2026-01-06", "open": 110.0},
            {"datetime": "2026-01-07", "open": 112.0},
            {"datetime": "2026-01-08", "open": 118.0},
            {"datetime": "2026-01-09", "open": 120.0},
            {"datetime": "2026-01-12", "open": 124.0},
        ]
    )
    intraday_rows = []
    for date, probs in {
        "2026-01-05": [0.72, 0.74, 0.20],
        "2026-01-07": [0.42, 0.44, 0.20],
        "2026-01-09": [0.80, 0.82, 0.20],
    }.items():
        for time, open_price, close_price, prob in zip(
            ["09:45:00", "10:00:00", "10:15:00"],
            [100.0, 102.0, 104.0],
            [101.0, 103.0, 98.0],
            probs,
            strict=True,
        ):
            intraday_rows.append(
                {
                    "datetime": f"{date} {time}",
                    "date": date,
                    "open": open_price,
                    "close": close_price,
                    "prob_lgb_xgb_consensus": prob,
                    "regime_gate_v2_dislocation": True,
                }
            )
    intraday = pd.DataFrame(intraday_rows)

    candidates = evaluate_observable_delayed_signal_candidates(
        daily,
        daily_prices=daily_prices,
        intraday=intraday,
        cost_bps_values=(20.0,),
        observation_bars_grid=(2,),
        signal_threshold_grid=(0.50,),
        risk_threshold_grid=(None, 0.30),
        quantiles=(0.50,),
    )

    assert {"hold_to_next_open", "risk_exit"}.issubset(set(candidates["exit_policy"]))
    assert set(candidates["condition_threshold_source"]) == {"validation_quantile"}
    assert set(candidates["selection_uses_test_segment"]) == {False}
    assert "delayed_signal_after_30m_risk_exit" in set(candidates["variant"])


def test_select_validation_robust_delayed_policy_requires_cost_and_skip_robustness():
    candidates = pd.DataFrame(
        [
            {
                "policy_id": "fragile_high_score",
                "selection_score": 120.0,
                "validation_drawdown_improvement_pct": 10.0,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 90.0,
                "validation_max_drawdown_pct": -10.0,
                "validation_skipped_periods": 60,
                "validation_periods": 250,
            },
            {
                "policy_id": "fragile_high_score",
                "selection_score": 120.0,
                "validation_drawdown_improvement_pct": 10.0,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 90.0,
                "validation_max_drawdown_pct": -10.0,
                "validation_skipped_periods": 60,
                "validation_periods": 250,
            },
            {
                "policy_id": "robust_lower_score",
                "selection_score": 95.0,
                "validation_drawdown_improvement_pct": 3.0,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 88.0,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 5,
                "validation_periods": 250,
            },
            {
                "policy_id": "robust_lower_score",
                "selection_score": 95.0,
                "validation_drawdown_improvement_pct": 3.0,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 87.0,
                "validation_max_drawdown_pct": -16.0,
                "validation_skipped_periods": 5,
                "validation_periods": 250,
            },
        ]
    )

    selected = select_validation_robust_delayed_policy(candidates)

    assert selected["policy_id"] == "robust_lower_score"
    assert selected["validation_passes_target"] is True


def test_select_validation_robust_delayed_policy_keeps_risk_exit_diagnostic_by_default():
    candidates = pd.DataFrame(
        [
            {
                "policy_id": "risk_exit_high_validation_score",
                "variant": "delayed_signal_after_30m_risk_exit",
                "exit_policy": "risk_exit",
                "selection_score": 140.0,
                "validation_drawdown_improvement_pct": 12.0,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 90.0,
                "validation_max_drawdown_pct": -6.0,
                "validation_skipped_periods": 5,
                "validation_periods": 250,
            },
            {
                "policy_id": "risk_exit_high_validation_score",
                "variant": "delayed_signal_after_30m_risk_exit",
                "exit_policy": "risk_exit",
                "selection_score": 140.0,
                "validation_drawdown_improvement_pct": 12.0,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 89.0,
                "validation_max_drawdown_pct": -7.0,
                "validation_skipped_periods": 5,
                "validation_periods": 250,
            },
            {
                "policy_id": "hold_robust_lower_score",
                "variant": "delayed_signal_after_30m",
                "exit_policy": "hold_to_next_open",
                "selection_score": 100.0,
                "validation_drawdown_improvement_pct": 4.0,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 88.0,
                "validation_max_drawdown_pct": -14.0,
                "validation_skipped_periods": 5,
                "validation_periods": 250,
            },
            {
                "policy_id": "hold_robust_lower_score",
                "variant": "delayed_signal_after_30m",
                "exit_policy": "hold_to_next_open",
                "selection_score": 100.0,
                "validation_drawdown_improvement_pct": 4.0,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 87.0,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 5,
                "validation_periods": 250,
            },
        ]
    )

    selected = select_validation_robust_delayed_policy(candidates)

    assert selected["policy_id"] == "hold_robust_lower_score"
    assert selected["exit_policy"] == "hold_to_next_open"

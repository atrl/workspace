from __future__ import annotations

import pandas as pd
import pytest

import src.workflow.cognitive_alpha_intraday_5min_risk_sleeve as risk_sleeve_workflow
import src.workflow.cognitive_alpha_intraday_local_fallback_overlay as local_fallback_workflow
import src.workflow.cognitive_alpha_intraday_overlay as overlay_workflow
from src.workflow.cognitive_alpha_intraday_5min_risk_sleeve import (
    _attach_external_context_to_5min_bars,
    evaluate_5min_risk_sleeve_candidates,
    simulate_5min_risk_sleeve,
)
from src.workflow.cognitive_alpha_intraday_overlay import (
    CognitiveAlphaIntradayOverlaySpec,
    build_intraday_optimization_plan,
    evaluate_conditional_overlay_candidates,
    evaluate_observable_delayed_signal_candidates,
    select_conditional_risk_sleeve_policy,
    select_intraday_flat_sleeve_policy,
    select_validation_robust_delayed_policy,
    summarize_execution_risk_diagnostics,
    simulate_observable_delayed_signal_overlay,
    simulate_execution_overlays,
)


def test_simulate_5min_risk_sleeve_keeps_daily_entry_and_reduces_after_trigger():
    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "walk_forward_fold": "wf_2026",
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
    bars_5min = pd.DataFrame(
        [
            {"datetime": "2026-01-05 09:35:00", "date": "2026-01-05", "open": 100.0, "close": 99.2},
            {"datetime": "2026-01-05 09:40:00", "date": "2026-01-05", "open": 99.1, "close": 97.4},
            {"datetime": "2026-01-05 09:45:00", "date": "2026-01-05", "open": 97.6, "close": 98.0},
        ]
    )

    rows = simulate_5min_risk_sleeve(
        daily,
        daily_prices=daily_prices,
        bars_5min=bars_5min,
        drawdown_threshold=-0.02,
        cost_bps=0.0,
    )

    by_variant = rows.set_index("variant")
    assert set(by_variant.index) == {
        "baseline_next_open",
        "risk_exit_full_5m",
        "risk_exit_full_5m_reenter_close",
        "risk_reduce_half_5m",
        "risk_reduce_quarter_5m",
        "risk_reduce_quarter_5m_reenter_close",
        "risk_reduce_eighth_5m",
    }
    assert by_variant.loc["baseline_next_open", "entry_time"] == "2026-01-05"
    assert by_variant.loc["baseline_next_open", "strategy_return"] == pytest.approx(0.10)
    assert by_variant.loc["risk_exit_full_5m", "entry_time"] == "2026-01-05 09:35:00"
    assert by_variant.loc["risk_exit_full_5m", "exit_time"] == "2026-01-05 09:40:00"
    assert by_variant.loc["risk_exit_full_5m", "strategy_return"] == pytest.approx(-0.026)
    assert by_variant.loc["risk_reduce_half_5m", "strategy_return"] == pytest.approx(0.037)
    assert by_variant.loc["risk_reduce_quarter_5m", "strategy_return"] == pytest.approx(0.0685)
    assert by_variant.loc["risk_reduce_eighth_5m", "strategy_return"] == pytest.approx(0.08425)
    assert by_variant.loc["risk_exit_full_5m", "skipped_reason"] is None


def test_simulate_5min_risk_sleeve_executes_on_next_5min_open_after_trigger():
    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "walk_forward_fold": "wf_2026",
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
    bars_5min = pd.DataFrame(
        [
            {"datetime": "2026-01-05 09:35:00", "date": "2026-01-05", "open": 100.0, "close": 99.2},
            {"datetime": "2026-01-05 09:40:00", "date": "2026-01-05", "open": 99.1, "close": 97.4},
            {"datetime": "2026-01-05 09:45:00", "date": "2026-01-05", "open": 97.6, "close": 98.0},
        ]
    )

    rows = simulate_5min_risk_sleeve(
        daily,
        daily_prices=daily_prices,
        bars_5min=bars_5min,
        drawdown_threshold=-0.02,
        cost_bps=0.0,
        execution_delay_bars=1,
        execution_price="open",
    )

    by_variant = rows.set_index("variant")
    assert by_variant.loc["risk_exit_full_5m", "entry_time"] == "2026-01-05 09:35:00"
    assert by_variant.loc["risk_exit_full_5m", "trigger_time"] == "2026-01-05 09:40:00"
    assert by_variant.loc["risk_exit_full_5m", "exit_time"] == "2026-01-05 09:45:00"
    assert by_variant.loc["risk_exit_full_5m", "trigger_price"] == pytest.approx(97.4)
    assert by_variant.loc["risk_exit_full_5m", "exit_price"] == pytest.approx(97.6)
    assert by_variant.loc["risk_exit_full_5m", "trigger_drawdown_from_entry"] == pytest.approx(-0.026)
    assert by_variant.loc["risk_exit_full_5m", "gross_return"] == pytest.approx(-0.024)
    assert by_variant.loc["risk_exit_full_5m", "strategy_return"] == pytest.approx(-0.024)
    assert by_variant.loc["risk_reduce_half_5m", "strategy_return"] == pytest.approx(0.038)
    assert by_variant.loc["risk_reduce_quarter_5m", "strategy_return"] == pytest.approx(0.069)
    assert by_variant.loc["risk_reduce_eighth_5m", "strategy_return"] == pytest.approx(0.0845)
    assert bool(by_variant.loc["risk_exit_full_5m", "risk_triggered"]) is True
    assert by_variant.loc["risk_exit_full_5m", "skipped_reason"] is None


def test_simulate_5min_risk_sleeve_can_reenter_at_close_after_intraday_risk_reduction():
    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "walk_forward_fold": "wf_2026",
                "target_position": 1.0,
                "strategy_return": 0.20,
            }
        ]
    )
    daily_prices = pd.DataFrame(
        [
            {"datetime": "2026-01-02", "open": 90.0},
            {"datetime": "2026-01-05", "open": 100.0},
            {"datetime": "2026-01-06", "open": 120.0},
        ]
    )
    bars_5min = pd.DataFrame(
        [
            {"datetime": "2026-01-05 09:35:00", "date": "2026-01-05", "open": 100.0, "close": 99.0},
            {"datetime": "2026-01-05 09:40:00", "date": "2026-01-05", "open": 99.0, "close": 95.0},
            {"datetime": "2026-01-05 16:00:00", "date": "2026-01-05", "open": 91.0, "close": 90.0},
        ]
    )

    rows = simulate_5min_risk_sleeve(
        daily,
        daily_prices=daily_prices,
        bars_5min=bars_5min,
        drawdown_threshold=-0.04,
        execution_delay_bars=0,
        execution_price="close",
        cost_bps=0.0,
        variants=("risk_exit_full_5m_reenter_close", "risk_reduce_quarter_5m_reenter_close"),
    )

    by_variant = rows.set_index("variant")
    assert by_variant.loc["baseline_next_open", "strategy_return"] == pytest.approx(0.20)
    assert by_variant.loc["risk_exit_full_5m_reenter_close", "entry_day_strategy_return"] == pytest.approx(-0.05)
    assert by_variant.loc["risk_exit_full_5m_reenter_close", "strategy_return"] == pytest.approx((95.0 / 100.0) * (120.0 / 90.0) - 1.0)
    assert by_variant.loc["risk_reduce_quarter_5m_reenter_close", "entry_day_strategy_return"] == pytest.approx(
        0.25 * (95.0 / 100.0 - 1.0) + 0.75 * (90.0 / 100.0 - 1.0)
    )
    assert by_variant.loc["risk_reduce_quarter_5m_reenter_close", "strategy_return"] == pytest.approx(
        0.75 * (120.0 / 100.0 - 1.0) + 0.25 * ((95.0 / 100.0) * (120.0 / 90.0) - 1.0)
    )


def test_simulate_5min_risk_sleeve_can_limit_output_to_requested_variant():
    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "walk_forward_fold": "wf_2026",
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
    bars_5min = pd.DataFrame(
        [
            {"datetime": "2026-01-05 09:35:00", "date": "2026-01-05", "open": 100.0, "close": 99.2},
            {"datetime": "2026-01-05 09:40:00", "date": "2026-01-05", "open": 99.1, "close": 97.4},
            {"datetime": "2026-01-05 09:45:00", "date": "2026-01-05", "open": 97.6, "close": 98.0},
        ]
    )

    rows = simulate_5min_risk_sleeve(
        daily,
        daily_prices=daily_prices,
        bars_5min=bars_5min,
        drawdown_threshold=-0.02,
        cost_bps=0.0,
        variants=("risk_reduce_eighth_5m",),
    )

    assert set(rows["variant"]) == {"baseline_next_open", "risk_reduce_eighth_5m"}
    by_variant = rows.set_index("variant")
    assert by_variant.loc["risk_reduce_eighth_5m", "strategy_return"] == pytest.approx(0.08425)


def test_selected_trades_replay_accepts_score_only_policy_without_drawdown_threshold():
    selected = {
        "policy_id": "risk_exit_full_5m_risk_score_only_dd_disabled_risk_0.60000",
        "variant": "risk_exit_full_5m",
        "trigger_mode": "risk_score_only",
        "drawdown_threshold": None,
        "risk_score_threshold": 0.60,
        "context_trigger_source": "disabled",
    }

    trades = risk_sleeve_workflow._simulate_selected_trades(
        selected,
        fold_daily=pd.DataFrame(
            [
                {
                    "datetime": "2026-01-02",
                    "segment": "test",
                    "walk_forward_fold": "wf_2026",
                    "target_position": 1.0,
                    "strategy_return": 0.10,
                }
            ]
        ),
        daily_prices=pd.DataFrame(
            [
                {"datetime": "2026-01-02", "open": 90.0},
                {"datetime": "2026-01-05", "open": 100.0},
                {"datetime": "2026-01-06", "open": 110.0},
            ]
        ),
        bars_5min=pd.DataFrame(
            [
                {"datetime": "2026-01-05 09:35:00", "date": "2026-01-05", "open": 100.0, "close": 99.8},
                {"datetime": "2026-01-05 09:40:00", "date": "2026-01-05", "open": 99.6, "close": 99.7},
            ]
        ),
        fold_scores=pd.DataFrame(
            [
                {
                    "datetime": "2026-01-05 09:35:00",
                    "date": "2026-01-05",
                    "risk_score": 0.75,
                    "prediction_available_gate": True,
                }
            ]
        ),
        required_cost_bps=(20.0,),
    )

    assert set(trades["variant"]) == {"baseline_next_open", "risk_exit_full_5m"}
    assert bool(trades.set_index("variant").loc["risk_exit_full_5m", "risk_triggered"]) is True


def test_5min_risk_scores_prefer_dedicated_adverse_event_risk_score():
    scores = risk_sleeve_workflow._risk_scores_for_fold(
        pd.DataFrame(
            [
                {
                    "datetime": "2026-01-05 09:45:00",
                    "date": "2026-01-05",
                    "walk_forward_fold": "wf_2026",
                    "selected_intraday_model": "lgb",
                    "prob_lgb": 0.80,
                    "adverse_event_risk_score": 0.65,
                    "prediction_available_gate": True,
                }
            ]
        ),
        fold="wf_2026",
    )

    assert scores.iloc[0]["risk_score"] == pytest.approx(0.65)
    assert scores.iloc[0]["risk_score_source"] == "adverse_event_risk_score"


def test_5min_risk_scores_fallback_inverts_execution_improvement_probability():
    scores = risk_sleeve_workflow._risk_scores_for_fold(
        pd.DataFrame(
            [
                {
                    "datetime": "2026-01-05 09:45:00",
                    "date": "2026-01-05",
                    "walk_forward_fold": "wf_2026",
                    "selected_intraday_model": "lgb",
                    "prob_lgb": 0.20,
                    "prediction_available_gate": True,
                }
            ]
        ),
        fold="wf_2026",
    )

    assert scores.iloc[0]["risk_score"] == pytest.approx(0.80)
    assert scores.iloc[0]["risk_score_source"] == "inverse_execution_improvement_probability"


def test_local_fallback_adverse_event_risk_head_trains_only_on_train_segment(monkeypatch):
    trained_targets: list[list[int]] = []

    class FakeRiskModel:
        def __init__(self, probability: float):
            self.probability = probability
            self.feature_importances_ = [1.0, 0.5]

        def fit(self, features: pd.DataFrame, target: pd.Series):
            assert list(features.columns) == ["feature_a", "feature_b"]
            trained_targets.append([int(value) for value in target.tolist()])
            return self

        def predict_proba(self, features: pd.DataFrame):
            return [[1.0 - self.probability, self.probability] for _ in range(len(features))]

    monkeypatch.setattr(
        local_fallback_workflow,
        "_models",
        lambda: {"lgb": FakeRiskModel(0.70), "xgb": FakeRiskModel(0.30)},
    )
    segments = {
        "train": pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-02 09:45:00", periods=4, freq="15min"),
                "date": ["2024-01-02"] * 4,
                "feature_a": [0.0, 1.0, 2.0, 3.0],
                "feature_b": [1.0, 1.0, 1.0, 1.0],
                "adverse_event_risk_target": [0, 1, 0, 1],
            }
        ),
        "valid": pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2025-01-02 09:45:00")],
                "date": ["2025-01-02"],
                "feature_a": [4.0],
                "feature_b": [1.0],
                "adverse_event_risk_target": [1],
            }
        ),
        "test": pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2026-01-02 09:45:00")],
                "date": ["2026-01-02"],
                "feature_a": [5.0],
                "feature_b": [1.0],
                "adverse_event_risk_target": [0],
            }
        ),
    }
    predictions = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2025-01-02 09:45:00"), pd.Timestamp("2026-01-02 09:45:00")],
            "date": ["2025-01-02", "2026-01-02"],
            "segment": ["valid", "test"],
            "prob_lgb": [0.6, 0.4],
        }
    )

    out, importance = local_fallback_workflow._append_adverse_event_risk_scores(
        predictions,
        segments=segments,
        feature_cols=["feature_a", "feature_b"],
    )

    assert trained_targets == [[0, 1, 0, 1], [0, 1, 0, 1]]
    assert out["risk_prob_lgb"].tolist() == pytest.approx([0.70, 0.70])
    assert out["risk_prob_xgb"].tolist() == pytest.approx([0.30, 0.30])
    assert out["adverse_event_risk_score"].tolist() == pytest.approx([0.50, 0.50])
    assert out["adverse_event_risk_score_source"].tolist() == ["lgb_xgb_adverse_event_consensus"] * 2
    assert set(importance["model"]) == {"adverse_event_risk_lgb", "adverse_event_risk_xgb"}


def test_evaluate_5min_risk_sleeve_candidates_reuses_trigger_replay_across_variants(monkeypatch):
    policies = [
        {
            "policy_id": "half_same_trigger",
            "variant": "risk_reduce_half_5m",
            "trigger_mode": "drawdown_only",
            "drawdown_threshold": -0.02,
            "risk_score_threshold": None,
            "context_trigger_source": "disabled",
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_uses_test_segment": False,
        },
        {
            "policy_id": "eighth_same_trigger",
            "variant": "risk_reduce_eighth_5m",
            "trigger_mode": "drawdown_only",
            "drawdown_threshold": -0.02,
            "risk_score_threshold": None,
            "context_trigger_source": "disabled",
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_uses_test_segment": False,
        },
    ]
    for policy in policies:
        policy.update(risk_sleeve_workflow._context_threshold_row({}))
    calls: list[tuple[str, ...]] = []

    def fake_policy_grid(*args, **kwargs):
        return policies

    def fake_simulate(*args, variants=None, **kwargs):
        calls.append(tuple(variants or ()))
        rows = []
        for segment in ["validation", "test"]:
            for variant, strategy_return, entry_day_return in [
                ("baseline_next_open", 0.10, -0.04),
                ("risk_reduce_half_5m", 0.09, -0.02),
                ("risk_reduce_eighth_5m", 0.092, -0.025),
            ]:
                if variant != "baseline_next_open" and variant not in set(variants or ()):
                    continue
                rows.append(
                    {
                        "datetime": "2026-01-02" if segment == "validation" else "2026-01-03",
                        "segment": segment,
                        "variant": variant,
                        "entry_date": "2026-01-05",
                        "target_position": 1.0,
                        "strategy_return": strategy_return,
                        "entry_day_strategy_return": entry_day_return,
                        "skipped_reason": None,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(risk_sleeve_workflow, "_validation_policy_grid", fake_policy_grid)
    monkeypatch.setattr(risk_sleeve_workflow, "simulate_5min_risk_sleeve", fake_simulate)

    candidates = evaluate_5min_risk_sleeve_candidates(
        pd.DataFrame(
            [
                {"datetime": "2026-01-02", "segment": "validation", "target_position": 1.0},
                {"datetime": "2026-01-03", "segment": "test", "target_position": 1.0},
            ]
        ),
        daily_prices=pd.DataFrame(
            [
                {"datetime": "2026-01-02", "open": 100.0},
                {"datetime": "2026-01-05", "open": 100.0},
                {"datetime": "2026-01-06", "open": 101.0},
            ]
        ),
        bars_5min=pd.DataFrame(
            [
                {
                    "datetime": "2026-01-05 09:35:00",
                    "date": "2026-01-05",
                    "open": 100.0,
                    "close": 99.0,
                    "risk_score": 0.8,
                    "prediction_available_gate": True,
                }
            ]
        ),
        intraday_scores=pd.DataFrame(
            [
                {
                    "datetime": "2026-01-05 09:35:00",
                    "date": "2026-01-05",
                    "risk_score": 0.8,
                    "prediction_available_gate": True,
                }
            ]
        ),
        cost_bps_values=(20.0,),
    )

    assert calls == [("risk_reduce_half_5m", "risk_reduce_eighth_5m")]
    assert set(candidates["variant"]) == {"risk_reduce_half_5m", "risk_reduce_eighth_5m"}


def test_simulate_5min_risk_sleeve_triggers_on_5min_context_stress():
    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "walk_forward_fold": "wf_2026",
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
    bars_5min = pd.DataFrame(
        [
            {
                "datetime": "2026-01-05 09:35:00",
                "date": "2026-01-05",
                "open": 100.0,
                "close": 99.6,
                "soxl_vwap_dist": -0.002,
                "soxx_cum_ret_from_open": -0.001,
                "soxl_rv_6": 0.004,
                "soxl_minus_3x_soxx_ret": 0.001,
            },
            {
                "datetime": "2026-01-05 09:40:00",
                "date": "2026-01-05",
                "open": 99.5,
                "close": 98.8,
                "soxl_vwap_dist": -0.014,
                "soxx_cum_ret_from_open": -0.008,
                "soxl_rv_6": 0.016,
                "soxl_minus_3x_soxx_ret": -0.006,
            },
            {
                "datetime": "2026-01-05 09:45:00",
                "date": "2026-01-05",
                "open": 98.6,
                "close": 98.7,
                "soxl_vwap_dist": -0.011,
                "soxx_cum_ret_from_open": -0.006,
                "soxl_rv_6": 0.014,
                "soxl_minus_3x_soxx_ret": -0.004,
            },
        ]
    )

    rows = simulate_5min_risk_sleeve(
        daily,
        daily_prices=daily_prices,
        bars_5min=bars_5min,
        drawdown_threshold=-0.05,
        trigger_mode="context_stress_or_drawdown",
        context_thresholds={
            "soxl_vwap_dist": -0.01,
            "soxx_cum_ret_from_open": -0.005,
            "soxl_rv_6": 0.012,
            "soxl_minus_3x_soxx_ret": -0.005,
        },
        execution_delay_bars=1,
        execution_price="open",
        cost_bps=0.0,
    )

    by_variant = rows.set_index("variant")
    assert by_variant.loc["risk_exit_full_5m", "trigger_time"] == "2026-01-05 09:40:00"
    assert by_variant.loc["risk_exit_full_5m", "exit_time"] == "2026-01-05 09:45:00"
    assert by_variant.loc["risk_exit_full_5m", "trigger_drawdown_from_entry"] == pytest.approx(-0.012)
    assert by_variant.loc["risk_exit_full_5m", "risk_execution_price"] == pytest.approx(98.6)
    assert by_variant.loc["risk_exit_full_5m", "strategy_return"] == pytest.approx(-0.014)
    assert by_variant.loc["risk_exit_full_5m", "context_trigger_source"] == "manual"
    assert by_variant.loc["risk_exit_full_5m", "context_soxl_vwap_dist"] == pytest.approx(-0.014)
    assert bool(by_variant.loc["risk_exit_full_5m", "risk_triggered"]) is True


def test_5min_risk_sleeve_uses_observable_external_15min_context_without_future_rows():
    bars_5min = pd.DataFrame(
        [
            {
                "datetime": "2026-01-05 09:35:00",
                "date": "2026-01-05",
                "open": 100.0,
                "close": 99.6,
                "soxl_vwap_dist": -0.012,
            },
            {
                "datetime": "2026-01-05 09:40:00",
                "date": "2026-01-05",
                "open": 99.5,
                "close": 99.1,
                "soxl_vwap_dist": -0.014,
            },
            {
                "datetime": "2026-01-05 09:45:00",
                "date": "2026-01-05",
                "open": 99.0,
                "close": 98.8,
                "soxl_vwap_dist": -0.016,
            },
        ]
    )
    external_15m = pd.DataFrame(
        [
            {
                "datetime": "2026-01-05 09:45:00",
                "external_context_available": True,
                "qqq_smh_soxx_lead_lag_3": -0.020,
                "semis_breadth_mom3_pos": 0.20,
                "semis_vwap_breadth": 0.25,
            }
        ]
    )

    merged, audit = _attach_external_context_to_5min_bars(bars_5min, external_15m)

    assert audit["external_15m_rows"] == 1
    assert audit["external_context_available_5min_rows"] == 1
    assert bool(merged.loc[merged["datetime"].eq(pd.Timestamp("2026-01-05 09:40:00")), "external_context_available"].iloc[0]) is False
    assert bool(merged.loc[merged["datetime"].eq(pd.Timestamp("2026-01-05 09:45:00")), "external_context_available"].iloc[0]) is True

    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "walk_forward_fold": "wf_2026",
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

    rows = simulate_5min_risk_sleeve(
        daily,
        daily_prices=daily_prices,
        bars_5min=merged,
        drawdown_threshold=-0.05,
        trigger_mode="context_stress_or_drawdown",
        context_thresholds={
            "soxl_vwap_dist": -0.01,
            "qqq_smh_soxx_lead_lag_3": -0.01,
            "semis_breadth_mom3_pos": 0.30,
        },
        execution_delay_bars=0,
        execution_price="close",
        cost_bps=0.0,
    )

    by_variant = rows.set_index("variant")
    assert by_variant.loc["risk_exit_full_5m", "trigger_time"] == "2026-01-05 09:45:00"
    assert by_variant.loc["risk_exit_full_5m", "context_qqq_smh_soxx_lead_lag_3"] == pytest.approx(-0.020)
    assert by_variant.loc["risk_exit_full_5m", "context_semis_breadth_mom3_pos"] == pytest.approx(0.20)


def test_5min_risk_sleeve_triggers_on_local_gap_reclaim_and_absorption_stress():
    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "walk_forward_fold": "wf_2026",
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
    bars_5min = pd.DataFrame(
        [
            {
                "datetime": "2026-01-05 09:35:00",
                "date": "2026-01-05",
                "open": 100.0,
                "close": 99.8,
                "soxl_vwap_dist": -0.003,
                "open_gap_reclaim_quality": 0.15,
                "soxl_volume_price_absorption": 0.2,
                "leveraged_replication_absorption": 0.1,
            },
            {
                "datetime": "2026-01-05 09:40:00",
                "date": "2026-01-05",
                "open": 99.7,
                "close": 99.0,
                "soxl_vwap_dist": -0.016,
                "open_gap_reclaim_quality": -0.45,
                "soxl_volume_price_absorption": 2.1,
                "leveraged_replication_absorption": 1.4,
            },
            {
                "datetime": "2026-01-05 09:45:00",
                "date": "2026-01-05",
                "open": 98.8,
                "close": 98.9,
                "soxl_vwap_dist": -0.012,
                "open_gap_reclaim_quality": -0.30,
                "soxl_volume_price_absorption": 1.8,
                "leveraged_replication_absorption": 1.2,
            },
        ]
    )

    rows = simulate_5min_risk_sleeve(
        daily,
        daily_prices=daily_prices,
        bars_5min=bars_5min,
        drawdown_threshold=-0.05,
        trigger_mode="context_stress_or_drawdown",
        context_thresholds={
            "soxl_vwap_dist": -0.01,
            "open_gap_reclaim_quality": -0.20,
            "soxl_volume_price_absorption": 1.0,
            "leveraged_replication_absorption": 1.0,
        },
        execution_delay_bars=1,
        execution_price="open",
        cost_bps=0.0,
    )

    by_variant = rows.set_index("variant")
    assert by_variant.loc["risk_exit_full_5m", "trigger_time"] == "2026-01-05 09:40:00"
    assert by_variant.loc["risk_exit_full_5m", "risk_execution_price"] == pytest.approx(98.8)
    assert by_variant.loc["risk_exit_full_5m", "context_open_gap_reclaim_quality"] == pytest.approx(-0.45)
    assert by_variant.loc["risk_exit_full_5m", "context_soxl_volume_price_absorption"] == pytest.approx(2.1)
    assert bool(by_variant.loc["risk_exit_full_5m", "context_stress_hit"]) is True


def test_validation_policy_grid_searches_pure_and_confirmed_intraday_score_triggers():
    policies = risk_sleeve_workflow._validation_policy_grid(
        pd.DataFrame(
            [
                {
                    "datetime": "2026-01-02",
                    "segment": "validation",
                    "target_position": 1.0,
                }
            ]
        ),
        daily_prices=pd.DataFrame(
            [
                {"datetime": "2026-01-02", "open": 100.0},
                {"datetime": "2026-01-05", "open": 100.0},
            ]
        ),
        bars_5min=pd.DataFrame(
            [
                {
                    "datetime": "2026-01-05 09:35:00",
                    "date": "2026-01-05",
                    "open": 100.0,
                    "close": 99.5,
                    "risk_score": 0.25,
                    "prediction_available_gate": True,
                },
                {
                    "datetime": "2026-01-05 09:40:00",
                    "date": "2026-01-05",
                    "open": 99.5,
                    "close": 99.0,
                    "risk_score": 0.35,
                    "prediction_available_gate": True,
                },
            ]
        ),
        risk_score_col="risk_score",
        gate_col="prediction_available_gate",
    )

    trigger_modes = {policy["trigger_mode"] for policy in policies}
    assert {
        "drawdown_only",
        "risk_score_only",
        "risk_score_or_drawdown",
        "risk_score_and_drawdown",
        "context_stress_or_drawdown",
    }.issubset(trigger_modes)


def test_build_intraday_optimization_plan_prefers_local_fallback_when_external_history_blocked():
    plan = build_intraday_optimization_plan(
        selected_model={
            "variant": "delayed_signal_after_30m",
            "passes_validation_and_test_target": True,
            "test_return_retention_pct": 86.1,
            "test_max_drawdown_pct": -11.0,
        },
        walk_forward_summary={
            "required_policy_satisfied": False,
            "blocking_reasons": [
                "annual_walk_forward_not_run",
                "insufficient_external_intraday_history",
                "source_intraday_walk_forward_not_satisfied",
            ],
            "missing_required_test_years": [2023, 2024, 2025, 2026],
        },
        conditional_walk_forward_summary={
            "required_policy_satisfied": False,
            "target_failure_years": [2023, 2024, 2025],
        },
        source_intraday_model={"external_context": {"provider_uri": "/tmp/external"}},
        local_intraday_manifest={
            "status": "ok",
            "freq": "15min",
            "calendar_start": "2018-12-31 09:45:00",
            "calendar_end": "2026-07-02 16:00:00",
            "instruments": ["SOXL", "SOXX"],
        },
        local_intraday_5min_manifest={
            "status": "ok",
            "freq": "5min",
            "calendar_start": "2018-12-31 09:35:00",
            "calendar_end": "2026-07-02 16:00:00",
            "instruments": ["SOXL", "SOXX"],
        },
        external_context_summary={
            "coverage_gap_analysis": {
                "full_walk_forward_history_available": False,
                "earliest_complete_symbol_regular_session": "2025-01-02 09:45:00",
                "missing_required_history_years": [2019, 2020, 2021, 2022, 2023, 2024],
            }
        },
    )

    assert plan["status"] == "needs_local_intraday_fallback"
    assert plan["primary_next_action"] == "run_soxl_soxx_intraday_only_annual_walk_forward"
    assert plan["external_context_status"] == "blocked_history_insufficient"
    assert plan["local_intraday_history"]["supports_required_start"] is True
    assert plan["local_intraday_history"]["supports_5min_probe"] is True
    assert plan["ranked_actions"][0]["action"] == "run_soxl_soxx_intraday_only_annual_walk_forward"
    assert "do_not_continue_external_threshold_search" in plan["guardrails"]


def test_build_intraday_optimization_plan_does_not_apply_external_gap_to_local_source():
    plan = build_intraday_optimization_plan(
        selected_model={
            "variant": "baseline_next_open",
            "test_return_retention_pct": 100.0,
            "test_max_drawdown_pct": -18.4,
        },
        walk_forward_summary={
            "required_policy_satisfied": False,
            "blocking_reasons": [],
            "missing_required_test_years": [],
        },
        conditional_walk_forward_summary={"required_policy_satisfied": False},
        source_intraday_model={"external_context": {}},
        local_intraday_manifest={
            "status": "ok",
            "freq": "15min",
            "calendar_start": "2018-12-31 09:45:00",
            "calendar_end": "2026-07-02 16:00:00",
            "instruments": ["SOXL", "SOXX"],
        },
        external_context_summary={
            "coverage_gap_analysis": {
                "full_walk_forward_history_available": False,
                "missing_required_history_years": [2019, 2020, 2021, 2022, 2023, 2024],
            }
        },
    )

    assert plan["external_context_status"] == "not_used"
    assert plan["status"] == "needs_new_intraday_factor_or_label"
    assert plan["primary_next_action"] == "replace_open_gate_with_delayed_observable_label"
    assert "do_not_continue_external_threshold_search" not in plan["guardrails"]


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
        "overlay_b_twap3",
        "overlay_b_twap4",
        "overlay_c_risk_exit",
    }
    by_variant = rows.set_index("variant")
    assert by_variant.loc["baseline_next_open", "strategy_return"] == 0.10
    assert by_variant.loc["overlay_a_signal_gate", "entry_time"] == "2026-01-05 10:15:00"
    assert by_variant.loc["overlay_a_signal_gate", "strategy_return"] == (110.0 / 104.0 - 1.0)
    assert by_variant.loc["overlay_b_twap3", "entry_time"] == "twap3_first_3_bars"
    assert by_variant.loc["overlay_b_twap3", "strategy_return"] == pytest.approx(110.0 / ((101.0 + 102.0 + 104.0) / 3.0) - 1.0)
    assert by_variant.loc["overlay_b_twap4", "entry_time"] == "twap4_first_4_bars"
    assert by_variant.loc["overlay_c_risk_exit", "exit_time"] == "2026-01-05 09:45:00"
    assert by_variant.loc["overlay_c_risk_exit", "strategy_return"] == (102.0 / 100.0 - 1.0)


def test_simulate_execution_overlays_marks_missing_prediction_coverage():
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
                "prob_lgb": None,
                "prediction_available_gate": False,
            },
            {
                "datetime": "2026-01-05 10:00:00",
                "date": "2026-01-05",
                "open": 102.0,
                "close": 103.0,
                "prob_lgb": None,
                "prediction_available_gate": False,
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
        prob_col="prob_lgb",
        gate_col="prediction_available_gate",
    )

    by_variant = rows.set_index("variant")
    assert by_variant.loc["overlay_a_signal_gate", "skipped_reason"] == "missing_intraday_prediction"
    assert by_variant.loc["overlay_c_risk_exit", "skipped_reason"] == "missing_intraday_prediction"
    assert by_variant.loc["overlay_c_risk_exit", "strategy_return"] == pytest.approx(0.10)


def test_simulate_execution_overlays_keeps_no_signal_separate_from_missing_predictions():
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
                "prob_lgb": 0.40,
                "prediction_available_gate": True,
            },
            {
                "datetime": "2026-01-05 10:00:00",
                "date": "2026-01-05",
                "open": 102.0,
                "close": 103.0,
                "prob_lgb": 0.42,
                "prediction_available_gate": True,
            },
        ]
    )

    rows = simulate_execution_overlays(
        daily,
        daily_prices=daily_prices,
        intraday=intraday,
        signal_threshold=0.55,
        risk_threshold=0.35,
        cost_bps=0.0,
        prob_col="prob_lgb",
        gate_col="prediction_available_gate",
    )

    by_variant = rows.set_index("variant")
    assert by_variant.loc["overlay_a_signal_gate", "skipped_reason"] == "no_15m_entry_signal"
    assert pd.isna(by_variant.loc["overlay_c_risk_exit", "skipped_reason"])


def test_summarize_execution_risk_diagnostics_quantifies_open_buy_dump_reduction():
    daily = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "test",
                "target_position": 1.0,
                "strategy_return": 0.05,
            }
        ]
    )
    daily_prices = pd.DataFrame(
        [
            {"datetime": "2026-01-02", "open": 90.0},
            {"datetime": "2026-01-05", "open": 100.0},
            {"datetime": "2026-01-06", "open": 105.0},
        ]
    )
    intraday = pd.DataFrame(
        [
            {
                "datetime": "2026-01-05 09:45:00",
                "date": "2026-01-05",
                "open": 100.0,
                "close": 95.0,
                "prob_lgb_xgb_consensus": 0.65,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2026-01-05 10:00:00",
                "date": "2026-01-05",
                "open": 96.0,
                "close": 96.0,
                "prob_lgb_xgb_consensus": 0.62,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2026-01-05 10:15:00",
                "date": "2026-01-05",
                "open": 96.0,
                "close": 96.0,
                "prob_lgb_xgb_consensus": 0.61,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2026-01-05 10:30:00",
                "date": "2026-01-05",
                "open": 96.0,
                "close": 96.0,
                "prob_lgb_xgb_consensus": 0.60,
                "regime_gate_v2_dislocation": True,
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
    ).assign(cost_bps=0.0)
    diagnostics = summarize_execution_risk_diagnostics(rows, open_buy_dump_threshold=-0.03)

    by_variant = diagnostics.set_index("variant")
    assert by_variant.loc["baseline_next_open", "open_buy_dump_periods"] == 1
    assert by_variant.loc["overlay_a_signal_gate", "open_buy_dump_periods"] == 0
    assert by_variant.loc["overlay_a_signal_gate", "open_buy_dump_period_reduction"] == 1
    assert by_variant.loc["baseline_next_open", "worst_entry_day_return_pct"] == pytest.approx(-4.0)
    assert by_variant.loc["overlay_a_signal_gate", "worst_entry_day_return_pct"] == pytest.approx(0.0)


def test_conditional_overlay_target_treats_nonpositive_baseline_as_improvement_gate():
    trades = pd.DataFrame(
        [
            {
                "datetime": "2022-01-03",
                "segment": "validation",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": -0.10,
                "entry_day_strategy_return": -0.04,
                "skipped_reason": None,
            },
            {
                "datetime": "2022-01-04",
                "segment": "validation",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": -0.05,
                "entry_day_strategy_return": -0.03,
                "skipped_reason": None,
            },
            {
                "datetime": "2023-01-03",
                "segment": "test",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": -0.08,
                "entry_day_strategy_return": -0.05,
                "skipped_reason": None,
            },
            {
                "datetime": "2023-01-04",
                "segment": "test",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": -0.02,
                "entry_day_strategy_return": -0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2022-01-03",
                "segment": "validation",
                "variant": "overlay_b_twap3",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": -0.10,
                "entry_day_strategy_return": -0.04,
                "skipped_reason": None,
            },
            {
                "datetime": "2022-01-04",
                "segment": "validation",
                "variant": "overlay_b_twap3",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": 0.20,
                "entry_day_strategy_return": -0.005,
                "skipped_reason": None,
            },
            {
                "datetime": "2023-01-03",
                "segment": "test",
                "variant": "overlay_b_twap3",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": -0.08,
                "entry_day_strategy_return": -0.05,
                "skipped_reason": None,
            },
            {
                "datetime": "2023-01-04",
                "segment": "test",
                "variant": "overlay_b_twap3",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": 0.12,
                "entry_day_strategy_return": -0.002,
                "skipped_reason": None,
            },
        ]
    )
    regime_features = pd.DataFrame(
        [
            {"datetime": "2022-01-03", "probability": 0.10, "target_position": 1.0},
            {"datetime": "2022-01-04", "probability": 0.20, "target_position": 1.0},
            {"datetime": "2023-01-03", "probability": 0.10, "target_position": 1.0},
            {"datetime": "2023-01-04", "probability": 0.20, "target_position": 1.0},
        ]
    )

    candidates = evaluate_conditional_overlay_candidates(
        trades,
        regime_features,
        cost_bps=20.0,
        variants=("overlay_b_twap3",),
        condition_columns=("probability",),
        quantiles=(0.50,),
        min_validation_coverage=0.01,
    )

    selected = candidates.loc[candidates["passes_validation_target"].astype(bool)].iloc[0]
    assert selected["validation_return_target_mode"] == "baseline_nonpositive_improvement"
    assert selected["test_return_target_mode"] == "baseline_nonpositive_improvement"
    assert selected["validation_return_retention_pct"] < 0.0
    assert selected["test_return_retention_pct"] < 0.0
    assert selected["validation_return_pct"] > selected["validation_baseline_return_pct"]
    assert selected["test_return_pct"] > selected["test_baseline_return_pct"]


def test_conditional_overlay_return_upper_bound_is_score_cap_not_hard_failure():
    trades = pd.DataFrame(
        [
            {
                "datetime": "2024-01-03",
                "segment": "validation",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": 0.05,
                "entry_day_strategy_return": 0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2024-01-04",
                "segment": "validation",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": 0.05,
                "entry_day_strategy_return": 0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2025-01-03",
                "segment": "test",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": 0.04,
                "entry_day_strategy_return": 0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2025-01-04",
                "segment": "test",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": 0.04,
                "entry_day_strategy_return": 0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2024-01-03",
                "segment": "validation",
                "variant": "overlay_b_twap3",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": 0.20,
                "entry_day_strategy_return": 0.02,
                "skipped_reason": None,
            },
            {
                "datetime": "2024-01-04",
                "segment": "validation",
                "variant": "overlay_b_twap3",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": 0.20,
                "entry_day_strategy_return": 0.02,
                "skipped_reason": None,
            },
            {
                "datetime": "2025-01-03",
                "segment": "test",
                "variant": "overlay_b_twap3",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": 0.12,
                "entry_day_strategy_return": 0.02,
                "skipped_reason": None,
            },
            {
                "datetime": "2025-01-04",
                "segment": "test",
                "variant": "overlay_b_twap3",
                "cost_bps": 20.0,
                "target_position": 1.0,
                "strategy_return": 0.12,
                "entry_day_strategy_return": 0.02,
                "skipped_reason": None,
            },
        ]
    )
    regime_features = pd.DataFrame(
        [
            {"datetime": "2024-01-03", "probability": 0.10, "target_position": 1.0},
            {"datetime": "2024-01-04", "probability": 0.20, "target_position": 1.0},
            {"datetime": "2025-01-03", "probability": 0.10, "target_position": 1.0},
            {"datetime": "2025-01-04", "probability": 0.20, "target_position": 1.0},
        ]
    )

    candidates = evaluate_conditional_overlay_candidates(
        trades,
        regime_features,
        cost_bps=20.0,
        variants=("overlay_b_twap3",),
        condition_columns=("probability",),
        quantiles=(0.50,),
        min_validation_coverage=0.01,
    )

    selected = candidates.loc[candidates["passes_validation_target"].astype(bool)].iloc[0]
    assert selected["validation_return_target_mode"] == "positive_retention_band"
    assert selected["validation_return_retention_pct"] > 95.0
    assert selected["validation_return_retention_score"] == pytest.approx(95.0)
    assert bool(selected["validation_return_target_satisfied"]) is True


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


def test_evaluate_conditional_overlay_candidates_reports_open_buy_dump_reduction():
    trades = pd.DataFrame(
        [
            {
                "datetime": "2026-01-02",
                "segment": "validation",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "strategy_return": 0.02,
                "entry_day_strategy_return": 0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-03",
                "segment": "validation",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "strategy_return": -0.09,
                "entry_day_strategy_return": -0.05,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-04",
                "segment": "test",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "strategy_return": -0.10,
                "entry_day_strategy_return": -0.06,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-05",
                "segment": "test",
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "strategy_return": -0.02,
                "entry_day_strategy_return": -0.04,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-02",
                "segment": "validation",
                "variant": "overlay_b_twap4",
                "cost_bps": 20.0,
                "strategy_return": 0.02,
                "entry_day_strategy_return": 0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-03",
                "segment": "validation",
                "variant": "overlay_b_twap4",
                "cost_bps": 20.0,
                "strategy_return": -0.01,
                "entry_day_strategy_return": -0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-04",
                "segment": "test",
                "variant": "overlay_b_twap4",
                "cost_bps": 20.0,
                "strategy_return": -0.02,
                "entry_day_strategy_return": -0.01,
                "skipped_reason": None,
            },
            {
                "datetime": "2026-01-05",
                "segment": "test",
                "variant": "overlay_b_twap4",
                "cost_bps": 20.0,
                "strategy_return": 0.01,
                "entry_day_strategy_return": 0.00,
                "skipped_reason": None,
            },
        ]
    )
    regime_features = pd.DataFrame(
        [
            {"datetime": "2026-01-02", "score": 0.20},
            {"datetime": "2026-01-03", "score": 0.80},
            {"datetime": "2026-01-04", "score": 0.90},
            {"datetime": "2026-01-05", "score": 0.95},
        ]
    )

    candidates = evaluate_conditional_overlay_candidates(
        trades,
        regime_features,
        cost_bps=20.0,
        variants=("overlay_b_twap4",),
        condition_columns=("score",),
        quantiles=(0.50,),
        min_validation_coverage=0.0,
    )

    selected = candidates.loc[
        candidates["variant"].eq("overlay_b_twap4")
        & candidates["condition_operator"].eq("ge")
    ].iloc[0]
    assert selected["condition_threshold"] == 0.50
    assert bool(selected["selection_uses_test_segment"]) is False
    assert selected["validation_open_buy_dump_period_reduction"] == 1
    assert selected["validation_crash_period_reduction"] == 1
    assert selected["test_open_buy_dump_period_reduction"] == 2
    assert selected["test_crash_period_reduction"] == 1
    assert selected["validation_worst_entry_day_return_improvement_pct"] == pytest.approx(4.0)


def test_select_conditional_risk_sleeve_policy_prefers_validation_risk_reduction():
    candidates = pd.DataFrame(
        [
            {
                "cost_bps": 20.0,
                "variant": "overlay_b_twap4",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.74,
                "selection_score": 98.0,
                "validation_return_retention_pct": 90.0,
                "validation_max_drawdown_pct": -14.0,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 2,
                "validation_crash_period_reduction": 1,
                "validation_worst_entry_day_return_improvement_pct": 4.0,
                "validation_drawdown_improvement_pct": 4.0,
                "test_return_retention_pct": 88.0,
                "test_max_drawdown_pct": -16.0,
                "test_open_buy_dump_period_reduction": 1,
                "test_crash_period_reduction": 1,
                "passes_validation_target": True,
                "passes_test_target": True,
            },
            {
                "cost_bps": 50.0,
                "variant": "overlay_b_twap4",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.74,
                "selection_score": 95.0,
                "validation_return_retention_pct": 87.0,
                "validation_max_drawdown_pct": -17.0,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 2,
                "validation_crash_period_reduction": 1,
                "validation_worst_entry_day_return_improvement_pct": 3.5,
                "validation_drawdown_improvement_pct": 2.0,
                "test_return_retention_pct": 86.0,
                "test_max_drawdown_pct": -17.0,
                "test_open_buy_dump_period_reduction": 1,
                "test_crash_period_reduction": 0,
                "passes_validation_target": True,
                "passes_test_target": True,
            },
            {
                "cost_bps": 20.0,
                "variant": "overlay_a_signal_gate",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.90,
                "condition_threshold": 0.72,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.90,
                "selection_score": 120.0,
                "validation_return_retention_pct": 91.0,
                "validation_max_drawdown_pct": -13.0,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 0,
                "validation_crash_period_reduction": 0,
                "validation_worst_entry_day_return_improvement_pct": 1.0,
                "validation_drawdown_improvement_pct": 5.0,
                "test_return_retention_pct": 80.0,
                "test_max_drawdown_pct": -15.0,
                "test_open_buy_dump_period_reduction": 4,
                "test_crash_period_reduction": 2,
                "passes_validation_target": True,
                "passes_test_target": False,
            },
            {
                "cost_bps": 50.0,
                "variant": "overlay_a_signal_gate",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.90,
                "condition_threshold": 0.72,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.90,
                "selection_score": 115.0,
                "validation_return_retention_pct": 90.0,
                "validation_max_drawdown_pct": -13.5,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 0,
                "validation_crash_period_reduction": 0,
                "validation_worst_entry_day_return_improvement_pct": 1.0,
                "validation_drawdown_improvement_pct": 5.0,
                "test_return_retention_pct": 79.0,
                "test_max_drawdown_pct": -15.0,
                "test_open_buy_dump_period_reduction": 4,
                "test_crash_period_reduction": 2,
                "passes_validation_target": True,
                "passes_test_target": False,
            },
        ]
    )

    selected = select_conditional_risk_sleeve_policy(candidates)

    assert selected["variant"] == "overlay_b_twap4"
    assert selected["selection_role"] == "conditional_risk_sleeve"
    assert selected["selection_uses_test_segment"] is False
    assert selected["selection_priority"][0] == "validation_min_required_cost_return_retention_pct_desc"
    assert selected["required_cost_validation_target_count"] == 2
    assert selected["test_open_buy_dump_period_reduction"] == 1


def test_select_conditional_risk_sleeve_policy_prefers_return_band_before_soft_cap_score():
    candidates = pd.DataFrame(
        [
            {
                "cost_bps": cost_bps,
                "variant": "overlay_b_twap3",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.75,
                "selection_score": 100.0 - cost_bps / 100.0,
                "validation_return_retention_pct": 92.0,
                "validation_return_retention_score": 92.0,
                "validation_return_retention_in_target_band": True,
                "validation_max_drawdown_pct": -14.5,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 2,
                "validation_crash_period_reduction": 1,
                "validation_worst_entry_day_return_improvement_pct": 2.0,
                "validation_drawdown_improvement_pct": 3.5,
                "test_return_retention_pct": 88.0,
                "test_max_drawdown_pct": -15.5,
                "test_open_buy_dump_period_reduction": 2,
                "test_crash_period_reduction": 1,
                "passes_validation_target": True,
                "passes_test_target": True,
            }
            for cost_bps in (20.0, 50.0)
        ]
        + [
            {
                "cost_bps": cost_bps,
                "variant": "overlay_b_twap4",
                "condition_column": "probability",
                "condition_operator": "ge",
                "condition_quantile": 0.55,
                "condition_threshold": 0.54,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.70,
                "selection_score": 125.0 - cost_bps / 100.0,
                "validation_return_retention_pct": 130.0,
                "validation_return_retention_score": 95.0,
                "validation_return_retention_in_target_band": False,
                "validation_max_drawdown_pct": -13.0,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 3,
                "validation_crash_period_reduction": 1,
                "validation_worst_entry_day_return_improvement_pct": 3.0,
                "validation_drawdown_improvement_pct": 4.0,
                "test_return_retention_pct": 120.0,
                "test_max_drawdown_pct": -14.0,
                "test_open_buy_dump_period_reduction": 3,
                "test_crash_period_reduction": 1,
                "passes_validation_target": True,
                "passes_test_target": True,
            }
            for cost_bps in (20.0, 50.0)
        ]
    )

    selected = select_conditional_risk_sleeve_policy(candidates)

    assert selected["variant"] == "overlay_b_twap3"
    assert selected["validation_return_retention_in_target_band"] is True


def test_select_conditional_risk_sleeve_policy_prefers_cost_buffered_twap_execution():
    candidates = pd.DataFrame(
        [
            {
                "cost_bps": 20.0,
                "variant": "overlay_b_twap4",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.75,
                "selection_score": 110.0,
                "validation_return_retention_pct": 90.0,
                "validation_max_drawdown_pct": -14.0,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 4,
                "validation_crash_period_reduction": 0,
                "validation_worst_entry_day_return_improvement_pct": 4.0,
                "validation_drawdown_improvement_pct": 4.0,
                "test_return_retention_pct": 85.4,
                "test_max_drawdown_pct": -16.0,
                "test_open_buy_dump_period_reduction": 3,
                "test_crash_period_reduction": 1,
                "passes_validation_target": True,
                "passes_test_target": True,
            },
            {
                "cost_bps": 50.0,
                "variant": "overlay_b_twap4",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.75,
                "selection_score": 96.0,
                "validation_return_retention_pct": 86.0,
                "validation_max_drawdown_pct": -17.7,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 4,
                "validation_crash_period_reduction": 0,
                "validation_worst_entry_day_return_improvement_pct": 4.0,
                "validation_drawdown_improvement_pct": 0.3,
                "test_return_retention_pct": 84.6,
                "test_max_drawdown_pct": -17.3,
                "test_open_buy_dump_period_reduction": 3,
                "test_crash_period_reduction": 1,
                "passes_validation_target": True,
                "passes_test_target": False,
            },
            {
                "cost_bps": 20.0,
                "variant": "overlay_b_twap3",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.75,
                "selection_score": 108.0,
                "validation_return_retention_pct": 92.8,
                "validation_max_drawdown_pct": -14.6,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 2,
                "validation_crash_period_reduction": 0,
                "validation_worst_entry_day_return_improvement_pct": 2.0,
                "validation_drawdown_improvement_pct": 3.7,
                "test_return_retention_pct": 88.4,
                "test_max_drawdown_pct": -15.4,
                "test_open_buy_dump_period_reduction": 2,
                "test_crash_period_reduction": 1,
                "passes_validation_target": True,
                "passes_test_target": True,
            },
            {
                "cost_bps": 50.0,
                "variant": "overlay_b_twap3",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.75,
                "selection_score": 101.0,
                "validation_return_retention_pct": 91.0,
                "validation_max_drawdown_pct": -17.3,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 2,
                "validation_crash_period_reduction": 0,
                "validation_worst_entry_day_return_improvement_pct": 2.0,
                "validation_drawdown_improvement_pct": 0.7,
                "test_return_retention_pct": 87.8,
                "test_max_drawdown_pct": -16.2,
                "test_open_buy_dump_period_reduction": 2,
                "test_crash_period_reduction": 1,
                "passes_validation_target": True,
                "passes_test_target": True,
            },
            {
                "cost_bps": 20.0,
                "variant": "overlay_a_signal_gate",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.75,
                "selection_score": 115.0,
                "validation_return_retention_pct": 93.0,
                "validation_max_drawdown_pct": -14.5,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 2,
                "validation_crash_period_reduction": 0,
                "validation_worst_entry_day_return_improvement_pct": 2.0,
                "validation_drawdown_improvement_pct": 3.5,
                "test_return_retention_pct": 76.0,
                "test_max_drawdown_pct": -15.0,
                "test_open_buy_dump_period_reduction": 4,
                "test_crash_period_reduction": 2,
                "passes_validation_target": True,
                "passes_test_target": False,
            },
            {
                "cost_bps": 50.0,
                "variant": "overlay_a_signal_gate",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "condition_threshold_source": "validation_quantile",
                "condition_valid_active_coverage": 0.75,
                "selection_score": 111.0,
                "validation_return_retention_pct": 92.0,
                "validation_max_drawdown_pct": -17.8,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
                "validation_open_buy_dump_period_reduction": 2,
                "validation_crash_period_reduction": 0,
                "validation_worst_entry_day_return_improvement_pct": 2.0,
                "validation_drawdown_improvement_pct": 0.2,
                "test_return_retention_pct": 75.0,
                "test_max_drawdown_pct": -15.8,
                "test_open_buy_dump_period_reduction": 4,
                "test_crash_period_reduction": 2,
                "passes_validation_target": True,
                "passes_test_target": False,
            },
        ]
    )

    selected = select_conditional_risk_sleeve_policy(candidates)

    assert selected["variant"] == "overlay_b_twap3"
    assert selected["selection_uses_test_segment"] is False
    assert selected["eligible_variants"] == ["overlay_b_twap3", "overlay_b_twap4"]
    assert selected["validation_min_required_cost_return_retention_pct"] == 91.0
    assert selected["required_cost_test_target_count"] == 2


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


def test_simulate_observable_delayed_signal_overlay_can_flatten_at_same_day_close():
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
                "datetime": "2026-01-05 15:45:00",
                "date": "2026-01-05",
                "open": 108.0,
                "close": 106.0,
                "prob_lgb_xgb_consensus": 0.55,
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
        exit_policy="same_day_close",
        cost_bps=0.0,
    )

    assert set(rows["variant"]) == {"baseline_next_open", "delayed_signal_after_30m_same_day_close"}
    by_variant = rows.set_index("variant")
    assert by_variant.loc["delayed_signal_after_30m_same_day_close", "entry_time"] == "2026-01-05 10:15:00"
    assert by_variant.loc["delayed_signal_after_30m_same_day_close", "exit_time"] == "2026-01-05 15:45:00"
    assert by_variant.loc["delayed_signal_after_30m_same_day_close", "exit_price"] == 106.0
    assert by_variant.loc["delayed_signal_after_30m_same_day_close", "strategy_return"] == (106.0 / 104.0 - 1.0)


def test_evaluate_observable_delayed_signal_candidates_includes_intraday_flat_policy():
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
        exit_policy_grid=("hold_to_next_open", "risk_exit", "same_day_close"),
        quantiles=(0.50,),
    )

    assert {"hold_to_next_open", "risk_exit", "same_day_close"}.issubset(set(candidates["exit_policy"]))
    assert set(candidates["condition_threshold_source"]) == {"validation_quantile"}
    assert set(candidates["selection_uses_test_segment"]) == {False}
    assert set(candidates["return_retention_upper_policy"]) == {"hard_gate"}
    assert "delayed_signal_after_30m_risk_exit" in set(candidates["variant"])
    assert "delayed_signal_after_30m_same_day_close" in set(candidates["variant"])


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


def test_select_validation_robust_delayed_policy_prefers_return_retention_after_risk_gates():
    candidates = pd.DataFrame(
        [
            {
                "policy_id": "low_retention_large_drawdown_repair",
                "selection_score": 110.0,
                "validation_drawdown_improvement_pct": 8.0,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 87.0,
                "validation_max_drawdown_pct": -10.0,
                "validation_skipped_periods": 5,
                "validation_periods": 250,
            },
            {
                "policy_id": "low_retention_large_drawdown_repair",
                "selection_score": 110.0,
                "validation_drawdown_improvement_pct": 8.0,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 86.0,
                "validation_max_drawdown_pct": -11.0,
                "validation_skipped_periods": 5,
                "validation_periods": 250,
            },
            {
                "policy_id": "higher_retention_still_reduces_drawdown",
                "selection_score": 100.0,
                "validation_drawdown_improvement_pct": 3.0,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 94.0,
                "validation_max_drawdown_pct": -14.0,
                "validation_skipped_periods": 5,
                "validation_periods": 250,
            },
            {
                "policy_id": "higher_retention_still_reduces_drawdown",
                "selection_score": 100.0,
                "validation_drawdown_improvement_pct": 3.0,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 93.0,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 5,
                "validation_periods": 250,
            },
        ]
    )

    selected = select_validation_robust_delayed_policy(candidates)

    assert selected["policy_id"] == "higher_retention_still_reduces_drawdown"
    assert selected["validation_passes_target"] is True


def test_select_validation_robust_delayed_policy_rejects_retention_above_hard_band_by_default():
    candidates = pd.DataFrame(
        [
            {
                "policy_id": "capped_upside_with_lower_drawdown",
                "selection_score": 99.0,
                "validation_drawdown_improvement_pct": 2.0,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 98.0,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 1,
                "validation_periods": 250,
            },
            {
                "policy_id": "capped_upside_with_lower_drawdown",
                "selection_score": 99.0,
                "validation_drawdown_improvement_pct": 2.0,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 97.0,
                "validation_max_drawdown_pct": -16.0,
                "validation_skipped_periods": 1,
                "validation_periods": 250,
            },
            {
                "policy_id": "in_band_but_weaker",
                "selection_score": 98.0,
                "validation_drawdown_improvement_pct": 2.0,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 92.0,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 1,
                "validation_periods": 250,
            },
            {
                "policy_id": "in_band_but_weaker",
                "selection_score": 98.0,
                "validation_drawdown_improvement_pct": 2.0,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 91.0,
                "validation_max_drawdown_pct": -16.0,
                "validation_skipped_periods": 1,
                "validation_periods": 250,
            },
        ]
    )

    selected = select_validation_robust_delayed_policy(candidates)

    assert selected["policy_id"] == "in_band_but_weaker"
    assert selected["return_retention_upper_policy"] == "hard_gate"
    assert selected["validation_passes_target"] is True


def test_select_validation_robust_delayed_policy_rejects_upper_boundary_solution():
    candidates = pd.DataFrame(
        [
            {
                "policy_id": "near_upper_boundary_solution",
                "selection_score": 120.0,
                "validation_drawdown_improvement_pct": 5.0,
                "condition_valid_active_coverage": 0.60,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 94.99,
                "validation_max_drawdown_pct": -14.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
            },
            {
                "policy_id": "near_upper_boundary_solution",
                "selection_score": 120.0,
                "validation_drawdown_improvement_pct": 5.0,
                "condition_valid_active_coverage": 0.60,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 94.98,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
            },
            {
                "policy_id": "in_band_with_validation_margin",
                "selection_score": 118.0,
                "validation_drawdown_improvement_pct": 4.0,
                "condition_valid_active_coverage": 0.55,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 94.90,
                "validation_max_drawdown_pct": -14.0,
                "validation_skipped_periods": 3,
                "validation_periods": 250,
            },
            {
                "policy_id": "in_band_with_validation_margin",
                "selection_score": 118.0,
                "validation_drawdown_improvement_pct": 4.0,
                "condition_valid_active_coverage": 0.55,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 94.85,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 3,
                "validation_periods": 250,
            },
        ]
    )

    selected = select_validation_robust_delayed_policy(candidates)

    assert selected["policy_id"] == "in_band_with_validation_margin"
    assert selected["required_cost_min_validation_band_margin"] == pytest.approx(0.10)
    assert selected["min_validation_retention_band_margin"] == pytest.approx(0.05)
    assert selected["validation_passes_target"] is True


def test_select_validation_robust_delayed_policy_prefers_lower_skip_after_retention_cap():
    candidates = pd.DataFrame(
        [
            {
                "policy_id": "extreme_sparse_repair",
                "selection_score": 130.0,
                "validation_drawdown_improvement_pct": 12.0,
                "condition_valid_active_coverage": 0.35,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 180.0,
                "validation_max_drawdown_pct": -8.0,
                "validation_skipped_periods": 30,
                "validation_periods": 250,
            },
            {
                "policy_id": "extreme_sparse_repair",
                "selection_score": 130.0,
                "validation_drawdown_improvement_pct": 12.0,
                "condition_valid_active_coverage": 0.35,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 170.0,
                "validation_max_drawdown_pct": -9.0,
                "validation_skipped_periods": 30,
                "validation_periods": 250,
            },
            {
                "policy_id": "stable_retention_repair",
                "selection_score": 105.0,
                "validation_drawdown_improvement_pct": 4.0,
                "condition_valid_active_coverage": 0.60,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 94.8,
                "validation_max_drawdown_pct": -14.0,
                "validation_skipped_periods": 3,
                "validation_periods": 250,
            },
            {
                "policy_id": "stable_retention_repair",
                "selection_score": 105.0,
                "validation_drawdown_improvement_pct": 4.0,
                "condition_valid_active_coverage": 0.60,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 94.0,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 3,
                "validation_periods": 250,
            },
        ]
    )

    selected = select_validation_robust_delayed_policy(candidates)

    assert selected["policy_id"] == "stable_retention_repair"
    assert selected["validation_skip_rate"] == pytest.approx(3 / 250)
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


def test_select_validation_robust_delayed_policy_allows_same_day_close_as_intraday_flat_candidate():
    candidates = pd.DataFrame(
        [
            {
                "policy_id": "hold_lower_retention",
                "variant": "delayed_signal_after_30m",
                "exit_policy": "hold_to_next_open",
                "selection_score": 100.0,
                "validation_drawdown_improvement_pct": 2.0,
                "condition_valid_active_coverage": 0.50,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 88.0,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
            },
            {
                "policy_id": "hold_lower_retention",
                "variant": "delayed_signal_after_30m",
                "exit_policy": "hold_to_next_open",
                "selection_score": 100.0,
                "validation_drawdown_improvement_pct": 2.0,
                "condition_valid_active_coverage": 0.50,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 87.0,
                "validation_max_drawdown_pct": -16.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
            },
            {
                "policy_id": "same_day_flat_higher_retention",
                "variant": "delayed_signal_after_30m_same_day_close",
                "exit_policy": "same_day_close",
                "selection_score": 105.0,
                "validation_drawdown_improvement_pct": 2.0,
                "condition_valid_active_coverage": 0.50,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 94.0,
                "validation_max_drawdown_pct": -14.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
            },
            {
                "policy_id": "same_day_flat_higher_retention",
                "variant": "delayed_signal_after_30m_same_day_close",
                "exit_policy": "same_day_close",
                "selection_score": 105.0,
                "validation_drawdown_improvement_pct": 2.0,
                "condition_valid_active_coverage": 0.50,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 93.0,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
            },
            {
                "policy_id": "risk_exit_highest_score",
                "variant": "delayed_signal_after_30m_risk_exit",
                "exit_policy": "risk_exit",
                "selection_score": 130.0,
                "validation_drawdown_improvement_pct": 5.0,
                "condition_valid_active_coverage": 0.50,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 95.0,
                "validation_max_drawdown_pct": -8.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
            },
            {
                "policy_id": "risk_exit_highest_score",
                "variant": "delayed_signal_after_30m_risk_exit",
                "exit_policy": "risk_exit",
                "selection_score": 130.0,
                "validation_drawdown_improvement_pct": 5.0,
                "condition_valid_active_coverage": 0.50,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 94.0,
                "validation_max_drawdown_pct": -9.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
            },
        ]
    )

    selected = select_validation_robust_delayed_policy(candidates)

    assert selected["policy_id"] == "same_day_flat_higher_retention"
    assert selected["exit_policy"] == "same_day_close"


def test_select_intraday_flat_sleeve_policy_prefers_validation_drawdown_over_coverage():
    candidates = pd.DataFrame(
        [
            {
                "policy_id": "wide_coverage_weaker_drawdown",
                "variant": "delayed_signal_after_30m_same_day_close",
                "exit_policy": "same_day_close",
                "selection_score": 120.0,
                "condition_valid_active_coverage": 0.80,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 140.0,
                "validation_max_drawdown_pct": -14.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
                "test_return_retention_pct": 110.0,
                "test_max_drawdown_pct": -12.0,
            },
            {
                "policy_id": "wide_coverage_weaker_drawdown",
                "variant": "delayed_signal_after_30m_same_day_close",
                "exit_policy": "same_day_close",
                "selection_score": 120.0,
                "condition_valid_active_coverage": 0.80,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 130.0,
                "validation_max_drawdown_pct": -15.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
                "test_return_retention_pct": 105.0,
                "test_max_drawdown_pct": -13.0,
            },
            {
                "policy_id": "lower_drawdown_flat_sleeve",
                "variant": "delayed_signal_after_30m_same_day_close",
                "exit_policy": "same_day_close",
                "selection_score": 90.0,
                "condition_valid_active_coverage": 0.20,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 94.0,
                "validation_max_drawdown_pct": -8.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
                "test_return_retention_pct": 95.0,
                "test_max_drawdown_pct": -7.0,
            },
            {
                "policy_id": "lower_drawdown_flat_sleeve",
                "variant": "delayed_signal_after_30m_same_day_close",
                "exit_policy": "same_day_close",
                "selection_score": 90.0,
                "condition_valid_active_coverage": 0.20,
                "cost_bps": 50.0,
                "validation_return_retention_pct": 92.0,
                "validation_max_drawdown_pct": -9.0,
                "validation_skipped_periods": 2,
                "validation_periods": 250,
                "test_return_retention_pct": 90.0,
                "test_max_drawdown_pct": -8.0,
            },
            {
                "policy_id": "hold_is_not_intraday_flat",
                "variant": "delayed_signal_after_30m",
                "exit_policy": "hold_to_next_open",
                "selection_score": 200.0,
                "condition_valid_active_coverage": 1.0,
                "cost_bps": 20.0,
                "validation_return_retention_pct": 200.0,
                "validation_max_drawdown_pct": -6.0,
                "validation_skipped_periods": 0,
                "validation_periods": 250,
                "test_return_retention_pct": 200.0,
                "test_max_drawdown_pct": -6.0,
            },
        ]
    )

    selected = select_intraday_flat_sleeve_policy(candidates)

    assert selected["policy_id"] == "lower_drawdown_flat_sleeve"
    assert selected["exit_policy"] == "same_day_close"
    assert selected["selection_role"] == "intraday_flat_sleeve"
    assert selected["selection_priority"][0] == "validation_max_drawdown_desc"
    assert selected["validation_passes_target"] is True
    assert selected["return_retention_upper_policy"] == "hard_gate"


def test_load_intraday_prediction_frame_uses_prediction_history_start(monkeypatch, tmp_path):
    recorder_dir = tmp_path / "source_recorder"
    prediction_dir = recorder_dir / "artifacts" / "model_selection"
    prediction_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "datetime": "2019-01-31 12:45:00",
                "segment": "valid",
                "prob_lgb_xgb_consensus": 0.62,
                "regime_gate_v2_dislocation": True,
            },
            {
                "datetime": "2019-01-31 13:00:00",
                "segment": "valid",
                "prob_lgb_xgb_consensus": 0.64,
                "regime_gate_v2_dislocation": True,
            },
        ]
    ).to_csv(prediction_dir / "all_model_predictions.csv", index=False)
    captured: dict[str, str] = {}

    def fake_load_qlib_feature_frame(config, *, instruments, fields, start, end):
        captured["start"] = start
        index = pd.MultiIndex.from_arrays(
            [
                ["SOXL", "SOXL"],
                pd.to_datetime(["2019-01-31 12:45:00", "2019-01-31 13:00:00"]),
            ],
            names=["instrument", "datetime"],
        )
        return pd.DataFrame({"$open": [10.0, 10.2], "$close": [10.1, 10.3]}, index=index)

    monkeypatch.setattr(overlay_workflow, "init_qlib_provider", lambda config: None)
    monkeypatch.setattr(overlay_workflow, "load_qlib_feature_frame", fake_load_qlib_feature_frame)

    frame = overlay_workflow._load_intraday_prediction_frame(
        CognitiveAlphaIntradayOverlaySpec(root=tmp_path),
        intraday_summary={"recorder_local_dir": str(recorder_dir)},
        as_of="2026-07-02",
        signal_config={
            "prob_col": "prob_lgb_xgb_consensus",
            "gate_col": "regime_gate_v2_dislocation",
            "gate_policy": "source_trade_gate",
        },
    )

    assert captured["start"] == "2019-01-31 09:45:00"
    assert frame["prob_lgb_xgb_consensus"].notna().all()


def test_walk_forward_metrics_records_required_annual_folds_and_fail_closed_missing_years():
    rows = []
    for date, segment, baseline_return, overlay_return in [
        ("2025-01-02", "validation", 0.10, 0.09),
        ("2025-01-03", "validation", 0.10, 0.09),
        ("2026-01-02", "test", 0.20, 0.18),
        ("2026-01-03", "test", 0.20, 0.18),
    ]:
        rows.append(
            {
                "datetime": date,
                "segment": segment,
                "variant": "baseline_next_open",
                "cost_bps": 20.0,
                "strategy_return": baseline_return,
                "target_position": 1.0,
                "entry_date": date,
                "skipped_reason": None,
            }
        )
        rows.append(
            {
                "datetime": date,
                "segment": segment,
                "variant": "overlay_a_signal_gate",
                "cost_bps": 20.0,
                "strategy_return": overlay_return,
                "target_position": 1.0,
                "entry_date": date,
                "skipped_reason": None,
            }
        )
    walk_forward = overlay_workflow._walk_forward_metrics(pd.DataFrame(rows), cost_bps=20.0)

    assert set(walk_forward["fold"]) == {"wf_2023", "wf_2024", "wf_2025", "wf_2026"}
    assert set(walk_forward["selection_uses_test_segment"]) == {False}
    missing = walk_forward.loc[walk_forward["status"].eq("infeasible")]
    assert set(missing["test_year"]) == {2023, 2024, 2025}
    assert missing["blocking_reason"].str.contains("missing_validation_data").all()
    fold_2025 = walk_forward.loc[walk_forward["fold"].eq("wf_2025")].iloc[0]
    assert fold_2025["valid_periods"] == 0
    assert fold_2025["test_periods"] == 2
    fold_2026 = walk_forward.loc[walk_forward["fold"].eq("wf_2026")].iloc[0]
    assert fold_2026["status"] == "ok"
    assert fold_2026["train_end"] == "2024-12-31"
    assert fold_2026["valid_start"] == "2025-01-01"
    assert fold_2026["valid_end"] == "2025-12-31"
    assert fold_2026["selected_variant"] == "overlay_a_signal_gate"
    assert bool(fold_2026["validation_passes_target"]) is True
    assert bool(fold_2026["test_passes_target"]) is True


def test_walk_forward_metrics_uses_fold_metadata_when_daily_replay_overlaps_years():
    rows = []
    for fold, test_year, segment, date, strategy_return in [
        ("wf_2024", 2024, "test", "2024-01-02", 0.50),
        ("wf_2025", 2025, "validation", "2024-01-02", 0.10),
        ("wf_2025", 2025, "test", "2025-01-02", 0.20),
    ]:
        for variant in ["baseline_next_open", "overlay_a_signal_gate"]:
            rows.append(
                {
                    "walk_forward_fold": fold,
                    "test_year": test_year,
                    "fold_segment": segment,
                    "datetime": date,
                    "segment": segment,
                    "variant": variant,
                    "cost_bps": 20.0,
                    "strategy_return": strategy_return,
                    "target_position": 1.0,
                    "entry_date": date,
                    "skipped_reason": None,
                }
            )

    walk_forward = overlay_workflow._walk_forward_metrics(pd.DataFrame(rows), cost_bps=20.0)

    fold_2025 = walk_forward.loc[walk_forward["fold"].eq("wf_2025")].iloc[0]
    assert fold_2025["status"] == "ok"
    assert fold_2025["valid_periods"] == 1
    assert fold_2025["test_periods"] == 1
    assert fold_2025["valid_total_return_pct"] == pytest.approx(10.0)
    assert fold_2025["test_total_return_pct"] == pytest.approx(20.0)


def test_walk_forward_metrics_fails_closed_when_intraday_predictions_are_missing():
    rows = []
    for segment, date in [("validation", "2022-01-03"), ("test", "2023-01-03")]:
        for variant, skipped_reason in [
            ("baseline_next_open", None),
            ("overlay_a_signal_gate", "missing_intraday_prediction"),
            ("overlay_b_twap4", None),
            ("overlay_c_risk_exit", "missing_intraday_prediction"),
        ]:
            rows.append(
                {
                    "walk_forward_fold": "wf_2023",
                    "test_year": 2023,
                    "fold_segment": segment,
                    "datetime": date,
                    "segment": segment,
                    "variant": variant,
                    "cost_bps": 20.0,
                    "strategy_return": 0.01,
                    "target_position": 1.0,
                    "entry_date": date,
                    "skipped_reason": skipped_reason,
                }
            )

    walk_forward = overlay_workflow._walk_forward_metrics(
        pd.DataFrame(rows),
        cost_bps=20.0,
        require_intraday_prediction_coverage=True,
    )

    fold_2023 = walk_forward.loc[walk_forward["fold"].eq("wf_2023")].iloc[0]
    assert fold_2023["status"] == "infeasible"
    assert fold_2023["blocking_reason"] == "missing_intraday_prediction_coverage"
    assert fold_2023["validation_missing_intraday_prediction_periods"] == 1
    assert fold_2023["test_missing_intraday_prediction_periods"] == 1


def test_conditional_risk_sleeve_walk_forward_selects_fold_local_validation_policy():
    rows = []
    regime_rows = []
    dates = [
        ("2025-01-02", "validation", 0.10),
        ("2025-01-03", "validation", 0.15),
        ("2025-01-06", "validation", 0.90),
        ("2025-01-07", "validation", 0.95),
        ("2026-01-02", "test", 0.10),
        ("2026-01-05", "test", 0.18),
        ("2026-01-06", "test", 0.88),
        ("2026-01-07", "test", 0.92),
    ]
    for date, segment, probability in dates:
        is_low_prob = probability < 0.50
        regime_rows.append(
            {
                "datetime": date,
                "walk_forward_fold": "wf_2026",
                "test_year": 2026,
                "fold_segment": segment,
                "probability": probability,
                "target_position": 1.0,
                "turnover": 0.1,
                "raw_target": 1.0,
            }
        )
        for cost_bps in [20.0, 50.0]:
            rows.append(
                {
                    "walk_forward_fold": "wf_2026",
                    "test_year": 2026,
                    "fold_segment": segment,
                    "datetime": date,
                    "segment": "stale_daily_segment",
                    "variant": "baseline_next_open",
                    "cost_bps": cost_bps,
                    "strategy_return": 0.02,
                    "entry_day_strategy_return": -0.04 if is_low_prob else 0.01,
                    "target_position": 1.0,
                    "entry_date": date,
                    "skipped_reason": None,
                }
            )
            rows.append(
                {
                    "walk_forward_fold": "wf_2026",
                    "test_year": 2026,
                    "fold_segment": segment,
                    "datetime": date,
                    "segment": "stale_daily_segment",
                    "variant": "overlay_b_twap3",
                    "cost_bps": cost_bps,
                    "strategy_return": 0.015 if is_low_prob else -0.03,
                    "entry_day_strategy_return": -0.01 if is_low_prob else 0.01,
                    "target_position": 1.0,
                    "entry_date": date,
                    "skipped_reason": None,
                }
            )

    walk_forward = overlay_workflow._conditional_risk_sleeve_walk_forward_metrics(
        pd.DataFrame(rows),
        pd.DataFrame(regime_rows),
        required_cost_bps=(20.0, 50.0),
        selection_cost_bps=20.0,
        condition_columns=("probability",),
    )

    assert set(walk_forward["fold"]) == {"wf_2023", "wf_2024", "wf_2025", "wf_2026"}
    assert set(walk_forward["selection_uses_test_segment"]) == {False}
    fold_2026 = walk_forward.loc[walk_forward["fold"].eq("wf_2026")].iloc[0]
    assert fold_2026["status"] == "ok"
    assert fold_2026["selected_variant"] == "overlay_b_twap3"
    assert fold_2026["condition_column"] == "probability"
    assert fold_2026["condition_operator"] == "le"
    assert fold_2026["required_cost_validation_target_count"] == 2
    assert fold_2026["required_cost_test_target_count"] == 2
    assert fold_2026["required_cost_validation_and_test_target_count"] == 2
    assert bool(fold_2026["validation_passes_target"]) is True
    assert bool(fold_2026["test_passes_target"]) is True
    assert fold_2026["validation_open_buy_dump_period_reduction"] > 0
    assert fold_2026["test_open_buy_dump_period_reduction"] > 0
    assert fold_2026["valid_periods"] == 4
    assert fold_2026["test_periods"] == 4


def test_conditional_risk_sleeve_walk_forward_fails_closed_without_validation_cost_target():
    rows = []
    regime_rows = []
    dates = [
        ("2025-01-02", "validation", 0.10),
        ("2025-01-03", "validation", 0.20),
        ("2025-01-06", "validation", 0.80),
        ("2025-01-07", "validation", 0.90),
        ("2026-01-02", "test", 0.10),
        ("2026-01-05", "test", 0.20),
    ]
    for date, segment, probability in dates:
        regime_rows.append(
            {
                "datetime": date,
                "walk_forward_fold": "wf_2026",
                "test_year": 2026,
                "fold_segment": segment,
                "probability": probability,
                "target_position": 1.0,
            }
        )
        for cost_bps in [20.0, 50.0]:
            rows.append(
                {
                    "walk_forward_fold": "wf_2026",
                    "test_year": 2026,
                    "fold_segment": segment,
                    "datetime": date,
                    "segment": segment,
                    "variant": "baseline_next_open",
                    "cost_bps": cost_bps,
                    "strategy_return": 0.02,
                    "entry_day_strategy_return": -0.01,
                    "target_position": 1.0,
                    "entry_date": date,
                    "skipped_reason": None,
                }
            )
            rows.append(
                {
                    "walk_forward_fold": "wf_2026",
                    "test_year": 2026,
                    "fold_segment": segment,
                    "datetime": date,
                    "segment": segment,
                    "variant": "overlay_b_twap3",
                    "cost_bps": cost_bps,
                    "strategy_return": -0.02,
                    "entry_day_strategy_return": -0.02,
                    "target_position": 1.0,
                    "entry_date": date,
                    "skipped_reason": None,
                }
            )

    walk_forward = overlay_workflow._conditional_risk_sleeve_walk_forward_metrics(
        pd.DataFrame(rows),
        pd.DataFrame(regime_rows),
        required_cost_bps=(20.0, 50.0),
        selection_cost_bps=20.0,
        condition_columns=("probability",),
    )

    fold_2026 = walk_forward.loc[walk_forward["fold"].eq("wf_2026")].iloc[0]
    assert fold_2026["status"] == "infeasible"
    assert fold_2026["blocking_reason"] == "failed_validation_target"
    assert pd.isna(fold_2026["selected_variant"])
    assert fold_2026["diagnostic_variant"] == "overlay_b_twap3"
    assert fold_2026["required_cost_validation_target_count"] == 0
    assert bool(fold_2026["selection_uses_test_segment"]) is False


def test_select_regime_policy_prefers_validation_drawdown_improvement_over_score():
    policy_selection = pd.DataFrame(
        [
            {
                "variant": "overlay_a_signal_gate",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_threshold": 0.65,
                "passes_validation_target": True,
                "passes_test_target": False,
                "passes_validation_and_test_target": False,
                "validation_return_retention_pct": 93.0,
                "validation_drawdown_improvement_pct": 2.9,
                "validation_return_pct": 120.0,
                "selection_score": 99.0,
            },
            {
                "variant": "overlay_b_twap4",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_threshold": 0.62,
                "passes_validation_target": True,
                "passes_test_target": True,
                "passes_validation_and_test_target": True,
                "validation_return_retention_pct": 89.0,
                "validation_drawdown_improvement_pct": 3.8,
                "validation_return_pct": 115.0,
                "selection_score": 97.0,
            },
        ]
    )

    selected = overlay_workflow._select_regime_policy(policy_selection)

    assert selected["variant"] == "overlay_b_twap4"
    assert selected["condition_threshold"] == 0.62
    assert selected["selection_uses_test_segment"] is False


def test_select_regime_policy_requires_validation_target_at_required_costs():
    policy_selection = pd.DataFrame(
        [
            {
                "cost_bps": 20.0,
                "variant": "overlay_a_signal_gate",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.60,
                "condition_threshold": 0.60,
                "passes_validation_target": True,
                "passes_test_target": True,
                "validation_return_retention_pct": 93.0,
                "validation_drawdown_improvement_pct": 4.5,
                "validation_return_pct": 125.0,
                "selection_score": 105.0,
            },
            {
                "cost_bps": 50.0,
                "variant": "overlay_a_signal_gate",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.60,
                "condition_threshold": 0.60,
                "passes_validation_target": False,
                "passes_test_target": False,
                "validation_return_retention_pct": 83.0,
                "validation_drawdown_improvement_pct": 4.5,
                "validation_return_pct": 108.0,
                "selection_score": 90.0,
            },
            {
                "cost_bps": 20.0,
                "variant": "overlay_b_twap4",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "passes_validation_target": True,
                "passes_test_target": True,
                "validation_return_retention_pct": 89.0,
                "validation_drawdown_improvement_pct": 3.8,
                "validation_return_pct": 115.0,
                "selection_score": 97.0,
            },
            {
                "cost_bps": 50.0,
                "variant": "overlay_b_twap4",
                "condition_column": "probability",
                "condition_operator": "le",
                "condition_quantile": 0.75,
                "condition_threshold": 0.62,
                "passes_validation_target": True,
                "passes_test_target": True,
                "validation_return_retention_pct": 88.0,
                "validation_drawdown_improvement_pct": 3.7,
                "validation_return_pct": 112.0,
                "selection_score": 96.0,
            },
        ]
    )

    selected = overlay_workflow._select_regime_policy(
        policy_selection,
        required_cost_bps=(20.0, 50.0),
        selection_cost_bps=20.0,
    )

    assert selected["variant"] == "overlay_b_twap4"
    assert selected["condition_threshold"] == 0.62
    assert selected["validation_cost_robust"] is True
    assert selected["required_cost_bps"] == [20.0, 50.0]
    assert selected["required_cost_validation_target_count"] == 2
    assert selected["required_cost_test_target_count"] == 2
    assert [row["cost_bps"] for row in selected["required_cost_results"]] == [20.0, 50.0]


def test_daily_position_source_audit_records_required_year_coverage(tmp_path):
    recorder_dir = tmp_path / "recorder"
    position_dir = recorder_dir / "artifacts" / "cognitive_alpha_position"
    position_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "datetime": "2022-01-03",
                "segment": "validation",
                "target_position": 0.5,
                "strategy_return": 0.01,
                "selected": True,
            },
            {
                "datetime": "2023-01-03",
                "segment": "test",
                "target_position": 0.5,
                "strategy_return": 0.01,
                "selected": True,
            },
            {
                "datetime": "2025-01-03",
                "segment": "validation",
                "target_position": 0.5,
                "strategy_return": 0.01,
                "selected": True,
            },
            {
                "datetime": "2026-01-05",
                "segment": "test",
                "target_position": 0.5,
                "strategy_return": 0.01,
                "selected": True,
            },
        ]
    ).to_csv(position_dir / "positions.csv", index=False)

    summary = {"recorder_id": "rec-1", "recorder_local_dir": str(recorder_dir)}
    audit = overlay_workflow._daily_position_source_audit(
        CognitiveAlphaIntradayOverlaySpec(root=tmp_path),
        daily_summary=summary,
        daily_positions_path=position_dir / "positions.csv",
    )

    assert audit.loc[0, "required_position_years"] == "2022,2023,2024,2025,2026"
    assert audit.loc[0, "selected_years"] == "2022,2023,2025,2026"
    assert audit.loc[0, "missing_required_position_years"] == "2024"
    assert bool(audit.loc[0, "satisfies_required_annual_walk_forward"]) is False
    assert audit.loc[0, "blocking_reason"] == "missing_required_position_years"


def test_load_daily_selected_positions_prefers_walk_forward_replay_artifact(tmp_path):
    recorder_dir = tmp_path / "recorder"
    position_dir = recorder_dir / "artifacts" / "cognitive_alpha_position"
    position_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "datetime": "2026-01-05",
                "segment": "test",
                "target_position": 0.1,
                "strategy_return": 0.001,
                "selected": True,
            }
        ]
    ).to_csv(position_dir / "positions.csv", index=False)
    pd.DataFrame(
        [
            {
                "walk_forward_fold": "wf_2023",
                "test_year": 2023,
                "fold_segment": "validation",
                "datetime": "2022-01-03",
                "target_position": 0.4,
                "strategy_return": 0.004,
                "selected": True,
            },
            {
                "walk_forward_fold": "wf_2023",
                "test_year": 2023,
                "fold_segment": "test",
                "datetime": "2023-01-03",
                "target_position": 0.5,
                "strategy_return": 0.005,
                "selected": True,
            },
        ]
    ).to_csv(position_dir / "walk_forward_positions.csv", index=False)

    summary = {"recorder_id": "rec-1", "recorder_local_dir": str(recorder_dir)}
    positions = overlay_workflow._load_daily_selected_positions(summary)
    audit = overlay_workflow._daily_position_source_audit(
        CognitiveAlphaIntradayOverlaySpec(root=tmp_path),
        daily_summary=summary,
    )

    assert positions["datetime"].dt.year.tolist() == [2022, 2023]
    assert positions["segment"].tolist() == ["validation", "test"]
    assert audit.loc[0, "artifact_name"] == "walk_forward_positions.csv"
    assert audit.loc[0, "selected_years"] == "2022,2023"

    latest_positions = overlay_workflow._load_daily_selected_positions(summary, prefer_walk_forward=False)
    assert latest_positions["datetime"].dt.year.tolist() == [2026]
    assert latest_positions["segment"].tolist() == ["test"]


def test_delayed_walk_forward_feasibility_uses_replay_validation_and_test_positions(tmp_path):
    spec = CognitiveAlphaIntradayOverlaySpec(root=tmp_path)
    daily_positions = pd.DataFrame(
        [
            {
                "walk_forward_fold": "wf_2023",
                "test_year": 2023,
                "fold_segment": "validation",
                "datetime": "2022-01-03",
                "target_position": 0.4,
                "strategy_return": 0.004,
            },
            {
                "walk_forward_fold": "wf_2023",
                "test_year": 2023,
                "fold_segment": "test",
                "datetime": "2023-01-03",
                "target_position": 0.5,
                "strategy_return": 0.005,
            },
            {
                "walk_forward_fold": "wf_2024",
                "test_year": 2024,
                "fold_segment": "validation",
                "datetime": "2023-01-03",
                "target_position": 0.5,
                "strategy_return": 0.005,
            },
            {
                "walk_forward_fold": "wf_2024",
                "test_year": 2024,
                "fold_segment": "test",
                "datetime": "2024-01-02",
                "target_position": 0.6,
                "strategy_return": 0.006,
            },
            {
                "walk_forward_fold": "wf_2025",
                "test_year": 2025,
                "fold_segment": "validation",
                "datetime": "2024-01-02",
                "target_position": 0.6,
                "strategy_return": 0.006,
            },
            {
                "walk_forward_fold": "wf_2025",
                "test_year": 2025,
                "fold_segment": "test",
                "datetime": "2025-01-02",
                "target_position": 0.7,
                "strategy_return": 0.007,
            },
            {
                "walk_forward_fold": "wf_2026",
                "test_year": 2026,
                "fold_segment": "validation",
                "datetime": "2025-01-02",
                "target_position": 0.7,
                "strategy_return": 0.007,
            },
            {
                "walk_forward_fold": "wf_2026",
                "test_year": 2026,
                "fold_segment": "test",
                "datetime": "2026-07-02",
                "target_position": 0.8,
                "strategy_return": 0.008,
            },
        ]
    )
    daily_positions = overlay_workflow._normalize_daily_positions(daily_positions)
    intraday_summary = {
        "external_intraday_context": {
            "coverage_start": "2019-01-02 09:45:00",
            "coverage_end": "2026-07-02 16:00:00",
        }
    }
    intraday_predictions = pd.DataFrame()

    feasibility = overlay_workflow._delayed_walk_forward_feasibility(
        spec,
        intraday_summary=intraday_summary,
        daily_positions=daily_positions,
        intraday_predictions=intraday_predictions,
    )

    assert set(feasibility["test_year"]) == {2023, 2024, 2025, 2026}
    assert feasibility["has_daily_position_valid_coverage"].all()
    assert feasibility["has_daily_position_test_coverage"].all()
    assert feasibility["feasible"].all()

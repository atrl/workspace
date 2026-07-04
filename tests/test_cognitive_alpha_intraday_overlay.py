from __future__ import annotations

import pandas as pd
import pytest

import src.workflow.cognitive_alpha_intraday_overlay as overlay_workflow
from src.workflow.cognitive_alpha_intraday_overlay import (
    CognitiveAlphaIntradayOverlaySpec,
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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import stage1_open_action_value as action_value
from src.workflow.stage1_open_action_value import ACTION_FEATURES


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "stage1_open_action_value_v1"


def test_open_action_features_are_known_before_open_execution():
    assert not any(name.startswith("early_") for name in ACTION_FEATURES)
    assert not any(name.startswith("log_volume_") for name in ACTION_FEATURES)
    assert {
        "portfolio_drawdown",
        "target_soxl_weight",
        "pretrade_soxl_weight",
        "effective_beta_delta",
        "open_turnover",
    } <= set(ACTION_FEATURES)


def test_open_route_is_independent_of_future_recovery_signal(monkeypatch):
    class HoldModel:
        def predict_proba(self, features):
            return np.tile([0.99, 0.01], (len(features), 1))

    dates = pd.date_range("2021-01-04", periods=2, freq="B")
    joined = pd.DataFrame(
        {
            "signal_date": dates,
            "trade_date": dates + pd.Timedelta(days=1),
            "exit_date": dates + pd.Timedelta(days=2),
            "open_soxl_weight": [0.10, 0.11],
            "open_soxx_weight": [0.20, 0.20],
            "open_cash_weight": [0.70, 0.69],
            "soxl_weight": [0.10, 0.15],
            "soxx_weight": [0.20, 0.25],
            "cash_weight": [0.70, 0.60],
            "recovery_action": [False, False],
            "soxl_return": [0.0, 0.0],
            "soxx_return": [0.0, 0.0],
            "early_ret_soxl": [0.0, 0.0],
            "early_ret_soxx": [0.0, 0.0],
        }
    )
    shadow = pd.DataFrame(
        [
            {**{feature: 0.0 for feature in ACTION_FEATURES}, "signal_date": date}
            for date in dates
        ]
    )
    monkeypatch.setattr(
        action_value, "_join_market_and_replay", lambda *args: joined.copy()
    )
    monkeypatch.setattr(
        action_value, "_build_training_frame", lambda *args, **kwargs: shadow.copy()
    )
    spec = action_value.Stage1OpenActionValueSpec(root=ROOT)

    def replay():
        return action_value._simulate_router(
            pd.DataFrame(),
            pd.DataFrame(),
            models={2021: HoldModel()},
            source_spec=action_value.IntradayRecoveryDirectAllocationSpec(root=ROOT),
            spec=spec,
            segment="causality_test",
        )

    no_recovery = replay()
    joined.loc[1, "recovery_action"] = True
    with_recovery = replay()

    # Day one establishes the same holdings. Only the future 10:31 signal
    # changes on day two, so the 09:31 execute/hold decision must be identical.
    pd.testing.assert_frame_equal(
        no_recovery[
            ["execute_probability", "execute_open_trade", "skipped_open_target"]
        ],
        with_recovery[
            ["execute_probability", "execute_open_trade", "skipped_open_target"]
        ],
    )
    assert no_recovery["execute_open_trade"].tolist() == [True, False]
    assert no_recovery["skipped_open_target"].tolist() == [False, True]
    assert no_recovery.iloc[1]["soxl_weight"] == pytest.approx(0.10 / 0.9997)
    assert no_recovery.iloc[1]["turnover"] == pytest.approx(0.0)
    # The later recovery still executes from the holdings actually kept at open.
    assert bool(with_recovery.iloc[1]["recovery_action"]) is True
    assert with_recovery.iloc[1]["soxl_weight"] == pytest.approx(0.15)
    assert with_recovery.iloc[1]["turnover"] == pytest.approx(0.40 - 0.30 / 0.9997)


def test_action_label_matures_after_exit_open_and_next_actual_target(monkeypatch):
    joined = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2016-12-29", "2016-12-30", "2017-01-03"]),
            "trade_date": pd.to_datetime(["2016-12-30", "2017-01-03", "2017-01-04"]),
            "exit_date": pd.to_datetime(["2017-01-03", "2017-01-04", "2017-01-05"]),
            "strategy_return": [0.0, 0.0, 0.0],
        }
    )
    monkeypatch.setattr(
        action_value, "_join_market_and_replay", lambda *args: joined.copy()
    )
    monkeypatch.setattr(
        action_value,
        "_action_features",
        lambda *args, **kwargs: {feature: 0.0 for feature in ACTION_FEATURES},
    )
    monkeypatch.setattr(
        action_value,
        "_branch_outcomes",
        lambda *args, **kwargs: {
            "execute_net": 0.0,
            "skip_net": 0.0,
            "execute_turnover": 0.0,
            "skip_turnover": 0.0,
            "execute_end": np.array([0.10, 0.20, 0.70]),
            "skip_end": np.array([0.10, 0.20, 0.70]),
        },
    )
    training = action_value._build_training_frame(
        pd.DataFrame(),
        pd.DataFrame(),
        source_spec=action_value.IntradayRecoveryDirectAllocationSpec(root=ROOT),
        label_cost_bps=25.0,
    )

    assert training.loc[0, "label_available_at"] == pd.Timestamp("2017-01-03T14:31Z")
    assert training.loc[1, "label_available_at"] == pd.Timestamp("2017-01-04T14:31Z")
    assert pd.isna(training.loc[2, "label_available_at"])
    assert pd.isna(training.loc[2, "execute_advantage"])


def test_qlib_training_splits_exclude_labels_maturing_at_test_open():
    signal_dates = pd.to_datetime(
        ["2016-12-28", "2016-12-29", "2016-12-30", "2017-01-03", "2022-12-29", "2022-12-30"]
    )
    training = pd.DataFrame(
        [{feature: 0.0 for feature in ACTION_FEATURES} for _ in signal_dates]
    )
    training["signal_date"] = signal_dates
    training["year"] = signal_dates.year
    training["execute_advantage"] = 0.01
    training["label_available_at"] = pd.to_datetime(
        [
            "2016-12-30T14:31Z", "2017-01-03T14:31Z", "2017-01-04T14:31Z",
            "2017-01-05T14:31Z", "2023-01-03T14:31Z", "2023-01-04T14:31Z",
        ],
        utc=True,
    )
    evaluation = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2017-01-03", "2022-01-03", "2023-01-03"]),
            "trade_date": pd.to_datetime(["2017-01-04", "2022-01-04", "2023-01-04"]),
        }
    )
    dataset, audit = action_value._build_qlib_dataset(
        training,
        action_value.Stage1OpenActionValueSpec(root=ROOT),
        evaluation_frame=evaluation,
    )
    year_train = action_value._prepare_mature_training(
        dataset, "train_2017", decision_at=audit["training_cutoffs"]["train_2017"]
    )
    outer_train = action_value._prepare_mature_training(
        dataset, "train_outer", decision_at=audit["training_cutoffs"]["train_outer"]
    )

    assert year_train.index.get_level_values("datetime").tolist() == signal_dates[:2].tolist()
    assert pd.Timestamp("2022-12-29") in outer_train.index.get_level_values("datetime")
    assert pd.Timestamp("2022-12-30") not in outer_train.index.get_level_values("datetime")
    assert audit["excluded_unavailable_label_rows"]["train_2017"] == 1
    assert audit["excluded_unavailable_label_rows"]["train_outer"] == 1
    assert "label_available_at" not in ACTION_FEATURES


def test_on_policy_open_uses_own_drawdown_and_never_future_intraday_data():
    class HoldModel:
        def predict_proba(self, features):
            return np.tile([0.99, 0.01], (len(features), 1))

    row = pd.Series({
        "soxl_weight": 0.11, "soxx_weight": 0.20, "cash_weight": 0.69,
        "tail_probability": 0.1, "alpha_probability": 0.8,
        "early_ret_soxl": 0.0, "early_ret_soxx": 0.0,
    })
    kwargs = {
        "pretrade_weights": np.array([0.10, 0.20, 0.70]),
        "peak": 1.0, "risk_state": 0, "risk_clock": 0, "history": [],
        "recovery_spec": action_value.IntradayRecoveryDirectAllocationSpec(root=ROOT),
        "model": HoldModel(), "action_spec": action_value.Stage1OpenActionValueSpec(root=ROOT),
    }
    normal = action_value.prepare_on_policy_open(row, equity=1.0, **kwargs)
    row["early_ret_soxl"], row["early_ret_soxx"] = 0.9, -0.9
    changed_future = action_value.prepare_on_policy_open(row, equity=1.0, **kwargs)
    distressed = action_value.prepare_on_policy_open(row, equity=0.8, **kwargs)

    assert normal["execute_open_target"] is False
    assert normal["features"] == changed_future["features"]
    assert changed_future["execute_open_target"] is False
    assert distressed["execute_open_target"] is True
    assert distressed["drawdown"] == pytest.approx(0.2)
    assert distressed["current_beta_cap"] < normal["current_beta_cap"]
    assert distressed["features"]["portfolio_drawdown"] == pytest.approx(0.2)


def test_development_annual_gate_does_not_select_on_outer_results():
    baseline = {
        "return_pct": 100.0, "max_drawdown_pct": -30.0, "worst_day_pct": -5.0,
        "cvar5_pct": -3.0, "worst_rolling_5d_pct": -10.0,
        "open_trade_days": 100, "total_turnover": 10.0,
    }
    candidate = {**baseline, "return_pct": 101.0, "open_trade_days": 90, "total_turnover": 9.0}
    annual = pd.DataFrame([
        {"variant": "open_action_value_candidate", "segment": "development_walk_forward", "year": 2020, "wealth_ratio_vs_baseline": 1.0},
        {"variant": "open_action_value_candidate", "segment": "outer_reused_diagnostic", "year": 2023, "wealth_ratio_vs_baseline": 0.1},
    ])
    replay = pd.DataFrame({"strategy_return": [0.1], "turnover": [0.2]})
    gates = action_value._promotion_gates(
        baseline_10=baseline, baseline_25=baseline,
        candidate_10=candidate, candidate_25=candidate, annual=annual,
        fallback=replay, baseline_replay=replay,
        spec=action_value.Stage1OpenActionValueSpec(root=ROOT),
    )
    assert gates["annual_wealth_floor_pass"] is True
    assert all(gates.values())


def test_cost_stress_reprices_fixed_events_and_updates_net_equity_weights():
    replay = pd.DataFrame({
        "execute_open_target": [True, False], "recovery_action": [False, False],
        "open_soxl_weight": [0.3, 0.3], "open_soxx_weight": [0.5, 0.5], "open_cash_weight": [0.2, 0.2],
        "soxl_weight": [0.3, 0.3], "soxx_weight": [0.5, 0.5], "cash_weight": [0.2, 0.2],
        "soxl_return": [0.0, 0.1], "soxx_return": [0.0, -0.05],
        "early_ret_soxl": [0.0, 0.0], "early_ret_soxx": [0.0, 0.0],
        "equity": [0.9992, 1.0042],
    })
    stress = action_value._stress(replay, cost_bps=25.0)
    assert stress.iloc[0]["fees"] == pytest.approx(0.002)
    assert stress.iloc[1]["fees"] == 0.0
    assert stress.iloc[1]["turnover"] == 0.0
    assert stress.iloc[1]["equity"] == pytest.approx(1.003)
    assert stress.iloc[1]["ending_cash_weight"] == pytest.approx(0.198 / 1.003)
    assert stress.iloc[1]["decision_source_equity"] == pytest.approx(1.0042)
    assert (1.0 + stress["strategy_return"]).prod() == pytest.approx(1.003)


def test_historical_card_is_invalidated_until_causal_revalidation():
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (CARD / summary["recovery_validation"]).read_text(encoding="utf-8")
    )
    assert summary["decision"] == "blocked_pending_causal_revalidation"
    assert summary["verification_status"] == "invalidated_pending_causal_revalidation"
    assert summary["development_gate_pass"] is False
    assert summary["selection_uses_outer_test"] is False
    assert validation["production_enabled"] is False
    assert validation["historical_recorder_id"] == summary["historical_result"]["recorder_id"]
    assert {item["issue"] for item in validation["invalidated_by"]} == {
        "future_recovery_signal_in_open_route",
        "training_labels_crossed_execution_split",
        "outer_annual_rows_in_development_gate",
        "fees_not_debited_from_next_pretrade_cash",
    }
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"


def test_annual_walk_forward_floor_and_outer_separation():
    annual = pd.read_csv(CARD / "annual_metrics.csv")
    candidate = annual[
        annual["variant"].eq("open_action_value_candidate")
        & annual["segment"].eq("development_walk_forward")
        & annual["year"].ge(2017)
    ]
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))

    assert candidate["wealth_ratio_vs_baseline"].min() >= 0.985
    assert summary["outer_role"] == "reused_diagnostic_only_not_model_selection"
    assert summary["production_change"] == "blocked_pending_causal_revalidation"
    assert summary["historical_result"]["development_gates"]["baseline_fallback_identity"] is True


def test_card_is_slim_and_notebook_is_qlib_native():
    forbidden = {
        "predictions.csv",
        "signals.csv",
        "portfolio_report.csv",
        "qlib_portfolio_report.csv",
        "qlib_positions.csv",
        "all_model_predictions.csv",
    }
    assert not forbidden.intersection(path.name for path in CARD.iterdir())
    notebook = json.loads(
        (CARD / "stage1_open_action_value_v1.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "qlib.init" in source
    assert "D.calendar" in source
    assert "DatasetH" in source
    assert "R.start" in source
    assert "run_stage1_open_action_value_card" in source
    assert "中文结论" in source

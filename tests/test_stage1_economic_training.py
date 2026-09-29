from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_economic_action import ECONOMIC_FEATURES
from src.workflow import stage1_economic_training as training


def _frame(periods=275):
    dates = pd.bdate_range("2016-01-04", periods=periods)
    return pd.DataFrame({
        "signal_date": dates, "trade_date": dates + pd.offsets.BDay(1), "exit_date": dates + pd.offsets.BDay(2),
        "segment": "development_walk_forward", "soxl_return": np.sin(np.arange(periods)) * .003,
        "soxx_return": np.cos(np.arange(periods)) * .001, "soxl_weight": .20,
        "soxx_weight": .5, "cash_weight": .3, "tail_probability": .1, "alpha_probability": .6,
        "early_ret_soxl": np.nan, "early_ret_soxx": np.nan,
    })


def _kwargs():
    return {"recovery_models": {}, "source_spec": IntradayRecoveryDirectAllocationSpec(root=Path(".")),
            "selected_model": {"action_beta": .6, "minimum_source_alpha_probability": .25, "probability_threshold": .825}}


def test_dataset_filters_signal_year_and_strict_information_cutoff():
    rows = pd.DataFrame([{name: float(index + 1) for name in ECONOMIC_FEATURES} for index in range(4)])
    rows["signal_date"] = pd.to_datetime(["2025-12-26", "2025-12-29", "2025-12-30", "2026-01-02"])
    rows["execute_advantage_bps"] = [10.0, 20.0, 30.0, 40.0]
    rows["label_available_at"] = pd.to_datetime([
        "2025-12-31T14:31:00Z", "2026-01-01T05:00:00Z", "2026-01-02T14:31:00Z", "2025-12-31T14:31:00Z",
    ], utc=True)
    dataset = training.build_economic_dataset(rows)
    mature, cutoff = training.mature_economic_rows(dataset, model_year=2026,
        first_decision_at=pd.Timestamp("2026-01-05T14:31:00Z"))
    assert cutoff == pd.Timestamp("2026-01-01T05:00:00Z")
    assert mature.execute_advantage_bps.tolist() == [10.0]


def test_future_year_returns_cannot_change_prior_annual_fit():
    frame = _frame()
    first = training.train_causal_economic_models(frame, **_kwargs())
    altered = frame.copy()
    altered.loc[altered.signal_date.dt.year.eq(2017), ["soxl_return", "soxx_return"]] = [.2, .1]
    second = training.train_causal_economic_models(altered, **_kwargs())
    assert first["final_model_year"] == 2017
    np.testing.assert_allclose(first["models"][2017].named_steps["ridge"].coef_, second["models"][2017].named_steps["ridge"].coef_, atol=1e-12)
    first_2017 = first["replay"].loc[first["replay"].signal_date.dt.year.eq(2017)].iloc[0]
    second_2017 = second["replay"].loc[second["replay"].signal_date.dt.year.eq(2017)].iloc[0]
    assert first_2017.predicted_net_advantage_bps == pytest.approx(second_2017.predicted_net_advantage_bps)
    diagnostic = first["diagnostics"].iloc[0]
    assert pd.Timestamp(diagnostic.label_available_at_max) < pd.Timestamp("2017-01-01", tz="America/New_York")
    assert first["models"][2016] is None
    assert first["dataset_audit"]["training_state"] == "causal_own_policy_path"


def test_counterfactual_execute_does_not_copy_observed_hold_or_recovery(monkeypatch):
    calls = []
    def recovery(row, **kwargs):
        calls.append(kwargs)
        return {"recovery_action": False, "target_weights": None}
    monkeypatch.setattr(training, "prepare_economic_recovery", recovery)
    opened = {"desired_open_weights": np.array([0., .5, .5]), "execution_target_weights": np.array([0., 0., 1.]),
              "history": [], "risk_clock": 3}
    row = pd.Series({"soxl_return": .0, "soxx_return": .02, "early_ret_soxl": .0, "early_ret_soxx": .005})
    results = {}
    for name, execute in [("execute", True), ("hold", False)]:
        results[name] = training._settle_branch(row, pretrade=np.array([0., 0., 1.]), opened=opened,
            execute=execute, cost_bps=25., recovery_model=None, source_spec=_kwargs()["source_spec"], selected_model={})
    assert len(calls) == 2
    assert calls[0]["pretrade_weights"][1] > 0
    assert calls[1]["pretrade_weights"][1] == 0
    assert all(call["open_state"]["risk_clock"] == 3 for call in calls)
    assert all(call["cost_bps"] == 25 for call in calls)
    assert results["execute"]["settled"]["open_turnover"] > 0
    assert results["hold"]["settled"]["open_turnover"] == 0


def test_common_net_reset_target_uses_each_branch_wealth_and_holdings():
    executed = {"settled": {"ending_weights": np.array([0., .6, .4]), "net_equity": 1.1, "net_return": .1}}
    held = {"settled": {"ending_weights": np.array([0., .4, .6]), "net_equity": 1.0, "net_return": .0}}
    sample = {"exit_date": pd.Timestamp("2022-01-04")}
    next_row = pd.Series({"trade_date": pd.Timestamp("2022-01-04")})
    training._finalize_label(sample, {"execute": executed, "hold": held}, next_row=next_row, next_target=np.array([0., .5, .5]))
    c = .0025
    # Sell branch: fee=c*(.6-.5*(1-fee)); buy branch: fee=c*(.5*(1-fee)-.4).
    execute_fee = c * .1 / (1 - .5 * c)
    hold_fee = c * .1 / (1 + .5 * c)
    assert sample["execute_reset_fee"] == pytest.approx(1.1 * execute_fee)
    assert sample["hold_reset_fee"] == pytest.approx(hold_fee)
    assert sample["execute_advantage_bps"] == pytest.approx(10_000 * (.1 - 1.1 * execute_fee + hold_fee))
    assert sample["label_available_at"] == pd.Timestamp("2022-01-04T14:31:00Z")


def test_portfolio_reset_does_not_erase_prior_training_rows():
    frame = _frame(275)
    frame.loc[frame.signal_date.dt.year.eq(2017), "segment"] = "outer_diagnostic"
    result = training.train_causal_economic_models(frame, **_kwargs())
    assert result["diagnostics"].iloc[0].training_rows >= 200
    assert len(result["training_rows"]) == len(frame)
    assert "unavailable_segment_reset_boundary" in result["training_rows"].label_status.tolist()
    assert result["dataset_audit"]["training_history_reset_on_segment"] is False


def test_days_since_trade_counts_completed_no_trade_sessions():
    frame = _frame(4)
    frame[["soxl_weight", "soxx_weight", "soxl_return", "soxx_return"]] = 0.0
    frame["cash_weight"] = 1.0
    replay = training.simulate_economic_policy(frame, models={}, **_kwargs())
    assert replay.days_since_trade.tolist() == [0, 1, 2, 3]
    assert replay.ending_days_since_trade.tolist() == [1, 2, 3, 4]
    assert not replay.execute_open_trade.any()


def test_only_simulation_accepts_explicit_last_row_terminal_close():
    frame = _frame(4)
    frame.loc[3, "exit_date"] = frame.loc[3, "trade_date"]
    frame.loc[3, "exit_basis"] = "terminal_close"
    replay = training.simulate_economic_policy(frame, models={}, **_kwargs())
    assert replay.iloc[-1].exit_basis == "terminal_close"
    with pytest.raises(ValueError, match="ordering"):
        training.train_causal_economic_models(frame, **_kwargs())
    frame.loc[2, "exit_date"] = frame.loc[2, "trade_date"]
    frame.loc[2, "exit_basis"] = "terminal_close"
    with pytest.raises(ValueError, match="ordering"):
        training.simulate_economic_policy(frame, models={}, **_kwargs())

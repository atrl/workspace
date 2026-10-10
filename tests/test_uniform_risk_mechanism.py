"""Fixtures and contracts for the uniform risk-mechanism analysis."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import uniform_risk_mechanism as R


def _frame(n=20, start="2020-01-01", **overrides):
    dates = pd.bdate_range(start, periods=n)
    frame = pd.DataFrame({
        "signal_date": dates, "trade_date": dates,
        "strategy_return": 0.001, "soxx_return": 0.002, "ending_cash_weight": 0.5,
        "drawdown_before": 0.05, "risk_state_before": 0, "risk_clock_before": 0,
        "risk_state_after": 0, "risk_clock_after": 0, "current_beta_cap": 1.0, "raw_beta": 1.0,
        "pretrade_beta": 1.0, "desired_beta": 1.0, "risk_q": 1.0,
        "stop_entered": False, "emergency_active": False, "trailing_loss_active": False,
        "clock_cap_binding": False, "emergency_cap_binding": False, "trailing_cap_binding": False,
        "max_soxl_binding": False, "min_cash_binding": False,
    })
    for key, value in overrides.items():
        frame[key] = value
    return frame


def _one(results, case):
    return [r for r in results if r["case"] == case][0]


def test_state_share_stop_and_emergency_runs():
    emergency = np.zeros(20, bool); emergency[5:8] = True  # 3-day run
    stop = np.zeros(20, bool); stop[3] = True; stop[12] = True
    frame = _frame(emergency_active=emergency, stop_entered=stop,
                   risk_state_before=[0] * 15 + [3] * 5)
    out = _one(R.analyze_risk_mechanism({"candidate::development_walk_forward::10": frame}),
               "candidate::development_walk_forward::10")
    assert out["status"] == "ok"
    assert out["emergency_longest_consecutive_days"] == 3
    assert out["stop_events"] == 2          # event pulse: each True is one stop
    assert out["emergency_entries"] == 1    # consecutive emergency is one episode
    assert out["state_share"]["0"] == pytest.approx(0.75)
    assert out["state_share"]["3"] == pytest.approx(0.25)


def test_binding_share_and_phase_stats():
    frame = _frame(clock_cap_binding=[True] * 4 + [False] * 16,
                   emergency_active=[True] * 5 + [False] * 15,
                   soxx_return=[-0.01] * 5 + [0.02] * 15,
                   strategy_return=[-0.005] * 5 + [0.003] * 15)
    out = _one(R.analyze_risk_mechanism({"x::development_walk_forward::25": frame}),
               "x::development_walk_forward::25")
    assert out["binding_share"]["clock_cap_binding"] == pytest.approx(0.2)
    em = out["phases"]["emergency_active"]
    assert em["days"] == 5 and em["soxx_down_days"] == 5
    assert em["mean_soxx_return"] == pytest.approx(-0.01)
    assert em["mean_strategy_return"] == pytest.approx(-0.005)


def test_paired_log_wealth_diff_aligned():
    raw = _frame(strategy_return=[0.0] * 20)
    cand = _frame(strategy_return=[0.01] * 20)
    out = _one(R.analyze_risk_mechanism({
        "raw_allocator_only::development_walk_forward::10": raw,
        "full_withhold_and_recovery::development_walk_forward::10": cand,
    }), "full_withhold_and_recovery::development_walk_forward::10")
    paired = out["paired_vs_raw"]
    assert paired["status"] == "ok"
    assert paired["paired_log_wealth_diff"] == pytest.approx(20 * np.log1p(0.01))
    assert out["status"] == "ok"


def test_paired_needs_data_on_misaligned_dates():
    raw = _frame(strategy_return=[0.0] * 20)
    cand = _frame(n=19, strategy_return=[0.01] * 19)
    out = _one(R.analyze_risk_mechanism({
        "raw_allocator_only::development_walk_forward::10": raw,
        "full_withhold_and_recovery::development_walk_forward::10": cand,
    }), "full_withhold_and_recovery::development_walk_forward::10")
    assert out["paired_vs_raw"]["status"] == "needs_data"


def test_missing_raw_reference_needs_data():
    cand = _frame()
    out = _one(R.analyze_risk_mechanism({"y::outer_reused_diagnostic::25": cand}),
               "y::outer_reused_diagnostic::25")
    assert out["paired_vs_raw"]["status"] == "needs_data"


def test_failclosed_missing_columns():
    frame = _frame().drop(columns=["emergency_active"])
    out = _one(R.analyze_risk_mechanism({"z::development_walk_forward::10": frame}),
               "z::development_walk_forward::10")
    assert out["status"] == "needs_data"
    assert "emergency_active" in out["reason"]


def test_failclosed_duplicate_and_unordered_dates():
    frame = _frame()
    dup = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
    out = _one(R.analyze_risk_mechanism({"z::development_walk_forward::10": dup}),
               "z::development_walk_forward::10")
    assert out["status"] == "needs_data" and "duplicate" in out["reason"]
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    out2 = _one(R.analyze_risk_mechanism({"z::development_walk_forward::10": shuffled}),
                "z::development_walk_forward::10")
    assert out2["status"] == "needs_data" and "ordered" in out2["reason"]


def test_raw_layer_only_optional_state_may_be_nan():
    frame = _frame()            # other raw state finite, flags False
    frame["current_beta_cap"] = np.nan
    frame["risk_q"] = np.nan
    out = _one(R.analyze_risk_mechanism({"raw_allocator_only::development_walk_forward::10": frame}),
               "raw_allocator_only::development_walk_forward::10")
    assert out["status"] == "ok"
    assert out["paired_vs_raw"]["status"] == "self_raw_reference"
    # a non-raw layer with a NaN numeric state field must fail closed
    bad = _frame(); bad["pretrade_beta"] = np.nan
    out2 = _one(R.analyze_risk_mechanism({"x::development_walk_forward::10": bad}),
                "x::development_walk_forward::10")
    assert out2["status"] == "needs_data" and "pretrade_beta" in out2["reason"]


def test_validate_returns_and_cash():
    frame = _frame(strategy_return=[-1.5] + [0.001] * 19)
    out = _one(R.analyze_risk_mechanism({"z::development_walk_forward::10": frame}),
               "z::development_walk_forward::10")
    assert out["status"] == "needs_data" and "le_minus_one" in out["reason"]
    frame2 = _frame(ending_cash_weight=[1.5] + [0.5] * 19)
    out2 = _one(R.analyze_risk_mechanism({"z::development_walk_forward::10": frame2}),
                "z::development_walk_forward::10")
    assert out2["status"] == "needs_data" and "cash" in out2["reason"]


def test_validate_bool_strings_rejected():
    frame = _frame()
    frame["emergency_active"] = "False"  # string must not be coerced truthy
    out = _one(R.analyze_risk_mechanism({"z::development_walk_forward::10": frame}),
               "z::development_walk_forward::10")
    assert out["status"] == "needs_data" and "nonbool_flag" in out["reason"]


def test_paired_by_state_and_phase_groups():
    raw = _frame(strategy_return=[0.0] * 20)
    cand = _frame(strategy_return=[0.01] * 20,
                  risk_state_before=[0] * 10 + [3] * 10,
                  emergency_active=[True] * 5 + [False] * 15)
    out = _one(R.analyze_risk_mechanism({
        "raw_allocator_only::development_walk_forward::10": raw,
        "candidate::development_walk_forward::10": cand,
    }), "candidate::development_walk_forward::10")
    paired = out["paired_vs_raw"]
    assert paired["status"] == "ok"
    assert set(paired["by_risk_state_before"]) == {"0", "3"}
    assert paired["by_phase"]["emergency_active"]["true"] == pytest.approx(5 * np.log1p(0.01))
    assert "phase_soxx" in paired


def test_grouped_case_list_shape():
    out = R.analyze_risk_mechanism({
        "raw_allocator_only::development_walk_forward::10": _frame(),
        "full_withhold_and_recovery::development_walk_forward::10": _frame(),
        "no_hold::outer_reused_diagnostic::25": _frame(),
    })
    assert isinstance(out, list) and len(out) == 3
    for row in out:
        assert {"case", "layer", "segment", "cost_bps", "status"} <= set(row)

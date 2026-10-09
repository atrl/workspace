from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_economic_action import ECONOMIC_FEATURES
from src.workflow import stage1_economic_training as sealed
from src.workflow import economic_cost_matched_training as matched


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


# --- parity: label25 reproduces the sealed control exactly ---

def test_label25_reproduces_sealed_control_replay_and_predictions():
    frame = _frame()
    control = sealed.train_causal_economic_models(frame, cost_bps=10.0, **_kwargs())
    book = matched.train_cost_matched_book(frame, label_cost_bps=25.0, evaluation_cost_bps=10.0, **_kwargs())
    # Replay returns and decisions identical.
    pd.testing.assert_series_equal(control["replay"]["strategy_return"], book["replay"]["strategy_return"])
    pd.testing.assert_series_equal(control["replay"]["execute_open_trade"], book["replay"]["execute_open_trade"])
    # Training labels identical (label fee is the same 25bp).
    c_lab = control["training_rows"].sort_values("signal_date").reset_index(drop=True)
    b_lab = book["training_rows"].sort_values("signal_date").reset_index(drop=True)
    np.testing.assert_allclose(c_lab["execute_advantage_bps"].to_numpy(float),
                               b_lab["execute_advantage_bps"].to_numpy(float), rtol=0, atol=1e-9)
    # Frozen annual models identical.
    for year in sorted(k for k in control["models"] if control["models"][k] is not None):
        np.testing.assert_allclose(
            control["models"][year].named_steps["ridge"].coef_,
            book["models"][year].named_steps["ridge"].coef_, rtol=0, atol=1e-12)


# --- label fee unit change ---

def test_label_fee_changes_labels_and_is_declared_per_book():
    frame = _frame()
    book10 = matched.train_cost_matched_book(frame, label_cost_bps=10.0, evaluation_cost_bps=10.0, **_kwargs())
    book25 = matched.train_cost_matched_book(frame, label_cost_bps=25.0, evaluation_cost_bps=25.0, **_kwargs())
    assert book10["dataset_audit"]["label_cost_bps"] == 10.0
    assert book25["dataset_audit"]["label_cost_bps"] == 25.0
    l10 = book10["training_rows"].sort_values("signal_date")["execute_advantage_bps"].to_numpy(float)
    l25 = book25["training_rows"].sort_values("signal_date")["execute_advantage_bps"].to_numpy(float)
    # A different reset fee must change the label magnitudes.
    assert not np.allclose(l10, l25, rtol=0, atol=1e-12)


def test_label_cost_is_explicit_and_global_is_untouched():
    # Guard: the sealed module's global must not be changed by the wrapper.
    before = sealed.LABEL_COST_BPS
    matched.train_cost_matched_book(_frame(4), label_cost_bps=10.0, evaluation_cost_bps=10.0, **_kwargs())
    assert sealed.LABEL_COST_BPS == before == 25.0


def test_rejects_negative_label_fee():
    with pytest.raises(ValueError, match="label_cost_bps"):
        matched.train_cost_matched_book(_frame(4), label_cost_bps=-1.0, **_kwargs())


# --- next-reset cost is accounted ---

def test_next_reset_fee_reduces_label_when_target_moves():
    frame = _frame()
    book = matched.train_cost_matched_book(frame, label_cost_bps=25.0, evaluation_cost_bps=10.0, **_kwargs())
    rows = book["training_rows"].dropna(subset=["execute_advantage_bps"])
    assert (rows["execute_reset_fee"] >= 0).all()
    assert (rows["hold_reset_fee"] >= 0).all()
    # The label is exactly the branch net difference adjusted by reset fees.
    lhs = rows["execute_advantage_bps"].to_numpy(float)
    rhs = 1e4 * (rows["execute_net_return"].to_numpy(float) - rows["hold_net_return"].to_numpy(float)
                 - rows["execute_reset_fee"].to_numpy(float) + rows["hold_reset_fee"].to_numpy(float))
    np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-6)


# --- two fee books do not share state ---

def test_two_fee_books_are_independent_state():
    frame = _frame()
    book10 = matched.train_cost_matched_book(frame, label_cost_bps=10.0, evaluation_cost_bps=10.0, **_kwargs())
    book25 = matched.train_cost_matched_book(frame, label_cost_bps=25.0, evaluation_cost_bps=25.0, **_kwargs())
    # A 25bp book replayed at 25bp differs from the 10bp book replayed at 10bp.
    assert not np.allclose(book10["replay"]["strategy_return"].to_numpy(float),
                           book25["replay"]["strategy_return"].to_numpy(float))
    # And the state/decisions are not carried across (separate model dicts).
    assert book10["models"] is not book25["models"]


# --- exact HOLD session charges zero open fee ---

def test_hold_sessions_charge_zero_open_fee():
    frame = _frame()
    book = matched.train_cost_matched_book(frame, label_cost_bps=25.0, evaluation_cost_bps=10.0, **_kwargs())
    holds = book["replay"][~book["replay"]["execute_open_trade"]]
    # A session with no open trade has zero open turnover and zero open fee.
    if len(holds):
        assert (holds["open_turnover"].abs() < 1e-12).all()


# --- maturity cutoff: future returns cannot change a prior fit ---

def test_future_year_returns_cannot_change_prior_fit_in_cost_book():
    frame = _frame()
    first = matched.train_cost_matched_book(frame, label_cost_bps=25.0, evaluation_cost_bps=10.0, **_kwargs())
    altered = frame.copy()
    altered.loc[altered.signal_date.dt.year.eq(2017), ["soxl_return", "soxx_return"]] = [.2, .1]
    second = matched.train_cost_matched_book(altered, label_cost_bps=25.0, evaluation_cost_bps=10.0, **_kwargs())
    assert first["final_model_year"] == 2017
    np.testing.assert_allclose(
        first["models"][2017].named_steps["ridge"].coef_,
        second["models"][2017].named_steps["ridge"].coef_, rtol=0, atol=1e-12)


def test_rejects_unregistered_window():
    with pytest.raises(ValueError, match="preregistered"):
        matched.train_cost_matched_book(_frame(4), label_cost_bps=25.0, update_through_year=2023, **_kwargs())


def test_rejects_nonfinite_or_negative_fees():
    for bad_label in (float("inf"), float("nan"), -1.0):
        with pytest.raises(ValueError, match="label_cost_bps"):
            matched.train_cost_matched_book(_frame(4), label_cost_bps=bad_label, **_kwargs())
    for bad_eval in (float("inf"), float("nan"), -5.0):
        with pytest.raises(ValueError, match="evaluation_cost_bps"):
            matched.train_cost_matched_book(_frame(4), label_cost_bps=10.0,
                                            evaluation_cost_bps=bad_eval, **_kwargs())

"""Structural risk-cap, causality, funded-integer and native-Qlib contracts."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.legacy_risk_budget import (
    build_legacy_risk_targets, replay_target_weights, run_native_target_replay,
)
from src.data.qlib_native import build_qlib_store_from_frame


def shadow_fixture():
    dates = pd.bdate_range("2024-01-01", periods=70)
    move = np.sin(np.arange(70)) * .01 + .001
    closes = pd.DataFrame({"SOXL": 100 * np.cumprod(1 + 3 * move),
                           "SOXX": 100 * np.cumprod(1 + move)}, index=dates)
    execution, signal = dates[61:], dates[60:-1]
    decisions = pd.DataFrame({"signal_date": signal, "planned_execution_date": execution,
        "signal_ready": True, "should_buy": True, "sell_type": None,
        "requested_shares": 3, "signal_account": 10000., "signal_exposure": .5})
    decisions.loc[1, ["should_buy", "sell_type"]] = [False, "normal_sell"]
    final = decisions.iloc[-1].copy()
    final.signal_date = dates[-1]
    final.planned_execution_date = pd.NaT
    decisions = pd.concat([decisions, final.to_frame().T], ignore_index=True)
    legacy = SimpleNamespace(daily=pd.DataFrame({"date": execution,
        "close": closes.loc[execution, "SOXL"].to_numpy()}), decisions=decisions)
    return legacy, closes


def test_intended_weights_include_new_prior_close_request_and_causal_vol_cap():
    legacy, closes = shadow_fixture()
    out = build_legacy_risk_targets(legacy, closes)
    expected = .5 + 3 * closes.loc[out.signal_date.iloc[0], "SOXL"] / 10000.
    assert out.raw_weight.iloc[0] == pytest.approx(expected)
    assert out.raw_weight.iloc[1] == pytest.approx(.5 - 3 * closes.loc[out.signal_date.iloc[1], "SOXL"] / 10000.)
    # Synthetic returns have an exact 3:1 volatility ratio, independent of data level.
    np.testing.assert_allclose(out.risk_cap, 1 / 3, atol=1e-14, rtol=0)
    np.testing.assert_allclose(out.capped_weight, 1 / 3, atol=1e-14, rtol=0)
    assert out.signal_date.lt(out.index).all() and len(out) == len(legacy.daily)
    # Change only data later than an evaluated prior-close signal: prior targets cannot move.
    changed = closes.copy()
    cutoff = out.signal_date.iloc[3]
    changed.loc[changed.index > cutoff, "SOXX"] *= 1.4
    revised = build_legacy_risk_targets(legacy, changed)
    pd.testing.assert_frame_equal(out.loc[out.signal_date <= cutoff], revised.loc[revised.signal_date <= cutoff])


@pytest.mark.parametrize("bad", ["missing_session", "same_day", "immature_sigma", "zero_sigma", "wrong_close", "fractional_shares"])
def test_target_builder_fails_closed_on_bad_source_or_timing(bad):
    legacy, closes = shadow_fixture()
    if bad == "missing_session": legacy.decisions = legacy.decisions.drop(index=1)
    elif bad == "same_day": legacy.decisions.loc[0, "signal_date"] = legacy.decisions.loc[0, "planned_execution_date"]
    elif bad == "immature_sigma": closes = closes.iloc[10:]
    elif bad == "zero_sigma": closes["SOXX"] = 100.
    elif bad == "wrong_close": legacy.daily.loc[0, "close"] += 1
    elif bad == "fractional_shares": legacy.decisions.loc[0, "requested_shares"] = 1.5
    with pytest.raises(ValueError, match="blocked:"):
        build_legacy_risk_targets(legacy, closes)


@pytest.fixture
def native_prices(tmp_path):
    dates = pd.bdate_range("2025-01-01", periods=8)
    market = pd.DataFrame({"open": 100. + np.arange(8), "close": 101. + np.arange(8)}, index=dates)
    rows = [{"datetime": date, "instrument": symbol, "open": float(row.open),
        "close": float(row.close), "high": float(row.close + 1), "low": float(row.open - 1),
        "volume": 1e7, "factor": 1.} for symbol in ("SOXL", "SOXX") for date, row in market.iterrows()]
    build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=tmp_path / "provider",
        as_of=str(dates[-1].date()), source="synthetic risk-budget native accounting")
    targets = pd.DataFrame({"signal_date": dates[:4], "raw_weight": [.5, 0., 1., .2]}, index=dates[1:5])
    return market, targets


@pytest.mark.parametrize("cost", [0., 10., 25.])
def test_native_target_replay_owns_fees_shares_and_continuous_nav(native_prices, cost):
    market, targets = native_prices
    result = run_native_target_replay(market, targets, weight_column="raw_weight", cost_bps=cost, initial_cash=1000.)
    assert result["audit"]["status"] == "passed"
    daily, trades = result["daily"], result["trades"]
    assert trades.amount.iloc[0] == 4 and trades.amount.iloc[1] == -4
    assert daily.shares.iloc[1] == 0 and (daily.cash >= 0).all()
    assert (trades.amount == np.trunc(trades.amount)).all()
    # A buy then sell at different opens must include both proportional fee legs.
    expected_cash = 1000. - 4 * 101 * (1 + cost / 10000.) + 4 * 102 * (1 - cost / 10000.)
    assert daily.cash.iloc[1] == pytest.approx(expected_cash)
    assert trades.signal_date.lt(trades.date).all()
    assert daily.shares.iloc[-1] > 0  # no fabricated terminal liquidation
    assert result["native"]["daily"].index.equals(targets.index)


def test_zero_shadow_target_keeps_cash_and_does_not_invent_soxx_fallback(native_prices):
    market, targets = native_prices
    targets.raw_weight = 0.
    result = run_native_target_replay(market, targets, weight_column="raw_weight", cost_bps=25, initial_cash=1000)
    assert result["trades"].empty
    assert result["daily"].account.eq(1000).all()
    assert result["native"]["daily"].net_return.eq(0).all()


def test_integer_target_error_is_bounded_without_negative_funding(native_prices):
    market, targets = native_prices
    result = replay_target_weights(market, targets, weight_column="raw_weight", cost_bps=25, initial_cash=1000)
    for row in result.daily.itertuples():
        opening = market.loc[row.date, "open"]
        nav_open = row.cash + row.shares * opening
        assert abs(row.open_exposure_after_cost - row.open_target_weight) <= opening / nav_open * 1.01


def test_missing_or_same_day_target_session_is_rejected(native_prices):
    market, targets = native_prices
    with pytest.raises(ValueError, match="every native session"):
        replay_target_weights(market, targets.iloc[[0, 2, 3]], weight_column="raw_weight", cost_bps=25)
    targets.loc[targets.index[1], "signal_date"] = targets.index[1]
    with pytest.raises(ValueError, match="previous native session"):
        replay_target_weights(market, targets, weight_column="raw_weight", cost_bps=25)

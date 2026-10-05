from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soxl_backtest.config.backtest_config import BacktestConfig
from soxl_backtest.position.position_manager import PositionManager
from soxl_backtest.strategy.buy_strategy import BuyStrategy
from src.backtest.legacy_soxl_replay import (
    LegacyDecisionEngine,
    LegacyOrderIntent,
    prepare_legacy_features,
    run_legacy_repaired,
)


def _prepared(prices, *, opens=None, panic=-100.0):
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=len(prices)),
        "price": prices, "close": prices, "open": opens or prices,
        "vix": 20.0, "MA20": 1000.0, "VOL_COEF": 1.0,
        "panic_score": panic, "signal_ready": True,
    })


def _raw(prices):
    result = _prepared(prices)
    return result[["date", "open", "close", "vix"]]


def test_legacy_initial_buy_does_not_read_last_row_as_previous_close():
    config = BacktestConfig({"VOLATILITY_MEDIAN": 0})
    for last_price in (50.0, 200.0):
        frame = _prepared([100.0, last_price])
        strategy = BuyStrategy(config, PositionManager(config))
        assert not strategy.should_buy(frame.iloc[0], frame, 0)
        assert strategy.cumulative_buy_cooldown == 0


def test_indicators_are_prefix_causal_and_require_actual_ma180_warmup():
    prices = list(np.exp(np.linspace(np.log(50), np.log(100), 185)))
    frame = _raw(prices)
    full = prepare_legacy_features(frame)
    prefix = prepare_legacy_features(frame.iloc[:182])
    pd.testing.assert_frame_equal(full.iloc[:182], prefix)
    assert not full.loc[:178, "signal_ready"].any()
    assert full.loc[179:, "signal_ready"].all()
    assert full.loc[179, "MA20"] == pytest.approx(np.mean(prices[:180]))


def test_each_decision_evaluates_buy_once_and_sell_only_if_no_buy(monkeypatch):
    engine = LegacyDecisionEngine(initial_cash=1000)
    frame = _prepared([100.0, 70.0])
    calls = {"buy": 0, "sell": 0}
    original_buy = engine.buy_strategy.should_buy
    original_sell = engine.sell_strategy.should_sell

    def buy(*args):
        calls["buy"] += 1
        return original_buy(*args)

    def sell(*args):
        calls["sell"] += 1
        return original_sell(*args)

    monkeypatch.setattr(engine.buy_strategy, "should_buy", buy)
    monkeypatch.setattr(engine.sell_strategy, "should_sell", sell)
    assert engine.decide(frame, 0)[0] is None
    intent, audit = engine.decide(frame, 1)
    assert intent is not None and audit["should_buy"]
    assert calls == {"buy": 2, "sell": 1}
    with pytest.raises(ValueError, match="already evaluated"):
        engine.decide(frame, 1)


@pytest.mark.parametrize("cost_bps", [0.0, 10.0, 25.0])
def test_frozen_close_request_executes_next_open_and_cash_funds_fee(cost_bps):
    engine = LegacyDecisionEngine(initial_cash=1000, cost_bps=cost_bps)
    frame = _prepared([100.0, 70.0, 60.0])
    engine.decide(frame, 0)
    intent, _ = engine.decide(frame, 1)
    assert intent.requested_shares == 4
    assert engine.position_manager.shares == 0
    assert engine.position_manager.capital == 1000
    fill = engine.settle(date=frame.iloc[2]["date"], open_price=400.0)
    assert fill["amount"] == 2
    assert fill["requested_shares"] == 4
    assert fill["price"] == 400.0
    assert fill["signal_date"] == frame.iloc[1]["date"]
    assert fill["fee"] == pytest.approx(800 * cost_bps / 10_000)
    assert fill["cash"] == pytest.approx(1000 - 800 - fill["fee"])
    assert fill["cash"] >= 0
    assert engine.settle(date=frame.iloc[2]["date"], open_price=400.0) is None


def test_next_open_does_not_recheck_signal_or_sell_trigger():
    engine = LegacyDecisionEngine(initial_cash=1000, cost_bps=10)
    frame = _prepared([100.0, 70.0, 150.0])
    engine.decide(frame, 0)
    engine.decide(frame, 1)
    fill = engine.settle(date=frame.iloc[2]["date"], open_price=150.0)
    assert fill["amount"] == 4
    assert fill["price"] != 70.0
    assert fill["side"] == "buy"


def test_weights_mark_daily_and_initial_build_uses_actual_position():
    engine = LegacyDecisionEngine(initial_cash=1000)
    frame = _prepared([100.0, 200.0], panic=0.0)
    engine.position_manager.capital = 600.0
    engine.position_manager.add_position(100.0, 4)
    engine.decide(frame, 0)
    assert engine.position_manager.position == pytest.approx(0.4)
    assert not engine.position_manager.initial_position_built
    engine.decide(frame, 1)
    assert engine.position_manager.position == pytest.approx(800 / 1400)
    assert engine.position_manager.initial_position_built


def test_lifo_lots_and_sale_cash_use_only_actual_held_shares():
    engine = LegacyDecisionEngine(initial_cash=1000, cost_bps=25)
    pm = engine.position_manager
    pm.capital = 800.0
    pm.add_position(10.0, 10)
    pm.add_position(20.0, 5)
    engine.pending = LegacyOrderIntent(pd.Timestamp("2020-01-01"), "sell", "normal_sell", 3, 30, 1250, .36, ())
    fill = engine.settle(date=pd.Timestamp("2020-01-02"), open_price=30.0)
    assert fill["amount"] == -3
    assert pm.position_stack == [{"price": 10.0, "shares": 10}, {"price": 20.0, "shares": 2}]
    assert pm.avg_price == pytest.approx(140 / 12)
    assert pm.capital == pytest.approx(800 + 90 * (1 - .0025))
    engine.pending = LegacyOrderIntent(pd.Timestamp("2020-01-02"), "sell", "stop_loss", 100, 30, 1250, .36, ())
    fill = engine.settle(date=pd.Timestamp("2020-01-03"), open_price=30.0)
    assert fill["amount"] == -12
    assert pm.shares == 0 and pm.position_stack == []
    assert pm.capital == pytest.approx(800 + 450 * (1 - .0025))


def test_buy_state_commits_once_only_on_actual_fill():
    frame = _prepared([100.0, 90.0, 80.0, 60.0])
    engine = LegacyDecisionEngine(initial_cash=1000, config_overrides={"DAILY_RETURN_THRESHOLD": -.5})
    for index in range(3):
        engine.decide(frame, index)
    intent, _ = engine.decide(frame, 3)
    assert intent is not None
    assert engine.buy_strategy.cumulative_buy_cooldown == 0
    assert engine.settle(date=pd.Timestamp("2020-01-07"), open_price=2000) is None
    assert engine.buy_strategy.cumulative_buy_cooldown == 0

    funded = LegacyDecisionEngine(initial_cash=1000, config_overrides={"DAILY_RETURN_THRESHOLD": -.5})
    for index in range(4):
        funded.decide(frame, index)
    fill = funded.settle(date=pd.Timestamp("2020-01-07"), open_price=60)
    assert fill is not None
    assert funded.buy_strategy.cumulative_buy_cooldown == 3
    assert funded.buy_strategy.last_buy_date is None  # preserve initial-branch economics
    assert funded.settle(date=pd.Timestamp("2020-01-07"), open_price=60) is None
    assert funded.buy_strategy.cumulative_buy_cooldown == 3


def test_full_replay_future_prefix_and_final_intent_are_causal():
    frame = _raw([100.0] * 180 + [70.0, 60.0, 50.0, 80.0, 100.0])
    frame.loc[181, "open"] = 90.0
    prefix = run_legacy_repaired(frame.iloc[:183], cost_bps=10)
    full = run_legacy_repaired(frame, cost_bps=10)
    pd.testing.assert_frame_equal(prefix.daily, full.daily.iloc[:183].reset_index(drop=True))
    pd.testing.assert_frame_equal(prefix.trades, full.trades.loc[full.trades["date"] <= frame.iloc[182]["date"]].reset_index(drop=True))
    assert (full.trades["date"] > full.trades["signal_date"]).all()
    assert full.trades.iloc[0]["price"] == 90.0
    assert full.daily.iloc[180]["shares"] == 0
    assert full.daily.iloc[181]["shares"] > 0
    assert full.daily["cash"].min() >= 0
    pd.testing.assert_series_equal(full.daily["exposure"],
                                  (full.daily["shares"] * full.daily["close"] / full.daily["account"]).rename("exposure"))
    final_signal_only = run_legacy_repaired(frame.iloc[:181], cost_bps=10)
    assert final_signal_only.trades.empty
    assert final_signal_only.decisions.iloc[-1]["status"] == "pending"


def test_input_missing_vix_duplicate_calendar_and_invalid_price_fail_closed():
    frame = _raw([100.0] * 181)
    frame.loc[180, "vix"] = np.nan
    with pytest.raises(ValueError, match="missing availability-aligned VIX"):
        run_legacy_repaired(frame)
    frame.loc[180, "vix"] = 20.0
    with pytest.raises(ValueError, match="unique and increasing"):
        prepare_legacy_features(pd.concat([frame, frame.iloc[[-1]]]))
    frame.loc[0, "open"] = 0.0
    with pytest.raises(ValueError, match="invalid legacy open"):
        prepare_legacy_features(frame)


def test_explicit_start_seeds_only_prior_close_and_can_fill_first_open():
    frame = _raw([100.0] * 180 + [70.0, 60.0, 50.0])
    frame.loc[181, "open"] = 90.0
    start = frame.iloc[181]["date"]
    result = run_legacy_repaired(frame, start=str(start.date()), cost_bps=10)
    assert result.daily.iloc[0]["date"] == start
    assert result.daily.iloc[0]["shares"] == int(30_000 / 70)
    assert result.trades.iloc[0]["date"] == start
    assert result.trades.iloc[0]["signal_date"] == frame.iloc[180]["date"]
    assert result.trades.iloc[0]["price"] == 90.0
    assert (result.trades["date"] >= start).all()
    assert result.decisions.iloc[0]["initial_signal_only"]
    assert result.decisions.iloc[0]["status"] == "filled"
    assert result.decisions["initial_signal_only"].sum() == 1
    assert result.daily.iloc[0]["net_return"] == pytest.approx(result.daily.iloc[0]["account"] / 100_000 - 1)


def test_explicit_start_without_ready_prior_close_fails_closed():
    frame = _raw([100.0] * 180)
    with pytest.raises(ValueError, match="requires a previous canonical close"):
        run_legacy_repaired(frame, start=str(frame.iloc[0]["date"].date()))
    with pytest.raises(ValueError, match="previous-close indicators or VIX not ready"):
        run_legacy_repaired(frame, start=str(frame.iloc[179]["date"].date()))

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qlib.backtest.account import Account
from qlib.backtest.position import Position

from src.analysis.convex_qlib import native_open_reconciliation
from src.analysis.integrated_allocation import native_allocation_reconciliation
from src.backtest.qlib_continuous_position import QlibContinuousPosition


def _position(cls=QlibContinuousPosition, *, amount=3150.124706141618):
    return cls(cash=100., position_dict={"SOXL": {"amount": amount, "price": 45.949546813964844}})


def test_native_default_deletes_real_residual_but_continuous_position_preserves_it():
    held, residual, price = 3150.124706141618, .0015480739584699961, 45.949546813964844
    sold = held - residual
    value, cost = sold * price, sold * price * .001
    original, precise = _position(Position), _position()
    original._sell_stock("SOXL", value, cost, price)
    precise._sell_stock("SOXL", value, cost, price)
    assert not original.check_stock("SOXL")  # Reproduces the installed Qlib defect.
    assert precise.get_stock_amount("SOXL") == pytest.approx(residual, abs=1e-12)
    assert isinstance(precise.get_stock_amount("SOXL"), np.float64)
    assert precise.get_cash() == pytest.approx(100 + value - cost)
    # At the actual next open, the default drops $0.0692393357 of real holdings.
    marked_difference = precise.get_stock_amount("SOXL") * 44.72611618041992
    assert marked_difference == pytest.approx(.06923933572241162, abs=1e-10)


def test_every_representable_positive_residual_remains_owned():
    position = _position(amount=1.)
    residual = math.ulp(1.)
    position._sell_stock("SOXL", 1. - residual, 0., 1.)
    assert position.check_stock("SOXL")
    assert position.get_stock_amount("SOXL") == residual


def test_true_liquidation_and_only_machine_roundoff_oversell_close():
    for sold in (1., np.nextafter(1., np.inf)):
        position = _position(amount=1.)
        position._sell_stock("SOXL", sold, 0., 1.)
        assert not position.check_stock("SOXL")
    position = _position(amount=1.)
    with pytest.raises(ValueError, match="only have"):
        position._sell_stock("SOXL", 1.00000001, 0., 1.)
    assert position.get_stock_amount("SOXL") == 1. and position.get_cash() == 100.


def test_division_overflow_fails_before_cash_or_stock_changes():
    position = _position(amount=1.)
    with pytest.raises(ValueError, match="nonfinite.*quantity"):
        position._sell_stock("SOXL", 1e308, 0., 1e-308)
    assert position.get_stock_amount("SOXL") == 1. and position.get_cash() == 100.


def test_delayed_cash_settlement_is_still_native():
    position = _position(amount=10.)
    position.settle_start(Position.ST_CASH)
    position._sell_stock("SOXL", 40., 1., 10.)
    assert position.get_stock_amount("SOXL") == 6.
    assert position.get_cash() == 100.
    assert position.get_cash(include_settle=True) == 139.
    position.settle_commit()
    assert position.get_cash() == 139.


def test_native_account_accepts_custom_position_without_global_patching():
    account = Account(init_cash=100., pos_type=QlibContinuousPosition, port_metr_enabled=False)
    assert isinstance(account.current_position, QlibContinuousPosition)
    assert account.current_position.__class__ is not Position


def test_native_backtest_retains_tiny_target_and_keeps_existing_tolerance():
    import qlib
    from qlib.data import D
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    provider = Path(__file__).resolve().parents[1] / "data/qlib/stable_excess_integration_20261002"
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    dates = pd.DatetimeIndex(D.calendar(start_time="2021-09-02", end_time="2021-09-09", freq="day"))
    opens = D.features(["SOXL", "SOXX"], ["$open"], start_time=dates[0], end_time=dates[-1], freq="day")["$open"].unstack("instrument").astype(float)
    current = np.array([0., 0., 1.]); market, path = [], []
    for i, target in enumerate((np.array([.33, 0., .67]), np.array([1.6557873226053374e-7, 0., 1 - 1.6557873226053374e-7]))):
        trade, end = dates[i + 1], dates[i + 2]
        returns = (opens.loc[end] / opens.loc[trade] - 1).to_numpy()
        executed = fee_affordable_target(target, pretrade_weights=current, cost_bps=10.)
        settled = settle_target_session(current, open_target_weights=executed, asset_returns=np.r_[returns, 0.], cost_bps=10.)
        row = {"signal_date": dates[i], "trade_date": trade, "exit_date": end,
               "soxl_return": returns[0], "soxx_return": returns[1]}
        market.append(row)
        path.append({**row, "strategy_return": settled["net_return"], "turnover": settled["turnover"],
                     "fees": settled["fees"], "cost_bps": 10., "execute_open_target": True,
                     **dict(zip(("soxl_weight", "soxx_weight", "cash_weight"), target))})
        current = settled["ending_weights"]
    args = (pd.DataFrame(market), pd.DataFrame(path))
    with pytest.raises(ValueError, match="native Qlib allocation reconciliation failed"):
        native_allocation_reconciliation(*args, provider_uri=provider, cost_bps=10.)
    result = native_allocation_reconciliation(*args, provider_uri=provider, cost_bps=10., native_reconciler=native_open_reconciliation)
    assert result["audit"]["tolerance"] == 1e-7
    assert result["audit"]["maximum_session_return_error"] < 1e-12
    assert result["positions"][dates[2]].get_stock_amount("SOXL") > 0
    assert result["audit"]["sealed_stage1_code_changed"] is False

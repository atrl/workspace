from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.native_raw_corporate import native_raw_validate
from src.backtest.raw_corporate_policy import make_schedule
from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
from src.data import build_qlib_store_from_frame
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec


class ConstantProbability:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, features):
        return np.tile([1 - self.probability, self.probability], (len(features), 1))


def _case(count):
    days = pd.bdate_range("2023-01-04", periods=count + 1)
    frame = pd.DataFrame({"signal_date": pd.bdate_range("2023-01-03", periods=count),
                          "trade_date": days[:-1], "exit_date": days[1:]})
    for name, value in {"soxl_weight": .1, "soxx_weight": .4, "cash_weight": .5,
                        "tail_probability": .1, "alpha_probability": .8}.items():
        frame[name] = value
    for asset in ("soxl", "soxx"):
        for field in ("ret", "range", "from_low", "volume"):
            frame[f"early_{field}_{asset}"] = 0.
    stamps = [day + pd.Timedelta(clock) for day in days[:-1]
              for clock in ("9h45min", "9h47min", "10h32min", "16h")]
    stamps += [days[-1] + pd.Timedelta(clock) for clock in ("9h45min", "9h47min")]
    points = pd.DataFrame(10., index=pd.DatetimeIndex(stamps), columns=["SOXL", "SOXX"])
    tariff = {"tiers": [{"through_shares": None, "per_share": 0.}],
              "minimum_per_order": 0., "maximum_order_fraction": 1.,
              "venue_per_share": 0., "venue_subdollar_fraction": 0.,
              "clearing_per_share": 0., "clearing_maximum_fraction": 0.,
              "sec_sale_fraction": 0., "taf_per_sold_share": 0.,
              "taf_maximum_per_trade": 0., "cat_per_share": 0.,
              "commission_pass_through_fraction": 0.}
    return frame, points, {"initial_cash": 1000., "tariff": tariff, "slippage_bps": [0, 1000]}


def _kwargs(plan, points, *, opening_probability=.01, **risk):
    return dict(point_prices=points, schedule={}, plan=plan,
                controller={"frozen_open_action_value_model": ConstantProbability(opening_probability),
                            "frozen_recovery_model": ConstantProbability(1.)},
                recovery_spec=IntradayRecoveryDirectAllocationSpec(root=Path.cwd(), **risk),
                action_spec=Stage1OpenActionValueSpec(root=Path.cwd()),
                selection={"action_beta": 1.2, "minimum_source_alpha_probability": .5,
                           "probability_threshold": .5})


def test_stage1_hold_and_cost_specific_risk_recovery_are_preserved():
    frame, points, plan = _case(2)
    frame.loc[1, ["soxl_weight", "cash_weight"]] = [.11, .49]
    kwargs = _kwargs(plan, points, stop_drawdown=.02, recovery_eligibility_drawdown=.01)
    low = replay_stage1_raw(frame, **kwargs, slippage_bps=0)
    high = replay_stage1_raw(frame, **kwargs, slippage_bps=1000)

    # Run the real Stage1 decision functions: a small upward target change is
    # held at zero cost, while the other account's own fee loss causes derisking
    # and its own recovery vote.  No COMMON cap or ALL3 vote can produce this.
    assert low["replay"].execute_open_target.tolist() == [True, False]
    assert low["replay"].recovery_action.tolist() == [False, False]
    assert high["replay"].execute_open_target.tolist() == [True, True]
    assert high["replay"].recovery_action.tolist() == [False, True]
    assert low["replay"].iloc[1].current_beta_cap == pytest.approx(.73)
    assert high["replay"].iloc[1].current_beta_cap == pytest.approx(.2)
    # The simultaneous SOXL and cash target survives; no beta-preserving T is
    # silently substituted for the frozen opening/recovery policy.
    assert low["points"].iloc[1].q_soxl == 10
    assert high["orders"].slot.eq("recovery").any()
    assert low["orders"].trade_date.nunique() == 1
    again = replay_stage1_raw(frame, **kwargs, slippage_bps=0)
    pd.testing.assert_frame_equal(low["replay"], again["replay"])
    pd.testing.assert_frame_equal(low["orders"], again["orders"])


def test_raw_dividend_funding_split_and_native_bridge(tmp_path):
    frame, points, plan = _case(3)
    frame[["soxl_weight", "soxx_weight", "cash_weight"]] = [0., 1., 0.]
    days = pd.DatetimeIndex([*frame.trade_date, frame.exit_date.iloc[-1]])
    points.loc[points.index.normalize() >= days[1], "SOXX"] = 9.
    points.loc[points.index.normalize() >= days[2], "SOXX"] = 3.
    cash_events = pd.DataFrame([{"instrument": "SOXX", "event_id": "div",
                                 "ex_date": days[1], "pay_date": days[1],
                                 "original_total_decimal": "1"}])
    split_events = pd.DataFrame([{"instrument": "SOXX", "event_id": "split",
                                  "effective_trading_date": days[2], "new_shares_per_old_share": 3}])
    kwargs = _kwargs(plan, points, opening_probability=1., minimum_cash_weight=0.)
    kwargs["schedule"] = make_schedule(cash_events, split_events, days, days[0], days[-1])
    path = replay_stage1_raw(frame, **kwargs, slippage_bps=0)
    marks = path["points"].set_index("stamp")
    entitlement = marks.loc[days[1] + pd.Timedelta("9h47min")]
    payment = marks.loc[days[2] + pd.Timedelta("9h47min")]
    assert entitlement.receivable == 100 and entitlement.cash == 0 and entitlement.q_soxx == 100
    assert not path["orders"].trade_date.eq(days[1]).any()  # R cannot fund requested buys.
    assert payment.receivable == 0 and payment.cash == 1 and payment.q_soxx == 333
    np.testing.assert_allclose(path["replay"].equity, 1., rtol=0, atol=1e-12)

    # Synthetic provider lives only in pytest's temporary directory.  Keep one
    # following native calendar point out of the policy grid and all metrics.
    guarded = pd.concat([points, pd.DataFrame([points.iloc[-1].to_dict()],
                        index=[points.index[-1] + pd.Timedelta(minutes=45)])])
    fields = ("open", "high", "low", "close", "execution_price", "raw_execution_price")
    raw = [{"datetime": stamp, "instrument": symbol, **{field: value for field in fields}}
           for symbol in points for stamp, value in guarded[symbol].items()]
    provider = tmp_path / "provider"
    build_qlib_store_from_frame(pd.DataFrame(raw), provider_uri=provider, freq="1min", region="us",
                               as_of=str(days[-1].date()), field_map={name: name for name in fields},
                               source="synthetic Stage1 raw native bridge regression")
    native = native_raw_validate(path, provider_uri=provider, point_prices=points,
                                 schedule=kwargs["schedule"], tariff=plan["tariff"],
                                 cost=0, initial_cash=plan["initial_cash"])
    assert native["audit"]["status"] == "passed"
    assert native["audit"]["max_relative_error"] < 1e-10
    assert len(native["native_corporate_receipts"]) == 3
    np.testing.assert_allclose(native["authority"].equity, 1., rtol=0, atol=1e-12)

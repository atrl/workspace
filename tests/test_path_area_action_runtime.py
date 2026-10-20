"""Prefix timing and per-account feature binding through a real native replay."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.path_area_action import replay_path_area_action
from src.data.path_area_evaluation import prefix_inputs_from_marks
from src.model.path_area_action import feature_order


def test_closed_prefix_and_cost_owned_action_context_reconcile_natively(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.data import build_qlib_store_from_frame
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec

    days = pd.date_range("2020-01-06", periods=3, freq="D")
    frame = pd.DataFrame({"signal_date": days[:2] - pd.Timedelta(days=1),
                          "trade_date": days[:2], "exit_date": days[1:],
                          "soxl_weight": .2, "soxx_weight": .65, "cash_weight": .15,
                          "tail_probability": .2, "alpha_probability": .8,
                          "early_ret_soxx": .01})
    rows = []
    for day in days[:2]:
        for j in range(15):
            for symbol in ("SOXL", "SOXX"):
                rows.append({"datetime": day + pd.Timedelta(hours=9, minutes=31+j),
                             "instrument": symbol, "$open": 10.,
                             "$close": 10. + j * (.01 if symbol == "SOXL" else .02),
                             "$price_valid": 1.})
    marks = pd.DataFrame(rows).set_index(["datetime", "instrument"]).sort_index()
    # One missing early bar remains a masked row, independent of later coverage.
    marks = marks.drop((days[1] + pd.Timedelta(hours=9, minutes=33), "SOXX"))
    prefix, _ = prefix_inputs_from_marks(marks, frame)
    later = pd.DataFrame([[99999., 99999., 1.]],
        index=pd.MultiIndex.from_tuples([(days[1] + pd.Timedelta(hours=16), "SOXX")],
                                       names=marks.index.names), columns=marks.columns)
    observed, _ = prefix_inputs_from_marks(pd.concat([marks, later]).sort_index(), frame)
    pd.testing.assert_frame_equal(prefix, observed)
    assert prefix.prefix_mask.tolist() == [1, 0]
    assert prefix.iloc[1].path_area_signed != prefix.iloc[1].path_area_signed

    stamps = [day + pd.Timedelta(clock) for day in days[:2]
              for clock in ("9h45min", "9h47min", "10h32min", "16h")]
    stamps += [days[-1] + pd.Timedelta(clock) for clock in ("9h45min", "9h47min")]
    prices = pd.DataFrame([[10. + (stamp.normalize()-days[0]).days]*2 for stamp in stamps],
                          index=pd.DatetimeIndex(stamps), columns=["SOXL", "SOXX"])
    tariff = {"tiers": [{"through_shares": None, "per_share": .0035}],
              "minimum_per_order": .35, "maximum_order_fraction": 1.,
              **{key: 0. for key in ("venue_per_share", "venue_subdollar_fraction",
                  "clearing_per_share", "clearing_maximum_fraction", "sec_sale_fraction",
                  "taf_per_sold_share", "taf_maximum_per_trade", "cat_per_share",
                  "commission_pass_through_fraction")}}

    class FixedProbability:
        def __init__(self):
            self.seen = []
        def predict_proba(self, values):
            assert tuple(values.columns) == feature_order()
            self.seen.append(values.copy())
            return np.array([[0., 1.]])

    models = [FixedProbability(), FixedProbability()]
    paths = []
    attempts = []
    for cost, model in zip((0, 5), models):
        paths.append(replay_path_area_action(frame, prefix=prefix, pipeline=model, arm="C",
            point_prices=prices, schedule={}, plan={"initial_cash": 100000., "tariff": tariff},
            controller={"frozen_recovery_model": None}, slippage_bps=cost,
            recovery_spec=IntradayRecoveryDirectAllocationSpec(root=tmp_path),
            action_spec=Stage1OpenActionValueSpec(root=tmp_path),
            selection={"action_beta": 1.2, "minimum_source_alpha_probability": .5,
                       "probability_threshold": .5},
            on_prediction_attempt=lambda: attempts.append(True)))
    assert len(attempts) == 4 and all(path["action_prediction_calls"] == 2 for path in paths)
    # The second day's endogenous state reflects its own previous fees.
    assert models[0].seen[1].pretrade_cash_weight.iloc[0] != models[1].seen[1].pretrade_cash_weight.iloc[0]
    assert all(len(model.seen) == 2 for model in models)
    with pytest.raises(ValueError, match="prefix and account days"):
        replay_path_area_action(frame, prefix=prefix.iloc[::-1], pipeline=models[0], arm="C",
            point_prices=prices, schedule={}, plan={}, controller={}, recovery_spec=None,
            action_spec=None, selection={}, slippage_bps=0)

    guard = prices.iloc[[-1]].copy()
    guard.index = pd.DatetimeIndex([prices.index[-1] + pd.Timedelta(minutes=1)])
    all_prices = pd.concat([prices, guard])
    fields = ("open", "high", "low", "close", "execution_price", "raw_execution_price")
    quote_rows = [{"datetime": stamp, "instrument": symbol,
                   **{name: float(value) for name in fields}}
                  for symbol in all_prices for stamp, value in all_prices[symbol].items()]
    provider = tmp_path / "native"
    build_qlib_store_from_frame(pd.DataFrame(quote_rows), provider_uri=provider,
        freq="1min", region="us", as_of="2020-01-08",
        field_map={name: name for name in fields}, source="synthetic P3 runtime regression")
    native = native_raw_validate(paths[1], provider_uri=provider, point_prices=prices,
                                 schedule={}, tariff=tariff, cost=5, initial_cash=100000.)
    assert native["audit"]["status"] == "passed"
    assert native["audit"]["max_relative_error"] <= 1e-7

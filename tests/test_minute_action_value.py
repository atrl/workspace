import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees
from src.backtest.minute_action_value import next_reset_advantage
from src.data.minute_action_marks import first_hour_features
from src.model.minute_action_value import MinuteActionValue
from src.workflow.minute_action_value import _dataset


def test_completed_hour_features_ignore_future_prices_and_refuse_missing_bar():
    stamps = pd.date_range("2023-01-04 09:31", periods=70, freq="1min")
    index = pd.MultiIndex.from_product([["SOXL", "SOXX"], stamps], names=["instrument", "datetime"])
    values = np.tile(np.linspace(100, 101, 70), 2)
    raw = pd.DataFrame({"$open": values, "$close": values * 1.00001, "$volume": 100.}, index=index)
    actual = first_hour_features(raw)
    changed = raw.copy()
    changed.loc[(slice(None), slice(pd.Timestamp("2023-01-04 10:31"), None)), ["$open", "$close"]] *= 3
    pd.testing.assert_frame_equal(actual, first_hour_features(changed))
    assert actual.feature_available_at.iloc[0] == pd.Timestamp("2023-01-04 10:31")
    broken = raw.drop(("SOXX", pd.Timestamp("2023-01-04 10:00")))
    assert first_hour_features(broken).empty
    invalid = raw.copy()
    invalid["$price_valid"] = 1.
    invalid.loc[("SOXX", pd.Timestamp("2023-01-04 10:00")), "$price_valid"] = 0.
    assert first_hour_features(invalid).empty


def test_paired_label_contains_overnight_return_and_both_reset_fees_without_mutation():
    tariff = json.loads((Path(__file__).resolve().parents[1] / "research/ibkr_tiered_minute_diagnosis_20261013/fee_schedule.json").read_text())["tariff"]
    context = {"quantities": np.array([0., 100.]), "cash": 10000., "desired": np.array([.1, .4, .5]),
               "known_nav": 20000., "fee_state": TieredFees(tariff, slippage_bps=2)}
    row = pd.Series({"trade_date": pd.Timestamp("2023-01-04"), "exit_date": pd.Timestamp("2023-01-05"),
        "soxl_action": 50., "soxx_action": 100., "raw_soxl_action": 50., "raw_soxx_action": 100.,
        "soxl_exit_open": 62.5, "soxx_exit_open": 105., "raw_soxl_exit_open": 62.5, "raw_soxx_exit_open": 105.})
    reset = np.array([0., .5, .5])
    result = next_reset_advantage(context, row, reset)
    assert 0 < result < .02  # gross advantage is exactly $400 before costs
    changed = row.copy()
    changed[["soxl_exit_open", "raw_soxl_exit_open"]] = 40.
    assert next_reset_advantage(context, changed, reset) < 0
    hold = deepcopy(context)
    hold["desired"] = np.array([0., .5, .5])
    assert next_reset_advantage(hold, row, reset) == 0
    assert context["fee_state"].monthly_shares == {}
    np.testing.assert_array_equal(context["quantities"], [0., 100.])


def test_qlib_fit_requires_strict_label_maturity_and_train_only_scaling():
    dates = pd.date_range("2023-01-04 10:31", periods=42, freq="1D")
    teacher = pd.DataFrame({"state": np.linspace(-1, 1, 42), "net_advantage": np.linspace(-.01, .01, 42),
        "label_available_at": dates + pd.Timedelta(days=1)}, index=dates)
    teacher.loc[dates[-2:], "label_available_at"] = pd.to_datetime(["2024-01-01", "2024-01-02"])
    contaminated = teacher.copy()
    contaminated.loc[dates[-2:], ["state", "net_advantage"]] = 1e8
    models = [MinuteActionValue(features=["state"], cutoff="2024-01-01", alpha=10, minimum_observations=10, hac_days=5)
              .fit(_dataset(data, ["state"], "2024-01-01")) for data in (teacher, contaminated)]
    assert models[0].audit["observations"] == 40
    np.testing.assert_array_equal(models[0].scaler.mean_, models[1].scaler.mean_)
    pd.testing.assert_frame_equal(models[0].predict_frame(teacher.iloc[:2]), models[1].predict_frame(teacher.iloc[:2]))

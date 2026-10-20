"""One concentrated synthetic contract for the P2 path-area/action bridge."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event
from src.backtest.path_area_teacher import (
    clocked_smooth_targets,
    label_local_actions,
    run_training_shadow,
)
from src.backtest.raw_corporate_policy import make_schedule
from src.data.path_area_bridge import (
    AREA_COLUMN,
    CONTROL_COLUMNS,
    build_path_area_inputs,
    route_prediction_rows,
    signed_path_area,
)
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.path_area_bridge import _require_execution_go
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec


def _tariff():
    return {
        "tiers": [{"through_shares": 300_000, "per_share": 0.0035},
                  {"through_shares": None, "per_share": 0.002}],
        "minimum_per_order": 0.35,
        "maximum_order_fraction": 1.0,
        "venue_per_share": 0.0,
        "venue_subdollar_fraction": 0.0,
        "clearing_per_share": 0.0,
        "clearing_maximum_fraction": 0.0,
        "sec_sale_fraction": 0.0,
        "taf_per_sold_share": 0.0,
        "taf_maximum_per_trade": 0.0,
        "cat_per_share": 0.0,
        "commission_pass_through_fraction": 0.0,
    }


def _marks(days):
    rows = []
    for number, day in enumerate(days):
        split = day >= pd.Timestamp("2020-02-04")
        soxl = 55.0 if day >= pd.Timestamp("2020-02-04") else 50.0
        soxx = 50.0 if split else 100.0
        for clock in ("9h45min", "9h47min", "10h32min", "16h"):
            stamp = day + pd.Timedelta(clock)
            rows.append({"stamp": stamp, "SOXL": soxl, "SOXX": soxx})
    return pd.DataFrame(rows).set_index("stamp").sort_index()


def test_fixed_path_oof_clock_physical_reset_and_missing_prefix_contract(tmp_path):
    with pytest.raises(PermissionError, match="separately reviewed"):
        _require_execution_go(Path.cwd(), {}, "implementation-only", "review-only",
                              tmp_path / "no_actual_go.json")
    # Same increments and endpoint preserve QV/covariance while reversing the
    # signed area; a straight line has zero area.
    clockwise = [(0.01, 0), (0, 0.01), (-0.01, 0), (0, -0.01)] + [(0, 0)] * 11
    counter = list(reversed(clockwise[:4])) + [(0, 0)] * 11

    def raw_path(increments):
        xy = [(0.0, 0.0)]
        for dx, dy in increments:
            previous = xy[-1]
            xy.append((previous[0] + dx, previous[1] + dy))
        return np.asarray([[100.0 * (1.0 + y), 100.0 * (1.0 + x)] for x, y in xy])

    area_a, controls_a = signed_path_area(raw_path(clockwise))
    area_b, controls_b = signed_path_area(raw_path(counter))
    assert area_a == pytest.approx(-area_b, abs=1e-14)
    assert controls_a == pytest.approx(controls_b, abs=1e-14)
    line_area, _ = signed_path_area(raw_path([(0.001, 0.002)] * 15))
    assert line_area == pytest.approx(0.0, abs=1e-14)

    # A complete prefix yields fixed fields; a missing prefix stays NaN/mask0.
    trade_dates = pd.to_datetime(["2020-01-31", "2020-02-03"])
    signal_dates = pd.to_datetime(["2020-01-30", "2020-01-31"])
    base = raw_path([(0.001, 0.002)] * 15)
    minute_rows = []
    stamps = pd.date_range(trade_dates[0] + pd.Timedelta(hours=9, minutes=31), periods=15, freq="min")
    for index, stamp in enumerate(stamps):
        for symbol_index, symbol in enumerate(("SOXL", "SOXX")):
            minute_rows.append({"datetime": stamp, "instrument": symbol,
                "$open": np.float32(base[0, symbol_index]),
                "$close": np.float32(base[index + 1, symbol_index]), "$price_valid": np.float32(1)})
    minute = pd.DataFrame(minute_rows).set_index(["datetime", "instrument"]).sort_index()
    minute = pd.concat({"feature": minute}, axis=1)
    timing = pd.DataFrame({"trade_date": trade_dates}, index=signal_dates)
    coverage = pd.DataFrame({"prefix_mask": [True, False]}, index=signal_dates)
    path_inputs = build_path_area_inputs(minute, timing, coverage)
    assert path_inputs.loc[signal_dates[0], AREA_COLUMN] == pytest.approx(0.0, abs=1e-14)
    assert path_inputs.loc[signal_dates[0], "prefix_mask"] == 1
    assert path_inputs.loc[signal_dates[1], "prefix_mask"] == 0
    assert path_inputs.loc[signal_dates[1], [AREA_COLUMN, *CONTROL_COLUMNS]].isna().all()

    # Actual trade year overrides signal year, including the cross-year row.
    routing = pd.DataFrame({"trade_date": pd.to_datetime(["2018-01-02", "2019-01-02", "2020-01-02"])},
                           index=pd.to_datetime(["2017-12-29", "2018-12-31", "2019-12-31"]))
    routed = route_prediction_rows(routing)
    assert list(routed[2018]) == [pd.Timestamp("2017-12-29")]
    assert list(routed[2019]) == [pd.Timestamp("2018-12-31")]
    assert list(routed[2020]) == [pd.Timestamp("2019-12-31")]

    # Original H1 feedback is used only before the next target's 09:46 clock.
    prediction = pd.DataFrame({"soxl_weight": [0.02, 0.04], "soxx_weight": [0.02, 0.03],
        "cash_weight": [0.96, 0.93], "tail_probability": [0.1, 0.1], "alpha_probability": [0.8, 0.8]},
        index=pd.to_datetime(["2018-12-28", "2018-12-31"]))
    h1_timing = pd.DataFrame({"trade_date": pd.to_datetime(["2018-12-31", "2019-01-02"]),
        "return_available_at": pd.to_datetime(["2019-01-02 09:31", "2019-01-03 09:31"]).tz_localize("America/New_York"),
        "soxl_forward_open_return": [0.10, 0.0], "soxx_forward_open_return": [0.0, 0.0]}, index=prediction.index)
    spec = __import__("src.workflow.pit_constituent_auxiliary_training_allocation",
                      fromlist=["PitAuxiliaryAllocationSpec"]).PitAuxiliaryAllocationSpec(root=Path.cwd())
    candidate = __import__("src.workflow.pit_constituent_auxiliary_training_allocation",
                           fromlist=["AuxiliaryTrainingCandidate"]).AuxiliaryTrainingCandidate(
                               name="pit_only_aux_005", auxiliary_weight=.05, include_union_backfill=False)
    smoothed = clocked_smooth_targets(prediction, h1_timing, spec=spec, candidate=candidate)
    assert list(smoothed.index.year) == [2018, 2019]
    assert smoothed.iloc[1].decision_at == pd.Timestamp("2019-01-02 09:46", tz="America/New_York")
    original = __import__("src.workflow.pit_constituent_auxiliary_training_allocation",
                          fromlist=["_smooth_weights"])._smooth_weights(
        prediction, spec=spec, candidate=candidate,
        asset_returns=h1_timing[["soxl_forward_open_return", "soxx_forward_open_return"]])
    np.testing.assert_allclose(smoothed[["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(float),
        original[["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(float), rtol=0, atol=1e-12)
    late = h1_timing.copy()
    late.loc[prediction.index[0], "return_available_at"] = pd.Timestamp("2019-01-02 09:47", tz="America/New_York")
    with pytest.raises(ValueError, match="not mature"):
        clocked_smooth_targets(prediction, late, spec=spec, candidate=candidate)

    # Where contracts agree (unit share basis, no corporate event), the legacy
    # adjusted/raw physical settlement and P2 raw settlement must agree.
    zero_tariff = _tariff()
    zero_tariff.update(minimum_per_order=0.0, maximum_order_fraction=1.0)
    zero_tariff["tiers"] = [{"through_shares": None, "per_share": 0.0}]
    raw_fees, adjusted_fees = TieredFees(zero_tariff, slippage_bps=0), TieredFees(zero_tariff, slippage_bps=0)
    raw_q, raw_cash, _ = __import__("src.backtest.raw_corporate_policy",
        fromlist=["settle_raw"]).settle_raw(np.zeros(2), 1_000.0, 0.0,
            desired=np.array([0.2, 0.5, 0.3]), prices=np.array([10.0, 20.0]),
            stamp=pd.Timestamp("2020-01-31 09:47"), fees=raw_fees)
    reference = settle_tiered_event(np.zeros(2), 1_000.0, desired=np.array([0.2, 0.5, 0.3]),
        adjusted_prices=np.array([10.0, 20.0]), raw_prices=np.array([10.0, 20.0]),
        stamp=pd.Timestamp("2020-01-31 09:47"), fees=adjusted_fees)
    np.testing.assert_array_equal(raw_q, reference["quantities"])
    assert raw_cash == pytest.approx(reference["cash"], abs=1e-12)

    # One continuous teacher around a month boundary, with an ex/pay event and
    # a next-day split. The first row is eligible because its CASH transition
    # remains below the frozen 5% guard; later missing-prefix labels remain valid.
    days = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-06"])
    teacher = pd.DataFrame({"signal_date": pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"]),
        "trade_date": days[:-1], "exit_date": days[1:],
        "soxl_weight": [0.02, 0.04, 0.04, 0.04],
        "soxx_weight": [0.02, 0.03, 0.03, 0.03],
        "cash_weight": [0.96, 0.93, 0.93, 0.93],
        "tail_probability": [0.1] * 4, "alpha_probability": [0.8] * 4,
        "prefix_mask": [1, 0, 1, 1], AREA_COLUMN: [0.01, np.nan, -0.01, 0.02]},
        index=days[:-1])
    for name in CONTROL_COLUMNS:
        teacher[name] = [0.01, np.nan, 0.02, 0.03]
    point_prices = _marks(days)
    cash_events = pd.DataFrame([{"instrument": "SOXX", "event_id": "div-1", "ex_date": days[1],
        "pay_date": days[1], "original_total_decimal": "0.25"}])
    split_events = pd.DataFrame([{"instrument": "SOXX", "event_id": "split-1",
        "effective_trading_date": days[2], "new_shares_per_old_share": 2}])
    schedule = make_schedule(cash_events, split_events, days, days[0], days[-1])
    recovery = IntradayRecoveryDirectAllocationSpec(root=Path.cwd(), minimum_cash_weight=0.15,
        maximum_soxl_weight=0.30, stop_drawdown=0.12, emergency_drawdown=0.22)
    action = Stage1OpenActionValueSpec(root=Path.cwd(), maximum_skippable_turnover=0.05,
                                       mandatory_derisk_beta_delta=0.0)
    shadow = run_training_shadow(teacher, point_prices=point_prices, schedule=schedule, sessions=days,
        recovery_spec=recovery, action_spec=action, tariff=_tariff(), initial_cash=100_000.0,
        slippage_bps=2.0, start_trade_date="2020-01-31")
    snapshot = {name: shadow[name].copy(deep=True) for name in ("orders", "points", "decisions", "trace", "corporate_receipts")}
    monthly = deepcopy(shadow["monthly_fee_shares"])
    labels = label_local_actions(shadow, point_prices=point_prices, schedule=schedule, sessions=days)
    pd.testing.assert_frame_equal(shadow["orders"], snapshot["orders"])
    pd.testing.assert_frame_equal(shadow["points"], snapshot["points"])
    pd.testing.assert_frame_equal(shadow["decisions"], snapshot["decisions"])
    pd.testing.assert_frame_equal(shadow["trace"], snapshot["trace"])
    pd.testing.assert_frame_equal(shadow["corporate_receipts"], snapshot["corporate_receipts"])
    assert shadow["monthly_fee_shares"] == monthly
    assert labels.loc[days[0], "label_status"] == "mature"
    assert "2020-01" in labels.loc[days[0], "execute_fee_month_shares"]
    assert "2020-02" in labels.loc[days[0], "execute_fee_month_shares"]
    assert "2020-01" not in labels.loc[days[0], "hold_fee_month_shares"]
    assert labels.loc[days[1], "prefix_mask"] == 0
    assert labels.loc[days[1], "label_status"] == "mature"
    assert labels.loc[days[1], "label_available_at"] == pd.Timestamp("2020-02-04 09:49", tz="America/New_York")
    exec_receipts = labels.loc[days[1], "execute_receipts"]
    hold_receipts = labels.loc[days[1], "hold_receipts"]
    assert [(item["kind"], item["id"]) for item in exec_receipts] == [
        (item["kind"], item["id"]) for item in hold_receipts]
    assert {"split", "pay"}.issubset({item["kind"] for item in exec_receipts})
    next_target = shadow["decisions"].loc[days[2], ["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(float)
    np.testing.assert_allclose(labels.loc[days[1], ["common_reset_target_soxl", "common_reset_target_soxx",
        "common_reset_target_cash"]].to_numpy(float), next_target, rtol=0, atol=0)
    assert pd.isna(labels.loc[days[3], "execute_delta"])
    assert labels.loc[days[3], "label_reason"] == "missing_next_common_teacher_target"

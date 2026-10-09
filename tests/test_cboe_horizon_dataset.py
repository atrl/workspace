"""Matched date/availability contracts using actual Qlib DatasetH readback."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.data import cboe_horizon_dataset as module
from src.data.cboe_volatility_source import FEATURES
from src.data.convex_allocation import FEATURE_COLUMNS as BASE_FEATURES
from src.data.qlib_native import build_qlib_store_from_frame


@pytest.fixture(scope="module")
def qlib_calendar(tmp_path_factory):
    dates = pd.bdate_range("2014-01-29", periods=14)
    rows = [{"instrument": symbol, "datetime": day, "open": float(100 + index), "close": float(100 + index)}
            for symbol in ("SOXL", "SOXX") for index, day in enumerate(dates)]
    build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=tmp_path_factory.mktemp("matched_cboe") / "provider",
        as_of=str(dates[-1].date()), source="synthetic matched Dataset calendar test")
    return dates


@pytest.fixture
def inputs(qlib_calendar, tmp_path, monkeypatch):
    calendar = qlib_calendar.copy()
    days = pd.Series(calendar, index=calendar)
    frame = pd.DataFrame({name: np.arange(len(calendar), dtype=float) / 100 + number
                          for number, name in enumerate(BASE_FEATURES)}, index=calendar)
    frame.index.name = "datetime"
    frame["signal_date"], frame["trade_date"], frame["exit_date"] = days, days.shift(-1), days.shift(-2)
    frame["feature_available_at"] = calendar.tz_localize("America/New_York") + pd.Timedelta(hours=16)
    frame["decision_at"] = calendar.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
    frame["return_available_at"] = pd.DatetimeIndex(frame.exit_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    frame["h5_exit_date"] = days.shift(-6)
    frame["h5_available_at"] = pd.DatetimeIndex(frame.h5_exit_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    for name, value in (("soxl_return", .003), ("soxx_return", .001)):
        frame[name] = value
        frame.loc[calendar[-2:], name] = np.nan
    for name, value in (("soxl_return_h5", .01), ("soxx_return_h5", .003)):
        frame[name] = value
        frame.loc[calendar[-6:], name] = np.nan
    external = pd.DataFrame({name: np.arange(len(calendar), dtype=float) / 10 + number
                            for number, name in enumerate(FEATURES)}, index=calendar)
    external["source_date"] = days.shift(1)
    external["available_at"] = pd.DatetimeIndex(external.source_date).tz_localize("America/New_York") + pd.Timedelta(hours=17)
    external["decision_at"] = frame.decision_at
    external = external.loc[external.source_date >= "2014-02-03"].copy()
    external.index.name = "datetime"
    primary = {"frame": frame, "calendar": calendar,
               "covariance": np.stack([np.eye(2) * (i + 1) for i in range(len(calendar))]),
               "audit": {"source": "synthetic Qlib input"}}
    plan = {"train_start": str(calendar[0].date()), "matched_training_start": "2014-02-04", "source": {"as_of": str(calendar[-1].date())},
            "external_source": {"provider_uri": "data/qlib/cboe_source", "source_start": "2014-02-03",
                "market_as_of": str(calendar[-1].date()), "store_as_of": "2026-10-08", "features": list(FEATURES)}}
    audit = {"store_as_of": "2026-10-08", "market_as_of": str(calendar[-1].date()),
             "source_start": "2014-02-03", "lag_sessions": 1,
             "historical_vintage_certified": False, "availability_timestamp_verified": False}
    calls = []
    monkeypatch.setattr(module, "prepare_horizon_inputs", lambda root, specification: deepcopy(primary))

    def load(provider, **kwargs):
        calls.append((provider, kwargs))
        return external.copy(deep=True), deepcopy(audit)

    monkeypatch.setattr(module, "load_cboe_features", load)
    return {"root": tmp_path, "plan": plan, "primary": primary, "external": external, "audit": audit, "calls": calls}


def prepare(inputs):
    return module.prepare_cboe_horizon_dataset(inputs["root"], inputs["plan"])


def test_exact_common_dataset_preserves_four_pending_h5_rows_and_full_calendar(inputs):
    from qlib.data.dataset import DatasetH

    old = deepcopy(inputs["primary"])
    result = prepare(inputs)
    frame, dataset = result["frame"], result["dataset"]
    assert isinstance(dataset, DatasetH)
    expected = inputs["primary"]["calendar"][4:-2]
    assert frame.index.equals(expected.rename("datetime"))
    assert result["audit"]["terminal_unsettled_daily_rows_excluded"] == 2
    assert result["audit"]["terminal_unmatured_h5_features_retained"] == 4
    assert frame.h5_available_at.isna().sum() == 4
    assert frame[["soxl_return_h5", "soxx_return_h5"]].isna().all(axis=1).sum() == 4
    assert np.isfinite(frame[[*BASE_FEATURES, *FEATURES]].to_numpy()).all()
    assert result["audit"]["missing_rows_dropped"] == 0
    assert dataset.calendar.equals(inputs["primary"]["calendar"])
    assert result["calendar"].equals(dataset.calendar)
    np.testing.assert_array_equal(result["covariance"], old["covariance"][4:-2])
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    assert list(prepared["feature"].columns) == [*BASE_FEATURES, *FEATURES]
    # Base/full learners consume the same object, labels and row inventory.
    base = prepared["feature"].loc[:, list(BASE_FEATURES)]
    full = prepared["feature"]
    pd.testing.assert_frame_equal(base, full.loc[:, list(BASE_FEATURES)], check_exact=True)
    pd.testing.assert_frame_equal(prepared["label"].droplevel("instrument"),
        frame[["soxl_return_h5", "soxx_return_h5"]], check_exact=True, check_freq=False)
    pd.testing.assert_frame_equal(inputs["primary"]["frame"], old["frame"], check_exact=True)
    provider, options = inputs["calls"][0]
    assert provider == inputs["root"] / "data/qlib/cboe_source"
    assert options["market_calendar"].equals(inputs["primary"]["calendar"])
    assert options["source_start"] == "2014-02-03"


@pytest.mark.parametrize("kind", ["external_missing_day", "interior_daily_nan", "primary_missing_day", "calendar_missing_day"])
def test_no_inner_join_or_silent_interior_drop(inputs, kind):
    primary, external = inputs["primary"], inputs["external"]
    day = primary["calendar"][6]
    if kind == "external_missing_day": external.drop(index=day, inplace=True)
    elif kind == "interior_daily_nan": primary["frame"].loc[day, "exit_date"] = pd.NaT
    elif kind == "primary_missing_day": primary["frame"].drop(index=day, inplace=True)
    else: primary["calendar"] = primary["calendar"].delete(6)
    with pytest.raises(ValueError, match="blocked"):
        prepare(inputs)


@pytest.mark.parametrize("kind", ["same_day", "late_availability", "early_source", "clock_difference", "source_clock"])
def test_external_lag_and_availability_are_strict(inputs, kind):
    frame = inputs["external"]
    day = frame.index[0]
    if kind == "same_day": frame.loc[day, "source_date"] = day
    elif kind == "early_source": frame.loc[day, "source_date"] = pd.Timestamp("2014-01-31")
    elif kind == "late_availability": frame.loc[day, "available_at"] = frame.loc[day, "decision_at"]
    elif kind == "source_clock": frame.loc[day, "available_at"] += pd.Timedelta(minutes=1)
    else: frame.loc[day, "decision_at"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="blocked"):
        prepare(inputs)


@pytest.mark.parametrize("kind", ["external_cutoff", "price_calendar_cutoff", "store_audit_cutoff", "matched_start", "family", "external_nan", "base_nan", "h5_interior_nan", "price_decision_clock", "price_outcome_clock"])
def test_schema_and_common_coverage_contracts_fail_closed(inputs, kind):
    plan, primary, external = inputs["plan"], inputs["primary"], inputs["external"]
    if kind == "external_cutoff": plan["external_source"]["market_as_of"] = "2014-02-13"
    elif kind == "price_calendar_cutoff": plan["source"]["as_of"] = plan["external_source"]["market_as_of"] = "2014-02-13"
    elif kind == "store_audit_cutoff": inputs["audit"]["store_as_of"] = "2026-10-07"
    elif kind == "matched_start": plan["matched_training_start"] = "2014-02-02"
    elif kind == "family": plan["external_source"]["features"] = list(FEATURES[:-1])
    elif kind == "external_nan": external.loc[:, list(FEATURES)] = np.nan
    elif kind == "base_nan": primary["frame"].loc[:, list(BASE_FEATURES)] = np.nan
    elif kind == "price_decision_clock": primary["frame"].loc[primary["calendar"][4], "decision_at"] += pd.Timedelta(minutes=1)
    elif kind == "price_outcome_clock": primary["frame"].loc[primary["calendar"][4], "return_available_at"] += pd.Timedelta(minutes=1)
    else: primary["frame"].loc[primary["calendar"][4], "h5_available_at"] = pd.NaT
    with pytest.raises(ValueError, match="blocked"):
        prepare(inputs)


def test_future_external_values_leave_earlier_dataset_rows_unchanged(inputs):
    original = prepare(inputs)
    cutoff = inputs["primary"]["calendar"][7]
    external = inputs["external"]
    external.loc[external.index > cutoff, list(FEATURES)] += 100
    revised = prepare(inputs)
    before = original["dataset"].prepare("all", col_set=["feature", "label", "metadata"])
    after = revised["dataset"].prepare("all", col_set=["feature", "label", "metadata"])
    mask = before.index.get_level_values("datetime") <= cutoff
    pd.testing.assert_frame_equal(before.loc[mask], after.loc[mask], check_exact=True)

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.data.convex_allocation import FEATURE_COLUMNS, fit_annual_estimator
from src.data.integrated_allocation import build_native_dataset
from src.model.matched_horizon_ridge import (
    LABEL_COLUMNS, METADATA_COLUMNS, fit_matched_horizon_ridge, predict_matched_horizon,
)


EXTRA = ("lag1_log_vix3m", "lag1_log_term", "lag1_log_vvix")
FULL = (*FEATURE_COLUMNS, *EXTRA)


@pytest.fixture(scope="module")
def frame_and_calendar():
    rng = np.random.default_rng(617)
    calendar = pd.bdate_range("2010-08-02", periods=1500)
    dates = pd.Series(calendar, index=calendar)
    frame = pd.DataFrame(rng.normal(size=(len(calendar), len(FULL))), index=calendar, columns=FULL)
    frame.index.name = "datetime"
    frame[FEATURE_COLUMNS[4]] = -frame[FEATURE_COLUMNS[0]]
    frame[FEATURE_COLUMNS[7]] = 2.  # A common constant uses exactly scale one.

    def clock(values, hour, minute=0):
        return pd.Series(pd.DatetimeIndex(values).tz_localize("America/New_York")
                         + pd.Timedelta(hours=hour, minutes=minute), index=calendar)

    frame["signal_date"] = dates
    frame["trade_date"], frame["exit_date"] = dates.shift(-1), dates.shift(-2)
    frame["feature_available_at"] = clock(dates, 16)
    frame["external_available_at"] = clock(dates.shift(1), 17)
    frame["decision_at"] = clock(dates, 16, 15)
    frame["return_available_at"] = clock(dates.shift(-2), 9, 31)
    frame["h5_available_at"] = clock(dates.shift(-6), 9, 31)
    returns = rng.normal(.001, .008, len(calendar))
    for symbol, multiplier in (("soxl", 2.8), ("soxx", 1.)):
        price = pd.Series(100 * np.cumprod(1 + multiplier * returns), index=calendar)
        frame[f"{symbol}_return_h5"] = price.shift(-6) / price.shift(-1) - 1.
    return frame.loc["2011-01-03":].copy(), calendar


def _dataset(frame, calendar):
    dataset = build_native_dataset(frame, feature_columns=FULL, label_columns=LABEL_COLUMNS,
        metadata_columns=METADATA_COLUMNS, as_of=str(calendar[-1].date()))
    dataset.horizon_sessions = 5
    dataset.calendar = calendar
    return dataset


def _fit(dataset, *, features=FEATURE_COLUMNS, **overrides):
    arguments = dict(year=2014, feature_columns=features, ridge_alpha=10., minimum_rows=252,
                     hac_lags=5, train_start="2011-01-03")
    arguments.update(overrides)
    return fit_matched_horizon_ridge(dataset, **arguments)


def test_matched_25_feature_fit_matches_legacy_ridge_hac_on_identical_rows(frame_and_calendar):
    frame, calendar = frame_and_calendar
    dataset = _dataset(frame, calendar)
    fit = _fit(dataset)
    # The old estimator requires exactly its 25-column feature schema, while
    # the new one selects a caller-declared projection of the same native data.
    legacy = build_native_dataset(frame, feature_columns=FEATURE_COLUMNS, label_columns=LABEL_COLUMNS,
        metadata_columns=METADATA_COLUMNS, as_of=str(calendar[-1].date()))
    legacy.horizon_sessions = 5
    expected = fit_annual_estimator(legacy, year=2014, ridge_alpha=10., minimum_training_rows=252, hac_lags=5)
    for key in ("feature_mean", "feature_scale", "prior_mean", "ridge_coefficients", "coefficient_covariance_hac"):
        np.testing.assert_array_equal(fit[key], expected[key])
    assert fit["feature_scale"][7] == 1.
    assert fit["audit"]["training_rows"] == expected["audit"]["training_rows"]
    assert fit["audit"]["last_label_available_at"] == pd.Timestamp(expected["audit"]["last_label_available_at"]).tz_convert("UTC").isoformat()


def test_base_and_full_share_maturity_dates_labels_and_hashes(frame_and_calendar):
    frame, calendar = frame_and_calendar
    dataset = _dataset(frame, calendar)
    base, full = _fit(dataset), _fit(dataset, features=FULL)
    for key in ("matched_rows_sha256", "label_rows_sha256", "training_dates_sha256", "training_rows", "latest_available_at"):
        assert base["audit"][key] == full["audit"][key]
    assert base["audit"]["training_frame_sha256"] != full["audit"]["training_frame_sha256"]
    np.testing.assert_array_equal(base["prior_mean"], full["prior_mean"])
    np.testing.assert_array_equal(base["feature_mean"], full["feature_mean"][:len(FEATURE_COLUMNS)])
    np.testing.assert_array_equal(base["feature_scale"], full["feature_scale"][:len(FEATURE_COLUMNS)])
    assert base["audit"]["invalid_rows_dropped"] == full["audit"]["invalid_rows_dropped"] == 0


def test_future_features_and_unmatured_labels_cannot_change_fit_or_earlier_predictions(frame_and_calendar):
    frame, calendar = frame_and_calendar
    dataset = _dataset(frame, calendar)
    fit = _fit(dataset, features=FULL)
    forecast = predict_matched_horizon(fit, dataset, signal_year=2014)
    altered = frame.copy()
    cutoff = pd.Timestamp("2014-01-01", tz="America/New_York")
    altered.loc[altered.h5_available_at.ge(cutoff), list(LABEL_COLUMNS)] = [.9, -.8]
    altered.loc[altered.index.year >= 2015, list(FULL)] = np.nan
    changed = _dataset(altered, calendar)
    later_fit = _fit(changed, features=FULL)
    for key in ("feature_mean", "feature_scale", "prior_mean", "ridge_coefficients", "coefficient_covariance_hac"):
        np.testing.assert_array_equal(fit[key], later_fit[key])
    assert fit["audit"] == later_fit["audit"]
    later = predict_matched_horizon(later_fit, changed, signal_year=2014)
    np.testing.assert_array_equal(forecast["ridge_mu"], later["ridge_mu"])
    np.testing.assert_array_equal(forecast["Vmean"], later["Vmean"])
    assert forecast["audit"] == later["audit"]


def test_prediction_never_requests_labels_and_retains_unknown_terminal_h5_outcomes(frame_and_calendar, monkeypatch):
    frame, calendar = frame_and_calendar
    dataset = _dataset(frame, calendar)
    year = int(frame.index[-1].year)
    fit = _fit(dataset, year=year)
    prepare = dataset.prepare
    calls = []

    def label_forbidden(*args, **kwargs):
        calls.append(kwargs["col_set"])
        assert "label" not in kwargs["col_set"]
        return prepare(*args, **kwargs)

    monkeypatch.setattr(dataset, "prepare", label_forbidden)
    prediction = predict_matched_horizon(fit, dataset, signal_year=year)
    assert calls == [["feature", "metadata"]]
    assert len(prediction["dates"]) == int((frame.index.year == year).sum())
    assert prediction["dates"][-1] == frame.index[-1]
    assert prediction["predictions"].label_available_at.tail(6).isna().all()
    assert np.isfinite(prediction["ridge_mu"]).all()
    assert prediction["Vmean"].shape == (len(prediction["dates"]), 2, 2)
    assert np.linalg.eigvalsh(prediction["Vmean"]).min() >= -1e-15


def test_current_query_covariance_is_correct_projection_of_joint_hac(frame_and_calendar):
    frame, calendar = frame_and_calendar
    dataset = _dataset(frame, calendar)
    fit = _fit(dataset, features=FULL)
    forecast = predict_matched_horizon(fit, dataset, signal_year=2014)
    row = frame.loc[forecast["dates"][3], list(FULL)].to_numpy(float)
    query = np.r_[1., (row - fit["feature_mean"]) / fit["feature_scale"]]
    projection = np.kron(np.eye(2), query.reshape(1, -1))
    np.testing.assert_allclose(forecast["Vmean"][3], projection @ fit["coefficient_covariance_hac"] @ projection.T, atol=1e-16)


@pytest.mark.parametrize("case", ["missing_feature", "missing_label", "external_naive", "price_naive",
    "bad_decision", "same_day_external", "early_price", "bad_trade", "bad_exit", "bad_h5_clock",
    "bad_h5_session", "future_h5_in_observed_tail", "bad_daily_clock", "duplicate_date", "unordered_date",
    "metadata_date_mismatch", "invalid_mature_feature", "invalid_mature_label", "minus_one_label"])
def test_invalid_source_schema_chronology_and_mature_values_fail_closed(frame_and_calendar, case):
    original, calendar = frame_and_calendar
    frame = original.copy()
    day = frame.index[30]
    if case == "missing_feature":
        frame = frame.drop(columns=EXTRA[0])
        dataset = build_native_dataset(frame, feature_columns=tuple(c for c in FULL if c != EXTRA[0]),
            label_columns=LABEL_COLUMNS, metadata_columns=METADATA_COLUMNS, as_of=str(calendar[-1].date()))
        dataset.calendar = calendar
    elif case == "missing_label":
        dataset = build_native_dataset(frame, feature_columns=FULL, label_columns=LABEL_COLUMNS[:1],
            metadata_columns=METADATA_COLUMNS, as_of=str(calendar[-1].date()))
        dataset.calendar = calendar
    else:
        if case == "external_naive":
            frame["external_available_at"] = frame.external_available_at.dt.tz_localize(None)
        elif case == "price_naive":
            frame["feature_available_at"] = frame.feature_available_at.dt.tz_localize(None)
        elif case == "bad_decision":
            frame.loc[day, "decision_at"] += pd.Timedelta(minutes=1)
        elif case == "same_day_external":
            frame.loc[day, "external_available_at"] = pd.Timestamp(day, tz="America/New_York") + pd.Timedelta(hours=17)
        elif case == "early_price":
            frame.loc[day, "feature_available_at"] -= pd.Timedelta(minutes=1)
        elif case == "bad_trade":
            frame.loc[day, "trade_date"] = frame.loc[day, "exit_date"]
        elif case == "bad_exit":
            frame.loc[day, "exit_date"] += pd.Timedelta(days=1)
        elif case == "bad_h5_clock":
            frame.loc[day, "h5_available_at"] -= pd.Timedelta(minutes=1)
        elif case == "bad_h5_session":
            frame.loc[day, "h5_available_at"] += pd.Timedelta(days=1)
        elif case == "future_h5_in_observed_tail":
            frame.loc[frame.index[-1], "h5_available_at"] = pd.Timestamp(calendar[-1], tz="America/New_York") + pd.Timedelta(days=10, hours=9, minutes=31)
        elif case == "bad_daily_clock":
            frame.loc[day, "return_available_at"] -= pd.Timedelta(minutes=1)
        elif case == "duplicate_date":
            frame.index = pd.DatetimeIndex([frame.index[0], *frame.index[:-1]], name="datetime")
        elif case == "unordered_date":
            pass  # Reverse the native prepare result below, after construction.
        elif case == "metadata_date_mismatch":
            frame.loc[day, "signal_date"] += pd.Timedelta(days=1)
        elif case == "invalid_mature_feature":
            frame.loc[day, EXTRA[0]] = np.nan
        elif case == "invalid_mature_label":
            frame.loc[day, LABEL_COLUMNS[0]] = np.nan
        else:
            frame.loc[day, LABEL_COLUMNS[0]] = -1.
        dataset = _dataset(frame, calendar)
        # StaticDataLoader itself rejects unsorted MultiIndex slicing. Test
        # this module's guard using an invalid response after construction.
        if case == "unordered_date":
            original_prepare = dataset.prepare
            dataset.prepare = lambda *a, **k: original_prepare(*a, **k).iloc[::-1]
    with pytest.raises(ValueError, match="blocked:"):
        _fit(dataset, features=FULL)


def test_missing_extended_feature_never_changes_the_maturity_cohort_by_dropping_a_row(frame_and_calendar):
    frame, calendar = frame_and_calendar
    bad = frame.copy()
    bad.loc[bad.index[30], EXTRA[0]] = np.nan
    baseline = _fit(_dataset(bad, calendar))
    reference = _fit(_dataset(frame, calendar))
    assert baseline["audit"]["matched_rows_sha256"] == reference["audit"]["matched_rows_sha256"]
    with pytest.raises(ValueError, match="rows cannot be dropped"):
        _fit(_dataset(bad, calendar), features=FULL)


@pytest.mark.parametrize("kwargs", [{"year": True}, {"year": 2014.5}, {"feature_columns": "one"},
    {"feature_columns": [FEATURE_COLUMNS[0]] * 2}, {"ridge_alpha": 0.}, {"ridge_alpha": np.inf},
    {"minimum_rows": 1}, {"minimum_rows": True}, {"minimum_rows": 9000},
    {"hac_lags": -1}, {"hac_lags": True}, {"hac_lags": 9000}, {"train_start": "2018-01-01"}])
def test_invalid_fit_contract_and_insufficient_history_fail_closed(frame_and_calendar, kwargs):
    frame, calendar = frame_and_calendar
    with pytest.raises(ValueError, match="blocked:"):
        _fit(_dataset(frame, calendar), **kwargs)


def test_wrong_model_year_and_invalid_parameters_cannot_be_used_for_inference(frame_and_calendar):
    frame, calendar = frame_and_calendar
    dataset = _dataset(frame, calendar)
    fit = _fit(dataset)
    with pytest.raises(ValueError, match="frozen annual model year"):
        predict_matched_horizon(fit, dataset, signal_year=2013)
    bad = deepcopy(fit)
    bad["model_available_at"] = "2014-01-01"
    with pytest.raises(ValueError, match="timezone"):
        predict_matched_horizon(bad, dataset, signal_year=2014)
    bad = deepcopy(fit)
    bad["feature_scale"][0] = 0.
    with pytest.raises(ValueError, match="invalid frozen model"):
        predict_matched_horizon(bad, dataset, signal_year=2014)

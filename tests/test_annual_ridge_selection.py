from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.data.convex_allocation import FEATURE_COLUMNS, LABEL_COLUMNS, build_horizon_dataset, predict_annual
from src.data.integrated_allocation import PRIMARY_METADATA
from src.model import annual_ridge_selection as selection


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(710)
    calendar = pd.bdate_range("2010-08-02", periods=1500)
    dates = pd.Series(calendar, index=calendar)
    frame = pd.DataFrame(rng.normal(size=(len(calendar), len(FEATURE_COLUMNS))), index=calendar, columns=FEATURE_COLUMNS)
    frame.index.name = "datetime"

    def clock(values, hour, minute=0):
        return pd.Series(pd.DatetimeIndex(values).tz_localize("America/New_York")
                         + pd.Timedelta(hours=hour, minutes=minute), index=calendar)

    frame["signal_date"] = dates
    frame["trade_date"], frame["exit_date"], frame["h5_exit_date"] = dates.shift(-1), dates.shift(-2), dates.shift(-6)
    frame["feature_available_at"] = clock(dates, 16)
    frame["decision_at"] = clock(dates, 16, 15)
    frame["return_available_at"] = clock(dates.shift(-2), 9, 31)
    frame["h5_available_at"] = clock(dates.shift(-6), 9, 31)
    frame["forecast_available_at"] = frame.h5_available_at
    frame["auxiliary_available_at"] = clock(dates.shift(-5), 16)
    y = rng.normal(0., .02, len(frame))
    frame["soxl_return_h5"], frame["soxx_return_h5"] = 2.5 * y, y
    frame.loc[frame.index[-6:], list(LABEL_COLUMNS)] = np.nan
    return frame.loc["2011-01-03":].copy(), calendar


def _dataset(frame, calendar):
    dataset = build_horizon_dataset(frame, as_of=str(calendar[-1].date()))
    dataset.calendar = calendar
    return dataset


def _select(frame, calendar, **overrides):
    arguments = dict(outer_year=2014, alphas=(10., 100., 1000.), minimum_training_rows=252,
                     minimum_validation_rows=126, hac_lags=5)
    arguments.update(overrides)
    return selection.select_annual_ridge(_dataset(frame, calendar), frame, **arguments)


def test_mature_prefix_and_candidate_rows_are_identical_and_auditable(synthetic):
    frame, calendar = synthetic
    result = _select(frame, calendar)
    audit = result["audit"]
    inner_cutoff, outer_cutoff = (pd.Timestamp(audit[key]) for key in ("inner_cutoff", "outer_cutoff"))
    expected_train = frame.loc[(frame.h5_available_at < inner_cutoff) & (frame.feature_available_at < inner_cutoff)]
    expected_validation = frame.loc[(frame.index.year == 2013) & (frame.h5_available_at < outer_cutoff)]
    assert audit["inner_training"]["training_rows"] == len(expected_train)
    assert audit["validation_rows"] == len(expected_validation)
    assert audit["validation_unmatured_suffix_rows"] > 0
    assert pd.Timestamp(audit["validation_label_available_at_max"]) < outer_cutoff
    assert set(result["candidate_scores"].candidate_name) == {"prior", "ridge_10", "ridge_100", "ridge_1000"}
    assert result["candidate_scores"].inner_training_frame_sha256.nunique() == 1
    assert result["control_fit"]["audit"]["ridge_alpha"] == 10.
    assert result["selected_fit"]["audit"]["training_frame_sha256"] == result["control_fit"]["audit"]["training_frame_sha256"]
    json.dumps(audit, allow_nan=False)


def test_future_outer_features_and_unmatured_labels_do_not_change_selection_or_refits(synthetic):
    frame, calendar = synthetic
    first = _select(frame, calendar)
    changed = frame.copy()
    cutoff = pd.Timestamp("2014-01-01", tz="America/New_York")
    changed.loc[changed.h5_available_at.ge(cutoff), list(LABEL_COLUMNS)] = [.8, -.7]
    changed.loc[changed.index.year >= 2014, list(FEATURE_COLUMNS)] = np.nan
    second = _select(changed, calendar)
    assert first["selected_name"] == second["selected_name"]
    assert first["audit"] == second["audit"]
    pd.testing.assert_frame_equal(first["candidate_scores"], second["candidate_scores"], check_exact=True)
    for name in ("selected_fit", "control_fit"):
        for key in ("feature_mean", "feature_scale", "ridge_coefficients", "prior_mean", "coefficient_covariance_hac"):
            np.testing.assert_array_equal(first[name][key], second[name][key])


def test_validation_score_is_asset_normalized_and_invariant_to_asset_scale(synthetic):
    frame, calendar = synthetic
    first = _select(frame, calendar)
    scaled = frame.copy()
    scaled["soxl_return_h5"] *= 2.
    second = _select(scaled, calendar)
    assert first["selected_name"] == second["selected_name"]
    np.testing.assert_allclose(first["candidate_scores"].normalized_mse,
                               second["candidate_scores"].normalized_mse, rtol=1e-13, atol=1e-14)
    variance = np.array(first["audit"]["inner_training_label_variances_ddof0"])
    scores = first["candidate_scores"]
    expected = (scores[["mse_soxl", "mse_soxx"]].to_numpy() / variance).mean(axis=1)
    np.testing.assert_array_equal(scores.normalized_mse, expected)


def test_prior_can_win_without_fitting_twenty_five_slopes_and_has_intercept_only_hac(synthetic):
    original, calendar = synthetic
    frame = original.copy()
    cutoff = pd.Timestamp("2013-01-01", tz="America/New_York")
    train = frame.h5_available_at.lt(cutoff)
    for label in LABEL_COLUMNS:
        frame.loc[train, label] -= frame.loc[train, label].mean()
        frame.loc[frame.index.year == 2013, label] = 0.
    result = _select(frame, calendar)
    assert result["selected_name"] == "prior"
    prior = result["selected_fit"]
    assert prior["model_kind"] == "historical_mean"
    assert prior["audit"]["actual_fitted_parameters"] == 2
    np.testing.assert_array_equal(prior["ridge_coefficients"][1:], 0.)
    width = len(FEATURE_COLUMNS) + 1
    occupied = prior["coefficient_covariance_hac"].copy()
    occupied[np.ix_([0, width], [0, width])] = 0.
    np.testing.assert_array_equal(occupied, 0.)
    query = frame.loc[frame.index.year == 2014].copy()
    predicted = predict_annual(prior, query)
    np.testing.assert_array_equal(predicted["ridge"], np.broadcast_to(prior["prior_mean"], predicted["ridge"].shape))
    np.testing.assert_allclose(predicted["Vmean"], np.broadcast_to(predicted["Vmean"][0], predicted["Vmean"].shape), atol=1e-20)
    query.loc[:, list(LABEL_COLUMNS)] = np.nan
    np.testing.assert_array_equal(predict_annual(prior, query)["ridge"], predicted["ridge"])


def test_inner_heldout_shrinkage_selects_regularization_instead_of_training_fit(synthetic):
    original, calendar = synthetic
    frame = original.copy()
    frame.loc[:, list(FEATURE_COLUMNS)] = 0.
    x = np.where(np.arange(len(frame)) % 2 == 0, 1., -1.)
    frame[FEATURE_COLUMNS[0]] = x
    rng = np.random.default_rng(41)
    noise = rng.normal(0, .0002, size=(len(frame), 2))
    frame["soxx_return_h5"] = .03 * x + noise[:, 0]
    frame["soxl_return_h5"] = .09 * x + noise[:, 1]
    validation = frame.index.year == 2013
    frame.loc[validation, "soxx_return_h5"] = .01 * x[validation] + noise[validation, 0]
    frame.loc[validation, "soxl_return_h5"] = .03 * x[validation] + noise[validation, 1]
    frame.loc[frame.index[-6:], list(LABEL_COLUMNS)] = np.nan
    result = _select(frame, calendar)
    assert result["selected_name"] == "ridge_1000"
    scores = result["candidate_scores"].set_index("candidate_name").normalized_mse
    assert scores.ridge_1000 < scores.ridge_100 < scores.ridge_10
    assert scores.ridge_1000 < scores.prior


@pytest.mark.parametrize("prior_wins", [True, False])
def test_only_exact_score_ties_use_prior_then_descending_alpha(synthetic, monkeypatch, prior_wins):
    frame, calendar = synthetic

    def tied_prediction(fit, query):
        values = np.zeros((len(query), 2))
        if fit["model_kind"] == "historical_mean" and not prior_wins:
            values[:] = 100.
        return {"ridge": values}

    monkeypatch.setattr(selection, "predict_annual", tied_prediction)
    result = _select(frame, calendar)
    assert result["selected_name"] == ("prior" if prior_wins else "ridge_1000")
    assert result["audit"]["exact_tie_order"] == ["prior", "ridge_1000", "ridge_100", "ridge_10"]


@pytest.mark.parametrize("case", ["missing_training_label", "missing_validation_label", "missing_validation_feature",
    "bad_h5_clock", "bad_h5_session", "missing_interior_h5_time", "late_decision", "naive_feature",
    "wrong_frame", "missing_calendar", "missing_native_row", "zero_training_variance"])
def test_invalid_maturity_data_or_frame_identity_fails_closed(synthetic, case):
    original, calendar = synthetic
    frame = original.copy()
    validation_day = frame.index[frame.index.year == 2013][30]
    if case == "missing_training_label":
        frame.loc[frame.index[20], LABEL_COLUMNS[0]] = np.nan
    elif case == "missing_validation_label":
        frame.loc[validation_day, LABEL_COLUMNS[0]] = np.nan
    elif case == "missing_validation_feature":
        frame.loc[validation_day, FEATURE_COLUMNS[0]] = np.nan
    elif case == "bad_h5_clock":
        frame.loc[validation_day, "h5_available_at"] -= pd.Timedelta(minutes=1)
    elif case == "bad_h5_session":
        frame.loc[validation_day, "h5_available_at"] += pd.Timedelta(days=1)
    elif case == "missing_interior_h5_time":
        frame.loc[validation_day, "h5_available_at"] = pd.NaT
    elif case == "late_decision":
        frame.loc[validation_day, "decision_at"] += pd.Timedelta(days=1)
    elif case == "naive_feature":
        frame["feature_available_at"] = frame.feature_available_at.dt.tz_localize(None)
    elif case == "missing_native_row":
        frame = frame.drop(index=validation_day)
    elif case == "zero_training_variance":
        frame.loc[frame.index.year < 2013, list(LABEL_COLUMNS)] = 0.
    dataset = _dataset(frame, calendar)
    if case == "wrong_frame":
        frame.loc[validation_day, FEATURE_COLUMNS[0]] += 1.
    elif case == "missing_calendar":
        del dataset.calendar
    with pytest.raises(ValueError, match="blocked:"):
        selection.select_annual_ridge(dataset, frame, outer_year=2014)


@pytest.mark.parametrize("kwargs", [{"outer_year": True}, {"alphas": ()}, {"alphas": (10., 10.)},
    {"alphas": (-1.,)}, {"alphas": (np.nan,)}, {"alphas": (True,)}, {"minimum_training_rows": 1},
    {"minimum_validation_rows": 9000}, {"hac_lags": True}, {"hac_lags": -1}])
def test_invalid_registered_candidate_contract_is_rejected(synthetic, kwargs):
    frame, calendar = synthetic
    with pytest.raises(ValueError, match="blocked:"):
        _select(frame, calendar, **kwargs)


def test_selector_preserves_input_and_refits_have_independent_mutable_buffers(synthetic):
    frame, calendar = synthetic
    before = frame.copy(deep=True)
    dataset = _dataset(frame, calendar)
    native_before = dataset.prepare("all", col_set=["feature", "label", "metadata"]).copy(deep=True)
    result = selection.select_annual_ridge(dataset, frame, outer_year=2014)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    pd.testing.assert_frame_equal(dataset.prepare("all", col_set=["feature", "label", "metadata"]), native_before, check_exact=True)
    control = deepcopy(result["control_fit"]["ridge_coefficients"])
    result["selected_fit"]["ridge_coefficients"][:] = 99.
    np.testing.assert_array_equal(result["control_fit"]["ridge_coefficients"], control)

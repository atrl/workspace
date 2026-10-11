"""Numerical equivalence + H20 fixed-candidate tests for the generic estimator math."""
import numpy as np
import pandas as pd
import pytest

from src.data.generic_horizon import attach_horizon_scope
from src.data.integrated_allocation import build_native_dataset
from src.data.convex_allocation import (FEATURE_COLUMNS, LABEL_COLUMNS, METADATA_COLUMNS,
                                        fit_annual_estimator, predict_annual)
from src.model import generic_horizon_estimator as GE

LABEL_H20 = ("soxl_return_h20", "soxx_return_h20")
METADATA_H20 = ("signal_date", "feature_available_at", "decision_at", "trade_date", "exit_date",
                "return_available_at", "exit_date_h20", "label_h20_available_at")


def _frame(n=900):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2014-01-02", periods=n)
    frame = pd.DataFrame(index=days)
    frame["signal_date"] = days
    frame["trade_date"] = days + pd.tseries.offsets.BDay(1)
    frame["exit_date"] = days + pd.tseries.offsets.BDay(2)
    frame["decision_at"] = (days + pd.Timedelta(hours=16, minutes=15)).tz_localize("America/New_York")
    frame["feature_available_at"] = (days + pd.Timedelta(hours=16)).tz_localize("America/New_York")
    frame["return_available_at"] = (frame["exit_date"] + pd.Timedelta(hours=9, minutes=31)).dt.tz_localize("America/New_York")
    frame["forecast_available_at"] = (frame["exit_date"] + pd.Timedelta(hours=9, minutes=30)).dt.tz_localize("America/New_York")
    frame["auxiliary_available_at"] = frame["decision_at"]
    frame["h5_exit_date"] = days + pd.tseries.offsets.BDay(6)
    frame["h5_available_at"] = (frame["h5_exit_date"] + pd.Timedelta(hours=9, minutes=31)).dt.tz_localize("America/New_York")
    frame["exit_date_h20"] = days + pd.tseries.offsets.BDay(21)
    frame["label_h20_available_at"] = (frame["exit_date_h20"] + pd.Timedelta(hours=9, minutes=31)).dt.tz_localize("America/New_York")
    for name in FEATURE_COLUMNS:
        frame[name] = rng.normal(size=n)
    for label in ("soxl_return_h5", "soxx_return_h5", "soxl_return_h20", "soxx_return_h20"):
        frame[label] = rng.normal(scale=0.01, size=n)
    return frame


def _dataset(frame, feature_columns, label_columns, metadata_columns, horizon):
    dataset = build_native_dataset(frame, feature_columns=feature_columns, label_columns=label_columns,
                                   metadata_columns=metadata_columns, as_of="2020-01-01")
    attach_horizon_scope(dataset, horizon)
    # bind the real mock calendar extended past the last signal to the label exits
    dataset.calendar = pd.bdate_range(frame.index.min(), frame.index.max() + pd.tseries.offsets.BDay(horizon + 5))
    return dataset


def test_generic_matches_sealed_h5_estimator_numerically():
    frame = _frame()
    ds5 = _dataset(frame, FEATURE_COLUMNS, LABEL_COLUMNS, METADATA_COLUMNS, 5)
    ref = fit_annual_estimator(ds5, year=2017, minimum_training_rows=50)
    gen = GE.fit_horizon_estimator(ds5, horizon_sessions=5, feature_columns=FEATURE_COLUMNS,
                                   label_columns=LABEL_COLUMNS, availability_column="h5_available_at",
                                   year=2017, minimum_training_rows=50)
    for key in ("feature_mean", "feature_scale", "prior_mean", "ridge_coefficients", "coefficient_covariance_hac"):
        assert np.allclose(np.asarray(ref[key]), np.asarray(gen[key]), rtol=0, atol=1e-12), key
    assert ref["audit"]["training_frame_sha256"] == gen["audit"]["training_frame_sha256"]
    assert gen["audit"]["hac_lags"] == 5 and gen["audit"]["horizon_sessions"] == 5

    query = frame.iloc[[-1]]
    pref, pgen = predict_annual(ref, query), GE.predict_horizon(gen, query, feature_columns=FEATURE_COLUMNS, horizon_sessions=5)
    for key in ("prior", "ridge", "Vmean"):
        assert np.allclose(pref[key], pgen[key], rtol=0, atol=1e-12), key


def test_h20_fixed_four_candidates_units_and_hac():
    frame = _frame()
    ds20 = _dataset(frame, FEATURE_COLUMNS, LABEL_H20, METADATA_H20, 20)
    fits = {"ridge_10": 10.0, "ridge_100": 100.0, "ridge_1000": 1000.0}
    models = {name: GE.fit_horizon_estimator(ds20, horizon_sessions=20, feature_columns=FEATURE_COLUMNS,
                                             label_columns=LABEL_H20, availability_column="label_h20_available_at",
                                             year=2017, ridge_alpha=alpha, minimum_training_rows=50,
                                             mean_covariance_units="twenty_session_simple_return_squared")
              for name, alpha in fits.items()}
    priors = {tuple(np.round(m["prior_mean"], 12)) for m in models.values()}
    assert len(priors) == 1  # intercept-only prior identical across alphas
    coeffs = {tuple(np.round(m["ridge_coefficients"].ravel(), 9)) for m in models.values()}
    assert len(coeffs) == 3  # each alpha yields a distinct Ridge fit
    for model in models.values():
        assert model["horizon_sessions"] == 20 and model["audit"]["hac_lags"] == 5
        assert model["audit"]["mean_covariance_units"] == "twenty_session_simple_return_squared"
        assert model["label_columns"] == LABEL_H20  # h20 labels, not an h5 swap
    query = frame.iloc[[-1]]
    prediction = GE.predict_horizon(models["ridge_10"], query, feature_columns=FEATURE_COLUMNS, horizon_sessions=20)
    assert prediction["Vmean"].shape == (1, 2, 2)
    assert (np.linalg.eigvalsh(prediction["Vmean"]) >= -1e-12).all()  # PSD Vmean


def test_horizon_scope_and_label_mismatch_blocked():
    frame = _frame()
    ds20 = _dataset(frame, FEATURE_COLUMNS, LABEL_H20, METADATA_H20, 20)
    with pytest.raises(ValueError, match="horizon-tagged DatasetH"):
        GE.fit_horizon_estimator(ds20, horizon_sessions=5, feature_columns=FEATURE_COLUMNS,
                                 label_columns=LABEL_H20, availability_column="label_h20_available_at",
                                 year=2017, minimum_training_rows=50)


def test_terminal_nat_labels_must_align_with_out_of_range_exit():
    from src.data.generic_horizon import build_horizon_labels
    cal = pd.bdate_range("2016-01-04", periods=50)
    rng = np.random.default_rng(3)
    prices = pd.DataFrame({"open_SOXL": 100 + np.arange(50.0), "open_SOXX": 50 + np.arange(50.0)}, index=cal)
    frame = pd.DataFrame(index=cal)
    frame["signal_date"] = cal
    frame["feature_available_at"] = (cal + pd.Timedelta(hours=16)).tz_localize("America/New_York")
    frame["decision_at"] = (cal + pd.Timedelta(hours=16, minutes=15)).tz_localize("America/New_York")
    dates = pd.Series(cal, index=cal)
    frame["exit_date"] = dates.shift(-2).reindex(cal).to_numpy()  # calendar t+2 (NaT at the shortened tail)
    frame["return_available_at"] = pd.DatetimeIndex(frame["exit_date"]).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    frame["f1"] = rng.normal(size=50)
    frame["f2"] = rng.normal(size=50)
    labels, _ = build_horizon_labels(frame, prices, horizon_sessions=20, availability_hour=9, availability_minute=31, calendar=cal)
    assert labels["soxl_return_h20"].isna().sum() == 21  # terminal unknown tail
    ds = build_native_dataset(labels, feature_columns=["f1", "f2"], label_columns=LABEL_H20,
                              metadata_columns=METADATA_H20, as_of="2020-01-01")
    attach_horizon_scope(ds, 20)
    ds.calendar = cal
    # terminal NaT aligns with out-of-range exit; validation passes
    GE.validate_native_horizon_dataset(ds, horizon_sessions=20, feature_columns=["f1", "f2"],
                                       label_columns=LABEL_H20, availability_column="label_h20_available_at")
    promoted = labels.copy()
    promoted.loc[promoted.index[20], "label_h20_available_at"] = pd.NaT  # interior NaT is illegal
    ds_bad = build_native_dataset(promoted, feature_columns=["f1", "f2"], label_columns=LABEL_H20,
                                  metadata_columns=METADATA_H20, as_of="2020-01-01")
    attach_horizon_scope(ds_bad, 20)
    ds_bad.calendar = cal
    with pytest.raises(ValueError):  # interior NaT is blocked (must align with an out-of-range exit)
        GE.validate_native_horizon_dataset(ds_bad, horizon_sessions=20, feature_columns=["f1", "f2"],
                                           label_columns=LABEL_H20, availability_column="label_h20_available_at")
    with pytest.raises(ValueError, match="terminal unknown tail"):
        GE._mature_tail_check(pd.Series([0.1, None, 0.2]))


def test_missing_feature_availability_on_one_row_blocks():
    frame = _frame()
    frame.iloc[10, frame.columns.get_loc("feature_available_at")] = None
    ds = _dataset(frame, FEATURE_COLUMNS, LABEL_H20, METADATA_H20, 20)
    with pytest.raises(ValueError, match="feature_available_at is missing"):
        GE.fit_horizon_estimator(ds, horizon_sessions=20, feature_columns=FEATURE_COLUMNS, label_columns=LABEL_H20,
                                 availability_column="label_h20_available_at", year=2017, minimum_training_rows=50)


def test_predict_accepts_index_and_checks_feature_clock():
    frame = _frame()
    ds = _dataset(frame, FEATURE_COLUMNS, LABEL_H20, METADATA_H20, 20)
    model = GE.fit_horizon_estimator(ds, horizon_sessions=20, feature_columns=FEATURE_COLUMNS, label_columns=LABEL_H20,
                                     availability_column="label_h20_available_at", year=2017, minimum_training_rows=50)
    query = frame.loc[frame.index.year == 2017].head(3).copy()
    by_column = GE.predict_horizon(model, query, feature_columns=FEATURE_COLUMNS, horizon_sessions=20)
    by_index = GE.predict_horizon(model, query.drop(columns=["signal_date"]), feature_columns=FEATURE_COLUMNS, horizon_sessions=20)
    assert np.allclose(by_column["ridge"], by_index["ridge"])
    leaked = query.copy()
    leaked.iloc[0, leaked.columns.get_loc("feature_available_at")] = pd.Timestamp("2030-01-01").tz_localize("America/New_York")
    with pytest.raises(ValueError, match="prediction clock"):
        GE.predict_horizon(model, leaked, feature_columns=FEATURE_COLUMNS, horizon_sessions=20)

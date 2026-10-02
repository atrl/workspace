from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from qlib.data.dataset import DatasetH
from sklearn.linear_model import Ridge

from src.data.convex_allocation import (
    FEATURE_COLUMNS, LABEL_COLUMNS, MEAN_COVARIANCE_UNITS, _psd_roundoff,
    build_horizon_dataset, build_horizon_frame, fit_annual_estimator, predict_annual,
)
from src.data.integrated_allocation import PRIMARY_FEATURES


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(314)
    calendar = pd.bdate_range("2011-01-03", periods=1500)
    dates = pd.Series(calendar, index=calendar)
    returns = rng.normal(.0004, .012, len(calendar))
    soxx = 100*np.cumprod(1+returns)
    soxl = 20*np.cumprod(1+3*returns+rng.normal(0, .002, len(calendar)))
    prices = pd.DataFrame({("open", "SOXL"): soxl, ("open", "SOXX"): soxx,
                           ("close", "SOXL"): soxl*1.002, ("close", "SOXX"): soxx*1.001}, index=calendar)
    features = rng.normal(size=(len(calendar), len(FEATURE_COLUMNS)))
    features[:, 4] = -features[:, 0]  # Existing reversal feature's exact redundancy.
    frame = pd.DataFrame(features, index=calendar, columns=FEATURE_COLUMNS)
    frame.index.name = "datetime"
    for name in PRIMARY_FEATURES[-3:]:
        frame[name] = 0.
    frame["signal_date"] = dates
    frame["trade_date"], frame["exit_date"] = dates.shift(-1), dates.shift(-2)

    def at(values, hour, minute=0):
        return pd.Series(pd.DatetimeIndex(values).tz_localize("America/New_York") + pd.Timedelta(hours=hour, minutes=minute), index=calendar)

    frame["feature_available_at"] = at(dates, 16)
    frame["decision_at"] = at(frame.trade_date, 9, 31)
    frame["return_available_at"] = at(frame.exit_date, 9, 30)
    frame["forecast_available_at"] = at(dates.shift(-6), 9, 30)
    frame["auxiliary_available_at"] = at(dates.shift(-5), 16)
    for symbol in ("SOXL", "SOXX"):
        price = prices[("open", symbol)]
        frame[f"{symbol.lower()}_return"] = price.shift(-2)/price.shift(-1)-1
    horizon, audit = build_horizon_frame(frame, prices)
    dataset = build_horizon_dataset(horizon, as_of=str(calendar[-1].date()))
    return frame, prices, horizon, dataset, audit


def test_h5_label_alignment_preserves_daily_replay_and_tail_features(data):
    original, prices, frame, dataset, audit = data
    pd.testing.assert_frame_equal(frame.loc[:, original.columns], original)
    for i, row in frame.iloc[:-6].iterrows():
        pos = prices.index.get_loc(i)
        expected = prices.iloc[pos+6][("open", "SOXX")]/prices.iloc[pos+1][("open", "SOXX")]-1
        assert row.soxx_return_h5 == pytest.approx(expected)
        assert row.h5_available_at == row.forecast_available_at
        assert row.feature_available_at < row.decision_at < row.h5_available_at
    assert frame.loc[:, list(LABEL_COLUMNS)].tail(6).isna().all().all()
    assert len(frame) == len(original) and audit["daily_returns_preserved"]
    assert isinstance(dataset, DatasetH)
    assert tuple(dataset.prepare("all", col_set="feature").columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 25


def test_ridge_matches_sklearn_with_prior_only_scaler_and_unpenalized_intercept(data):
    _, _, frame, dataset, _ = data
    fit = fit_annual_estimator(dataset, year=2015)
    cutoff = pd.Timestamp("2015-01-01", tz="America/New_York")
    train = frame.loc[frame.h5_available_at.lt(cutoff)]
    X = train.loc[:, list(FEATURE_COLUMNS)].to_numpy()
    Y = train.loc[:, list(LABEL_COLUMNS)].to_numpy()
    standardized = (X-fit["feature_mean"])/fit["feature_scale"]
    reference = Ridge(alpha=10.).fit(standardized, Y)
    np.testing.assert_allclose(fit["ridge_coefficients"][0], reference.intercept_, atol=1e-12)
    np.testing.assert_allclose(fit["ridge_coefficients"][1:].T, reference.coef_, atol=1e-12)
    np.testing.assert_allclose(fit["prior_mean"], Y.mean(axis=0), atol=1e-14)
    assert pd.Timestamp(fit["audit"]["last_label_available_at"]) < cutoff
    assert fit["audit"]["hac"]["maxlags"] == 5
    json.dumps(fit["audit"], allow_nan=False)


def test_unmatured_and_future_observations_cannot_change_fit(data):
    _, _, frame, dataset, _ = data
    cutoff = pd.Timestamp("2015-01-01", tz="America/New_York")
    first = fit_annual_estimator(dataset, year=2015)
    changed = frame.copy()
    changed.loc[changed.h5_available_at.ge(cutoff), list(LABEL_COLUMNS)] = .9
    changed.loc[changed.index.year >= 2015, list(FEATURE_COLUMNS)] *= 1000
    second = fit_annual_estimator(build_horizon_dataset(changed, as_of=str(frame.index[-1].date())), year=2015)
    for field in ("prior_mean", "feature_mean", "feature_scale", "ridge_coefficients", "coefficient_covariance_hac"):
        np.testing.assert_array_equal(first[field], second[field])
    assert first["audit"]["training_frame_sha256"] == second["audit"]["training_frame_sha256"]


def test_hac5_uses_joint_residual_scores_and_mean_query_projection(data):
    _, _, frame, dataset, _ = data
    fit = fit_annual_estimator(dataset, year=2015)
    train = frame[frame.h5_available_at.lt(pd.Timestamp(fit["model_available_at"]))]
    X = train.loc[:, list(FEATURE_COLUMNS)].to_numpy()
    A = np.c_[np.ones(len(X)), (X-fit["feature_mean"])/fit["feature_scale"]]
    E = train.loc[:, list(LABEL_COLUMNS)].to_numpy()-A@fit["ridge_coefficients"]
    scores = np.stack([np.kron(e, a) for e, a in zip(E, A)])
    scores -= scores.mean(axis=0)
    S = scores.T@scores
    for lag in range(1, 6):
        cross = sum(np.outer(scores[i], scores[i-lag]) for i in range(lag, len(scores)))
        S += (1-lag/6)*(cross+cross.T)
    inverse = np.linalg.inv(A.T@A+np.diag([0]+[10]*25))
    bread = np.kron(np.eye(2), inverse)
    np.testing.assert_allclose(fit["coefficient_covariance_hac"], bread@S@bread.T, atol=1e-13)
    query = frame.loc[frame.index.year == 2015].iloc[:5]
    predicted = predict_annual(fit, query)
    a = np.r_[1., (query.iloc[0].loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float)-fit["feature_mean"])/fit["feature_scale"]]
    G = np.kron(np.eye(2), a.reshape(1, -1))
    np.testing.assert_allclose(predicted["Vmean"][0], G@fit["coefficient_covariance_hac"]@G.T, atol=1e-13)
    assert predicted["Vmean"].shape == (5, 2, 2)
    assert np.linalg.eigvalsh(predicted["Vmean"]).min() >= -1e-14
    assert predicted["audit"]["mean_covariance_units"] == MEAN_COVARIANCE_UNITS
    assert predicted["audit"]["daily_risk_covariance_units"] != MEAN_COVARIANCE_UNITS


def test_prediction_does_not_read_future_labels_or_old_forecast_slots(data):
    _, _, frame, dataset, _ = data
    fit = fit_annual_estimator(dataset, year=2015)
    query = frame.loc[frame.index.year == 2015].iloc[:10]
    first = predict_annual(fit, query)
    changed = query.copy()
    changed.loc[:, [*LABEL_COLUMNS, *PRIMARY_FEATURES[-3:]]] = 1e9
    second = predict_annual(fit, changed)
    for field in ("prior", "ridge", "Vmean"):
        np.testing.assert_array_equal(first[field], second[field])
    with pytest.raises(ValueError, match="retrospective OOF"):
        predict_annual(fit, frame.loc[frame.index.year == 2014].iloc[:2])


def test_covariance_repairs_only_roundoff():
    repaired, audit = _psd_roundoff(np.diag([-1e-16, 1.]))
    assert audit["clipped_roundoff_eigenvalues"] == 1 and repaired[0, 0] == 0.
    with pytest.raises(ValueError, match="materially negative"):
        _psd_roundoff(np.diag([-.01, 1.]))
    with pytest.raises(ValueError, match="asymmetric"):
        _psd_roundoff(np.array([[1., .3], [.7, 1.]]))


def test_invalid_maturity_alignment_insufficient_history_and_prediction_fail_closed(data):
    original, prices, frame, dataset, _ = data
    bad = original.copy()
    bad["forecast_available_at"] = bad["return_available_at"]
    with pytest.raises(ValueError, match="availability differs"):
        build_horizon_frame(bad, prices)
    with pytest.raises(ValueError, match="insufficient strictly mature"):
        fit_annual_estimator(dataset, year=2012, minimum_training_rows=1000)
    fit = fit_annual_estimator(dataset, year=2015)
    bad_query = frame.loc[frame.index.year == 2015].iloc[:2].copy()
    bad_query.iloc[0, bad_query.columns.get_loc(FEATURE_COLUMNS[0])] = np.nan
    with pytest.raises(ValueError, match="invalid annual prediction"):
        predict_annual(fit, bad_query)
    changed = deepcopy(fit)
    changed["coefficient_covariance_hac"] = -np.eye(52)
    with pytest.raises(ValueError, match="materially negative"):
        predict_annual(changed, frame.loc[frame.index.year == 2015].iloc[:2])

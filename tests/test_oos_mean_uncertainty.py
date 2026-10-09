import json

import numpy as np
import pandas as pd
import pytest

from src.model.oos_mean_uncertainty import fit_oos_mean_error


def _predictions(rows=40):
    dates = pd.bdate_range("2018-12-03", periods=rows)
    issued = dates.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
    available = (dates + pd.offsets.BDay(6)).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    return pd.DataFrame({"signal_date": dates, "forecast_available_at": issued,
        "label_available_at": available, "predicted_soxl": .01, "predicted_soxx": .005,
        "realized_soxl": .03, "realized_soxx": -.005, "model_year": dates.year})


def _fit(frame, cutoff="2019-12-31T00:00:00-05:00", **kwargs):
    return fit_oos_mean_error(frame, cutoff=pd.Timestamp(cutoff), minimum_rows=kwargs.pop("minimum_rows", 10),
                              hac_lags=kwargs.pop("hac_lags", 5), **kwargs)


def test_constant_oos_bias_survives_as_floor_without_dividing_by_sample_size():
    result = _fit(_predictions())
    bias = np.array([.02, -.01])
    np.testing.assert_allclose(result["bias"], bias, atol=1e-15)
    np.testing.assert_allclose(result["hac_covariance_of_mean"], np.zeros((2, 2)), atol=1e-30)
    np.testing.assert_allclose(result["floor"], np.outer(bias, bias), atol=1e-18)
    longer = _fit(_predictions(rows=80))
    np.testing.assert_allclose(longer["floor"], result["floor"], atol=1e-18)
    assert result["audit"]["row_count"] == 40
    assert result["audit"]["response_order"] == ["SOXL", "SOXX"]
    assert "not predictive RMSE" in result["audit"]["interpretation"]
    json.dumps(result["audit"], allow_nan=False)


def test_unbiased_hac_matches_direct_bartlett_covariance_of_mean_not_error_variance():
    frame = _predictions()
    rng = np.random.default_rng(507)
    residual = rng.normal(0., .02, size=(len(frame), 2))
    residual -= residual.mean(axis=0)
    frame[["realized_soxl", "realized_soxx"]] = frame[["predicted_soxl", "predicted_soxx"]].to_numpy() + residual
    result = _fit(frame, hac_lags=3)
    centered = residual - residual.mean(axis=0)
    meat = centered.T @ centered
    for lag in range(1, 4):
        cross = centered[lag:].T @ centered[:-lag]
        meat += (1 - lag / 4) * (cross + cross.T)
    expected = meat / len(frame) ** 2
    np.testing.assert_allclose(result["bias"], 0., atol=1e-16)
    np.testing.assert_allclose(result["hac_covariance_of_mean"], expected, atol=1e-18)
    np.testing.assert_allclose(result["floor"], expected, atol=1e-18)
    assert not np.allclose(result["floor"], residual.T @ residual / len(frame), atol=1e-8)
    assert np.linalg.eigvalsh(result["floor"]).min() >= -1e-18


def test_future_and_unmatured_numerical_values_never_change_fit_or_consumed_hash():
    frame = _predictions(rows=100)
    cutoff = pd.Timestamp("2019-01-01", tz="America/New_York")
    original = _fit(frame, cutoff=cutoff)
    future = frame.label_available_at >= cutoff
    altered = frame.copy()
    altered.loc[future, ["predicted_soxl", "predicted_soxx", "realized_soxl", "realized_soxx"]] = [np.nan, np.inf, -2., 1e200]
    result = _fit(altered, cutoff=cutoff)
    for key in ("bias", "hac_covariance_of_mean", "floor"):
        np.testing.assert_array_equal(original[key], result[key])
    assert original["audit"] == result["audit"]
    assert pd.Timestamp(result["audit"]["latest_label_available_at"]) < cutoff
    assert result["audit"]["row_count"] == int((~future).sum())


def test_future_issuance_metadata_is_not_consumed_and_index_or_extra_columns_do_not_change_hash():
    frame = _predictions(rows=100)
    cutoff = pd.Timestamp("2019-01-01", tz="America/New_York")
    first = _fit(frame, cutoff=cutoff)
    changed = frame.copy()
    future = changed.forecast_available_at >= cutoff
    changed.loc[future, "model_year"] = 2099
    changed.loc[future, "forecast_available_at"] = pd.NaT
    changed.loc[future, "label_available_at"] = pd.NaT
    changed.index = np.arange(len(changed)) * 13
    changed["unrelated_future_feature"] = np.nan
    second = _fit(changed, cutoff=cutoff)
    assert first["audit"] == second["audit"]
    np.testing.assert_array_equal(first["floor"], second["floor"])


def test_cutoff_equality_excludes_the_label_and_timezone_conversion_is_equivalent():
    frame = _predictions()
    cutoff = frame.label_available_at.iloc[15]
    local = _fit(frame, cutoff=cutoff)
    utc = _fit(frame, cutoff=cutoff.tz_convert("UTC"))
    assert local["audit"] == utc["audit"]
    assert local["audit"]["row_count"] == 15
    changed = frame.copy()
    changed.loc[15:, "realized_soxl"] = np.nan
    np.testing.assert_array_equal(_fit(changed, cutoff=cutoff)["floor"], local["floor"])


@pytest.mark.parametrize("case", ["missing_column", "missing_prediction", "missing_realized",
    "missing_forecast_time", "missing_label_time", "naive_forecast", "naive_label", "wrong_clock",
    "same_time_label", "early_label", "reordered_label", "duplicate_signal", "unordered_signal",
    "intraday_signal", "aware_signal", "future_model", "past_model", "noninteger_model",
    "minus_one_prediction", "minus_one_realized", "infinite_realized", "duplicate_column"])
def test_invalid_past_forecast_identity_chronology_and_values_fail_closed(case):
    frame = _predictions()
    if case == "missing_column":
        frame = frame.drop(columns="model_year")
    elif case == "missing_prediction":
        frame.loc[4, "predicted_soxl"] = np.nan
    elif case == "missing_realized":
        frame.loc[4, "realized_soxx"] = np.nan
    elif case == "missing_forecast_time":
        frame.loc[4, "forecast_available_at"] = pd.NaT
    elif case == "missing_label_time":
        frame.loc[4, "label_available_at"] = pd.NaT
    elif case == "naive_forecast":
        frame["forecast_available_at"] = frame.forecast_available_at.dt.tz_localize(None)
    elif case == "naive_label":
        frame["label_available_at"] = frame.label_available_at.dt.tz_localize(None)
    elif case == "wrong_clock":
        frame.loc[4, "forecast_available_at"] += pd.Timedelta(minutes=1)
    elif case == "same_time_label":
        frame.loc[4, "label_available_at"] = frame.loc[4, "forecast_available_at"]
    elif case == "early_label":
        frame.loc[4, "label_available_at"] = frame.loc[4, "forecast_available_at"] - pd.Timedelta(minutes=1)
    elif case == "reordered_label":
        frame.loc[[4, 5], "label_available_at"] = frame.loc[[5, 4], "label_available_at"].to_numpy()
    elif case == "duplicate_signal":
        frame.loc[4, "signal_date"] = frame.loc[3, "signal_date"]
    elif case == "unordered_signal":
        frame = frame.iloc[::-1]
    elif case == "intraday_signal":
        frame.loc[4, "signal_date"] += pd.Timedelta(hours=1)
    elif case == "aware_signal":
        frame["signal_date"] = frame.signal_date.dt.tz_localize("UTC")
    elif case == "future_model":
        frame.loc[4, "model_year"] += 1
    elif case == "past_model":
        frame.loc[4, "model_year"] -= 1
    elif case == "noninteger_model":
        frame["model_year"] = frame.model_year.astype(float) + .5
    elif case == "minus_one_prediction":
        frame.loc[4, "predicted_soxl"] = -1.
    elif case == "minus_one_realized":
        frame.loc[4, "realized_soxx"] = -1.
    elif case == "infinite_realized":
        frame.loc[4, "realized_soxl"] = np.inf
    else:
        frame = pd.concat([frame, frame[["model_year"]]], axis=1)
    with pytest.raises(ValueError, match="blocked:"):
        _fit(frame)


@pytest.mark.parametrize("kwargs", [{"minimum_rows": 1}, {"minimum_rows": 10.5},
    {"minimum_rows": True}, {"minimum_rows": 100}, {"hac_lags": -1},
    {"hac_lags": 40}, {"hac_lags": True}, {"hac_lags": 1.5}])
def test_invalid_estimation_contract_or_insufficient_rows_rejects(kwargs):
    with pytest.raises(ValueError, match="blocked:"):
        _fit(_predictions(), **kwargs)


def test_naive_cutoff_and_empty_inputs_fail_closed():
    with pytest.raises(ValueError, match="timezone-aware"):
        fit_oos_mean_error(_predictions(), cutoff="2020-01-01")
    with pytest.raises(ValueError, match="blocked:"):
        _fit(_predictions().iloc[:0])
    with pytest.raises(ValueError, match="insufficient"):
        _fit(_predictions(), cutoff="2018-01-01T00:00:00Z")


def test_default_minimum_applies_and_supplied_frame_is_not_mutated():
    frame = _predictions(rows=300)
    original = frame.copy(deep=True)
    result = fit_oos_mean_error(frame, cutoff=pd.Timestamp("2021-01-01", tz="UTC"))
    assert result["audit"]["row_count"] == 300
    assert result["audit"]["minimum_rows"] == 252
    assert result["audit"]["hac_lags"] == 5
    pd.testing.assert_frame_equal(frame, original, check_exact=True)

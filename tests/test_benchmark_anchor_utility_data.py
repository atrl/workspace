from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.data.benchmark_anchor_utility import (
    FEATURE_COLUMNS, PRICE_FEATURES, build_anchor_dataset, fit_feature_scaler,
    prepare_training_rows, transform_features,
)


def _market_time(days, hour, minute=0):
    stamps = pd.DatetimeIndex(pd.to_datetime(days)).tz_localize("America/New_York")
    return pd.Series(stamps + pd.Timedelta(hours=hour, minutes=minute), index=days.index)


@pytest.fixture
def frame():
    dates = pd.bdate_range("2011-01-03", "2014-01-10", name="datetime")
    t = np.arange(len(dates))
    data = pd.DataFrame({name: np.sin(t * (.01 + j * .002)) + j / 100 for j, name in enumerate(PRICE_FEATURES)}, index=dates)
    data[PRICE_FEATURES[0]] = .1  # exactly constant, even if mean rounding is not
    days = pd.Series(dates, index=dates)
    data["signal_date"] = days
    data["trade_date"], data["exit_date"] = days.shift(-1), days.shift(-2)
    data["feature_available_at"] = _market_time(days, 16)
    data["decision_at"] = _market_time(data.trade_date, 9, 31)
    data["return_available_at"] = _market_time(data.exit_date, 9, 30)
    data["forecast_available_at"] = _market_time(days.shift(-6), 9, 30)
    data["auxiliary_available_at"] = _market_time(days.shift(-5), 16)
    data["soxl_return"] = .01 * np.sin(t * .3)
    data["soxx_return"] = .004 * np.cos(t * .3)
    data.loc[data.exit_date.isna(), ["soxl_return", "soxx_return"]] = np.nan
    data["forecast_available"] = (data.index.year >= 2012).astype(float)
    data["future_vol_prediction"] = .02 + .003 * np.sin(t * .1)
    data.loc[data.forecast_available.eq(0), "future_vol_prediction"] = np.nan
    data["forecast_model_available_at"] = pd.Series([
        pd.Timestamp(f"{date.year}-01-01", tz="America/New_York") if date.year >= 2012 else pd.NaT
        for date in dates], index=dates)
    return data


def _training(frame):
    return prepare_training_rows(build_anchor_dataset(frame, as_of="2014-01-10"), 2013)


def test_native_dataset_preserves_all_rows_and_daily_maturity_not_h5(frame):
    dataset = build_anchor_dataset(frame, as_of="2014-01-10")
    assert len(dataset.prepare("all")) == len(frame)
    assert dataset.prepare("all", col_set=["feature"])["feature"]["future_vol_prediction"].isna().any()
    rows, audit = prepare_training_rows(dataset, 2013)
    cutoff = pd.Timestamp("2013-01-01", tz="America/New_York")
    expected = frame.feature_available_at.lt(cutoff) & frame.return_available_at.lt(cutoff)
    assert rows.index.equals(frame.index[expected])
    assert audit["eligible_rows"] == int(expected.sum())
    assert (rows.forecast_available_at >= cutoff).any()
    assert not audit["h5_or_vol_target_availability_used_for_policy_maturity"]
    boundary = rows.index[-1]
    changed = frame.copy()
    changed.loc[boundary, "return_available_at"] = cutoff
    later, _ = _training(changed)
    assert boundary not in later.index


def test_scalers_are_prior_only_available_vol_only_serializable_and_mask_after_scaling(frame):
    rows, audit = _training(frame)
    scaler = fit_feature_scaler(rows, audit, "price_vol")
    ready = rows.forecast_available.eq(1)
    assert scaler["vol_mean"] == pytest.approx(rows.loc[ready, "future_vol_prediction"].mean())
    assert scaler["vol_scale"] == pytest.approx(rows.loc[ready, "future_vol_prediction"].std(ddof=0))
    assert scaler["price_scale"][0] == 1
    assert scaler["price_mean"][0] == .1
    result = transform_features(rows, json.loads(json.dumps(scaler)), "price_vol")
    assert result["features"].shape == (len(rows), 27)
    assert not result["invalid_forecast_mask"].any()
    assert (result["features"][~ready, 25:] == 0).all()
    assert (result["features"][:, 0] == 0).all()
    assert (result["features"][ready, 26] == 1).all()
    for variant, zero_start in (("constant", 0), ("price", 25)):
        masked = transform_features(rows, fit_feature_scaler(rows, audit, variant), variant)
        assert (masked["features"][:, zero_start:] == 0).all()
    future = frame.copy()
    future.loc[future.index.year >= 2013, list(PRICE_FEATURES)] = 9999
    future.loc[future.index.year >= 2013, "future_vol_prediction"] = -100
    future_rows, future_audit = _training(future)
    same = fit_feature_scaler(future_rows, future_audit, "price_vol")
    assert same == scaler


@pytest.mark.parametrize("field,value", [
    ("future_vol_prediction", np.nan), ("future_vol_prediction", -.01),
    ("future_vol_prediction", 0.), ("forecast_available", 2.),
    ("forecast_model_available_at", pd.NaT),
    ("forecast_model_available_at", pd.Timestamp("2014-01-01", tz="America/New_York")),
])
def test_bad_used_training_forecasts_tag_whole_fold_without_affecting_other_variants(frame, field, value):
    point = pd.Timestamp("2012-06-04")
    frame.loc[point, field] = value
    rows, audit = _training(frame)
    assert point in rows.index
    for variant in ("constant", "price"):
        scaler = fit_feature_scaler(rows, audit, variant)
        assert scaler["eligible"]
        assert not transform_features(rows, scaler, variant)["invalid_forecast_mask"].any()
    invalid = fit_feature_scaler(rows, audit, "price_vol")
    assert not invalid["eligible"]
    assert invalid["invalid_training_forecast_dates"] == ["2012-06-04"]
    assert invalid["vol_mean"] == 0 and invalid["vol_scale"] == 1
    assert not invalid["vol_fitted"]
    assert invalid["vol_scaler_status"] == "invalid_training_forecasts"


def test_replay_failure_is_one_date_not_hindsight_annual_fallback(frame):
    rows, audit = _training(frame)
    scaler = fit_feature_scaler(rows, audit, "price_vol")
    replay = frame.loc["2013-06-03":"2013-06-07"].copy()
    replay.loc["2013-06-05", "future_vol_prediction"] = -1
    result = transform_features(replay, scaler, "price_vol")
    assert result["invalid_forecast_mask"].tolist() == [False, False, True, False, False]
    assert np.isfinite(result["features"]).all()
    assert result["features"][2, 25] == 0
    assert result["audit"]["invalid_forecast_dates"] == ["2013-06-05"]
    changed_labels = replay.copy()
    changed_labels[["soxl_return", "soxx_return"]] = 999
    unchanged = transform_features(changed_labels, scaler, "price_vol")
    np.testing.assert_array_equal(result["features"], unchanged["features"])
    assert result["audit"] == unchanged["audit"]


@pytest.mark.parametrize("field,value,error", [
    (PRICE_FEATURES[3], np.nan, "nonfinite common"),
    ("soxx_return", -1., "impossible mature simple returns"),
    ("soxl_return", np.nan, "nonfinite common"),
    ("return_available_at", pd.NaT, "cannot silently drop"),
])
def test_invalid_common_mature_rows_abort_instead_of_filtering(frame, field, value, error):
    frame.loc["2012-06-04", field] = value
    with pytest.raises(ValueError, match=error):
        _training(frame)


def test_immutable_audit_and_scaler_bindings_and_insufficient_history(frame):
    rows, audit = _training(frame)
    changed = rows.copy()
    changed.iloc[0, 2] += .01
    with pytest.raises(ValueError, match="exact mature training audit"):
        fit_feature_scaler(changed, audit, "price")
    scaler = fit_feature_scaler(rows, audit, "price")
    scaler["price_mean"][0] += 1
    with pytest.raises(ValueError, match="frozen scaler identity"):
        transform_features(rows, scaler, "price")
    dataset = build_anchor_dataset(frame, as_of="2014-01-10")
    dataset.anchor_source_contract["frame_sha256"] = "changed"
    with pytest.raises(ValueError, match="native Dataset changed"):
        prepare_training_rows(dataset, 2013)
    with pytest.raises(ValueError, match="insufficient strictly mature"):
        prepare_training_rows(build_anchor_dataset(frame, as_of="2014-01-10"), 2011)
    with pytest.raises(TypeError, match="native Qlib DatasetH"):
        prepare_training_rows(frame, 2013)


def test_entirely_unavailable_head_is_not_fabricated_as_a_fitted_scaler(frame):
    frame["forecast_available"] = 0.
    frame["future_vol_prediction"] = np.inf  # unused values are neutral, not estimates
    frame["forecast_model_available_at"] = pd.NaT
    rows, audit = _training(frame)
    scaler = fit_feature_scaler(rows, audit, "price_vol")
    assert scaler["eligible"] and not scaler["vol_fitted"]
    assert scaler["vol_scaler_status"] == "unavailable_prior_forecasts"
    transformed = transform_features(rows, scaler, "price_vol")
    assert not transformed["invalid_forecast_mask"].any()
    assert (transformed["features"][:, 25:] == 0).all()
    ready = rows.iloc[:1].copy()
    ready["forecast_available"] = 1.
    ready["future_vol_prediction"] = .02
    ready["forecast_model_available_at"] = pd.Timestamp("2011-01-01", tz="America/New_York")
    assert transform_features(ready, scaler, "price_vol")["invalid_forecast_mask"].all()

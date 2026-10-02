from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.data.conditional_variance import (
    COVARIANCE_FIELDS, VARIANTS, apply_calibration, build_calibration_dataset,
    build_raw_variances, calibrate_annual, load_qualified_primary,
    primary_frame_from_qualified_qlib,
)
from src.data.price_basis import FIELDS, PUBLISHED_BASIS, _sha, price_frame_sha256
from src.data.qlib_native import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame


@pytest.fixture(scope="module")
def market():
    # A synthetic calendar is written/read through Qlib in the loader test;
    # production never constructs exchange sessions using business weekdays.
    dates = pd.bdate_range("2010-01-01", "2017-01-31", name="datetime")
    t = np.arange(len(dates))
    parts = []
    for symbol, amplitude in (("SOXL", .026), ("SOXX", .012)):
        returns = .0003 + amplitude * (np.sin(t * .31) + .25 * np.cos(t * .113))
        opening = 20 * np.exp(np.cumsum(returns))
        closing = opening * (1 + .004 * np.sin(t * .17))
        frame = pd.DataFrame({"$open": opening, "$close": closing,
                              "$high": np.maximum(opening, closing) * 1.01,
                              "$low": np.minimum(opening, closing) * .99,
                              "$volume": 100000. * np.exp(.2 * np.sin(t * .23))}, index=dates)
        frame["instrument"] = symbol
        parts.append(frame.reset_index().set_index(["instrument", "datetime"]))
    raw = pd.concat(parts).loc[:, ["$" + field for field in FIELDS]].sort_index()
    primary, prices, audit = primary_frame_from_qualified_qlib(raw, dates)
    forecast = primary.copy()
    forecast["forecast_available"] = (forecast.index.year >= 2013).astype(float)
    forecast["future_vol_prediction"] = .014 + .002 * np.sin(np.arange(len(forecast)) * .13)
    forecast.loc[forecast.forecast_available.eq(0), "future_vol_prediction"] = 0
    risk = build_raw_variances(forecast, prices)
    return raw, dates, forecast, prices, risk


def _dataset(frame, risk):
    return build_calibration_dataset(frame, risk, as_of="2017-01-31")


def test_log_volume_features_are_split_scale_invariant(market):
    raw, calendar, _, _, _ = market
    changed = raw.copy()
    changed["$volume"] *= 60
    before, _, _ = primary_frame_from_qualified_qlib(raw, calendar)
    after, _, _ = primary_frame_from_qualified_qlib(changed, calendar)
    np.testing.assert_allclose(before[["soxl_volume_z20", "soxx_volume_z20"]],
                               after[["soxl_volume_z20", "soxx_volume_z20"]], rtol=0, atol=2e-11)
    changed.iloc[0, changed.columns.get_loc("$volume")] = 0
    with pytest.raises(ValueError, match="positive finite qualified volume"):
        primary_frame_from_qualified_qlib(changed, calendar)


def test_covariance_and_observed_controls_are_causal_and_have_distinct_price_clocks(market):
    _, _, frame, prices, risk = market
    point = pd.Timestamp("2015-06-01")
    changed = prices.copy()
    changed.loc[changed.index > point] *= np.linspace(1.3, 5, (changed.index > point).sum())[:, None]
    after = build_raw_variances(frame, changed)
    np.testing.assert_array_equal(risk.loc[:point, [*COVARIANCE_FIELDS, "m_open20", "m_close20"]],
                                  after.loc[:point, [*COVARIANCE_FIELDS, "m_open20", "m_close20"]])
    changed = prices.copy()
    changed.loc[point, ("close", "SOXX")] *= 1.1
    clock = build_raw_variances(frame, changed)
    assert clock.loc[point, "m_open20"] == risk.loc[point, "m_open20"]
    assert clock.loc[point, "m_close20"] != risk.loc[point, "m_close20"]
    expected = prices[("open", "SOXX")].pct_change(fill_method=None).loc[:point].tail(20).var(ddof=1)
    assert risk.loc[point, "m_open20"] == pytest.approx(expected)


def test_common_native_calibration_budget_means_and_strict_maturity(market):
    _, _, frame, _, risk = market
    original = calibrate_annual(_dataset(frame, risk), 2016)
    common = pd.to_datetime(original["audit"]["common_dates"])
    selected_returns = frame.loc[common, "soxx_return"].to_numpy()
    assert original["B"] == pytest.approx(np.mean((selected_returns - selected_returns.mean()) ** 2))
    for name in VARIANTS:
        assert original["c"][name] * risk.loc[common, f"m_{name}"].mean() == pytest.approx(original["B"])
    assert original["audit"]["h5_mean_rows"] > len(common)
    changed = frame.copy()
    boundary = common[-1]
    changed.loc[boundary, "forecast_available_at"] = pd.Timestamp("2016-01-01", tz="America/New_York")
    modified = calibrate_annual(_dataset(changed, risk), 2016)
    assert str(boundary.date()) not in modified["audit"]["common_dates"]
    assert modified["audit"]["common_rows"] == len(common) - 1
    future = frame.copy()
    future.loc[future.index.year >= 2016, ["soxx_return", "soxl_return_h5", "soxx_return_h5"]] = 999
    unchanged = calibrate_annual(_dataset(future, risk), 2016)
    assert unchanged == original


def test_invalid_eligible_estimator_is_not_dropped_or_squared_into_validity(market):
    _, _, frame, prices, risk = market
    original = calibrate_annual(_dataset(frame, risk), 2016)
    point = pd.Timestamp(original["audit"]["common_dates"][50])
    changed = frame.copy()
    changed.loc[point, "future_vol_prediction"] = -.03
    bad_risk = build_raw_variances(changed, prices)
    assert pd.isna(bad_risk.loc[point, "m_learned"])
    audited = calibrate_annual(_dataset(changed, bad_risk), 2016)
    assert audited["audit"]["common_dates"] == original["audit"]["common_dates"]
    assert audited["B"] == original["B"]
    assert audited["c"]["learned"] is None
    assert audited["calibration_invalid_dates"]["learned"] == [str(point.date())]
    future = bad_risk.loc[bad_risk.index.year == 2016]
    fallback = apply_calibration(audited, future, "learned")
    baseline = apply_calibration(audited, future, "expanding")
    np.testing.assert_array_equal(fallback["covariance"], baseline["covariance"])
    assert fallback["fallback"].all()
    assert fallback["audit"]["scientifically_disqualified"]
    broken = frame.copy()
    broken.loc[point, "soxx_return"] = np.nan
    with pytest.raises(ValueError, match="rows cannot be dropped"):
        calibrate_annual(_dataset(broken, risk), 2016)


def test_fixed_budget_does_not_cancel_conditional_risk_and_support_only_flags(market):
    _, _, frame, _, risk = market
    calibration = calibrate_annual(_dataset(frame, risk), 2016)
    sample = risk.loc[risk.index.year == 2016].iloc[:3].copy()
    result = apply_calibration(calibration, sample, "learned")
    changed = sample.copy()
    changed["m_learned"] *= 100
    changed["future_vol_prediction"] *= 10
    more_risk = apply_calibration(calibration, changed, "learned")
    np.testing.assert_allclose(more_risk["covariance"], result["covariance"] * 100, rtol=1e-14)
    np.testing.assert_array_equal(more_risk["benchmark_variance"], result["benchmark_variance"])
    original_max_soxx = np.sqrt(result["benchmark_variance"] / result["covariance"][:, 1, 1])
    changed_max_soxx = np.sqrt(more_risk["benchmark_variance"] / more_risk["covariance"][:, 1, 1])
    np.testing.assert_allclose(changed_max_soxx, original_max_soxx / 10)
    assert more_risk["out_of_support"].all()
    assert not more_risk["fallback"].any()
    changed.iloc[0, changed.columns.get_loc("future_vol_prediction")] = -1
    invalid = apply_calibration(calibration, changed, "learned")
    control = apply_calibration(calibration, sample, "expanding")
    np.testing.assert_array_equal(invalid["covariance"][0], control["covariance"][0])
    assert invalid["fallback"].tolist() == [True, False, False]


def test_invalid_common_control_and_future_calibration_use_fail_closed(market):
    _, _, frame, _, risk = market
    calibration = calibrate_annual(_dataset(frame, risk), 2016)
    with pytest.raises(ValueError, match="before its cutoff"):
        apply_calibration(calibration, risk.loc["2015-12-01":"2015-12-31"], "expanding")
    bad = risk.loc["2016-01-01":"2016-01-10"].copy()
    bad.iloc[0, bad.columns.get_loc("cov_soxx_soxx")] = 0
    with pytest.raises(ValueError, match="invalid common expanding covariance"):
        apply_calibration(calibration, bad, "learned")
    with pytest.raises(TypeError, match="native Qlib DatasetH"):
        calibrate_annual(frame, 2016)
    other_predictions = risk.copy()
    other_predictions.iloc[0, other_predictions.columns.get_loc("future_vol_prediction")] += .001
    with pytest.raises(ValueError, match="different forecasts"):
        _dataset(frame, other_predictions)


def test_estimator_overflow_falls_back_but_common_overflow_raises(market):
    _, _, frame, _, risk = market
    calibration = calibrate_annual(_dataset(frame, risk), 2016)
    calibration["c"]["expanding"] = .5
    calibration["c"]["learned"] = 1.
    extreme = risk.loc[risk.index.year == 2016].iloc[:1].copy()
    extreme.loc[:, list(COVARIANCE_FIELDS)] = [1e308, 0., .0001]
    extreme["m_expanding"] = .0001
    extreme["future_vol_prediction"] = 1.
    extreme["m_learned"] = 1.
    result = apply_calibration(calibration, extreme, "learned")
    assert result["fallback"].all()
    assert result["covariance"][0, 0, 0] == 5e307
    calibration["c"]["expanding"] = 10.
    with pytest.raises(ValueError, match="invalid common calibrated covariance"):
        apply_calibration(calibration, extreme, "learned")


@pytest.mark.parametrize("column,value,error", [
    ("soxx_return", -2., "invalid mature calibration return"),
    ("soxl_return", -1., "invalid mature calibration return"),
    ("soxl_return_h5", -2., "invalid strictly mature H5"),
    ("future_vol_5d", np.nan, "invalid mature volatility label"),
    ("future_vol_5d", -.001, "invalid mature volatility label"),
])
def test_impossible_or_missing_mature_labels_block_without_dropping(market, column, value, error):
    _, _, frame, _, risk = market
    changed = frame.copy()
    changed.loc["2014-06-02", column] = value
    with pytest.raises(ValueError, match=error):
        calibrate_annual(_dataset(changed, risk), 2016)


def test_zero_realized_volatility_is_physically_valid(market):
    _, _, frame, _, risk = market
    changed = frame.copy()
    changed.loc["2014-06-02", "future_vol_5d"] = 0.
    result = calibrate_annual(_dataset(changed, risk), 2016)
    assert "2014-06-02" in result["audit"]["common_dates"]
    assert pd.Timestamp(result["audit"]["h5_label_available_at_max"]) < pd.Timestamp(result["cutoff"])


def test_common_dates_require_h5_maturity_even_if_vol_metadata_is_earlier(market):
    _, _, frame, _, risk = market
    before = calibrate_annual(_dataset(frame, risk), 2016)
    changed = frame.copy()
    point = pd.Timestamp("2014-06-02")
    changed.loc[point, "h5_available_at"] = pd.Timestamp("2016-01-01", tz="America/New_York")
    after = calibrate_annual(_dataset(changed, risk), 2016)
    assert str(point.date()) in before["audit"]["common_dates"]
    assert str(point.date()) not in after["audit"]["common_dates"]
    assert after["audit"]["common_rows"] == before["audit"]["common_rows"] - 1
    assert after["audit"]["h5_mean_rows"] == before["audit"]["h5_mean_rows"] - 1


def test_native_loader_binds_parent_recorder_marker_calendar_and_content(market, tmp_path, monkeypatch):
    from qlib.workflow import R

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    raw, calendar, _, _, _ = market
    provider = tmp_path / "provider"
    source = raw.reset_index().rename(columns={"$" + field: field for field in FIELDS})
    build_qlib_store_from_frame(source, provider_uri=provider, as_of="2017-01-31", field_map={f: f for f in FIELDS})
    loaded = load_qlib_feature_frame(QlibProviderConfig(provider_uri=provider), instruments=["SOXL", "SOXX"],
                                    fields=["$" + f for f in FIELDS], start="2010-01-01", end="2017-01-31").sort_index()
    manifest = {"qualification": {"qualification_pass": True, "as_of": "2017-01-31", "required_full_start": "2010-01-01"},
                "reference_calendar": {"dates": calendar.strftime("%Y-%m-%d").tolist()},
                "native_readback": {"frame_sha256": price_frame_sha256(loaded)}}
    (provider / "price_basis_manifest.json").write_text(json.dumps(manifest))
    (provider / "price_basis_status.json").write_text(json.dumps({"status": "qualified", "manifest_sha256": _sha(manifest)}))
    uri = str(tmp_path / "recorder")
    with R.start(experiment_name="synthetic_price_parent", uri=uri):
        recorder = R.get_recorder()
        R.save_objects(artifact_path="data", **{"provider_manifest.pkl": manifest})
        recorder_id = recorder.id
    summary = {"recorder_id": recorder_id, "experiment": "synthetic_price_parent", "recorder_uri": uri,
               "basis": PUBLISHED_BASIS, "source_qualification_pass": True, "data_as_of": "2017-01-31",
               "provider_uri": "provider", "native_readback": manifest["native_readback"]}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    kwargs = dict(provider_uri="provider", parent_summary="summary.json", expected_recorder_id=recorder_id, as_of="2017-01-31")
    result = load_qualified_primary(tmp_path, **kwargs)
    assert result["audit"]["native_frame_sha256"] == price_frame_sha256(loaded)
    assert not result["primary_frame"].empty
    with pytest.raises(ValueError, match="parent card identity"):
        load_qualified_primary(tmp_path, **(kwargs | {"expected_recorder_id": "different"}))
    feature = provider / "features/soxl/open.day.bin"
    values = np.frombuffer(feature.read_bytes(), dtype="<f4").copy()
    values[50] += 1
    feature.write_bytes(values.tobytes())
    with pytest.raises(ValueError, match="native content hash changed"):
        load_qualified_primary(tmp_path, **kwargs)

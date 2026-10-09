from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.data import oos_horizon_inputs as inputs
from src.data.conditional_variance import primary_frame_from_qualified_qlib
from src.data.convex_allocation import FEATURE_COLUMNS, build_horizon_dataset, predict_annual
from src.data.integrated_allocation import frame_sha256
from src.data.qlib_native import build_qlib_store_from_frame


@pytest.fixture(scope="module")
def native_source(tmp_path_factory):
    from qlib.data import D

    calendar = pd.bdate_range("2010-08-02", periods=1400)
    rng = np.random.default_rng(1009)
    common = rng.normal(.0003, .012, len(calendar))
    rows = []
    for symbol, multiplier in (("SOXX", 1.), ("SOXL", 2.7)):
        opened = 100. * np.cumprod(1. + multiplier * common + rng.normal(0, .001, len(calendar)))
        closed = opened * (1. + rng.normal(0, .003, len(calendar)))
        for i, day in enumerate(calendar):
            rows.append({"instrument": symbol, "datetime": day, "open": opened[i], "close": closed[i],
                "high": max(opened[i], closed[i]) * 1.01, "low": min(opened[i], closed[i]) * .99,
                "volume": 1_000_000. + i * 701. + float(rng.integers(1, 100000))})
    provider = tmp_path_factory.mktemp("oos_native") / "provider"
    build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=provider, as_of=str(calendar[-1].date()))
    raw = D.features(["SOXL", "SOXX"], ["$open", "$high", "$low", "$close", "$volume"],
        start_time=calendar[0], end_time=calendar[-1]).sort_index()
    frame, prices, audit = primary_frame_from_qualified_qlib(raw, calendar, train_start="2011-01-03")
    bundle = {"primary_frame": frame, "execution_prices": prices, "calendar": calendar,
              "audit": {"source": "synthetic native Qlib readback; qualification loader independently tested",
                        "features": audit}}
    plan = {"source": {"provider_uri": str(provider), "summary": "synthetic_summary.json",
        "recorder_id": "synthetic_only", "as_of": str(calendar[-1].date())}, "train_start": "2011-01-03",
        "label_precision": "float64", "fit_years": [2013, 2014], "ridge_alpha": 10.,
        "minimum_training_rows": 252, "hac_lags": 5}
    return provider, bundle, plan


def _prepare(native_source, monkeypatch, *, source=None, plan=None):
    import qlib

    provider, original, registered = native_source
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    # This seam isolates transformation from the separately enforced parent
    # qualification identity; all feature/price observations still came from D.
    monkeypatch.setattr(inputs, "load_qualified_primary", lambda *args, **kwargs: deepcopy(original if source is None else source))
    return inputs.prepare_horizon_inputs(provider.parent, deepcopy(registered if plan is None else plan))


def test_native_source_to_dataseth_is_exact_and_labels_use_float64_with_separate_clocks(native_source, monkeypatch):
    from qlib.data.dataset import DatasetH

    _, source, plan = native_source
    before = deepcopy(source)
    bundle = _prepare(native_source, monkeypatch)
    frame, prices = bundle["frame"], source["execution_prices"].astype(np.float64)
    assert isinstance(bundle["dataset"], DatasetH)
    prepared = bundle["dataset"].prepare("all", col_set=["feature", "label", "metadata"])
    prepared.columns = prepared.columns.get_level_values(1)
    prepared.index = pd.DatetimeIndex(prepared.index.get_level_values("datetime"))
    pd.testing.assert_frame_equal(prepared, frame.loc[:, prepared.columns], check_exact=True, check_freq=False)
    for symbol in ("SOXL", "SOXX"):
        opening = prices[("open", symbol)]
        for offset, suffix in ((2, "return"), (6, "return_h5")):
            column = f"{symbol.lower()}_{suffix}"
            expected = (opening.shift(-offset) / opening.shift(-1) - 1).reindex(frame.index)
            assert frame[column].dtype == np.dtype("float64")
            np.testing.assert_array_equal(frame[column].to_numpy(), expected.to_numpy())
    assert frame[["soxl_return_h5", "soxx_return_h5"]].tail(6).isna().all().all()
    assert len(bundle["dataset"].prepare("all")) == len(frame)
    assert set(frame.decision_at.dt.strftime("%H:%M")) == {"16:15"}
    assert set(frame.return_available_at.dropna().dt.strftime("%H:%M")) == {"09:31"}
    assert set(frame.h5_available_at.dropna().dt.strftime("%H:%M")) == {"09:31"}
    pd.testing.assert_series_equal(frame.forecast_available_at, frame.h5_available_at, check_names=False)
    assert bundle["audit"]["original_frame_sha256"] == frame_sha256(source["primary_frame"])
    assert bundle["audit"]["transformed_frame_sha256"] != bundle["audit"]["original_frame_sha256"]
    assert bundle["audit"]["dataset_sha256"] == frame_sha256(prepared)
    pd.testing.assert_frame_equal(source["primary_frame"], before["primary_frame"], check_exact=True)
    pd.testing.assert_frame_equal(source["execution_prices"], before["execution_prices"], check_exact=True)


def test_annual_forecast_rows_and_vmean_are_bound_to_same_sorted_signal_indices(native_source, monkeypatch):
    _, _, plan = native_source
    bundle = _prepare(native_source, monkeypatch)
    predictions, vmean, models, audits = inputs.fit_annual_oos(bundle, plan)
    expected_dates = bundle["frame"].index[bundle["frame"].index.year.isin(plan["fit_years"])]
    pd.testing.assert_index_equal(predictions.index, expected_dates)
    assert predictions.index.is_monotonic_increasing and not predictions.index.has_duplicates
    assert vmean.shape == (len(predictions), 2, 2)
    assert set(models) == set(plan["fit_years"])
    for year in plan["fit_years"]:
        part = bundle["frame"].loc[bundle["frame"].index.year == year]
        reference = predict_annual(models[year], part)
        mask = predictions.index.year == year
        np.testing.assert_array_equal(predictions.loc[mask, ["predicted_soxl", "predicted_soxx"]], reference["ridge"])
        np.testing.assert_array_equal(vmean[mask], reference["Vmean"])
        assert predictions.loc[mask, "model_year"].eq(year).all()
        assert predictions.loc[mask, "training_frame_sha256"].eq(models[year]["audit"]["training_frame_sha256"]).all()
        pd.testing.assert_series_equal(predictions.loc[mask, "forecast_available_at"], part.decision_at, check_names=False)
        assert pd.Timestamp(models[year]["audit"]["last_label_available_at"]) < pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    assert len(audits) == len(plan["fit_years"])


def test_future_labels_and_features_cannot_change_earlier_annual_model_or_oos_predictions(native_source, monkeypatch):
    _, _, original_plan = native_source
    plan = deepcopy(original_plan)
    plan["fit_years"] = [2013]
    bundle = _prepare(native_source, monkeypatch)
    expected, vmean, models, _ = inputs.fit_annual_oos(bundle, plan)
    changed = bundle["frame"].copy(deep=True)
    cutoff = pd.Timestamp("2013-01-01", tz="America/New_York")
    changed.loc[changed.h5_available_at.ge(cutoff), ["soxl_return_h5", "soxx_return_h5"]] = [.8, -.8]
    changed.loc[changed.index.year >= 2014, list(FEATURE_COLUMNS)] = 1000.
    changed_bundle = {**bundle, "frame": changed, "dataset": build_horizon_dataset(changed, as_of=plan["source"]["as_of"])}
    actual, later_vmean, later_models, _ = inputs.fit_annual_oos(changed_bundle, plan)
    for key in ("feature_mean", "feature_scale", "ridge_coefficients", "coefficient_covariance_hac", "prior_mean"):
        np.testing.assert_array_equal(models[2013][key], later_models[2013][key])
    columns = ["predicted_soxl", "predicted_soxx", "training_frame_sha256", "forecast_available_at"]
    pd.testing.assert_frame_equal(expected[columns], actual[columns], check_exact=True)
    np.testing.assert_array_equal(vmean, later_vmean)
    assert models[2013]["audit"]["training_frame_sha256"] == later_models[2013]["audit"]["training_frame_sha256"]


@pytest.mark.parametrize("years", [[], [2014, 2013], [2013, 2013], [2013.5], [True], [2030]])
def test_empty_missing_or_ambiguous_fit_years_fail_closed(native_source, monkeypatch, years):
    _, _, original = native_source
    plan = deepcopy(original)
    plan["fit_years"] = years
    bundle = _prepare(native_source, monkeypatch)
    with pytest.raises(ValueError, match="blocked:"):
        inputs.fit_annual_oos(bundle, plan)


def test_source_calendar_mismatch_is_rejected(native_source, monkeypatch):
    _, original, _ = native_source
    source = deepcopy(original)
    source["execution_prices"] = source["execution_prices"].iloc[:-1]
    with pytest.raises(ValueError, match="qualified price calendar differs"):
        _prepare(native_source, monkeypatch, source=source)


def test_dataseth_readback_drift_is_detected(native_source, monkeypatch):
    real_builder = inputs.build_horizon_dataset

    def altered_builder(frame, **kwargs):
        dataset = real_builder(frame, **kwargs)
        prepare = dataset.prepare

        def corrupt_readback(*args, **kw):
            result = prepare(*args, **kw).copy()
            result.iloc[0, 0] += 1.
            return result

        monkeypatch.setattr(dataset, "prepare", corrupt_readback)
        return dataset

    monkeypatch.setattr(inputs, "build_horizon_dataset", altered_builder)
    with pytest.raises(AssertionError):
        _prepare(native_source, monkeypatch)

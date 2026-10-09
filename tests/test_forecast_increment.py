"""Synthetic maturity, paired-loss and interpretation contracts only."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis.forecast_increment import analyze_forecast_increment


@pytest.fixture
def forecasts():
    dates = pd.bdate_range("2024-01-02", periods=18)
    x = np.array([-.12, .05, -.03, .08, -.01, .04, -.07, .09, .02, -.02, .11, -.05])
    frame = pd.DataFrame({"signal_date": dates[:12],
        "label_available_at": [day.to_pydatetime() + timedelta(hours=9, minutes=31)
                               for day in dates[6:].tz_localize("America/New_York")],
        "model_year": 2024})
    for asset, scale in (("soxl", 1.), ("soxx", .4)):
        y = x * scale
        frame[f"realized_{asset}"] = y
        frame[f"base_{asset}"] = y + .03
        frame[f"full_{asset}"] = y + .01
        frame[f"prior_{asset}"] = 0.
    return frame


def analyze(frame, **kwargs):
    params = {"evaluation_cutoff": pd.Timestamp("2024-03-01", tz="UTC"),
              "development_years": (2024,), "mean_blocks": (2, 3), "resamples": 200}
    params.update(kwargs)
    return analyze_forecast_increment(frame, **params)


def test_known_mse_increment_calibration_and_exact_error_decomposition(forecasts):
    original = forecasts.copy(deep=True)
    result = analyze(forecasts)
    metrics = result["metrics"].query("window == 'development'")
    full = metrics.query("model == 'full'")
    np.testing.assert_allclose(full.mse, .0001, atol=1e-15)
    np.testing.assert_allclose(full.calibration_slope, 1., atol=1e-14)
    np.testing.assert_allclose(full.calibration_intercept, -.01, atol=1e-14)
    np.testing.assert_allclose(full.pearson_ic, 1., atol=1e-14)
    np.testing.assert_allclose(full.spearman_ic, 1., atol=1e-14)
    np.testing.assert_allclose(metrics.mse, metrics.squared_error_bias + metrics.error_variance_ddof0, atol=1e-15)
    inc = result["increments"].query("window == 'development' and reference == 'base'")
    np.testing.assert_allclose(inc.mse_improvement_reference_minus_full, .0008, atol=1e-15)
    np.testing.assert_allclose(inc.mse_improvement_reference_minus_full,
                               inc.squared_bias_improvement + inc.error_variance_improvement, atol=1e-15)
    assert metrics.query("model == 'prior'").pearson_ic.isna().all()
    assert metrics.query("model == 'prior'").calibration_slope.isna().all()
    assert result["audit"]["promotion_eligible"] is False
    assert result["audit"]["calibration_applied_to_predictions"] is False
    pd.testing.assert_frame_equal(forecasts, original, check_exact=True)


def test_paired_bootstrap_constant_improvement_and_family_are_explicit(forecasts):
    result = analyze(forecasts)["bootstrap"]
    base = result.query("reference == 'base'")
    np.testing.assert_allclose(base.mse_improvement_lower_bound, .0008, atol=1e-15)
    assert base.lower_bound_positive.all()
    assert result.declared_family_size.eq(4).all()
    assert result.per_comparison_alpha.eq(.0125).all()
    assert set(result.mean_block) == {2, 3}
    pd.testing.assert_frame_equal(result, analyze(forecasts)["bootstrap"], check_exact=True)


def test_strict_cutoff_excludes_equal_time_and_unmatured_suffix_values(forecasts):
    cutoff = forecasts.label_available_at.iloc[8]
    first = analyze(forecasts, evaluation_cutoff=cutoff)
    altered = forecasts.copy()
    altered.loc[8:, ["realized_soxl", "realized_soxx"]] = np.nan
    altered.loc[10:, "label_available_at"] = pd.NaT
    second = analyze(altered, evaluation_cutoff=cutoff)
    assert first["audit"]["mature_rows"] == second["audit"]["mature_rows"] == 8
    assert second["audit"]["terminal_unmatured_rows"] == 4
    for key in ("metrics", "increments", "bootstrap"):
        pd.testing.assert_frame_equal(first[key], second[key], check_exact=True)


@pytest.mark.parametrize("case", ["missing_mature_label", "missing_prediction", "missing_interior_time",
    "naive_time", "future_model_year", "duplicate_signal", "unordered_signal", "noncausal_label", "late_signal"])
def test_invalid_rows_never_silently_disappear(forecasts, case):
    frame = forecasts.copy()
    if case == "missing_mature_label":
        frame.loc[3, "realized_soxx"] = np.nan
    elif case == "missing_prediction":
        frame.loc[3, "full_soxl"] = np.nan
    elif case == "missing_interior_time":
        frame.loc[3, "label_available_at"] = pd.NaT
    elif case == "naive_time":
        frame["label_available_at"] = frame.label_available_at.dt.tz_localize(None)
    elif case == "future_model_year":
        frame.loc[3, "model_year"] = 2025
    elif case == "duplicate_signal":
        frame.loc[3, "signal_date"] = frame.signal_date.iloc[2]
    elif case == "unordered_signal":
        frame = frame.iloc[::-1]
    elif case == "noncausal_label":
        frame.loc[0, "label_available_at"] = pd.Timestamp("2024-01-02 09:31", tz="America/New_York")
    else:
        frame.loc[11, "signal_date"] = pd.Timestamp("2024-04-01")
    with pytest.raises(ValueError, match="blocked:"):
        analyze(frame)


def test_missing_terminal_prediction_is_rejected_even_if_label_is_unmatured(forecasts):
    cutoff = forecasts.label_available_at.iloc[8]
    forecasts.loc[11, "prior_soxx"] = np.nan
    with pytest.raises(ValueError, match="nonfinite forecast"):
        analyze(forecasts, evaluation_cutoff=cutoff)


def test_identical_forecasts_have_no_invented_positive_increment(forecasts):
    for asset in ("soxl", "soxx"):
        forecasts[f"full_{asset}"] = forecasts[f"base_{asset}"]
    result = analyze(forecasts)
    assert result["increments"].query("reference == 'base'").mse_improvement_reference_minus_full.eq(0).all()
    boot = result["bootstrap"].query("reference == 'base'")
    assert boot.mse_improvement_lower_bound.eq(0).all()
    assert not boot.lower_bound_positive.any()


def test_annual_and_reused_scopes_preserve_the_same_matched_rows(forecasts):
    later = forecasts.copy()
    later["signal_date"] = [day.replace(year=2025) for day in later.signal_date]
    later["label_available_at"] = [day.replace(year=2025) for day in later.label_available_at]
    later["model_year"] = 2025
    combined = pd.concat([forecasts, later], ignore_index=True)
    result = analyze(combined, evaluation_cutoff=pd.Timestamp("2025-03-01", tz="UTC"))
    assert set(result["metrics"].window) == {"all", "development", "reused_diagnostic", "year_2024", "year_2025"}
    assert result["metrics"].query("window == 'all'").rows.eq(24).all()
    assert result["metrics"].query("window != 'all'").rows.eq(12).all()
    assert set(result["bootstrap"].window) == {"development", "reused_diagnostic"}
    assert result["audit"]["mature_rows"] == 24


def test_explicit_aware_cutoff_and_bootstrap_sample_size_are_required(forecasts):
    with pytest.raises(ValueError, match="aware timestamp"):
        analyze(forecasts, evaluation_cutoff="2024-03-01")
    with pytest.raises(ValueError, match="insufficient mature"):
        analyze(forecasts, mean_blocks=(20,))

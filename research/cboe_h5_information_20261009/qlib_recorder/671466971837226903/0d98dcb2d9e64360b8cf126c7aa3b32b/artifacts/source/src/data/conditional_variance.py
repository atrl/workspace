"""Causal, common-date variance calibration on a qualified native Qlib basis.

This module neither fits forecasts nor optimizes portfolios. Five-session mean
rewards and one-session covariance/budget units remain explicitly separate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.convex_allocation import LABEL_COLUMNS as H5_LABELS, build_horizon_frame
from src.data.integrated_allocation import (
    PRIMARY_FEATURES, PRIMARY_LABELS, PRIMARY_METADATA, build_native_dataset,
    frame_sha256, primary_frame_from_qlib,
)
from src.data.price_basis import FIELDS, INSTRUMENTS, PUBLISHED_BASIS, _sha, price_frame_sha256
from src.data.qlib_native import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame
from src.workflow.benchmark_risk_excess import COVARIANCE_FIELDS, expanding_covariance

VARIANTS = ("expanding", "open20", "close20", "learned")
MARGINAL_FIELDS = tuple(f"m_{name}" for name in VARIANTS)
RISK_UNITS = "one_session_simple_return_squared"
MEAN_UNITS = "five_session_simple_return"


def primary_frame_from_qualified_qlib(raw, calendar, *, train_start="2011-01-03"):
    """Reuse the primary transform with log(volume), not log1p(volume).

    Positive source volume is mandatory. A later multiplicative split adjustment
    adds a constant to log volume and leaves its rolling z-score unchanged.
    Window/minimum/std/zero-variance convention are otherwise unchanged.
    """
    volume = raw["$volume"].to_numpy(float)
    if not np.isfinite(volume).all() or (volume <= 0).any():
        raise ValueError("blocked: log-volume features require positive finite qualified volume")
    frame, prices = primary_frame_from_qlib(raw, raw, calendar, train_start=train_start)
    for symbol in INSTRUMENTS:
        log_volume = np.log(raw.xs(symbol, level="instrument")["$volume"].reindex(calendar).astype(float))
        rolling = log_volume.rolling(20, min_periods=10)
        frame[f"{symbol.lower()}_volume_z20"] = ((log_volume - rolling.mean()) / (rolling.std(ddof=0) + 1e-8)).reindex(frame.index)
    if not np.isfinite(frame[list(PRIMARY_FEATURES)].to_numpy(float)).all():
        raise ValueError("blocked: incomplete qualified causal features")
    frame, horizon_audit = build_horizon_frame(frame, prices)
    return frame, prices, {"primary_frame_sha256": frame_sha256(frame),
        "feature_columns": list(PRIMARY_FEATURES), "price_basis": PUBLISHED_BASIS,
        "volume_z20": "(log(volume)-rolling20_mean)/(rolling20_std_ddof0+1e-8); min_periods10",
        "volume_feature_change": "log1p replaced by log before any new fitting; no other feature recipe change",
        "horizon": horizon_audit}


def load_qualified_primary(root, *, provider_uri, parent_summary, expected_recorder_id,
                           as_of, train_start="2011-01-03") -> dict[str, Any]:
    """Read D features only after card/Recorder/marker identity checks."""
    from qlib.data import D
    from qlib.workflow import R

    root = Path(root).resolve()
    provider = (root / provider_uri).resolve()
    summary_path = root / parent_summary
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    manifest_bytes = (provider / "price_basis_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    marker = json.loads((provider / "price_basis_status.json").read_text())
    if (summary.get("recorder_id") != expected_recorder_id or summary.get("basis") != PUBLISHED_BASIS
            or summary.get("source_qualification_pass") is not True or summary.get("data_as_of") != as_of
            or (root / summary["provider_uri"]).resolve() != provider):
        raise ValueError("blocked: qualified parent card identity mismatch")
    if (marker.get("status") != "qualified" or marker.get("manifest_sha256") != _sha(manifest)
            or manifest["qualification"].get("qualification_pass") is not True
            or manifest["qualification"].get("as_of") != as_of):
        raise ValueError("blocked: qualified provider marker or as_of mismatch")
    init_qlib_provider(QlibProviderConfig(provider_uri=provider))
    with R.uri_context(summary["recorder_uri"]):
        recorder = R.get_recorder(experiment_name=summary["experiment"], recorder_id=expected_recorder_id)
        recorded_manifest = recorder.load_object("data/provider_manifest.pkl")
    if _sha(recorded_manifest) != _sha(manifest):
        raise ValueError("blocked: provider manifest differs from parent Qlib Recorder")
    start = manifest["qualification"]["required_full_start"]
    raw = load_qlib_feature_frame(QlibProviderConfig(provider_uri=provider), instruments=list(INSTRUMENTS),
                                 fields=["$" + field for field in FIELDS], start=start, end=as_of).sort_index()
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    expected_dates = manifest["reference_calendar"]["dates"]
    if calendar.strftime("%Y-%m-%d").tolist() != expected_dates or str(calendar[-1].date()) != as_of:
        raise ValueError("blocked: qualified provider full calendar changed")
    if sorted(D.list_instruments(D.instruments("all"), as_list=True)) != sorted(INSTRUMENTS):
        raise ValueError("blocked: qualified provider instrument universe changed")
    native_hash = price_frame_sha256(raw)
    if (native_hash != manifest["native_readback"]["frame_sha256"]
            or native_hash != summary["native_readback"]["frame_sha256"]):
        raise ValueError("blocked: qualified provider native content hash changed")
    if ((provider / "price_basis_manifest.json").read_bytes() != manifest_bytes
            or summary_path.read_bytes() != summary_bytes
            or json.loads((provider / "price_basis_status.json").read_text()) != marker):
        raise ValueError("blocked: source metadata changed during native read")
    frame, prices, feature_audit = primary_frame_from_qualified_qlib(raw, calendar, train_start=train_start)
    return {"primary_frame": frame, "execution_prices": prices, "raw_ohlcv": raw, "calendar": calendar,
        "audit": {"parent_recorder_id": expected_recorder_id, "parent_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "provider_uri": str(provider), "provider_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "native_frame_sha256": native_hash, "as_of": as_of, "features": feature_audit,
            "source_api": ["qlib.workflow.R.get_recorder", "qlib.data.D.calendar", "qlib.data.D.instruments", "qlib.data.D.features"]}}


def _dates(frame):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("blocked: ordered unique native session dates are required")


def build_raw_variances(forecast_frame, execution_prices) -> pd.DataFrame:
    """Observed prices through signal OPEN/CLOSE; supplied annual OOF forecasts."""
    _dates(forecast_frame)
    _dates(execution_prices)
    prices = execution_prices.loc[:, [(mark, symbol) for mark in ("open", "close") for symbol in INSTRUMENTS]]
    if not np.isfinite(prices.to_numpy(float)).all() or (prices <= 0).any().any():
        raise ValueError("blocked: incomplete positive native risk prices")
    if not forecast_frame.index.isin(prices.index).all():
        raise ValueError("blocked: forecast dates missing from native prices")
    covariance = expanding_covariance(prices)
    result = covariance.reindex(forecast_frame.index).copy()
    result["m_expanding"] = result["cov_soxx_soxx"]
    for mark in ("open", "close"):
        returns = prices[(mark, "SOXX")].astype(float).pct_change(fill_method=None)
        result[f"m_{mark}20"] = returns.rolling(20, min_periods=20).var(ddof=1).reindex(result.index)
    available = pd.to_numeric(forecast_frame["forecast_available"], errors="coerce")
    if not available.isin([0., 1.]).all():
        raise ValueError("blocked: forecast_available must explicitly indicate 0 or 1")
    prediction = pd.to_numeric(forecast_frame["future_vol_prediction"], errors="coerce")
    valid = available.eq(1) & np.isfinite(prediction) & prediction.gt(0)
    with np.errstate(over="ignore", invalid="ignore"):
        result["m_learned"] = prediction.pow(2).where(valid)
    result["forecast_available"] = available
    result["future_vol_prediction"] = prediction
    result["forecast_valid"] = valid & np.isfinite(result.m_learned) & result.m_learned.gt(0)
    # annual_forecasts fits at January 1; allow an explicit earlier frozen model
    # timestamp from the orchestrator rather than infer a later refit.
    if "forecast_model_available_at" in forecast_frame:
        model_time = pd.to_datetime(forecast_frame["forecast_model_available_at"], utc=True)
    else:
        model_time = pd.Series([pd.Timestamp(f"{date.year}-01-01", tz="America/New_York") for date in result.index], index=result.index)
    result["forecast_model_available_at"] = model_time.where(available.eq(1))
    known = pd.to_datetime(forecast_frame.feature_available_at, utc=True)
    if (available.eq(1) & (pd.to_datetime(result.forecast_model_available_at, utc=True).gt(known)
                           | result.forecast_model_available_at.isna())).any():
        raise ValueError("blocked: forecast model was unavailable at signal time")
    result.attrs["risk_source"] = {"expanding": covariance.attrs["source"], "units": RISK_UNITS,
        "open20_last_observation": "signal-day open", "close20_last_observation": "signal-day close",
        "windows": 20, "rolling_ddof": 1, "negative_vol_predictions_squared": False,
        "raw_variances_sha256": frame_sha256(result)}
    return result


def build_calibration_dataset(forecast_frame, raw_variances, *, as_of):
    if not forecast_frame.index.equals(raw_variances.index):
        raise ValueError("blocked: calibration inputs have different session rows")
    forecast_columns = ["future_vol_prediction", "forecast_available"]
    if not np.array_equal(forecast_frame[forecast_columns].to_numpy(float),
                          raw_variances[forecast_columns].to_numpy(float), equal_nan=True):
        raise ValueError("blocked: calibration frame and risk table contain different forecasts")
    frame = forecast_frame.copy()
    for column in (*COVARIANCE_FIELDS, *MARGINAL_FIELDS, "forecast_model_available_at"):
        frame[column] = raw_variances[column]
    features = (*PRIMARY_FEATURES, *COVARIANCE_FIELDS, *MARGINAL_FIELDS)
    labels = (*PRIMARY_LABELS, *H5_LABELS)
    metadata = (*PRIMARY_METADATA, "h5_exit_date", "h5_available_at", "forecast_model_available_at")
    return build_native_dataset(frame, feature_columns=features, label_columns=labels, metadata_columns=metadata, as_of=as_of)


def _covariance_array(frame):
    values = frame.loc[:, list(COVARIANCE_FIELDS)].to_numpy(float)
    matrices = np.empty((len(values), 2, 2), dtype=float)
    matrices[:, 0, 0], matrices[:, 0, 1] = values[:, 0], values[:, 1]
    matrices[:, 1, 0], matrices[:, 1, 1] = values[:, 1], values[:, 2]
    return matrices


def _check_common_covariance(frame):
    matrices = _covariance_array(frame)
    if not np.isfinite(matrices).all() or (matrices[:, 1, 1] <= 0).any() or (matrices[:, 0, 0] < 0).any():
        raise ValueError("blocked: invalid common expanding covariance; no fallback is possible")
    scale = np.max(np.abs(matrices), axis=(1, 2))
    if (np.linalg.eigvalsh(matrices)[:, 0] < -256 * np.finfo(float).eps * scale).any():
        raise ValueError("blocked: materially non-PSD common covariance")
    if not np.array_equal(frame["m_expanding"].to_numpy(float), matrices[:, 1, 1]):
        raise ValueError("blocked: expanding marginal differs from common covariance")
    return matrices


def calibrate_annual(dataset, year, *, min_rows=252, train_start="2011-01-03") -> dict[str, Any]:
    """Fit moments only on strictly mature DatasetH rows; never drop bad data."""
    from qlib.data.dataset import DatasetH

    if not isinstance(dataset, DatasetH):
        raise TypeError("blocked: annual calibration requires native Qlib DatasetH")
    cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"]).copy()
    prepared.columns = [column[-1] for column in prepared.columns]
    prepared.index = pd.DatetimeIndex(prepared.index.get_level_values("datetime"))
    _dates(prepared)
    eligible = prepared.loc[prepared.index >= pd.Timestamp(train_start)].copy()
    known = pd.to_datetime(eligible.feature_available_at, utc=True).lt(cutoff)
    h5_mature = known & pd.to_datetime(eligible.h5_available_at, utc=True).lt(cutoff)
    means = eligible.loc[h5_mature, list(H5_LABELS)]
    if len(means) < min_rows or not np.isfinite(means.to_numpy(float)).all() or (means.to_numpy(float) <= -1).any():
        raise ValueError("blocked: insufficient or invalid strictly mature H5 prior-mean rows")
    available = pd.to_numeric(eligible.forecast_available, errors="coerce")
    if not available.isin([0., 1.]).all():
        raise ValueError("blocked: invalid OOF availability indicator")
    common_mask = known & available.eq(1)
    for name in ("return_available_at", "forecast_available_at", "h5_available_at", "forecast_model_available_at"):
        common_mask &= pd.to_datetime(eligible[name], utc=True).lt(cutoff)
    common = eligible.loc[common_mask].copy()
    if len(common) < min_rows:
        raise ValueError("blocked: insufficient common mature OOF calibration dates")
    _check_common_covariance(common)
    daily_labels = common[["soxl_return", "soxx_return"]].to_numpy(float)
    if not np.isfinite(daily_labels).all() or (daily_labels <= -1).any():
        raise ValueError("blocked: invalid mature calibration return; rows cannot be dropped")
    vol_labels = common.future_vol_5d.to_numpy(float)
    if not np.isfinite(vol_labels).all() or (vol_labels < 0).any():
        raise ValueError("blocked: invalid mature volatility label; rows cannot be dropped")
    returns = daily_labels[:, 1]
    target_mean = float(returns.mean())
    budget = float(np.mean(np.square(returns - target_mean)))
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("blocked: common annual risk budget must be positive finite")
    coefficients, support, invalid_dates = {}, {}, {}
    for variant in VARIANTS:
        values = common[f"m_{variant}"].to_numpy(float)
        valid = np.isfinite(values) & (values > 0)
        if variant == "learned":
            prediction = common.future_vol_prediction.to_numpy(float)
            valid &= np.isfinite(prediction) & (prediction > 0)
        invalid_dates[variant] = common.index[~valid].strftime("%Y-%m-%d").tolist()
        coefficients[variant] = float(budget / values.mean()) if valid.all() else None
        support[variant] = {"min": float(values.min()), "max": float(values.max())} if valid.all() else None
        coefficient = coefficients[variant]
        if coefficient is not None and (not np.isfinite(coefficient) or coefficient <= 0):
            coefficients[variant] = None
            invalid_dates[variant] = common.index.strftime("%Y-%m-%d").tolist()
        if variant == "expanding" and coefficients[variant] is None:
            raise ValueError("blocked: common expanding calibration is invalid")
    dates = common.index.strftime("%Y-%m-%d").tolist()
    return {"year": int(year), "cutoff": cutoff.isoformat(), "B": budget, "c": coefficients,
        "mu_H5": means.mean().to_numpy(float).tolist(), "support": support,
        "calibration_invalid_dates": invalid_dates, "calibration_available": {v: coefficients[v] is not None for v in VARIANTS},
        "risk_units": RISK_UNITS, "mean_units": MEAN_UNITS,
        "audit": {"common_dates": dates, "common_dates_sha256": _sha(dates), "common_rows": len(common),
            "common_frame_sha256": frame_sha256(common), "minimum_rows": int(min_rows),
            "target_soxx_return_mean": target_mean, "budget_ddof": 0,
            "h5_mean_rows": len(means), "h5_mean_frame_sha256": frame_sha256(means),
            "h5_first_date": str(means.index[0].date()), "h5_last_date": str(means.index[-1].date()),
            "h5_label_available_at_max": pd.to_datetime(eligible.loc[h5_mature, "h5_available_at"], utc=True).max().isoformat(),
            "label_available_at_max": pd.to_datetime(common.forecast_available_at, utc=True).max().isoformat(),
            "maturity_rule": "feature, 1d return, H5/vol5 outcome and forecast model availability strictly before Jan1",
            "normalization": "same B and same common dates for all c_j = B / mean(raw_marginal_j)",
            "invalid_rows_dropped": 0, "source_api": "qlib.data.dataset.DatasetH.prepare"}}


def apply_calibration(calibration, raw_variances, variant) -> dict[str, Any]:
    """Scale covariance with fixed annual B; invalid estimators only account via fallback."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variance variant: {variant}")
    _dates(raw_variances)
    if (raw_variances.index < pd.Timestamp(calibration["cutoff"]).tz_localize(None)).any():
        raise ValueError("blocked: annual calibration cannot be used before its cutoff")
    common = _check_common_covariance(raw_variances)
    budget = float(calibration["B"])
    control_c = calibration["c"]["expanding"]
    if not np.isfinite(budget) or budget <= 0 or control_c is None or not np.isfinite(control_c) or control_c <= 0:
        raise ValueError("blocked: invalid common annual calibration")
    marginal = raw_variances[f"m_{variant}"].to_numpy(float)
    valid = np.isfinite(marginal) & (marginal > 0)
    if variant == "learned":
        prediction = raw_variances.future_vol_prediction.to_numpy(float)
        valid &= raw_variances.forecast_available.eq(1).to_numpy() & np.isfinite(prediction) & (prediction > 0)
    coefficient = calibration["c"][variant]
    coefficient_valid = coefficient is not None and np.isfinite(coefficient) and coefficient > 0
    fallback = ~valid if coefficient_valid else np.ones(len(common), dtype=bool)
    factor = np.full(len(common), float(control_c))
    if coefficient_valid and variant != "expanding":
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            proposed = coefficient * marginal / common[:, 1, 1]
        bad_factor = ~np.isfinite(proposed) | (proposed <= 0)
        fallback |= bad_factor
        factor[~fallback] = proposed[~fallback]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        control = common * float(control_c)
        scaled = common * factor[:, None, None]
    if not np.isfinite(control).all() or (control[:, 1, 1] <= 0).any():
        raise ValueError("blocked: invalid common calibrated covariance")
    bad_covariance = ~np.isfinite(scaled).all(axis=(1, 2)) | (scaled[:, 1, 1] <= 0)
    fallback |= bad_covariance
    scaled[fallback] = control[fallback]
    bounds = calibration["support"][variant]
    support_flag = valid & ((marginal < bounds["min"]) | (marginal > bounds["max"])) if bounds else np.zeros(len(common), dtype=bool)
    dates = raw_variances.index
    return {"covariance": scaled, "benchmark_variance": np.full(len(common), budget), "fallback": fallback,
        "out_of_support": support_flag, "audit": {"variant": variant, "rows": len(common),
            "fallback_count": int(fallback.sum()), "fallback_dates": dates[fallback].strftime("%Y-%m-%d").tolist(),
            "estimator_integrity_pass": not bool(fallback.any()), "scientifically_disqualified": bool(fallback.any()),
            "support_flag_count": int(support_flag.sum()),
            "support_flag_dates": dates[support_flag].strftime("%Y-%m-%d").tolist(), "support_flags_change_exposure": False,
            "B": budget, "c": coefficient, "calibration_cutoff": calibration["cutoff"], "risk_units": RISK_UNITS,
            "fallback_rule": "same-date calibrated expanding covariance, accounting only; any fallback disqualifies",
            "formula": "Sigma_j = (c_j*m_j/Sigma_expanding_SOXX_SOXX)*Sigma_expanding; B fixed annually"}}

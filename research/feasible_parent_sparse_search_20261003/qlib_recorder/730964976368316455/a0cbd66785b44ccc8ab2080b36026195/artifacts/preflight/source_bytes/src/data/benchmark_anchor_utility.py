"""Native mature daily training rows and frozen inputs for anchored utility.

Forecast failures are variant-specific. They never remove an observation, alter
common return labels, or retroactively replace a year's valid decisions.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from src.data.integrated_allocation import (
    PRIMARY_FEATURES, PRIMARY_METADATA, RETURN_LABELS, build_native_dataset, frame_sha256,
)

PRICE_FEATURES = tuple(PRIMARY_FEATURES[:-3])
FEATURE_COLUMNS = (*PRICE_FEATURES, "future_vol_prediction", "forecast_available")
METADATA_COLUMNS = (*PRIMARY_METADATA, "forecast_model_available_at")
VARIANTS = ("constant", "price", "price_vol")
SCHEMA_VERSION = "benchmark_anchor_features_v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _dates(frame):
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates
            or not frame.index.is_monotonic_increasing or frame.index.tz is not None):
        raise ValueError("blocked: expected unique ordered timezone-naive Qlib session dates")


def _variant(value):
    if value not in VARIANTS:
        raise ValueError(f"blocked: unknown anchor variant {value}")


def build_anchor_dataset(frame: pd.DataFrame, *, as_of: str):
    """Preserve all rows, including invalid/unavailable extra forecast values."""
    _dates(frame)
    if frame.empty or frame.index.max() > pd.Timestamp(as_of):
        raise ValueError("blocked: empty anchor source or dates beyond as_of")
    selected_columns = [*FEATURE_COLUMNS, *RETURN_LABELS, *METADATA_COLUMNS]
    missing = set(selected_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"blocked: anchor source columns missing {sorted(missing)}")
    selected = frame.loc[:, selected_columns].copy()
    selected.index.name = "datetime"
    dataset = build_native_dataset(selected, feature_columns=FEATURE_COLUMNS, label_columns=RETURN_LABELS,
                                   metadata_columns=METADATA_COLUMNS, as_of=as_of)
    dataset.anchor_source_contract = {
        "schema_version": SCHEMA_VERSION, "as_of": as_of,
        "rows": len(selected), "dates": selected.index.strftime("%Y-%m-%d").tolist(),
        "columns": selected_columns, "frame_sha256": frame_sha256(selected),
    }
    return dataset


def _native_frame(dataset):
    from qlib.data.dataset import DatasetH

    if not isinstance(dataset, DatasetH) or not hasattr(dataset, "anchor_source_contract"):
        raise TypeError("blocked: anchor preparation requires its native Qlib DatasetH contract")
    frame = dataset.prepare("all", col_set=["feature", "label", "metadata"]).copy()
    frame.columns = [column[-1] for column in frame.columns]
    if frame.index.get_level_values("instrument").nunique() != 1:
        raise ValueError("blocked: anchor Dataset has an unexpected instrument grain")
    frame.index = pd.DatetimeIndex(frame.index.get_level_values("datetime"), name="datetime")
    _dates(frame)
    contract = dataset.anchor_source_contract
    if (list(frame.columns) != contract["columns"] or len(frame) != contract["rows"]
            or frame.index.strftime("%Y-%m-%d").tolist() != contract["dates"]
            or frame_sha256(frame) != contract["frame_sha256"]):
        raise ValueError("blocked: native Dataset changed source rows, values or date identity")
    return frame, contract


def _numeric(frame, columns):
    try:
        values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError("blocked: common anchor inputs are not numeric") from error
    if not np.isfinite(values).all():
        raise ValueError("blocked: nonfinite common anchor inputs; rows cannot be dropped")
    return values


def _aware_times(series, *, field, missing_ok=True):
    for value in series.loc[series.notna()]:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            raise ValueError(f"blocked: {field} needs explicit timezone information")
    converted = pd.to_datetime(series, utc=True, errors="raise")
    if not missing_ok and converted.isna().any():
        raise ValueError(f"blocked: missing {field}")
    return converted


def prepare_training_rows(dataset, year: int, *, min_rows=252, train_start="2011-01-03"):
    """Exact daily-label maturity; H5/vol-target availability does not purge rows."""
    frame, contract = _native_frame(dataset)
    cutoff = pd.Timestamp(f"{int(year)}-01-01", tz="America/New_York")
    feature_time = _aware_times(frame.feature_available_at, field="feature_available_at", missing_ok=False)
    return_time = _aware_times(frame.return_available_at, field="return_available_at")
    if (return_time.isna() & frame.exit_date.notna()).any():
        raise ValueError("blocked: observed exit date lacks return availability; cannot silently drop a row")
    eligible = (frame.index >= pd.Timestamp(train_start)) & feature_time.lt(cutoff) & return_time.lt(cutoff)
    rows = frame.loc[eligible].copy()
    if len(rows) < min_rows:
        raise ValueError("blocked: insufficient strictly mature daily anchor training rows")
    _numeric(rows, PRICE_FEATURES)
    returns = _numeric(rows, RETURN_LABELS)
    if (returns <= -1).any():
        raise ValueError("blocked: impossible mature simple returns; rows cannot be dropped")
    signal_dates = pd.to_datetime(rows.signal_date)
    if not pd.DatetimeIndex(signal_dates).equals(rows.index):
        raise ValueError("blocked: signal dates differ from native Dataset index")
    decision = _aware_times(rows.decision_at, field="decision_at", missing_ok=False)
    if not (feature_time.loc[rows.index].lt(decision) & decision.lt(return_time.loc[rows.index])).all():
        raise ValueError("blocked: causal feature/decision/outcome chronology is invalid")
    dates = rows.index.strftime("%Y-%m-%d").tolist()
    audit = {
        "schema_version": SCHEMA_VERSION, "model_year": int(year), "cutoff": cutoff.isoformat(),
        "train_start": train_start, "as_of": contract["as_of"], "minimum_rows": int(min_rows),
        "source_rows": len(frame), "eligible_rows": len(rows), "excluded_unmatured_or_prestart": int((~eligible).sum()),
        "eligible_dates": dates, "eligible_dates_sha256": _hash(dates),
        "source_frame_sha256": contract["frame_sha256"], "training_rows_sha256": frame_sha256(rows),
        "first_signal_date": dates[0], "last_signal_date": dates[-1],
        "feature_available_at_max": feature_time.loc[rows.index].max().isoformat(),
        "return_available_at_max": return_time.loc[rows.index].max().isoformat(),
        "maturity_rule": "features and one-day outcomes strictly before fit-year Jan1 America/New_York",
        "h5_or_vol_target_availability_used_for_policy_maturity": False,
        "invalid_rows_dropped": 0, "source_api": "qlib.data.dataset.DatasetH.prepare",
    }
    return rows, audit


def _forecast_status(rows):
    """Bad available forecasts/metadata are flagged; unavailable values are unused."""
    availability = pd.to_numeric(rows["forecast_available"], errors="coerce").to_numpy(float)
    prediction = pd.to_numeric(rows["future_vol_prediction"], errors="coerce").to_numpy(float)
    valid_flag = np.isfinite(availability) & np.isin(availability, [0., 1.])
    available = valid_flag & (availability == 1.)
    reasons = [[] for _ in range(len(rows))]
    for i in np.flatnonzero(~valid_flag):
        reasons[i].append("invalid_forecast_availability_indicator")
    for i in np.flatnonzero(available & (~np.isfinite(prediction) | (prediction <= 0))):
        reasons[i].append("available_forecast_nonfinite_or_nonpositive")
    # This check is deliberately local to used forecasts. Unused extra metadata
    # cannot invalidate a constant/price-only policy.
    for i in np.flatnonzero(available):
        try:
            fitted = pd.Timestamp(rows.iloc[i]["forecast_model_available_at"])
            known = pd.Timestamp(rows.iloc[i]["feature_available_at"])
            if pd.isna(fitted) or pd.isna(known) or fitted.tzinfo is None or known.tzinfo is None or fitted > known:
                reasons[i].append("forecast_model_unavailable_at_signal")
        except (ValueError, TypeError, KeyError):
            reasons[i].append("forecast_model_unavailable_at_signal")
    invalid = np.array([bool(reason) for reason in reasons])
    return prediction, available, invalid, reasons


def _mean_scale(values):
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("blocked: scaler requires finite observed training values")
    constant = np.ptp(values, axis=0) == 0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
    mean[constant] = values[0, constant]
    scale[constant] = 1.
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("blocked: nonconstant scaler has an invalid mean/scale; no clipping")
    return mean, scale, constant


def fit_feature_scaler(rows, audit, variant):
    """Serializable prior-only scaler; invalid used training forecasts tag a null fold."""
    _variant(variant)
    _dates(rows)
    if (frame_sha256(rows) != audit["training_rows_sha256"] or len(rows) != audit["eligible_rows"]
            or rows.index.strftime("%Y-%m-%d").tolist() != audit["eligible_dates"]):
        raise ValueError("blocked: scaler rows differ from the exact mature training audit")
    mean, scale, constant = _mean_scale(_numeric(rows, PRICE_FEATURES))
    invalid = np.zeros(len(rows), dtype=bool)
    reasons = [[] for _ in range(len(rows))]
    available = np.zeros(len(rows), dtype=bool)
    vol_mean, vol_scale, vol_fitted = 0., 1., False
    if variant == "price_vol":
        prediction, available, invalid, reasons = _forecast_status(rows)
        if available.any() and not invalid.any():
            means, scales, _ = _mean_scale(prediction[available, None])
            vol_mean, vol_scale, vol_fitted = float(means[0]), float(scales[0]), True
    status = ("unused" if variant != "price_vol" else "invalid_training_forecasts" if invalid.any()
              else "fitted" if vol_fitted else "unavailable_prior_forecasts")
    result = {
        "schema_version": SCHEMA_VERSION, "variant": variant, "feature_columns": list(FEATURE_COLUMNS),
        "eligible": not bool(invalid.any()), "price_mean": mean.tolist(), "price_scale": scale.tolist(),
        "constant_price_columns": [PRICE_FEATURES[i] for i in np.flatnonzero(constant)],
        "vol_mean": vol_mean, "vol_scale": vol_scale, "vol_fitted": vol_fitted,
        "vol_scaler_status": status, "available_training_forecast_rows": int(available.sum()),
        "invalid_training_forecast_dates": rows.index[invalid].strftime("%Y-%m-%d").tolist(),
        "invalid_training_forecast_reasons": {str(rows.index[i].date()): reasons[i] for i in np.flatnonzero(invalid)},
        "cutoff": audit["cutoff"], "model_year": audit["model_year"], "training_rows": len(rows),
        "training_rows_sha256": audit["training_rows_sha256"], "eligible_dates_sha256": audit["eligible_dates_sha256"],
        "price_scaler_ddof": 0, "availability_scaled": False, "disabled_channels_masked_after_scaling": True,
        "unfitted_vol_values_are_neutral_placeholders": not vol_fitted,
    }
    return result | {"scaler_sha256": _hash(result)}


def transform_features(rows, scaler, variant):
    """Finite model inputs and a *per-date* forecast failure mask, without row removal.

    Workflow responsibility: a training-ineligible scaler anchors its entire
    fitted fold. A valid scaler with a bad replay forecast forces SOXX only on
    that date, records disqualification, then resumes on actual carried holdings.
    """
    _variant(variant)
    _dates(rows)
    payload = {key: value for key, value in scaler.items() if key != "scaler_sha256"}
    if (scaler.get("schema_version") != SCHEMA_VERSION or scaler.get("variant") != variant
            or scaler.get("feature_columns") != list(FEATURE_COLUMNS) or _hash(payload) != scaler.get("scaler_sha256")):
        raise ValueError("blocked: frozen scaler identity or variant mismatch")
    values = _numeric(rows, PRICE_FEATURES)
    mean, scale = np.asarray(scaler["price_mean"], float), np.asarray(scaler["price_scale"], float)
    if mean.shape != (25,) or scale.shape != (25,) or not np.isfinite(np.r_[mean, scale]).all() or (scale <= 0).any():
        raise ValueError("blocked: invalid frozen price scaler")
    result = np.zeros((len(rows), 27), dtype=float)
    result[:, :25] = (values - mean) / scale
    invalid = np.zeros(len(rows), dtype=bool)
    reasons = [[] for _ in range(len(rows))]
    if variant == "price_vol":
        prediction, available, invalid, reasons = _forecast_status(rows)
        if not scaler["vol_fitted"]:
            for i in np.flatnonzero(available):
                reasons[i].append("no_fitted_prior_volatility_scaler")
            invalid |= available
        usable = available & ~invalid
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (prediction[usable] - float(scaler["vol_mean"])) / float(scaler["vol_scale"])
        positions = np.flatnonzero(usable)
        nonfinite = ~np.isfinite(standardized)
        for i in positions[nonfinite]:
            reasons[i].append("nonfinite_standardized_forecast")
            invalid[i] = True
        result[positions[~nonfinite], 25] = standardized[~nonfinite]
        result[:, 26] = available.astype(float)
    elif variant == "constant":
        result[:, :] = 0.
    if not np.isfinite(result).all():
        raise ValueError("blocked: nonfinite standardized common inputs; no feature clipping")
    input_columns = list(PRICE_FEATURES)
    if variant == "price_vol":
        input_columns += ["future_vol_prediction", "forecast_available", "forecast_model_available_at", "feature_available_at"]
    input_identity = frame_sha256(rows.loc[:, input_columns])
    return {"features": result, "invalid_forecast_mask": invalid,
        "audit": {"schema_version": SCHEMA_VERSION, "variant": variant, "rows": len(rows),
            "scaler_sha256": scaler["scaler_sha256"], "scaler_training_eligible": scaler["eligible"],
            "input_frame_sha256": input_identity, "features_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
            "invalid_forecast_count": int(invalid.sum()),
            "invalid_forecast_dates": rows.index[invalid].strftime("%Y-%m-%d").tolist(),
            "invalid_forecast_reasons": {str(rows.index[i].date()): reasons[i] for i in np.flatnonzero(invalid)},
            "row_count_preserved": True, "forecast_failure_action": "workflow must force SOXX at invalid dates; no retroactive annual replacement"}}

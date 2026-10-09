"""Annual Qlib H5 Ridge estimators with feature-independent maturity cohorts.

Feature choices are supplied by the caller. Both the baseline and the extended
model consume the same date/label cohort; invalid mature feature values raise
instead of silently changing that cohort. No feature or parameter search runs.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.convex_allocation import _hac_coefficient_covariance, _psd_roundoff
from src.data.integrated_allocation import frame_sha256


LABEL_COLUMNS = ("soxl_return_h5", "soxx_return_h5")
METADATA_COLUMNS = ("signal_date", "feature_available_at", "external_available_at",
                    "h5_available_at", "decision_at", "trade_date", "exit_date", "return_available_at")
AVAILABILITY_COLUMNS = ("feature_available_at", "external_available_at", "h5_available_at", "return_available_at")


def _year(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= 9999:
        raise ValueError("blocked: model and signal years must be integer calendar years")
    return int(value)


def _features(columns):
    if isinstance(columns, str):
        raise ValueError("blocked: feature_columns must be a sequence of unique feature names")
    try:
        names = tuple(columns)
    except TypeError as error:
        raise ValueError("blocked: feature_columns must be a sequence of unique feature names") from error
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or set(names) & set((*LABEL_COLUMNS, *METADATA_COLUMNS))):
        raise ValueError("blocked: feature_columns must be unique non-label feature names")
    return names


def _sessions(values, *, name, allow_missing=False):
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values))
    except (TypeError, ValueError) as error:
        raise ValueError(f"blocked: invalid {name} session dates") from error
    if dates.tz is not None or (dates.hasnans and not allow_missing) or not dates.equals(dates.normalize()):
        raise ValueError(f"blocked: {name} requires timezone-naive midnight session dates")
    return dates


def _aware(values, *, name, allow_missing=False):
    parsed = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"blocked: invalid {name} availability") from error
        if pd.isna(timestamp):
            if not allow_missing:
                raise ValueError(f"blocked: {name} must have finite aware availability")
            parsed.append(pd.NaT)
        elif timestamp.tzinfo is None:
            raise ValueError(f"blocked: {name} requires explicit timezone")
        else:
            parsed.append(timestamp.tz_convert("UTC"))
    return pd.DatetimeIndex(pd.to_datetime(parsed, utc=True))


def _clock(dates, hour, minute=0):
    return (pd.DatetimeIndex(dates).tz_localize("America/New_York")
            + pd.Timedelta(hours=hour, minutes=minute)).tz_convert("UTC")


def _prepare(dataset, feature_columns, *, labels):
    from qlib.data.dataset import DatasetH

    if not isinstance(dataset, DatasetH):
        raise ValueError("blocked: matched annual model requires native Qlib DatasetH")
    groups = ["feature", "label", "metadata"] if labels else ["feature", "metadata"]
    prepared = dataset.prepare("all", col_set=groups)
    if prepared.empty or not isinstance(prepared.columns, pd.MultiIndex) or prepared.columns.has_duplicates:
        raise ValueError("blocked: empty or ambiguous native Dataset schema")
    expected = [("feature", name) for name in feature_columns] + [("metadata", name) for name in METADATA_COLUMNS]
    if labels:
        expected += [("label", name) for name in LABEL_COLUMNS]
    if set(expected) - set(prepared.columns):
        raise ValueError("blocked: native Dataset lacks registered features, labels or availability metadata")
    if isinstance(prepared.index, pd.MultiIndex):
        if "datetime" not in prepared.index.names:
            raise ValueError("blocked: native Dataset needs a datetime index")
        dates = _sessions(prepared.index.get_level_values("datetime"), name="source")
    else:
        dates = _sessions(prepared.index, name="source")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("blocked: source rows must have unique ordered signal dates")
    metadata = prepared["metadata"].loc[:, list(METADATA_COLUMNS)].copy()
    signals = _sessions(metadata.signal_date, name="signal")
    if not signals.equals(dates):
        raise ValueError("blocked: signal metadata differs from native Dataset index")
    trade = _sessions(metadata.trade_date, name="trade", allow_missing=True)
    exit_dates = _sessions(metadata.exit_date, name="exit", allow_missing=True)
    times = {name: _aware(metadata[name], name=name, allow_missing=name in ("h5_available_at", "return_available_at"))
             for name in (*AVAILABILITY_COLUMNS, "decision_at")}
    if not times["feature_available_at"].equals(_clock(signals, 16)):
        raise ValueError("blocked: price features must be available at signal-day 16:00")
    if not times["decision_at"].equals(_clock(signals, 16, 15)):
        raise ValueError("blocked: decisions must be at signal-day 16:15")
    external_local = times["external_available_at"].tz_convert("America/New_York")
    external_dates = external_local.tz_localize(None).normalize()
    if (not times["external_available_at"].equals(_clock(external_dates, 17))
            or not (external_dates < signals).all()):
        raise ValueError("blocked: external information must be from a prior session at 17:00")
    if not ((times["feature_available_at"] < times["decision_at"])
            & (times["external_available_at"] < times["decision_at"])).all():
        raise ValueError("blocked: feature information is unavailable at decision time")
    known_trade = ~trade.isna()
    known_exit = ~exit_dates.isna()
    if (not (signals[known_trade] < trade[known_trade]).all()
            or not (times["decision_at"][known_trade] < _clock(trade[known_trade], 9, 30)).all()
            or not (trade[known_exit] < exit_dates[known_exit]).all()):
        raise ValueError("blocked: decision, trade and exit chronology differs")
    if not times["return_available_at"].equals(_clock(exit_dates, 9, 31)):
        raise ValueError("blocked: daily return must mature at exit-session 09:31")
    h5_known = ~times["h5_available_at"].isna()
    if not (times["h5_available_at"][h5_known] > times["return_available_at"][h5_known]).all():
        raise ValueError("blocked: H5 maturity must follow its daily return maturity")
    calendar = getattr(dataset, "calendar", None)
    if calendar is not None:
        calendar = _sessions(calendar, name="native calendar")
        if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
            raise ValueError("blocked: native calendar must be ordered and unique")
        positions = calendar.get_indexer(signals)
        if (positions < 0).any():
            raise ValueError("blocked: signal absent from bound native calendar")
        if ((positions == 0).any()
                or not times["external_available_at"].equals(_clock(calendar[positions - 1], 17))):
            raise ValueError("blocked: external availability must use the prior native session at17:00")

        def shifted(offset):
            shifted_dates = np.full(len(signals), np.datetime64("NaT"), dtype="datetime64[ns]")
            valid = positions + offset < len(calendar)
            shifted_dates[valid] = calendar.to_numpy()[positions[valid] + offset]
            return pd.DatetimeIndex(shifted_dates)

        if not trade.equals(shifted(1)) or not exit_dates.equals(shifted(2)):
            raise ValueError("blocked: trade/exit differ from next native sessions")
        if not times["h5_available_at"].equals(_clock(shifted(6), 9, 31)):
            raise ValueError("blocked: H5 maturity differs from six-session native-calendar clock")
        horizon_scope = "bound native calendar: entry+1, daily exit+2, H5 exit+6 at09:31; unknown tail remains NaT"
    else:
        horizon_scope = "chronology only; caller did not bind a native calendar to DatasetH"
    # Keep the native Qlib index and selected column order for model-specific
    # training identity; availability validation never drops rows or labels.
    selected = [("feature", name) for name in feature_columns]
    if labels:
        selected += [("label", name) for name in LABEL_COLUMNS]
    selected += [("metadata", name) for name in METADATA_COLUMNS]
    return prepared.loc[:, selected].copy(), dates, times, horizon_scope


def fit_matched_horizon_ridge(dataset, *, year: int, feature_columns: Sequence[str],
                             ridge_alpha: float, minimum_rows: int, hac_lags: int,
                             train_start: str) -> dict:
    """Fit annual H5 Ridge on one feature-independent strictly mature cohort."""
    year, features = _year(year), _features(feature_columns)
    if (isinstance(ridge_alpha, (bool, np.bool_)) or not np.isscalar(ridge_alpha)
            or not np.isfinite(ridge_alpha) or ridge_alpha <= 0
            or isinstance(minimum_rows, (bool, np.bool_))
            or not isinstance(minimum_rows, (int, np.integer)) or minimum_rows < len(features) + 2
            or isinstance(hac_lags, (bool, np.bool_))
            or not isinstance(hac_lags, (int, np.integer)) or hac_lags < 0):
        raise ValueError("blocked: invalid fixed Ridge/HAC estimation contract")
    start = _sessions([train_start], name="train_start")[0]
    cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    prepared, dates, times, scope = _prepare(dataset, features, labels=True)
    mature = dates >= start
    for name in AVAILABILITY_COLUMNS:
        mature &= ~times[name].isna() & (times[name] < cutoff)
    train = prepared.loc[mature].copy()
    if len(train) < minimum_rows or hac_lags >= len(train):
        raise ValueError("blocked: insufficient common strictly mature training rows")
    X, Y = train["feature"].to_numpy(dtype=np.float64), train["label"].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all() or not np.isfinite(Y).all() or (Y <= -1).any():
        raise ValueError("blocked: incomplete or invalid mature features/labels; rows cannot be dropped")
    mean, raw_scale = X.mean(axis=0), X.std(axis=0, ddof=0)
    scale = np.where(raw_scale > 1e-12, raw_scale, 1.)
    design = np.c_[np.ones(len(X)), (X - mean) / scale]
    gram = design.T @ design
    penalty = np.diag(np.r_[0., np.full(len(features), ridge_alpha)])
    inverse = np.linalg.solve(gram + penalty, np.eye(len(features) + 1))
    coefficients = inverse @ (design.T @ Y)
    covariance, hac_audit = _hac_coefficient_covariance(design, Y - design @ coefficients, inverse, maxlags=int(hac_lags))
    common_columns = [("label", name) for name in LABEL_COLUMNS] + [("metadata", name) for name in METADATA_COLUMNS]
    common = train.loc[:, common_columns]
    training_dates = dates[mature]
    audit = {"model_year": year, "model_available_at": cutoff.isoformat(), "horizon_sessions": 5,
        "feature_columns": list(features), "training_rows": len(train), "minimum_training_rows": int(minimum_rows),
        "train_start": start.date().isoformat(), "first_training_signal": training_dates[0].date().isoformat(),
        "last_training_signal": training_dates[-1].date().isoformat(),
        "last_label_available_at": times["h5_available_at"][mature].max().isoformat(),
        "latest_available_at": {name: times[name][mature].max().isoformat() for name in AVAILABILITY_COLUMNS},
        "training_frame_sha256": frame_sha256(train), "matched_rows_sha256": frame_sha256(common),
        "label_rows_sha256": frame_sha256(train["label"]),
        "training_dates_sha256": hashlib.sha256(training_dates.asi8.tobytes()).hexdigest(),
        "ridge_alpha": float(ridge_alpha), "ridge_objective": "sum squared errors + alpha*sum squared slopes; unpenalized intercept",
        "scaler": "common prior strictly mature rows; population standard deviation; constants scale1",
        "constant_feature_count": int((raw_scale <= 1e-12).sum()), "hac": hac_audit,
        "maturity_rule": "feature, external, daily return and H5 availability all strictly before January1 NewYork",
        "horizon_validation": scope, "invalid_rows_dropped": 0,
        "mean_covariance_units": "five_session_simple_return_squared", "source_api": "qlib.data.dataset.DatasetH.prepare"}
    return {"model_year": year, "model_available_at": cutoff.isoformat(), "horizon_sessions": 5,
        "feature_columns": features, "feature_mean": mean, "feature_scale": scale, "prior_mean": Y.mean(axis=0),
        "ridge_coefficients": coefficients, "coefficient_covariance_hac": covariance, "audit": audit}


def predict_matched_horizon(fit: dict, dataset, *, signal_year: int) -> dict:
    """Infer using feature/metadata groups only, retaining unknown tail labels."""
    year = _year(signal_year)
    if fit.get("horizon_sessions") != 5 or fit.get("model_year") != year:
        raise ValueError("blocked: forecast year must equal its frozen annual model year")
    features = _features(fit["feature_columns"])
    prepared, dates, times, scope = _prepare(dataset, features, labels=False)
    selected = dates.year == year
    if not selected.any():
        raise ValueError("blocked: registered prediction year has no native rows")
    part, signal_dates = prepared.loc[selected], dates[selected]
    model_at = _aware([fit["model_available_at"]], name="model_available_at")[0]
    expected_model_at = pd.Timestamp(f"{year}-01-01", tz="America/New_York").tz_convert("UTC")
    if model_at != expected_model_at or (times["feature_available_at"][selected] < model_at).any():
        raise ValueError("blocked: annual model cannot produce retrospective forecasts")
    X = part["feature"].to_numpy(dtype=np.float64)
    p = len(features) + 1
    mean, scale = np.asarray(fit["feature_mean"]), np.asarray(fit["feature_scale"])
    coefficients, covariance = np.asarray(fit["ridge_coefficients"]), np.asarray(fit["coefficient_covariance_hac"])
    prior = np.asarray(fit["prior_mean"])
    if (mean.shape != (p - 1,) or scale.shape != (p - 1,) or coefficients.shape != (p, 2)
            or covariance.shape != (2 * p, 2 * p) or prior.shape != (2,)
            or not all(np.isfinite(value).all() for value in (X, mean, scale, coefficients, covariance, prior))
            or (scale <= 0).any()):
        raise ValueError("blocked: invalid frozen model or inference feature values")
    covariance, stored_psd = _psd_roundoff(covariance)
    design = np.c_[np.ones(len(X)), (X - mean) / scale]
    means = design @ coefficients
    projected = np.einsum("ni,aibj,nj->nab", design, covariance.reshape(2, p, 2, p), design)
    repaired, repairs = zip(*[_psd_roundoff(value) for value in projected])
    if not np.isfinite(means).all():
        raise ValueError("blocked: nonfinite matched H5 forecasts")
    prior_means = np.broadcast_to(prior, means.shape).copy()
    table = pd.DataFrame({"signal_date": signal_dates, "forecast_available_at": times["decision_at"][selected],
        "label_available_at": times["h5_available_at"][selected], "model_year": year,
        "predicted_soxl": means[:, 0], "predicted_soxx": means[:, 1],
        "prior_soxl": prior_means[:, 0], "prior_soxx": prior_means[:, 1]}, index=signal_dates)
    table.index.name = "datetime"
    return {"dates": signal_dates, "predictions": table, "ridge_mu": means, "prior_mu": prior_means,
        "Vmean": np.stack(repaired), "audit": {"model_year": year, "model_available_at": model_at.isoformat(),
            "rows": len(part), "feature_columns": list(features),
            "training_frame_sha256": fit["audit"]["training_frame_sha256"],
            "matched_rows_sha256": fit["audit"]["matched_rows_sha256"],
            "inference_frame_sha256": frame_sha256(part), "horizon_validation": scope,
            "labels_requested": False, "unknown_tail_labels_do_not_filter_inference": True,
            "prediction_covariance_clipped_roundoff_eigenvalues": sum(item["clipped_roundoff_eigenvalues"] for item in repairs),
            "maximum_prediction_covariance_adjustment": max(item["maximum_entry_adjustment"] for item in repairs),
            "stored_covariance_psd_audit": stored_psd,
            "interpretation": "conditional H5 mean uncertainty; excludes ridge bias, drift and model-selection uncertainty"}}

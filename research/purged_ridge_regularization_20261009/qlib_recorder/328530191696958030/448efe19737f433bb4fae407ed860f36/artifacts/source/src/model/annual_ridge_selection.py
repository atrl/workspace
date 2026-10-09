"""Nested annual H5 Ridge selection using a completed prior validation year.

All candidates share price-only Qlib rows and one inner information cutoff.
Selection uses normalized forecast error, never outer-year outcomes or returns
from an allocated portfolio. A historical-mean candidate is an explicit model.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.data.convex_allocation import (
    FEATURE_COLUMNS, LABEL_COLUMNS, _hac_coefficient_covariance,
    fit_annual_estimator, predict_annual,
)
from src.data.integrated_allocation import frame_sha256


REQUIRED_METADATA = ("signal_date", "feature_available_at", "decision_at", "trade_date",
                     "exit_date", "return_available_at", "h5_available_at")


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _calendar(values, name, *, allow_missing=False):
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values))
    except (TypeError, ValueError) as error:
        raise ValueError(f"blocked: invalid {name}") from error
    _require(dates.tz is None and (allow_missing or not dates.hasnans)
             and dates.equals(dates.normalize()), f"{name} must be naive native session dates")
    return dates


def _aware(values, name, *, allow_missing=False):
    stamps = []
    for value in values:
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"blocked: invalid {name}") from error
        _require((allow_missing and pd.isna(stamp)) or (not pd.isna(stamp) and stamp.tzinfo is not None),
                 f"{name} needs explicit timezone-aware availability")
        stamps.append(pd.NaT if pd.isna(stamp) else stamp.tz_convert("UTC"))
    return pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))


def _clock(dates, hour, minute=0):
    return (pd.DatetimeIndex(dates).tz_localize("America/New_York")
            + pd.Timedelta(hours=hour, minutes=minute)).tz_convert("UTC")


def _inputs(dataset, frame):
    from qlib.data.dataset import DatasetH

    _require(isinstance(dataset, DatasetH) and getattr(dataset, "horizon_sessions", None) == 5,
             "nested selection requires the native H5 DatasetH")
    _require(isinstance(frame, pd.DataFrame) and not frame.empty and not frame.columns.has_duplicates,
             "nested selection requires a complete dated inference frame")
    calendar = getattr(dataset, "calendar", None)
    _require(calendar is not None, "nested selection requires a bound native calendar")
    calendar = _calendar(calendar, "calendar")
    _require(len(calendar) > 6 and not calendar.has_duplicates and calendar.is_monotonic_increasing,
             "native calendar must be unique and ordered")
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    _require(not prepared.empty and isinstance(prepared.columns, pd.MultiIndex)
             and not prepared.columns.has_duplicates, "native H5 schema is empty or ambiguous")
    _require(tuple(prepared["feature"].columns) == FEATURE_COLUMNS
             and tuple(prepared["label"].columns) == LABEL_COLUMNS
             and set(REQUIRED_METADATA).issubset(prepared["metadata"].columns),
             "nested selection requires the unchanged 25-feature/two-label schema")
    _require(isinstance(prepared.index, pd.MultiIndex) and "datetime" in prepared.index.names,
             "native H5 rows need a datetime index")
    dates = _calendar(prepared.index.get_level_values("datetime"), "source dates")
    _require(not dates.has_duplicates and dates.is_monotonic_increasing,
             "source dates must be unique and ordered")
    given_dates = _calendar(frame.index, "frame dates")
    _require(dates.equals(given_dates), "DatasetH and inference frame dates differ")
    positions = calendar.get_indexer(dates)
    _require((positions >= 0).all() and np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)),
             "source rows must cover every bound native session between their endpoints")
    columns = [*FEATURE_COLUMNS, *LABEL_COLUMNS, *REQUIRED_METADATA]
    _require(set(columns).issubset(frame.columns), "inference frame lacks registered columns")
    flat = prepared.copy()
    flat.columns = flat.columns.get_level_values(1)
    flat.index = dates
    try:
        pd.testing.assert_frame_equal(flat.loc[:, columns], frame.loc[:, columns], check_exact=True,
                                      check_freq=False, check_names=False)
    except AssertionError as error:
        raise ValueError("blocked: DatasetH and frame feature/label/timing values differ") from error
    metadata = prepared["metadata"]
    _require(_calendar(metadata.signal_date, "signal dates").equals(dates), "signal dates differ from native rows")

    def shifted(offset):
        output = np.full(len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        valid = positions + offset < len(calendar)
        output[valid] = calendar.to_numpy()[positions[valid] + offset]
        return pd.DatetimeIndex(output)

    _require(_calendar(metadata.trade_date, "trade dates", allow_missing=True).equals(shifted(1))
             and _calendar(metadata.exit_date, "exit dates", allow_missing=True).equals(shifted(2)),
             "trade and exit must be the next two native sessions")
    times = {name: _aware(metadata[name], name, allow_missing=name in ("h5_available_at", "return_available_at"))
             for name in ("feature_available_at", "decision_at", "return_available_at", "h5_available_at")}
    _require(times["feature_available_at"].equals(_clock(dates, 16))
             and times["decision_at"].equals(_clock(dates, 16, 15)), "price and decision clocks must be16:00/16:15")
    _require(times["return_available_at"].equals(_clock(shifted(2), 9, 31))
             and times["h5_available_at"].equals(_clock(shifted(6), 9, 31)),
             "daily/H5 outcomes must mature at native +2/+6 session09:31")
    return prepared, dates, times


def _training(prepared, dates, times, year, minimum_rows, hac_lags):
    cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    mature = ((dates >= pd.Timestamp("2011-01-03")) & (times["feature_available_at"] < cutoff)
              & (times["h5_available_at"] < cutoff))
    train = prepared.loc[mature].copy()
    _require(len(train) >= minimum_rows and hac_lags < len(train), "insufficient strictly mature training rows")
    X, Y = train["feature"].to_numpy(float), train["label"].to_numpy(float)
    _require(np.isfinite(X).all() and np.isfinite(Y).all() and (Y > -1).all(),
             "invalid mature training values; rows cannot be dropped")
    return train, {"model_year": int(year), "model_available_at": cutoff.isoformat(), "training_rows": len(train),
        "first_training_signal": dates[mature][0].date().isoformat(),
        "last_training_signal": dates[mature][-1].date().isoformat(),
        "last_label_available_at": times["h5_available_at"][mature].max().isoformat(),
        "training_frame_sha256": frame_sha256(train),
        "training_labels_sha256": frame_sha256(train["label"])}


def _historical_mean(train, training_audit, *, hac_lags):
    """Two actual intercept parameters, embedded in the legacy query schema."""
    labels = train["label"].to_numpy(float)
    mean = labels.mean(axis=0)
    hac, hac_audit = _hac_coefficient_covariance(np.ones((len(labels), 1)), labels - mean,
        np.array([[1. / len(labels)]]), maxlags=hac_lags)
    width = len(FEATURE_COLUMNS) + 1
    coefficients = np.zeros((width, 2))
    coefficients[0] = mean
    embedded = np.zeros((2 * width, 2 * width))
    embedded[np.ix_([0, width], [0, width])] = hac
    return {"model_kind": "historical_mean", "model_year": training_audit["model_year"],
        "model_available_at": training_audit["model_available_at"], "horizon_sessions": 5,
        "feature_columns": FEATURE_COLUMNS, "feature_mean": np.zeros(len(FEATURE_COLUMNS)),
        "feature_scale": np.ones(len(FEATURE_COLUMNS)), "prior_mean": mean,
        "ridge_coefficients": coefficients, "coefficient_covariance_hac": embedded,
        "audit": {**training_audit, "model_kind": "historical_mean", "actual_fitted_parameters": 2,
            "coefficients_per_response": 1, "compatibility_coefficients_per_response": width,
            "ridge_alpha": None, "scaler": "not fitted; zero/one compatibility buffers multiply zero slopes",
            "hac": hac_audit, "mean_covariance_units": "five_session_simple_return_squared",
            "source_api": "qlib.data.dataset.DatasetH.prepare",
            "interpretation": "historical training mean with intercept-only HAC; not a SOXX allocation or a fitted25-slope model"}}


def _ridge(dataset, year, alpha, minimum_rows, hac_lags, expected_hash):
    fitted = fit_annual_estimator(dataset, year=year, ridge_alpha=alpha,
        minimum_training_rows=minimum_rows, hac_lags=hac_lags)
    _require(fitted["audit"]["training_frame_sha256"] == expected_hash,
             "Ridge candidates consumed different training X/Y/metadata")
    fitted["model_kind"] = "ridge"
    fitted["audit"]["model_kind"] = "ridge"
    return fitted


def select_annual_ridge(dataset, frame, *, outer_year, alphas=(10., 100., 1000.),
                        minimum_training_rows=252, minimum_validation_rows=126, hac_lags=5) -> dict:
    """Choose prior/Ridge on mature inner-year H5 error, then refit before outer year.

    The score is the mean of each asset's validation MSE divided by that
    asset's inner-training variance (ddof=0). Only exact floating-point ties
    invoke the declared prior-first, descending-alpha order. The outer control
    always remains Ridge10. All eligible validation rows and candidates remain.
    """
    _require(isinstance(outer_year, (int, np.integer)) and not isinstance(outer_year, (bool, np.bool_))
             and 2 <= outer_year <= 9999, "outer_year must be an integer calendar year")
    try:
        registered_alphas = tuple(alphas)
        _require(not any(isinstance(alpha, (bool, np.bool_)) for alpha in registered_alphas),
                 "boolean values are not Ridge penalties")
        penalties = tuple(float(alpha) for alpha in registered_alphas)
    except (TypeError, ValueError) as error:
        raise ValueError("blocked: alphas must be a finite positive fixed candidate list") from error
    _require(penalties and len(set(penalties)) == len(penalties)
             and all(np.isfinite(alpha) and alpha > 0 for alpha in penalties),
             "alphas must be unique finite positive fixed candidates")
    for count, name, lower in ((minimum_training_rows, "minimum_training_rows", len(FEATURE_COLUMNS) + 2),
                               (minimum_validation_rows, "minimum_validation_rows", 1), (hac_lags, "hac_lags", 0)):
        _require(isinstance(count, (int, np.integer)) and not isinstance(count, (bool, np.bool_)) and count >= lower,
                 name + " is invalid")
    prepared, dates, times = _inputs(dataset, frame)
    outer_year, inner_year = int(outer_year), int(outer_year) - 1
    outer_cutoff = pd.Timestamp(f"{outer_year}-01-01", tz="America/New_York")
    inner_train, inner_audit = _training(prepared, dates, times, inner_year, minimum_training_rows, hac_lags)
    inner_year_rows = dates.year == inner_year
    _require(inner_year_rows.any(), "inner validation signal year is unavailable")
    validation_mature = (times["feature_available_at"][inner_year_rows] < outer_cutoff) \
        & (times["h5_available_at"][inner_year_rows] < outer_cutoff)
    count = int(validation_mature.sum())
    _require(count >= minimum_validation_rows and validation_mature[:count].all()
             and not validation_mature[count:].any(),
             "validation requires a complete mature prefix with no interior missing rows")
    validation_mask = inner_year_rows.copy()
    validation_mask[inner_year_rows] = validation_mature
    validation = prepared.loc[validation_mask].copy()
    query = frame.loc[dates[validation_mask]].copy()
    Y = validation["label"].to_numpy(float)
    _require(np.isfinite(validation["feature"].to_numpy(float)).all()
             and np.isfinite(Y).all() and (Y > -1).all(),
             "invalid mature validation values; rows cannot be dropped")
    variances = inner_train["label"].to_numpy(float).var(axis=0, ddof=0)
    _require(np.isfinite(variances).all() and (variances > 0).all(),
             "both inner-training label variances must be positive finite")
    inner_models = {"prior": _historical_mean(inner_train, inner_audit, hac_lags=hac_lags)}
    candidate_alphas = {"prior": None}
    for alpha in penalties:
        name = "ridge_" + format(alpha, ".17g")
        inner_models[name] = _ridge(dataset, inner_year, alpha, minimum_training_rows, hac_lags,
                                    inner_audit["training_frame_sha256"])
        candidate_alphas[name] = alpha
    priority = ["prior", *["ridge_" + format(alpha, ".17g") for alpha in sorted(penalties, reverse=True)]]
    scores = []
    for name, model in inner_models.items():
        predicted = predict_annual(model, query)["ridge"]
        _require(predicted.shape == Y.shape and np.isfinite(predicted).all(), "invalid candidate validation forecasts")
        with np.errstate(over="ignore", invalid="ignore"):
            mse = np.square(predicted - Y).mean(axis=0)
            normalized = mse / variances
            score = float(normalized.mean())
        _require(np.isfinite(score) and np.isfinite(normalized).all(), "nonfinite normalized candidate score")
        scores.append({"candidate_name": name, "model_kind": model["model_kind"], "ridge_alpha": candidate_alphas[name],
            "normalized_mse": score, "mse_soxl": float(mse[0]), "mse_soxx": float(mse[1]),
            "normalized_mse_soxl": float(normalized[0]), "normalized_mse_soxx": float(normalized[1]),
            "inner_training_frame_sha256": model["audit"]["training_frame_sha256"],
            "validation_rows": count, "exact_tie_priority": priority.index(name)})
    best_score = min(item["normalized_mse"] for item in scores)
    exact_ties = [item["candidate_name"] for item in scores if item["normalized_mse"] == best_score]
    selected = min(exact_ties, key=priority.index)
    outer_train, outer_audit = _training(prepared, dates, times, outer_year, minimum_training_rows, hac_lags)
    control = _ridge(dataset, outer_year, 10., minimum_training_rows, hac_lags, outer_audit["training_frame_sha256"])
    if selected == "prior":
        fitted = _historical_mean(outer_train, outer_audit, hac_lags=hac_lags)
    elif candidate_alphas[selected] == 10.:
        fitted = deepcopy(control)
    else:
        fitted = _ridge(dataset, outer_year, candidate_alphas[selected], minimum_training_rows, hac_lags,
                        outer_audit["training_frame_sha256"])
    for item in scores:
        item["selected"] = item["candidate_name"] == selected
    audit = {"outer_year": outer_year, "inner_year": inner_year,
        "inner_cutoff": inner_audit["model_available_at"], "outer_cutoff": outer_cutoff.isoformat(),
        "inner_training": inner_audit, "outer_training": outer_audit,
        "validation_rows": count, "validation_signal_year_rows": int(inner_year_rows.sum()),
        "validation_unmatured_suffix_rows": int(inner_year_rows.sum()) - count,
        "validation_first_signal": dates[validation_mask][0].date().isoformat(),
        "validation_last_signal": dates[validation_mask][-1].date().isoformat(),
        "validation_label_available_at_max": times["h5_available_at"][validation_mask].max().isoformat(),
        "validation_frame_sha256": frame_sha256(validation),
        "validation_labels_sha256": frame_sha256(validation["label"]),
        "inner_training_label_variances_ddof0": variances.tolist(), "response_order": ["SOXL", "SOXX"],
        "score": "mean_over_assets(validation_MSE / inner_training_label_variance_ddof0)",
        "candidate_scores": scores, "alphas": list(penalties), "selected_name": selected,
        "selected_alpha": candidate_alphas[selected], "control_alpha": 10.,
        "exact_tie_order": priority, "exact_best_ties": exact_ties,
        "feature_columns": list(FEATURE_COLUMNS), "horizon_sessions": 5,
        "validation_rule": "inner signal year only; H5 label and feature availability strictly before outer January1",
        "outer_training_rule": "refit after selection using only mature features and H5 labels before outer January1",
        "outer_outcomes_used_for_selection": False, "invalid_rows_dropped": 0,
        "scope": "nested historical forecasting comparison; no portfolio-based selection or forward-performance claim"}
    return {"selected_name": selected, "control_fit": control, "selected_fit": fitted,
        "candidate_scores": pd.DataFrame(scores), "audit": audit, "inner_models": inner_models}

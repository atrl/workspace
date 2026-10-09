"""Native Qlib H5 mean estimators for the registered convex allocation study.

The five-session reward labels and their mean-estimation covariance are distinct
from the daily covariance used by the portfolio feasibility constraint. No
portfolio optimization, risk-budget choice, or feature search occurs here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.integrated_allocation import (
    PRIMARY_FEATURES, PRIMARY_METADATA, build_native_dataset, frame_sha256,
)


HORIZON_SESSIONS = 5
FEATURE_COLUMNS = tuple(PRIMARY_FEATURES[:-3])
LABEL_COLUMNS = ("soxl_return_h5", "soxx_return_h5")
METADATA_COLUMNS = (*PRIMARY_METADATA, "h5_exit_date", "h5_available_at")
MEAN_COVARIANCE_UNITS = "five_session_simple_return_squared"
DAILY_RISK_COVARIANCE_UNITS = "one_session_simple_return_squared"


def _open_at(days: pd.Series) -> pd.Series:
    values = pd.DatetimeIndex(pd.to_datetime(days)).tz_localize("America/New_York")
    return pd.Series(values + pd.Timedelta(hours=9, minutes=30), index=days.index)


def build_horizon_frame(primary_frame: pd.DataFrame, execution_prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach H5 labels to frozen Qlib rows without changing daily replay fields.

    Input frames must be retained Qlib observations/Recorder objects. A signal
    at t close enters at open[t+1] and its H5 label matures at open[t+6]. Unknown
    terminal labels remain missing; ready inference features remain present.
    """
    for frame in (primary_frame, execution_prices):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("blocked: H5 inputs require unique ordered Qlib dates")
    if primary_frame.empty or not primary_frame.index.isin(execution_prices.index).all():
        raise ValueError("blocked: H5 signal dates are absent from frozen execution prices")
    if not np.isfinite(primary_frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("blocked: incomplete H5 causal features")
    result = primary_frame.copy(deep=True)
    calendar = execution_prices.index
    dates = pd.Series(calendar, index=calendar)
    entry, end = dates.shift(-1), dates.shift(-(HORIZON_SESSIONS + 1))
    expected_trade_dates = entry.reindex(result.index)
    if not pd.to_datetime(result.trade_date).equals(expected_trade_dates.rename("trade_date")):
        raise ValueError("blocked: H5 entry alignment differs from the frozen daily workflow")
    for symbol, label in zip(("SOXL", "SOXX"), LABEL_COLUMNS):
        opening = execution_prices[("open", symbol)].astype("float64")
        if not np.isfinite(opening.to_numpy()).all() or (opening <= 0).any():
            raise ValueError("blocked: invalid native H5 opening prices")
        result[label] = (opening.shift(-(HORIZON_SESSIONS + 1)) / opening.shift(-1) - 1).reindex(result.index)
    result["h5_exit_date"] = end.reindex(result.index)
    result["h5_available_at"] = _open_at(result.h5_exit_date)
    expected = pd.to_datetime(result.forecast_available_at, utc=True)
    if not expected.equals(pd.to_datetime(result.h5_available_at, utc=True).rename("forecast_available_at")):
        raise ValueError("blocked: H5 label availability differs from the original five-return maturity")
    complete = result.loc[:, list(LABEL_COLUMNS)].notna().all(axis=1)
    if not complete.any() or result.loc[complete, "h5_available_at"].isna().any():
        raise ValueError("blocked: H5 reward labels lack valid observation times")
    audit = {"horizon_sessions": HORIZON_SESSIONS, "formula": "open[t+6]/open[t+1]-1",
        "signal": "close[t]", "entry": "open[t+1]", "outcome_available": "open[t+6],09:30 America/New_York",
        "feature_columns": list(FEATURE_COLUMNS), "label_columns": list(LABEL_COLUMNS),
        "primary_input_sha256": frame_sha256(primary_frame), "execution_prices_sha256": frame_sha256(execution_prices),
        "horizon_frame_sha256": frame_sha256(result), "rows": len(result), "complete_labels": int(complete.sum()),
        "last_feature_date": str(result.index.max().date()), "last_complete_label_signal": str(result.index[complete].max().date()),
        "last_complete_label_available_at": result.loc[complete, "h5_available_at"].max().isoformat(),
        "daily_returns_preserved": True, "mean_covariance_units": MEAN_COVARIANCE_UNITS,
        "daily_risk_covariance_units": DAILY_RISK_COVARIANCE_UNITS,
        "qlib_docs_reference": "https://qlib.readthedocs.io/en/latest/component/data.html"}
    return result, audit


def build_horizon_dataset(frame: pd.DataFrame, *, as_of: str):
    dataset = build_native_dataset(frame, feature_columns=FEATURE_COLUMNS, label_columns=LABEL_COLUMNS,
                                   metadata_columns=METADATA_COLUMNS, as_of=as_of)
    dataset.horizon_sessions = HORIZON_SESSIONS
    dataset.mean_covariance_units = MEAN_COVARIANCE_UNITS
    return dataset


def _psd_roundoff(matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    """Repair floating-point negative eigenvalues only; reject material defects."""
    value = np.asarray(matrix, dtype="float64")
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.isfinite(value).all():
        raise ValueError("blocked: invalid covariance matrix")
    scale = max(float(np.max(np.abs(value))), np.finfo(float).tiny)
    symmetry_error = float(np.max(np.abs(value - value.T)))
    symmetry_tolerance = 256 * np.finfo(float).eps * len(value) * scale
    if symmetry_error > symmetry_tolerance:
        raise ValueError("blocked: materially asymmetric covariance")
    symmetric = (value + value.T) / 2
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    eigen_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    tolerance = 256 * np.finfo(float).eps * len(value) * eigen_scale
    if eigenvalues[0] < -tolerance:
        raise ValueError("blocked: materially negative covariance eigenvalue")
    count = int((eigenvalues < 0).sum())
    repaired = (vectors * np.maximum(eigenvalues, 0)) @ vectors.T if count else symmetric
    repaired = (repaired + repaired.T) / 2
    return repaired, {"minimum_eigenvalue_before": float(eigenvalues[0]), "roundoff_tolerance": tolerance,
        "clipped_roundoff_eigenvalues": count, "maximum_entry_adjustment": float(np.max(np.abs(repaired-value))),
        "symmetry_error": symmetry_error, "policy": "roundoff only; material negative eigenvalues fail closed"}


def _hac_coefficient_covariance(design: np.ndarray, residual: np.ndarray, inverse_penalized_gram: np.ndarray,
                                 *, maxlags: int) -> tuple[np.ndarray, dict]:
    """Joint response-major Bartlett HAC sandwich for the penalized estimator.

    Residual-score centering removes the sample estimating-equation offset
    caused by the fixed ridge penalty. No n/(n-k) finite-sample correction is
    applied. HAC5 captures local residual-score dependence, including overlap;
    it does not account for shrinkage bias or historical model selection.
    """
    n, p = design.shape
    if residual.shape != (n, 2) or not 0 <= maxlags < n:
        raise ValueError("blocked: invalid joint HAC residual dimensions or lag")
    scores = np.concatenate([design * residual[:, j:j+1] for j in range(2)], axis=1)
    score_offset = scores.mean(axis=0)
    scores = scores - score_offset
    meat = scores.T @ scores
    for lag in range(1, maxlags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1. - lag / (maxlags + 1.)) * (cross + cross.T)
    bread = np.kron(np.eye(2), inverse_penalized_gram)
    covariance, psd = _psd_roundoff(bread @ meat @ bread.T)
    return covariance, {"method": "joint_residual_score_Bartlett_HAC", "maxlags": maxlags,
        "response_order": ["SOXL", "SOXX"], "coefficients_per_response": p,
        "score_centering": "subtract prior-training sample score mean, including ridge penalty offset",
        "maximum_absolute_score_offset": float(np.abs(score_offset).max()),
        "finite_sample_correction": False, "coefficient_covariance_psd_audit": psd,
        "limitation": "conditional parameter-estimation uncertainty; excludes ridge bias, nonstationarity and model-search uncertainty"}


def fit_annual_estimator(dataset, *, year: int, ridge_alpha: float = 10., minimum_training_rows: int = 252,
                         hac_lags: int = 5) -> dict[str, Any]:
    """Fit prior means and joint Ridge only from native DatasetH mature rows."""
    from qlib.data.dataset import DatasetH

    if not isinstance(dataset, DatasetH) or getattr(dataset, "horizon_sessions", None) != HORIZON_SESSIONS:
        raise ValueError("blocked: annual H5 estimators require the native horizon DatasetH")
    if not isinstance(year, (int, np.integer)) or not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("blocked: invalid annual year or ridge penalty")
    if minimum_training_rows < len(FEATURE_COLUMNS) + 2 or not isinstance(hac_lags, int) or hac_lags < 0:
        raise ValueError("blocked: insufficient estimation contract or invalid HAC lag")
    cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    if tuple(prepared["feature"].columns) != FEATURE_COLUMNS or tuple(prepared["label"].columns) != LABEL_COLUMNS:
        raise ValueError("blocked: H5 feature or label schema differs")
    available = pd.to_datetime(prepared[("metadata", "h5_available_at")], utc=True)
    feature_known = pd.to_datetime(prepared[("metadata", "feature_available_at")], utc=True)
    signal_dates = prepared.index.get_level_values("datetime")
    mature = (available.notna() & available.lt(cutoff.tz_convert("UTC"))
              & feature_known.notna() & feature_known.lt(cutoff.tz_convert("UTC"))
              & (signal_dates >= pd.Timestamp("2011-01-03")))
    train = prepared.loc[mature].copy()
    if len(train) < minimum_training_rows or hac_lags >= len(train):
        raise ValueError("blocked: insufficient strictly mature H5 training rows")
    X, Y = train["feature"].to_numpy(dtype="float64"), train["label"].to_numpy(dtype="float64")
    if not np.isfinite(X).all() or not np.isfinite(Y).all() or (Y <= -1).any():
        raise ValueError("blocked: incomplete mature H5 features or labels")
    if train.index.get_level_values("datetime").max().year >= year:
        raise ValueError("blocked: annual estimator consumed a future signal year")
    mean, raw_scale = X.mean(axis=0), X.std(axis=0, ddof=0)
    scale = np.where(raw_scale > 1e-12, raw_scale, 1.)
    design = np.c_[np.ones(len(X)), (X-mean)/scale]
    gram = design.T @ design
    penalty = np.diag(np.r_[0., np.full(len(FEATURE_COLUMNS), ridge_alpha)])
    inverse = np.linalg.solve(gram + penalty, np.eye(len(FEATURE_COLUMNS)+1))
    coefficients = inverse @ (design.T @ Y)
    residual = Y-design@coefficients
    coefficient_covariance, hac_audit = _hac_coefficient_covariance(design, residual, inverse, maxlags=hac_lags)
    training_hash = frame_sha256(train)
    audit = {"model_year": int(year), "model_available_at": cutoff.isoformat(), "horizon_sessions": HORIZON_SESSIONS,
        "training_rows": len(train), "minimum_training_rows": minimum_training_rows,
        "first_training_signal": str(train.index.get_level_values("datetime").min().date()),
        "last_training_signal": str(train.index.get_level_values("datetime").max().date()),
        "last_label_available_at": available.loc[train.index].max().isoformat(),
        "training_frame_sha256": training_hash, "ridge_alpha": ridge_alpha,
        "ridge_objective": "sum squared errors + alpha*sum squared slopes; unpenalized intercept",
        "scaler": "prior strictly mature training rows, population standard deviation; constants use scale1",
        "constant_feature_count": int((raw_scale <= 1e-12).sum()),
        "mean_covariance_units": MEAN_COVARIANCE_UNITS, "daily_risk_covariance_units": DAILY_RISK_COVARIANCE_UNITS,
        "maturity_rule": "H5 label and features strictly before January1 NewYork; no overlapping unfinished labels",
        "source_api": "qlib.data.dataset.DatasetH.prepare", "hac": hac_audit}
    return {"model_year": int(year), "model_available_at": cutoff.isoformat(), "horizon_sessions": HORIZON_SESSIONS,
        "feature_columns": FEATURE_COLUMNS, "feature_mean": mean, "feature_scale": scale,
        "prior_mean": Y.mean(axis=0), "ridge_coefficients": coefficients,
        "coefficient_covariance_hac": coefficient_covariance, "audit": audit}


def predict_annual(fit: dict, frame: pd.DataFrame) -> dict[str, Any]:
    """Return H5 prior/Ridge means and HAC mean covariance; no portfolio action."""
    if fit.get("horizon_sessions") != HORIZON_SESSIONS or tuple(fit.get("feature_columns", ())) != FEATURE_COLUMNS:
        raise ValueError("blocked: annual estimator schema differs")
    X = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
    p = len(FEATURE_COLUMNS)+1
    mean, scale = np.asarray(fit["feature_mean"]), np.asarray(fit["feature_scale"])
    coefficients = np.asarray(fit["ridge_coefficients"])
    covariance = np.asarray(fit["coefficient_covariance_hac"])
    prior = np.asarray(fit["prior_mean"])
    if (X.ndim != 2 or len(X) == 0 or mean.shape != (p-1,) or scale.shape != (p-1,)
            or coefficients.shape != (p, 2) or covariance.shape != (2*p, 2*p) or prior.shape != (2,)
            or not all(np.isfinite(value).all() for value in (X, mean, scale, coefficients, covariance, prior))
            or (scale <= 0).any()):
        raise ValueError("blocked: invalid annual prediction inputs or parameters")
    model_at = pd.Timestamp(fit["model_available_at"])
    if model_at.tzinfo is None:
        raise ValueError("blocked: model availability requires timezone")
    feature_times = pd.to_datetime(frame.feature_available_at, utc=True)
    if feature_times.isna().any() or feature_times.lt(model_at.tz_convert("UTC")).any():
        raise ValueError("blocked: annual model cannot produce retrospective OOF predictions")
    covariance, stored_psd = _psd_roundoff(covariance)
    design = np.c_[np.ones(len(X)), (X-mean)/scale]
    ridge = design @ coefficients
    blocks = covariance.reshape(2, p, 2, p)
    raw_covariance = np.einsum("ni,aibj,nj->nab", design, blocks, design)
    repaired, psd_audits = zip(*[_psd_roundoff(value) for value in raw_covariance])
    return {"prior": np.broadcast_to(prior, (len(X), 2)).copy(), "ridge": ridge,
        "Vmean": np.stack(repaired), "audit": {"model_year": fit["model_year"],
            "model_available_at": fit["model_available_at"], "training_frame_sha256": fit["audit"]["training_frame_sha256"],
            "rows": len(frame), "response_order": ["SOXL", "SOXX"], "horizon_sessions": HORIZON_SESSIONS,
            "mean_covariance_units": MEAN_COVARIANCE_UNITS, "daily_risk_covariance_units": DAILY_RISK_COVARIANCE_UNITS,
            "prediction_covariance_clipped_roundoff_eigenvalues": sum(a["clipped_roundoff_eigenvalues"] for a in psd_audits),
            "maximum_prediction_covariance_adjustment": max(a["maximum_entry_adjustment"] for a in psd_audits),
            "stored_covariance_psd_audit": stored_psd,
            "interpretation": "uncertainty of conditional H5 mean; not event-return variance or guaranteed confidence coverage"}}

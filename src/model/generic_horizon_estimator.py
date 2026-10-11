"""Parameterized native horizon estimator math (H5 default; H20 uses the same protocol).

Real math extracted from ``convex_allocation.fit_annual_estimator`` / ``predict_annual``:
standardized joint Ridge (population std, constant -> scale 1), unpenalized intercept,
Bartlett HAC coefficient covariance, and Vmean = design @ Cov @ design^T. The parent
intercept-only historical-mean prior (``annual_ridge_selection._historical_mean``) is also
provided so the selection's ``prior`` candidate uses the parent prior fit and its own HAC
uncertainty, not a ridge prior_mean.

Horizon, label columns, availability column and unit strings are explicit parameters. The
unit strings are NOT free-form: they must equal ``mean_variance_unit(horizon_sessions)``
and ``DAILY_RISK_UNIT``. HAC lag stays at the parent value (5).

The sealed H5 entry is left UNCHANGED; a focused test asserts numerical equivalence.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.generic_horizon import DAILY_RISK_UNIT, mean_variance_unit, predictor_clock, strict_positive_int
from src.data.integrated_allocation import frame_sha256

PARENT_HAC_LAGS = 5
PARENT_TRAIN_START = "2011-01-03"


def native_eligibility(prepared, eligibility_column=None):
    """Explicit source mask; retain all native rows for chronology checks."""
    if eligibility_column is None:
        return pd.Series(True, index=prepared.index)
    key = ('metadata', eligibility_column)
    if key not in prepared or not prepared[key].isin([0, 1]).all():
        raise ValueError('blocked: eligibility requires explicit 0/1 metadata on every row')
    return prepared[key].astype(bool)


def training_identity(train):
    """Comparable between arms even when their feature schemas differ."""
    import hashlib
    return dict(training_index_sha256=hashlib.sha256('\n'.join(map(str, train.index)).encode()).hexdigest(),
                training_label_sha256=frame_sha256(train['label']))


def _strict_year(value, name="year") -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"blocked: {name} must be an integer calendar year")
    if not 1900 <= int(value) <= 9999:
        raise ValueError(f"blocked: {name} out of range")
    return int(value)


def _strict_count(value, name) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"blocked: {name} must be a positive integer")
    if int(value) < 1:
        raise ValueError(f"blocked: {name} must be a positive integer")
    return int(value)


def _units(horizon_sessions: int, mean_covariance_units, daily_risk_covariance_units) -> tuple[str, str]:
    expected_mean, expected_daily = mean_variance_unit(horizon_sessions), DAILY_RISK_UNIT
    if mean_covariance_units not in (None, expected_mean):
        raise ValueError(f"blocked: mean covariance unit must be {expected_mean!r}")
    if daily_risk_covariance_units not in (None, expected_daily):
        raise ValueError(f"blocked: daily risk unit must be {expected_daily!r}")
    return expected_mean, expected_daily


def _resolve_side(name):
    from src.data.convex_allocation import _hac_coefficient_covariance, _psd_roundoff
    return _hac_coefficient_covariance, _psd_roundoff


def _mature_tail_check(available: pd.Series) -> None:
    """NaT labels may only appear as a contiguous terminal-unknown suffix."""
    has_label = available.notna().to_numpy()
    expected = np.array([True] * int(has_label.sum()) + [False] * int((~has_label).sum()))
    if not np.array_equal(has_label, expected):
        raise ValueError("blocked: NaT labels may only appear at the terminal unknown tail")


def fit_horizon_estimator(dataset, *, horizon_sessions: int = 5, feature_columns, label_columns,
                          availability_column: str, year: int, ridge_alpha: float = 10.,
                          minimum_training_rows: int = 252, hac_lags: int = PARENT_HAC_LAGS,
                          train_start: str = PARENT_TRAIN_START,
                          mean_covariance_units: str | None = None,
                          daily_risk_covariance_units: str | None = None,
                          eligibility_column: str | None = None) -> dict[str, Any]:
    """Fit the prior mean and joint Ridge only from native mature rows for this horizon."""
    from qlib.data.dataset import DatasetH

    horizon_sessions = strict_positive_int(horizon_sessions)
    year = _strict_year(year)
    minimum_training_rows = _strict_count(minimum_training_rows, "minimum_training_rows")
    if isinstance(hac_lags, (bool, np.bool_)) or not isinstance(hac_lags, (int, np.integer)) or int(hac_lags) < 0:
        raise ValueError("blocked: invalid HAC lag")
    hac_lags = int(hac_lags)
    feature_columns, label_columns = tuple(feature_columns), tuple(label_columns)
    mean_unit, daily_unit = _units(horizon_sessions, mean_covariance_units, daily_risk_covariance_units)
    if not isinstance(dataset, DatasetH) or getattr(dataset, "horizon_sessions", None) != horizon_sessions:
        raise ValueError("blocked: the horizon-tagged DatasetH is required")
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("blocked: invalid ridge penalty")
    if minimum_training_rows < len(feature_columns) + 2:
        raise ValueError("blocked: insufficient estimation contract")

    cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    if tuple(prepared["feature"].columns) != feature_columns or tuple(prepared["label"].columns) != label_columns:
        raise ValueError("blocked: feature or label schema differs")
    validate_native_horizon_dataset(dataset, horizon_sessions=horizon_sessions, feature_columns=feature_columns,
                                    label_columns=label_columns, availability_column=availability_column,
                                    calendar=getattr(dataset, "calendar", None), eligibility_column=eligibility_column)
    eligible = native_eligibility(prepared, eligibility_column)
    available = pd.to_datetime(prepared[("metadata", availability_column)], utc=True)
    feature_known = pd.to_datetime(prepared[("metadata", "feature_available_at")], utc=True)
    if feature_known.loc[eligible].isna().any():
        raise ValueError("blocked: feature availability is missing on at least one row")
    signal_dates = pd.DatetimeIndex(prepared.index.get_level_values("datetime"))
    prediction = pd.Series(predictor_clock(signal_dates), index=prepared.index)
    if bool((feature_known > prediction).any()):
        raise ValueError("blocked: feature_available_at exceeds the row prediction clock")
    _mature_tail_check(available)
    mature = (available.notna() & available.lt(cutoff.tz_convert("UTC"))
              & feature_known.notna() & feature_known.lt(cutoff.tz_convert("UTC"))
              & (signal_dates >= pd.Timestamp(train_start)) & eligible)
    train = prepared.loc[mature].copy()
    if len(train) < minimum_training_rows or hac_lags >= len(train):
        raise ValueError("blocked: insufficient strictly mature training rows")
    X, Y = train["feature"].to_numpy(dtype="float64"), train["label"].to_numpy(dtype="float64")
    if not np.isfinite(X).all() or not np.isfinite(Y).all() or (Y <= -1).any():
        raise ValueError("blocked: incomplete mature features or labels")
    if train.index.get_level_values("datetime").max().year >= year:
        raise ValueError("blocked: annual estimator consumed a future signal year")
    hac_fn, _ = _resolve_side("hac")
    mean, raw_scale = X.mean(axis=0), X.std(axis=0, ddof=0)
    scale = np.where(raw_scale > 1e-12, raw_scale, 1.)
    design = np.c_[np.ones(len(X)), (X - mean) / scale]
    gram = design.T @ design
    penalty = np.diag(np.r_[0., np.full(len(feature_columns), ridge_alpha)])
    inverse = np.linalg.solve(gram + penalty, np.eye(len(feature_columns) + 1))
    coefficients = inverse @ (design.T @ Y)
    residual = Y - design @ coefficients
    coefficient_covariance, hac_audit = hac_fn(design, residual, inverse, maxlags=hac_lags)
    audit = {"model_year": year, "model_available_at": cutoff.isoformat(), "horizon_sessions": horizon_sessions,
             "training_rows": len(train), "minimum_training_rows": minimum_training_rows,
             "first_training_signal": str(train.index.get_level_values("datetime").min().date()),
             "last_training_signal": str(train.index.get_level_values("datetime").max().date()),
             "last_label_available_at": available.loc[train.index].max().isoformat(),
             "training_frame_sha256": frame_sha256(train), "ridge_alpha": ridge_alpha,
             "ridge_objective": "sum squared errors + alpha*sum squared slopes; unpenalized intercept",
             "scaler": "prior strictly mature training rows, population standard deviation; constants use scale1",
             "constant_feature_count": int((raw_scale <= 1e-12).sum()),
             "mean_covariance_units": mean_unit, "daily_risk_covariance_units": daily_unit,
             "hac_lags": hac_lags, "hac": hac_audit,
             "maturity_rule": f"label ({availability_column}) and features strictly before January1 NewYork",
             "source_api": "qlib.data.dataset.DatasetH.prepare"}
    audit.update(training_identity(train), eligibility_column=eligibility_column,
                 source_ineligible_rows=int((~eligible).sum()))
    return {"model_year": year, "model_available_at": cutoff.isoformat(), "horizon_sessions": horizon_sessions,
            "feature_columns": feature_columns, "label_columns": label_columns,
            "mean_covariance_units": mean_unit, "daily_risk_covariance_units": daily_unit,
            "feature_mean": mean, "feature_scale": scale, "prior_mean": Y.mean(axis=0),
            "ridge_coefficients": coefficients, "coefficient_covariance_hac": coefficient_covariance, "audit": audit}


def fit_horizon_prior(dataset, *, horizon_sessions: int = 5, feature_columns, label_columns,
                      availability_column: str, year: int, minimum_training_rows: int = 252,
                      hac_lags: int = PARENT_HAC_LAGS, train_start: str = PARENT_TRAIN_START,
                      mean_covariance_units: str | None = None,
                      daily_risk_covariance_units: str | None = None,
                      eligibility_column: str | None = None) -> dict[str, Any]:
    """Parent intercept-only historical-mean prior with its own HAC intercept uncertainty."""
    from qlib.data.dataset import DatasetH

    horizon_sessions = strict_positive_int(horizon_sessions)
    year = _strict_year(year)
    minimum_training_rows = _strict_count(minimum_training_rows, "minimum_training_rows")
    feature_columns, label_columns = tuple(feature_columns), tuple(label_columns)
    mean_unit, daily_unit = _units(horizon_sessions, mean_covariance_units, daily_risk_covariance_units)
    if not isinstance(dataset, DatasetH) or getattr(dataset, "horizon_sessions", None) != horizon_sessions:
        raise ValueError("blocked: the horizon-tagged DatasetH is required")
    validate_native_horizon_dataset(dataset, horizon_sessions=horizon_sessions, feature_columns=feature_columns,
                                    label_columns=label_columns, availability_column=availability_column,
                                    calendar=getattr(dataset, "calendar", None), eligibility_column=eligibility_column)
    cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    eligible = native_eligibility(prepared, eligibility_column)
    available = pd.to_datetime(prepared[("metadata", availability_column)], utc=True)
    feature_known = pd.to_datetime(prepared[("metadata", "feature_available_at")], utc=True)
    if feature_known.loc[eligible].isna().any():
        raise ValueError("blocked: feature availability is missing on at least one row")
    signal_dates = pd.DatetimeIndex(prepared.index.get_level_values("datetime"))
    mature = (available.notna() & available.lt(cutoff.tz_convert("UTC")) & feature_known.notna()
              & feature_known.lt(cutoff.tz_convert("UTC")) & (signal_dates >= pd.Timestamp(train_start)) & eligible)
    train = prepared.loc[mature]
    if len(train) < minimum_training_rows:
        raise ValueError("blocked: insufficient strictly mature prior rows")
    labels = train["label"].to_numpy(dtype="float64")
    if not np.isfinite(labels).all() or (labels <= -1).any():
        raise ValueError('blocked: incomplete mature prior labels')
    mean = labels.mean(axis=0)
    hac_fn, _ = _resolve_side("hac")
    hac, hac_audit = hac_fn(np.ones((len(labels), 1)), labels - mean,
                            np.array([[1.0 / len(labels)]]), maxlags=hac_lags)
    width = len(feature_columns) + 1
    coefficients = np.zeros((width, 2))
    coefficients[0] = mean
    embedded = np.zeros((2 * width, 2 * width))
    embedded[np.ix_([0, width], [0, width])] = hac
    return {"model_kind": "historical_mean", "model_year": year, "model_available_at": cutoff.isoformat(),
            "horizon_sessions": horizon_sessions, "feature_columns": feature_columns, "label_columns": label_columns,
            "mean_covariance_units": mean_unit, "daily_risk_covariance_units": daily_unit,
            "training_rows": int(len(train)), "training_frame_sha256": frame_sha256(train),
            "prior_mean": mean, "feature_mean": np.zeros(len(feature_columns)), "feature_scale": np.ones(len(feature_columns)),
            "ridge_coefficients": coefficients, "coefficient_covariance_hac": embedded,
            "audit": {"model_year": year, "model_available_at": cutoff.isoformat(), "horizon_sessions": horizon_sessions,
                      "model_kind": "historical_mean", "hac_lags": hac_lags, "hac": hac_audit,
                      "mean_covariance_units": mean_unit, "daily_risk_covariance_units": daily_unit,
                      "training_frame_sha256": frame_sha256(train), **training_identity(train),
                      "eligibility_column": eligibility_column, "source_ineligible_rows": int((~eligible).sum())}}


def predict_horizon(fit: dict, frame: pd.DataFrame, *, feature_columns, horizon_sessions: int = 5,
                    mean_covariance_units: str | None = None,
                    daily_risk_covariance_units: str | None = None) -> dict[str, Any]:
    """Return prior/Ridge means and HAC mean covariance for the horizon; no portfolio action."""
    feature_columns = tuple(feature_columns)
    horizon_sessions = strict_positive_int(horizon_sessions)
    mean_unit, daily_unit = _units(horizon_sessions, mean_covariance_units, daily_risk_covariance_units)
    if fit.get("horizon_sessions") != horizon_sessions or tuple(fit.get("feature_columns", ())) != feature_columns:
        raise ValueError("blocked: horizon estimator schema differs")
    if fit.get("mean_covariance_units") != mean_unit:
        raise ValueError("blocked: fitted model unit does not match the requested horizon unit")
    raw_signal = frame["signal_date"] if "signal_date" in frame.columns else pd.Series(frame.index)
    signal = pd.to_datetime(pd.Series(np.asarray(raw_signal)))
    if not (signal.dt.year == fit["model_year"]).all():
        raise ValueError("blocked: query signal year must equal the model year")
    feature_times = pd.to_datetime(frame["feature_available_at"], utc=True)
    if feature_times.isna().any():
        raise ValueError("blocked: query feature_available_at must be explicit timezone-aware on every row")
    if bool((feature_times.to_numpy() > predictor_clock(signal).to_numpy()).any()):
        raise ValueError("blocked: query feature_available_at exceeds the row prediction clock")
    X = frame.loc[:, list(feature_columns)].to_numpy(dtype="float64")
    p = len(feature_columns) + 1
    mean, scale = np.asarray(fit["feature_mean"]), np.asarray(fit["feature_scale"])
    coefficients = np.asarray(fit["ridge_coefficients"])
    covariance = np.asarray(fit["coefficient_covariance_hac"])
    prior = np.asarray(fit["prior_mean"])
    if (X.ndim != 2 or len(X) == 0 or mean.shape != (p - 1,) or scale.shape != (p - 1,)
            or coefficients.shape != (p, 2) or covariance.shape != (2 * p, 2 * p) or prior.shape != (2,)
            or not all(np.isfinite(value).all() for value in (X, mean, scale, coefficients, covariance, prior))
            or (scale <= 0).any()):
        raise ValueError("blocked: invalid prediction inputs or parameters")
    model_at = pd.Timestamp(fit["model_available_at"])
    if model_at.tzinfo is None:
        raise ValueError("blocked: model availability requires timezone")
    if bool(feature_times.lt(model_at.tz_convert("UTC")).any()):
        raise ValueError("blocked: annual model cannot produce retrospective OOF predictions")
    _, psd_fn = _resolve_side("psd")
    covariance, stored_psd = psd_fn(covariance)
    design = np.c_[np.ones(len(X)), (X - mean) / scale]
    ridge = design @ coefficients
    blocks = covariance.reshape(2, p, 2, p)
    raw_covariance = np.einsum("ni,aibj,nj->nab", design, blocks, design)
    repaired, psd_audits = zip(*[psd_fn(value) for value in raw_covariance])
    return {"prior": np.broadcast_to(prior, (len(X), 2)).copy(), "ridge": ridge, "Vmean": np.stack(repaired),
            "audit": {"model_year": fit["model_year"], "model_available_at": fit["model_available_at"],
                      "training_frame_sha256": fit["audit"]["training_frame_sha256"], "rows": len(frame),
                      "response_order": ["SOXL", "SOXX"], "horizon_sessions": horizon_sessions,
                      "model_kind": fit.get("model_kind", "ridge"),
                      "mean_covariance_units": mean_unit, "daily_risk_covariance_units": daily_unit,
                      "prediction_covariance_clipped_roundoff_eigenvalues": sum(a["clipped_roundoff_eigenvalues"] for a in psd_audits),
                      "maximum_prediction_covariance_adjustment": max(a["maximum_entry_adjustment"] for a in psd_audits),
                      "stored_covariance_psd_audit": stored_psd,
                      "interpretation": "uncertainty of the conditional horizon mean; not event-return variance"}}


def _shifted(calendar, positions, offset):
    out = np.full(len(positions), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = positions + offset < len(calendar)
    out[valid] = calendar.to_numpy()[positions[valid] + offset]
    return pd.DatetimeIndex(out)


def _aware(values, name: str) -> pd.Series:
    """Parse to UTC while rejecting any naive (non-tz-aware) non-null input."""
    raw = pd.Series(values)
    for value in raw.dropna():
        if pd.Timestamp(value).tzinfo is None:
            raise ValueError(f"blocked: {name} must be explicitly timezone-aware")
    return pd.to_datetime(raw, utc=True)


def validate_native_horizon_dataset(dataset, *, horizon_sessions: int, feature_columns, label_columns,
                                    availability_column: str, calendar=None, eligibility_column=None) -> dict:
    """Uniform pre-fit chronology/identity check on the native horizon DatasetH."""
    from qlib.data.dataset import DatasetH

    horizon_sessions = strict_positive_int(horizon_sessions)
    if not isinstance(dataset, DatasetH) or getattr(dataset, "horizon_sessions", None) != horizon_sessions:
        raise ValueError("blocked: the horizon-tagged DatasetH is required")
    cal = pd.DatetimeIndex(calendar if calendar is not None else getattr(dataset, "calendar", []))
    if len(cal) == 0 or cal.hasnans or cal.duplicated().any() or not cal.is_monotonic_increasing:
        raise ValueError("blocked: a bound native calendar (unique, ordered) is required")
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    if tuple(prepared["feature"].columns) != tuple(feature_columns) or tuple(prepared["label"].columns) != tuple(label_columns):
        raise ValueError("blocked: feature or label schema differs")
    dates = pd.DatetimeIndex(prepared.index.get_level_values("datetime"))
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("blocked: dataset rows must be unique and ordered")
    positions = cal.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("blocked: dataset rows must be on the bound native calendar")
    if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
        raise ValueError("blocked: dataset rows must be contiguous on the bound native calendar")
    meta = prepared["metadata"]
    eligible = native_eligibility(prepared, eligibility_column)
    if availability_column not in meta.columns or "feature_available_at" not in meta.columns:
        raise ValueError("blocked: availability metadata is missing")
    feature_at = _aware(meta["feature_available_at"], "feature_available_at")
    if feature_at.loc[eligible].isna().any():
        raise ValueError("blocked: feature_available_at is missing on at least one row")
    prediction = pd.Series(predictor_clock(dates), index=prepared.index)
    if bool((feature_at > prediction).any()):
        raise ValueError("blocked: feature_available_at exceeds the row prediction clock")
    label_at = _aware(meta[availability_column], availability_column)
    exit_label = _shifted(cal, positions, horizon_sessions + 1)
    expected = pd.Series((exit_label.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)).tz_convert("UTC"),
                         index=prepared.index)
    lv, ev = label_at.to_numpy(), pd.Series(expected).to_numpy()
    if not np.array_equal(pd.isna(lv), pd.isna(ev)):
        raise ValueError(f"blocked: {availability_column} NaT must align exactly with out-of-range exit t+{horizon_sessions + 1}")
    known = ~pd.isna(ev)
    if not np.array_equal(lv[known], ev[known]):
        raise ValueError(f"blocked: {availability_column} must equal exit t+{horizon_sessions + 1} 09:31 America/New_York")
    for column, offset, label in (("trade_date", 1, "next native session (t+1)"),
                                  ("exit_date", 2, "second native session (t+2)")):
        if column in meta.columns:
            pv, xv = pd.to_datetime(meta[column]).to_numpy(), _shifted(cal, positions, offset).to_numpy()
            if not np.array_equal(pd.isna(pv), pd.isna(xv)) or not np.array_equal(pv[~pd.isna(xv)], xv[~pd.isna(xv)]):
                raise ValueError(f"blocked: {column} must be the {label}")
    _mature_tail_check(label_at)
    return {"horizon_sessions": horizon_sessions, "calendar_rows": int(len(cal)), "rows": int(len(dates)),
            "contiguous": True, "terminal_nat_aligned": True, "chronology_ok": True, "calendar_bound": True,
            "eligibility_column": eligibility_column, "source_ineligible_rows": int((~eligible).sum())}

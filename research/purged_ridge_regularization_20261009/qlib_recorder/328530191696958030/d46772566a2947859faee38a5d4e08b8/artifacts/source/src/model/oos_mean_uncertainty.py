"""Past-OOF bias and HAC ambiguity floor for benchmark-relative mean forecasts.

This estimates the mean of observed forecast errors, not an event-return
variance or a guaranteed confidence interval for tomorrow's conditional mean.
The upstream Qlib Recorder must retain genuinely frozen annual forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.convex_allocation import _hac_coefficient_covariance, _psd_roundoff
from src.data.integrated_allocation import frame_sha256


PREDICTED = ("predicted_soxl", "predicted_soxx")
REALIZED = ("realized_soxl", "realized_soxx")
COLUMNS = ("signal_date", "forecast_available_at", "label_available_at",
           *PREDICTED, *REALIZED, "model_year")


def _aware(value, name):
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"blocked: {name} requires a finite timezone-aware timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"blocked: {name} requires a finite timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def fit_oos_mean_error(predictions: pd.DataFrame, *, cutoff,
                       minimum_rows: int = 252, hac_lags: int = 5) -> dict:
    """Fit a two-asset OOF mean-error floor using only strictly mature rows.

    Annual ``model_year`` must equal ``signal_date.year``. Forecasts are issued
    at signal-day 16:15 America/New_York, before next-session execution; labels
    must become available later. Future/unmatured numerical values do not enter
    validation, estimation, or the consumed-row hash. Already-issued forecasts
    still require complete chronology metadata, even when labels are pending.

    ``bias = mean(realized - predicted)``; Bartlett HAC estimates covariance of
    that sample mean. ``floor = outer(bias, bias) + HAC`` preserves persistent
    bias as ambiguity instead of silently dividing it by sample size. This is
    an annual bias ambiguity floor, not full predictive RMSE or guaranteed
    confidence coverage. Historical model selection and drift remain outside
    its uncertainty claim.
    """
    known_before = _aware(cutoff, "cutoff")
    if (isinstance(minimum_rows, bool) or not isinstance(minimum_rows, (int, np.integer))
            or minimum_rows < 2 or isinstance(hac_lags, bool)
            or not isinstance(hac_lags, (int, np.integer)) or hac_lags < 0):
        raise ValueError("blocked: minimum_rows and hac_lags require valid integer counts")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty or set(COLUMNS) - set(predictions):
        raise ValueError("blocked: nonempty OOF predictions with the complete schema are required")
    if predictions.columns.has_duplicates:
        raise ValueError("blocked: duplicate OOF prediction columns")
    source = predictions.loc[:, list(COLUMNS)].copy().reset_index(drop=True)
    try:
        signals = pd.DatetimeIndex(pd.to_datetime(source.signal_date))
    except (TypeError, ValueError) as exc:
        raise ValueError("blocked: invalid OOF signal dates") from exc
    if (signals.hasnans or signals.tz is not None or signals.has_duplicates
            or not signals.is_monotonic_increasing or not signals.equals(signals.normalize())):
        raise ValueError("blocked: OOF signal dates must be ordered unique finite naive sessions")
    source["signal_date"] = signals
    expected_issue = (signals.tz_localize("America/New_York")
                      + pd.Timedelta(hours=16, minutes=15)).tz_convert("UTC")
    # The fixed issuance clock identifies future rows without inspecting their
    # predictions, realized outcomes, model metadata or availability values.
    issued_mask = expected_issue < known_before
    issued = source.loc[issued_mask].copy()
    if issued.empty:
        raise ValueError("blocked: insufficient strictly mature OOF rows")
    for column in ("forecast_available_at", "label_available_at"):
        issued[column] = pd.DatetimeIndex([_aware(value, column) for value in issued[column]])
    if not pd.DatetimeIndex(issued.forecast_available_at).equals(expected_issue[issued_mask]):
        raise ValueError("blocked: OOF forecast must be issued at signal-day 16:15 America/New_York")
    if not issued.forecast_available_at.lt(issued.label_available_at).all():
        raise ValueError("blocked: OOF forecast must precede label availability")
    if (not issued.label_available_at.is_monotonic_increasing
            or issued.label_available_at.duplicated().any()):
        raise ValueError("blocked: OOF label availability must be ordered and unique")
    model_years = pd.to_numeric(issued.model_year, errors="coerce").to_numpy(float)
    if not np.isfinite(model_years).all() or not np.array_equal(model_years, issued.signal_date.dt.year.to_numpy()):
        raise ValueError("blocked: OOF model_year must equal its signal year; retrospective refits are forbidden")
    issued["model_year"] = model_years.astype(np.int64)
    mature = issued.loc[issued.label_available_at.lt(known_before)].copy().reset_index(drop=True)
    count = len(mature)
    if count < minimum_rows or hac_lags >= count:
        raise ValueError("blocked: insufficient strictly mature OOF rows for declared minimum or HAC lag")
    try:
        values = mature.loc[:, [*PREDICTED, *REALIZED]].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("blocked: mature OOF returns must be finite numbers") from exc
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("blocked: mature OOF returns must be finite and greater than -1")
    # Canonical dtypes and a fresh positional index make the hash a function
    # solely of the consumed records, independent of unrelated future rows.
    mature.loc[:, [*PREDICTED, *REALIZED]] = values
    mature = mature.astype(dict.fromkeys((*PREDICTED, *REALIZED), np.float64))
    errors = values[:, 2:] - values[:, :2]
    with np.errstate(over="ignore", invalid="ignore"):
        bias = errors.mean(axis=0)
        hac, hac_audit = _hac_coefficient_covariance(
            np.ones((count, 1)), errors, np.array([[1. / count]]), maxlags=int(hac_lags))
        floor, floor_audit = _psd_roundoff(np.outer(bias, bias) + hac)
    if not np.isfinite(bias).all():
        raise ValueError("blocked: nonfinite OOF mean error")
    audit = {
        "cutoff": known_before.isoformat(), "row_count": count,
        "minimum_rows": int(minimum_rows), "hac_lags": int(hac_lags),
        "oos_rows_sha256": frame_sha256(mature),
        "first_signal_date": mature.signal_date.iloc[0].date().isoformat(),
        "last_signal_date": mature.signal_date.iloc[-1].date().isoformat(),
        "latest_forecast_available_at": mature.forecast_available_at.max().isoformat(),
        "latest_label_available_at": mature.label_available_at.max().isoformat(),
        "response_order": ["SOXL", "SOXX"], "residual": "realized_minus_predicted",
        "floor_formula": "outer(mean_oof_error, mean_oof_error) + HAC_covariance_of_mean_oof_error",
        "units": "forecast_horizon_simple_return_squared",
        "maturity_rule": "forecast and label availability strictly before cutoff",
        "issuance_clock": "signal-day 16:15 America/New_York; model_year equals signal year",
        "hac": hac_audit, "floor_psd_audit": floor_audit,
        "interpretation": "OOS residual mean plus annual bias ambiguity floor; not predictive RMSE or guaranteed confidence coverage",
        "provenance_scope": "clock and annual identity checked; upstream Qlib Recorder must attest frozen models and prior-only training hashes",
        "limitation": "HAC is conditional on its dependence assumptions; no guarantee against drift, ridge bias changes or adaptive model search",
    }
    return {"bias": bias, "hac_covariance_of_mean": hac, "floor": floor, "audit": audit}

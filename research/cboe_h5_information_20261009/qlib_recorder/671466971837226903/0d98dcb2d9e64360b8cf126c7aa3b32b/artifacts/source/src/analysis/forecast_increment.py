"""Matched, mature H5 forecast diagnostics from Qlib Recorder predictions.

This analyzer cannot verify source vintages, feature availability, model fitting
boundaries or Recorder identity. The caller owns those provenance checks.
Calibration and error decompositions are descriptions, never transformations
of the forecasts. No output grants production eligibility.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


ASSETS = ("soxl", "soxx")
MODELS = ("base", "full", "prior")
SCOPE = "reused_history_diagnostic_not_forward_or_promotion_evidence"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _aware(value):
    timestamp = pd.Timestamp(value)
    _require(not pd.isna(timestamp) and timestamp.tzinfo is not None,
             "evaluation_cutoff must be an explicit aware timestamp")
    return timestamp.tz_convert("UTC")


def _correlation(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _statistics(prediction, realized):
    error = prediction - realized
    mean_error = float(error.mean())
    if np.ptp(prediction) == 0:
        slope, intercept = None, None
    else:
        centered = prediction - prediction.mean()
        slope = float(centered @ (realized - realized.mean()) / (centered @ centered))
        intercept = float(realized.mean() - slope * prediction.mean())
    return {"mse": float(np.square(error).mean()), "mean_prediction_error": mean_error,
        "squared_error_bias": mean_error ** 2, "error_variance_ddof0": float(error.var()),
        "pearson_ic": _correlation(prediction, realized),
        "spearman_ic": _correlation(pd.Series(prediction).rank().to_numpy(),
                                     pd.Series(realized).rank().to_numpy()),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "calibration_status": "constant_prediction_undefined" if slope is None else "descriptive_ols_y_on_prediction"}


def _paired_stationary_means(values, *, mean_block, resamples, seed):
    """Shared circular geometric-block indices for all paired comparisons."""
    random = np.random.default_rng(seed)
    count = len(values)
    index = random.integers(count, size=resamples)
    totals = values[index].copy()
    for _ in range(1, count):
        restart = random.random(resamples) < 1 / mean_block
        index = (index + 1) % count
        index[restart] = random.integers(count, size=int(restart.sum()))
        totals += values[index]
    return totals / count


def analyze_forecast_increment(
    predictions: pd.DataFrame,
    *,
    evaluation_cutoff,
    development_years=tuple(range(2016, 2023)),
    mean_blocks=(20, 60),
    resamples=2000,
) -> dict:
    """Return metrics, increments, bootstrap and maturity audit.

    Input columns: signal_date, label_available_at, model_year, and
    realized/base/full/prior_{soxl,soxx}. Asset labels are H5 simple returns.
    No row is dropped for missing fields. All forecasts must be finite, and
    both realized labels must be finite on every strictly mature row. Only a
    terminal not-yet-mature suffix may have unavailable or missing labels.

    Positive MSE improvement means MSE(reference)-MSE(full). Pooled and annual
    calibration is ex-post, with no refit, sign flip or prediction adjustment.
    The exact MSE identity is squared mean error plus error variance (ddof=0).

    Bootstrap covers the development and reused-diagnostic windows separately,
    not the overlapping pooled-all window or individual years. Within each
    window, the declared family is two assets times two references = four.
    Both fixed block lengths are sensitivity diagnostics, not candidates to
    select. Bounds condition on the retained path, ignore historical research
    selection/refit uncertainty, and do not establish causal information gains.
    """
    cutoff = _aware(evaluation_cutoff)
    _require(isinstance(predictions, pd.DataFrame) and not predictions.empty
             and not predictions.columns.has_duplicates, "predictions need unique columns and nonempty rows")
    required = {"signal_date", "label_available_at", "model_year",
                *[f"{model}_{asset}" for model in ("realized", *MODELS) for asset in ASSETS]}
    _require(required.issubset(predictions.columns), "prediction fields are incomplete")
    years, blocks = tuple(development_years), tuple(mean_blocks)
    _require(bool(years) and all(isinstance(y, Integral) and not isinstance(y, bool) for y in years)
             and years == tuple(range(years[0], years[-1] + 1)), "development years must be ordered and contiguous")
    _require(bool(blocks) and len(set(blocks)) == len(blocks)
             and all(isinstance(b, Integral) and not isinstance(b, bool) and b >= 1 for b in blocks),
             "mean blocks must be explicit unique positive integers")
    _require(isinstance(resamples, Integral) and not isinstance(resamples, bool) and resamples >= 200,
             "bootstrap requires at least 200 resamples")
    frame = predictions.copy().reset_index(drop=True)
    dates = pd.to_datetime(frame.signal_date, errors="raise")
    _require(dates.notna().all() and dates.dt.tz is None and dates.eq(dates.dt.normalize()).all()
             and dates.is_monotonic_increasing and not dates.duplicated().any(),
             "signals must be unique ordered timezone-naive session dates")
    _require(dates.max() <= cutoff.tz_convert("America/New_York").normalize().tz_localize(None),
             "signal rows extend beyond evaluation cutoff")
    frame["signal_date"] = dates
    for value in frame.label_available_at:
        _require(pd.isna(value) or pd.Timestamp(value).tzinfo is not None,
                 "known label availability must have a timezone")
    available = pd.to_datetime(frame.label_available_at, utc=True)
    known = available.notna()
    _require(available.loc[known].is_monotonic_increasing,
             "label availability must not go backwards")
    _require((available.loc[known].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
              > dates.loc[known]).all(), "H5 labels cannot mature on or before the signal date")
    mature = known & available.lt(cutoff)
    count = int(mature.sum())
    _require(count >= 3 and mature.iloc[:count].all() and not mature.iloc[count:].any(),
             "mature observations must be a complete prefix; no missing interior rows")
    numeric_forecasts = [f"{model}_{asset}" for model in MODELS for asset in ASSETS]
    numeric_labels = [f"realized_{asset}" for asset in ASSETS]
    for columns, rows in ((numeric_forecasts, frame), (numeric_labels, frame.iloc[:count])):
        _require(all(pd.api.types.is_numeric_dtype(rows[c]) and not pd.api.types.is_bool_dtype(rows[c])
                     for c in columns), "forecast and label columns must be numeric")
        _require(np.isfinite(rows[columns].to_numpy(float)).all(),
                 "nonfinite forecast or mature label; interior row dropping is prohibited")
    _require((frame.iloc[:count][numeric_labels].to_numpy(float) > -1).all(),
             "realized simple returns must exceed -1")
    _require(pd.api.types.is_numeric_dtype(frame.model_year)
             and not pd.api.types.is_bool_dtype(frame.model_year), "model_year must be numeric")
    model_year = frame.model_year.to_numpy(float)
    _require(np.isfinite(model_year).all() and (model_year == np.floor(model_year)).all()
             and (model_year >= 1900).all() and (model_year <= dates.dt.year.to_numpy()).all(),
             "model_year cannot be nonintegral or later than signal year")
    scored = frame.iloc[:count].copy()
    _require(scored.signal_date.dt.year.ge(years[0]).all(), "predictions precede declared development start")
    _require(set(years).issubset(set(scored.signal_date.dt.year)), "declared development years are missing")
    # Calendar completeness and model training clocks remain caller audits.
    masks = {"all": np.ones(count, dtype=bool),
             "development": scored.signal_date.dt.year.isin(years).to_numpy(),
             "reused_diagnostic": scored.signal_date.dt.year.gt(years[-1]).to_numpy()}
    masks.update({f"year_{year}": scored.signal_date.dt.year.eq(year).to_numpy()
                  for year in sorted(scored.signal_date.dt.year.unique())})
    metrics, increments, bootstrap = [], [], []
    comparisons = [(asset, reference) for asset in ASSETS for reference in ("base", "prior")]
    family_size, alpha, seed = len(comparisons), .05 / len(comparisons), 1729
    for window, mask in masks.items():
        sample = scored.loc[mask]
        if sample.empty:
            continue
        _require(len(sample) >= 3, "each reported window needs at least three mature rows")
        common = {"window": window, "rows": len(sample),
            "start_signal": str(sample.signal_date.iloc[0].date()),
            "end_signal": str(sample.signal_date.iloc[-1].date()), "evidence_scope": SCOPE}
        losses = []
        for asset in ASSETS:
            realized = sample[f"realized_{asset}"].to_numpy(float)
            stats = {model: _statistics(sample[f"{model}_{asset}"].to_numpy(float), realized) for model in MODELS}
            for model in MODELS:
                metrics.append({**common, "asset": asset, "model": model, **stats[model]})
            full_loss = np.square(sample[f"full_{asset}"].to_numpy(float) - realized)
            for reference in ("base", "prior"):
                reference_loss = np.square(sample[f"{reference}_{asset}"].to_numpy(float) - realized)
                difference = reference_loss - full_loss
                losses.append(difference)
                improvement = float(difference.mean())
                increments.append({**common, "asset": asset, "reference": reference,
                    "mse_improvement_reference_minus_full": improvement,
                    "relative_mse_improvement_pct": 100 * improvement / stats[reference]["mse"]
                        if stats[reference]["mse"] > 0 else None,
                    "squared_bias_improvement": stats[reference]["squared_error_bias"] - stats["full"]["squared_error_bias"],
                    "error_variance_improvement": stats[reference]["error_variance_ddof0"] - stats["full"]["error_variance_ddof0"],
                    "pearson_ic_change": None if stats["full"]["pearson_ic"] is None or stats[reference]["pearson_ic"] is None
                        else stats["full"]["pearson_ic"] - stats[reference]["pearson_ic"],
                    "spearman_ic_change": None if stats["full"]["spearman_ic"] is None or stats[reference]["spearman_ic"] is None
                        else stats["full"]["spearman_ic"] - stats[reference]["spearman_ic"]})
        if window not in ("development", "reused_diagnostic"):
            continue
        matrix = np.column_stack(losses)
        for block in blocks:
            _require(len(sample) >= 2 * block, "insufficient mature observations for declared bootstrap block")
            means = _paired_stationary_means(matrix, mean_block=block, resamples=resamples, seed=seed)
            for j, (asset, reference) in enumerate(comparisons):
                lower = float(np.quantile(means[:, j], alpha))
                bootstrap.append({**common, "asset": asset, "reference": reference,
                    "mean_block": block, "resamples": resamples, "random_seed": seed,
                    "declared_family_size": family_size, "one_sided_family_confidence": .95,
                    "per_comparison_alpha": alpha,
                    "mse_improvement_reference_minus_full": float(matrix[:, j].mean()),
                    "mse_improvement_lower_bound": lower, "lower_bound_positive": lower > 0,
                    "nonpositive_draw_fraction_not_pvalue": float((means[:, j] <= 0).mean()),
                    "method": "paired_circular_stationary_bootstrap_mean_squared_error_difference",
                    "limitation": "Conditional on reused fixed forecasts; no refit or historical search adjustment; no production permission."})
    return {"metrics": pd.DataFrame(metrics), "increments": pd.DataFrame(increments),
        "bootstrap": pd.DataFrame(bootstrap), "audit": {
            "input_rows": len(frame), "mature_rows": count, "terminal_unmatured_rows": len(frame) - count,
            "first_unmatured_signal": None if count == len(frame) else str(frame.signal_date.iloc[count].date()),
            "maximum_scored_label_available_at": str(available.iloc[:count].max()),
            "evaluation_cutoff": str(cutoff), "strict_maturity": "label_available_at < evaluation_cutoff",
            "development_years": list(years), "bootstrap_family": "two_assets_times_two_references_within_each_window",
            "declared_family_size": family_size, "mean_blocks": list(blocks),
            "calibration_applied_to_predictions": False, "prediction_sign_flips": False,
            "training_and_calendar_provenance_verified": False, "promotion_eligible": False,
            "evidence_scope": SCOPE}}

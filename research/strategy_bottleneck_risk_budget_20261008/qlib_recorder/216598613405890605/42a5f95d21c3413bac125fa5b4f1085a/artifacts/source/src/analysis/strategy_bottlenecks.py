"""Descriptive attribution of frozen forecasts loaded from a Qlib Recorder.

The caller owns Recorder identity/hash verification. No rows are removed, no
forecast is fitted or changed, and no trading policy is evaluated here. The
full-minus-base increment includes jointly refitted base coefficients; it is
not a causal attribution to an individual feature. Annual loss contributions
do not authorize excluding years. Fee events overlap and are not a portfolio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.frozen_cash_information import cash_hurdle, correlation


_HEAD_LEGS = {"cash5": 1, "leverage5": 2}
_COSTS = (10., 25.)
_FAMILIES = ("base", "base_plus_earnings")


def _checked_predictions(predictions, target_heads):
    if isinstance(target_heads, str):
        raise ValueError("blocked: target_heads must be a sequence of distinct heads")
    heads = tuple(target_heads)
    if not heads or len(set(heads)) != len(heads) or any(h not in _HEAD_LEGS for h in heads):
        raise ValueError("blocked: distinct cash5/leverage5 target heads required")
    if (not isinstance(predictions, pd.DataFrame) or predictions.empty
            or not isinstance(predictions.index, pd.DatetimeIndex)
            or predictions.index.tz is not None or predictions.index.hasnans
            or predictions.index.has_duplicates or not predictions.index.is_monotonic_increasing
            or predictions.columns.has_duplicates):
        raise ValueError("blocked: nonempty ordered unique naive Qlib signal rows required")
    columns = [name for head in heads for name in (head, *(f"{head}_{f}" for f in _FAMILIES))]
    missing = set(columns) - set(predictions.columns)
    if missing:
        raise ValueError(f"blocked: frozen prediction columns missing {sorted(missing)}")
    try:
        values = predictions.loc[:, columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("blocked: numeric frozen prediction rows required") from exc
    if not np.isfinite(values).all():
        raise ValueError("blocked: all frozen prediction rows must be finite; no row dropping")
    return heads


def _mse_attribution(target, base, full):
    try:
        with np.errstate(over="raise", invalid="raise"):
            increment = full - base
            residual = target - base
            base_mse = float(np.mean(np.square(residual)))
            full_mse = float(np.mean(np.square(full - target)))
            energy = float(np.mean(np.square(increment)))
            alignment = float(2 * np.mean(increment * residual))
            difference = full_mse - base_mse
            result = {
                "base_mse": base_mse, "full_mse": full_mse,
                "extra_forecast_energy": energy,
                "twice_residual_alignment": alignment,
                "full_minus_base_MSE": difference,
                "SSE_difference": difference * len(target),
                "identity_absolute_error": abs(difference - (energy - alignment)),
                "extra_forecast_rms_bp": float(10000 * np.sqrt(energy)),
                "increment_variance": float(np.var(increment)),
                "increment_mean_square": float(np.mean(increment) ** 2),
                "twice_increment_residual_covariance": float(2 * np.mean(
                    (increment - increment.mean()) * (residual - residual.mean()))),
                "twice_increment_residual_mean_product": float(2 * increment.mean() * residual.mean()),
                "base_bias_bp": float(10000 * np.mean(base - target)),
                "full_bias_bp": float(10000 * np.mean(full - target)),
            }
    except FloatingPointError as exc:
        raise ValueError("blocked: nonfinite forecast attribution arithmetic") from exc
    if not np.isfinite(list(result.values())).all():
        raise ValueError("blocked: nonfinite forecast attribution arithmetic")
    # Include both original losses in the tolerance scale: their subtraction
    # can cancel even when the increment's two terms are exactly computed.
    scale = max(base_mse, full_mse, energy, abs(alignment), np.finfo(float).tiny)
    result["identity_within_roundoff"] = result["identity_absolute_error"] <= 128 * np.finfo(float).eps * scale
    if not result["identity_within_roundoff"]:
        raise ValueError("blocked: forecast MSE identity failed")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        for name, p, y, rank in (
            ("increment_residual_correlation", increment, residual, False),
            ("base_ic", base, target, True), ("full_ic", full, target, True),
        ):
            value, reason = correlation(p, y, rank=rank)
            result[name] = np.nan if value is None else value
            result[name + "_undefined_reason"] = reason
    result["delta_ic_full_minus_base"] = result["full_ic"] - result["base_ic"]
    return result


def _event_calibration(target, forecast, *, head, cost):
    hurdle = _HEAD_LEGS[head] * cash_hurdle(cost)
    mask = forecast > hurdle
    count = int(mask.sum())
    result = {
        "cost_bps": cost, "events": count, "coverage": count / len(target),
        "hurdle_bp": 10000 * hurdle,
        "forecast_mean_net_bp": np.nan, "realized_mean_net_bp": np.nan,
        "overestimation_bp": np.nan, "positive_event_fraction": np.nan,
        "conditional_mean_undefined_reason": "no_events" if not count else None,
        "event_scope": "isolated_overlapping_H5_SOXX_roundtrip_not_portfolio_not_compounded_not_annualized",
    }
    if count:
        result.update(
            forecast_mean_net_bp=float(10000 * np.mean(forecast[mask] - hurdle)),
            realized_mean_net_bp=float(10000 * np.mean(target[mask] - hurdle)),
            overestimation_bp=float(10000 * np.mean(forecast[mask] - target[mask])),
            positive_event_fraction=float(np.mean(target[mask] > hurdle)),
        )
        if not np.isfinite([result[k] for k in (
            "forecast_mean_net_bp", "realized_mean_net_bp", "overestimation_bp", "positive_event_fraction"
        )]).all():
            raise ValueError("blocked: nonfinite event attribution arithmetic")
    return result


def analyze_forecast_increment(predictions, target_heads=("cash5", "leverage5")):
    """Attribute existing forecast losses and fixed-cost events, without fitting.

    ``predictions`` is the caller-verified Qlib Recorder DataFrame, indexed by
    unique chronological signal dates. Each head requires its target column,
    ``<head>_base`` and ``<head>_base_plus_earnings``. All supplied rows enter the
    pooled slice and exactly one annual slice. Fractions of pooled SSE are
    signed contributions and can be negative or exceed one; a zero pooled
    difference has an undefined fraction. Constant correlations and no-event
    conditional means remain undefined rather than being reported as zero.
    """
    heads = _checked_predictions(predictions, target_heads)
    windows = [("pooled", predictions), *(
        (str(year), frame) for year, frame in predictions.groupby(predictions.index.year, sort=True)
    )]
    losses, events = [], []
    for head in heads:
        pooled_sse = None
        for label, frame in windows:
            target, base, full = (
                frame[c].to_numpy(dtype=np.float64)
                for c in (head, f"{head}_base", f"{head}_base_plus_earnings")
            )
            identity = {"head": head, "slice": label, "rows": len(frame),
                        "first_signal": frame.index[0].isoformat(), "last_signal": frame.index[-1].isoformat()}
            row = {**identity, **_mse_attribution(target, base, full)}
            if label == "pooled":
                pooled_sse = row["SSE_difference"]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                fraction = np.divide(row["SSE_difference"], pooled_sse) if pooled_sse != 0 else np.nan
            row["SSE_difference_fraction_of_pooled"] = float(fraction) if np.isfinite(fraction) else np.nan
            row["SSE_fraction_undefined_reason"] = (
                "zero_pooled_SSE_difference" if pooled_sse == 0 else
                "undefined_numeric_fraction" if not np.isfinite(fraction) else None)
            losses.append(row)
            for cost in _COSTS:
                for family, forecast in zip(_FAMILIES, (base, full)):
                    events.append({**identity, "family": family,
                                   **_event_calibration(target, forecast, head=head, cost=cost)})
    return {"mse_attribution": pd.DataFrame(losses), "event_calibration": pd.DataFrame(events)}


__all__ = ["analyze_forecast_increment"]

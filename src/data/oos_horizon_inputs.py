"""Qualified Qlib features, annual H5 forecasts, and separate causal clocks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.conditional_variance import load_qualified_primary
from src.data.convex_allocation import build_horizon_dataset, fit_annual_estimator, predict_annual
from src.data.integrated_allocation import frame_sha256
from src.workflow.benchmark_risk_excess import expanding_covariance, COVARIANCE_FIELDS


def prepare_horizon_inputs(root, plan):
    """Preserve the parent feature recipe; recalculate labels from native opens.

    No old source card is modified or re-signed. Original and transformed input
    identities remain distinct. Unknown terminal outcomes remain in DatasetH.
    """
    from qlib.data import D

    source = plan["source"]
    bundle = load_qualified_primary(root, provider_uri=source["provider_uri"],
        parent_summary=source["summary"], expected_recorder_id=source["recorder_id"],
        as_of=source["as_of"], train_start=plan["train_start"])
    frame = bundle["primary_frame"].copy()
    original_hash = frame_sha256(frame)
    prices = bundle["execution_prices"].astype(float)
    calendar = pd.DatetimeIndex(D.calendar(freq="day", future=False))
    if not prices.index.equals(calendar) or str(calendar[-1].date()) != source["as_of"]:
        raise ValueError("blocked: qualified price calendar differs")
    dates = pd.Series(calendar, index=calendar)
    for symbol in ("SOXL", "SOXX"):
        opening = prices[("open", symbol)]
        frame[f"{symbol.lower()}_return"] = (opening.shift(-2) / opening.shift(-1) - 1).reindex(frame.index)
        frame[f"{symbol.lower()}_return_h5"] = (opening.shift(-6) / opening.shift(-1) - 1).reindex(frame.index)
    frame["h5_exit_date"] = dates.shift(-6).reindex(frame.index)
    frame["decision_at"] = frame.index.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
    for key, days in (("return_available_at", frame.exit_date), ("h5_available_at", frame.h5_exit_date)):
        frame[key] = pd.DatetimeIndex(days).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    frame["forecast_available_at"] = frame.h5_available_at
    dataset = build_horizon_dataset(frame, as_of=source["as_of"])
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"]).copy()
    prepared.columns = prepared.columns.get_level_values(1)
    prepared.index = pd.DatetimeIndex(prepared.index.get_level_values("datetime"))
    pd.testing.assert_frame_equal(prepared, frame.loc[:, prepared.columns], check_exact=True, check_freq=False)
    covariance = expanding_covariance(prices).reindex(frame.index)
    packed = covariance.loc[:, list(COVARIANCE_FIELDS)].to_numpy(float)
    if not np.isfinite(packed).all():
        raise ValueError("blocked: incomplete past-only covariance")
    sigma = np.empty((len(frame), 2, 2))
    sigma[:, 0, 0], sigma[:, 0, 1] = packed[:, 0], packed[:, 1]
    sigma[:, 1, 0], sigma[:, 1, 1] = packed[:, 1], packed[:, 2]
    return {"dataset": dataset, "frame": frame, "calendar": calendar, "covariance": sigma,
        "audit": {"qualified_source": bundle["audit"], "original_frame_sha256": original_hash,
            "transformed_frame_sha256": frame_sha256(frame), "dataset_sha256": frame_sha256(prepared),
            "label_precision": plan["label_precision"], "decision_clock": "signal date 16:15 America/New_York",
            "outcome_clock": "exit date 09:31 America/New_York", "covariance_source": covariance.attrs}}


def fit_annual_oos(bundle, plan):
    """Fit each annual model once; never re-predict past calibration years."""
    frame, dataset = bundle["frame"], bundle["dataset"]
    years = plan["fit_years"]
    if (not years or any(isinstance(y, (bool, np.bool_)) or not isinstance(y, (int, np.integer)) for y in years)
            or list(years) != sorted(set(years)) or any(not (frame.index.year == y).any() for y in years)):
        raise ValueError("blocked: fit_years require available unique increasing integer years")
    models, tables, covariance, audits = {}, [], {}, []
    for year in plan["fit_years"]:
        fit = fit_annual_estimator(dataset, year=year, ridge_alpha=plan["ridge_alpha"],
            minimum_training_rows=plan["minimum_training_rows"], hac_lags=plan["hac_lags"])
        part = frame.loc[frame.index.year == year]
        if part.empty:
            raise ValueError("blocked: registered forecast year is unavailable")
        forecast = predict_annual(fit, part)
        prediction = pd.DataFrame({"signal_date": part.signal_date,
            "forecast_available_at": part.decision_at, "label_available_at": part.h5_available_at,
            "model_year": year, "training_frame_sha256": fit["audit"]["training_frame_sha256"],
            "predicted_soxl": forecast["ridge"][:, 0], "predicted_soxx": forecast["ridge"][:, 1],
            "realized_soxl": part.soxl_return_h5, "realized_soxx": part.soxx_return_h5}, index=part.index)
        models[year], covariance[year] = fit, forecast["Vmean"]
        audits.append(fit["audit"])
        tables.append(prediction)
    predictions = pd.concat(tables).sort_index()
    vmean = np.concatenate([covariance[year] for year in plan["fit_years"]])
    if not np.isfinite(predictions[["predicted_soxl", "predicted_soxx"]].to_numpy(float)).all():
        raise ValueError("blocked: nonfinite OOS forecasts")
    return predictions, vmean, models, pd.DataFrame(audits)

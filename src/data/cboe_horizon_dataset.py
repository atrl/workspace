"""Matched native DatasetH for a single, lagged Cboe information family."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cboe_volatility_source import FEATURES, load_cboe_features
from src.data.convex_allocation import FEATURE_COLUMNS as BASE_FEATURES
from src.data.integrated_allocation import frame_sha256
from src.data.oos_horizon_inputs import prepare_horizon_inputs


def prepare_cboe_horizon_dataset(root, plan):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    primary = prepare_horizon_inputs(root, plan)
    external = plan["external_source"]
    if external["market_as_of"] != plan["source"]["as_of"]:
        raise ValueError("blocked: external/price research cutoff differs")
    features, source_audit = load_cboe_features(root / external["provider_uri"],
        market_calendar=primary["calendar"], source_start=external["source_start"],
        market_as_of=external["market_as_of"])
    if tuple(external["features"]) != FEATURES:
        raise ValueError("blocked: registered Cboe feature family differs")
    all_frame = primary["frame"]
    calendar = pd.DatetimeIndex(primary["calendar"])
    if (calendar.empty or str(calendar[-1].date()) != plan["source"]["as_of"]
            or source_audit["market_as_of"] != external["market_as_of"]
            or source_audit["store_as_of"] != external["store_as_of"]
            or source_audit["source_start"] != external["source_start"]
            or source_audit["lag_sessions"] != 1):
        raise ValueError("blocked: price calendar or external source cutoff differs")
    expected_index = calendar[calendar >= pd.Timestamp(plan["train_start"])]
    if not all_frame.index.equals(expected_index):
        raise ValueError("blocked: primary frame omits canonical sessions")
    dates = pd.Series(calendar, index=calendar)
    for column, offset in (("signal_date", 0), ("trade_date", 1), ("exit_date", 2), ("h5_exit_date", 6)):
        expected = dates.shift(-offset).reindex(all_frame.index)
        if not pd.DatetimeIndex(all_frame[column]).equals(pd.DatetimeIndex(expected)):
            raise ValueError("blocked: primary session boundaries contain an interior gap")
    expected_decision = all_frame.index.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
    if not pd.DatetimeIndex(all_frame.decision_at).equals(expected_decision):
        raise ValueError("blocked: primary decision clock differs from prior close")
    for column, days in (("return_available_at", all_frame.exit_date), ("h5_available_at", all_frame.h5_exit_date)):
        expected = pd.DatetimeIndex(days).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
        if not pd.DatetimeIndex(all_frame[column]).equals(expected):
            raise ValueError("blocked: primary outcome maturity differs")
    eligible = (all_frame.index >= pd.Timestamp(plan["matched_training_start"])) & all_frame.exit_date.notna()
    frame = all_frame.loc[eligible].copy()
    if frame.empty or not frame.index.isin(features.index).all():
        raise ValueError("blocked: missing common price/Cboe sessions")
    if frame.index[0] != pd.Timestamp(plan["matched_training_start"]):
        raise ValueError("blocked: matched training start is not an observed eligible session")
    context = features.reindex(frame.index)
    if not context.decision_at.equals(frame.decision_at):
        raise ValueError("blocked: external/price decision clocks differ")
    source_dates = pd.DatetimeIndex(dates.shift(1).reindex(frame.index))
    available = source_dates.tz_localize("America/New_York") + pd.Timedelta(hours=17)
    if (not pd.DatetimeIndex(context.source_date).equals(source_dates)
            or not pd.DatetimeIndex(context.available_at).equals(available)
            or not context.available_at.lt(frame.decision_at).all()):
        raise ValueError("blocked: external source is not the prior native session at its declared availability")
    for field in FEATURES:
        frame[field] = context[field]
    frame["external_available_at"] = context.available_at
    frame["external_source_date"] = context.source_date
    if not np.isfinite(frame[[*BASE_FEATURES, *FEATURES]].to_numpy(float)).all():
        raise ValueError("blocked: incomplete common inference features")
    labels = ("soxl_return_h5", "soxx_return_h5")
    metadata = ("signal_date", "feature_available_at", "external_available_at", "h5_available_at",
        "decision_at", "trade_date", "exit_date", "return_available_at", "external_source_date")
    native = pd.concat({"feature": frame[[*BASE_FEATURES, *FEATURES]], "label": frame[list(labels)],
        "metadata": frame[list(metadata)]}, axis=1)
    native.index = pd.MultiIndex.from_arrays([frame.index, ["SOXL_SOXX_CASH"] * len(frame)],
        names=["datetime", "instrument"])
    dataset = DatasetH(handler=DataHandlerLP(instruments=None, start_time=frame.index[0], end_time=frame.index[-1],
        data_loader=StaticDataLoader(config=native), infer_processors=[], learn_processors=[]),
        segments={"all": (str(frame.index[0].date()), str(frame.index[-1].date()))})
    dataset.calendar = primary["calendar"].copy()
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"]).copy()
    pd.testing.assert_frame_equal(prepared, native, check_exact=True)
    return {"frame": frame, "dataset": dataset, "calendar": primary["calendar"],
        "covariance": primary["covariance"][np.flatnonzero(eligible)],
        "audit": {"primary": primary["audit"], "external": source_audit,
            "matched_training_start": plan["matched_training_start"], "matched_frame_sha256": frame_sha256(frame),
            "dataset_sha256": frame_sha256(prepared), "base_feature_columns": list(BASE_FEATURES),
            "extra_feature_columns": list(FEATURES), "rows": len(frame),
            "missing_rows_dropped": 0, "terminal_unsettled_daily_rows_excluded": int(all_frame.exit_date.isna().sum()),
            "terminal_unmatured_h5_features_retained": int(frame.h5_available_at.isna().sum())}}

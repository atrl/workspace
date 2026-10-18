"""Deterministic FINRA inputs from a verified source Recorder DatasetH frame.

The caller verifies the Recorder and raw-decimal lineage before calling this
adapter, then writes/reads the derived Qlib feature store before model use.
Historical 18:00 availability is an explicit assumption, not vintage or receipt
certification. This module neither loads market returns nor creates sessions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FINRA_BINARY_FEATURES = ("finra_both_ready", "chicago_regime", "fractional_regime")
FINRA_NUMERIC_FEATURES = (
    "log1p_SOXL_CNMS_TotalVolume", "log1p_SOXX_CNMS_TotalVolume", "finra_short_mix",
)
FINRA_FEATURES = FINRA_BINARY_FEATURES + FINRA_NUMERIC_FEATURES
FINRA_FEATURE_MASKS = {name: "finra_both_ready" for name in FINRA_NUMERIC_FEATURES}
_SOURCE_FIELDS = ("short_ratio", "total_volume", "source_ready")


def _daily_dates(index: pd.Index, *, role: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
        raise ValueError(f"blocked: {role} must use timezone-naive Qlib daily dates")
    if index.hasnans or not index.equals(index.normalize()):
        raise ValueError(f"blocked: {role} contains missing or intraday dates")
    return index


def build_finra_allocation_features(
    source_prepared: pd.DataFrame, *, signal_index: pd.DatetimeIndex,
    decision_at: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map source trade date d to the same primary signal date d, without fill.

    ``source_prepared`` is the float64, Decimal-derived source DatasetH.prepare
    result, indexed by ``datetime`` and ``instrument`` in either level order.
    ``signal_index`` and ``decision_at`` come from the existing primary Qlib
    timing surface. A NaT decision is retained and marked unexecutable; its
    source observation is not replaced by a fabricated next-session clock.
    """
    dates = _daily_dates(signal_index, role="primary signal index")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("blocked: primary signal dates must be unique and ordered")
    if not decision_at.index.equals(signal_index):
        raise ValueError("blocked: decision clocks do not exactly align with primary signals")
    frame = source_prepared
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame["feature"]
    if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {"datetime", "instrument"}:
        raise ValueError("blocked: FINRA source requires a native two-level Qlib index")
    if frame.index.nlevels != 2 or frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("blocked: duplicate or ambiguous FINRA source coordinates")
    source_dates = _daily_dates(frame.index.get_level_values("datetime"), role="FINRA source")
    symbols = frame.index.get_level_values("instrument")
    if not symbols.isin(["SOXL", "SOXX"]).all():
        raise ValueError("blocked: FINRA source contains an unexpected instrument")
    source = frame.loc[:, list(_SOURCE_FIELDS)].copy()
    if any(dtype != np.dtype("float64") for dtype in source.dtypes):
        raise ValueError("blocked: use Decimal-derived float64 source DatasetH, not the float32 Qlib cache")
    if not source.source_ready.isin([0., 1.]).all():
        raise ValueError("blocked: FINRA source readiness must be explicit binary values")
    ready = source.source_ready.eq(1)
    valid = (np.isfinite(source.short_ratio) & source.short_ratio.between(0., 1.)
             & np.isfinite(source.total_volume) & source.total_volume.gt(0.))
    if (ready & ~valid).any():
        raise ValueError("blocked: a ready FINRA source row violates numeric invariants")
    source.index = pd.MultiIndex.from_arrays([source_dates, symbols], names=["datetime", "instrument"])

    assumed = pd.Series((dates + pd.Timedelta(hours=18)).tz_localize("America/New_York"),
                        index=signal_index, name="assumed_available_at")
    # Reject naive clocks instead of silently guessing their timezone. All-NaT
    # clocks are allowed because the original calendar can have an unknown tail.
    clocks = []
    for value in decision_at:
        stamp = pd.Timestamp(value)
        if pd.notna(stamp) and stamp.tzinfo is None:
            raise ValueError("blocked: nonmissing decision clocks must be timezone-aware")
        clocks.append(stamp)
    decision = pd.Series(pd.to_datetime(clocks, utc=True).tz_convert("America/New_York"),
                         index=signal_index, name="decision_at")
    executable = decision.notna()
    known = decision.loc[executable]
    if not (known.dt.hour.eq(9) & known.dt.minute.eq(29) & known.dt.second.eq(0)
            & known.dt.microsecond.eq(0) & known.dt.nanosecond.eq(0)).all():
        raise ValueError("blocked: existing decision clock is not 09:29 America/New_York")
    if not assumed.loc[executable].lt(known).all():
        raise ValueError("blocked: assumed FINRA availability is not strictly before decision")

    aligned = {}
    for symbol in ("SOXL", "SOXX"):
        wanted = pd.MultiIndex.from_arrays([dates, [symbol] * len(dates)], names=source.index.names)
        aligned[symbol] = source.reindex(wanted).set_axis(signal_index)
    both_ready = aligned["SOXL"].source_ready.eq(1) & aligned["SOXX"].source_ready.eq(1)
    features = pd.DataFrame(index=signal_index, columns=list(FINRA_FEATURES), dtype=np.float64)
    features["finra_both_ready"] = both_ready.astype(float)
    features["chicago_regime"] = (dates >= pd.Timestamp("2018-09-10")).astype(float)
    features["fractional_regime"] = (dates >= pd.Timestamp("2026-02-23")).astype(float)
    for symbol in ("SOXL", "SOXX"):
        features.loc[both_ready, f"log1p_{symbol}_CNMS_TotalVolume"] = np.log1p(
            aligned[symbol].loc[both_ready, "total_volume"])
    features.loc[both_ready, "finra_short_mix"] = (
        aligned["SOXL"].loc[both_ready, "short_ratio"] - aligned["SOXX"].loc[both_ready, "short_ratio"])
    metadata = pd.DataFrame({"source_date": dates, "assumed_available_at": assumed,
                             "decision_at": decision, "decision_executable": executable}, index=signal_index)
    metadata.attrs["historical_availability"] = "conditional 18:00 ET; original vintage and first receipt unqualified"
    return features, metadata

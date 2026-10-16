"""Causal monthly valuation inputs; source qualification remains a separate gate."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.qlib_native import QlibProviderConfig, init_qlib_provider


def monthly_decision_schedule(provider_uri: Path, *, as_of: str) -> pd.DataFrame:
    """Only complete calendar months with an observed next-session execution date."""
    from qlib.data import D

    init_qlib_provider(QlibProviderConfig(provider_uri=provider_uri))
    days = pd.DatetimeIndex(D.calendar(freq="day"))
    days = days[days <= pd.Timestamp(as_of).normalize()]
    if days.has_duplicates or not days.is_monotonic_increasing:
        raise ValueError("invalid Qlib calendar")
    rows = []
    for day, following in zip(days[:-1], days[1:]):
        if day.to_period("M") != following.to_period("M"):
            rows.append({
                "signal_at": (day + pd.Timedelta(hours=16, minutes=15)).tz_localize("America/New_York"),
                "execution_at": (following + pd.Timedelta(hours=9, minutes=47)).tz_localize("America/New_York"),
            })
    return pd.DataFrame(rows, columns=["signal_at", "execution_at"])


def _utc(values, name):
    if any(pd.isna(v) or pd.Timestamp(v).tzinfo is None for v in values):
        raise ValueError(f"{name} requires explicit timezone and known clock")
    return pd.to_datetime(values, utc=True)


def align_monthly_valuations(events: pd.DataFrame, decisions: pd.DataFrame, *,
                             lookback_months=60, minimum_snapshots=36,
                             maximum_age_days=45, minimum_coverage=.95) -> pd.DataFrame:
    """Freeze each monthly state at its own cutoff, including its historical rank.

    observation_id identifies an economic snapshot, source_id its release/version.
    Repeated snapshots do not manufacture extra rank history. Negative earnings
    yields are valid. Missing company earnings are never relabelled profitable.
    Values here must already use the source-qualified basket aggregation method.
    Decisions must come from monthly_decision_schedule; this pure aligner does
    not certify exchange calendars or create return labels. For repeated older
    observations, the first historically selected value supplies rank history.
    """
    required = {"valuation_at", "available_at", "observation_id", "source_id", "method_id",
                "valuation_earnings_yield", "earnings_change", "loss_weight", "coverage_weight"}
    if not required.issubset(events):
        raise ValueError(f"missing valuation columns: {sorted(required-set(events))}")
    if not 0 < minimum_snapshots <= lookback_months or maximum_age_days < 0 or not 0 < minimum_coverage <= 1:
        raise ValueError("invalid frozen eligibility settings")
    e = events.copy()
    for col in ["valuation_at", "available_at"]:
        e[col] = _utc(e[col], col)
    if (e.available_at < e.valuation_at).any():
        raise ValueError("valuation cannot be known before its economic timestamp")
    for col in ["observation_id", "source_id", "method_id"]:
        if e[col].isna().any() or e[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"missing {col}")
    if e.duplicated(["valuation_at", "available_at"]).any():
        raise ValueError("ambiguous simultaneous valuation versions")
    if e.groupby("observation_id").valuation_at.nunique().gt(1).any():
        raise ValueError("economic observation identity changed")
    for col in ["coverage_weight", "loss_weight"]:
        if not e[col].between(0, 1).all():
            raise ValueError(f"invalid {col}")
    if (e.loss_weight > e.coverage_weight).any():
        raise ValueError("known loss weight exceeds known earnings coverage")
    e = e.sort_values(["valuation_at", "available_at"])
    cutoffs = pd.DatetimeIndex(_utc(decisions.signal_at, "signal_at"))
    if cutoffs.has_duplicates or not cutoffs.is_monotonic_increasing:
        raise ValueError("decisions must be unique and chronological")
    if cutoffs.tz_convert("America/New_York").tz_localize(None).to_period("M").duplicated().any():
        raise ValueError("multiple decisions in a calendar month")
    history, rows = [], []
    for cutoff in cutoffs:
        eligible = e.loc[(e.available_at <= cutoff) & (e.valuation_at <= cutoff)]
        row = {"signal_at": cutoff, "eligible": False, "status": "no_available_snapshot",
               "earnings_yield_percentile": np.nan, "rank_prior_snapshots": 0}
        if not eligible.empty:
            event = eligible.iloc[-1]
            row.update(event.to_dict())
            row["unknown_earnings_weight"] = 1 - float(event.coverage_weight)
            age = (cutoff-event.valuation_at).total_seconds()/86400
            row["snapshot_age_days"] = age
            finite = np.isfinite([event.valuation_earnings_yield, event.earnings_change]).all()
            known = age <= maximum_age_days and event.coverage_weight >= minimum_coverage and finite
            row["source_eligible"] = bool(known)
            row["status"] = "stale_or_incomplete" if not known else "insufficient_rank_history"
            if known:
                start = cutoff - pd.DateOffset(months=lookback_months)
                prior = [h for h in history if h["signal_at"] >= start
                         and h["method_id"] == event.method_id
                         and h["observation_id"] != event.observation_id]
                unique = {h["observation_id"]: h for h in reversed(prior)}
                values = np.array([h["valuation_earnings_yield"] for h in unique.values()])
                row["rank_prior_snapshots"] = len(values)
                if len(values) >= minimum_snapshots:
                    value = event.valuation_earnings_yield
                    row["earnings_yield_percentile"] = float(((values < value).sum()+.5*(values == value).sum())/len(values))
                    row.update(eligible=True, status="qualified_monthly_features")
                history.append(dict(row))
        rows.append(row)
    return pd.DataFrame(rows)

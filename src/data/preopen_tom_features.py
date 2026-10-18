"""Pure, conditional NYSE preopen calendar receiver; no I/O or model encoding.

The caller owns Qlib loading, source hashes, original row identity and any known
prefix data gaps.  This module never confirms that the decision date opened.
Its three masks themselves contain calendar information and belong to BOTH
matched arms.  The nullable proxy is the sole additional numeric coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.workflow.calendar_tom_source import _events_at, _planned_state


_ZONE = "America/New_York"
_VENUE = "NYSE"
_OPEN = {"planned_weekday_open", "early_close", "planned_reopening"}
_CLOSED = {"closed", "weekend"}
_KINDS = {"weekday_baseline", "planned_closed", "early_close", "temporary_closed", "reopening"}
_VERSIONS = {"append_only_first_version", "later_corroborating_version",
             "later_changed_notice_requires_asof"}
_CONFLICT_FREE = {"none", "dated_change_apply_only_when_known"}
_LEDGER_COLUMNS = {"assertion_id", "document_id", "venue", "assertion_kind",
                   "session_date", "effective_from", "effective_until",
                   "knowledge_at", "version_status", "conflict_status"}
_TIMING_COLUMNS = {"decision_at", "label_available_at", "return_available_at", "account_eligible"}
CANONICAL_COLUMNS = [
    "signal_date", "original_decision_at", "decision_date", "source_as_of",
    "prefix_available", "current_plan_known", "current_planned_state",
    "monthend_horizon_known", "first3_branch_known", "first3_branch_value_nullable",
    "expected_last_branch_known", "expected_last_branch_value_nullable",
    "proxy_known", "preopen_tom_proxy_nullable", "missing_announcement_reason",
    "amendment_gap_reason", "conflict_reason", "prefix_gap_reason", "version_gap_reason",
    "clock_gap_reason", "current_plan_document_ids", "horizon_document_ids",
    "knowledge_document_ids", "knowledge_version_ids", "knowledge_max_at",
    "historical_PIT_qualified", "account_eligible", "prefix_evidence_class",
    "receiver_calendar", "next_planned_open_after_month_end",
]


def _civil(value, name):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError(f"{name} must be a naive civil date")
    return result


def _aware_ny(series, name, *, allow_nat):
    if not isinstance(series.dtype, pd.DatetimeTZDtype) or str(series.dt.tz) != _ZONE:
        raise ValueError(f"{name} must preserve the original America/New_York timezone")
    if not allow_nat and series.isna().any():
        raise ValueError(f"{name} contains unknown timestamps")


def _validate(calendar, timing, ledger, as_of, prefix_start, amendment_gaps, prefix_gaps):
    if prefix_start > as_of:
        raise ValueError("prefix coverage starts after source as_of")
    if not isinstance(calendar, pd.DatetimeIndex) or calendar.empty:
        raise ValueError("calendar must be a nonempty DatetimeIndex")
    if (calendar.tz is not None or calendar.hasnans or calendar.has_duplicates
            or not calendar.is_monotonic_increasing or not calendar.equals(calendar.normalize())):
        raise ValueError("calendar has ambiguous or non-civil dates")
    if calendar[-1] > as_of:
        raise ValueError("calendar contains realized dates beyond source as_of")
    index = timing.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is not None or index.hasnans
            or index.has_duplicates or not index.is_monotonic_increasing
            or not index.equals(index.normalize())):
        raise ValueError("duplicate or inconsistent signal identity")
    if len(index) and index[-1] > as_of:
        raise ValueError("signal identity exceeds source as_of")
    if "decision_at" not in timing or not set(timing.columns) <= _TIMING_COLUMNS:
        raise ValueError("timing may contain only original clock metadata and account_eligible")
    _aware_ny(timing.decision_at, "decision_at", allow_nat=True)
    decisions = timing.decision_at.dropna()
    if decisions.duplicated().any():
        raise ValueError("duplicate nonempty decision mapping")
    for signal, clock in decisions.items():
        if (clock.hour, clock.minute, clock.second, clock.microsecond, clock.nanosecond) != (9, 29, 0, 0, 0):
            raise ValueError("nonempty decision clock differs from frozen09:29")
        if clock.tz_localize(None).normalize() <= signal:
            raise ValueError("decision date must be after its original signal date")
    for column in set(timing) & {"label_available_at", "return_available_at"}:
        if not isinstance(timing[column].dtype, pd.DatetimeTZDtype):
            raise ValueError(f"{column} lost timestamp metadata dtype")
    if "account_eligible" in timing and not pd.api.types.is_bool_dtype(timing.account_eligible.dtype):
        raise ValueError("account_eligible must be boolean metadata")
    if not _LEDGER_COLUMNS <= set(ledger):
        raise ValueError("required knowledge ledger columns missing")
    if ledger.assertion_id.isna().any() or ledger.assertion_id.duplicated().any():
        raise ValueError("duplicate or missing knowledge assertion identity")
    if ledger.document_id.isna().any() or not ledger.assertion_kind.isin(_KINDS).all():
        raise ValueError("invalid ledger document or assertion kind")
    _aware_ny(ledger.knowledge_at, "ledger.knowledge_at", allow_nat=False)
    for column in ["effective_from", "effective_until", "session_date"]:
        values = ledger[column]
        if not pd.api.types.is_datetime64_dtype(values.dtype) or getattr(values.dt, "tz", None) is not None:
            raise ValueError(f"ledger.{column} must contain naive dates")
        if not values.dropna().eq(values.dropna().dt.normalize()).all():
            raise ValueError(f"ledger.{column} contains intraday dates")
    if (ledger.effective_from.isna().any() or ledger.effective_until.isna().any()
            or (ledger.effective_until < ledger.effective_from).any()):
        raise ValueError("invalid ledger effective interval")
    events = ledger.loc[~ledger.assertion_kind.eq("weekday_baseline")]
    if events.session_date.isna().any():
        raise ValueError("dated ledger event has no session date")
    for gap in amendment_gaps:
        if not {"id", "venues", "affected_calendar_years", "unknown_from"} <= set(gap):
            raise ValueError("amendment gap schema missing")
        clock = pd.Timestamp(gap["unknown_from"])
        if pd.isna(clock) or clock.tzinfo is None:
            raise ValueError("amendment gap must have an explicit timezone")
    for gap in prefix_gaps:
        if not {"id", "start", "end"} <= set(gap):
            raise ValueError("prefix gap schema missing")
        if _civil(gap["end"], "prefix gap end") < _civil(gap["start"], "prefix gap start"):
            raise ValueError("invalid prefix gap interval")


@dataclass(frozen=True)
class _State:
    state: str
    documents: tuple = ()
    assertions: tuple = ()
    missing: tuple = ()
    amendment: tuple = ()
    conflict: tuple = ()
    version: tuple = ()


def _lookup(known, day, clock, amendment_gaps):
    # An old closed cell is not evidence that resolves a missing amendment.
    gaps = tuple(sorted(gap["id"] for gap in amendment_gaps
                        if _VENUE in gap["venues"] and day.year in gap["affected_calendar_years"]
                        and clock >= pd.Timestamp(gap["unknown_from"])))
    if gaps:
        return _State("unknown", amendment=gaps)
    base = known.loc[known.assertion_kind.eq("weekday_baseline")
                     & known.effective_from.le(day) & known.effective_until.ge(day)]
    if base.empty:
        return _State("unknown", missing=(f"no_known_ordinary_plan:{day.date()}",))
    base = base.loc[base.knowledge_at.eq(base.knowledge_at.max())]
    ordinary = known.loc[known.document_id.isin(base.document_id) & known.session_date.eq(day)
                         & known.assertion_kind.isin(["planned_closed", "early_close"])]
    exceptions = known.loc[known.session_date.eq(day)
                           & known.assertion_kind.isin(["temporary_closed", "reopening"])]
    if not exceptions.empty:
        exceptions = exceptions.loc[exceptions.knowledge_at.eq(exceptions.knowledge_at.max())]
    relevant_ids = set(base.assertion_id) | set(ordinary.assertion_id) | set(exceptions.assertion_id)
    relevant = known.loc[known.assertion_id.isin(relevant_ids)]
    state, documents, conflicting = _planned_state(known, day)
    assertions = tuple(sorted(relevant.assertion_id))
    unresolved = relevant.loc[~relevant.conflict_status.isin(_CONFLICT_FREE)]
    bad_version = relevant.loc[~relevant.version_status.isin(_VERSIONS)]
    conflicts = (f"conflicting_plan:{day.date()}",) if conflicting or not unresolved.empty else ()
    versions = (f"unqualified_version:{day.date()}",) if not bad_version.empty else ()
    if conflicts or versions:
        return _State("unknown", documents, assertions, conflict=conflicts, version=versions)
    return _State(state, documents, assertions)


def _announced_horizon(day, lookup):
    """Only announced dates; no observed-calendar lookup, and no source _horizon."""
    end = day + pd.offsets.MonthEnd(0)
    evidence, later_open = [], False
    for planned_day in pd.date_range(day, end, freq="D"):
        state = lookup(planned_day)
        evidence.append(state)
        if state.state == "unknown":
            return False, None, pd.NaT, evidence
        if planned_day > day and state.state in _OPEN:
            later_open = True
    planned_day = end + pd.DateOffset(days=1)
    # Each lookup needs a finite dated baseline; exhaustion returns unknown.
    # No arbitrary shorter lookahead substitutes for the next planned open.
    while True:
        state = lookup(planned_day)
        evidence.append(state)
        if state.state == "unknown":
            return False, None, pd.NaT, evidence
        if state.state in _OPEN:
            return True, not later_open, planned_day, evidence
        planned_day += pd.DateOffset(days=1)


def build_preopen_tom_features(*, calendar: pd.DatetimeIndex, timing: pd.DataFrame,
                               ledger: pd.DataFrame, amendment_gaps: list,
                               as_of: str, prefix_start: date,
                               prefix_gaps: tuple = ()) -> pd.DataFrame:
    """Return canonical nullable features indexed by the unchanged signal dates.

    ``timing.decision_at`` must retain its original NY timezone and09:29 clock.
    Optional account_eligible is passed through, never used by feature logic;
    absent metadata is NA, while missing/out-of-as_of decisions are ineligible.
    ``prefix_start`` and any explicit inclusive prefix_gaps are upstream source
    coverage metadata. Missing dates themselves are not guessed to be data gaps.
    """
    end = _civil(as_of, "as_of")
    start = _civil(prefix_start, "prefix_start")
    _validate(calendar, timing, ledger, end, start, amendment_gaps, prefix_gaps)
    rows, plan_cache = [], {}
    for signal, source in timing.iterrows():
        clock = source.decision_at
        day = pd.NaT if pd.isna(clock) else clock.tz_localize(None).normalize()
        valid = pd.notna(day) and day <= end
        past = calendar[calendar < day] if pd.notna(day) else calendar[:0]
        if pd.notna(day) and (past.empty or past[-1] != signal):
            raise ValueError("original signal is not the last d-prior core session")
        row = {"signal_date": signal, "original_decision_at": clock,
               "decision_date": day, "source_as_of": end,
               "prefix_available": False, "current_plan_known": False,
               "current_planned_state": "unknown", "monthend_horizon_known": False,
               "first3_branch_known": False, "first3_branch_value_nullable": np.nan,
               "expected_last_branch_known": False, "expected_last_branch_value_nullable": np.nan,
               "proxy_known": False, "preopen_tom_proxy_nullable": np.nan,
               "missing_announcement_reason": "", "amendment_gap_reason": "",
               "conflict_reason": "", "prefix_gap_reason": "", "version_gap_reason": "",
               "clock_gap_reason": "", "current_plan_document_ids": (),
               "horizon_document_ids": (), "knowledge_document_ids": (),
               "knowledge_version_ids": (), "knowledge_max_at": pd.NaT,
               "historical_PIT_qualified": False,
               "account_eligible": source.get("account_eligible", pd.NA) if valid else False,
               "prefix_evidence_class": "conditional_core_common_session_snapshot_not_official_NYSE_opens",
               "receiver_calendar": _VENUE, "next_planned_open_after_month_end": pd.NaT}
        if not valid:
            row["clock_gap_reason"] = "original_decision_missing" if pd.isna(day) else "decision_beyond_asof"
            rows.append(row)
            continue
        # The only numerical use of actual sessions is the strict d-prior prefix.
        month_start = day.replace(day=1)
        prefix_reasons = [] if start <= month_start else ["month_start_before_prefix_coverage"]
        prefix_reasons.extend(gap["id"] for gap in prefix_gaps
                              if pd.Timestamp(gap["start"]) < day and pd.Timestamp(gap["end"]) >= month_start)
        row["prefix_available"] = not prefix_reasons
        row["prefix_gap_reason"] = "|".join(sorted(set(prefix_reasons)))
        known = _events_at(ledger, _VENUE, clock)
        context = (tuple(known.assertion_id), tuple(sorted(gap["id"] for gap in amendment_gaps
                   if _VENUE in gap["venues"] and clock >= pd.Timestamp(gap["unknown_from"]))))

        def lookup(planned_day):
            key = (context, planned_day)
            if key not in plan_cache:
                plan_cache[key] = _lookup(known, planned_day, clock, amendment_gaps)
            return plan_cache[key]

        current = lookup(day)
        row["current_planned_state"] = current.state
        row["current_plan_known"] = current.state in _OPEN | _CLOSED
        row["current_plan_document_ids"] = current.documents
        horizon_known, last_value, next_open, horizon = _announced_horizon(day, lookup)
        row["monthend_horizon_known"] = horizon_known
        row["next_planned_open_after_month_end"] = next_open
        if current.state in _CLOSED:
            row.update(first3_branch_known=True, first3_branch_value_nullable=0.,
                       expected_last_branch_known=True, expected_last_branch_value_nullable=0.)
        elif current.state in _OPEN:
            if row["prefix_available"]:
                count_before = int((past >= month_start).sum())
                row.update(first3_branch_known=True,
                           first3_branch_value_nullable=float(count_before + 1 in (1, 2, 3)))
            if horizon_known:
                row.update(expected_last_branch_known=True,
                           expected_last_branch_value_nullable=float(last_value))
        left, right = row["first3_branch_value_nullable"], row["expected_last_branch_value_nullable"]
        if (row["first3_branch_known"] and left == 1.) or (row["expected_last_branch_known"] and right == 1.):
            row.update(proxy_known=True, preopen_tom_proxy_nullable=1.)
        elif row["first3_branch_known"] and row["expected_last_branch_known"]:
            row.update(proxy_known=True, preopen_tom_proxy_nullable=0.)
        evidence = [current, *horizon]
        for attribute, output in [("missing", "missing_announcement_reason"),
                                  ("amendment", "amendment_gap_reason"),
                                  ("conflict", "conflict_reason"), ("version", "version_gap_reason")]:
            row[output] = "|".join(sorted({item for state in evidence for item in getattr(state, attribute)}))
        row["horizon_document_ids"] = tuple(sorted({doc for state in horizon for doc in state.documents}))
        row["knowledge_document_ids"] = tuple(sorted({doc for state in evidence for doc in state.documents}))
        row["knowledge_version_ids"] = tuple(sorted({key for state in evidence for key in state.assertions}))
        cited = known.loc[known.assertion_id.isin(row["knowledge_version_ids"])]
        row["knowledge_max_at"] = pd.NaT if cited.empty else cited.knowledge_at.max()
        rows.append(row)
    result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS, index=timing.index.copy())
    result.index.name = "signal_date"
    result["original_decision_at"] = timing.decision_at.copy()
    for column in ["signal_date", "decision_date", "source_as_of", "next_planned_open_after_month_end"]:
        result[column] = pd.to_datetime(result[column])
    result["knowledge_max_at"] = pd.to_datetime(result.knowledge_max_at, utc=True).dt.tz_convert(_ZONE)
    for column in ["prefix_available", "current_plan_known", "monthend_horizon_known",
                   "first3_branch_known", "expected_last_branch_known", "proxy_known", "historical_PIT_qualified"]:
        result[column] = result[column].astype(bool)
    result["account_eligible"] = pd.array(result.account_eligible, dtype="boolean")
    for column in ["first3_branch_value_nullable", "expected_last_branch_value_nullable", "preopen_tom_proxy_nullable"]:
        result[column] = result[column].astype("float64")
    return result

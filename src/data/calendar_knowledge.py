"""Parse pinned announcement bytes into append-only source assertions.

This is a pure ingestion helper, not an exchange calendar, a Qlib data reader,
or a feature builder.  In particular, a weekday_baseline asserts only an
announced ordinary annual schedule; it does not assert that no later emergency
closure occurred.  Consumers must filter knowledge_at before resolving versions.

document_specs is keyed by receipt id.  Each admitted specification contains:
  parser_type: ice_calendar | nasdaq_notice
  publication: {precision: day | timestamp, value: ISO value, anchor: body text}
  scope_anchor: verbatim normalized visible text supporting the named venues
  venues: [NYSE, NYSE_ARCA] for ICE, or [NASDAQ] for Nasdaq
  calendar: {table_index, years, full_years} for ICE
  events: [{assertion_kind: temporary_closed | reopening, date, anchor}] otherwise

For day precision, published_at is NaT and knowledge_at is the following local
midnight.  That convention does not establish a true historical receipt time.
Source spans index canonical_visible_text(raw), NOT byte offsets in the HTML.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup


_NY = ZoneInfo("America/New_York")
_MONTHS = {name.lower(): number for number, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}
_DATE = re.compile(
    r"(?:(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
    r"(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:\s*(?:st|nd|rd|th))?"
    r"(?:\s*,?\s+(\d{4}))?", re.I)
_EARLY = re.compile(
    r"Each market will close early at (\d{1,2}):(\d{2})\s*p\.m\."
    r"\s*(?:\([^)]*\))?\s*on (.+?)\.(?=\s|$)", re.I)
_HOLIDAY = re.compile(
    r"^(?:New Year.?s Day|Martin Luther King.*Day|Washington.?s Birthday|"
    r"Presidents.? Day|Good Friday|Memorial Day|Juneteenth.*Day|"
    r"Independence Day|Labor Day|Thanksgiving Day|Christmas Day)\s*[*†┼]*$", re.I)

LEDGER_COLUMNS = [
    "assertion_id", "document_id", "raw_sha256", "venue", "assertion_kind",
    "session_date", "session_date_or_rule", "effective_from", "effective_until",
    "published_date", "published_at", "publication_precision", "knowledge_at",
    "knowledge_basis", "scope_source_span", "publication_source_span",
    "source_span", "source_text", "source_html_locator", "rule_priority",
    "rule_year", "holiday_name", "cash_close_time", "schedule_state",
    "version_status", "supersedes_assertion_id", "conflict_status",
    "historical_vintage_status",
]
STATUS_COLUMNS = [
    "document_id", "parser_status", "gap_reason", "raw_sha256", "expected_sha256",
    "raw_byte_count", "http_status", "assertion_count", "receipt_status",
]


def canonical_visible_text(raw: bytes) -> str:
    """Stable text surface used by every recorded character span."""
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def _text(node: Any) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _span(text: str, anchor: str, occurrence: int = 0) -> dict[str, Any]:
    if not isinstance(anchor, str) or not anchor or occurrence < 0:
        raise ValueError("invalid source anchor")
    starts = [match.start() for match in re.finditer(re.escape(anchor), text)]
    if occurrence >= len(starts):
        raise ValueError("source anchor absent")
    start = starts[occurrence]
    return {"surface": "canonical_visible_text_v1", "start": start,
            "end": start + len(anchor), "occurrence": occurrence,
            "match_count": len(starts)}


def _dates(text: str, default_year: int | None = None) -> list[date]:
    dates = []
    for match in _DATE.finditer(text):
        year = int(match[4]) if match[4] else default_year
        if year is None:
            raise ValueError("explicit year required in early-close sentence")
        parsed = date(year, _MONTHS[match[2].lower()], int(match[3]))
        if match[1] and parsed.strftime("%A").lower() != match[1].lower():
            raise ValueError("source weekday disagrees with stated date")
        dates.append(parsed)
    return dates


def _publication(spec: dict, text: str) -> dict:
    publication = spec["publication"]
    anchor = publication["anchor"]
    source_span = _span(text, anchor, publication.get("occurrence", 0))
    precision = publication["precision"]
    if precision == "day":
        day = date.fromisoformat(publication["value"])
        # A dateline must itself carry the date; metadata alone is insufficient.
        parsed_anchor = pd.to_datetime(anchor, errors="raise")
        if parsed_anchor.date() != day or parsed_anchor.tzinfo is not None:
            raise ValueError("publication day anchor disagrees with specification")
        knowledge = pd.Timestamp(datetime.combine(day + timedelta(days=1),
                                                  datetime.min.time(), tzinfo=_NY))
        published_at = pd.NaT
        basis = "date_only_next_local_midnight_convention_not_first_receipt"
    elif precision == "timestamp":
        value = publication["value"]
        if value not in anchor:
            raise ValueError("exact publication timestamp absent from anchor")
        published_at = pd.Timestamp(value)
        if published_at.tzinfo is None:
            raise ValueError("publication timestamp must include timezone")
        published_at = published_at.tz_convert(_NY)
        knowledge = published_at
        day = published_at.date()
        basis = "explicit_body_timestamp"
    else:
        raise ValueError("unknown publication precision")
    return {"published_date": day.isoformat(), "published_at": published_at,
            "publication_precision": precision, "knowledge_at": knowledge,
            "knowledge_basis": basis, "publication_source_span": source_span}


def _venues(spec: dict, text: str) -> tuple[list[str], dict]:
    anchor = spec["scope_anchor"]
    scope = _span(text, anchor, spec.get("scope_occurrence", 0))
    venues = spec["venues"]
    if not venues or len(set(venues)) != len(venues):
        raise ValueError("venues must be nonempty and distinct")
    if spec["parser_type"] == "ice_calendar":
        if not set(venues) <= {"NYSE", "NYSE_ARCA"}:
            raise ValueError("ICE venue mapping is limited to named cash-equity venues")
        if "cash equity markets" not in anchor.lower() or "europe" in anchor.lower():
            raise ValueError("ICE scope is not U.S. cash-equity markets")
        required = {"NYSE": "New York Stock Exchange", "NYSE_ARCA": "NYSE Arca"}
        if any(required[venue].lower() not in anchor.lower() for venue in venues):
            raise ValueError("requested venue absent from scope anchor")
    elif spec["parser_type"] == "nasdaq_notice":
        if venues != ["NASDAQ"] or not re.search(r"nasdaq", anchor, re.I):
            raise ValueError("Nasdaq notice cannot be mapped to another venue")
        if not re.search(r"U\.S\.\s+equit", anchor, re.I):
            raise ValueError("Nasdaq U.S. equity scope missing")
    else:
        raise ValueError("unsupported parser_type")
    return venues, scope


def _row(common: dict, venue: str, kind: str, text: str, anchor: str,
         *, session: date | None = None, year: int | None = None,
         holiday: str | None = None, close_time: str | None = None,
         locator: str = "normalized_text", source_span: dict | None = None) -> dict:
    baseline = kind == "weekday_baseline"
    if baseline and year is None:
        raise ValueError("baseline year missing")
    if not baseline and session is None:
        raise ValueError("explicit event date missing")
    effective_from = date(year, 1, 1) if baseline else session
    effective_until = date(year, 12, 31) if baseline else session
    state = {"weekday_baseline": "ordinary_weekday_rule_not_realized_open",
             "planned_closed": "closed", "temporary_closed": "closed",
             "reopening": "open", "early_close": "open_short_session"}[kind]
    span = source_span if source_span is not None else _span(text, anchor)
    if text[span["start"]:span["end"]] != anchor:
        raise ValueError("source span does not reproduce source text")
    row = {**common, "venue": venue, "assertion_kind": kind,
           "session_date": pd.NaT if baseline else pd.Timestamp(session),
           "session_date_or_rule": f"weekday_baseline:{year}" if baseline else session.isoformat(),
           "effective_from": pd.Timestamp(effective_from),
           "effective_until": pd.Timestamp(effective_until),
           "source_span": span, "source_text": anchor,
           "source_html_locator": locator, "rule_priority": 20 if kind in
           {"temporary_closed", "reopening"} else 10,
           "rule_year": year if baseline else session.year,
           "holiday_name": holiday, "cash_close_time": close_time,
           "schedule_state": state, "version_status": "append_only_first_version",
           "supersedes_assertion_id": None, "conflict_status": "none"}
    identity = "|".join(str(row[key]) for key in
                        ["document_id", "raw_sha256", "venue", "assertion_kind",
                         "session_date_or_rule", "source_html_locator", "source_text"])
    row["assertion_id"] = hashlib.sha256(identity.encode()).hexdigest()
    return row


def _ice_rows(raw: bytes, spec: dict, common: dict, text: str,
              venues: list[str]) -> list[dict]:
    config = spec["calendar"]
    years, full = set(config["years"]), set(config["full_years"])
    if not years or not full <= years or any(type(y) is not int for y in years):
        raise ValueError("invalid named calendar years")
    tables = BeautifulSoup(raw, "html.parser").find_all("table")
    table = tables[config["table_index"]]
    table_text = _text(table)
    table_span = _span(text, table_text)
    source_cursor = table_span["start"]
    events, seen, seen_holidays = [], set(), {year: set() for year in years}
    column_years = None
    current_year = None
    for ri, tr in enumerate(table.find_all("tr")):
        row_text = _text(tr)
        start = text.find(row_text, source_cursor, table_span["end"])
        if row_text and start < 0:
            raise ValueError("table row cannot be located on canonical text surface")
        source_cursor = start + len(row_text) if row_text else source_cursor
        row_span = {"surface": "canonical_visible_text_v1", "start": start,
                    "end": source_cursor, "container": "table", "row_index": ri}
        cells = [(_text(cell), ci) for ci, cell in enumerate(
            tr.find_all(["td", "th"], recursive=False)) if _text(cell)]
        if not cells:
            continue
        values = [value for value, _ in cells]
        section = re.fullmatch(r"(\d{4}) Holidays", values[0], re.I)
        if section and len(values) == 1:
            current_year = int(section[1])
            if current_year not in years:
                raise ValueError("unexpected year section")
            seen.add(current_year)
            continue
        if values[0].upper() == "HOLIDAY":
            column_years = [int(value) for value in values[1:]]
            if len(set(column_years)) != len(column_years) or set(column_years) != years:
                raise ValueError("table column years disagree with pinned specification")
            seen.update(column_years)
            continue
        if not _HOLIDAY.fullmatch(values[0]):
            if len(values) > 1:
                raise ValueError("unrecognized calendar row")
            continue
        row_years = column_years if column_years is not None else [current_year]
        if current_year is None and column_years is None:
            raise ValueError("holiday precedes its named year")
        if len(values) != len(row_years) + 1:
            raise ValueError("holiday/date column count mismatch")
        for year, (value, ci) in zip(row_years, cells[1:]):
            if values[0] in seen_holidays[year]:
                raise ValueError("duplicate holiday row in one year")
            seen_holidays[year].add(values[0])
            # A dash is explicit non-observation; it is not a closed date.
            if re.fullmatch(r"[—–-]\s*\**", value):
                continue
            dates = _dates(value, year)
            if len(dates) == 1:
                session = dates[0]
            elif "observed" in value.lower():
                observed = re.split(r"observed", value, flags=re.I)[-1]
                observed_dates = _dates(observed, year)
                if len(observed_dates) != 1:
                    raise ValueError("observed date is ambiguous")
                session = observed_dates[0]
            else:
                raise ValueError("holiday date is missing or ambiguous")
            if session.year != year:
                raise ValueError("holiday year differs from its table year")
            anchor = _text(tr)
            for venue in venues:
                events.append(_row(common, venue, "planned_closed", text, anchor,
                                   session=session, holiday=values[0],
                                   locator=f"table[{config['table_index']}]/tr[{ri}]/cell[{ci}]",
                                   source_span=row_span))
    if seen != years or any(not seen_holidays[year] for year in years):
        raise ValueError("named calendar year missing from source table")
    # Full-year designation is separately frozen, not inferred from row count.
    for year in sorted(full):
        for venue in venues:
            events.append(_row(common, venue, "weekday_baseline", text, table_text,
                               year=year, locator=f"table[{config['table_index']}]/year[{year}]"))
    matches = list(_EARLY.finditer(text))
    if not matches:
        raise ValueError("early-close sentences missing or unparseable")
    for mi, match in enumerate(matches):
        hour, minute = int(match[1]), int(match[2])
        if not 1 <= hour <= 12 or not 0 <= minute <= 59:
            raise ValueError("invalid cash early-close time")
        close_time = f"{hour % 12 + 12:02d}:{minute:02d}"
        dates = _dates(match[3])
        if not dates or any(day.year not in years for day in dates):
            raise ValueError("early-close dates outside named years")
        for session in dates:
            for venue in venues:
                events.append(_row(common, venue, "early_close", text, match[0],
                                   session=session, close_time=close_time,
                                   locator=f"early_close_sentence[{mi}]"))
    # A document cannot simultaneously close and shorten the same session.
    closed = {(r["venue"], r["session_date"]) for r in events
              if r["assertion_kind"] == "planned_closed"}
    if any((r["venue"], r["session_date"]) in closed for r in events
           if r["assertion_kind"] == "early_close"):
        raise ValueError("one document closes and early-closes the same session")
    return events


def _nasdaq_rows(spec: dict, common: dict, text: str, venues: list[str]) -> list[dict]:
    events = []
    if not spec.get("events"):
        raise ValueError("Nasdaq explicit event list is empty")
    for i, event in enumerate(spec["events"]):
        anchor = event["anchor"]
        _span(text, anchor)
        if re.search(r"\b(likely|expects?|expected|may|might|pending|tentative|if)\b", anchor, re.I):
            raise ValueError("tentative announcement cannot become a definite assertion")
        session = date.fromisoformat(event["date"])
        dates = _dates(anchor, date.fromisoformat(common["published_date"]).year)
        if dates != [session]:
            raise ValueError("notice date differs from the sole explicit source date")
        kind = event["assertion_kind"]
        if kind == "temporary_closed":
            if not re.search(r"\bwill (?:be closed|close)\b|decision to close", anchor, re.I):
                raise ValueError("definite closure wording missing")
        elif kind == "reopening":
            if not re.search(r"\bwill (?:be open|open)\b", anchor, re.I):
                raise ValueError("definite reopening wording missing")
        else:
            raise ValueError("unsupported Nasdaq assertion kind")
        events.append(_row(common, venues[0], kind, text, anchor, session=session,
                           locator=f"pinned_notice_sentence[{i}]"))
    return events


def _versions(rows: list[dict]) -> list[dict]:
    """Annotate later records without invalidating earlier knowledge states."""
    ordered = sorted(rows, key=lambda r: (r["knowledge_at"], r["document_id"], r["assertion_id"]))
    prior: dict[tuple, list[dict]] = {}
    for row in ordered:
        key = (row["venue"], "baseline", row["rule_year"]) if row["assertion_kind"] == "weekday_baseline" else (
            row["venue"], "event", row["session_date"])
        existing = prior.setdefault(key, [])
        state = (row["schedule_state"], row["cash_close_time"])
        if existing:
            latest = existing[-1]
            same_clock = [r for r in existing if r["knowledge_at"] == row["knowledge_at"]]
            contradictory = [r for r in same_clock
                             if (r["schedule_state"], r["cash_close_time"]) != state]
            if contradictory:
                row["conflict_status"] = "unresolved_same_knowledge_time"
                row["version_status"] = "same_clock_conflict"
                for previous in contradictory:
                    previous["conflict_status"] = row["conflict_status"]
                    previous["version_status"] = row["version_status"]
            elif latest["knowledge_at"] < row["knowledge_at"]:
                row["supersedes_assertion_id"] = latest["assertion_id"]
                changed = (latest["schedule_state"], latest["cash_close_time"]) != state
                row["version_status"] = "later_changed_notice_requires_asof" if changed else "later_corroborating_version"
                if changed:
                    row["conflict_status"] = "dated_change_apply_only_when_known"
        existing.append(row)
    return ordered


def build_knowledge_ledger(documents: list, raw_bodies: dict[str, bytes], *,
                           document_specs: dict) -> dict[str, Any]:
    """Return source assertions and explicit failures; perform no I/O.

    Missing/failed/unpinned raw bodies and parser errors emit a document gap and
    zero assertions for that document.  They never become claims of no closure.
    Successful parsing is not full-period, immutable-vintage or PIT acceptance.
    """
    ids = [document["id"] for document in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate document id")
    statuses, gaps, rows = [], [], []
    for document in documents:
        ident = document["id"]
        receipt = document.get("native_capture", document)
        expected = receipt.get("sha256")
        raw = raw_bodies.get(ident)
        actual = hashlib.sha256(raw).hexdigest() if isinstance(raw, bytes) else None
        status = {"document_id": ident, "parser_status": "gap", "gap_reason": None,
                  "raw_sha256": actual, "expected_sha256": expected,
                  "raw_byte_count": len(raw) if isinstance(raw, bytes) else None,
                  "http_status": receipt.get("http_status"), "assertion_count": 0,
                  "receipt_status": receipt.get("status")}
        try:
            if receipt.get("http_status") != 200:
                raise ValueError("receipt_has_no_successful_raw_response")
            if raw is None:
                raise ValueError("raw_body_missing")
            if not isinstance(raw, bytes):
                raise ValueError("raw_body_must_be_bytes")
            if not expected or actual != expected:
                raise ValueError("raw_sha256_unpinned_or_mismatched")
            if receipt.get("bytes") is not None and receipt["bytes"] != len(raw):
                raise ValueError("raw_byte_count_mismatch")
            if ident not in document_specs:
                raise ValueError("document_spec_missing")
            spec = document_specs[ident]
            if spec.get("raw_sha256", actual) != actual:
                raise ValueError("parser_spec_raw_sha256_mismatch")
            text = canonical_visible_text(raw)
            publication = _publication(spec, text)
            if document.get("publication_date") not in (None, publication["published_date"]):
                raise ValueError("receipt_publication_date_disagrees_with_body")
            venues, scope = _venues(spec, text)
            common = {"document_id": ident, "raw_sha256": actual, **publication,
                      "scope_source_span": scope,
                      "historical_vintage_status": "current_dated_source_not_immutable_first_vintage"}
            parsed = (_ice_rows(raw, spec, common, text, venues) if spec["parser_type"] == "ice_calendar"
                      else _nasdaq_rows(spec, common, text, venues))
            if not parsed:
                raise ValueError("no_assertions_parsed")
            rows.extend(parsed)
            status.update(parser_status="parsed_conditional_source", assertion_count=len(parsed))
        except (KeyError, ValueError, IndexError, TypeError) as error:
            status["gap_reason"] = str(error)
            gaps.append({"document_id": ident, "reason": str(error), "assertions_admitted": 0})
        statuses.append(status)
    for ident in sorted(set(raw_bodies) - set(ids)):
        gaps.append({"document_id": ident, "reason": "raw_id_not_in_document_receipts", "assertions_admitted": 0})
    ledger = pd.DataFrame(_versions(rows), columns=LEDGER_COLUMNS)
    for column in ["session_date", "effective_from", "effective_until"]:
        ledger[column] = pd.to_datetime(ledger[column])
    for column in ["knowledge_at", "published_at"]:
        ledger[column] = pd.to_datetime(ledger[column], utc=True).dt.tz_convert(_NY)
    return {"ledger": ledger, "document_status": pd.DataFrame(statuses, columns=STATUS_COLUMNS),
            "source_gaps": gaps}

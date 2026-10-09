"""Issuer cash-event qualification, not an investment or account ledger.

Amounts remain exact Decimal strings. Economic dates and evidence availability
are separate; neither a current download nor an archive capture invents an
announcement timestamp. The caller supplies the Qlib-derived native calendar.
"""
from __future__ import annotations

import base64
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET

import pandas as pd


ISHARES_FIELDS = ["Record Date", "Ex-Date", "Payable Date", "Total Distribution",
                  "Income", "ST Cap Gains", "LT Cap Gains", "Return of Capital"]
DIREXION_FIELDS = ["Record Date", "Ex Date", "Pay Date", "Income Dividend",
                   "Short-Term Capital Gain", "Long-Term Capital Gain"]
_SS = "urn:schemas-microsoft-com:office:spreadsheet"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _decimal(value):
    _require(isinstance(value, (str, Decimal)) and str(value).strip() != "", "missing decimal amount")
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("blocked: invalid decimal amount") from error
    _require(amount.is_finite() and amount >= 0, "amount must be finite and nonnegative")
    return amount


def _date(value):
    _require(isinstance(value, str), "date must be an explicit source string")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError("blocked: invalid or ambiguous event date")


def _normal_dates(frame, columns):
    frame.attrs["source_tokens"] = frame.to_dict("records")
    for name in columns:
        frame[name] = frame[name].map(_date)
    return frame


def parse_ishares_distribution_workbook(raw_bytes):
    """Parse only the unique SpreadsheetML sheet, retaining original amounts.

    The real workbook has an invalid bare ampersand in an unrelated hyperlink.
    Isolating the sheet avoids repairing arbitrary XML or changing source bytes.
    The selected sheet itself must be valid, explicit and unambiguous XML.
    """
    _require(isinstance(raw_bytes, bytes), "workbook must be original bytes")
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("blocked: workbook is not UTF-8 SpreadsheetML") from error
    sheets = re.findall(r'<(?:ss:)?Worksheet\b[^>]*\bss:Name=["\']Distributions["\'][^>]*>.*?</(?:ss:)?Worksheet>',
                        text, re.S)
    _require(len(sheets) == 1, "need exactly one Distributions worksheet")
    try:
        wrapper = ET.fromstring(f'<root xmlns:ss="{_SS}">{sheets[0]}</root>')
    except ET.ParseError as error:
        raise ValueError("blocked: invalid Distributions worksheet XML") from error
    rows = []
    for row in wrapper.findall(f".//{{{_SS}}}Row"):
        values, cursor = {}, 1
        for cell in row:
            _require(cell.tag == f"{{{_SS}}}Cell", "unexpected row element")
            _require(not any(k.endswith(("Formula", "MergeAcross", "MergeDown")) for k in cell.attrib),
                     "formulas and merged distribution cells are unsupported")
            position = cell.attrib.get(f"{{{_SS}}}Index", str(cursor))
            _require(position.isdigit() and cursor <= int(position) <= len(ISHARES_FIELDS),
                     "duplicate, reversed or out-of-range cell index")
            position = int(position)
            nodes = list(cell)
            _require(len(nodes) == 1 and nodes[0].tag == f"{{{_SS}}}Data" and len(nodes[0]) == 0,
                     "distribution cell must contain one plain Data element")
            values[position] = nodes[0].text or ""
            cursor = position + 1
        _require(set(values) == set(range(1, len(ISHARES_FIELDS) + 1)), "missing distribution cell")
        rows.append([values[i] for i in range(1, len(ISHARES_FIELDS) + 1)])
    _require(len(rows) > 1 and rows[0] == ISHARES_FIELDS, "unexpected issuer sheet header or empty sheet")
    frame = pd.DataFrame(rows[1:], columns=ISHARES_FIELDS)
    _require(frame.apply(lambda c: c.map(lambda v: bool(v.strip()))).to_numpy().all(), "blank source cell")
    return _normal_dates(frame, ISHARES_FIELDS[:3])


def parse_direxion_distribution_table(table_csv_bytes):
    """Read the frozen, explicitly labelled browser-DOM table transcription."""
    _require(isinstance(table_csv_bytes, bytes), "DOM table capture must be bytes")
    try:
        rows = list(csv.reader(io.StringIO(table_csv_bytes.decode("utf-8-sig"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("blocked: invalid DOM table CSV") from error
    _require(len(rows) > 1 and rows[0] == DIREXION_FIELDS, "unexpected DOM table header or empty table")
    _require(all(len(r) == len(DIREXION_FIELDS) and all(v.strip() for v in r) for r in rows[1:]),
             "missing or extra DOM table cell/row")
    return _normal_dates(pd.DataFrame(rows[1:], columns=DIREXION_FIELDS), DIREXION_FIELDS[:3])


def _calendar(values):
    dates = pd.DatetimeIndex(values)
    _require(len(dates) > 0 and dates.tz is None and not dates.hasnans
             and dates.is_unique and dates.is_monotonic_increasing
             and dates.equals(dates.normalize()), "native calendar must be complete, unique, ordered naive dates")
    return dates


def _calendar_status(date, calendar):
    day = pd.Timestamp(date)
    if day < calendar[0] or day > calendar[-1]:
        return "outside_reference_range"
    _require(day in calendar, f"in-range event date absent from native calendar: {date}")
    return "native_session"


def _source_file(plan, relative):
    items = [f for f in plan["source_archive"]["files"] if f["archive_relative_path"] == relative]
    _require(len(items) == 1, "source identity is absent or ambiguous")
    return items[0]


def qualify_etf_corporate_actions(source_rows, split_events, *, native_calendar,
                                 market_as_of, source_contract):
    """Normalize all declared rows without filling history or running a ledger."""
    plan = source_contract
    calendar, as_of = _calendar(native_calendar), _date(market_as_of)
    _require(as_of == plan["market_as_of"] and str(calendar[-1].date()) == as_of,
             "market cutoff and native calendar differ")
    _require(set(source_rows) == set(plan["sources"]), "issuer inventory differs")
    _require(split_events == plan["split_events"], "split events differ from frozen source contract")
    splits, events = [], []
    for event in split_events:
        effective, record, payable = map(_date, (event["effective_trading_date"], event["record_date"],
                                               event["payable_after_close_date"]))
        ratio = event["new_shares_per_old_share"]
        _require(type(ratio) is int and ratio > 0 and record <= payable < effective <= as_of,
                 "invalid or future effective split")
        status = _calendar_status(effective, calendar)
        _require(status == "native_session", "declared split lacks native reference coverage")
        following = calendar[calendar > pd.Timestamp(payable)]
        _require(len(following) and str(following[0].date()) == effective,
                 "split effective date is not first native session after payable close")
        source = _source_file(plan, event["evidence"])
        _require(source["capture_kind"] == "http_response_body", "split anchor is not valid archived HTTP content")
        splits.append({"event_id": f'{event["instrument"]}:split:{effective}',
            "instrument": event["instrument"], "event_type": "split", "record_date": record,
            "payable_after_close_date": payable, "effective_trading_date": effective,
            "new_shares_per_old_share": ratio, "source_hash": source["sha256"],
            "source_version": event["evidence"], "source_capture_kind": source["capture_kind"],
            "basis_evidence_grade": "primary_document_ratio_and_effective_date_contract",
            "reference_calendar_status": status, "archive_timestamp_utc": None, "announcement_at": None})
    _require(len({r["event_id"] for r in splits}) == len(splits), "duplicate split event")
    discrepancies = {(v["instrument"], v["ex_date"]): v for v in plan["declared_discrepancies"]}
    for symbol, source in plan["sources"].items():
        frame = source_rows[symbol]
        _require(list(frame.columns) == source["source_fields"], "source fields differ from contract")
        _require(len(frame) == source["expected_distribution_rows"], "source row count differs")
        is_soxx = symbol == "SOXX"
        version = "soxx_ishares_workbook_20261009" if is_soxx else "soxl_direxion_dom_table_20261009"
        identity = plan["source_versions"][version]
        dates = []
        for number, row in enumerate(frame.to_dict("records"), 1):
            ex, record, pay = map(_date, (row["Ex-Date" if is_soxx else "Ex Date"], row["Record Date"],
                                       row["Payable Date" if is_soxx else "Pay Date"]))
            _require(ex <= pay and ex <= record <= pay and ex <= as_of, "invalid dates or future economic event")
            dates.append(ex)
            if is_soxx:
                components = [_decimal(row[k]) for k in ISHARES_FIELDS[4:]]
                total = _decimal(row["Total Distribution"])
                _require(abs(total - sum(components)) <= _decimal(source["component_total_abs_tolerance_decimal"]),
                         "issuer total differs from component precision contract")
            else:
                components = [Decimal(0) if row[k] == "—" else _decimal(row[k]) for k in DIREXION_FIELDS[3:]]
                components.append(Decimal(0))  # this issuer schema has no return-of-capital column
                total = sum(components)
            multiplier = Decimal(1)
            if is_soxx:
                for split in splits:
                    if split["instrument"] == symbol and ex < split["effective_trading_date"]:
                        multiplier *= split["new_shares_per_old_share"]
            difference = discrepancies.get((symbol, ex))
            basis_version, grade = version, source["basis_evidence_grade"]
            archive_stamp, archive_digest, known_by = None, None, None
            if difference and symbol == "SOXL" and "evidence_integrity" in difference:
                basis_version = "soxl_direxion_distribution_archive_20141224"
                historical = plan["source_versions"][basis_version]
                archive_stamp, archive_digest = historical["archive_timestamp_utc"], historical["archive_cdx_sha1_base32"]
                known_by = historical["source_evidence_known_by_utc"]
                _require(total == _decimal(difference["issuer_original_amount_decimal"]), "2014 exact issuer amount differs")
                grade = "exact_archived_original_issuer_rate_after_ex_date_before_split"
            events.append({"event_id": f"{symbol}:cash:{ex}", "instrument": symbol, "event_type": "cash_distribution",
                "source_row": number, "source_capture_kind": identity["capture_kind"],
                "source_hash": identity["body_sha256"], "source_version": version, "basis_source_version": basis_version,
                "ex_date": ex, "record_date": record, "pay_date": pay, "source_cash_basis": source["cash_basis"],
                "source_total_decimal": str(total), "source_income_decimal": str(components[0]),
                "source_short_term_gain_decimal": str(components[1]), "source_long_term_gain_decimal": str(components[2]),
                "source_return_of_capital_decimal": str(components[3]),
                "original_share_multiplier_decimal": str(multiplier), "original_total_decimal": str(total * multiplier),
                "original_income_decimal": str(components[0] * multiplier),
                "original_short_term_gain_decimal": str(components[1] * multiplier),
                "original_long_term_gain_decimal": str(components[2] * multiplier),
                "original_return_of_capital_decimal": str(components[3] * multiplier),
                "return_of_capital_source_field_present": is_soxx,
                "basis_evidence_grade": grade, "reference_calendar_status": _calendar_status(ex, calendar),
                "record_reference_calendar_status": _calendar_status(record, calendar),
                "pay_reference_calendar_status": _calendar_status(pay, calendar),
                "payable_status_at_market_as_of": "unpaid_receivable_not_spendable" if pay > as_of else "scheduled_pay_date_passed_not_actual_account_receipt",
                "declared_discrepancy_id": f"{symbol}:{ex}" if difference else None,
                "archive_timestamp_utc": archive_stamp, "archive_cdx_sha1_base32": archive_digest,
                "source_evidence_known_by_utc": known_by, "announcement_at": None})
        _require(len(set(dates)) == len(dates), "duplicate source ex-date")
        _require(min(dates) == source["expected_ex_date_min"] and max(dates) == source["expected_ex_date_max"],
                 "source date extent differs")
    cash, split_frame = pd.DataFrame(events), pd.DataFrame(splits)
    positive = cash.original_total_decimal.map(Decimal) > 0
    counts = {}
    for symbol, source in plan["sources"].items():
        mask = cash.instrument.eq(symbol)
        pos, zero = int((positive & mask).sum()), int((~positive & mask).sum())
        _require(pos == source["expected_positive_rows"] and zero == source["expected_zero_rows"],
                 "positive/zero event counts differ")
        counts[symbol] = {"rows": int(mask.sum()), "positive_rows": pos, "zero_rows": zero,
            "outside_reference_range": int((mask & cash.reference_calendar_status.eq("outside_reference_range")).sum())}
    _require(cash.event_id.is_unique, "duplicate canonical event identity")
    audit = {"status": "usable_under_declared_source_contract", "cash_event_rows": len(cash),
        "split_event_rows": len(split_frame), "all_original_source_rows_retained": True, "by_instrument": counts,
        "native_calendar_rows": len(calendar), "native_calendar_start": str(calendar[0].date()),
        "native_calendar_end": str(calendar[-1].date()), "market_as_of": as_of,
        "unpaid_receivable_events": cash.loc[cash.payable_status_at_market_as_of.eq("unpaid_receivable_not_spendable"),
            ["event_id", "pay_date", "original_total_decimal"]].to_dict("records"),
        "announcement_timestamps_invented": False, "historical_vintage_certified": False,
        "account_ledger_run": False, "model_fits": 0, "strategy_replays": 0, "investment_return_certified": False}
    return cash, split_frame, audit


def compare_futu_cash_events(cash_events, rehabilitation, *, source_contract):
    """Record exact differences; never replace issuer Decimal amounts."""
    plan, rows = source_contract, []
    declared = {(v["instrument"], v["ex_date"]): v for v in plan["declared_discrepancies"]}
    for symbol, source in plan["sources"].items():
        vendor = rehabilitation[symbol].copy()
        _require("ex_div_date" in vendor and "per_cash_div" in vendor, "vendor event schema missing")
        positive = {}
        for record in vendor.to_dict("records"):
            value = record["per_cash_div"]
            if value is None or pd.isna(value):
                continue  # a split-only event, not a missing issuer cash row
            amount = _decimal(str(value))
            if amount == 0:
                continue
            date = _date(str(record["ex_div_date"])[:10])
            _require(date not in positive, "duplicate positive vendor event")
            positive[date] = amount
        issuer = cash_events[cash_events.instrument.eq(symbol)]
        issuer_positive = {r["ex_date"]: r for r in issuer.to_dict("records") if _decimal(r["original_total_decimal"]) > 0}
        tolerance = _decimal(plan["source_comparison_precision_contract"][symbol.lower() + "_undeclared_difference_max_abs_decimal"])
        for date in sorted(set(issuer_positive) | set(positive)):
            event, amount = issuer_positive.get(date), positive.get(date)
            exception = declared.get((symbol, date))
            _require(event is not None, "vendor positive event absent from complete issuer table")
            issuer_amount = _decimal(event["original_total_decimal"])
            if amount is None:
                _require(exception is not None and exception.get("futu_cash_event_present") is False,
                         "undeclared missing vendor cash event")
                _require(issuer_amount == _decimal(exception["expected_original_share_amount_decimal"]),
                         "declared missing event amount changed")
                difference, status = None, "declared_vendor_missing_event"
            else:
                difference = issuer_amount - amount
                if exception:
                    expected = exception.get("issuer_original_amount_decimal", exception.get("issuer_amount_decimal"))
                    if expected is not None:
                        _require(issuer_amount == _decimal(expected) and amount == _decimal(exception["futu_amount_decimal"]),
                                 "declared source conflict values changed")
                    status = "declared_source_difference"
                else:
                    _require(abs(difference) <= tolerance, "undeclared material issuer/vendor cash difference")
                    status = "exact_amount_match" if difference == 0 else "within_declared_display_precision"
            rows.append({"instrument": symbol, "ex_date": date, "issuer_original_total_decimal": str(issuer_amount),
                "futu_amount_decimal": str(amount) if amount is not None else None,
                "difference_decimal": str(difference) if difference is not None else None,
                "status": status, "issuer_amount_modified": False})
    frame = pd.DataFrame(rows)
    return frame, {"positive_issuer_events": len(frame), "status_counts": frame.status.value_counts().to_dict(),
        "all_issuer_amounts_preserved": True, "comparison_is_diagnostic_only": True,
        "exact_decimal_differences_retained": True}


def verify_historical_2014_source(html_bytes, cdx_bytes, contract):
    """Verify the exact archived issuer rate without assigning ex-ante timing."""
    from bs4 import BeautifulSoup

    _require(hashlib.sha256(html_bytes).hexdigest() == contract["body_sha256"], "historical HTML hash differs")
    digest = base64.b32encode(hashlib.sha1(html_bytes).digest()).decode()
    _require(digest == contract["cdx_sha1_base32"], "historical HTML differs from CDX digest")
    data = json.loads(cdx_bytes)
    _require(isinstance(data, list) and len(data) > 1, "CDX response missing")
    records = [dict(zip(data[0], r, strict=True)) for r in data[1:]]
    captured = pd.Timestamp(contract["capture_utc"]).strftime("%Y%m%d%H%M%S")
    matches = [r for r in records if r.get("original") == contract["original_url"] and r.get("timestamp") == captured]
    _require(len(matches) == 1 and matches[0].get("digest") == digest and matches[0].get("statuscode") == "200",
             "CDX original URL/timestamp/status/digest do not match")
    soup = BeautifulSoup(html_bytes, "html.parser")
    rows = [r for r in soup.find_all("tr") if "SOXL" in r.get_text()]
    _require(len(rows) == 1, "historical issuer SOXL row is ambiguous")
    values = [c.get_text(" ", strip=True) for c in rows[0].find_all(["td", "th"])]
    _require(len(values) == 5 and values[0] == "SOXL" and values[2] == "25459W458", "historical ticker/CUSIP schema differs")
    _require(_decimal(values[3].replace("$", "").replace(",", "")) == _decimal(contract["exact_amount_usd_decimal"])
             and _decimal(values[4]) == _decimal(contract["exact_original_share_rate_decimal"]),
             "historical issuer total/rate differs")
    table = rows[0].find_parent("table")
    header = [c.get_text(" ", strip=True) for c in table.find("tr").find_all(["td", "th"])]
    _require(header == ["Ticker", "Fund", "CUSIP", "Distribution Amount", "Distribution Per Share Rate"],
             "historical issuer column meanings differ")
    # The event schedule is in the list immediately preceding this income table.
    prior = table.find_previous("ul")
    schedule = prior.get_text(" ", strip=True) if prior else ""
    event_dates = {}
    for key, label in (("economic_ex_date", "Ex Date"), ("record_date", "Record Date"), ("pay_date", "Pay Date")):
        matches = re.findall(label.replace(" ", r"\s*") + r"\s*[-–:]?\s*(\d{1,2}/\d{1,2}/\d{4})", schedule, re.I)
        _require(len(matches) == 1, "historical event schedule missing or ambiguous")
        event_dates[key] = _date(matches[0])
    _require(event_dates["economic_ex_date"] <= event_dates["record_date"] <= event_dates["pay_date"],
             "historical event date order is invalid")
    return {"status": "exact_historical_issuer_rate_verified", "sha256": contract["body_sha256"],
        "cdx_sha1_base32": digest, "archive_timestamp_utc": contract["capture_utc"],
        **event_dates,
        "original_share_rate_decimal": values[4], "distribution_total_decimal": contract["exact_amount_usd_decimal"],
        "announcement_at": None,
        "known_by_is_after_ex_date": pd.Timestamp(contract["capture_utc"]).date().isoformat() > event_dates["economic_ex_date"],
        "ex_ante_availability_certified": False}

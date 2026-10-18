"""Date-only corporate-event admission for a conditional historical source study.

Inputs are existing Recorder objects and hash-verified raw-source metadata.  This
module neither reads prices nor certifies historical publication or absence of
events in real time.  Every event, including explicit zero cash, disables the
temporary increment in both arms without deleting the account session.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re

import pandas as pd


SYMBOLS = ("SOXL", "SOXX")
REHAB_FIELDS = (
    "ex_div_date", "split_base", "split_ert", "join_base", "join_ert", "split_ratio",
    "per_cash_div", "special_dividend", "bonus_base", "bonus_ert", "per_share_div_ratio",
    "transfer_base", "transfer_ert", "per_share_trans_ratio", "allot_base", "allot_ert",
    "allotment_ratio", "allotment_price", "add_base", "add_ert", "stk_spo_ratio",
    "stk_spo_price", "spin_off_base", "spin_off_ert", "spin_off_ratio",
    "forward_adj_factorA", "forward_adj_factorB", "backward_adj_factorA", "backward_adj_factorB",
)
_UNSUPPORTED = (
    "join_base", "join_ert", "special_dividend", "bonus_base", "bonus_ert",
    "per_share_div_ratio", "transfer_base", "transfer_ert", "per_share_trans_ratio",
    "allot_base", "allot_ert", "allotment_ratio", "allotment_price", "add_base",
    "add_ert", "stk_spo_ratio", "stk_spo_price", "spin_off_base", "spin_off_ert",
    "spin_off_ratio",
)


def _require(condition, message):
    if not condition:
        raise ValueError("needs_data: corporate event " + message)


def _date(value):
    _require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None,
             "date must be a local YYYY-MM-DD literal")
    stamp = pd.Timestamp(value)
    _require(pd.notna(stamp) and stamp.strftime("%Y-%m-%d") == value, "invalid date")
    return stamp


def _decimal(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("needs_data: corporate event invalid numeric field") from error
    _require(result.is_finite(), "nonfinite numeric field")
    return result


def audit_event_coverage(
    cash_events, split_events, rehab_by_symbol, *, sessions, source_contract,
    rehab_source_manifest, original_qualified_through="2026-09-25", as_of="2026-10-09",
):
    """Return session mask, audit, and event-membership differences, without labels.

    ``sessions`` must be the exact native account-session index, not generated
    weekdays.  The caller must verify Recorder identities and the manifest/raw
    rehab byte hashes before calling.  Missing schemas, dates or duplicate events
    fail closed; date-locatable unsupported/mismatched events remain unknown and
    disable that session.  An unresolved tail denies the entire extension.
    """
    original, cutoff = _date(original_qualified_through), _date(as_of)
    calendar = pd.DatetimeIndex(sessions)
    _require(len(calendar) > 0 and calendar.tz is None and not calendar.hasnans
             and not calendar.has_duplicates and calendar.is_monotonic_increasing
             and calendar.equals(calendar.normalize()), "requires exact sorted native session dates")
    _require(calendar[-1] <= cutoff and original <= cutoff, "session/cutoff chronology")
    _require(source_contract.get("market_as_of") == original_qualified_through,
             "original qualification date differs from source contract")
    _require(set(source_contract.get("sources", {})) == set(SYMBOLS)
             and set(rehab_by_symbol) == set(SYMBOLS), "instrument universe differs")
    manifest = rehab_source_manifest
    _require(manifest.get("schema_version") == "semis_minute_source_v1"
             and manifest.get("complete") is True and manifest.get("adjustment") == "NONE"
             and manifest.get("session") == "RTH", "same-snapshot manifest incomplete or incompatible")
    snapshot_start, snapshot_end = _date(manifest["start"]), _date(manifest["as_of"])
    _require(snapshot_start <= snapshot_end, "snapshot date interval")
    _require({"instrument", "event_id", "ex_date", "original_total_decimal"} <= set(cash_events)
             and {"instrument", "event_id", "effective_trading_date", "new_shares_per_old_share"}
             <= set(split_events), "issuer schema incomplete")
    _require(set(cash_events.instrument) == set(SYMBOLS)
             and set(split_events.instrument) <= set(SYMBOLS), "issuer instrument mismatch")
    _require(not cash_events.event_id.duplicated().any() and not split_events.event_id.duplicated().any(),
             "duplicate issuer event identity")

    issuer, vendor, unknown, differences = {}, {}, {}, []
    def issue(symbol, day, kind, status, unresolved=False):
        differences.append({"instrument": symbol, "event_date": str(day.date()),
                            "event_kind": kind, "status": status, "unresolved": bool(unresolved)})
        if unresolved:
            unknown.setdefault(day, set()).add(symbol + ":" + status)

    for symbol in SYMBOLS:
        records = cash_events.loc[cash_events.instrument.eq(symbol)]
        contract = source_contract["sources"][symbol]
        _require(len(records) == contract["expected_distribution_rows"], "issuer row count changed")
        dates = [_date(x) for x in records.ex_date]
        _require(len(set(dates)) == len(dates)
                 and min(dates) == _date(contract["expected_ex_date_min"])
                 and max(dates) == _date(contract["expected_ex_date_max"]), "issuer date coverage changed")
        for row, day in zip(records.to_dict("records"), dates):
            amount = _decimal(row["original_total_decimal"])
            _require(amount >= 0 and day <= original, "issuer amount/date outside original contract")
            issuer[(symbol, day, "cash")] = {"zero": amount == 0, "id": str(row["event_id"])}
    for row in split_events.to_dict("records"):
        day, ratio = _date(row["effective_trading_date"]), _decimal(row["new_shares_per_old_share"])
        _require(day <= original and ratio > 0 and ratio == ratio.to_integral_value(), "split date/ratio")
        key = (row["instrument"], day, "split")
        _require(key not in issuer, "duplicate issuer split date")
        issuer[key] = {"ratio": ratio, "id": str(row["event_id"])}

    receipt_bounds = {}
    for symbol in SYMBOLS:
        frame = rehab_by_symbol[symbol]
        _require(set(frame.columns) == set(REHAB_FIELDS), symbol + " rehab schema changed")
        entries = [x for x in manifest.get("rehab", []) if x.get("symbol") == symbol]
        _require(len(entries) == 1 and len(frame) == entries[0].get("rows"), symbol + " rehab binding/count")
        entry = entries[0]
        _require(re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) is not None
                 and entry.get("path") == symbol + "_rehab.parquet", symbol + " rehab identity missing")
        fetched = pd.Timestamp(entry.get("fetched_at"))
        _require(pd.notna(fetched) and fetched.tz is not None, symbol + " rehab receipt timestamp missing")
        receipt_bounds[symbol] = fetched.isoformat()
        dates = [_date(x) for x in frame.ex_div_date]
        _require(len(set(dates)) == len(dates) and all(x <= snapshot_end for x in dates),
                 symbol + " duplicate or post-snapshot event date")
        for row, day in zip(frame.to_dict("records"), dates):
            unsupported = [x for x in _UNSUPPORTED if pd.notna(row[x])]
            if unsupported:
                issue(symbol, day, "unknown", "unsupported_fields:" + ",".join(unsupported), True)
            found = False
            if pd.notna(row["per_cash_div"]):
                cash = _decimal(row["per_cash_div"])
                if cash < 0:
                    issue(symbol, day, "cash", "negative_vendor_cash", True)
                vendor[(symbol, day, "cash")] = {}
                found = True
            base, extra = row["split_base"], row["split_ert"]
            if pd.notna(base) or pd.notna(extra):
                if pd.isna(base) or pd.isna(extra):
                    issue(symbol, day, "split", "incomplete_exact_split_units", True)
                else:
                    base, extra = _decimal(base), _decimal(extra)
                    if base <= 0 or extra <= 0:
                        issue(symbol, day, "split", "invalid_exact_split_units", True)
                    else:
                        vendor[(symbol, day, "split")] = {"ratio": extra / base}
                found = True
            elif pd.notna(row["split_ratio"]) and _decimal(row["split_ratio"]) != 1:
                issue(symbol, day, "split", "split_ratio_without_exact_units", True)
                found = True
            if not found and not unsupported:
                issue(symbol, day, "unknown", "affine_only_or_unclassified_rehab_event", True)

    declared_missing = {
        (x["instrument"], _date(x["ex_date"]))
        for x in source_contract.get("declared_discrepancies", [])
        if x.get("futu_cash_event_present") is False
    }
    for key in sorted(set(issuer) | set(vendor)):
        symbol, day, kind = key
        left, right = issuer.get(key), vendor.get(key)
        if left is not None and right is not None:
            if kind == "split" and left["ratio"] != right["ratio"]:
                issue(symbol, day, kind, "issuer_vendor_split_ratio_conflict", True)
            continue
        if left is not None and kind == "cash" and left["zero"]:
            issue(symbol, day, kind, "issuer_explicit_zero_retained_no_vendor_cash_required")
        elif left is not None and kind == "cash" and (symbol, day) in declared_missing:
            issue(symbol, day, kind, "previously_declared_vendor_omission_issuer_event_retained")
        else:
            issue(symbol, day, kind, "vendor_event_absent_from_issuer" if left is None
                  else "issuer_event_absent_from_vendor", True)

    # A known date outside the native session set but inside its contiguous
    # range cannot be silently ignored: it may be an ambiguous effective-date
    # convention and would otherwise manufacture apparently event-free days.
    all_dates = {key[1] for key in set(issuer) | set(vendor)} | set(unknown)
    off_calendar = sorted(d for d in all_dates if calendar[0] <= d <= calendar[-1] and d not in calendar)
    tail_unknown = sorted(d for d in unknown if original < d <= cutoff)
    # This tests retrieval of the completed snapshot, not an event's first-public time.
    receipts_after_snapshot = all(pd.Timestamp(x).tz_convert("UTC").date() > snapshot_end.date()
                                  for x in receipt_bounds.values())
    tail_supported = (snapshot_end >= cutoff and snapshot_start <= calendar[0]
                      and receipts_after_snapshot and not tail_unknown and not off_calendar)
    qualified_through = cutoff if tail_supported else min(original, snapshot_end)
    rows = []
    for day in calendar:
        keys = [key for key in set(issuer) | set(vendor) if key[1] == day]
        reasons = []
        if off_calendar:
            reasons.append("unresolved_event_session_mapping")
        if day < snapshot_start or day > qualified_through:
            reasons.append("outside_qualified_snapshot_coverage")
        if day in unknown:
            reasons.append("unknown_or_conflicting_event")
        if keys:
            reasons.append("cash_or_split_event_including_explicit_zero")
        rows.append({"trade_date": day, "event_source_ready": not reasons,
                     "shared_no_increment": bool(reasons),
                     "issuer_event_count": sum(key in issuer for key in keys),
                     "rehab_event_count": sum(key in vendor for key in keys),
                     "unknown_event": day in unknown,
                     "reason": ";".join(reasons) if reasons else "conditional_snapshot_non_event",
                     "historical_PIT_qualified": False})
    mask = pd.DataFrame(rows).set_index("trade_date")
    difference_frame = pd.DataFrame(differences, columns=["instrument", "event_date", "event_kind", "status", "unresolved"])
    if len(difference_frame):
        difference_frame = difference_frame.sort_values(["event_date", "instrument", "status"]).reset_index(drop=True)
    audit = {
        "status": "conditional_snapshot_event_mask_ready" if not off_calendar else "needs_event_session_mapping",
        "qualification": "frozen_current_history_non_event_classification_not_historical_PIT",
        "event_tail_supported": bool(tail_supported), "qualified_through": str(qualified_through.date()),
        "original_qualified_through": original_qualified_through,
        "requested_as_of": as_of, "snapshot_as_of": str(snapshot_end.date()),
        "tail_basis": "Completed same-snapshot rehab with no unresolved additional tail events; issuer card itself is not rewritten or extended.",
        "rehab_fetched_at": receipt_bounds,
        "unknown_or_conflict_dates": [str(d.date()) for d in sorted(unknown)],
        "unknown_or_conflict_dates_in_requested_sessions": [str(d.date()) for d in sorted(unknown) if d in calendar],
        "off_native_session_dates": [str(d.date()) for d in off_calendar],
        "tail_unknown_dates": [str(d.date()) for d in tail_unknown],
        "session_rows": len(mask), "conditional_non_event_sessions": int(mask.event_source_ready.sum()),
        "shared_no_increment_sessions": int(mask.shared_no_increment.sum()),
        "issuer_cash_rows": len(cash_events), "issuer_explicit_zero_rows": sum(
            value.get("zero", False) for value in issuer.values()),
        "issuer_split_rows": len(split_events),
        "byte_identity_verified_by_caller_required": True,
        "historical_PIT_qualified": False, "independent_forward_qualified": False,
        "new_ROD_values_or_signs": 0, "new_returns_or_labels": 0,
        "event_day_policy": "Both arms no-increment; retain day, baseline holdings and corporate accounting.",
        "no_event_proof_limit": "Absence in a current historical table is conditional snapshot classification, not contemporaneous proof that no event was known or existed.",
    }
    return mask, audit, difference_frame

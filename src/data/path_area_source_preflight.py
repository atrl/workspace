"""Pure P1 qualifications for an unexecuted path-area/action bridge.

Callers own all authorization, byte identities, bounded Qlib reads, resident
object audits and Recorder persistence. Nothing here loads a source, calls a
model, constructs a path statistic, creates action labels or runs an account.
Unknown coverage is never evidence of an empty pool or permission for P2.
"""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from src.data.minute_allocation_features import label_maturity
from src.model.minute_pit_allocator import MinutePitAllocator, TARGETS
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    COMMON_FEATURES, AuxiliaryTrainingCandidate, PitAuxiliaryAllocationNet, _standardize_pair,
)

NY = "America/New_York"
CLOCKS = ("decision_at", "return_available_at", "label_available_at")
AUX_LABELS = ("synthetic_2x_crash_label", "synthetic_2x_upside_label", "training_weight")
PIT_FIELDS = ("$pit_available", "$pit_member", "$instrument_type")
MINUTE_FIELDS = ("$open", "$high", "$low", "$close", "$volume", "$price_valid")
SYMBOLS = ("SOXL", "SOXX")


def _require(condition, message):
    if not condition:
        raise ValueError("path P1: " + message)


def _calendar(values):
    dates = pd.DatetimeIndex(values)
    _require(len(dates) > 0 and dates.tz is None and not dates.hasnans
             and dates.is_unique and dates.is_monotonic_increasing
             and dates.equals(dates.normalize()), "invalid native daily calendar")
    return dates


def _native_index(frame, *, primary=False):
    _require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.MultiIndex)
             and frame.index.names == ["datetime", "instrument"]
             and frame.index.is_unique, "invalid native training index")
    dates = pd.DatetimeIndex(frame.index.get_level_values("datetime"))
    _require(dates.tz is None and not dates.hasnans and dates.equals(dates.normalize()),
             "training dates must be naive normalized sessions")
    if primary:
        _require(dates.is_unique and dates.is_monotonic_increasing
                 and set(frame.index.get_level_values("instrument")) == {"SOXL_SOXX_CASH"},
                 "primary index/order/instrument differs")
    _require(frame.columns.is_unique and isinstance(frame.columns, pd.MultiIndex)
             and frame.columns.nlevels == 2, "training columns must be unique two-level columns")
    return dates


def frame_digest(frame):
    """Hash schema, full index/order and values; do not coerce or reorder them."""
    schema = {"columns": list(frame.columns), "dtypes": list(map(str, frame.dtypes)),
              "index_names": list(frame.index.names),
              "index_dtypes": [str(frame.index.get_level_values(i).dtype)
                               for i in range(frame.index.nlevels)]}
    h = hashlib.sha256(json.dumps(schema, sort_keys=True, default=str).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="<u8").tobytes())
    return h.hexdigest()


def _ny_clock(series, name):
    _require(isinstance(series.dtype, pd.DatetimeTZDtype) and str(series.dt.tz) == NY,
             name + " must retain original America/New_York timezone")
    return series


def validate_training_frames(primary, auxiliary, *, base_features,
                             expected_primary_rows=3014, expected_auxiliary_rows=20741):
    """Qualify full prepared PG frames, preserving their original 85/20 schema."""
    dates = _native_index(primary, primary=True)
    aux_dates = _native_index(auxiliary)
    _require(primary.shape == (expected_primary_rows, 85), "primary shape differs")
    _require(auxiliary.shape == (expected_auxiliary_rows, 20), "auxiliary shape differs")
    _require(len(base_features) == 39 and len(set(base_features)) == 39
             and set(base_features).issubset(primary["feature"].columns), "39 primary inputs differ")
    _require(len(primary["feature"].columns) == 61
             and list(primary["common"].columns) == list(COMMON_FEATURES)
             and list(primary["label"].columns) == list(TARGETS)
             and set(primary["meta"].columns) == set(CLOCKS), "primary 61/16/5/3 groups differ")
    _require(list(auxiliary["feature"].columns) == list(COMMON_FEATURES)
             and list(auxiliary["label"].columns) == list(AUX_LABELS)
             and list(auxiliary["meta"].columns) == ["observations_end"], "auxiliary 16/3/1 groups differ")
    _require(len(dates) > 0 and dates.min() >= pd.Timestamp("2011-01-03")
             and dates.max() <= pd.Timestamp("2022-12-30"), "primary residency outside train interval")
    _require(not len(aux_dates) or aux_dates.max() <= pd.Timestamp("2022-12-30"),
             "auxiliary residency exceeds train interval")
    for name in CLOCKS:
        clock = _ny_clock(primary[("meta", name)], name)
        _require(clock.notna().all() and clock.lt(pd.Timestamp("2023-01-01", tz=NY)).all(),
                 "PG clock not fully matured before cutoff: " + name)
    _require(np.isfinite(primary["label"].to_numpy(float)).all(), "original PG targets not finite")
    observations = pd.to_datetime(auxiliary[("meta", "observations_end")])
    _require(observations.dt.tz is None and observations.notna().all()
             and observations.le(pd.Timestamp("2022-12-30")).all(), "auxiliary observation-end invalid")
    return {"status": "qualified_train_frames", "primary_shape": list(primary.shape),
            "auxiliary_shape": list(auxiliary.shape), "primary_sha256": frame_digest(primary),
            "auxiliary_sha256": frame_digest(auxiliary),
            "five_targets_sha256": frame_digest(primary.loc[:, [("label", x) for x in TARGETS]]),
            "full_index_sha256": frame_digest(primary.iloc[:, :0]),
            "resident_object_audit_required_from_workflow": True, "P2_admitted": False}


def fold_boundaries(calendar, years=(2018, 2019, 2020, 2021, 2022), purge_rows=20):
    dates = _calendar(calendar)
    _require(purge_rows == 20 and len(years) == len(set(years)), "fold/purge contract differs")
    result = {}
    for year in years:
        start = pd.Timestamp(year=int(year), month=1, day=1)
        pos = dates.searchsorted(start) - purge_rows - 1
        _require(0 <= pos < len(dates) and dates[-1] >= start, "calendar does not cover fold boundary")
        result[int(year)] = {"year": int(year), "cutoff": start.tz_localize(NY),
                             "train_end": dates[pos]}
    return result


def build_fold_views(primary, calendar, *, base_features,
                     years=(2018, 2019, 2020, 2021, 2022), purge_rows=20):
    """Reproduce original eligibility/scalers, without model initialization/forward."""
    dates = _native_index(primary, primary=True)
    cal = _calendar(calendar)
    before = frame_digest(primary)
    meta = primary["meta"].copy()
    meta.index = dates
    for name in CLOCKS:
        _ny_clock(meta[name], name)
        _require(meta[name].notna().all(), "PG clock missing")
    timing = meta.copy()
    for name, clock in (("trade_date", "decision_at"), ("exit_date", "return_available_at")):
        timing[name] = meta[clock].dt.tz_localize(None).dt.normalize()
    expected = label_maturity(timing[["trade_date", "exit_date"]], cal)
    for name in CLOCKS:
        _require(meta[name].equals(expected[name]), "native calendar/clock mismatch: " + name)
    timing["trade_year"] = timing.trade_date.dt.year
    timing["next_target_present"] = timing.exit_date.isin(timing.trade_date)
    folds = fold_boundaries(cal, years, purge_rows)
    maturity = primary[("meta", "label_available_at")]
    finite = np.isfinite(primary["label"].loc[:, list(TARGETS)].to_numpy(float)).all(axis=1)
    for year, fold in folds.items():
        eligible = (dates <= fold["train_end"]) & maturity.notna() & maturity.lt(fold["cutoff"]) & finite
        train = primary.loc[eligible].copy()
        fold.update({"train": train, "eligible": eligible.copy(), "training_rows": len(train),
                     "status": "qualified" if len(train) >= 100 else "needs_training_rows",
                     "prediction_index": primary.index[timing.trade_year.to_numpy() == year]})
        if not len(train):
            fold.update(primary_scaler=None, common_scaler=None, scaler_finite=False)
            continue
        # Exact original implementation, including median imputation and ddof=1.
        px, _, ps = _standardize_pair(train["feature"], train["feature"], base_features)
        cx, _, cs = _standardize_pair(train["common"], train["common"], COMMON_FEATURES)
        fold.update(primary_scaler={k: v.to_dict() for k, v in ps.items()},
                    common_scaler={k: v.to_dict() for k, v in cs.items()},
                    scaler_finite=bool(torch.isfinite(px).all() and torch.isfinite(cx).all()),
                    five_targets_sha256=frame_digest(train.loc[:, [("label", x) for x in TARGETS]]))
    _require(frame_digest(primary) == before, "primary source mutated")
    return {"timing": timing, "folds": folds, "primary_sha256": before, "P2_admitted": False}


def qualify_saved_model(model, fold, *, base_seed, base_features, expected_spec,
                        feature_masks, auxiliary_rows=None):
    """Inspect a future-authorized object only; never initialize or invoke it."""
    failures, unknowns = [], []
    def check(ok, name):
        if not ok:
            failures.append(name)
    check(type(model) is MinutePitAllocator, "original_model_class")
    check(getattr(model, "base_features", None) == list(base_features) and len(base_features) == 39,
          "original_39_feature_order")
    check(getattr(model, "extra_features", None) == [], "no_extra_features")
    check(getattr(model, "feature_masks", None) == feature_masks, "inactive_feature_masks_exact")
    check(getattr(model, "seed", None) == int(base_seed) + fold["year"], "seed_plus_trade_year")
    check(getattr(model, "cutoff", None) == fold["cutoff"], "cutoff")
    check(getattr(model, "train_end", None) == fold["train_end"], "purged_train_end")
    spec = getattr(model, "spec", None)
    check(spec is not None and asdict(spec) == expected_spec, "spec_exact")
    candidate = AuxiliaryTrainingCandidate(name="pit_only_aux_005", auxiliary_weight=.05,
                                            include_union_backfill=False)
    check(getattr(model, "candidate", None) == candidate, "PIT_only_candidate")
    expected_shapes = {"tail_encoder.0.weight": (8, 16), "tail_encoder.0.bias": (8,),
                       "tail_head.weight": (1, 8), "tail_head.bias": (1,),
                       "alpha_head.weight": (1, 8), "alpha_head.bias": (1,),
                       "allocation_head.0.weight": (16, 47), "allocation_head.0.bias": (16,),
                       "allocation_head.2.weight": (3, 16), "allocation_head.2.bias": (3,)}
    network = getattr(model, "network", None)
    if network is None:
        failures.append("missing_network")
    else:
        parameters = list(network.named_parameters())
        check(type(network) is PitAuxiliaryAllocationNet and list(dict(parameters)) == list(expected_shapes)
              and {k: tuple(v.shape) for k, v in parameters} == expected_shapes
              and sum(v.numel() for _, v in parameters) == 973, "973_parameter_shapes")
        check(all(bool(torch.isfinite(v.detach()).all()) for _, v in parameters), "finite_saved_parameters")
    for name in ("primary_scaler", "common_scaler"):
        check(fold.get(name) is not None and getattr(model, name, None) == fold.get(name), name + "_exact")
    check(fold.get("scaler_finite", False), "fold_input_support")
    train = fold["train"]
    check(len(train) >= 100 and fold.get("status") == "qualified", "original_training_minimum")
    audit = getattr(model, "audit", {})
    check(audit.get("training_rows") == len(train), "training_row_audit")
    if len(train):
        train_dates = train.index.get_level_values("datetime")
        check(audit.get("first_training_signal") == str(train_dates.min().date()), "first_signal_audit")
        check(audit.get("last_training_signal") == str(train_dates.max().date()), "last_signal_audit")
        check(audit.get("last_mature_label_at") == str(train[("meta", "label_available_at")].max()),
              "last_maturity_audit")
    check(audit.get("cutoff") == str(fold["cutoff"]) and audit.get("epochs") == 300
          and audit.get("checkpoint_policy") == "fixed_final_epoch", "fixed_training_audit")
    if auxiliary_rows is None:
        unknowns.append("fold_auxiliary_provenance_not_qualified")
    else:
        check(audit.get("auxiliary_rows") == auxiliary_rows, "fold_auxiliary_rows")
    return {"year": fold["year"], "base_seed": base_seed, "failures": failures,
            "unknowns": unknowns, "static_qualified": not failures and not unknowns,
            "status": "failed" if failures else "unknown" if unknowns else "qualified",
            "forward_calls": 0, "P2_admitted": False}


def _native_values(frame, fields):
    _require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.MultiIndex)
             and frame.index.nlevels == 2
             and set(frame.index.names) == {"datetime", "instrument"}
             and frame.index.is_unique and list(frame.columns) == list(fields), "native query schema differs")
    dates = pd.DatetimeIndex(frame.index.get_level_values("datetime"))
    _require(dates.tz is None and not dates.hasnans, "native timestamps must be naive New York")
    return frame.reorder_levels(["datetime", "instrument"]), dates


def qualify_pit_pool(flags, *, calendar, instruments, start, end, source_qualified,
                     instrument_spans=None):
    """Require observed, valid flags on every declared instrument/session cell."""
    cal = _calendar(calendar)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    _require(start <= end and cal[0] <= start and cal[-1] >= end, "membership calendar bounds differ")
    _require(len(instruments) == len(set(instruments)) and len(instruments) > 0, "invalid frozen PIT universe")
    if flags is None:
        return {"status": "unknown", "empty_pool_proven": False, "qualified_rows": 0,
                "unknowns": ["flag_query_unavailable"], "P2_admitted": False}
    frame, dates = _native_values(flags, PIT_FIELDS)
    _require(set(frame.index.get_level_values("instrument")).issubset(instruments)
             and (dates >= start).all() and (dates <= end).all(), "membership query outside allowlist")
    if instrument_spans is not None:
        _require(set(instrument_spans) == set(instruments), "instrument spans do not cover frozen universe")
    expected = []
    for instrument in instruments:
        lo, hi = (start, end) if instrument_spans is None else map(pd.Timestamp, instrument_spans[instrument])
        _require(lo <= hi, "invalid declared instrument lifespan")
        expected.extend((day, instrument) for day in cal[(cal >= max(start, lo)) & (cal <= min(end, hi))])
    grid = pd.MultiIndex.from_tuples(expected, names=["datetime", "instrument"])
    values = frame.reindex(grid)
    numeric = values.to_numpy(float)
    finite = np.isfinite(numeric).all(axis=1)
    valid = finite & np.isin(numeric[:, 0], [0, 1]) & np.isin(numeric[:, 1], [0, 1]) & np.isin(numeric[:, 2], [0, 1])
    admitted = valid & (numeric[:, 0] == 1) & (numeric[:, 1] == 1) & (numeric[:, 2] == 0)
    unknowns = []
    if not source_qualified:
        unknowns.append("source_identity_or_coverage_unqualified")
    if not len(grid):
        unknowns.append("no_expected_cells")
    if (~valid).any():
        unknowns.append("missing_or_invalid_flag_cells")
    empty = not unknowns and not admitted.any()
    return {"status": "unknown" if unknowns else "proven_empty" if empty else "nonempty_needs_raw_auxiliary",
            "empty_pool_proven": bool(empty), "qualified_rows": int(admitted.sum()),
            "expected_cells": len(grid), "missing_rows": int((~grid.isin(frame.index)).sum()),
            "nonfinite_cells": int((~finite).sum()), "invalid_cells": int((finite & ~valid).sum()),
            "type0_rows": int((valid & (numeric[:, 2] == 0)).sum()),
            "PIT_available_rows": int((valid & (numeric[:, 0] == 1)).sum()),
            "member_rows": int((valid & (numeric[:, 1] == 1)).sum()),
            "unknowns": unknowns, "historical_origin_lineage_certified": False, "P2_admitted": False}


def qualify_minute_coverage(bars, timing, schedule, *, source_clock_qualified):
    """Prefix mask and separate execution coverage; no returns or path statistic."""
    frame, dates = _native_values(bars, MINUTE_FIELDS)
    _require(set(frame.index.get_level_values("instrument")).issubset(SYMBOLS), "unexpected minute instrument")
    _require((dates <= pd.Timestamp("2022-12-30 16:00")).all(), "minute residency exceeds train bound")
    _require(timing.index.is_unique and {"trade_date", "exit_date", "decision_at"}.issubset(timing),
             "timing schema differs")
    _require(schedule.index.is_unique and {"open", "close"}.issubset(schedule), "schedule schema differs")
    session_dates = pd.DatetimeIndex(schedule.index)
    if session_dates.tz is not None:
        session_dates = session_dates.tz_localize(None)
    session_dates = session_dates.normalize()
    _require(session_dates.is_unique, "duplicate schedule session")
    schedule = schedule.copy()
    schedule.index = session_dates

    def bar_good(stamp, field):
        for symbol in SYMBOLS:
            key = (stamp, symbol)
            if key not in frame.index:
                return False
            row = frame.loc[key]
            value = row[field]
            if not (np.isfinite(value) and value > 0 and row["$price_valid"] == 1):
                return False
        return True

    rows = []
    for signal, row in timing.iterrows():
        day, exit_day = pd.Timestamp(row.trade_date), pd.Timestamp(row.exit_date)
        _require(day.tzinfo is None and exit_day.tzinfo is None and day == day.normalize()
                 and exit_day == exit_day.normalize(), "trade/exit dates must be native sessions")
        decision = pd.Timestamp(row.decision_at)
        _require(str(decision.tzinfo) == NY and decision == day.tz_localize(NY) + pd.Timedelta(hours=9, minutes=29),
                 "allocation clock differs")
        prefix_grid = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=15, freq="min")
        prefix_observed = bar_good(prefix_grid[0], "$open") and all(bar_good(t, "$close") for t in prefix_grid)
        schedule_known = day in schedule.index
        close_stamp = None
        if schedule_known:
            opening, closing = (pd.Timestamp(schedule.loc[day, name]) for name in ("open", "close"))
            _require(opening.tzinfo is not None and closing.tzinfo is not None, "schedule clock timezone missing")
            opening, closing = opening.tz_convert(NY), closing.tz_convert(NY)
            schedule_known = opening == day.tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)
            schedule_known &= closing.date() == day.date() and closing > opening
            if schedule_known:
                close_stamp = closing.tz_localize(None)
        prefix_known = bool(source_clock_qualified and schedule_known)
        rows.append({"signal_date": signal, "trade_date": day, "exit_date": exit_day,
                     "trade_year": day.year, "source_clock_qualified": bool(source_clock_qualified),
                     "schedule_known": bool(schedule_known), "prefix_observed_complete": bool(prefix_observed),
                     "prefix_mask": bool(prefix_known and prefix_observed),
                     "prefix_qualification_known": prefix_known,
                     "known_0945_close": bool(source_clock_qualified and bar_good(prefix_grid[-1], "$close")),
                     "fill_0947_open": bool(source_clock_qualified and bar_good(day + pd.Timedelta(hours=9, minutes=48), "$open")),
                     "recovery_1032_open": bool(source_clock_qualified and bar_good(day + pd.Timedelta(hours=10, minutes=33), "$open")),
                     "session_close": bool(source_clock_qualified and close_stamp is not None and bar_good(close_stamp, "$close")),
                     "session_close_at": close_stamp,
                     "next_0947_open": bool(source_clock_qualified and bar_good(exit_day + pd.Timedelta(hours=9, minutes=48), "$open")),
                     "next_target_present": bool(row.get("next_target_present", False)),
                     "actual_first_receipt_certified": False})
    return pd.DataFrame(rows).set_index("signal_date")


def corporate_metadata_qualification(cash_events, split_events, *, sessions, start, end, source_qualified):
    """Date/unit prerequisites only. No account, entitlement or price calculation."""
    cal = _calendar(sessions)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    unknowns, selected = [], []
    if not source_qualified:
        unknowns.append("corporate_source_identity_or_coverage_unqualified")
    required = {"cash": ("event_id", "instrument", "ex_date", "pay_date", "original_total_decimal"),
                "split": ("event_id", "instrument", "effective_trading_date", "new_shares_per_old_share")}
    residency = {}
    for kind, source in (("cash", cash_events), ("split", split_events)):
        if source is None:
            unknowns.append(kind + "_object_unavailable")
            continue
        _require(isinstance(source, pd.DataFrame) and set(required[kind]).issubset(source),
                 kind + " metadata schema differs")
        _require(source.index.is_unique and source.event_id.notna().all()
                 and not source.event_id.duplicated().any(), "duplicate/missing corporate event identity")
        date_col = "ex_date" if kind == "cash" else "effective_trading_date"
        all_dates = pd.to_datetime(source[date_col])
        _require(all_dates.dt.tz is None and all_dates.notna().all()
                 and all_dates.equals(all_dates.dt.normalize()), "invalid corporate event dates")
        residency[kind] = {"whole_object_rows": len(source), "post_train_rows": int((all_dates > end).sum())}
        for number in source.index[source.instrument.isin(SYMBOLS) & all_dates.between(start, end)]:
            row, day = source.loc[number], all_dates.loc[number]
            if day not in cal:
                unknowns.append("event_outside_native_sessions:" + str(row.event_id))
            item = {"event_id": str(row.event_id), "instrument": row.instrument, "date": day, "kind": kind}
            if kind == "split":
                try:
                    ratio = Decimal(str(row.new_shares_per_old_share))
                    valid = ratio.is_finite() and ratio > 0 and ratio == ratio.to_integral_value()
                except InvalidOperation:
                    valid = False
                if not valid:
                    unknowns.append("unsupported_split_units:" + str(row.event_id))
            else:
                pay = pd.Timestamp(row.pay_date)
                _require(pay.tzinfo is None and not pd.isna(pay) and pay == pay.normalize(), "invalid pay date")
                if pay < day:
                    unknowns.append("pay_before_ex:" + str(row.event_id))
                following = cal[cal > pay]
                item["available_date"] = following[0] if len(following) else pd.NaT
                if not len(following):
                    unknowns.append("next_pay_session_unknown:" + str(row.event_id))
                try:
                    rate = Decimal(str(row.original_total_decimal))
                    valid = rate.is_finite() and rate >= 0
                except InvalidOperation:
                    valid = False
                if not valid:
                    unknowns.append("invalid_cash_decimal:" + str(row.event_id))
            selected.append(item)
    events = pd.DataFrame(selected, columns=["event_id", "instrument", "date", "kind", "available_date"])
    if len(events):
        for key, group in events.groupby(["date", "instrument"]):
            if set(group.kind) == {"cash", "split"}:
                unknowns.append("ambiguous_same_day_split_dividend:" + str(key))
    return {"status": "unknown" if unknowns else "qualified_metadata_only", "unknowns": unknowns,
            "events": events, "residency": residency, "prices_or_accounts_examined": False,
            "actual_event_announcement_PIT_certified": False, "P2_admitted": False}

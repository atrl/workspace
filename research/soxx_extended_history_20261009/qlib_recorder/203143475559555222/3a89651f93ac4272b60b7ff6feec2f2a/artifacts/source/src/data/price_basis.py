"""Strict, data-only qualification of source candles before a new Qlib snapshot.

No rows are repaired, interpolated, deduplicated or discarded.  Source metadata
is an evidence declaration: the notebook must archive and verify the referenced
raw responses.  This module binds those identities to rows and validates the
declared convention; it cannot certify a vendor's historical price truth.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.qlib_native import (
    QlibProviderConfig,
    build_qlib_store_from_frame,
    init_qlib_provider,
    load_qlib_feature_frame,
)

FIELDS = ("open", "high", "low", "close", "volume")
INSTRUMENTS = ("SOXL", "SOXX")
SCHEMA_VERSION = "price_basis_v1"
PUBLISHED_BASIS = "published_futu_affine_v1"


class PriceBasisError(ValueError):
    """A source or snapshot failed a required data qualification gate."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def price_frame_sha256(frame: pd.DataFrame) -> str:
    """Content and schema identity, including row order; attrs are separate."""
    digest = hashlib.sha256()
    digest.update(_json({"columns": list(frame.columns), "dtypes": [str(x) for x in frame.dtypes]}).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _date(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        raise PriceBasisError("daily session dates must be timezone-naive")
    if pd.isna(result) or result != result.normalize():
        raise PriceBasisError("daily session date must be valid midnight")
    return result


def load_qlib_price_calendar(
    provider_uri: str | Path, start: str, as_of: str,
) -> dict[str, Any]:
    """Freeze dates from D.calendar, never a weekday/calendar approximation."""
    from qlib.data import D

    start_date, end_date = _date(start), _date(as_of)
    config = QlibProviderConfig(provider_uri=Path(provider_uri))
    init_qlib_provider(config)
    dates = pd.DatetimeIndex(D.calendar(start_time=start, end_time=as_of, freq="day"))
    if len(dates) == 0 or dates[0] != start_date or dates[-1] != end_date:
        raise PriceBasisError("reference Qlib calendar does not cover exact requested endpoints")
    calendar_file = config.resolved_provider_uri / "calendars/day.txt"
    payload = {
        "provider_uri": str(config.resolved_provider_uri),
        "source_api": "qlib.data.D.calendar", "freq": "day", "region": "us",
        "requested_start": start_date.date().isoformat(),
        "as_of": end_date.date().isoformat(),
        "dates": dates.strftime("%Y-%m-%d").tolist(),
        "calendar_file_sha256": hashlib.sha256(calendar_file.read_bytes()).hexdigest(),
    }
    return payload | {"calendar_sha256": _sha(payload)}


def _calendar_dates(calendar: Mapping[str, Any], as_of: str) -> pd.DatetimeIndex:
    payload = {key: value for key, value in calendar.items() if key != "calendar_sha256"}
    if calendar.get("source_api") != "qlib.data.D.calendar" or calendar.get("freq") != "day":
        raise PriceBasisError("a provenance-bearing native Qlib daily calendar is required")
    if _sha(payload) != calendar.get("calendar_sha256"):
        raise PriceBasisError("calendar content hash mismatch")
    dates = pd.DatetimeIndex([_date(x) for x in calendar["dates"]])
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise PriceBasisError("reference calendar must be nonempty, unique and increasing")
    if dates[-1] != _date(as_of) or calendar["as_of"] != as_of:
        raise PriceBasisError("reference calendar must end at exact as_of")
    if dates[0] != _date(calendar["requested_start"]):
        raise PriceBasisError("reference calendar start mismatch")
    return dates


def _session_date(value: Any) -> pd.Timestamp:
    try:
        return _date(value)
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def _row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("instrument", "datetime", *FIELDS, "basis", "original_adjustment", "affine_A", "affine_B",
            "split_price_factor", "rehab_event_count", "rehab_frame_sha256")
    return {key: row[key] for key in keys if key in row}


def normalize_futu_candles(raw: pd.DataFrame, source_identity: Mapping[str, Any]) -> pd.DataFrame:
    """Normalize SDK daily bars without fixing invalid observations.

    The identity must describe the original response artifacts, QFQ/RTH request,
    retrieval vintage, timezone, volume basis, and before/after rehabilitation
    hashes. Invalid values remain in the result and fail qualification.
    """
    required = {"code", "time_key", *FIELDS}
    if not required.issubset(raw.columns):
        raise PriceBasisError(f"Futu raw columns missing: {sorted(required - set(raw.columns))}")
    identity = json.loads(_json(dict(source_identity)))
    identity_hash = _sha(identity)
    source = raw.reset_index(drop=True)
    result = pd.DataFrame(index=source.index)
    result["instrument"] = source["code"].map({"US.SOXL": "SOXL", "US.SOXX": "SOXX"})
    result["datetime"] = pd.to_datetime(source["time_key"].map(_session_date))
    for field in FIELDS:
        result[field] = pd.to_numeric(source[field], errors="coerce").astype(float)
    result["source_identity_sha256"] = identity_hash
    result["source_row_number"] = np.arange(len(source), dtype=np.int64)
    result["source_code"] = source["code"].astype(str)
    result["source_time_key"] = source["time_key"].astype(str)
    result["source_raw_row_sha256"] = [_sha(row) for row in source.to_dict("records")]
    result["normalized_row_sha256"] = [_sha(_row_payload(row)) for row in result.to_dict("records")]
    result.attrs["source_identities"] = {identity_hash: identity}
    result.attrs["raw_frame_sha256"] = price_frame_sha256(raw)
    return result


def _identity_errors(identity: Mapping[str, Any], as_of: str, required_start: str) -> list[str]:
    errors = []
    published = identity.get("adjustment") == PUBLISHED_BASIS
    secondary = published and identity.get("source") == "Yahoo"
    expected = {
        "source": "Yahoo" if secondary else "Futu",
        "adjustment": PUBLISHED_BASIS if published else "QFQ", "session": "RTH", "extended_time": False,
        "timezone": "America/New_York", "volume_unit": "shares",
        "volume_basis": "split_adjusted_shares",
    }
    if published:
        expected["original_adjustment"] = "SPLIT_ONLY" if secondary else "NONE"
        expected["basis_adapter"] = PUBLISHED_BASIS
    for key, value in expected.items():
        if key not in identity or identity[key] != value:
            errors.append(f"{key}: expected {value!r}")
    if not identity.get("retrieval_id"):
        errors.append("missing retrieval_id")
    if _aware_timestamp(identity.get("retrieved_at_utc")) is None:
        errors.append("missing timezone-aware retrieved_at_utc")
    try:
        start, end = _date(identity.get("requested_start")), _date(identity.get("requested_end"))
        if start > end:
            errors.append("source request start exceeds end")
        if not secondary and start > _date(required_start):
            errors.append("source request starts after full required calendar")
        if end > _date(as_of) or (not secondary and end != _date(as_of)):
            errors.append("source request end differs from as_of")
    except (ValueError, TypeError):
        errors.append("invalid source request date bounds")
    artifacts = identity.get("raw_artifacts", [])
    if not artifacts or any(not x.get("name") or not _is_hash(x.get("sha256")) for x in artifacts):
        errors.append("missing named raw artifact SHA256 evidence")
    rehab = identity.get("rehabilitation", {})
    for instrument in INSTRUMENTS:
        record = rehab.get(instrument, {})
        before, after = record.get("before_sha256"), record.get("after_sha256")
        if not _is_hash(before) or before != after:
            errors.append(f"{instrument}: missing or changed rehabilitation evidence")
        before_time = _aware_timestamp(record.get("before_observed_at_utc"))
        after_time = _aware_timestamp(record.get("after_observed_at_utc"))
        if before_time is None or after_time is None or before_time > after_time:
            errors.append(f"{instrument}: missing or invalid rehabilitation observation times")
    return errors


def _aware_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        result = pd.Timestamp(value)
        return result if not pd.isna(result) and result.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_price_basis(
    frame: pd.DataFrame, calendar: Mapping[str, Any], as_of: str,
    required_model_start: str = "2011-01-03",
) -> dict[str, Any]:
    """Report every invalid/missing row; full required coverage is mandatory.

    Source identity mappings live in ``frame.attrs['source_identities']``. This
    supports independent per-row source identities without mislabelling a future
    secondary source as Futu. SDK QFQ and the separately declared published
    affine adapter are accepted; other bases need an additional source contract.
    """
    dates = _calendar_dates(calendar, as_of)
    model_start = _date(required_model_start)
    model_dates = dates[dates >= model_start]
    if model_dates.empty:
        raise PriceBasisError("required model interval is empty")
    required = {"instrument", "datetime", *FIELDS, "source_identity_sha256", "source_row_number",
                "source_raw_row_sha256", "normalized_row_sha256", "source_code", "source_time_key"}
    missing_columns = sorted(required - set(frame.columns))
    identities = frame.attrs.get("source_identities", {})
    identity_errors = {}
    for key, identity in identities.items():
        errors = _identity_errors(identity, as_of, dates[0].date().isoformat())
        if key != _sha(identity):
            errors.append("source identity hash mismatch")
        if errors:
            identity_errors[key] = errors
    declared_bases = sorted({str(identity.get("adjustment")) for identity in identities.values()})
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "qualification_pass": False,
        "as_of": as_of, "required_full_start": dates[0].date().isoformat(),
        "required_model_start": required_model_start,
        "expected_instruments": list(INSTRUMENTS), "required_full_sessions": len(dates),
        "required_model_sessions": len(model_dates), "row_count": len(frame),
        "frame_sha256": price_frame_sha256(frame), "calendar_sha256": calendar["calendar_sha256"],
        "source_identities_sha256": _sha(identities), "missing_columns": missing_columns,
        "source_identity_errors": identity_errors,
        "declared_price_bases": declared_bases,
        "basis_consistency_pass": len(declared_bases) == 1,
        "missing_dates_full": {}, "missing_dates_model": {}, "invalid_rows": [],
        "full_coverage_pass": False, "model_coverage_pass": False,
        "source_evidence_scope": "identity declarations and row binding; raw response archive verification belongs to the recorder notebook",
        "coverage_scope": "relative to the frozen native Qlib reference calendar; not independent exchange-calendar certification",
    }
    if missing_columns:
        for instrument in INSTRUMENTS:
            result["missing_dates_full"][instrument] = dates.strftime("%Y-%m-%d").tolist()
            result["missing_dates_model"][instrument] = model_dates.strftime("%Y-%m-%d").tolist()
        return result
    normalized_dates = pd.to_datetime(frame["datetime"].map(_session_date))
    duplicate = frame.duplicated(["instrument", "datetime"], keep=False).to_numpy()
    duplicate_lineage = frame.duplicated(["source_identity_sha256", "source_row_number"], keep=False).to_numpy()
    for position, row in enumerate(frame.to_dict("records")):
        issues = []
        date = normalized_dates.iloc[position]
        if row["instrument"] not in INSTRUMENTS:
            issues.append("unknown_instrument")
        if pd.isna(date) or date not in dates:
            issues.append("invalid_or_outside_calendar_date")
        if not pd.isna(date) and date > _date(as_of):
            issues.append("after_as_of")
        if duplicate[position]:
            issues.append("duplicate_instrument_date")
        values = pd.to_numeric(pd.Series({key: row[key] for key in FIELDS}), errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            issues.append("nonfinite_or_nonpositive_ohlcv")
        else:
            op, high, low, close, _ = values
            if not (low <= op <= high and low <= close <= high):
                issues.append("physical_ohlc_order")
            with np.errstate(over="ignore", under="ignore"):
                stored = values.astype(np.float32)
            if not np.isfinite(stored).all() or (stored <= 0).any():
                issues.append("not_positive_finite_in_qlib_float32")
        if row["source_identity_sha256"] not in identities:
            issues.append("unknown_source_identity")
        ordinal = row["source_row_number"]
        valid_ordinal = isinstance(ordinal, (int, np.integer)) and not isinstance(ordinal, bool) and ordinal >= 0
        identity = identities.get(row["source_identity_sha256"], {})
        source_symbol = (row["source_code"] if identity.get("source") == "Yahoo"
                         else {"US.SOXL": "SOXL", "US.SOXX": "SOXX"}.get(row["source_code"]))
        if (duplicate_lineage[position] or not _is_hash(row["source_raw_row_sha256"])
                or not valid_ordinal or source_symbol != row["instrument"]
                or _session_date(row["source_time_key"]) != date):
            issues.append("invalid_row_lineage")
        if row["normalized_row_sha256"] != _sha(_row_payload(row)):
            issues.append("normalized_row_hash_mismatch")
        if identity.get("adjustment") == PUBLISHED_BASIS:
            transform_keys = {"basis", "original_adjustment", "affine_A", "affine_B", "split_price_factor",
                              "rehab_event_count", "rehab_frame_sha256"}
            if not transform_keys.issubset(row):
                issues.append("missing_published_transform_lineage")
            elif (row["basis"] != PUBLISHED_BASIS
                    or row["original_adjustment"] != identity.get("original_adjustment")
                    or not _is_hash(row["rehab_frame_sha256"])):
                issues.append("invalid_published_transform_lineage")
        try:
            if not (_date(identity["requested_start"]) <= date <= _date(identity["requested_end"])):
                issues.append("row_outside_source_request")
        except (KeyError, TypeError, ValueError):
            issues.append("row_source_request_unverifiable")
        if issues:
            result["invalid_rows"].append({
                "row_position": position, "instrument": str(row["instrument"]),
                "datetime": str(date), "source_identity_sha256": row["source_identity_sha256"],
                "source_row_number": row["source_row_number"], "issues": issues,
            })
    for instrument in INSTRUMENTS:
        present = pd.DatetimeIndex(normalized_dates.loc[frame["instrument"] == instrument].dropna())
        result["missing_dates_full"][instrument] = dates.difference(present).strftime("%Y-%m-%d").tolist()
        result["missing_dates_model"][instrument] = model_dates.difference(present).strftime("%Y-%m-%d").tolist()
    result["full_coverage_pass"] = not any(result["missing_dates_full"].values())
    result["model_coverage_pass"] = not any(result["missing_dates_model"].values())
    result["qualification_pass"] = bool(
        result["full_coverage_pass"] and not result["invalid_rows"] and identities and not identity_errors
        and result["basis_consistency_pass"]
    )
    return result


def build_qualified_price_store(
    frame: pd.DataFrame, provider_uri: str | Path, as_of: str, source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a new provider only with a passed, content-bound qualification.

    The destination is reserved exclusively before invoking the existing native
    builder. Any failed readback leaves a marked failed new directory for audit;
    it is never accepted as a qualified snapshot and cannot be overwritten here.
    """
    calendar = source_manifest.get("calendar", {})
    supplied = source_manifest.get("qualification", {})
    if supplied.get("qualification_pass") is not True:
        raise PriceBasisError("explicit passed qualification is required; incomplete history cannot be shortened")
    audit = validate_price_basis(frame, calendar, as_of, supplied["required_model_start"])
    if not audit["qualification_pass"] or _sha(audit) != _sha(supplied):
        raise PriceBasisError("qualification no longer matches input rows, calendar or identities")
    identities = source_manifest.get("source_identities")
    if identities is None and "source_identity" in source_manifest:
        identity = source_manifest["source_identity"]
        identities = {_sha(identity): identity}
    if identities != frame.attrs.get("source_identities"):
        raise PriceBasisError("manifest source identities differ from qualified row identities")
    destination = Path(provider_uri).absolute()
    if destination.exists() or destination.is_symlink():
        raise PriceBasisError("destination must be new; existing providers are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    marker = destination / "price_basis_status.json"
    marker.write_text(_json({"status": "building", "qualification": audit}))
    try:
        manifest = build_qlib_store_from_frame(
            frame, provider_uri=destination, as_of=as_of,
            field_map={field: field for field in FIELDS}, source="qualified_source_candles",
        )
        loaded = load_qlib_feature_frame(
            QlibProviderConfig(provider_uri=destination), instruments=list(INSTRUMENTS),
            fields=[f"${field}" for field in FIELDS], start=audit["required_full_start"], end=as_of,
        ).sort_index()
        from qlib.data import D

        read_dates = pd.DatetimeIndex(D.calendar(freq="day")).strftime("%Y-%m-%d").tolist()
        if read_dates != calendar["dates"]:
            raise PriceBasisError("native readback calendar differs from qualified reference")
        listed = sorted(D.list_instruments(D.instruments("all"), as_list=True))
        if listed != sorted(INSTRUMENTS):
            raise PriceBasisError("native readback instrument universe mismatch")
        expected = frame.set_index(["instrument", "datetime"])[list(FIELDS)].sort_index().astype(np.float32)
        expected.columns = [f"${field}" for field in FIELDS]
        if not loaded.index.equals(expected.index) or not np.array_equal(loaded.to_numpy(), expected.to_numpy()):
            raise PriceBasisError("native D.features readback differs from source float32 storage values")
        raw_values = frame.set_index(["instrument", "datetime"])[list(FIELDS)].sort_index().to_numpy(float)
        quantization = np.max(np.abs(expected.to_numpy(float) - raw_values), axis=0)
        native_readback = {
            "status": "passed", "frame_sha256": price_frame_sha256(loaded),
            "rows": len(loaded), "storage_dtype": "float32",
            "max_absolute_quantization_by_field": dict(zip(FIELDS, quantization.tolist())),
            "source_api": ["qlib.data.D.calendar", "qlib.data.D.instruments", "qlib.data.D.list_instruments", "qlib.data.D.features"],
        }
        result = manifest | {
            "price_basis_schema": SCHEMA_VERSION, "qualification": audit,
            "reference_calendar": dict(calendar), "source_identities": identities,
            "source_manifest_sha256": _sha(source_manifest), "native_readback": native_readback,
        }
        (destination / "price_basis_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        marker.write_text(_json({"status": "qualified", "manifest_sha256": _sha(result)}))
        return result
    except Exception as error:
        marker.write_text(_json({"status": "failed", "error": str(error)}))
        raise

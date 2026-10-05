"""Diagnostic historical earnings surprises, conservatively aligned through Qlib.

The vendor response is a current historical snapshot, not an archived point-in-
time consensus database. No value from this module qualifies for promotion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.qlib_native import (
    QlibProviderConfig, build_qlib_store_from_frame, init_qlib_provider,
)

SYMBOLS = ("NVDA", "AMD", "INTC")
FEATURE_COLUMNS = tuple(f"earnings_surprise_{s.lower()}" for s in SYMBOLS)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _number(value: Any) -> float:
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return float("nan")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError("earnings values must be finite or explicitly missing")
    return parsed


def parse_earnings_events(payload: dict, symbol: str, *, as_of: str,
                          source_sha256: str, source_url: str) -> pd.DataFrame:
    """Validate unfiltered raw events. Future reports fail, never truncate.

    A missing actual or estimate invalidates state at its availability session;
    it is not silently dropped or carried through from an earlier quarter.
    """
    if symbol not in SYMBOLS or payload.get("symbol") != symbol:
        raise ValueError("earnings source symbol mismatch")
    raw = payload.get("quarterlyEarnings")
    if not isinstance(raw, list) or not raw:
        raise ValueError("earnings source has no quarterly events")
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError("source_sha256 must identify the raw source")
    if "apikey=" in source_url.lower() or "token=" in source_url.lower():
        raise ValueError("source URL must not contain credentials")
    rows = []
    for row in raw:
        report = pd.Timestamp(row["reportedDate"])
        fiscal = pd.Timestamp(row["fiscalDateEnding"])
        if pd.isna(report) or pd.isna(fiscal) or report != report.normalize() or fiscal != fiscal.normalize():
            raise ValueError("invalid earnings report/fiscal date")
        if report.tzinfo is not None or fiscal.tzinfo is not None:
            raise ValueError("earnings source dates must be timezone-naive dates")
        if report > pd.Timestamp(as_of).normalize():
            raise ValueError("raw reportedDate exceeds as_of")
        if fiscal > report:
            raise ValueError("fiscal period ends after its report")
        actual, estimate = _number(row.get("reportedEPS")), _number(row.get("estimatedEPS"))
        valid = bool(np.isfinite(actual) and np.isfinite(estimate))
        denom = abs(actual) + abs(estimate)
        surprise = (0.0 if denom == 0 else (actual - estimate) / denom) if valid else float("nan")
        rows.append({"symbol": symbol, "reported_date": report, "fiscal_date_ending": fiscal,
                     "reported_eps": actual, "estimated_eps": estimate,
                     "surprise": surprise, "valid": valid,
                     "report_time_vendor": row.get("reportTime", "unknown"),
                     "source_sha256": source_sha256, "source_url": source_url})
    events = pd.DataFrame(rows).sort_values(["reported_date", "fiscal_date_ending"])
    # Identical duplicates are benign; contradictory duplicates must fail closed.
    events = events.drop_duplicates()
    if events.reported_date.duplicated().any():
        raise ValueError("conflicting earnings events on the same report date")
    if events.fiscal_date_ending.duplicated().any():
        raise ValueError("multiple historical snapshots for one fiscal period")
    return events.reset_index(drop=True)


def align_earnings_surprise(events: pd.DataFrame, calendar: pd.DatetimeIndex
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """After any report, wait until a strictly later session has fully closed.

    Trading then occurs at the next session's open. Weekend reports therefore
    wait through Monday close as well. No future values or backwards fill.
    """
    calendar = pd.DatetimeIndex(calendar, name="datetime")
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Qlib calendar must be nonempty, ordered and unique")
    if calendar.tz is not None or not calendar.equals(calendar.normalize()):
        raise ValueError("Qlib daily calendar must contain timezone-naive dates")
    if set(events.symbol.unique()) != set(SYMBOLS):
        raise ValueError("all three declared source companies are required")
    if events.duplicated(["symbol", "reported_date"]).any():
        raise ValueError("duplicate symbol/report date")
    result = pd.DataFrame(index=calendar, columns=FEATURE_COLUMNS, dtype=float)
    audit = []
    for symbol in SYMBOLS:
        part = events.loc[events.symbol.eq(symbol)].sort_values("reported_date")
        positions = calendar.searchsorted(pd.DatetimeIndex(part.reported_date), side="right")
        values = np.full(len(calendar), np.nan)
        for i, (_, event) in enumerate(part.iterrows()):
            pos = int(positions[i])
            end = int(positions[i + 1]) if i + 1 < len(positions) else len(calendar)
            if pos < len(calendar):
                values[pos:end] = float(event.surprise) if bool(event.valid) else np.nan
            row = event.to_dict()
            row["availability_session"] = str(calendar[pos].date()) if pos < len(calendar) else None
            row["availability_at"] = (str(calendar[pos].date()) + "T17:00:00 America/New_York") if pos < len(calendar) else None
            row["earliest_trade_session"] = str(calendar[pos + 1].date()) if pos + 1 < len(calendar) else None
            row["before_reference_calendar"] = bool(pd.Timestamp(event.reported_date) < calendar[0])
            row["within_reference_calendar"] = bool(pos < len(calendar))
            audit.append(row)
        result[f"earnings_surprise_{symbol.lower()}"] = values
    return result, pd.DataFrame(audit)


def build_earnings_surprise_provider(snapshot_dir: Path, provider_uri: Path,
                                      reference_provider: Path, *,
                                      start: str = "2011-01-03", end: str = "2026-09-25",
                                      as_of: str = "2026-10-05") -> dict:
    """Build a separate feature provider and compare every cell via native D."""
    from qlib.data import D
    snapshot_dir, provider_uri = Path(snapshot_dir).resolve(), Path(provider_uri).resolve()
    reference_provider = Path(reference_provider).resolve()
    if provider_uri == reference_provider:
        raise ValueError("feature provider must not overwrite the reference provider")
    if provider_uri.exists() and any(provider_uri.iterdir()):
        raise ValueError("feature provider already exists; refuse to mutate evidence")
    snap_manifest = json.loads((snapshot_dir / "snapshot_manifest.json").read_text())
    if snap_manifest["as_of"] != as_of or snap_manifest["symbols"] != list(SYMBOLS):
        raise ValueError("snapshot declaration does not match requested build")
    parts = []
    for symbol in SYMBOLS:
        meta = snap_manifest["sources"][symbol]
        path = snapshot_dir / meta["file"]
        digest = sha256_file(path)
        if digest != meta["sha256"]:
            raise ValueError("raw snapshot hash mismatch")
        parts.append(parse_earnings_events(json.loads(path.read_text()), symbol,
                                          as_of=as_of, source_sha256=digest, source_url=meta["source_url"]))
    events = pd.concat(parts, ignore_index=True)
    init_qlib_provider(QlibProviderConfig(provider_uri=reference_provider, kernels=1))
    reference_calendar = pd.DatetimeIndex(D.calendar(freq="day"), name="datetime")
    if pd.Timestamp(end) > reference_calendar[-1] or pd.Timestamp(end) > pd.Timestamp(as_of):
        raise ValueError("requested end exceeds Qlib calendar/as_of")
    if pd.Timestamp(start) < reference_calendar[0]:
        raise ValueError("requested start precedes reference Qlib calendar")
    daily, audit = align_earnings_surprise(events, reference_calendar)
    daily = daily.loc[start:end]
    if daily.empty or not daily.index.isin(reference_calendar).all():
        raise ValueError("empty or noncanonical feature window")
    frame = daily.reset_index()
    frame["instrument"] = "SOXX"
    manifest = build_qlib_store_from_frame(
        frame, provider_uri=provider_uri, market="all", region="us", as_of=as_of,
        field_map={"$" + column: column for column in FEATURE_COLUMNS},
        source=str(snapshot_dir / "snapshot_manifest.json"))
    init_qlib_provider(QlibProviderConfig(provider_uri=provider_uri, kernels=1))
    native = D.features(["SOXX"], ["$" + c for c in FEATURE_COLUMNS],
                        start_time=start, end_time=end, freq="day").xs("SOXX", level="instrument")
    native.columns = FEATURE_COLUMNS
    native = native.reindex(daily.index)
    expected = daily.astype("float32")
    if not np.array_equal(native.to_numpy(), expected.to_numpy(), equal_nan=True):
        raise ValueError("Qlib feature readback does not match aligned events")
    repo_root = provider_uri.parents[2]
    relative = lambda p: str(Path(p).relative_to(repo_root))
    audit.to_csv(snapshot_dir / "events_availability.csv", index=False)
    event_digest = sha256_file(snapshot_dir / "events_availability.csv")
    manifest.update({
        "provider_uri": relative(provider_uri), "source": relative(snapshot_dir / "snapshot_manifest.json"),
        "reference_provider": relative(reference_provider),
        "reference_manifest_sha256": sha256_file(reference_provider / "qlib_store_manifest.json"),
        "reference_calendar_sha256": sha256_file(reference_provider / "calendars/day.txt"),
        "snapshot_manifest_sha256": sha256_file(snapshot_dir / "snapshot_manifest.json"),
        "event_audit": relative(snapshot_dir / "events_availability.csv"), "event_audit_sha256": event_digest,
        "snapshot_dir": relative(snapshot_dir), "diagnostic_only": True, "no_strict_pit": True,
        "no_promotion": True, "signal_source_symbols": list(SYMBOLS),
        "hypothesis_scope": "Three fixed company signals, not a complete historical sector basket",
        "feature_formula": "(reportedEPS-estimatedEPS)/(abs(reportedEPS)+abs(estimatedEPS)); both zero -> 0",
        "availability_rule": "first Qlib session strictly after reportedDate, after full close; trade next session open",
        "missing_rule": "Missing actual or estimate invalidates state until next valid event; never forward-fill through missing reports",
        "pit_limitation": "Current vendor historical snapshot lacks archived consensus vintages and historical correction timestamps; conservative release lag does not repair this limitation",
        "source_limitations": snap_manifest.get("limitations", []),
        "historical_source_quality": snap_manifest.get("historical_source_quality", {}),
        "readback": {"status": "exact_float32_equal_with_nan", "kernels": 1, "rows": len(daily),
                     "missing_by_feature": {c: int(daily[c].isna().sum()) for c in FEATURE_COLUMNS}},
        "event_counts": {s: int(events.symbol.eq(s).sum()) for s in SYMBOLS},
    })
    manifest["feature_sha256"] = {str(p.relative_to(provider_uri)): sha256_file(p) for p in sorted((provider_uri / "features").rglob("*.bin"))}
    manifest["provider_metadata_sha256"] = {name: sha256_file(provider_uri / name) for name in ("calendars/day.txt", "instruments/all.txt")}
    manifest["source_files_sha256"] = {relative(snapshot_dir / meta["file"]): meta["sha256"] for meta in snap_manifest["sources"].values()}
    manifest["source_files_sha256"].update({relative(snapshot_dir / "snapshot_manifest.json"): manifest["snapshot_manifest_sha256"], relative(snapshot_dir / "events_availability.csv"): event_digest})
    _dump(provider_uri / "qlib_store_manifest.json", manifest)
    return manifest


def load_earnings_surprise_features(provider_uri: Path, *, start: str, end: str
                                    ) -> tuple[pd.DataFrame, dict]:
    """Read only the declared separate provider through native Qlib APIs."""
    from qlib.data import D
    provider_uri = Path(provider_uri).resolve()
    manifest = json.loads((provider_uri / "qlib_store_manifest.json").read_text())
    if pd.Timestamp(start) < pd.Timestamp(manifest["calendar_start"]) or pd.Timestamp(end) > pd.Timestamp(manifest["calendar_end"]):
        raise ValueError("requested features exceed registered provider coverage")
    repo_root = provider_uri.parents[2]
    for name, digest in manifest["source_files_sha256"].items():
        if sha256_file(repo_root / name) != digest:
            raise ValueError("earnings source hash mismatch")
    for name, digest in manifest["provider_metadata_sha256"].items():
        if sha256_file(provider_uri / name) != digest:
            raise ValueError("earnings provider metadata hash mismatch")
    for name, digest in manifest["feature_sha256"].items():
        if sha256_file(provider_uri / name) != digest:
            raise ValueError("earnings feature provider hash mismatch")
    init_qlib_provider(QlibProviderConfig(provider_uri=provider_uri, kernels=1))
    frame = D.features(["SOXX"], ["$" + c for c in FEATURE_COLUMNS], start_time=start,
                       end_time=end, freq="day").xs("SOXX", level="instrument")
    frame.columns = FEATURE_COLUMNS
    return frame.sort_index(), manifest

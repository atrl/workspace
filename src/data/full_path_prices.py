"""Frozen NAS point marks with separate decision, fill and fee price clocks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.minute_price_basis import normalize_none_history, published_intraday_basis
from src.data.qlib_native import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame


def _source_frames(source, receipt, *, as_of, workspace_root):
    if receipt.get("as_of") != as_of:
        raise ValueError("blocked: source as_of differs")
    frames, events = {}, {}
    if receipt.get("schema_version") == "canonical_none_refresh_v1":
        if workspace_root is None:
            raise ValueError("blocked: canonical source requires explicit workspace_root")
        if receipt.get("source_adjustment") != "NONE" or receipt.get("session") != "RTH":
            raise ValueError("blocked: canonical source basis/session differs")
        root = Path(workspace_root).resolve()
        relative = Path(receipt["raw_cache"])
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
            raise ValueError("blocked: canonical cache path escapes workspace")
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["raw_sha256"]:
            raise ValueError("blocked: canonical raw source bytes changed")
        raw = pd.read_parquet(path)
        required = {"symbol", "timestamp", "date", "adjustment_type", "interval", "bar_start_at", "is_final", "price_valid"}
        if required - set(raw) or not raw.adjustment_type.eq("NONE").all() or not raw.interval.eq("1m").all():
            raise ValueError("blocked: canonical minute schema/basis differs")
        if raw.duplicated(["symbol", "timestamp"]).any():
            raise ValueError("blocked: duplicate canonical minutes")
        for symbol in ("SOXL", "SOXX"):
            frames[symbol] = raw.loc[raw.symbol.eq(symbol)].copy()
            event_path = source.parent / (symbol + "_rehab.parquet")
            if hashlib.sha256(event_path.read_bytes()).hexdigest() != receipt["rehab_sha256"][symbol]:
                raise ValueError("blocked: canonical corporate-action bytes changed")
            events[symbol] = pd.read_parquet(event_path)
    elif receipt.get("schema_version") == "semis_minute_source_v1":
        if not receipt.get("complete") or receipt.get("adjustment") != "NONE" or receipt.get("session") != "RTH":
            raise ValueError("blocked: source completeness/basis/session differs")
        for symbol in ("SOXL", "SOXX"):
            entry = next(e for e in receipt["entries"] if e["symbol"] == symbol)
            rehab = next(e for e in receipt["rehab"] if e["symbol"] == symbol)
            for item in (entry, rehab):
                path = source.parent / item["path"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError("blocked: observed source/event bytes changed")
            frames[symbol] = normalize_none_history(pd.read_parquet(source.parent / entry["path"]), symbol=symbol,
                interval="1m", fetched_at=entry["fetched_at"])
            events[symbol] = pd.read_parquet(source.parent / rehab["path"])
    else:
        raise ValueError("blocked: unsupported minute source schema")
    return frames, events


def build_full_path_prices(*, source_manifest, source_sha256, provider_uri, start, as_of, workspace_root=None):
    source = Path(source_manifest)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("blocked: source receipt differs from frozen path plan")
    receipt = json.loads(source.read_text())
    frames, events = _source_frames(source, receipt, as_of=as_of, workspace_root=workspace_root)
    import exchange_calendars as xc
    schedule = xc.get_calendar("XNYS", start=start, end=as_of).schedule
    expected_calendar = {str(day.date()) for day in schedule.index}
    closing = {str(day.date()): pd.Timestamp(row.close).tz_convert("America/New_York") for day, row in schedule.iterrows()}
    pieces = []
    for symbol, raw in frames.items():
        if raw.date.gt(as_of).any():
            raise ValueError("blocked: source extends beyond as_of")
        if not isinstance(raw.timestamp.dtype, pd.DatetimeTZDtype):
            raise ValueError("blocked: timezone-aware minute timestamps required")
        if not raw.date.eq(raw.timestamp.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")).all():
            raise ValueError("blocked: source date and local timestamp differ")
        raw = raw.loc[raw.date.ge(start)].copy()
        if set(raw.date) != expected_calendar:
            raise ValueError("needs_data: observed source is missing a real trading session")
        clock = raw.timestamp.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
        actual_close = raw.timestamp.eq(raw.date.map(closing))
        chosen = raw.loc[clock.isin(["09:45", "09:48", "10:33"]) | actual_close].copy()
        selected_clock = chosen.timestamp.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
        chosen["slot"] = np.select([selected_clock.eq("09:45"), selected_clock.eq("09:48"), selected_clock.eq("10:33")],
            ["known", "open", "recovery"], default="close")
        if (not chosen.groupby("date").size().reindex(sorted(expected_calendar), fill_value=0).eq(4).all()
                or not chosen.price_valid.all() or not chosen.is_final.fillna(False).all()):
            raise ValueError("needs_data: known/fill/session-close mark missing, invalid or unfinished")
        pieces.append(chosen)
    points, basis = published_intraday_basis(pd.concat(pieces, ignore_index=True), events, as_of=as_of)
    # Point marks are actual selected opens/closes, not synthetic OHLC bars.
    is_open = points.slot.isin(["open", "recovery"])
    points["execution_price"] = np.where(is_open, points.open, points.close)
    points["raw_execution_price"] = np.where(is_open, points.raw_open, points.raw_close)
    points["datetime"] = points.timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None).where(~is_open,
        points.bar_start_at.dt.tz_convert("America/New_York").dt.tz_localize(None))
    points["instrument"] = points.symbol
    fields = ["open", "high", "low", "close", "volume", "execution_price", "raw_execution_price"]
    for field in ("open", "high", "low", "close"):
        points[field] = points.execution_price
    manifest = build_qlib_store_from_frame(points, provider_uri=provider_uri, freq="1min", as_of=as_of,
        field_map={field: field for field in fields}, source="NAS observed decision/fill/session-close point marks")
    manifest.update(source_manifest_sha256=source_sha256, source_schema=receipt["schema_version"], price_basis=basis,
        calendar_role="sparse_observed_events_not_complete_minute_bars", decision_information="09:45 close available09:46",
        fill_clock="09:47 actual opening; no current fill price input to policy", known_pressure_day_retained=True,
        live_receipt_fill_dividend_parity_qualified=False)
    (Path(provider_uri) / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    quotes = load_qlib_feature_frame(QlibProviderConfig(Path(provider_uri), freq="1min"), instruments=["SOXL", "SOXX"],
        fields=["$execution_price", "$raw_execution_price"], start=start, end=as_of + " 16:00:00")
    return quotes, manifest

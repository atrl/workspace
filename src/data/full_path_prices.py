"""Frozen NAS point marks with separate decision, fill and fee price clocks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.minute_price_basis import normalize_none_history, published_intraday_basis
from src.data.qlib_native import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame


def build_full_path_prices(*, source_manifest, source_sha256, provider_uri, start, as_of):
    source = Path(source_manifest)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("blocked: source receipt differs from frozen path plan")
    receipt = json.loads(source.read_text())
    if not receipt["complete"] or receipt["adjustment"] != "NONE" or receipt["as_of"] != as_of:
        raise ValueError("blocked: source completeness/basis/as_of differs")
    import exchange_calendars as xc
    expected_calendar = {str(day.date()) for day in xc.get_calendar("XNYS", start=start, end=as_of).sessions}
    pieces, events, expected_days = [], {}, None
    for symbol in ("SOXL", "SOXX"):
        entry = next(e for e in receipt["entries"] if e["symbol"] == symbol)
        rehab = next(e for e in receipt["rehab"] if e["symbol"] == symbol)
        for item in (entry, rehab):
            path = source.parent / item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("blocked: observed source/event bytes changed")
        raw = normalize_none_history(pd.read_parquet(source.parent / entry["path"]), symbol=symbol,
                                     interval="1m", fetched_at=entry["fetched_at"])
        if raw.date.gt(as_of).any():
            raise ValueError("blocked: source extends beyond as_of")
        raw = raw.loc[raw.date.ge(start)].copy()
        days = set(raw.date)
        if days != expected_calendar:
            raise ValueError("needs_data: observed source is missing a real trading session")
        if expected_days is not None and days != expected_days:
            raise ValueError("needs_data: instrument calendars differ")
        expected_days = days
        clock = raw.timestamp.dt.strftime("%H:%M")
        last = raw.groupby("date").timestamp.transform("max")
        chosen = raw.loc[clock.isin(["09:45", "09:48", "10:33"]) | raw.timestamp.eq(last)].copy()
        chosen["slot"] = np.select([chosen.timestamp.dt.strftime("%H:%M").eq("09:45"),
            chosen.timestamp.dt.strftime("%H:%M").eq("09:48"), chosen.timestamp.dt.strftime("%H:%M").eq("10:33")],
            ["known", "open", "recovery"], default="close")
        if not chosen.groupby("date").size().eq(4).all() or not chosen.price_valid.all():
            raise ValueError("needs_data: known/fill/session-close mark missing or invalid")
        pieces.append(chosen)
        events[symbol] = pd.read_parquet(source.parent / rehab["path"])
    points, basis = published_intraday_basis(pd.concat(pieces, ignore_index=True), events, as_of=as_of)
    # Point marks are actual selected opens/closes, not synthetic OHLC bars.
    is_open = points.slot.isin(["open", "recovery"])
    points["execution_price"] = np.where(is_open, points.open, points.close)
    points["raw_execution_price"] = np.where(is_open, points.raw_open, points.raw_close)
    points["datetime"] = points.timestamp.dt.tz_localize(None).where(~is_open,
        points.bar_start_at.dt.tz_localize(None))
    points["instrument"] = points.symbol
    fields = ["open", "high", "low", "close", "volume", "execution_price", "raw_execution_price"]
    for field in ("open", "high", "low", "close"):
        points[field] = points.execution_price
    manifest = build_qlib_store_from_frame(points, provider_uri=provider_uri, freq="1min", as_of=as_of,
        field_map={field: field for field in fields}, source="NAS observed decision/fill/session-close point marks")
    manifest.update(source_manifest_sha256=source_sha256, price_basis=basis,
        calendar_role="sparse_observed_events_not_complete_minute_bars", decision_information="09:45 close available09:46",
        fill_clock="09:47 actual opening; no current fill price input to policy", known_pressure_day_retained=True,
        live_receipt_fill_dividend_parity_qualified=False)
    (Path(provider_uri) / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    quotes = load_qlib_feature_frame(QlibProviderConfig(Path(provider_uri), freq="1min"), instruments=["SOXL", "SOXX"],
        fields=["$execution_price", "$raw_execution_price"], start=start, end=as_of + " 16:00:00")
    return quotes, manifest

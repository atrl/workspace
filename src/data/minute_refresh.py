"""Core minute refresh from unadjusted observations, then published research basis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.intraday import fetch_futu_intraday_bars, merge_intraday_bars, read_intraday_cache, resample_intraday_bars
from src.data.intraday_quality import audit_minute_coverage, plan_minute_repairs
from src.data.minute_intake import atomic_json, atomic_parquet
from src.data.minute_price_basis import published_intraday_basis
from src.data.qlib_native import build_qlib_store_from_frame


def publish_core_stores(raw: pd.DataFrame, rehab: dict[str, pd.DataFrame], *, root: Path,
                        as_of: str, source: str) -> dict:
    published, basis = published_intraday_basis(raw, rehab, as_of=as_of)
    atomic_parquet(raw, root / "data/processed/intraday/soxl_soxx_none_1m.parquet")
    # Raw OHLC/amounts and receipt clocks remain in the NAS-owned NONE plane;
    # the slim compatibility cache and native stores are disposable consumers.
    columns = ["provider", "symbol", "interval", "timestamp", "date", "session",
               "open", "high", "low", "close", "volume", "turnover", "vwap",
               "source_tz", "adjusted", "fetched_at", "adjustment_type", "price_valid",
               "is_final", "affine_A", "affine_B", "split_price_factor"]
    atomic_parquet(published[columns], root / "data/processed/intraday/soxl_soxx_1m.parquet")
    manifests = {}
    for freq in ("1min", "5min", "15min"):
        if freq == "1min":
            bars = published.rename(columns={"symbol": "instrument", "timestamp": "datetime"}).copy()
            fields = ["open", "high", "low", "close", "volume", "turnover", "vwap", "price_valid"]
        else:
            bars = resample_intraday_bars(published, freq=freq, symbols=["SOXL", "SOXX"])
            fields = ["open", "high", "low", "close", "volume", "turnover", "bar_index", "bar_count",
                      "observed_1m_count", "expected_1m_count", "is_complete"]
        manifest = build_qlib_store_from_frame(bars, provider_uri=root / "data/qlib" / ("soxl_intraday_" + freq),
            region="us", freq=freq, as_of=as_of, field_map={"$" + name: name for name in fields}, source=source)
        manifest.update(price_basis=basis, timestamp_convention="right_edge", source_adjustment="NONE",
            research_adjustment="PUBLISHED_FUTU_AFFINE", malformed_source_rows_quarantined=True)
        atomic_json(manifest, Path(manifest["provider_uri"]) / "qlib_store_manifest.json")
        manifests[freq] = manifest
    return {"manifests": manifests, "basis_audit": basis, "cache_rows": len(published),
            "cache_end": str(published.timestamp.max())}


def refresh_published_core(*, root: Path, session_date: str, start_session_date: str | None,
                           host: str, port: int) -> dict:
    """Repair closed historical gaps and refresh the tail, independently of orders."""
    import exchange_calendars as xc
    from futu import OpenQuoteContext, RET_OK, SysConfig

    nas_binding = None
    if (root / "data/nas_sources.json").exists():
        from src.data.nas_market_data import load_current_nas_binding, materialize_nas_snapshot

        nas_binding = load_current_nas_binding(workspace_root=root)
        materialize_nas_snapshot(nas_binding, workspace_root=root, include_derived=True)
    path = root / "data/processed/intraday/soxl_soxx_none_1m.parquet"
    raw = read_intraday_cache(path)
    if raw.empty or not raw.adjustment_type.eq("NONE").all():
        raise ValueError("blocked: canonical raw NONE minute plane unavailable")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    calendar = xc.get_calendar("XNYS", start=raw.date.min(), end=session_date).schedule
    audit = audit_minute_coverage(raw, calendar, symbols=["SOXL", "SOXX"],
        timestamp_convention="right_edge", observed_at=pd.Timestamp.now(tz="UTC"))
    repairs = plan_minute_repairs(audit)
    # Malformed vendor candles remain quarantined; a repeated identical request
    # does not become a successful repair. Missing dates/minutes are backfilled.
    requests = {(row.symbol, row.start, row.end) for row in repairs.itertuples() if row.missing_minutes > 0}
    tail = min([session_date, str(raw.date.max()), *([start_session_date] if start_session_date else [])])
    requests.update((symbol, tail, session_date) for symbol in ("SOXL", "SOXX"))
    receipts = []
    for symbol, start, end in sorted(requests):
        fetched = fetch_futu_intraday_bars(symbol, start=start, end=end, host=host, port=port,
                                           adjust="none", session="RTH")
        fetched["adjustment_type"] = "NONE"
        fetched["price_valid"] = fetched.high.ge(fetched[["open", "close", "low"]].max(axis=1)) & fetched.low.le(fetched[["open", "close", "high"]].min(axis=1))
        fetched["bar_end_at"] = fetched.timestamp
        fetched["bar_start_at"] = fetched.timestamp - pd.Timedelta(minutes=1)
        fetched["received_at"] = pd.to_datetime(fetched.fetched_at, utc=True)
        fetched["is_final"] = fetched.timestamp.dt.tz_convert("UTC") + pd.Timedelta(minutes=1) <= fetched.received_at
        fetched["availability_contract"] = "historical_right_edge_plus_60s_assumption_not_live_receipt_certification"
        raw = merge_intraday_bars(raw, fetched)
        receipts.append({"symbol": symbol, "start": start, "end": end, "rows": len(fetched)})
    SysConfig.set_all_thread_daemon(True)
    context = OpenQuoteContext(host=host, port=port)
    rehab = {}
    try:
        for symbol in ("SOXL", "SOXX"):
            ret, events = context.get_rehab("US." + symbol)
            if ret != RET_OK:
                raise ValueError("needs_data: published events unavailable for " + symbol)
            rehab[symbol] = events
    finally:
        context.close()
    updated = audit_minute_coverage(raw, calendar, symbols=["SOXL", "SOXX"],
        timestamp_convention="right_edge", observed_at=pd.Timestamp.now(tz="UTC"))
    result = publish_core_stores(raw, rehab, root=root, as_of=session_date,
                                source="canonical NONE minute refresh; published Futu events")
    result.update(session_date=session_date, fetch_start=tail, repair_requests=receipts,
                  prior_raw_sha256=before, remaining_source_anomalies=updated.loc[updated.repair_required].to_dict("records"))
    if nas_binding is not None:
        from src.data.nas_market_data import publish_nas_snapshot, register_nas_source

        source_dir = root / "data/raw/external/semis_minute_refresh"
        files = {"data/processed/intraday/soxl_soxx_none_1m.parquet": "source",
                 "data/processed/intraday/soxl_soxx_1m.parquet": "derived"}
        for symbol, events in rehab.items():
            event_path = source_dir / (symbol + "_rehab.parquet")
            atomic_parquet(events, event_path)
            files[str(event_path.relative_to(root))] = "source"
        source_manifest = source_dir / "source_manifest.json"
        atomic_json({"schema_version": "canonical_none_refresh_v1", "as_of": session_date,
            "parent_source_id": nas_binding["source_id"], "parent_snapshot": nas_binding["snapshot_root"],
            "raw_cache": "data/processed/intraday/soxl_soxx_none_1m.parquet",
            "repair_requests": receipts, "source_adjustment": "NONE", "session": "RTH",
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rehab_sha256": {symbol: hashlib.sha256((source_dir / (symbol + "_rehab.parquet")).read_bytes()).hexdigest()
                             for symbol in rehab}}, source_manifest)
        files[str(source_manifest.relative_to(root))] = "source"
        inherited = [{**entry, "snapshot_root": entry.get("snapshot_root", nas_binding["snapshot_root"])}
                     for entry in nas_binding["objects"] if entry["local_path"] not in files]
        context_as_of = nas_binding.get("metadata", {}).get("context_as_of", nas_binding["as_of"])
        context_invalid = int(nas_binding.get("metadata", {}).get("context_invalid_rows", 0))
        total_invalid = int(updated.invalid_rows.sum()) + context_invalid
        published_binding = publish_nas_snapshot(workspace_root=root, source_manifest=source_manifest,
            files=files, as_of=session_date, inherited_objects=inherited,
            source_metadata={**{key: value for key, value in nas_binding.get("metadata", {}).items() if key.startswith("context_")},
                "context_as_of": context_as_of, "collector": "mac_futu_opend",
                "local_store_policy": "disposable Qlib cache",
                "quality_status": "partial" if total_invalid else "ok",
                "malformed_rows": int(updated.invalid_rows.sum()), "core_rows": int(len(raw))})
        result["nas_source"] = register_nas_source(published_binding, workspace_root=root,
            error_count=total_invalid,
            observation={"quality_status": "partial" if total_invalid else "ok",
                "core_rows": int(len(raw)), "core_last_bar": str(raw.timestamp.max()),
                "context_as_of": context_as_of, "context_invalid_rows": context_invalid, "missing_closed_minutes": int(updated.missing_minutes.sum()),
                "malformed_rows": int(updated.invalid_rows.sum()),
                "research_adjustment": "published_futu_affine_v1", "source_adjustment": "NONE"})
    return result

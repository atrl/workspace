from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import (
    QlibProviderConfig,
    build_qlib_store_from_frame,
    fetch_polygon_aggregate_bars,
    init_qlib_provider,
    load_qlib_feature_frame,
    merge_intraday_bars,
    polygon_intraday_to_qlib_frame,
    read_intraday_cache,
    write_intraday_cache,
)
from src.workflow.intraday_bar_model import _recorder_files, _remove_transient_code_snapshots


EXPERIMENT = "intraday_external_context_data_v1"
REQUESTED_SYMBOLS = ("QQQ", "SMH", "SOXX", "NVDA", "AVGO", "AMD", "TSM", "ASML")
QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "highfreq": "https://qlib.readthedocs.io/en/latest/component/highfreq.html",
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
}
QLIB_NATIVE_API = [
    "qlib.init",
    "qlib.data.D.features",
    "qlib.workflow.R.start",
    "qlib.workflow.Recorder.log_artifact",
    "build_qlib_store_from_frame",
]
ALLOWED_DATA_STATUS = {
    "ok",
    "ok_partial_insufficient_for_full_walk_forward",
    "blocked_missing_api_key",
    "blocked_provider_rate_limit",
    "blocked_provider_timeout",
    "blocked_provider_history",
    "blocked_no_rows",
}


@dataclass(frozen=True)
class IntradayExternalContextDataSpec:
    root: Path
    experiment: str = EXPERIMENT
    requested_symbols: tuple[str, ...] = REQUESTED_SYMBOLS
    start: str = "2019-01-02"
    end: str = "2026-07-02"
    full_walk_forward_start: str = "2019-01-02"
    as_of: str = "2026-07-02"
    frequency: str = "15min"
    market: str = "all"
    region: str = "us"
    source: str = "polygon_aggs_custom_bars"

    @property
    def output_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def recorder_uri(self) -> Path:
        return self.output_dir / "qlib_recorder"

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_external_intraday_15min"

    @property
    def fallback_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def raw_cache_dir(self) -> Path:
        return self.root / "data" / "raw" / "intraday" / "polygon_external_15min"


def run_intraday_external_context_data_card(
    *,
    root: str | Path | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    spec = IntradayExternalContextDataSpec(root=_resolve_root(root))
    return _run_card(spec, api_key=api_key or os.environ.get("POLYGON_API_KEY"))


def _resolve_root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[2]


def _run_card(spec: IntradayExternalContextDataSpec, *, api_key: str | None) -> dict[str, Any]:
    from qlib.workflow import R

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)
    workflow_yaml = _write_workflow_yaml(spec)

    source_audit, normalized = _fetch_requested_symbols(spec, api_key=api_key)
    qlib_frame = polygon_intraday_to_qlib_frame(normalized) if not normalized.empty else pd.DataFrame()
    coverage_audit = _coverage_audit(spec, qlib_frame=qlib_frame, source_audit=source_audit)
    available_symbols = sorted(symbol for symbol in spec.requested_symbols if symbol in set(qlib_frame.get("symbol", [])))
    missing_symbols = sorted(set(spec.requested_symbols) - set(available_symbols))
    provider_built = bool(not qlib_frame.empty and set(available_symbols) == set(spec.requested_symbols))
    data_status = _data_status(
        spec,
        source_audit=source_audit,
        available_symbols=available_symbols,
        qlib_frame=qlib_frame,
        api_key=api_key,
    )

    manifest = _write_or_build_provider(spec, qlib_frame=qlib_frame) if provider_built else _blocked_manifest(spec, data_status)
    qlib_feature_sample = _qlib_feature_sample(spec, available_symbols=available_symbols, provider_built=provider_built)
    analysis_summary = _analysis_summary(
        spec,
        data_status=data_status,
        source_audit=source_audit,
        coverage_audit=coverage_audit,
        manifest=manifest,
        available_symbols=available_symbols,
        missing_symbols=missing_symbols,
    )
    sidecars = _write_sidecars(
        spec,
        source_audit=source_audit,
        coverage_audit=coverage_audit,
        manifest=manifest,
        analysis_summary=analysis_summary,
        data_status=data_status,
    )

    R.set_uri(spec.recorder_uri.resolve().as_posix())
    with R.start(experiment_name=spec.experiment):
        recorder = R.get_recorder()
        R.log_params(
            experiment=spec.experiment,
            source=spec.source,
            start=spec.start,
            end=spec.end,
            frequency=spec.frequency,
            requested_symbols=",".join(spec.requested_symbols),
            data_status=data_status,
        )
        R.log_metrics(
            available_symbol_count=float(len(available_symbols)),
            missing_symbol_count=float(len(missing_symbols)),
            regular_session_rows=float(len(qlib_frame)),
        )
        recorder.log_artifact((spec.output_dir / sidecars["source_audit"]).as_posix(), artifact_path="intraday_external_context")
        recorder.log_artifact(
            (spec.output_dir / "qlib_feature_sample.csv").as_posix(),
            artifact_path="intraday_external_context",
        )
        for rel_path in sidecars.values():
            recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
        recorder_id = recorder.id
        recorder_local_dir = Path(recorder.get_local_dir())

    (spec.output_dir / "qlib_feature_sample.csv").unlink(missing_ok=True)
    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    summary = _summary(
        spec,
        workflow_yaml=workflow_yaml,
        data_status=data_status,
        source_audit=source_audit,
        coverage_audit=coverage_audit,
        manifest=manifest,
        sidecars=sidecars,
        available_symbols=available_symbols,
        missing_symbols=missing_symbols,
        recorder_id=recorder_id,
        recorder_local_dir=recorder_local_dir,
        recorder_artifacts=recorder_artifacts,
        provider_built=provider_built,
    )
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (spec.output_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")
    return summary


def _fetch_requested_symbols(
    spec: IntradayExternalContextDataSpec,
    *,
    api_key: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    if not api_key:
        for symbol in spec.requested_symbols:
            audit_rows.append(
                {
                    "symbol": symbol,
                    "status": "blocked_missing_api_key",
                    "http_status": 0,
                    "row_count": 0,
                    "page_count": 0,
                    "message": "POLYGON_API_KEY missing",
                    "requested_start": spec.start,
                    "requested_end": spec.end,
                    "coverage_start": "",
                    "coverage_end": "",
                    "provider_history_truncated": True,
                    "raw_cache_path": "",
                    "cache_row_count": 0,
                    "fetched_row_count": 0,
                    "fetch_range_count": 0,
                    "blocked_range_count": 0,
                    "chunk_statuses": "",
                }
            )
        return pd.DataFrame(audit_rows), pd.DataFrame()

    for symbol in spec.requested_symbols:
        frame, audit = _fetch_symbol_with_cache(spec, symbol, api_key=api_key)
        audit_rows.append(audit)
        if not frame.empty:
            frames.append(frame)
    normalized = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame(audit_rows), normalized


def _fetch_symbol_with_cache(
    spec: IntradayExternalContextDataSpec,
    symbol: str,
    *,
    api_key: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = _read_cached_symbol_frame(spec, symbol)
    cache_row_count = int(len(cached))
    seeded_provider_row_count = 0
    cache_source = "raw_cache" if cache_row_count else ""
    seeded = _read_existing_provider_symbol_frame(spec, symbol)
    if not seeded.empty:
        merged_seeded = merge_intraday_bars(seeded, cached)
        seeded_provider_row_count = max(0, int(len(merged_seeded)) - int(len(cached)))
        if seeded_provider_row_count:
            write_intraday_cache(merged_seeded, _raw_cache_path(spec, symbol))
            cached = merged_seeded
            cache_source = "existing_qlib_provider" if cache_row_count == 0 else "raw_cache+existing_qlib_provider"

    ranges = _missing_fetch_ranges(spec, cached)
    if not ranges and not cached.empty:
        status = "ok_seeded_provider_cache" if seeded_provider_row_count else "ok_cached"
        return cached, _cached_source_audit(
            spec,
            symbol,
            cached,
            status=status,
            cache_row_count=cache_row_count or int(len(cached)),
            seeded_provider_row_count=seeded_provider_row_count,
            cache_source=cache_source,
        )

    merged = cached.copy()
    chunk_audits: list[dict[str, Any]] = []
    fetched_row_count = 0
    blocked_range_count = 0
    cache_path = _raw_cache_path(spec, symbol)

    for range_start, range_end in ranges:
        frame, audit = fetch_polygon_aggregate_bars(
            symbol,
            start=range_start,
            end=range_end,
            api_key=api_key,
            multiplier=15,
            max_retries=0,
            retry_sleep_seconds=0.0,
        )
        chunk_audits.append(audit)
        if not frame.empty:
            fetched_row_count += int(len(frame))
            merged = merge_intraday_bars(merged, frame)
            write_intraday_cache(merged, cache_path)
        status = str(audit.get("status", ""))
        if status in {"blocked_provider_rate_limit", "blocked_provider_timeout"}:
            blocked_range_count += 1
            break
        if status == "ok_partial_provider_rate_limit":
            blocked_range_count += 1
            break
        if status.startswith("blocked_"):
            blocked_range_count += 1

    if merged.empty:
        audit = chunk_audits[-1] if chunk_audits else _blocked_no_rows_audit(spec, symbol)
        return pd.DataFrame(), {
            **audit,
            "raw_cache_path": "",
            "cache_row_count": cache_row_count,
            "seeded_provider_row_count": seeded_provider_row_count,
            "fetched_row_count": fetched_row_count,
            "fetch_range_count": len(chunk_audits),
            "blocked_range_count": blocked_range_count,
            "chunk_statuses": ",".join(str(row.get("status", "")) for row in chunk_audits),
            "cache_source": cache_source,
        }

    return merged, _merged_source_audit(
        spec,
        symbol,
        merged,
        cache_row_count=cache_row_count,
        seeded_provider_row_count=seeded_provider_row_count,
        fetched_row_count=fetched_row_count,
        fetch_range_count=len(chunk_audits),
        blocked_range_count=blocked_range_count,
        chunk_statuses=[str(row.get("status", "")) for row in chunk_audits],
        cache_source=cache_source,
    )


def _raw_cache_path(spec: IntradayExternalContextDataSpec, symbol: str) -> Path:
    return spec.raw_cache_dir / f"{symbol.upper()}_{spec.frequency}.parquet"


def _read_cached_symbol_frame(
    spec: IntradayExternalContextDataSpec,
    symbol: str,
) -> pd.DataFrame:
    path = _raw_cache_path(spec, symbol)
    if not path.exists():
        return pd.DataFrame()
    frame = read_intraday_cache(path)
    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame.loc[frame["symbol"].eq(symbol.upper())].copy()
    if frame.empty:
        return pd.DataFrame()
    return frame


def _read_existing_provider_symbol_frame(
    spec: IntradayExternalContextDataSpec,
    symbol: str,
) -> pd.DataFrame:
    manifest_path = spec.provider_uri / "qlib_store_manifest.json"
    if not manifest_path.exists():
        return pd.DataFrame()
    try:
        sample = load_qlib_feature_frame(
            QlibProviderConfig(
                provider_uri=spec.provider_uri,
                region=spec.region,
                market=spec.market,
                freq=spec.frequency,
            ),
            instruments=[symbol.upper()],
            fields=["$open", "$high", "$low", "$close", "$volume", "$turnover", "$vwap"],
            start=f"{spec.start} 00:00:00",
            end=f"{spec.end} 23:59:59",
        )
    except Exception:
        return pd.DataFrame()
    if sample.empty:
        return pd.DataFrame()
    frame = sample.reset_index()
    if "datetime" not in frame:
        return pd.DataFrame()
    timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")
    else:
        timestamps = timestamps.dt.tz_convert("America/New_York")
    interval = spec.frequency.replace("min", "m")
    normalized = pd.DataFrame(
        {
            "provider": "qlib_seed",
            "symbol": symbol.upper(),
            "interval": interval,
            "timestamp": timestamps,
            "date": timestamps.dt.date.astype(str),
            "session": "regular",
            "open": pd.to_numeric(frame["$open"], errors="coerce") if "$open" in frame else pd.NA,
            "high": pd.to_numeric(frame["$high"], errors="coerce") if "$high" in frame else pd.NA,
            "low": pd.to_numeric(frame["$low"], errors="coerce") if "$low" in frame else pd.NA,
            "close": pd.to_numeric(frame["$close"], errors="coerce") if "$close" in frame else pd.NA,
            "volume": pd.to_numeric(frame["$volume"], errors="coerce") if "$volume" in frame else pd.NA,
            "turnover": pd.to_numeric(frame["$turnover"], errors="coerce") if "$turnover" in frame else pd.NA,
            "vwap": pd.to_numeric(frame["$vwap"], errors="coerce") if "$vwap" in frame else pd.NA,
            "transactions": pd.NA,
            "source_tz": "America/New_York",
            "adjusted": True,
            "fetched_at": "existing_qlib_provider_seed",
        }
    )
    normalized = normalized.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    if normalized.empty:
        return pd.DataFrame()
    return normalized.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last").reset_index(
        drop=True
    )


def _cached_source_audit(
    spec: IntradayExternalContextDataSpec,
    symbol: str,
    frame: pd.DataFrame,
    *,
    status: str = "ok_cached",
    cache_row_count: int | None = None,
    seeded_provider_row_count: int = 0,
    cache_source: str = "raw_cache",
) -> dict[str, Any]:
    coverage_start = _format_min(frame["timestamp"].min()) if "timestamp" in frame else ""
    coverage_end = _format_min(frame["timestamp"].max()) if "timestamp" in frame else ""
    provider_history_truncated = True
    if coverage_start and coverage_end:
        provider_history_truncated = bool(
            pd.Timestamp(frame["timestamp"].min()).date() > pd.Timestamp(spec.start).date()
            or pd.Timestamp(frame["timestamp"].max()).date() < pd.Timestamp(spec.end).date()
        )
    return {
        "symbol": symbol.upper(),
        "status": status,
        "http_status": 0,
        "row_count": int(len(frame)),
        "page_count": 0,
        "message": "raw cache hit",
        "requested_start": spec.start,
        "requested_end": spec.end,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "provider_history_truncated": provider_history_truncated,
        "raw_cache_path": _raw_cache_path(spec, symbol).as_posix(),
        "cache_row_count": int(len(frame)) if cache_row_count is None else int(cache_row_count),
        "seeded_provider_row_count": seeded_provider_row_count,
        "fetched_row_count": 0,
        "fetch_range_count": 0,
        "blocked_range_count": 0,
        "chunk_statuses": "",
        "cache_source": cache_source,
    }


def _merged_source_audit(
    spec: IntradayExternalContextDataSpec,
    symbol: str,
    frame: pd.DataFrame,
    *,
    cache_row_count: int,
    seeded_provider_row_count: int,
    fetched_row_count: int,
    fetch_range_count: int,
    blocked_range_count: int,
    chunk_statuses: list[str],
    cache_source: str,
) -> dict[str, Any]:
    coverage_start = _format_min(frame["timestamp"].min()) if "timestamp" in frame else ""
    coverage_end = _format_min(frame["timestamp"].max()) if "timestamp" in frame else ""
    provider_history_truncated = _frame_history_truncated(spec, frame)
    if fetched_row_count > 0:
        if cache_row_count == 0 and "ok_partial_provider_rate_limit" in chunk_statuses:
            status = "ok_partial_provider_rate_limit"
        elif cache_row_count == 0:
            status = "ok_partial_provider_history" if provider_history_truncated or blocked_range_count else "ok"
        else:
            status = "ok_partial_cache_updated" if provider_history_truncated or blocked_range_count else "ok_cache_updated"
    elif cache_row_count > 0:
        status = "ok_cached_partial_fetch_blocked" if provider_history_truncated or blocked_range_count else "ok_cached"
    elif seeded_provider_row_count > 0:
        status = (
            "ok_seeded_provider_partial_fetch_blocked"
            if provider_history_truncated or blocked_range_count
            else "ok_seeded_provider_cache"
        )
    else:
        status = "blocked_no_rows"
    return {
        "symbol": symbol.upper(),
        "status": status,
        "http_status": 0,
        "row_count": int(len(frame)),
        "page_count": 0,
        "message": "raw cache updated" if fetched_row_count else "raw cache hit",
        "requested_start": spec.start,
        "requested_end": spec.end,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "provider_history_truncated": provider_history_truncated,
        "raw_cache_path": _raw_cache_path(spec, symbol).as_posix(),
        "cache_row_count": cache_row_count,
        "seeded_provider_row_count": seeded_provider_row_count,
        "fetched_row_count": fetched_row_count,
        "fetch_range_count": fetch_range_count,
        "blocked_range_count": blocked_range_count,
        "chunk_statuses": ",".join(status for status in chunk_statuses if status),
        "cache_source": cache_source,
    }


def _blocked_no_rows_audit(spec: IntradayExternalContextDataSpec, symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol.upper(),
        "status": "blocked_no_rows",
        "http_status": 0,
        "row_count": 0,
        "page_count": 0,
        "message": "no fetch ranges",
        "requested_start": spec.start,
        "requested_end": spec.end,
        "coverage_start": "",
        "coverage_end": "",
        "provider_history_truncated": True,
        "raw_cache_path": "",
        "cache_row_count": 0,
        "seeded_provider_row_count": 0,
        "fetched_row_count": 0,
        "fetch_range_count": 0,
        "blocked_range_count": 0,
        "chunk_statuses": "",
        "cache_source": "",
    }


def _missing_fetch_ranges(spec: IntradayExternalContextDataSpec, cached: pd.DataFrame) -> list[tuple[str, str]]:
    requested_start = pd.Timestamp(spec.start).date()
    requested_end = pd.Timestamp(spec.end).date()
    if requested_start > requested_end:
        return []
    if cached.empty:
        return _split_date_range_recent_first(requested_start, requested_end)

    timestamps = pd.to_datetime(cached["timestamp"], errors="coerce")
    min_date = timestamps.min().date()
    max_date = timestamps.max().date()
    ranges: list[tuple[str, str]] = []
    if max_date < requested_end:
        after_start = (pd.Timestamp(max_date) + pd.Timedelta(days=1)).date()
        ranges.extend(_split_date_range_recent_first(after_start, requested_end))
    if min_date > requested_start:
        before_end = (pd.Timestamp(min_date) - pd.Timedelta(days=1)).date()
        ranges.extend(_split_date_range_recent_first(requested_start, before_end))
    return ranges


def _split_date_range_recent_first(start: Any, end: Any) -> list[tuple[str, str]]:
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if start_date > end_date:
        return []
    ranges: list[tuple[str, str]] = []
    for year in range(end_date.year, start_date.year - 1, -1):
        chunk_start = max(start_date, pd.Timestamp(year=year, month=1, day=1).date())
        chunk_end = min(end_date, pd.Timestamp(year=year, month=12, day=31).date())
        if chunk_start <= chunk_end:
            ranges.append((chunk_start.isoformat(), chunk_end.isoformat()))
    return ranges


def _frame_history_truncated(spec: IntradayExternalContextDataSpec, frame: pd.DataFrame) -> bool:
    if frame.empty or "timestamp" not in frame:
        return True
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().all():
        return True
    return bool(
        timestamps.min().date() > pd.Timestamp(spec.start).date()
        or timestamps.max().date() < pd.Timestamp(spec.end).date()
    )


def _coverage_audit(
    spec: IntradayExternalContextDataSpec,
    *,
    qlib_frame: pd.DataFrame,
    source_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in spec.requested_symbols:
        symbol_frame = qlib_frame.loc[qlib_frame["symbol"].eq(symbol)] if not qlib_frame.empty else pd.DataFrame()
        source_row = source_audit.loc[source_audit["symbol"].eq(symbol)]
        rows.append(
            {
                "symbol": symbol,
                "status": str(source_row["status"].iloc[0]) if not source_row.empty else "blocked_no_rows",
                "coverage_start": _format_min(symbol_frame["datetime"].min()) if not symbol_frame.empty else "",
                "coverage_end": _format_min(symbol_frame["datetime"].max()) if not symbol_frame.empty else "",
                "regular_session_rows": int(len(symbol_frame)),
                "first_session": str(symbol_frame["date"].min()) if not symbol_frame.empty else "",
                "last_session": str(symbol_frame["date"].max()) if not symbol_frame.empty else "",
            }
        )
    return pd.DataFrame(rows)


def _data_status(
    spec: IntradayExternalContextDataSpec,
    *,
    source_audit: pd.DataFrame,
    available_symbols: list[str],
    qlib_frame: pd.DataFrame,
    api_key: str | None,
) -> str:
    if not api_key:
        return "blocked_missing_api_key"
    statuses = set(source_audit["status"].astype(str)) if not source_audit.empty else {"blocked_no_rows"}
    if not available_symbols:
        if "blocked_provider_rate_limit" in statuses:
            return "blocked_provider_rate_limit"
        if "blocked_provider_timeout" in statuses:
            return "blocked_provider_timeout"
        if "blocked_provider_history" in statuses:
            return "blocked_provider_history"
        return "blocked_no_rows"
    if set(available_symbols) != set(spec.requested_symbols):
        return "ok_partial_insufficient_for_full_walk_forward"
    coverage_start = pd.Timestamp(qlib_frame["datetime"].min()).date().isoformat()
    if coverage_start > spec.full_walk_forward_start:
        return "ok_partial_insufficient_for_full_walk_forward"
    return "ok"


def _write_or_build_provider(spec: IntradayExternalContextDataSpec, *, qlib_frame: pd.DataFrame) -> dict[str, Any]:
    manifest = build_qlib_store_from_frame(
        qlib_frame,
        provider_uri=spec.provider_uri,
        market=spec.market,
        region=spec.region,
        freq=spec.frequency,
        field_map={
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$turnover": "turnover",
            "$vwap": "vwap",
            "$bar_index": "bar_index",
            "$bar_count": "bar_count",
        },
        as_of=spec.as_of,
        source=spec.source,
    )
    return manifest


def _blocked_manifest(spec: IntradayExternalContextDataSpec, status: str) -> dict[str, Any]:
    return {
        "operation": "qlib_store_build",
        "status": status,
        "source": spec.source,
        "provider_uri": "",
        "market": spec.market,
        "region": spec.region,
        "freq": spec.frequency,
        "as_of": spec.as_of,
        "instruments": [],
        "fields": [],
        "row_count": 0,
        "source_api": QLIB_NATIVE_API,
    }


def _qlib_feature_sample(
    spec: IntradayExternalContextDataSpec,
    *,
    available_symbols: list[str],
    provider_built: bool,
) -> pd.DataFrame:
    if provider_built and available_symbols:
        sample = load_qlib_feature_frame(
            QlibProviderConfig(provider_uri=spec.provider_uri, region=spec.region, market=spec.market, freq=spec.frequency),
            instruments=available_symbols[: min(2, len(available_symbols))],
            fields=["$close", "$vwap"],
            start=f"{spec.start} 09:45:00",
            end=f"{spec.end} 16:00:00",
        )
        frame = sample.reset_index().head(64)
    else:
        init_qlib_provider(QlibProviderConfig(provider_uri=spec.fallback_provider_uri, region=spec.region, market=spec.market))
        frame = load_qlib_feature_frame(
            QlibProviderConfig(provider_uri=spec.fallback_provider_uri, region=spec.region, market=spec.market),
            instruments=["SOXL"],
            fields=["$close"],
            start=spec.end,
            end=spec.end,
        ).reset_index()
        frame["external_context_status"] = "blocked_no_external_provider_rows"
    frame.to_csv(spec.output_dir / "qlib_feature_sample.csv", index=False)
    return frame


def _analysis_summary(
    spec: IntradayExternalContextDataSpec,
    *,
    data_status: str,
    source_audit: pd.DataFrame,
    coverage_audit: pd.DataFrame,
    manifest: dict[str, Any],
    available_symbols: list[str],
    missing_symbols: list[str],
) -> dict[str, Any]:
    return {
        "source": "intraday_external_context_data_card",
        "data_status": data_status,
        "requested_symbols": list(spec.requested_symbols),
        "available_symbols": available_symbols,
        "missing_symbols": missing_symbols,
        "requested_range": [spec.start, spec.end],
        "full_walk_forward_required_range": [spec.full_walk_forward_start, spec.end],
        "raw_cache_dir": spec.raw_cache_dir.resolve().as_posix(),
        "source_status_counts": source_audit["status"].value_counts().to_dict() if not source_audit.empty else {},
        "coverage_rows": int(len(coverage_audit)),
        "regular_session_rows": int(coverage_audit["regular_session_rows"].sum()) if not coverage_audit.empty else 0,
        "provider_manifest_status": manifest.get("status"),
        "provider_manifest_hash": manifest.get("hash", ""),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API,
    }


def _write_sidecars(
    spec: IntradayExternalContextDataSpec,
    *,
    source_audit: pd.DataFrame,
    coverage_audit: pd.DataFrame,
    manifest: dict[str, Any],
    analysis_summary: dict[str, Any],
    data_status: str,
) -> dict[str, str]:
    sidecars = {
        "analysis_summary": "analysis_summary.json",
        "coverage_audit": "coverage_audit.csv",
        "failure_modes": "failure_modes.md",
        "qlib_store_manifest": "qlib_store_manifest.json",
        "source_audit": "source_audit.csv",
    }
    source_audit.to_csv(spec.output_dir / sidecars["source_audit"], index=False)
    coverage_audit.to_csv(spec.output_dir / sidecars["coverage_audit"], index=False)
    (spec.output_dir / sidecars["qlib_store_manifest"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["failure_modes"]).write_text(_failure_modes(data_status), encoding="utf-8")
    return sidecars


def _summary(
    spec: IntradayExternalContextDataSpec,
    *,
    workflow_yaml: Path,
    data_status: str,
    source_audit: pd.DataFrame,
    coverage_audit: pd.DataFrame,
    manifest: dict[str, Any],
    sidecars: dict[str, str],
    available_symbols: list[str],
    missing_symbols: list[str],
    recorder_id: str,
    recorder_local_dir: Path,
    recorder_artifacts: list[str],
    provider_built: bool,
) -> dict[str, Any]:
    if data_status not in ALLOWED_DATA_STATUS:
        raise ValueError(f"unexpected external context data status: {data_status}")
    return {
        "experiment": spec.experiment,
        "card_type": "intraday_external_context_data",
        "workflow_config": workflow_yaml.name,
        "as_of": spec.as_of,
        "market": spec.market,
        "region": spec.region,
        "source": spec.source,
        "frequency": spec.frequency,
        "requested_range": [spec.start, spec.end],
        "full_walk_forward_required_range": [spec.full_walk_forward_start, spec.end],
        "raw_cache_dir": spec.raw_cache_dir.resolve().as_posix(),
        "requested_symbols": list(spec.requested_symbols),
        "available_symbols": available_symbols,
        "missing_symbols": missing_symbols,
        "objective_context_gap_closed_or_audited": list(spec.requested_symbols),
        "data_status": data_status,
        "provider_uri": spec.provider_uri.resolve().as_posix() if provider_built else "",
        "provider_manifest": spec.provider_uri.joinpath("qlib_store_manifest.json").as_posix() if provider_built else "",
        "source_audit_status_counts": source_audit["status"].value_counts().to_dict() if not source_audit.empty else {},
        "coverage_audit": {
            "regular_session_rows": int(coverage_audit["regular_session_rows"].sum()) if not coverage_audit.empty else 0,
            "symbols_with_rows": available_symbols,
        },
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API,
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.resolve().as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": workflow_yaml.name,
            "primary_artifact": f"{spec.experiment}.ipynb",
            "machine_entrypoint": "summary.json",
            "kept_sidecars": list(sidecars.values()),
            "recorder_only_artifacts": {
                "source_audit": "artifacts/intraday_external_context/source_audit.csv",
                "qlib_feature_sample": "artifacts/intraday_external_context/qlib_feature_sample.csv",
            },
            "reproducible_from_recorder": True,
        },
        "manifest_status": manifest.get("status"),
        "manifest_hash": manifest.get("hash", ""),
    }


def _write_workflow_yaml(spec: IntradayExternalContextDataSpec) -> Path:
    path = spec.output_dir / "workflow.yaml"
    lines = [
        f"experiment: {spec.experiment}",
        "card_type: intraday_external_context_data",
        "qlib:",
        f"  provider_uri: {spec.provider_uri.resolve().as_posix()}",
        f"  recorder_uri: {spec.recorder_uri.resolve().as_posix()}",
        f"  market: {spec.market}",
        f"  region: {spec.region}",
        f"  frequency: {spec.frequency}",
        "source:",
        f"  provider: {spec.source}",
        f"  requested_range: [{spec.start}, {spec.end}]",
        f"  raw_cache_dir: {spec.raw_cache_dir.resolve().as_posix()}",
        "  requested_symbols:",
    ]
    lines.extend(f"    - {symbol}" for symbol in spec.requested_symbols)
    lines.extend(
        [
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  machine_entrypoint: summary.json",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _failure_modes(data_status: str) -> str:
    return "\n".join(
        [
            "# Failure Modes",
            "",
            f"- data_status: {data_status}",
            "- blocked_missing_api_key: POLYGON_API_KEY was absent; no external minute context can be trusted.",
            "- blocked_provider_rate_limit: provider throttled the request; rerun with backoff or a paid quota window.",
            "- blocked_provider_history: provider plan did not authorize the requested range or symbol.",
            "- ok_partial_insufficient_for_full_walk_forward: some data was built, but it is not enough to retrain the full 2019-2026 walk-forward external context model.",
            "- no fake breadth columns are created when QQQ/SMH/constituent rows are missing.",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 中文结论",
            "",
            "- 做了什么：建立外部 15m 上下文数据卡，审计 QQQ/SMH/SOXX 与 NVDA/AVGO/AMD/TSM/ASML 是否可进入 qlib provider。",
            f"- 数据状态：{summary['data_status']}；可用={summary['available_symbols']}；缺失={summary['missing_symbols']}。",
            f"- 覆盖范围：请求 {summary['requested_range'][0]} 到 {summary['requested_range'][1]}，完整 walk-forward 需要 {summary['full_walk_forward_required_range'][0]} 到 {summary['full_walk_forward_required_range'][1]}。",
            "- 用途：这张卡只解决外部 15m 上下文数据和缺口审计，不直接证明日内模型可晋升。",
            "- 风险：Polygon 权限、历史窗口和限流会影响可重复性；缺失 symbol 不会被空值因子伪装成 breadth。",
            f"- qlib recorder：{summary['recorder_id']}",
        ]
    )


def _format_min(value: Any) -> str:
    if pd.isna(value):
        return ""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

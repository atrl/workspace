from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

import pandas as pd
import requests

from .qlib_native import QlibProviderConfig, audit_qlib_provider, build_qlib_store_from_frame


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_COLUMNS = [
    "date",
    "soxl_open",
    "soxl_high",
    "soxl_low",
    "soxl_close",
    "soxl_volume",
    "soxx_open",
    "soxx_high",
    "soxx_low",
    "soxx_close",
    "soxx_volume",
    "vix",
]
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
DEFAULT_PROVIDER_SEQUENCE = ("futu", "yahoo_api", "polygon", "yfinance")
QLIB_FIELD_MAP = {
    "$open": "open",
    "$high": "high",
    "$low": "low",
    "$close": "close",
    "$volume": "volume",
    "$vix": "vix",
}


@dataclass(frozen=True)
class MarketDataUpdatePaths:
    panel_csv: Path = REPO_ROOT / "data" / "processed" / "panels" / "soxl_soxx_vix_full.csv"
    raw_symbol_dir: Path = REPO_ROOT / "data" / "raw" / "symbols"
    raw_external_dir: Path = REPO_ROOT / "data" / "raw" / "external"
    qlib_provider_uri: Path = REPO_ROOT / "data" / "qlib" / "soxl"
    registry_yaml: Path = REPO_ROOT / "data" / "datasets.yaml"


@dataclass(frozen=True)
class MarketDataFrames:
    provider: str
    source_api: list[str]
    soxl: pd.DataFrame
    soxx: pd.DataFrame
    vix: pd.DataFrame
    raw_provider_frames: dict[str, pd.DataFrame] = field(default_factory=dict)


def _default_end_date() -> str:
    return date.today().isoformat()


def _fetch_start_from_existing(existing: pd.DataFrame, *, overlap_days: int, fallback: str) -> str:
    if existing.empty or "date" not in existing:
        return fallback
    last = pd.to_datetime(existing["date"], errors="coerce").max()
    if pd.isna(last):
        return fallback
    return (last.date() - timedelta(days=overlap_days)).isoformat()


def _read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PANEL_COLUMNS)
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"canonical panel missing date column: {path.as_posix()}")
    return _normalize_panel(frame)


def _normalize_date_column(frame: pd.DataFrame, *, date_col: str = "date") -> pd.Series:
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    return dates.dt.normalize().dt.date.astype("string")


def _normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"market frame missing columns: {missing}")
    normalized = frame.loc[:, OHLCV_COLUMNS].copy()
    normalized["date"] = _normalize_date_column(normalized)
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    normalized = normalized.drop_duplicates("date", keep="last").reset_index(drop=True)
    return normalized


def _normalize_vix_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "vix" not in frame.columns:
        if "close" in frame.columns:
            frame = frame.rename(columns={"close": "vix"})
        else:
            raise ValueError("vix frame missing vix or close column")
    normalized = frame.loc[:, ["date", "vix"]].copy()
    normalized["date"] = _normalize_date_column(normalized)
    normalized["vix"] = pd.to_numeric(normalized["vix"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "vix"]).sort_values("date")
    normalized = normalized.drop_duplicates("date", keep="last").reset_index(drop=True)
    return normalized


def _normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PANEL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"canonical panel missing columns: {missing}")
    normalized = frame.loc[:, PANEL_COLUMNS].copy()
    normalized["date"] = _normalize_date_column(normalized)
    for column in PANEL_COLUMNS:
        if column != "date":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["date", "soxl_close", "soxx_close", "vix"])
    normalized = normalized.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return normalized


def _date_filtered(frame: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["date"] = _normalize_date_column(normalized)
    return normalized.loc[(normalized["date"] >= start) & (normalized["date"] <= end)].reset_index(drop=True)


def combine_market_frames(frames: MarketDataFrames) -> pd.DataFrame:
    soxl = _normalize_ohlcv_frame(frames.soxl).rename(
        columns={column: f"soxl_{column}" for column in ["open", "high", "low", "close", "volume"]}
    )
    soxx = _normalize_ohlcv_frame(frames.soxx).rename(
        columns={column: f"soxx_{column}" for column in ["open", "high", "low", "close", "volume"]}
    )
    vix = _normalize_vix_frame(frames.vix)
    panel = soxl.merge(soxx, on="date", how="inner").merge(vix, on="date", how="inner")
    return _normalize_panel(panel.loc[:, PANEL_COLUMNS])


def merge_market_panel(
    existing: pd.DataFrame,
    fetched: pd.DataFrame,
    *,
    replace_overlap: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    existing = _normalize_panel(existing) if not existing.empty else pd.DataFrame(columns=PANEL_COLUMNS)
    fetched = _normalize_panel(fetched) if not fetched.empty else pd.DataFrame(columns=PANEL_COLUMNS)
    if existing.empty:
        return fetched, int(len(fetched)), 0
    if fetched.empty:
        return existing, 0, 0

    existing_dates = set(existing["date"].astype(str))
    fetched_dates = set(fetched["date"].astype(str))
    appended_rows = len(fetched_dates - existing_dates)
    replaced_rows = len(existing_dates & fetched_dates) if replace_overlap else 0

    if replace_overlap:
        merged = pd.concat([existing, fetched], ignore_index=True)
        merged = merged.drop_duplicates("date", keep="last")
    else:
        append_only = fetched.loc[~fetched["date"].astype(str).isin(existing_dates)]
        merged = pd.concat([existing, append_only], ignore_index=True)
    return _normalize_panel(merged), int(appended_rows), int(replaced_rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def sync_local_data_surfaces(panel: pd.DataFrame, *, paths: MarketDataUpdatePaths) -> dict[str, Any]:
    normalized = _normalize_panel(panel)
    written: dict[str, Any] = {"raw_symbols": {}, "raw_external": {}}
    for symbol in ["SOXL", "SOXX"]:
        prefix = symbol.lower()
        symbol_frame = normalized[
            [
                "date",
                f"{prefix}_open",
                f"{prefix}_high",
                f"{prefix}_low",
                f"{prefix}_close",
                f"{prefix}_volume",
            ]
        ].rename(
            columns={
                f"{prefix}_open": "open",
                f"{prefix}_high": "high",
                f"{prefix}_low": "low",
                f"{prefix}_close": "close",
                f"{prefix}_volume": "volume",
            }
        )
        symbol_path = paths.raw_symbol_dir / f"{symbol}.csv"
        _write_csv(symbol_frame.loc[:, OHLCV_COLUMNS], symbol_path)
        written["raw_symbols"][symbol] = symbol_path

    vix_frame = normalized.loc[:, ["date", "vix"]]
    vix_path = paths.raw_external_dir / "VIX.csv"
    _write_csv(vix_frame, vix_path)
    written["raw_external"]["VIX"] = vix_path
    return written


def market_panel_to_qlib_frame(panel: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_panel(panel)
    rows: list[pd.DataFrame] = []
    for symbol in ["SOXL", "SOXX"]:
        prefix = symbol.lower()
        symbol_frame = normalized[
            [
                "date",
                f"{prefix}_open",
                f"{prefix}_high",
                f"{prefix}_low",
                f"{prefix}_close",
                f"{prefix}_volume",
                "vix",
            ]
        ].rename(
            columns={
                "date": "datetime",
                f"{prefix}_open": "open",
                f"{prefix}_high": "high",
                f"{prefix}_low": "low",
                f"{prefix}_close": "close",
                f"{prefix}_volume": "volume",
            }
        )
        symbol_frame.insert(1, "instrument", symbol)
        rows.append(symbol_frame)
    qlib_frame = pd.concat(rows, ignore_index=True)
    qlib_frame["datetime"] = pd.to_datetime(qlib_frame["datetime"]).dt.normalize()
    return qlib_frame.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def refresh_qlib_store(panel: pd.DataFrame, *, paths: MarketDataUpdatePaths) -> dict[str, Any]:
    normalized = _normalize_panel(panel)
    qlib_frame = market_panel_to_qlib_frame(normalized)
    as_of = str(normalized["date"].max())
    manifest = build_qlib_store_from_frame(
        qlib_frame,
        provider_uri=paths.qlib_provider_uri,
        market="all",
        region="us",
        freq="day",
        instrument_col="instrument",
        datetime_col="datetime",
        field_map=QLIB_FIELD_MAP,
        as_of=as_of,
        source=paths.panel_csv.resolve().as_posix(),
    )
    audit = audit_qlib_provider(
        QlibProviderConfig(provider_uri=paths.qlib_provider_uri, region="us", market="all", freq="day"),
        instruments=["SOXL", "SOXX"],
        fields=["$close", "$volume", "$vix"],
        start=str(normalized["date"].min()),
        end=as_of,
    )
    return {"manifest": manifest, "audit": audit}


def _yahoo_period(value: str, *, end: bool = False) -> int:
    ts = pd.Timestamp(value)
    if end:
        ts = ts + pd.Timedelta(days=1)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def _request_json(url: str, *, params: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 soxl-qlib-data-refresh"},
    )
    response.raise_for_status()
    return response.json()


def _fetch_yahoo_chart_payload(symbol: str, *, start: str, end: str) -> dict[str, Any]:
    encoded = quote(symbol, safe="")
    params = {
        "period1": _yahoo_period(start),
        "period2": _yahoo_period(end, end=True),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    errors: list[str] = []
    for host in ["query2.finance.yahoo.com", "query1.finance.yahoo.com"]:
        try:
            return _request_json(f"https://{host}/v8/finance/chart/{encoded}", params=params, timeout=30.0)
        except Exception as exc:  # pragma: no cover - network fallback is live-provider behavior
            errors.append(f"{host}: {exc}")
    raise RuntimeError(f"yahoo chart failed for {symbol}: {'; '.join(errors)}")


def _fetch_yahoo_chart_ohlcv(symbol: str, *, start: str, end: str) -> pd.DataFrame:
    payload = _fetch_yahoo_chart_payload(symbol, start=start, end=end)
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"yahoo chart error for {symbol}: {error}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"yahoo chart returned no result for {symbol}")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_block = (result.get("indicators", {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York").date.astype(str),
            "open": quote_block.get("open", []),
            "high": quote_block.get("high", []),
            "low": quote_block.get("low", []),
            "close": quote_block.get("close", []),
            "volume": quote_block.get("volume", []),
        }
    )
    return _normalize_ohlcv_frame(frame)


def _fetch_yahoo_chart_vix(*, start: str, end: str) -> pd.DataFrame:
    frame = _fetch_yahoo_chart_ohlcv("^VIX", start=start, end=end)
    return _normalize_vix_frame(frame.loc[:, ["date", "close"]].rename(columns={"close": "vix"}))


def _fetch_yfinance_ohlcv(symbol: str, *, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    history = yf.Ticker(symbol).history(start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat())
    if history.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}")
    history = history.reset_index()
    date_col = "Date" if "Date" in history.columns else history.columns[0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(history[date_col]).dt.date.astype(str),
            "open": history["Open"],
            "high": history["High"],
            "low": history["Low"],
            "close": history["Close"],
            "volume": history["Volume"],
        }
    )
    return _normalize_ohlcv_frame(frame)


def _fetch_yfinance_compatible_ohlcv(symbol: str, *, start: str, end: str) -> pd.DataFrame:
    errors: list[str] = []
    for source, fetcher in [
        ("yfinance.Ticker.history", _fetch_yfinance_ohlcv),
        ("query2/query1.finance.yahoo.com/v8/finance/chart", _fetch_yahoo_chart_ohlcv),
        ("polygon.io/v2/aggs", _fetch_polygon_ohlcv),
    ]:
        try:
            return fetcher(symbol, start=start, end=end)
        except Exception as exc:  # pragma: no cover - live provider fallback
            errors.append(f"{source}: {exc}")
    raise RuntimeError(f"no yfinance-compatible OHLCV source succeeded for {symbol}: {'; '.join(errors)}")


def _fetch_polygon_ohlcv(symbol: str, *, start: str, end: str) -> pd.DataFrame:
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is not configured")
    payload = _request_json(
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
        params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key},
    )
    results = payload.get("results") or []
    if not results:
        raise RuntimeError(f"polygon returned no rows for {symbol}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime([row["t"] for row in results], unit="ms", utc=True).date.astype(str),
            "open": [row.get("o") for row in results],
            "high": [row.get("h") for row in results],
            "low": [row.get("l") for row in results],
            "close": [row.get("c") for row in results],
            "volume": [row.get("v") for row in results],
        }
    )
    return _normalize_ohlcv_frame(frame)


def _fetch_cboe_vix(*, start: str, end: str) -> pd.DataFrame:
    response = requests.get(
        "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 soxl-qlib-data-refresh"},
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    frame.columns = [column.strip().lower() for column in frame.columns]
    date_col = "date" if "date" in frame.columns else frame.columns[0]
    close_col = "close" if "close" in frame.columns else frame.columns[-1]
    normalized = pd.DataFrame({"date": pd.to_datetime(frame[date_col]).dt.date.astype(str), "vix": frame[close_col]})
    return _date_filtered(_normalize_vix_frame(normalized), start=start, end=end)


def _fetch_vix_with_fallback(*, start: str, end: str) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for source, fetcher in [
        ("yahoo.chart:^VIX", _fetch_yahoo_chart_vix),
        ("cboe:VIX_History.csv", _fetch_cboe_vix),
    ]:
        try:
            frame = fetcher(start=start, end=end)
            if not frame.empty:
                return frame, source
            errors.append(f"{source}: empty")
        except Exception as exc:  # pragma: no cover - exercised by live provider fallback
            errors.append(f"{source}: {exc}")
    raise RuntimeError(f"no VIX provider succeeded: {'; '.join(errors)}")


def _futu_autype(name: str):
    from futu import AuType

    normalized = name.lower()
    if normalized in {"qfq", "forward"}:
        return AuType.QFQ
    if normalized in {"hfq", "backward"}:
        return AuType.HFQ
    return AuType.NONE


def _fetch_futu_market_frames(*, start: str, end: str, host: str, port: int, adjust: str) -> MarketDataFrames:
    from futu import KLType, OpenQuoteContext, RET_OK

    quote_ctx = OpenQuoteContext(host=host, port=port)
    raw_frames: dict[str, pd.DataFrame] = {}
    normalized_frames: dict[str, pd.DataFrame] = {}
    try:
        for symbol in ["SOXL", "SOXX"]:
            code = f"US.{symbol}"
            page_req_key = None
            parts: list[pd.DataFrame] = []
            while True:
                ret, data, page_req_key = quote_ctx.request_history_kline(
                    code,
                    start=start,
                    end=end,
                    ktype=KLType.K_DAY,
                    autype=_futu_autype(adjust),
                    max_count=1000,
                    page_req_key=page_req_key,
                )
                if ret != RET_OK:
                    raise RuntimeError(f"futu returned error for {code}: {data}")
                if not data.empty:
                    parts.append(data.copy())
                if page_req_key is None:
                    break
            if not parts:
                raise RuntimeError(f"futu returned no rows for {code}")
            raw = pd.concat(parts, ignore_index=True)
            raw.insert(0, "symbol", code)
            raw.insert(1, "date", pd.to_datetime(raw["time_key"]).dt.date.astype(str))
            raw_frames[f"futu_history_kline_US_{symbol}"] = raw
            normalized_frames[symbol] = _normalize_ohlcv_frame(
                raw.rename(columns={"time_key": "datetime"}).assign(date=pd.to_datetime(raw["time_key"]).dt.date)
            )
    finally:
        quote_ctx.close()

    vix, vix_source = _fetch_vix_with_fallback(start=start, end=end)
    return MarketDataFrames(
        provider="futu",
        source_api=["futu.OpenQuoteContext.request_history_kline", vix_source],
        soxl=normalized_frames["SOXL"],
        soxx=normalized_frames["SOXX"],
        vix=vix,
        raw_provider_frames=raw_frames,
    )


def fetch_market_frames(
    *,
    provider: str,
    start: str,
    end: str,
    futu_host: str | None = None,
    futu_port: int | None = None,
    futu_adjust: str = "qfq",
) -> MarketDataFrames:
    provider = provider.lower()
    if provider == "futu":
        return _fetch_futu_market_frames(
            start=start,
            end=end,
            host=futu_host or os.environ.get("FUTU_HOST", "127.0.0.1"),
            port=int(futu_port or os.environ.get("FUTU_PORT", "11111")),
            adjust=futu_adjust,
        )
    if provider == "yahoo_api":
        return MarketDataFrames(
            provider="yahoo_api",
            source_api=["query1.finance.yahoo.com/v8/finance/chart"],
            soxl=_fetch_yahoo_chart_ohlcv("SOXL", start=start, end=end),
            soxx=_fetch_yahoo_chart_ohlcv("SOXX", start=start, end=end),
            vix=_fetch_yahoo_chart_vix(start=start, end=end),
        )
    if provider == "polygon":
        vix, vix_source = _fetch_vix_with_fallback(start=start, end=end)
        return MarketDataFrames(
            provider="polygon",
            source_api=["polygon.io/v2/aggs", vix_source],
            soxl=_fetch_polygon_ohlcv("SOXL", start=start, end=end),
            soxx=_fetch_polygon_ohlcv("SOXX", start=start, end=end),
            vix=vix,
        )
    if provider == "yfinance":
        vix, vix_source = _fetch_vix_with_fallback(start=start, end=end)
        return MarketDataFrames(
            provider="yfinance",
            source_api=[
                "yfinance.Ticker.history",
                "query2/query1.finance.yahoo.com/v8/finance/chart",
                "polygon.io/v2/aggs",
                vix_source,
            ],
            soxl=_fetch_yfinance_compatible_ohlcv("SOXL", start=start, end=end),
            soxx=_fetch_yfinance_compatible_ohlcv("SOXX", start=start, end=end),
            vix=vix,
        )
    raise ValueError(f"unknown market data provider: {provider}")


def _choose_provider_frames(
    *,
    provider: str,
    provider_sequence: Sequence[str],
    start: str,
    end: str,
    old_last_date: str | None,
    fetch_kwargs: dict[str, Any],
) -> tuple[MarketDataFrames, pd.DataFrame, list[dict[str, Any]]]:
    candidates = list(provider_sequence if provider == "auto" else [provider])
    attempts: list[dict[str, Any]] = []
    first_success: tuple[MarketDataFrames, pd.DataFrame] | None = None
    freshest_success: tuple[MarketDataFrames, pd.DataFrame] | None = None
    freshest_last_date = ""
    for candidate in candidates:
        started = time.monotonic()
        try:
            frames = fetch_market_frames(provider=candidate, start=start, end=end, **fetch_kwargs)
            fetched_panel = combine_market_frames(frames)
            fetched_last_date = str(fetched_panel["date"].max()) if not fetched_panel.empty else None
            new_rows = (
                int((fetched_panel["date"].astype(str) > old_last_date).sum())
                if old_last_date and not fetched_panel.empty
                else int(len(fetched_panel))
            )
            attempts.append(
                {
                    "provider": candidate,
                    "status": "ok",
                    "rows": int(len(fetched_panel)),
                    "last_date": fetched_last_date,
                    "new_rows_after_existing": new_rows,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "source_api": frames.source_api,
                }
            )
            if first_success is None:
                first_success = (frames, fetched_panel)
            if fetched_last_date and fetched_last_date > freshest_last_date:
                freshest_success = (frames, fetched_panel)
                freshest_last_date = fetched_last_date
            if not old_last_date or (fetched_last_date and fetched_last_date > old_last_date):
                return frames, fetched_panel, attempts
        except Exception as exc:
            attempts.append(
                {
                    "provider": candidate,
                    "status": "error",
                    "error": str(exc),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
    if provider == "auto" and freshest_success is not None:
        return freshest_success[0], freshest_success[1], attempts
    if first_success is not None:
        return first_success[0], first_success[1], attempts
    raise RuntimeError(f"all market data providers failed: {attempts}")


def _provider_raw_append_rows(existing_path: Path, update: pd.DataFrame) -> pd.DataFrame:
    update = update.copy()
    if "date" in update.columns:
        update = update.sort_values("date")
    if not existing_path.exists():
        return update.reset_index(drop=True)

    keys = [key for key in ["code", "time_key"] if key in update.columns]
    if not keys and "date" in update.columns:
        keys = ["date"]
    if not keys:
        return update.iloc[0:0].copy()

    existing = pd.read_csv(existing_path, usecols=lambda column: column in set(keys))
    existing_keys = set(map(tuple, existing.loc[:, keys].astype(str).itertuples(index=False, name=None)))
    update_keys = update.loc[:, keys].astype(str).apply(tuple, axis=1)
    append_rows = update.loc[~update_keys.isin(existing_keys)]
    return append_rows.reset_index(drop=True)


def sync_provider_raw_frames(frames: MarketDataFrames, *, paths: MarketDataUpdatePaths) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for name, raw_frame in frames.raw_provider_frames.items():
        path = paths.raw_external_dir / f"{name}.csv"
        append_rows = _provider_raw_append_rows(path, raw_frame)
        if path.exists():
            if not append_rows.empty:
                append_rows.to_csv(path, mode="a", header=False, index=False)
        else:
            _write_csv(append_rows, path)
        written[name] = path
    return written


def _csv_summary(path: Path, *, relative_to: Path = REPO_ROOT) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "exists": path.exists(),
        "path": path.relative_to(relative_to).as_posix() if path.exists() else path.as_posix(),
    }
    if not path.exists():
        return summary | {"rows": 0, "start": None, "end": None}
    frame = pd.read_csv(path)
    summary["rows"] = int(len(frame))
    date_col = next((column for column in ["date", "datetime", "time_key"] if column in frame.columns), None)
    if date_col is None or frame.empty:
        summary["start"] = None
        summary["end"] = None
        return summary
    dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
    summary["start"] = dates.min().date().isoformat() if not dates.empty else None
    summary["end"] = dates.max().date().isoformat() if not dates.empty else None
    return summary


def update_dataset_registry(*, paths: MarketDataUpdatePaths, provider_frames: MarketDataFrames | None = None) -> dict[str, Any]:
    import yaml

    registry: dict[str, Any]
    if paths.registry_yaml.exists():
        registry = yaml.safe_load(paths.registry_yaml.read_text(encoding="utf-8")) or {}
    else:
        registry = {}
    registry["generated_at"] = datetime.now(timezone.utc).isoformat()
    registry.setdefault("processed_panels", {})["soxl_soxx_vix_full"] = _csv_summary(paths.panel_csv)
    registry.setdefault("symbols", {})["SOXL"] = _csv_summary(paths.raw_symbol_dir / "SOXL.csv")
    registry.setdefault("symbols", {})["SOXX"] = _csv_summary(paths.raw_symbol_dir / "SOXX.csv")
    registry.setdefault("external", {})["VIX"] = _csv_summary(paths.raw_external_dir / "VIX.csv")
    if provider_frames is not None:
        for name in provider_frames.raw_provider_frames:
            registry.setdefault("external", {})[name] = _csv_summary(paths.raw_external_dir / f"{name}.csv")
    provider_uri = paths.qlib_provider_uri
    registry.setdefault("qlib", {})["SOXL"] = {
        "exists": provider_uri.exists(),
        "has_calendar": (provider_uri / "calendars" / "day.txt").exists(),
        "has_instruments": (provider_uri / "instruments" / "all.txt").exists(),
        "path": provider_uri.relative_to(REPO_ROOT).as_posix() if provider_uri.exists() else provider_uri.as_posix(),
    }
    paths.registry_yaml.parent.mkdir(parents=True, exist_ok=True)
    paths.registry_yaml.write_text(yaml.safe_dump(registry, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return {
        "registry_path": paths.registry_yaml.resolve().as_posix(),
        "generated_at": registry["generated_at"],
        "updated_keys": [
            "processed_panels.soxl_soxx_vix_full",
            "symbols.SOXL",
            "symbols.SOXX",
            "external.VIX",
            "qlib.SOXL",
        ],
    }


def update_market_data(
    *,
    paths: MarketDataUpdatePaths | None = None,
    provider: str = "auto",
    provider_sequence: Sequence[str] = DEFAULT_PROVIDER_SEQUENCE,
    start: str | None = None,
    end: str | None = None,
    refresh_overlap_days: int = 10,
    replace_overlap: bool = False,
    local_only: bool = False,
    skip_qlib: bool = False,
    skip_registry: bool = False,
    dry_run: bool = False,
    futu_host: str | None = None,
    futu_port: int | None = None,
    futu_adjust: str = "qfq",
) -> dict[str, Any]:
    paths = paths or MarketDataUpdatePaths()
    existing = _read_panel(paths.panel_csv)
    old_last_date = str(existing["date"].max()) if not existing.empty else None
    start = start or _fetch_start_from_existing(existing, overlap_days=refresh_overlap_days, fallback="2010-03-11")
    end = end or _default_end_date()

    frames: MarketDataFrames | None = None
    attempts: list[dict[str, Any]] = []
    fetched_panel = pd.DataFrame(columns=PANEL_COLUMNS)
    appended_rows = 0
    replaced_rows = 0
    result_provider = "local"

    if local_only:
        if existing.empty:
            raise ValueError("local-only market data update requires an existing canonical panel")
        updated_panel = existing
    else:
        frames, fetched_panel, attempts = _choose_provider_frames(
            provider=provider,
            provider_sequence=provider_sequence,
            start=start,
            end=end,
            old_last_date=old_last_date,
            fetch_kwargs={
                "futu_host": futu_host,
                "futu_port": futu_port,
                "futu_adjust": futu_adjust,
            },
        )
        result_provider = frames.provider
        updated_panel, appended_rows, replaced_rows = merge_market_panel(
            existing,
            fetched_panel,
            replace_overlap=replace_overlap,
        )
        if not dry_run:
            _write_csv(updated_panel, paths.panel_csv)

    surfaces: dict[str, Any] = {}
    provider_raw: dict[str, Path] = {}
    qlib_result: dict[str, Any] | None = None
    registry_result: dict[str, Any] | None = None
    if not dry_run:
        surfaces = sync_local_data_surfaces(updated_panel, paths=paths)
        if frames is not None:
            provider_raw = sync_provider_raw_frames(frames, paths=paths)
        if not skip_qlib:
            qlib_result = refresh_qlib_store(updated_panel, paths=paths)
        if not skip_registry:
            registry_result = update_dataset_registry(paths=paths, provider_frames=frames)

    new_last_date = str(updated_panel["date"].max()) if not updated_panel.empty else None
    return {
        "operation": "market_data_update",
        "provider": result_provider,
        "provider_requested": provider,
        "provider_sequence": list(provider_sequence),
        "source_api": frames.source_api if frames is not None else ["local_csv"],
        "attempts": attempts,
        "start": start,
        "end": end,
        "old_last_date": old_last_date,
        "new_last_date": new_last_date,
        "fetched_rows": int(len(fetched_panel)),
        "total_rows": int(len(updated_panel)),
        "appended_rows": appended_rows,
        "replaced_overlap_rows": replaced_rows,
        "replace_overlap": replace_overlap,
        "local_only": local_only,
        "dry_run": dry_run,
        "panel_csv": paths.panel_csv.resolve().as_posix(),
        "surfaces": {
            "raw_symbols": {key: value.resolve().as_posix() for key, value in surfaces.get("raw_symbols", {}).items()},
            "raw_external": {key: value.resolve().as_posix() for key, value in surfaces.get("raw_external", {}).items()},
            "provider_raw": {key: value.resolve().as_posix() for key, value in provider_raw.items()},
        },
        "qlib": qlib_result,
        "registry": registry_result,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh SOXL/SOXX/VIX canonical data and qlib store.")
    parser.add_argument(
        "--provider",
        choices=["auto", "futu", "yahoo_api", "polygon", "yfinance"],
        default=os.environ.get("SOXL_DATA_PROVIDER", "auto"),
    )
    parser.add_argument(
        "--provider-sequence",
        default=",".join(DEFAULT_PROVIDER_SEQUENCE),
        help="Comma-separated providers used when --provider=auto.",
    )
    parser.add_argument("--start", help="Inclusive fetch start date. Defaults to last panel date minus overlap.")
    parser.add_argument("--end", help="Inclusive fetch end date. Defaults to today.")
    parser.add_argument("--refresh-overlap-days", type=int, default=10)
    parser.add_argument("--replace-overlap", action="store_true", help="Replace fetched overlapping dates.")
    parser.add_argument("--local-only", action="store_true", help="Only sync raw and qlib surfaces from local panel.")
    parser.add_argument("--skip-qlib", action="store_true", help="Do not rebuild data/qlib/soxl.")
    parser.add_argument("--skip-registry", action="store_true", help="Do not update data/datasets.yaml.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--futu-host", default=os.environ.get("FUTU_HOST"))
    parser.add_argument("--futu-port", type=int, default=int(os.environ.get("FUTU_PORT", "11111")))
    parser.add_argument("--futu-adjust", default=os.environ.get("FUTU_ADJUST", "qfq"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    provider_sequence = [provider.strip() for provider in args.provider_sequence.split(",") if provider.strip()]
    result = update_market_data(
        provider=args.provider,
        provider_sequence=provider_sequence,
        start=args.start,
        end=args.end,
        refresh_overlap_days=args.refresh_overlap_days,
        replace_overlap=args.replace_overlap,
        local_only=args.local_only,
        skip_qlib=args.skip_qlib,
        skip_registry=args.skip_registry,
        dry_run=args.dry_run,
        futu_host=args.futu_host,
        futu_port=args.futu_port,
        futu_adjust=args.futu_adjust,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

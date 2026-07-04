from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


NEW_YORK_TZ = ZoneInfo("America/New_York")

BASE_INTRADAY_FEATURES = [
    "BAR_COUNT_1M",
    "OPEN_RANGE_RETURN_30M",
    "CLOSE_LOCATION",
    "REALIZED_VOL_1M",
    "VOLUME_CONCENTRATION_CLOSE_30M",
]
EXPANDED_INTRADAY_FEATURES = [
    "FIRST_HOUR_RETURN",
    "AMPLITUDE",
    "DRAWDOWN_FROM_OPEN",
    "LATE_RETURN_30M",
    "CLOSE_VWAP_DEVIATION",
]


def _as_new_york_timestamp(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize(NEW_YORK_TZ)
    return timestamps.dt.tz_convert(NEW_YORK_TZ)


def _classify_session(timestamp: pd.Timestamp) -> str:
    local = timestamp.tz_convert(NEW_YORK_TZ) if timestamp.tzinfo is not None else timestamp.tz_localize(NEW_YORK_TZ)
    minute = local.hour * 60 + local.minute
    if 9 * 60 + 30 <= minute <= 16 * 60:
        return "regular"
    if minute < 9 * 60 + 30:
        return "premarket"
    return "afterhours"


def _classify_aggregate_session(start: pd.Timestamp, end: pd.Timestamp) -> str:
    local_start = start.tz_convert(NEW_YORK_TZ) if start.tzinfo is not None else start.tz_localize(NEW_YORK_TZ)
    local_end = end.tz_convert(NEW_YORK_TZ) if end.tzinfo is not None else end.tz_localize(NEW_YORK_TZ)
    start_minute = local_start.hour * 60 + local_start.minute
    end_minute = local_end.hour * 60 + local_end.minute
    regular_start = 9 * 60 + 30
    regular_end = 16 * 60
    if start_minute >= regular_start and end_minute <= regular_end:
        return "regular"
    if end_minute <= regular_start:
        return "premarket"
    return "afterhours"


def normalize_futu_intraday_frame(
    raw: pd.DataFrame,
    *,
    symbol: str,
    adjusted: bool = True,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    required = {"time_key", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"futu intraday frame missing columns: {missing}")

    timestamps = _as_new_york_timestamp(raw["time_key"])
    normalized = pd.DataFrame(
        {
            "provider": "futu",
            "symbol": symbol.upper(),
            "interval": "1m",
            "timestamp": timestamps,
            "date": timestamps.dt.date.astype(str),
            "session": [_classify_session(ts) for ts in timestamps],
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw["volume"], errors="coerce"),
            "turnover": pd.to_numeric(raw["turnover"], errors="coerce") if "turnover" in raw else np.nan,
            "source_tz": "America/New_York",
            "adjusted": bool(adjusted),
            "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
        }
    )
    normalized = normalized.dropna(subset=["timestamp", "open", "high", "low", "close"])
    normalized = normalized.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")
    return normalized.reset_index(drop=True)


def normalize_polygon_aggregate_frame(
    raw: pd.DataFrame | list[dict[str, Any]],
    *,
    symbol: str,
    multiplier: int = 15,
    adjusted: bool = True,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    """Normalize Polygon aggregate bars into the repo canonical intraday schema.

    Polygon aggregate timestamps are the start of the aggregate window. The
    qlib intraday stores in this repo use right-edge timestamps, so a 09:30
    15-minute aggregate is stored at 09:45.
    """

    frame = pd.DataFrame(raw).copy()
    required = {"t", "o", "h", "l", "c", "v"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"polygon aggregate frame missing columns: {missing}")

    start_timestamps = pd.to_datetime(frame["t"], unit="ms", utc=True, errors="coerce").dt.tz_convert(NEW_YORK_TZ)
    timestamps = start_timestamps + pd.Timedelta(minutes=int(multiplier))
    volume = pd.to_numeric(frame["v"], errors="coerce")
    vwap = pd.to_numeric(frame["vw"], errors="coerce") if "vw" in frame else pd.Series(np.nan, index=frame.index)
    normalized = pd.DataFrame(
        {
            "provider": "polygon",
            "symbol": symbol.upper(),
            "interval": f"{int(multiplier)}m",
            "timestamp": timestamps,
            "date": timestamps.dt.date.astype(str),
            "session": [
                _classify_aggregate_session(start, end) for start, end in zip(start_timestamps, timestamps, strict=False)
            ],
            "open": pd.to_numeric(frame["o"], errors="coerce"),
            "high": pd.to_numeric(frame["h"], errors="coerce"),
            "low": pd.to_numeric(frame["l"], errors="coerce"),
            "close": pd.to_numeric(frame["c"], errors="coerce"),
            "volume": volume,
            "turnover": vwap * volume,
            "vwap": vwap,
            "transactions": pd.to_numeric(frame["n"], errors="coerce") if "n" in frame else np.nan,
            "source_tz": "UTC",
            "adjusted": bool(adjusted),
            "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
        }
    )
    normalized = normalized.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    normalized = normalized.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")
    return normalized.reset_index(drop=True)


def polygon_intraday_to_qlib_frame(
    frame: pd.DataFrame,
    *,
    regular_session_only: bool = True,
) -> pd.DataFrame:
    required = {
        "symbol",
        "timestamp",
        "date",
        "session",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"polygon intraday frame missing columns: {missing}")

    result = frame.copy()
    result["timestamp"] = _as_new_york_timestamp(result["timestamp"])
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["date"] = pd.to_datetime(result["date"]).dt.date.astype(str)
    if regular_session_only:
        result = result.loc[result["session"].astype(str).str.lower() == "regular"].copy()
    result = result.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    if result.empty:
        return pd.DataFrame(
            columns=[
                "instrument",
                "datetime",
                "date",
                "symbol",
                "interval",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
                "vwap",
                "transactions",
                "bar_index",
                "bar_count",
            ]
        )
    for column in ["turnover", "vwap", "transactions"]:
        if column not in result:
            result[column] = np.nan
    result = result.sort_values(["symbol", "timestamp"]).copy()
    result["interval"] = result["interval"].astype(str).str.replace(r"^(\d+)m$", r"\1min", regex=True)
    result["bar_index"] = result.groupby(["symbol", "date"]).cumcount()
    result["bar_count"] = result.groupby(["symbol", "date"])["timestamp"].transform("count")
    result["instrument"] = result["symbol"]
    result = result.rename(columns={"timestamp": "datetime"})
    columns = [
        "instrument",
        "datetime",
        "date",
        "symbol",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "vwap",
        "transactions",
        "bar_index",
        "bar_count",
    ]
    return result[columns].sort_values(["instrument", "datetime"]).reset_index(drop=True)


def fetch_polygon_aggregate_bars(
    symbol: str,
    *,
    start: str,
    end: str,
    api_key: str,
    multiplier: int = 15,
    timespan: str = "minute",
    adjusted: bool = True,
    base_url: str = "https://api.polygon.io",
    timeout: float = 30.0,
    max_pages: int = 50,
    max_retries: int = 4,
    retry_sleep_seconds: float = 12.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import requests

    symbol = symbol.upper()
    requested_audit = {
        "requested_start": start,
        "requested_end": end,
        "coverage_start": "",
        "coverage_end": "",
        "provider_history_truncated": True,
    }
    url = f"{base_url.rstrip('/')}/v2/aggs/ticker/{symbol}/range/{int(multiplier)}/{timespan}/{start}/{end}"
    params: dict[str, Any] = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    rows: list[dict[str, Any]] = []
    page_count = 0
    last_status: int | None = None
    message = ""
    current_url: str | None = url
    current_params: dict[str, Any] | None = params

    while current_url and page_count < max_pages:
        attempts = 0
        while True:
            try:
                response = requests.get(current_url, params=current_params, timeout=timeout)
            except requests.exceptions.RequestException as exc:
                return pd.DataFrame(), {
                    "symbol": symbol,
                    "status": "blocked_provider_timeout",
                    "http_status": 0,
                    "row_count": 0,
                    "page_count": page_count,
                    "message": str(exc),
                    **requested_audit,
                }
            last_status = int(response.status_code)
            if response.status_code != 429 or attempts >= max_retries:
                break
            attempts += 1
            retry_after = response.headers.get("Retry-After")
            sleep_seconds = float(retry_after) if retry_after else retry_sleep_seconds
            time.sleep(sleep_seconds)
        try:
            payload = response.json()
        except ValueError:
            payload = {"status": "ERROR", "message": response.text[:240]}
        message = str(payload.get("message") or payload.get("status") or "")
        if response.status_code != 200:
            if rows:
                status = (
                    "ok_partial_provider_rate_limit" if response.status_code == 429 else "ok_partial_provider_history"
                )
                return _polygon_rows_with_audit(
                    rows,
                    symbol=symbol,
                    multiplier=multiplier,
                    adjusted=adjusted,
                    status=status,
                    http_status=last_status,
                    page_count=page_count,
                    message=message,
                    start=start,
                    end=end,
                )
            blocked_status = "blocked_provider_rate_limit" if response.status_code == 429 else "blocked_provider_history"
            return pd.DataFrame(), {
                "symbol": symbol,
                "status": blocked_status,
                "http_status": last_status,
                "row_count": 0,
                "page_count": page_count,
                "message": message,
                **requested_audit,
            }
        page_rows = payload.get("results") or []
        rows.extend(page_rows)
        page_count += 1
        next_url = payload.get("next_url")
        if not next_url:
            current_url = None
        else:
            current_url = str(next_url)
            current_params = {"apiKey": api_key}

    return _polygon_rows_with_audit(
        rows,
        symbol=symbol,
        multiplier=multiplier,
        adjusted=adjusted,
        status="ok" if rows else "blocked_no_rows",
        http_status=last_status,
        page_count=page_count,
        message=message,
        start=start,
        end=end,
    )


def _polygon_rows_with_audit(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    multiplier: int,
    adjusted: bool,
    status: str,
    http_status: int | None,
    page_count: int,
    message: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized = (
        normalize_polygon_aggregate_frame(
            rows,
            symbol=symbol,
            multiplier=multiplier,
            adjusted=adjusted,
        )
        if rows
        else pd.DataFrame()
    )
    if normalized.empty and status.startswith("ok"):
        status = "blocked_no_rows"
    coverage_start = _format_timestamp_for_audit(normalized["timestamp"].min()) if not normalized.empty else ""
    coverage_end = _format_timestamp_for_audit(normalized["timestamp"].max()) if not normalized.empty else ""
    provider_history_truncated = True
    if not normalized.empty:
        provider_history_truncated = bool(
            pd.Timestamp(normalized["timestamp"].min()).date() > pd.Timestamp(start).date()
            or pd.Timestamp(normalized["timestamp"].max()).date() < pd.Timestamp(end).date()
        )
    return normalized, {
        "symbol": symbol,
        "status": status,
        "http_status": http_status,
        "row_count": int(len(normalized)),
        "page_count": page_count,
        "message": message,
        "requested_start": start,
        "requested_end": end,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "provider_history_truncated": provider_history_truncated,
    }


def _format_timestamp_for_audit(value: Any) -> str:
    if pd.isna(value):
        return ""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def merge_intraday_bars(existing: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = fetched.copy()
    elif fetched.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, fetched], ignore_index=True)
    if merged.empty:
        return merged
    merged["timestamp"] = _as_new_york_timestamp(merged["timestamp"])
    merged["symbol"] = merged["symbol"].astype(str).str.upper()
    merged = merged.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")
    return merged.reset_index(drop=True)


def fetch_futu_intraday_bars(
    symbol: str,
    *,
    start: str,
    end: str,
    host: str = "127.0.0.1",
    port: int = 11111,
    adjust: str = "qfq",
    extended_time: bool = False,
    session: str = "N/A",
) -> pd.DataFrame:
    from futu import AuType, KLType, OpenQuoteContext, RET_OK

    adjust_name = adjust.lower()
    if adjust_name in {"qfq", "forward"}:
        autype = AuType.QFQ
    elif adjust_name in {"hfq", "backward"}:
        autype = AuType.HFQ
    else:
        autype = AuType.NONE

    code = f"US.{symbol.upper()}"
    quote_ctx = OpenQuoteContext(host=host, port=port)
    parts: list[pd.DataFrame] = []
    page_req_key = None
    try:
        while True:
            ret, data, page_req_key = quote_ctx.request_history_kline(
                code,
                start=start,
                end=end,
                ktype=KLType.K_1M,
                autype=autype,
                max_count=1000,
                page_req_key=page_req_key,
                extended_time=extended_time,
                session=session,
            )
            if ret != RET_OK:
                raise RuntimeError(f"futu intraday returned error for {code}: {data}")
            if not data.empty:
                parts.append(data.copy())
            if page_req_key is None:
                break
    finally:
        quote_ctx.close()
    if not parts:
        raise RuntimeError(f"futu intraday returned no rows for {code}")
    return normalize_futu_intraday_frame(
        pd.concat(parts, ignore_index=True),
        symbol=symbol,
        adjusted=adjust_name != "none",
    )


def read_intraday_cache(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if not frame.empty and "timestamp" in frame:
        frame["timestamp"] = _as_new_york_timestamp(frame["timestamp"])
    return frame


def write_intraday_cache(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def resample_intraday_bars(
    minute_frame: pd.DataFrame,
    *,
    freq: str,
    symbols: Sequence[str] | None = None,
    regular_session_only: bool = True,
) -> pd.DataFrame:
    required = {"symbol", "timestamp", "date", "session", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(minute_frame.columns))
    if missing:
        raise ValueError(f"intraday minute frame missing columns: {missing}")

    frame = minute_frame.copy()
    frame["timestamp"] = _as_new_york_timestamp(frame["timestamp"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if symbols is not None:
        wanted = {symbol.upper() for symbol in symbols}
        frame = frame.loc[frame["symbol"].isin(wanted)].copy()
    if regular_session_only:
        frame = frame.loc[frame["session"].astype(str).str.lower() == "regular"].copy()
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "instrument",
                "datetime",
                "date",
                "symbol",
                "interval",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
                "bar_index",
                "bar_count",
            ]
        )

    rows: list[pd.DataFrame] = []
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "turnover" in frame.columns:
        aggregations["turnover"] = "sum"
    for (symbol, session_date), bars in frame.sort_values("timestamp").groupby(["symbol", "date"], sort=True):
        resampled = (
            bars.set_index("timestamp")
            .resample(freq, label="right", closed="right", origin="start_day")
            .agg(aggregations)
            .dropna(subset=["open", "high", "low", "close"])
        )
        if resampled.empty:
            continue
        if "turnover" not in resampled.columns:
            resampled["turnover"] = np.nan
        resampled = resampled.reset_index().rename(columns={"timestamp": "datetime"})
        resampled["instrument"] = symbol
        resampled["symbol"] = symbol
        resampled["date"] = session_date
        resampled["interval"] = freq
        resampled["bar_index"] = np.arange(len(resampled), dtype=int)
        resampled["bar_count"] = len(resampled)
        rows.append(resampled)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    columns = [
        "instrument",
        "datetime",
        "date",
        "symbol",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "bar_index",
        "bar_count",
    ]
    return result[columns].sort_values(["instrument", "datetime"]).reset_index(drop=True)


def _safe_ratio(numerator: float, denominator: float, default: float = np.nan) -> float:
    if denominator == 0 or pd.isna(denominator):
        return default
    return float(numerator / denominator)


def _next_calendar_date(session_date: pd.Timestamp, calendar: list[pd.Timestamp]) -> str | None:
    for candidate in calendar:
        if candidate > session_date:
            return candidate.date().isoformat()
    return None


def build_t1_intraday_feature_frame(
    minute_frame: pd.DataFrame,
    *,
    calendar: Sequence[str | pd.Timestamp],
    symbol: str = "SOXL",
    regular_session_only: bool = True,
    open_minutes: int = 30,
    close_minutes: int = 30,
) -> pd.DataFrame:
    required = {"symbol", "timestamp", "date", "session", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(minute_frame.columns))
    if missing:
        raise ValueError(f"intraday minute frame missing columns: {missing}")

    calendar_dates = sorted(pd.to_datetime(pd.Series(list(calendar))).dt.normalize().dropna().unique())
    calendar_ts = [pd.Timestamp(value) for value in calendar_dates]
    frame = minute_frame.copy()
    frame["timestamp"] = _as_new_york_timestamp(frame["timestamp"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame.loc[frame["symbol"] == symbol.upper()].copy()
    if regular_session_only:
        frame = frame.loc[frame["session"].astype(str).str.lower() == "regular"].copy()
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    rows: list[dict[str, float | str]] = []
    prefix = f"INTRADAY_{symbol.upper()}"
    for session_date, bars in frame.sort_values("timestamp").groupby("date"):
        feature_date = _next_calendar_date(pd.Timestamp(session_date), calendar_ts)
        if feature_date is None:
            continue
        bars = bars.sort_values("timestamp").copy()
        first_open = float(bars["open"].iloc[0])
        last_close = float(bars["close"].iloc[-1])
        day_high = float(bars["high"].max())
        day_low = float(bars["low"].min())
        total_volume = float(bars["volume"].sum())
        first_timestamp = bars["timestamp"].iloc[0]
        last_timestamp = bars["timestamp"].iloc[-1]
        open_window = bars.loc[bars["timestamp"] <= first_timestamp + pd.Timedelta(minutes=open_minutes - 1)]
        first_hour_window = bars.loc[bars["timestamp"] <= first_timestamp + pd.Timedelta(minutes=59)]
        close_window = bars.loc[bars["timestamp"] >= last_timestamp - pd.Timedelta(minutes=close_minutes - 1)]
        if open_window.empty:
            open_window = bars.head(1)
        if first_hour_window.empty:
            first_hour_window = bars.head(1)
        if close_window.empty:
            close_window = bars.tail(1)
        returns = bars["close"].astype(float).pct_change().dropna()
        if "turnover" in bars and pd.to_numeric(bars["turnover"], errors="coerce").notna().any():
            turnover = pd.to_numeric(bars["turnover"], errors="coerce").fillna(0.0)
            vwap = _safe_ratio(float(turnover.sum()), total_volume)
        else:
            vwap = _safe_ratio(float((bars["close"].astype(float) * bars["volume"].astype(float)).sum()), total_volume)
        close_location = _safe_ratio(last_close - day_low, day_high - day_low, default=0.5)
        rows.append(
            {
                "date": feature_date,
                f"{prefix}_BAR_COUNT_1M": int(len(bars)),
                f"{prefix}_OPEN_RANGE_RETURN_30M": _safe_ratio(float(open_window["close"].iloc[-1]), first_open) - 1.0,
                f"{prefix}_CLOSE_LOCATION": close_location,
                f"{prefix}_REALIZED_VOL_1M": float(returns.std(ddof=0)) if len(returns) else 0.0,
                f"{prefix}_VOLUME_CONCENTRATION_CLOSE_30M": _safe_ratio(
                    float(close_window["volume"].sum()), total_volume, default=0.0
                ),
                f"{prefix}_FIRST_HOUR_RETURN": _safe_ratio(float(first_hour_window["close"].iloc[-1]), first_open) - 1.0,
                f"{prefix}_AMPLITUDE": _safe_ratio(day_high, day_low) - 1.0,
                f"{prefix}_DRAWDOWN_FROM_OPEN": _safe_ratio(day_low, first_open) - 1.0,
                f"{prefix}_LATE_RETURN_30M": _safe_ratio(last_close, float(close_window["close"].iloc[0])) - 1.0,
                f"{prefix}_CLOSE_VWAP_DEVIATION": _safe_ratio(last_close, vwap) - 1.0,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

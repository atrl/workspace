from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
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

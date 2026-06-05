from __future__ import annotations

"""Intraday OHLCV intake and daily-safe feature aggregation."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from src.soxl_quant.settings import ROOT

NEW_YORK_TZ = "America/New_York"
POLYGON_API_KEY_ENV = "POLYGON_API_KEY"

CANONICAL_INTRADAY_COLUMNS = [
    "provider",
    "symbol",
    "interval",
    "timestamp",
    "date",
    "session",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "source_tz",
    "adjusted",
    "fetched_at",
]

INTRADAY_FEATURE_COLUMNS = [
    "date",
    "provider",
    "session",
    "interval",
    "bars_soxl",
    "bars_soxx",
    "opening_range_30",
    "close_location_value",
    "realized_intraday_vol",
    "volume_concentration_open_30",
    "soxl_soxx_intraday_relative_volume",
    "intraday_tracking_error",
]

_FUTU_KTYPE_BY_INTERVAL = {"1m": "K_1M"}
_POLYGON_RANGE_BY_INTERVAL = {"1m": (1, "minute")}


def canonical_intraday_path(*, root: Path = ROOT) -> Path:
    return root / "data" / "processed" / "intraday" / "soxl_soxx_1m.parquet"


def intraday_feature_csv_path(*, root: Path = ROOT) -> Path:
    return root / "data" / "raw" / "external" / "intraday_features_soxl_1m.csv"


def intraday_raw_path(
    provider: str,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    session: str,
    *,
    root: Path = ROOT,
) -> Path:
    safe = f"{symbol.upper()}_{interval}_{start}_{end}_{session}.parquet"
    return root / "data" / "raw" / "intraday" / provider.lower() / safe


def _empty_canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_INTRADAY_COLUMNS)


def _normalize_timestamp(values: Iterable[Any] | pd.Series, *, source_tz: str) -> pd.Series:
    timestamps = pd.to_datetime(pd.Series(values), errors="coerce")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(source_tz)
    else:
        timestamps = timestamps.dt.tz_convert(source_tz)
    return timestamps.dt.tz_convert(NEW_YORK_TZ)


def _regular_session_mask(timestamps: pd.Series) -> pd.Series:
    minute_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    return (minute_of_day >= 9 * 60 + 30) & (minute_of_day < 16 * 60)


def _filter_session(frame: pd.DataFrame, session: str) -> pd.DataFrame:
    if frame.empty or session != "regular":
        return frame
    return frame[_regular_session_mask(frame["timestamp"])].copy()


def _canonical_from_ohlcv(
    *,
    provider: str,
    raw: pd.DataFrame,
    symbol: str,
    interval: str,
    session: str,
    timestamps: pd.Series,
    adjusted: bool,
    source_tz: str,
    fetched_at: pd.Timestamp | None,
) -> pd.DataFrame:
    if raw.empty:
        return _empty_canonical_frame()
    fetched = fetched_at if fetched_at is not None else pd.Timestamp.now(tz="UTC")
    frame = pd.DataFrame(
        {
            "provider": provider,
            "symbol": symbol.upper(),
            "interval": interval,
            "timestamp": timestamps,
            "session": session,
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("volume", 0.0), errors="coerce").fillna(0.0).astype(float),
            "turnover": pd.to_numeric(raw.get("turnover", 0.0), errors="coerce").fillna(0.0).astype(float),
            "source_tz": source_tz,
            "adjusted": pd.Series([bool(adjusted)] * len(raw), dtype=object),
            "fetched_at": fetched,
        }
    )
    frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame = _filter_session(frame, session)
    frame = frame[CANONICAL_INTRADAY_COLUMNS]
    return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def normalize_futu_history_kline(
    frame: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    session: str,
    adjusted: bool = True,
    fetched_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_canonical_frame()
    raw = frame.copy()
    if "time_key" not in raw.columns:
        raise KeyError("Futu history K-line frame missing time_key")
    timestamps = _normalize_timestamp(raw["time_key"], source_tz=NEW_YORK_TZ)
    return _canonical_from_ohlcv(
        provider="futu",
        raw=raw,
        symbol=symbol,
        interval=interval,
        session=session,
        timestamps=timestamps,
        adjusted=adjusted,
        source_tz=NEW_YORK_TZ,
        fetched_at=fetched_at,
    )


def normalize_polygon_aggregates(
    payload: dict[str, Any] | pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    session: str,
    fetched_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    adjusted = True
    if isinstance(payload, pd.DataFrame):
        raw = payload.copy()
    else:
        adjusted = bool(payload.get("adjusted", True))
        raw = pd.DataFrame(payload.get("results") or [])
    if raw.empty:
        return _empty_canonical_frame()
    rename = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    raw = raw.rename(columns=rename)
    if "t" not in raw.columns:
        raise KeyError("Polygon aggregate payload missing t")
    timestamps = pd.to_datetime(raw["t"], unit="ms", utc=True).dt.tz_convert(NEW_YORK_TZ)
    raw["turnover"] = 0.0
    return _canonical_from_ohlcv(
        provider="polygon",
        raw=raw,
        symbol=symbol,
        interval=interval,
        session=session,
        timestamps=timestamps,
        adjusted=adjusted,
        source_tz="UTC",
        fetched_at=fetched_at,
    )


@dataclass
class FutuIntradayProvider:
    host: str = "127.0.0.1"
    port: int = 11111
    max_count: int = 1000
    adjusted: bool = True
    sdk: Any | None = None

    provider_name = "futu"

    def _sdk(self):
        if self.sdk is not None:
            return self.sdk
        try:
            import futu
        except Exception as exc:  # pragma: no cover - environment-dependent import.
            raise RuntimeError("futu-api not installed") from exc
        return futu

    def fetch(self, symbol: str, *, interval: str, start: str, end: str, session: str) -> pd.DataFrame:
        if interval not in _FUTU_KTYPE_BY_INTERVAL:
            raise ValueError(f"unsupported Futu intraday interval: {interval}")
        futu = self._sdk()
        quote_ctx = futu.OpenQuoteContext(host=self.host, port=self.port)
        chunks: list[pd.DataFrame] = []
        page_req_key = None
        try:
            while True:
                ret, data, page_req_key = quote_ctx.request_history_kline(
                    code=f"US.{symbol.upper()}",
                    start=start,
                    end=end,
                    ktype=getattr(futu.KLType, _FUTU_KTYPE_BY_INTERVAL[interval]),
                    autype=futu.AuType.QFQ if self.adjusted else futu.AuType.NONE,
                    fields=[futu.KL_FIELD.ALL],
                    max_count=self.max_count,
                    page_req_key=page_req_key,
                    extended_time=session == "extended",
                    session=futu.Session.RTH if session == "regular" else futu.Session.ALL,
                )
                if ret != futu.RET_OK:
                    raise RuntimeError(f"Futu history K-line failed for {symbol}: {data}")
                if isinstance(data, pd.DataFrame) and not data.empty:
                    chunks.append(data)
                if page_req_key is None:
                    break
        finally:
            close = getattr(quote_ctx, "close", None)
            if callable(close):
                close()
        raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        return normalize_futu_history_kline(
            raw,
            symbol=symbol,
            interval=interval,
            session=session,
            adjusted=self.adjusted,
        )


@dataclass
class PolygonIntradayProvider:
    api_key: str | None = None
    adjusted: bool = True
    base_url: str = "https://api.polygon.io"
    timeout: float = 60.0
    session_obj: Any | None = None

    provider_name = "polygon"

    def _api_key(self) -> str:
        key = (self.api_key or os.environ.get(POLYGON_API_KEY_ENV) or "").strip()
        if not key:
            raise RuntimeError(f"{POLYGON_API_KEY_ENV} environment variable not set")
        return key

    def fetch(self, symbol: str, *, interval: str, start: str, end: str, session: str) -> pd.DataFrame:
        if interval not in _POLYGON_RANGE_BY_INTERVAL:
            raise ValueError(f"unsupported Polygon intraday interval: {interval}")
        api_key = self._api_key()
        multiplier, timespan = _POLYGON_RANGE_BY_INTERVAL[interval]
        url = f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{start}/{end}"
        client = self.session_obj or requests.Session()
        rows: list[dict[str, Any]] = []
        adjusted = self.adjusted
        first = True
        while url:
            if first:
                params = {
                    "adjusted": str(self.adjusted).lower(),
                    "sort": "asc",
                    "limit": 50000,
                    "apiKey": api_key,
                }
            else:
                params = None if "apiKey=" in url else {"apiKey": api_key}
            response = client.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            rows.extend(payload.get("results") or [])
            adjusted = bool(payload.get("adjusted", adjusted))
            url = payload.get("next_url")
            first = False
        return normalize_polygon_aggregates(
            {"ticker": symbol.upper(), "adjusted": adjusted, "results": rows},
            symbol=symbol,
            interval=interval,
            session=session,
        )


def build_daily_intraday_features(
    canonical: pd.DataFrame,
    *,
    provider_order: tuple[str, ...] = ("polygon", "futu"),
    interval: str = "1m",
    session: str = "regular",
) -> pd.DataFrame:
    if canonical.empty:
        return pd.DataFrame(columns=INTRADAY_FEATURE_COLUMNS)
    frame = canonical.copy()
    frame = frame[(frame["interval"] == interval) & (frame["session"] == session)]
    if frame.empty:
        return pd.DataFrame(columns=INTRADAY_FEATURE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for date, full_day in frame.groupby("date", sort=True):
        day = pd.DataFrame()
        for provider in provider_order:
            candidate = full_day[full_day["provider"] == provider]
            if not candidate[candidate["symbol"] == "SOXL"].empty:
                day = candidate
                break
        if day.empty:
            day = full_day
        soxl = day[day["symbol"] == "SOXL"].sort_values("timestamp")
        soxx = day[day["symbol"] == "SOXX"].sort_values("timestamp")
        if soxl.empty:
            continue
        minute_of_day = soxl["timestamp"].dt.hour * 60 + soxl["timestamp"].dt.minute
        open_30 = soxl[(minute_of_day >= 9 * 60 + 30) & (minute_of_day < 10 * 60)]
        if open_30.empty:
            open_30 = soxl.head(min(len(soxl), 30))
        high = float(soxl["high"].max())
        low = float(soxl["low"].min())
        last_close = float(soxl["close"].iloc[-1])
        first_open = float(soxl["open"].iloc[0])
        total_volume = float(soxl["volume"].sum())
        opening_range = float((open_30["high"].max() - open_30["low"].min()) / (first_open + 1e-8))
        close_location = float((last_close - low) / (high - low + 1e-8))
        log_ret = np.log(soxl["close"].astype(float)).diff().dropna()
        realized_vol = float(np.sqrt(np.square(log_ret).sum()))
        volume_conc = float(open_30["volume"].sum() / (total_volume + 1e-8))
        soxl_return = float(last_close / first_open - 1.0)
        if soxx.empty:
            soxx_volume = 0.0
            soxx_return = 0.0
        else:
            soxx_volume = float(soxx["volume"].sum())
            soxx_return = float(soxx["close"].iloc[-1] / (soxx["open"].iloc[0] + 1e-8) - 1.0)
        rows.append(
            {
                "date": str(date),
                "provider": str(soxl["provider"].iloc[0]),
                "session": session,
                "interval": interval,
                "bars_soxl": int(len(soxl)),
                "bars_soxx": int(len(soxx)),
                "opening_range_30": opening_range,
                "close_location_value": close_location,
                "realized_intraday_vol": realized_vol,
                "volume_concentration_open_30": volume_conc,
                "soxl_soxx_intraday_relative_volume": float(total_volume / (soxx_volume + 1e-8)) if soxx_volume else 0.0,
                "intraday_tracking_error": soxl_return - 3.0 * soxx_return,
            }
        )
    return pd.DataFrame(rows, columns=INTRADAY_FEATURE_COLUMNS)


def write_intraday_outputs(
    frames: dict[tuple[str, str], pd.DataFrame],
    *,
    interval: str,
    start: str,
    end: str,
    session: str,
    root: Path = ROOT,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    canonical_frames: list[pd.DataFrame] = []
    for (provider, symbol), frame in sorted(frames.items()):
        path = intraday_raw_path(provider, symbol, interval, start, end, session, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        paths[f"raw:{provider}:{symbol}"] = path
        canonical_frames.append(frame)
    canonical = pd.concat(canonical_frames, ignore_index=True) if canonical_frames else _empty_canonical_frame()
    canonical = canonical.sort_values(["provider", "symbol", "timestamp"]).reset_index(drop=True)
    canonical_path = canonical_intraday_path(root=root)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(canonical_path, index=False)
    paths["canonical"] = canonical_path

    features = build_daily_intraday_features(canonical, interval=interval, session=session)
    feature_path = intraday_feature_csv_path(root=root)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_path, index=False)
    paths["features"] = feature_path
    return paths


__all__ = [
    "CANONICAL_INTRADAY_COLUMNS",
    "INTRADAY_FEATURE_COLUMNS",
    "FutuIntradayProvider",
    "PolygonIntradayProvider",
    "build_daily_intraday_features",
    "canonical_intraday_path",
    "intraday_feature_csv_path",
    "intraday_raw_path",
    "normalize_futu_history_kline",
    "normalize_polygon_aggregates",
    "write_intraday_outputs",
]

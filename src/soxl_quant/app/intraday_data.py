from __future__ import annotations

"""Intraday SOXL/SOXX data intake and normalization utilities."""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

from src.soxl_quant.app.futu import call_with_connect_timeout, check_opend_port

NEW_YORK_TZ = "America/New_York"
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
POLYGON_API_KEY_ENV = "POLYGON_API_KEY"


def _interval_to_polygon(interval: str) -> tuple[int, str]:
    if interval == "1m":
        return 1, "minute"
    if interval.endswith("m") and interval[:-1].isdigit():
        return int(interval[:-1]), "minute"
    if interval.endswith("h") and interval[:-1].isdigit():
        return int(interval[:-1]), "hour"
    raise ValueError(f"Unsupported intraday interval: {interval}")


def _session_for_timestamp(timestamp: pd.Timestamp) -> str:
    local = timestamp.tz_convert(NEW_YORK_TZ)
    hhmm = local.hour * 100 + local.minute
    if 930 <= hhmm <= 1600:
        return "regular"
    return "extended"


def _prepare_futu_log_home(log_home: Path | str | None) -> str | None:
    if log_home is None:
        return os.environ.get("HOME")
    previous = os.environ.get("HOME")
    path = Path(log_home)
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(path)
    return previous


def _import_futu_sdk(log_home: Path | str | None = None):
    _prepare_futu_log_home(log_home)
    try:
        import futu
    except ModuleNotFoundError as exc:
        raise RuntimeError("futu-api not installed") from exc
    except Exception as exc:
        raise RuntimeError(f"failed to import futu-api: {type(exc).__name__}: {exc}") from exc
    return futu


def normalize_intraday_bars(
    frame: pd.DataFrame,
    *,
    provider: str,
    symbol: str,
    interval: str,
    timestamp_column: str,
    source_tz: str = NEW_YORK_TZ,
    adjusted: bool = True,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_INTRADAY_COLUMNS)

    result = frame.copy()
    timestamps = pd.to_datetime(result[timestamp_column], errors="coerce")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(source_tz)
    else:
        timestamps = timestamps.dt.tz_convert(NEW_YORK_TZ)
    timestamps = timestamps.dt.tz_convert(NEW_YORK_TZ)
    fetched = fetched_at or datetime.now(timezone.utc).isoformat()
    turnover = result["turnover"] if "turnover" in result.columns else pd.Series(0.0, index=result.index)

    normalized = pd.DataFrame(
        {
            "provider": provider,
            "symbol": symbol.upper().replace("US.", ""),
            "interval": interval,
            "timestamp": timestamps,
            "open": pd.to_numeric(result.get("open"), errors="coerce"),
            "high": pd.to_numeric(result.get("high"), errors="coerce"),
            "low": pd.to_numeric(result.get("low"), errors="coerce"),
            "close": pd.to_numeric(result.get("close"), errors="coerce"),
            "volume": pd.to_numeric(result.get("volume"), errors="coerce").fillna(0.0),
            "turnover": pd.to_numeric(turnover, errors="coerce").fillna(0.0),
            "source_tz": source_tz,
            "adjusted": bool(adjusted),
            "fetched_at": fetched,
        }
    ).dropna(subset=["timestamp", "open", "high", "low", "close"])
    normalized["date"] = normalized["timestamp"].dt.strftime("%Y-%m-%d")
    normalized["session"] = normalized["timestamp"].map(_session_for_timestamp)
    normalized = normalized.sort_values(["symbol", "timestamp", "provider"]).reset_index(drop=True)
    return normalized[CANONICAL_INTRADAY_COLUMNS]


@dataclass(frozen=True)
class PolygonIntradayProvider:
    api_key_env: str = POLYGON_API_KEY_ENV

    def fetch(self, symbol: str, *, start: str, end: str, interval: str = "1m") -> pd.DataFrame:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} environment variable not set")
        multiplier, timespan = _interval_to_polygon(interval)
        ticker = symbol.upper().replace("US.", "")
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
        response = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key},
            timeout=60,
        )
        response.raise_for_status()
        rows = response.json().get("results") or []
        if not rows:
            return pd.DataFrame(columns=CANONICAL_INTRADAY_COLUMNS)
        raw = pd.DataFrame(rows)
        converted = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(raw["t"], unit="ms", utc=True),
                "open": raw["o"],
                "high": raw["h"],
                "low": raw["l"],
                "close": raw["c"],
                "volume": raw["v"],
                "turnover": raw.get("vw", 0.0),
            }
        )
        return normalize_intraday_bars(
            converted,
            provider="polygon",
            symbol=ticker,
            interval=interval,
            timestamp_column="timestamp",
            source_tz="UTC",
            adjusted=True,
        )


@dataclass(frozen=True)
class FutuIntradayProvider:
    host: str = "127.0.0.1"
    port: int = 11111
    log_home: Path | None = None
    connect_timeout_sec: float = 20.0

    def fetch(self, symbol: str, *, start: str, end: str, interval: str = "1m") -> pd.DataFrame:
        if not check_opend_port(self.host, self.port, timeout=min(float(self.connect_timeout_sec), 5.0)):
            raise RuntimeError(f"Futu OpenD is not reachable at {self.host}:{self.port}")
        futu = _import_futu_sdk(self.log_home)
        if hasattr(futu, "SysConfig"):
            futu.SysConfig.set_all_thread_daemon(True)

        ktype = getattr(futu.KLType, "K_1M" if interval == "1m" else f"K_{interval.upper()}", None)
        if ktype is None:
            raise ValueError(f"Unsupported Futu interval: {interval}")
        code = symbol if symbol.startswith("US.") else f"US.{symbol.upper()}"
        ctx = call_with_connect_timeout(
            lambda: futu.OpenQuoteContext(host=self.host, port=self.port),
            seconds=float(self.connect_timeout_sec),
        )
        rows: list[pd.DataFrame] = []
        page_req_key = None
        try:
            while True:
                ret, data, page_req_key = call_with_connect_timeout(
                    lambda: ctx.request_history_kline(
                        code=code,
                        start=start,
                        end=end,
                        ktype=ktype,
                        max_count=1000,
                        page_req_key=page_req_key,
                    ),
                    seconds=float(self.connect_timeout_sec),
                )
                if ret != futu.RET_OK:
                    raise RuntimeError(f"request_history_kline failed: {data}")
                if data is not None and len(data) > 0:
                    rows.append(data)
                if page_req_key is None:
                    break
        finally:
            ctx.close()
        raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if raw.empty:
            return pd.DataFrame(columns=CANONICAL_INTRADAY_COLUMNS)
        timestamp_col = "time_key" if "time_key" in raw.columns else ("datetime" if "datetime" in raw.columns else raw.columns[0])
        return normalize_intraday_bars(
            raw,
            provider="futu",
            symbol=code,
            interval=interval,
            timestamp_column=timestamp_col,
            source_tz=NEW_YORK_TZ,
            adjusted=True,
        )


def filter_session(frame: pd.DataFrame, session: str) -> pd.DataFrame:
    if session == "all":
        return frame
    if session not in {"regular", "extended"}:
        raise ValueError(f"Unsupported session: {session}")
    return frame.loc[frame["session"] == session].reset_index(drop=True)


def build_canonical_intraday_panel(
    frames: Iterable[pd.DataFrame],
    *,
    provider_priority: Sequence[str] = ("futu", "polygon"),
) -> pd.DataFrame:
    combined = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=[*CANONICAL_INTRADAY_COLUMNS, "source_count", "close_disagreement_abs"])
    priority = {provider: idx for idx, provider in enumerate(provider_priority)}
    combined = combined.copy()
    combined["_priority"] = combined["provider"].map(priority).fillna(len(priority)).astype(int)
    keys = ["symbol", "interval", "timestamp"]
    disagreements = (
        combined.groupby(keys, dropna=False)["close"]
        .agg(source_count="count", close_min="min", close_max="max")
        .reset_index()
    )
    disagreements["close_disagreement_abs"] = disagreements["close_max"] - disagreements["close_min"]
    selected = combined.sort_values([*keys, "_priority"]).drop_duplicates(keys, keep="first").drop(columns=["_priority"])
    selected = selected.merge(disagreements[[*keys, "source_count", "close_disagreement_abs"]], on=keys, how="left")
    return selected.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def write_intraday_surfaces(
    raw_frames: dict[tuple[str, str], pd.DataFrame],
    *,
    root: Path,
    interval: str,
    session: str,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    paths: dict[str, str] = {}
    filtered_frames: list[pd.DataFrame] = []
    for (provider, symbol), frame in raw_frames.items():
        filtered = filter_session(frame, session)
        filtered_frames.append(filtered)
        raw_path = root / "data" / "raw" / "intraday" / provider / f"{symbol.lower()}_{interval}.parquet"
        paths[f"raw:{provider}:{symbol}"] = str(raw_path.relative_to(root))
        if not dry_run:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            filtered.to_parquet(raw_path, index=False)
    canonical = build_canonical_intraday_panel(filtered_frames)
    canonical_path = root / "data" / "processed" / "intraday" / f"soxl_soxx_{interval}.parquet"
    paths["canonical"] = str(canonical_path.relative_to(root))
    if not dry_run:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_parquet(canonical_path, index=False)
    return canonical, paths


def build_intraday_daily_features(frame: pd.DataFrame, *, output_path: Path | None = None) -> pd.DataFrame:
    if frame.empty:
        result = pd.DataFrame()
    else:
        bars = frame.loc[(frame["symbol"] == "SOXL") & (frame["session"] == "regular")].copy()
        bars["ret"] = bars.groupby("date")["close"].pct_change().fillna(0.0)
        rows = []
        for date, group in bars.groupby("date", sort=True):
            group = group.sort_values("timestamp")
            total_volume = float(group["volume"].sum())
            close_window = min(30, max(1, len(group) // 2))
            close_30_volume = float(group.tail(close_window)["volume"].sum())
            rows.append(
                {
                    "date": (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "INTRADAY_SOXL_BAR_COUNT_1M": int(len(group)),
                    "INTRADAY_SOXL_OPEN_RANGE_RETURN_30M": float(group.head(30)["close"].iloc[-1] / group["open"].iloc[0] - 1.0),
                    "INTRADAY_SOXL_CLOSE_LOCATION": float(
                        (group["close"].iloc[-1] - group["low"].min()) / (group["high"].max() - group["low"].min() + 1e-8)
                    ),
                    "INTRADAY_SOXL_REALIZED_VOL_1M": float(group["ret"].std(ddof=0) or 0.0),
                    "INTRADAY_SOXL_VOLUME_CONCENTRATION_CLOSE_30M": close_30_volume / total_volume
                    if total_volume > 0
                    else 0.0,
                }
            )
        result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result

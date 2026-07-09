from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from .market_update import _fetch_yahoo_chart_ohlcv
from .qlib_native import build_qlib_store_from_frame


HARDWARE_CYCLE_INSTRUMENT = "HARDWARE_CYCLE"
HARDWARE_CYCLE_PROVIDER_NAME = "hardware_cycle_v1"
HARDWARE_CYCLE_AIMULTIPLE_URL = "https://aimultiple.com/gpu-index"
HARDWARE_CYCLE_USER_AGENT = "Mozilla/5.0 soxl-qlib-hardware-cycle"

HARDWARE_PROXY_GROUPS: dict[str, tuple[str, ...]] = {
    "ai_accelerator": ("NVDA", "AMD", "AVGO", "TSM", "ASML"),
    "cpu_compute": ("AMD", "INTC", "QCOM", "MRVL", "ARM"),
    "memory_storage": ("MU", "WDC", "STX"),
    "enterprise_storage": ("NTAP", "PSTG", "HPE", "HPQ"),
    "cloud_capex": ("AMZN", "MSFT", "GOOGL", "ORCL", "META"),
    "ai_server": ("SMCI", "DELL"),
    "power_cooling": ("VRT", "ETN", "PWR"),
}
HARDWARE_PROXY_SYMBOLS = tuple(dict.fromkeys(symbol for symbols in HARDWARE_PROXY_GROUPS.values() for symbol in symbols))

HARDWARE_LEVEL_FIELDS = tuple(f"{group}_index" for group in HARDWARE_PROXY_GROUPS)
HARDWARE_COUNT_FIELDS = tuple(f"{group}_available_count" for group in HARDWARE_PROXY_GROUPS)
CLOUD_GPU_DAILY_PROXY_GROUPS: dict[str, tuple[str, ...]] = {
    "cloud_gpu_daily_hopper_proxy": ("NVDA", "SMCI", "DELL", "VRT"),
    "cloud_gpu_daily_modern_proxy": ("NVDA", "AMD", "AVGO", "TSM", "ASML"),
    "cloud_gpu_daily_hyperscaler_demand_proxy": ("AMZN", "MSFT", "GOOGL", "ORCL", "META"),
    "cloud_gpu_daily_server_supply_proxy": ("SMCI", "DELL", "VRT", "ETN", "PWR"),
}
CLOUD_GPU_DAILY_DERIVED_FIELDS = (
    "cloud_gpu_daily_supply_demand_pressure_index",
    "cloud_gpu_daily_server_tightness_index",
    "cloud_gpu_daily_memory_pressure_index",
)
CLOUD_GPU_DAILY_PROXY_FIELDS = tuple(
    [f"{group}_index" for group in CLOUD_GPU_DAILY_PROXY_GROUPS] + list(CLOUD_GPU_DAILY_DERIVED_FIELDS)
)
CLOUD_GPU_DAILY_PROXY_COUNT_FIELDS = tuple(f"{group}_available_count" for group in CLOUD_GPU_DAILY_PROXY_GROUPS)


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _decode_script_payload(script: str) -> str:
    try:
        return script.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return script


def _script_bodies(html_text: str) -> list[str]:
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html_text, flags=re.IGNORECASE | re.DOTALL)
    return scripts or [html_text]


def _iter_nivo_chart_objects(html_text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    seen: set[str] = set()
    for body in _script_bodies(html_text):
        for candidate in [body, _decode_script_payload(body)]:
            pos = 0
            while True:
                idx = candidate.find('{"id":"nivo', pos)
                if idx < 0:
                    break
                try:
                    obj, end = decoder.raw_decode(candidate[idx:])
                except json.JSONDecodeError:
                    pos = idx + 1
                    continue
                pos = idx + max(end, 1)
                if obj.get("type") != "chart-line":
                    continue
                stable = json.dumps(obj, sort_keys=True)
                if stable in seen:
                    continue
                seen.add(stable)
                yield obj


def _cloud_gpu_field_name(series: dict[str, Any]) -> str | None:
    billing = _slug(series.get("billing_type") or "ON_DEMAND")
    if series.get("gpu_model") is not None:
        gpu = _slug(series.get("id") or series.get("gpu_model"))
        return f"cloud_gpu_{gpu}_{billing}"
    if series.get("category") is not None:
        category = _slug(series.get("category") or series.get("id"))
        return f"cloud_gpu_{category}_{billing}"
    return None


def extract_aimultiple_gpu_index_series(html_text: str) -> pd.DataFrame:
    """Extract cloud GPU monthly index points from AIMultiple's embedded chart payloads."""

    rows: list[dict[str, Any]] = []
    for chart_index, chart in enumerate(_iter_nivo_chart_objects(html_text), start=1):
        for series in chart.get("data", {}).get("data", []):
            field = _cloud_gpu_field_name(series)
            if field is None:
                continue
            for point in series.get("data", []):
                rows.append(
                    {
                        "datetime": pd.to_datetime(point.get("x"), errors="coerce"),
                        "field": field,
                        "value": pd.to_numeric(point.get("y"), errors="coerce"),
                        "series_id": str(series.get("id", "")),
                        "category": str(series.get("category", "")),
                        "gpu_model": str(series.get("gpu_model", "")),
                        "billing_type": str(series.get("billing_type", "ON_DEMAND")),
                        "chart_index": chart_index,
                        "source": "aimultiple_cloud_gpu_index",
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "field",
                "value",
                "series_id",
                "category",
                "gpu_model",
                "billing_type",
                "chart_index",
                "source",
            ]
        )
    frame = frame.dropna(subset=["datetime", "field", "value"]).copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame["value"] = frame["value"].astype("float64")
    return frame.sort_values(["field", "datetime", "chart_index"]).drop_duplicates(
        ["field", "datetime"],
        keep="last",
    )


def fetch_aimultiple_gpu_index_series(*, url: str = HARDWARE_CYCLE_AIMULTIPLE_URL) -> pd.DataFrame:
    response = requests.get(url, timeout=30, headers={"User-Agent": HARDWARE_CYCLE_USER_AGENT})
    response.raise_for_status()
    return extract_aimultiple_gpu_index_series(response.text)


def fetch_hardware_proxy_price_frames(
    *,
    start: str,
    end: str,
    symbols: Iterable[str] = HARDWARE_PROXY_SYMBOLS,
    allow_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        normalized = str(symbol).upper()
        try:
            frames[normalized] = _fetch_yahoo_chart_ohlcv(normalized, start=start, end=end)
        except Exception as exc:
            if not allow_missing:
                raise
            errors[normalized] = str(exc)
    if not frames:
        raise RuntimeError(f"no hardware proxy price frames could be fetched: {errors}")
    return frames


def _normalize_proxy_close_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"{symbol} proxy frame requires date and close columns")
    normalized = frame[["date", "close"]].copy()
    normalized["datetime"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    normalized[symbol] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized = normalized.dropna(subset=["datetime", symbol])
    return normalized[["datetime", symbol]].sort_values("datetime").groupby("datetime", as_index=False).last()


def _rebased_equal_weight_index(price_frame: pd.DataFrame, symbols: Iterable[str]) -> pd.Series:
    symbol_list = [symbol for symbol in symbols if symbol in price_frame.columns]
    if not symbol_list:
        return pd.Series(np.nan, index=price_frame.index, dtype="float64")
    prices = price_frame[symbol_list].astype("float64").ffill()
    rebased_parts = []
    for symbol in symbol_list:
        series = prices[symbol]
        first_valid = series.dropna()
        if first_valid.empty or abs(float(first_valid.iloc[0])) <= 1e-12:
            continue
        rebased_parts.append(series / float(first_valid.iloc[0]) * 100.0)
    if not rebased_parts:
        return pd.Series(np.nan, index=price_frame.index, dtype="float64")
    return pd.concat(rebased_parts, axis=1).mean(axis=1, skipna=True)


def _proxy_available_count(price_frame: pd.DataFrame, symbols: Iterable[str]) -> pd.Series:
    symbol_list = [symbol for symbol in symbols if symbol in price_frame.columns]
    if not symbol_list:
        return pd.Series(0.0, index=price_frame.index, dtype="float64")
    return price_frame[symbol_list].notna().sum(axis=1).astype("float64")


def _safe_index_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype("float64")
    den = pd.to_numeric(denominator, errors="coerce").astype("float64")
    den = den.where(den.abs() > 1e-12)
    return (num / den * 100.0).replace([np.inf, -np.inf], np.nan)


def _pivot_cloud_gpu_index(cloud_gpu_index: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    if cloud_gpu_index.empty:
        return pd.DataFrame(index=calendar)
    calendar_index = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    raw = cloud_gpu_index.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.normalize()
    raw["field"] = raw["field"].astype(str)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    pivot = raw.pivot_table(index="datetime", columns="field", values="value", aggfunc="last").sort_index()
    columns: dict[str, pd.Series] = {}
    for field in pivot.columns:
        points = pivot[field].dropna().sort_index()
        mapped_points = _map_source_points_to_calendar(points, calendar_index)
        aligned = mapped_points.reindex(calendar_index).ffill()
        available = aligned.notna().astype("float64")
        source_observed = pd.Series(0.0, index=calendar_index, dtype="float64")
        if not mapped_points.empty:
            source_observed.loc[mapped_points.index] = 1.0
        is_carried = ((available > 0.0) & (source_observed <= 0.0)).astype("float64")
        last_update = pd.Series(pd.NaT, index=calendar_index, dtype="datetime64[ns]")
        if not mapped_points.empty:
            last_update.loc[mapped_points.index] = mapped_points.index
        last_update = last_update.ffill()
        staleness = pd.Series(
            (calendar_index - pd.DatetimeIndex(last_update)).days,
            index=calendar_index,
            dtype="float64",
        )
        staleness = staleness.where(available > 0.0)
        columns[str(field)] = aligned
        columns[f"{field}_available"] = available
        columns[f"{field}_source_observed"] = source_observed
        columns[f"{field}_is_carried"] = is_carried
        columns[f"{field}_staleness_days"] = staleness
    return pd.DataFrame(columns, index=calendar_index)


def _map_source_points_to_calendar(points: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    mapped: dict[pd.Timestamp, float] = {}
    for source_date, value in points.items():
        pos = calendar.searchsorted(pd.Timestamp(source_date), side="left")
        if pos >= len(calendar):
            continue
        mapped[pd.Timestamp(calendar[pos])] = float(value)
    if not mapped:
        return pd.Series(dtype="float64")
    return pd.Series(mapped, dtype="float64").sort_index()


def build_hardware_cycle_daily_frame(
    proxy_frames: dict[str, pd.DataFrame],
    cloud_gpu_index: pd.DataFrame,
    *,
    calendar: Iterable[str | pd.Timestamp],
) -> pd.DataFrame:
    """Build one qlib-ready daily hardware-cycle instrument from raw source frames."""

    calendar_index = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize().sort_values().unique()
    if len(calendar_index) == 0:
        raise ValueError("hardware cycle calendar cannot be empty")
    prices = pd.DataFrame(index=calendar_index)
    for raw_symbol, frame in proxy_frames.items():
        symbol = str(raw_symbol).upper()
        normalized = _normalize_proxy_close_frame(symbol, frame).set_index("datetime")
        prices = prices.join(normalized[[symbol]], how="left")
    prices = prices.ffill()

    daily = pd.DataFrame({"datetime": calendar_index})
    daily["instrument"] = HARDWARE_CYCLE_INSTRUMENT
    group_indices: dict[str, pd.Series] = {}
    group_counts: dict[str, pd.Series] = {}
    for group, symbols in HARDWARE_PROXY_GROUPS.items():
        group_indices[group] = _rebased_equal_weight_index(prices, symbols)
        group_counts[group] = _proxy_available_count(prices, symbols)
        daily[f"{group}_index"] = group_indices[group].to_numpy(dtype="float64")
        daily[f"{group}_available_count"] = group_counts[group].to_numpy(dtype="float64")

    for group, symbols in CLOUD_GPU_DAILY_PROXY_GROUPS.items():
        daily[f"{group}_index"] = _rebased_equal_weight_index(prices, symbols).to_numpy(dtype="float64")
        daily[f"{group}_available_count"] = _proxy_available_count(prices, symbols).to_numpy(dtype="float64")

    daily["cloud_gpu_daily_supply_demand_pressure_index"] = _safe_index_ratio(
        group_indices["ai_accelerator"],
        group_indices["cloud_capex"],
    ).to_numpy(dtype="float64")
    daily["cloud_gpu_daily_server_tightness_index"] = _safe_index_ratio(
        pd.Series(daily["cloud_gpu_daily_server_supply_proxy_index"].to_numpy(dtype="float64"), index=calendar_index),
        group_indices["cloud_capex"],
    ).to_numpy(dtype="float64")
    daily["cloud_gpu_daily_memory_pressure_index"] = _safe_index_ratio(
        group_indices["memory_storage"],
        group_indices["ai_accelerator"],
    ).to_numpy(dtype="float64")

    cloud = _pivot_cloud_gpu_index(cloud_gpu_index, calendar_index)
    if not cloud.empty:
        daily = daily.join(cloud.reset_index(drop=True))
    return daily.replace([np.inf, -np.inf], np.nan)


def _calendar_from_soxl_provider(root: Path, *, start: str, end: str) -> list[str]:
    calendar_path = root / "data" / "qlib" / "soxl" / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise FileNotFoundError(f"SOXL qlib calendar missing: {calendar_path.as_posix()}")
    dates = pd.to_datetime([line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return [pd.Timestamp(value).date().isoformat() for value in dates[mask]]


def build_hardware_cycle_qlib_store_from_frame(
    frame: pd.DataFrame,
    *,
    provider_uri: Path,
    as_of: str,
) -> dict[str, Any]:
    return build_qlib_store_from_frame(
        frame,
        provider_uri=provider_uri,
        market="all",
        region="us",
        freq="day",
        instrument_col="instrument",
        datetime_col="datetime",
        as_of=as_of,
        source="hardware_cycle_daily_public_market_proxy_and_cloud_gpu_index",
    )


def build_hardware_cycle_qlib_store(
    *,
    root: Path,
    start: str,
    end: str,
    provider_uri: Path | None = None,
    raw_dir: Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    provider_uri = provider_uri or root / "data" / "qlib" / HARDWARE_CYCLE_PROVIDER_NAME
    raw_dir = raw_dir or root / "data" / "raw" / "external" / "hardware_cycle"
    raw_dir.mkdir(parents=True, exist_ok=True)

    requested_symbols = list(HARDWARE_PROXY_SYMBOLS)
    proxy_frames = fetch_hardware_proxy_price_frames(start=start, end=end, symbols=requested_symbols)
    cloud_gpu = fetch_aimultiple_gpu_index_series()
    calendar = _calendar_from_soxl_provider(root, start=start, end=end)
    daily = build_hardware_cycle_daily_frame(proxy_frames, cloud_gpu, calendar=calendar)
    daily_path = raw_dir / "hardware_cycle_daily.csv"
    cloud_path = raw_dir / "aimultiple_cloud_gpu_index.csv"
    daily.to_csv(daily_path, index=False)
    cloud_gpu.to_csv(cloud_path, index=False)

    manifest = build_hardware_cycle_qlib_store_from_frame(daily, provider_uri=provider_uri, as_of=end)
    raw_manifest = {
        "source": "hardware_cycle_daily_public_market_proxy_and_cloud_gpu_index",
        "proxy_symbols": requested_symbols,
        "proxy_symbols_fetched": sorted(proxy_frames),
        "proxy_symbols_missing": [symbol for symbol in requested_symbols if symbol not in proxy_frames],
        "proxy_groups": {group: list(symbols) for group, symbols in HARDWARE_PROXY_GROUPS.items()},
        "daily_proxy_groups": {group: list(symbols) for group, symbols in CLOUD_GPU_DAILY_PROXY_GROUPS.items()},
        "daily_proxy_fields": list(CLOUD_GPU_DAILY_PROXY_FIELDS),
        "cloud_gpu_source": HARDWARE_CYCLE_AIMULTIPLE_URL,
        "raw_daily_csv": daily_path.as_posix(),
        "raw_cloud_gpu_csv": cloud_path.as_posix(),
        "calendar_start": min(calendar),
        "calendar_end": max(calendar),
        "row_count": int(len(daily)),
        "cloud_gpu_field_count": int(cloud_gpu["field"].nunique()) if not cloud_gpu.empty else 0,
        "limitations": [
            "AIMultiple cloud GPU index is a reconstructed monthly external index with shorter history than market proxies.",
            "Monthly cloud GPU values are forward-filled as daily state; source_observed, is_carried, and staleness_days fields preserve update provenance.",
            "cloud_gpu_daily_* fields are daily public-market proxy indices from trading-day closes; they are not direct rental quote history.",
            "Direct daily GPU rental indices are not fetched unless a licensed or host-authenticated source is provided.",
            "PricePerGig product-level storage data is excluded from training due to ML/caching restrictions.",
            "All model-facing hardware-cycle factors must be lagged by the factor builder.",
        ],
    }
    raw_manifest_path = raw_dir / "hardware_cycle_raw_manifest.json"
    raw_manifest_path.write_text(json.dumps(raw_manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["raw_manifest"] = raw_manifest_path.as_posix()
    (Path(manifest["provider_uri"]) / "qlib_store_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest

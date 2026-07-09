from __future__ import annotations

import json
import re
from io import StringIO
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
FREE_GPU_PRICE_HISTORY_BASE_URL = "https://freellm.org/history"
FREE_GPU_PRICE_LATEST_META_URL = "https://freellm.org/latest/meta.json"
FREE_GPU_PRICE_LATEST_URL = "https://freellm.org/latest/all.json"
FREE_GPU_PRICE_SOURCE = "freellm_gpu_cloud_pricing_history"
FREE_GPU_PRICE_MODELS = (
    "gb200",
    "b200",
    "h200",
    "h100",
    "a100",
    "l40s",
    "rtx_pro_6000",
    "rtx_4090",
    "mi300x",
)
FREE_GPU_PRICE_FIELDS = tuple(f"free_cloud_gpu_{model}_price" for model in FREE_GPU_PRICE_MODELS)
FREE_GPU_PRICE_SNAPSHOT_COLUMNS = (
    "datetime",
    "field",
    "value",
    "provider_count",
    "quote_count",
    "source",
)
FRED_HARDWARE_SALES_PRICE_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_HARDWARE_SALES_PRICE_SOURCE = "fred_bls_hardware_sales_price_index"
BLS_HARDWARE_SALES_PRICE_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_HARDWARE_SALES_PRICE_SOURCE = "bls_public_api_hardware_sales_price_index"
FRED_HARDWARE_SALES_PRICE_SERIES: dict[str, dict[str, str]] = {
    "hardware_sales_storage_device_ppi": {
        "series_id": "PCU334112334112",
        "description": "Producer Price Index by Industry: Computer Storage Device Manufacturing",
    },
    "hardware_sales_storage_device_except_parts_ppi": {
        "series_id": "PCU3341123341121",
        "description": "Producer Price Index by Industry: Computer Storage Devices Except Parts",
    },
    "hardware_sales_storage_device_primary_ppi": {
        "series_id": "PCU334112334112P",
        "description": "Producer Price Index by Industry: Computer Storage Device Manufacturing Primary Products",
    },
    "hardware_sales_server_ppi": {
        "series_id": "PCU3341113341115",
        "description": "Producer Price Index by Industry: Host Computers, Mainframes, Unix and PC Servers",
    },
    "hardware_sales_computer_peripheral_parts_ppi": {
        "series_id": "WPU1154",
        "description": "Producer Price Index by Commodity: Computer Peripheral Equipment and Parts",
    },
    "hardware_sales_semiconductor_related_ppi": {
        "series_id": "PCU334413334413",
        "description": "Producer Price Index by Industry: Semiconductor and Related Device Manufacturing",
    },
    "hardware_sales_computer_peripheral_equipment_ppi": {
        "series_id": "PCU3341--3341--",
        "fred_series_id": "PCU33413341",
        "description": "Producer Price Index by Industry: Computer and Peripheral Equipment Manufacturing",
    },
}
HARDWARE_SALES_PRICE_FIELDS = tuple(FRED_HARDWARE_SALES_PRICE_SERIES)
FRED_HARDWARE_SALES_PRICE_COLUMNS = (
    "datetime",
    "field",
    "value",
    "series_id",
    "source",
)

HARDWARE_PROXY_GROUPS: dict[str, tuple[str, ...]] = {
    "ai_accelerator": ("NVDA", "AMD", "AVGO", "TSM", "ASML"),
    "cpu_compute": ("AMD", "INTC", "QCOM", "MRVL", "ARM"),
    "memory_storage": ("MU", "WDC", "STX"),
    "enterprise_storage": ("NTAP", "PSTG", "HPE", "HPQ"),
    "cloud_capex": ("AMZN", "MSFT", "GOOGL", "ORCL", "META"),
    "ai_server": ("SMCI", "DELL"),
    "power_cooling": ("VRT", "ETN", "PWR"),
    "storage_hdd_manufacturer": ("WDC", "STX"),
    "storage_nand_ssd": ("MU", "WDC", "SIMO"),
    "networking_interconnect": ("ANET", "CSCO", "CIEN", "COHR", "LITE", "GLW"),
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
HARDWARE_DAILY_DERIVED_FIELDS = (
    "storage_hdd_cloud_demand_pressure_index",
    "storage_hdd_vs_nand_index",
    "networking_cloud_demand_pressure_index",
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


def _empty_free_gpu_price_history() -> pd.DataFrame:
    return pd.DataFrame(columns=list(FREE_GPU_PRICE_SNAPSHOT_COLUMNS))


def _free_gpu_model_key(row: dict[str, Any]) -> str | None:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("product", "gpu_model", "model", "name", "sku", "description")
    ).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", text)
    patterns = (
        ("gb200", r"(?<![a-z0-9])gb200(?![a-z0-9])"),
        ("b200", r"(?<![a-z0-9])b200(?![a-z0-9])"),
        ("h200", r"(?<![a-z0-9])h200(?![a-z0-9])"),
        ("h100", r"(?<![a-z0-9])h100(?![a-z0-9])"),
        ("a100", r"(?<![a-z0-9])a100(?![a-z0-9])"),
        ("l40s", r"(?<![a-z0-9])l40s(?![a-z0-9])"),
        ("rtx_pro_6000", r"rtx\s+pro\s+6000|rtx\s+6000|pro\s+6000"),
        ("rtx_4090", r"rtx\s+4090|(?<![a-z0-9])4090(?![a-z0-9])"),
        ("mi300x", r"(?<![a-z0-9])mi300x(?![a-z0-9])"),
    )
    for model, pattern in patterns:
        if re.search(pattern, normalized):
            return model
    return None


def _free_gpu_price_value(row: dict[str, Any]) -> float | None:
    for key in (
        "price_per_hour_usd",
        "price_per_gpu_hour_usd",
        "price_per_gpu_hour",
        "hourly_price_usd",
        "secure_price",
        "community_price",
        "price",
        "raw_price",
    ):
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", value)
            if match is None:
                continue
            value = match.group(0).replace(",", "")
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            continue
        price = float(numeric)
        if price > 0.0:
            return price
    return None


def extract_freellm_gpu_price_snapshot(
    rows: list[dict[str, Any]],
    *,
    snapshot_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Extract daily free cloud-GPU rental price indices from a freellm snapshot."""

    snapshot_ts = pd.Timestamp(snapshot_date).normalize()
    parsed: list[dict[str, Any]] = []
    for row in rows:
        model = _free_gpu_model_key(row)
        price = _free_gpu_price_value(row)
        if model is None or price is None:
            continue
        parsed.append(
            {
                "datetime": snapshot_ts,
                "field": f"free_cloud_gpu_{model}_price",
                "value": price,
                "provider": str(row.get("provider") or "").strip().lower(),
            }
        )
    if not parsed:
        return _empty_free_gpu_price_history()

    frame = pd.DataFrame(parsed)
    grouped = []
    for field, field_frame in frame.groupby("field", sort=True):
        providers = field_frame["provider"].replace("", np.nan).dropna()
        grouped.append(
            {
                "datetime": snapshot_ts,
                "field": str(field),
                "value": round(float(field_frame["value"].median()), 6),
                "provider_count": int(providers.nunique()),
                "quote_count": int(len(field_frame)),
                "source": FREE_GPU_PRICE_SOURCE,
            }
        )
    return pd.DataFrame(grouped, columns=list(FREE_GPU_PRICE_SNAPSHOT_COLUMNS)).sort_values(
        ["datetime", "field"],
    )


def _freellm_latest_snapshot_date(
    session: requests.Session,
    *,
    meta_url: str,
) -> pd.Timestamp | None:
    response = session.get(meta_url, timeout=20)
    response.raise_for_status()
    meta = response.json()
    generated_at = meta.get("generated_at_utc") if isinstance(meta, dict) else None
    if not generated_at:
        return None
    return pd.Timestamp(generated_at).tz_localize(None).normalize()


def fetch_freellm_gpu_price_history(
    *,
    start: str,
    end: str,
    base_url: str = FREE_GPU_PRICE_HISTORY_BASE_URL,
    meta_url: str = FREE_GPU_PRICE_LATEST_META_URL,
    latest_url: str = FREE_GPU_PRICE_LATEST_URL,
    max_scan_days: int = 120,
) -> pd.DataFrame:
    """Fetch no-auth daily cloud-GPU rental price history snapshots from freellm."""

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        return _empty_free_gpu_price_history()

    session = requests.Session()
    session.headers.update({"User-Agent": HARDWARE_CYCLE_USER_AGENT})
    try:
        latest_ts = _freellm_latest_snapshot_date(session, meta_url=meta_url) or end_ts
    except Exception:
        latest_ts = end_ts
    latest_ts = min(latest_ts, end_ts)
    scan_start = max(start_ts, latest_ts - pd.Timedelta(days=max_scan_days))
    if scan_start > latest_ts:
        return _empty_free_gpu_price_history()

    snapshots: list[pd.DataFrame] = []
    for snapshot_ts in pd.date_range(scan_start, latest_ts, freq="D"):
        url = f"{base_url.rstrip('/')}/{snapshot_ts.date().isoformat()}/all.json"
        try:
            response = session.get(url, timeout=20)
        except requests.RequestException:
            continue
        if response.status_code == 404:
            continue
        if not response.ok:
            continue
        try:
            rows = response.json()
        except ValueError:
            continue
        if not isinstance(rows, list):
            continue
        snapshot = extract_freellm_gpu_price_snapshot(rows, snapshot_date=snapshot_ts)
        if not snapshot.empty:
            snapshots.append(snapshot)

    if not snapshots and start_ts <= latest_ts <= end_ts:
        try:
            response = session.get(latest_url, timeout=20)
            response.raise_for_status()
            rows = response.json()
            if isinstance(rows, list):
                snapshot = extract_freellm_gpu_price_snapshot(rows, snapshot_date=latest_ts)
                if not snapshot.empty:
                    snapshots.append(snapshot)
        except Exception:
            pass

    if not snapshots:
        return _empty_free_gpu_price_history()
    frame = pd.concat(snapshots, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["provider_count"] = pd.to_numeric(frame["provider_count"], errors="coerce").fillna(0.0).astype("int64")
    frame["quote_count"] = pd.to_numeric(frame["quote_count"], errors="coerce").fillna(0.0).astype("int64")
    return (
        frame.dropna(subset=["datetime", "field", "value"])
        .sort_values(["datetime", "field"])
        .drop_duplicates(["datetime", "field"], keep="last")
        .reset_index(drop=True)
    )


def _empty_hardware_sales_price_index() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(FRED_HARDWARE_SALES_PRICE_COLUMNS))
    frame.attrs["errors"] = {}
    return frame


def extract_fred_hardware_sales_price_series(
    csv_text: str,
    *,
    series_id: str,
    field: str,
) -> pd.DataFrame:
    """Extract one official FRED/BLS hardware sales price index series."""

    raw = pd.read_csv(StringIO(csv_text))
    if "observation_date" not in raw.columns or series_id not in raw.columns:
        raise ValueError(f"FRED series {series_id} requires observation_date and value column")
    frame = raw[["observation_date", series_id]].copy()
    frame["datetime"] = pd.to_datetime(frame["observation_date"], errors="coerce").dt.normalize()
    frame["value"] = pd.to_numeric(frame[series_id], errors="coerce")
    frame = frame.dropna(subset=["datetime", "value"])
    if frame.empty:
        return _empty_hardware_sales_price_index()
    out = pd.DataFrame(
        {
            "datetime": frame["datetime"],
            "field": str(field),
            "value": frame["value"].astype("float64"),
            "series_id": str(series_id),
            "source": FRED_HARDWARE_SALES_PRICE_SOURCE,
        }
    )
    return out.sort_values(["field", "datetime"]).drop_duplicates(["field", "datetime"], keep="last")


def fetch_fred_hardware_sales_price_series(
    *,
    series_map: dict[str, dict[str, str]] = FRED_HARDWARE_SALES_PRICE_SERIES,
    base_url: str = FRED_HARDWARE_SALES_PRICE_BASE_URL,
    allow_missing: bool = True,
) -> pd.DataFrame:
    """Fetch official no-auth monthly hardware sales price indices from FRED/BLS."""

    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": HARDWARE_CYCLE_USER_AGENT})
    for field, meta in series_map.items():
        series_id = str(meta.get("fred_series_id") or meta["series_id"])
        try:
            response = session.get(base_url, params={"id": series_id}, timeout=30)
            response.raise_for_status()
            frame = extract_fred_hardware_sales_price_series(
                response.text,
                series_id=series_id,
                field=field,
            )
        except Exception as exc:
            if not allow_missing:
                raise
            errors[str(field)] = str(exc)
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        empty = _empty_hardware_sales_price_index()
        empty.attrs["errors"] = errors
        return empty
    out = pd.concat(frames, ignore_index=True).sort_values(["field", "datetime"]).reset_index(drop=True)
    out.attrs["errors"] = errors
    return out


def extract_bls_hardware_sales_price_payload(
    payload: dict[str, Any],
    *,
    series_id_to_field: dict[str, str],
) -> pd.DataFrame:
    """Extract official BLS Public API monthly hardware sales price index values."""

    rows: list[dict[str, Any]] = []
    results = payload.get("Results") if isinstance(payload, dict) else None
    series_list = results.get("series", []) if isinstance(results, dict) else []
    for series in series_list:
        if not isinstance(series, dict):
            continue
        series_id = str(series.get("seriesID") or "")
        field = series_id_to_field.get(series_id)
        if field is None:
            continue
        for point in series.get("data", []):
            if not isinstance(point, dict):
                continue
            period = str(point.get("period") or "")
            if not re.fullmatch(r"M(0[1-9]|1[0-2])", period):
                continue
            rows.append(
                {
                    "datetime": pd.Timestamp(year=int(point["year"]), month=int(period[1:]), day=1),
                    "field": field,
                    "value": pd.to_numeric(point.get("value"), errors="coerce"),
                    "series_id": series_id,
                    "source": BLS_HARDWARE_SALES_PRICE_SOURCE,
                }
            )
    if not rows:
        return _empty_hardware_sales_price_index()
    frame = pd.DataFrame(rows)
    frame = frame.dropna(subset=["datetime", "field", "value"])
    if frame.empty:
        return _empty_hardware_sales_price_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame["value"] = frame["value"].astype("float64")
    return frame.sort_values(["field", "datetime"]).drop_duplicates(["field", "datetime"], keep="last")


def fetch_bls_hardware_sales_price_series(
    *,
    series_map: dict[str, dict[str, str]] = FRED_HARDWARE_SALES_PRICE_SERIES,
    api_url: str = BLS_HARDWARE_SALES_PRICE_API_URL,
    start_year: int = 1992,
    end_year: int | None = None,
    chunk_years: int = 10,
    allow_missing: bool = True,
) -> pd.DataFrame:
    """Fetch official no-auth BLS monthly hardware sales price indices in 10-year chunks."""

    if chunk_years < 1:
        raise ValueError("chunk_years must be positive")
    end_year = int(end_year or pd.Timestamp.utcnow().year)
    start_year = int(start_year)
    if end_year < start_year:
        return _empty_hardware_sales_price_index()

    series_id_to_field = {str(meta["series_id"]): str(field) for field, meta in series_map.items()}
    series_ids = list(series_id_to_field)
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": HARDWARE_CYCLE_USER_AGENT, "Content-Type": "application/json"})
    for chunk_start in range(start_year, end_year + 1, chunk_years):
        chunk_end = min(end_year, chunk_start + chunk_years - 1)
        payload = {
            "seriesid": series_ids,
            "startyear": str(chunk_start),
            "endyear": str(chunk_end),
        }
        try:
            response = session.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            response_payload = response.json()
            frame = extract_bls_hardware_sales_price_payload(
                response_payload,
                series_id_to_field=series_id_to_field,
            )
        except Exception as exc:
            if not allow_missing:
                raise
            errors[f"{chunk_start}-{chunk_end}"] = str(exc)
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        empty = _empty_hardware_sales_price_index()
        empty.attrs["errors"] = errors
        return empty
    out = pd.concat(frames, ignore_index=True).sort_values(["field", "datetime"]).drop_duplicates(
        ["field", "datetime"],
        keep="last",
    )
    missing_series = set(series_id_to_field) - set(out["series_id"].dropna().astype(str).unique())
    for series_id in sorted(missing_series):
        errors[series_id_to_field[series_id]] = f"no observations returned for {series_id}"
    out = out.reset_index(drop=True)
    out.attrs["errors"] = errors
    return out


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


def _pivot_hardware_sales_price_index(
    hardware_sales_price_index: pd.DataFrame | None,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    if hardware_sales_price_index is None or hardware_sales_price_index.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(calendar).normalize().sort_values().unique())
    return _pivot_cloud_gpu_index(hardware_sales_price_index, calendar)


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


def _pivot_free_gpu_price_history(
    free_gpu_price_history: pd.DataFrame | None,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    if free_gpu_price_history is None or free_gpu_price_history.empty:
        return pd.DataFrame(index=calendar)
    calendar_index = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    raw = free_gpu_price_history.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce").dt.normalize()
    raw["field"] = raw["field"].astype(str)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    for column in ("provider_count", "quote_count"):
        if column not in raw.columns:
            raw[column] = 0.0
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0.0)
    raw = raw.dropna(subset=["datetime", "field", "value"])
    raw = raw.loc[raw["field"].str.startswith("free_cloud_gpu_") & raw["field"].str.endswith("_price")]
    if raw.empty:
        return pd.DataFrame(index=calendar_index)

    columns: dict[str, pd.Series] = {}
    for field, field_frame in raw.sort_values(["datetime", "field"]).groupby("field", sort=True):
        mapped_values: dict[pd.Timestamp, float] = {}
        mapped_provider_counts: dict[pd.Timestamp, float] = {}
        mapped_quote_counts: dict[pd.Timestamp, float] = {}
        for row in field_frame.itertuples(index=False):
            pos = calendar_index.searchsorted(pd.Timestamp(row.datetime), side="left")
            if pos >= len(calendar_index):
                continue
            mapped_date = pd.Timestamp(calendar_index[pos])
            mapped_values[mapped_date] = float(row.value)
            mapped_provider_counts[mapped_date] = float(row.provider_count)
            mapped_quote_counts[mapped_date] = float(row.quote_count)

        values = pd.Series(np.nan, index=calendar_index, dtype="float64")
        provider_counts = pd.Series(0.0, index=calendar_index, dtype="float64")
        quote_counts = pd.Series(0.0, index=calendar_index, dtype="float64")
        if mapped_values:
            mapped_index = pd.DatetimeIndex(sorted(mapped_values))
            values.loc[mapped_index] = [mapped_values[pd.Timestamp(date)] for date in mapped_index]
            provider_counts.loc[mapped_index] = [
                mapped_provider_counts.get(pd.Timestamp(date), 0.0) for date in mapped_index
            ]
            quote_counts.loc[mapped_index] = [
                mapped_quote_counts.get(pd.Timestamp(date), 0.0) for date in mapped_index
            ]
        available = values.notna().astype("float64")
        columns[str(field)] = values
        columns[f"{field}_available"] = available
        columns[f"{field}_source_observed"] = available.copy()
        columns[f"{field}_provider_count"] = provider_counts
        columns[f"{field}_quote_count"] = quote_counts
    return pd.DataFrame(columns, index=calendar_index)


def build_hardware_cycle_daily_frame(
    proxy_frames: dict[str, pd.DataFrame],
    cloud_gpu_index: pd.DataFrame,
    *,
    free_gpu_price_history: pd.DataFrame | None = None,
    hardware_sales_price_index: pd.DataFrame | None = None,
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
    daily["storage_hdd_cloud_demand_pressure_index"] = _safe_index_ratio(
        group_indices["storage_hdd_manufacturer"],
        group_indices["cloud_capex"],
    ).to_numpy(dtype="float64")
    daily["storage_hdd_vs_nand_index"] = _safe_index_ratio(
        group_indices["storage_hdd_manufacturer"],
        group_indices["storage_nand_ssd"],
    ).to_numpy(dtype="float64")
    daily["networking_cloud_demand_pressure_index"] = _safe_index_ratio(
        group_indices["networking_interconnect"],
        group_indices["cloud_capex"],
    ).to_numpy(dtype="float64")

    cloud = _pivot_cloud_gpu_index(cloud_gpu_index, calendar_index)
    if not cloud.empty:
        daily = daily.join(cloud.reset_index(drop=True))
    free_gpu = _pivot_free_gpu_price_history(free_gpu_price_history, calendar_index)
    if not free_gpu.empty:
        daily = daily.join(free_gpu.reset_index(drop=True))
    sales_price = _pivot_hardware_sales_price_index(hardware_sales_price_index, calendar_index)
    if not sales_price.empty:
        daily = daily.join(sales_price.reset_index(drop=True))
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
        source="hardware_cycle_daily_public_market_proxy_cloud_gpu_index_free_gpu_prices_and_bls_fred_sales_prices",
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
    free_gpu_error: str | None = None
    try:
        free_gpu = fetch_freellm_gpu_price_history(start=start, end=end)
    except Exception as exc:
        free_gpu_error = str(exc)
        free_gpu = _empty_free_gpu_price_history()
    sales_price_error: str | None = None
    try:
        hardware_sales_price = fetch_bls_hardware_sales_price_series(end_year=pd.Timestamp(end).year)
    except Exception as exc:
        sales_price_error = str(exc)
        try:
            hardware_sales_price = fetch_fred_hardware_sales_price_series()
        except Exception as fallback_exc:
            sales_price_error = f"BLS API failed: {sales_price_error}; FRED CSV failed: {fallback_exc}"
            hardware_sales_price = _empty_hardware_sales_price_index()
    sales_price_errors = dict(hardware_sales_price.attrs.get("errors", {}))
    if sales_price_error is not None:
        sales_price_errors["_fetch"] = sales_price_error
    if sales_price_errors and not hardware_sales_price.empty:
        sales_price_status = "partial"
    elif sales_price_errors:
        sales_price_status = "fetch_failed"
    else:
        sales_price_status = "ok"
    calendar = _calendar_from_soxl_provider(root, start=start, end=end)
    daily = build_hardware_cycle_daily_frame(
        proxy_frames,
        cloud_gpu,
        free_gpu_price_history=free_gpu,
        hardware_sales_price_index=hardware_sales_price,
        calendar=calendar,
    )
    daily_path = raw_dir / "hardware_cycle_daily.csv"
    cloud_path = raw_dir / "aimultiple_cloud_gpu_index.csv"
    free_gpu_path = raw_dir / "freellm_gpu_price_history.csv"
    sales_price_path = raw_dir / "bls_hardware_sales_price_index.csv"
    daily.to_csv(daily_path, index=False)
    cloud_gpu.to_csv(cloud_path, index=False)
    free_gpu.to_csv(free_gpu_path, index=False)
    hardware_sales_price.to_csv(sales_price_path, index=False)

    manifest = build_hardware_cycle_qlib_store_from_frame(daily, provider_uri=provider_uri, as_of=end)
    raw_manifest = {
        "source": "hardware_cycle_daily_public_market_proxy_cloud_gpu_index_free_gpu_prices_and_bls_fred_sales_prices",
        "proxy_symbols": requested_symbols,
        "proxy_symbols_fetched": sorted(proxy_frames),
        "proxy_symbols_missing": [symbol for symbol in requested_symbols if symbol not in proxy_frames],
        "proxy_groups": {group: list(symbols) for group, symbols in HARDWARE_PROXY_GROUPS.items()},
        "daily_proxy_groups": {group: list(symbols) for group, symbols in CLOUD_GPU_DAILY_PROXY_GROUPS.items()},
        "daily_proxy_fields": list(CLOUD_GPU_DAILY_PROXY_FIELDS),
        "hardware_daily_derived_fields": list(HARDWARE_DAILY_DERIVED_FIELDS),
        "cloud_gpu_source": HARDWARE_CYCLE_AIMULTIPLE_URL,
        "free_gpu_price_source": f"{FREE_GPU_PRICE_HISTORY_BASE_URL}/YYYY-MM-DD/all.json",
        "free_gpu_price_latest_meta": FREE_GPU_PRICE_LATEST_META_URL,
        "free_gpu_price_status": "ok" if free_gpu_error is None else "fetch_failed",
        "free_gpu_price_error": free_gpu_error,
        "free_gpu_price_field_count": int(free_gpu["field"].nunique()) if not free_gpu.empty else 0,
        "free_gpu_price_snapshot_count": int(free_gpu["datetime"].nunique()) if not free_gpu.empty else 0,
        "free_gpu_price_start": (
            pd.Timestamp(free_gpu["datetime"].min()).date().isoformat() if not free_gpu.empty else None
        ),
        "free_gpu_price_end": (
            pd.Timestamp(free_gpu["datetime"].max()).date().isoformat() if not free_gpu.empty else None
        ),
        "free_gpu_price_quote_rows": int(free_gpu["quote_count"].sum()) if not free_gpu.empty else 0,
        "hardware_sales_price_source": BLS_HARDWARE_SALES_PRICE_API_URL,
        "hardware_sales_price_fred_csv_source": FRED_HARDWARE_SALES_PRICE_BASE_URL,
        "hardware_sales_price_series": FRED_HARDWARE_SALES_PRICE_SERIES,
        "hardware_sales_price_status": sales_price_status,
        "hardware_sales_price_error": sales_price_error,
        "hardware_sales_price_errors": sales_price_errors,
        "hardware_sales_price_field_count": (
            int(hardware_sales_price["field"].nunique()) if not hardware_sales_price.empty else 0
        ),
        "hardware_sales_price_observation_count": int(len(hardware_sales_price)),
        "hardware_sales_price_start": (
            pd.Timestamp(hardware_sales_price["datetime"].min()).date().isoformat()
            if not hardware_sales_price.empty
            else None
        ),
        "hardware_sales_price_end": (
            pd.Timestamp(hardware_sales_price["datetime"].max()).date().isoformat()
            if not hardware_sales_price.empty
            else None
        ),
        "hardware_sales_price_series_ids": (
            sorted(hardware_sales_price["series_id"].dropna().astype(str).unique().tolist())
            if not hardware_sales_price.empty and "series_id" in hardware_sales_price
            else []
        ),
        "raw_daily_csv": daily_path.as_posix(),
        "raw_cloud_gpu_csv": cloud_path.as_posix(),
        "raw_free_gpu_price_csv": free_gpu_path.as_posix(),
        "raw_hardware_sales_price_csv": sales_price_path.as_posix(),
        "calendar_start": min(calendar),
        "calendar_end": max(calendar),
        "row_count": int(len(daily)),
        "cloud_gpu_field_count": int(cloud_gpu["field"].nunique()) if not cloud_gpu.empty else 0,
        "limitations": [
            "AIMultiple cloud GPU index is a reconstructed monthly external index with shorter history than market proxies.",
            "Monthly cloud GPU values are forward-filled as daily state; source_observed, is_carried, and staleness_days fields preserve update provenance.",
            "cloud_gpu_daily_* fields are daily public-market proxy indices from trading-day closes; they are not direct rental quote history.",
            "free_cloud_gpu_* fields are free daily provider snapshot rental prices from freellm history; missing dates are not forward-filled.",
            "Free direct GPU price history is cloud rental and instance-hour pricing, not hardware sales pricing.",
            "Free direct GPU rental price history is short and may be stale relative to the market-proxy calendar.",
            "FRED/BLS hardware_sales_* fields are official monthly producer sales price indices mapped to the first available qlib trading day and forward-filled as staleness-tracked state, not direct daily product quotes.",
            "storage_hdd_* and networking_* daily fields are public-market proxy indices from listed supply-chain equities, not hard-disk spot or contract prices.",
            "PricePerGig product-level storage data is excluded from training due to ML/caching restrictions.",
            "All model-facing hardware-cycle factors must be lagged by the factor builder.",
        ],
    }
    raw_manifest_path = raw_dir / "hardware_cycle_raw_manifest.json"
    raw_manifest_path.write_text(json.dumps(raw_manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["raw_manifest"] = raw_manifest_path.as_posix()
    manifest["free_gpu_price_source"] = raw_manifest["free_gpu_price_source"]
    manifest["free_gpu_price_status"] = raw_manifest["free_gpu_price_status"]
    manifest["free_gpu_price_field_count"] = raw_manifest["free_gpu_price_field_count"]
    manifest["free_gpu_price_snapshot_count"] = raw_manifest["free_gpu_price_snapshot_count"]
    manifest["free_gpu_price_start"] = raw_manifest["free_gpu_price_start"]
    manifest["free_gpu_price_end"] = raw_manifest["free_gpu_price_end"]
    manifest["free_gpu_price_quote_rows"] = raw_manifest["free_gpu_price_quote_rows"]
    manifest["hardware_sales_price_source"] = raw_manifest["hardware_sales_price_source"]
    manifest["hardware_sales_price_fred_csv_source"] = raw_manifest["hardware_sales_price_fred_csv_source"]
    manifest["hardware_sales_price_status"] = raw_manifest["hardware_sales_price_status"]
    manifest["hardware_sales_price_field_count"] = raw_manifest["hardware_sales_price_field_count"]
    manifest["hardware_sales_price_observation_count"] = raw_manifest["hardware_sales_price_observation_count"]
    manifest["hardware_sales_price_start"] = raw_manifest["hardware_sales_price_start"]
    manifest["hardware_sales_price_end"] = raw_manifest["hardware_sales_price_end"]
    manifest["hardware_sales_price_series_ids"] = raw_manifest["hardware_sales_price_series_ids"]
    (Path(manifest["provider_uri"]) / "qlib_store_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest

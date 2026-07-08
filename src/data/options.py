from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


NEW_YORK_TZ = ZoneInfo("America/New_York")


def normalize_polygon_option_contracts(
    raw: pd.DataFrame | list[dict[str, Any]],
    *,
    underlying: str,
    as_of: str | None = None,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(raw).copy()
    columns = [
        "provider",
        "underlying",
        "option_ticker",
        "contract_type",
        "expiration_date",
        "strike_price",
        "shares_per_contract",
        "exercise_style",
        "primary_exchange",
        "cfi",
        "as_of",
        "fetched_at",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    required = {"ticker", "contract_type", "expiration_date", "strike_price"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"polygon option contracts missing columns: {missing}")

    expiration = pd.to_datetime(frame["expiration_date"], errors="coerce")
    normalized = pd.DataFrame(
        {
            "provider": "polygon",
            "underlying": underlying.upper(),
            "option_ticker": frame["ticker"].astype(str),
            "contract_type": frame["contract_type"].astype(str).str.lower(),
            "expiration_date": expiration.dt.date,
            "strike_price": pd.to_numeric(frame["strike_price"], errors="coerce"),
            "shares_per_contract": pd.to_numeric(frame.get("shares_per_contract", np.nan), errors="coerce"),
            "exercise_style": frame.get("exercise_style", "").astype(str) if "exercise_style" in frame else "",
            "primary_exchange": frame.get("primary_exchange", "").astype(str) if "primary_exchange" in frame else "",
            "cfi": frame.get("cfi", "").astype(str) if "cfi" in frame else "",
            "as_of": as_of or "",
            "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
        }
    )
    normalized = normalized.dropna(subset=["expiration_date", "strike_price"])
    normalized["expiration_date"] = pd.to_datetime(normalized["expiration_date"]).dt.date.astype(str)
    return (
        normalized.sort_values(["expiration_date", "strike_price", "contract_type", "option_ticker"])
        .drop_duplicates(["option_ticker"], keep="last")
        .reset_index(drop=True)
    )


def normalize_polygon_option_daily_aggregates(
    raw: pd.DataFrame | list[dict[str, Any]],
    *,
    option_ticker: str,
    underlying: str,
    adjusted: bool = True,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(raw).copy()
    columns = [
        "provider",
        "underlying",
        "option_ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
        "adjusted",
        "fetched_at",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    required = {"t", "o", "h", "l", "c", "v"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"polygon option daily aggregates missing columns: {missing}")

    timestamps = pd.to_datetime(frame["t"], unit="ms", utc=True, errors="coerce").dt.tz_convert(NEW_YORK_TZ)
    normalized = pd.DataFrame(
        {
            "provider": "polygon",
            "underlying": underlying.upper(),
            "option_ticker": option_ticker,
            "date": timestamps.dt.date,
            "open": pd.to_numeric(frame["o"], errors="coerce"),
            "high": pd.to_numeric(frame["h"], errors="coerce"),
            "low": pd.to_numeric(frame["l"], errors="coerce"),
            "close": pd.to_numeric(frame["c"], errors="coerce"),
            "volume": pd.to_numeric(frame["v"], errors="coerce"),
            "vwap": pd.to_numeric(frame["vw"], errors="coerce") if "vw" in frame else np.nan,
            "transactions": pd.to_numeric(frame["n"], errors="coerce") if "n" in frame else np.nan,
            "adjusted": bool(adjusted),
            "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
        }
    )
    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.date.astype(str)
    return (
        normalized.sort_values(["option_ticker", "date"])
        .drop_duplicates(["option_ticker", "date"], keep="last")
        .reset_index(drop=True)
    )


def fetch_polygon_option_contracts(
    *,
    underlying: str,
    api_key: str,
    expiration_start: str,
    expiration_end: str,
    expired: bool | None = None,
    as_of: str | None = None,
    base_url: str = "https://api.polygon.io",
    timeout: float = 30.0,
    limit: int = 1000,
    max_pages: int = 50,
    max_retries: int = 4,
    retry_sleep_seconds: float = 12.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import requests

    params: dict[str, Any] = {
        "underlying_ticker": underlying.upper(),
        "expiration_date.gte": expiration_start,
        "expiration_date.lte": expiration_end,
        "limit": int(limit),
        "sort": "expiration_date",
        "apiKey": api_key,
    }
    if expired is not None:
        params["expired"] = "true" if expired else "false"
    if as_of:
        params["as_of"] = as_of

    rows: list[dict[str, Any]] = []
    page_count = 0
    last_status: int | None = None
    message = ""
    current_url: str | None = f"{base_url.rstrip('/')}/v3/reference/options/contracts"
    current_params: dict[str, Any] | None = params

    while current_url and page_count < max_pages:
        response, payload = _polygon_get_json_with_retries(
            current_url,
            params=current_params,
            timeout=timeout,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        last_status = int(response.status_code)
        message = str(payload.get("message") or payload.get("status") or "")
        if response.status_code != 200:
            status = "ok_partial_provider_rate_limit" if rows and response.status_code == 429 else "blocked_provider"
            if response.status_code == 429 and not rows:
                status = "blocked_provider_rate_limit"
            return _option_contract_rows_with_audit(
                rows,
                underlying=underlying,
                as_of=as_of,
                status=status,
                http_status=last_status,
                page_count=page_count,
                message=message,
                expiration_start=expiration_start,
                expiration_end=expiration_end,
                expired=expired,
            )
        rows.extend(payload.get("results") or [])
        page_count += 1
        next_url = payload.get("next_url")
        if not next_url:
            current_url = None
        else:
            current_url = str(next_url)
            current_params = {"apiKey": api_key}

    status = "ok" if rows else "blocked_no_rows"
    return _option_contract_rows_with_audit(
        rows,
        underlying=underlying,
        as_of=as_of,
        status=status,
        http_status=last_status,
        page_count=page_count,
        message=message,
        expiration_start=expiration_start,
        expiration_end=expiration_end,
        expired=expired,
    )


def fetch_polygon_option_daily_bars(
    option_ticker: str,
    *,
    underlying: str,
    api_key: str,
    start: str,
    end: str,
    adjusted: bool = True,
    base_url: str = "https://api.polygon.io",
    timeout: float = 30.0,
    max_retries: int = 4,
    retry_sleep_seconds: float = 12.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/v2/aggs/ticker/{option_ticker}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    response, payload = _polygon_get_json_with_retries(
        url,
        params=params,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep_seconds=retry_sleep_seconds,
    )
    rows = payload.get("results") or []
    normalized = normalize_polygon_option_daily_aggregates(
        rows,
        option_ticker=option_ticker,
        underlying=underlying,
        adjusted=adjusted,
    )
    status = str(payload.get("status") or "")
    if response.status_code == 429:
        status = "blocked_provider_rate_limit"
    elif response.status_code != 200:
        status = "blocked_provider_history"
    elif normalized.empty:
        status = "blocked_no_rows"
    elif status in {"", "OK", "DELAYED"}:
        status = "ok"
    coverage_start = str(normalized["date"].min()) if not normalized.empty else ""
    coverage_end = str(normalized["date"].max()) if not normalized.empty else ""
    return normalized, {
        "option_ticker": option_ticker,
        "underlying": underlying.upper(),
        "status": status,
        "provider_status": payload.get("status", ""),
        "http_status": int(response.status_code),
        "row_count": int(len(normalized)),
        "message": str(payload.get("message") or payload.get("error") or ""),
        "requested_start": start,
        "requested_end": end,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "provider_history_truncated": bool(
            normalized.empty
            or pd.Timestamp(coverage_start).date() > pd.Timestamp(start).date()
            or pd.Timestamp(coverage_end).date() < pd.Timestamp(end).date()
        ),
    }


def select_daily_option_targets(
    contracts: pd.DataFrame,
    spot_frame: pd.DataFrame,
    *,
    min_dte: int = 25,
    max_dte: int = 35,
    target_dte: int = 30,
    target_specs: Sequence[str] = ("atm_call", "atm_put"),
) -> pd.DataFrame:
    required_contracts = {"option_ticker", "contract_type", "expiration_date", "strike_price"}
    missing_contracts = sorted(required_contracts - set(contracts.columns))
    if missing_contracts:
        raise ValueError(f"contracts missing columns: {missing_contracts}")
    required_spot = {"date", "underlying_open"}
    missing_spot = sorted(required_spot - set(spot_frame.columns))
    if missing_spot:
        raise ValueError(f"spot_frame missing columns: {missing_spot}")

    contract_frame = contracts.copy()
    contract_frame["expiration_date"] = pd.to_datetime(contract_frame["expiration_date"]).dt.date
    contract_frame["contract_type"] = contract_frame["contract_type"].astype(str).str.lower()
    contract_frame["strike_price"] = pd.to_numeric(contract_frame["strike_price"], errors="coerce")
    contract_frame = contract_frame.dropna(subset=["expiration_date", "strike_price"])

    rows: list[dict[str, Any]] = []
    for spot_row in spot_frame.sort_values("date").itertuples(index=False):
        trade_date = pd.Timestamp(getattr(spot_row, "date")).date()
        underlying_open = float(getattr(spot_row, "underlying_open"))
        available = contract_frame.copy()
        available["dte"] = (pd.to_datetime(available["expiration_date"]) - pd.Timestamp(trade_date)).dt.days
        available = available.loc[(available["dte"] >= min_dte) & (available["dte"] <= max_dte)].copy()
        if available.empty:
            continue
        for target_name in target_specs:
            contract_type, strike_target = _target_contract_type_and_strike(target_name, underlying_open)
            subset = available.loc[available["contract_type"] == contract_type].copy()
            if subset.empty:
                continue
            subset["dte_distance"] = (subset["dte"] - target_dte).abs()
            subset["strike_distance"] = (subset["strike_price"] - strike_target).abs()
            selected = subset.sort_values(["dte_distance", "strike_distance", "expiration_date", "strike_price"]).iloc[0]
            rows.append(
                {
                    "date": trade_date.isoformat(),
                    "underlying": str(getattr(spot_row, "underlying", "SOXL")).upper(),
                    "underlying_open": underlying_open,
                    "target": target_name,
                    "option_ticker": selected["option_ticker"],
                    "contract_type": selected["contract_type"],
                    "expiration_date": selected["expiration_date"].isoformat(),
                    "dte": int(selected["dte"]),
                    "strike_price": float(selected["strike_price"]),
                    "strike_to_open": float(selected["strike_price"]) / underlying_open,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "underlying",
                "underlying_open",
                "target",
                "option_ticker",
                "contract_type",
                "expiration_date",
                "dte",
                "strike_price",
                "strike_to_open",
            ]
        )
    return pd.DataFrame(rows).sort_values(["date", "target"]).reset_index(drop=True)


def merge_option_daily_bars(existing: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = fetched.copy()
    elif fetched.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, fetched], ignore_index=True)
    if merged.empty:
        return merged
    merged["date"] = pd.to_datetime(merged["date"]).dt.date.astype(str)
    merged["option_ticker"] = merged["option_ticker"].astype(str)
    return (
        merged.sort_values(["option_ticker", "date"])
        .drop_duplicates(["option_ticker", "date"], keep="last")
        .reset_index(drop=True)
    )


def write_option_daily_snapshot(
    *,
    output_dir: Path,
    contracts: pd.DataFrame,
    selections: pd.DataFrame,
    bars: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts.to_parquet(output_dir / "contracts.parquet", index=False)
    selections.to_csv(output_dir / "selected_contracts.csv", index=False)
    bars.to_parquet(output_dir / "option_daily_bars.parquet", index=False)

    import json

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _polygon_get_json_with_retries(
    url: str,
    *,
    params: dict[str, Any] | None,
    timeout: float,
    max_retries: int,
    retry_sleep_seconds: float,
):
    import requests

    attempts = 0
    while True:
        response = requests.get(url, params=params, timeout=timeout)
        try:
            payload = response.json()
        except ValueError:
            payload = {"status": "ERROR", "message": response.text[:240]}
        if response.status_code != 429 or attempts >= max_retries:
            return response, payload
        attempts += 1
        retry_after = response.headers.get("Retry-After")
        sleep_seconds = float(retry_after) if retry_after else retry_sleep_seconds
        time.sleep(sleep_seconds)


def _option_contract_rows_with_audit(
    rows: list[dict[str, Any]],
    *,
    underlying: str,
    as_of: str | None,
    status: str,
    http_status: int | None,
    page_count: int,
    message: str,
    expiration_start: str,
    expiration_end: str,
    expired: bool | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized = normalize_polygon_option_contracts(rows, underlying=underlying, as_of=as_of) if rows else pd.DataFrame()
    if normalized.empty and status == "ok":
        status = "blocked_no_rows"
    return normalized, {
        "underlying": underlying.upper(),
        "status": status,
        "http_status": http_status,
        "row_count": int(len(normalized)),
        "page_count": page_count,
        "message": message,
        "expiration_start": expiration_start,
        "expiration_end": expiration_end,
        "expired": expired,
        "as_of": as_of or "",
    }


def _target_contract_type_and_strike(target_name: str, underlying_open: float) -> tuple[str, float]:
    name = target_name.lower()
    if name == "atm_call":
        return "call", underlying_open
    if name == "atm_put":
        return "put", underlying_open
    if name == "put_90":
        return "put", 0.90 * underlying_open
    if name == "call_115":
        return "call", 1.15 * underlying_open
    raise ValueError(f"unsupported option target spec: {target_name}")


def option_ticker_date_bounds(selections: pd.DataFrame, *, fetch_until: str) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame(columns=["option_ticker", "start", "end"])
    frame = selections.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"]).dt.date.astype(str)
    bounds = (
        frame.groupby("option_ticker", as_index=False)
        .agg(start=("date", "min"), expiration_date=("expiration_date", "max"))
        .sort_values("option_ticker")
    )
    fetch_until_date = pd.Timestamp(fetch_until).date()
    bounds["end"] = [
        min(pd.Timestamp(expiration).date(), fetch_until_date).isoformat() for expiration in bounds["expiration_date"]
    ]
    return bounds[["option_ticker", "start", "end"]]


def build_option_snapshot_manifest(
    *,
    underlying: str,
    provider: str,
    requested_start: str,
    requested_end: str,
    fetch_until: str,
    target_specs: Iterable[str],
    min_dte: int,
    max_dte: int,
    contracts: pd.DataFrame,
    selections: pd.DataFrame,
    bars: pd.DataFrame,
    contract_audits: Sequence[dict[str, Any]],
    bar_audits: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provider": provider,
        "underlying": underlying.upper(),
        "requested_start": requested_start,
        "requested_end": requested_end,
        "fetch_until": fetch_until,
        "target_specs": list(target_specs),
        "min_dte": int(min_dte),
        "max_dte": int(max_dte),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contracts": {
            "rows": int(len(contracts)),
            "unique_option_tickers": int(contracts["option_ticker"].nunique()) if "option_ticker" in contracts else 0,
            "expiration_start": str(contracts["expiration_date"].min()) if not contracts.empty else "",
            "expiration_end": str(contracts["expiration_date"].max()) if not contracts.empty else "",
        },
        "selections": {
            "rows": int(len(selections)),
            "trade_dates": int(selections["date"].nunique()) if "date" in selections else 0,
            "unique_option_tickers": int(selections["option_ticker"].nunique()) if "option_ticker" in selections else 0,
        },
        "bars": {
            "rows": int(len(bars)),
            "unique_option_tickers": int(bars["option_ticker"].nunique()) if "option_ticker" in bars else 0,
            "date_start": str(bars["date"].min()) if not bars.empty else "",
            "date_end": str(bars["date"].max()) if not bars.empty else "",
        },
        "audits": {
            "contract_requests": list(contract_audits),
            "bar_requests": list(bar_audits),
        },
        "limitations": [
            "Polygon/Massive rate limits can leave a partial snapshot; inspect audits.bar_requests statuses.",
            "Contract definitions are selected by expiration window and do not prove each contract was listed on every trade date.",
            "Daily aggregates are OHLCV bars, not full bid/ask NBBO, IV surface, or Greeks.",
        ],
    }

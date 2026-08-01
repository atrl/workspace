from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.core import stable_hash

from .market_update import _fetch_yahoo_chart_ohlcv, _fetch_yahoo_chart_payload
from .qlib_native import QlibProviderConfig, audit_qlib_provider, build_qlib_store_from_frame


SYMBOLS = ("SOXL", "SOXX", "SOXS")
PRICE_COLUMNS = ("open", "high", "low", "close")
NASDAQ_HISTORY_URL = "https://api.nasdaq.com/api/quote/{symbol}/historical"


@dataclass(frozen=True)
class SoxsDefensiveDataPaths:
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw" / "soxs_defensive_v1"

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_soxx_soxs"

    @property
    def source_manifest(self) -> Path:
        return self.raw_dir / "source_manifest.json"


def _parse_nasdaq_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def normalize_nasdaq_history(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("Nasdaq history returned no rows")
    raw = pd.DataFrame(rows)
    required = {"date", *PRICE_COLUMNS}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Nasdaq history missing columns: {missing}")
    result = pd.DataFrame({"date": pd.to_datetime(raw["date"], errors="coerce")})
    for column in PRICE_COLUMNS:
        result[column] = _parse_nasdaq_number(raw[column])
    result = result.dropna(subset=["date", *PRICE_COLUMNS]).sort_values("date")
    return result.drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch_nasdaq_history(symbol: str, *, start: str, end: str) -> pd.DataFrame:
    response = requests.get(
        NASDAQ_HISTORY_URL.format(symbol=symbol),
        params={"assetclass": "etf", "fromdate": start, "todate": end, "limit": 10_000},
        headers={
            "User-Agent": "Mozilla/5.0 soxs-qlib-research",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or {}
    rows = (data.get("tradesTable") or {}).get("rows") or []
    return normalize_nasdaq_history(rows)


def fetch_incremental_history_with_fallback(
    symbol: str,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch a short incremental window without making Nasdaq a single point of failure."""

    try:
        return fetch_nasdaq_history(symbol, start=start, end=end), {
            "incremental_price_source": "nasdaq",
            "incremental_fallback": False,
        }
    except Exception as error:
        yahoo = _fetch_yahoo_chart_ohlcv(symbol, start=start, end=end)
        return yahoo.loc[:, ["date", *PRICE_COLUMNS]].copy(), {
            "incremental_price_source": "yahoo_chart",
            "incremental_fallback": True,
            "nasdaq_error": str(error)[:500],
        }


def fetch_yahoo_adjustment_factors(symbol: str, *, start: str, end: str) -> pd.DataFrame:
    payload = _fetch_yahoo_chart_payload(symbol, start=start, end=end)
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        raise ValueError(f"Yahoo adjustment payload missing result for {symbol}")
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjusted = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []
    timestamps = result.get("timestamp") or []
    close = quote.get("close") or []
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None),
            "close": close,
            "adjusted_close": adjusted,
        }
    )
    factors["adjustment_factor"] = pd.to_numeric(factors["adjusted_close"], errors="coerce") / pd.to_numeric(
        factors["close"], errors="coerce"
    )
    factors = factors.dropna(subset=["date", "adjustment_factor"])
    if factors.empty or factors["adjustment_factor"].le(0.0).any():
        raise ValueError(f"Yahoo adjustment factors invalid for {symbol}")
    return factors.loc[:, ["date", "adjustment_factor"]].drop_duplicates("date", keep="last")


def merge_yahoo_adjustment_factors(
    prices: pd.DataFrame,
    adjustments: pd.DataFrame,
    *,
    symbol: str,
    max_trailing_business_days: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = prices.merge(adjustments, on="date", how="left", validate="one_to_one")
    missing = merged.loc[merged["adjustment_factor"].isna(), "date"].sort_values()
    if missing.empty:
        return merged, {
            "adjustment_factor_source_end": pd.Timestamp(adjustments["date"].max()).date().isoformat(),
            "trailing_adjustment_fill_dates": [],
        }

    source_end = pd.Timestamp(adjustments["date"].max())
    missing_dates = pd.to_datetime(missing)
    trailing_days = pd.bdate_range(source_end + pd.Timedelta(days=1), missing_dates.max())
    latest_factor = float(
        adjustments.sort_values("date").iloc[-1]["adjustment_factor"]
    )
    safe_trailing_gap = (
        bool((missing_dates > source_end).all())
        and len(trailing_days) <= max_trailing_business_days
        and np.isclose(latest_factor, 1.0, atol=1e-8)
    )
    if not safe_trailing_gap:
        raise ValueError(f"Yahoo dividend adjustment factors do not cover stitched {symbol} history")

    merged["adjustment_factor"] = merged["adjustment_factor"].fillna(latest_factor)
    return merged, {
        "adjustment_factor_source_end": source_end.date().isoformat(),
        "trailing_adjustment_fill_dates": [
            pd.Timestamp(value).date().isoformat() for value in missing_dates
        ],
    }


def stitch_yahoo_to_nasdaq(yahoo: pd.DataFrame, nasdaq: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    yahoo_frame = yahoo.loc[:, ["date", *PRICE_COLUMNS]].copy()
    yahoo_frame["date"] = pd.to_datetime(yahoo_frame["date"])
    nasdaq_frame = nasdaq.loc[:, ["date", *PRICE_COLUMNS]].copy()
    nasdaq_frame["date"] = pd.to_datetime(nasdaq_frame["date"])
    overlap = yahoo_frame.merge(nasdaq_frame, on="date", suffixes=("_yahoo", "_nasdaq"))
    if len(overlap) < 20:
        raise ValueError("SOXS provider stitch requires at least 20 common Yahoo/Nasdaq rows")
    anchor = overlap.sort_values("date").head(40)
    scales: dict[str, float] = {}
    max_anchor_error = 0.0
    for column in PRICE_COLUMNS:
        ratios = anchor[f"{column}_nasdaq"] / anchor[f"{column}_yahoo"]
        scale = float(ratios.replace([np.inf, -np.inf], np.nan).dropna().median())
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"invalid Yahoo/Nasdaq stitch scale for {column}")
        scales[column] = scale
        relative_error = (anchor[f"{column}_yahoo"] * scale / anchor[f"{column}_nasdaq"] - 1.0).abs()
        max_anchor_error = max(max_anchor_error, float(relative_error.max()))
    if max_anchor_error > 0.02:
        raise ValueError(f"Yahoo/Nasdaq stitch anchor error too large: {max_anchor_error:.6f}")

    nasdaq_start = nasdaq_frame["date"].min()
    early = yahoo_frame.loc[yahoo_frame["date"] < nasdaq_start].copy()
    for column, scale in scales.items():
        early[column] *= scale
    combined = pd.concat([early, nasdaq_frame], ignore_index=True).sort_values("date")
    combined = combined.drop_duplicates("date", keep="last").reset_index(drop=True)
    returns = combined["close"].pct_change().dropna()
    max_abs_return = float(returns.abs().max())
    if max_abs_return > 0.60:
        bad_date = combined.loc[returns.abs().idxmax(), "date"].date().isoformat()
        raise ValueError(f"unresolved corporate-action discontinuity on {bad_date}: {max_abs_return:.6f}")
    audit = {
        "yahoo_rows": int(len(yahoo_frame)),
        "nasdaq_rows": int(len(nasdaq_frame)),
        "nasdaq_start": nasdaq_start.date().isoformat(),
        "combined_rows": int(len(combined)),
        "stitch_scales": scales,
        "max_anchor_relative_error": max_anchor_error,
        "max_abs_close_return": max_abs_return,
    }
    return combined, audit


def build_soxs_defensive_provider(
    *,
    root: Path | str,
    start: str = "2010-03-11",
    end: str,
    refresh_overlap_days: int = 120,
) -> dict[str, Any]:
    paths = SoxsDefensiveDataPaths(root=Path(root).resolve())
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    stitched: dict[str, pd.DataFrame] = {}
    audits: dict[str, Any] = {}
    for symbol in SYMBOLS:
        raw_path = paths.raw_dir / f"{symbol}.csv"
        existing = _read_existing_symbol(raw_path, symbol=symbol)
        fetch_start = start
        if not existing.empty:
            fetch_start = max(
                pd.Timestamp(start),
                existing["date"].max() - pd.Timedelta(days=refresh_overlap_days),
            ).date().isoformat()
        if existing.empty:
            nasdaq = fetch_nasdaq_history(symbol, start=fetch_start, end=end)
            yahoo = _fetch_yahoo_chart_ohlcv(symbol, start=fetch_start, end=end)
            combined, audit = stitch_yahoo_to_nasdaq(yahoo, nasdaq)
            refresh_mode = "initial_yahoo_nasdaq_stitch"
        else:
            combined, fetch_audit = fetch_incremental_history_with_fallback(
                symbol,
                start=fetch_start,
                end=end,
            )
            combined = combined.loc[:, ["date", *PRICE_COLUMNS]].copy()
            combined["date"] = pd.to_datetime(combined["date"])
            audit = {
                "nasdaq_rows": int(len(combined)),
                "nasdaq_start": pd.Timestamp(combined["date"].min()).date().isoformat(),
                "incremental_source": (
                    "nasdaq_or_yahoo_chart_price_with_yahoo_distribution_adjustment"
                ),
                **fetch_audit,
            }
            refresh_mode = "append_only_overlap_validation"
        adjustments = fetch_yahoo_adjustment_factors(symbol, start=fetch_start, end=end)
        combined, adjustment_audit = merge_yahoo_adjustment_factors(
            combined,
            adjustments,
            symbol=symbol,
        )
        audit.update(adjustment_audit)
        for column in PRICE_COLUMNS:
            combined[column] *= combined["adjustment_factor"]
        audit["dividend_adjustment_factor_min"] = float(combined["adjustment_factor"].min())
        audit["dividend_adjustment_factor_max"] = float(combined["adjustment_factor"].max())
        combined = combined.drop(columns="adjustment_factor")
        combined = combined.loc[
            (combined["date"] >= pd.Timestamp(fetch_start)) & (combined["date"] <= pd.Timestamp(end))
        ].reset_index(drop=True)
        if not existing.empty:
            combined = _merge_overlap_preserving_history(
                existing, combined, refresh_start=fetch_start
            )
        combined = combined.loc[combined["date"] >= pd.Timestamp(start)].sort_values("date").reset_index(drop=True)
        audit.update(
            {
                "refresh_mode": refresh_mode,
                "refresh_start": fetch_start,
                "refresh_overlap_days": int(refresh_overlap_days),
                "preserved_rows": int(len(existing)),
                "appended_rows": int((combined["date"] > existing["date"].max()).sum())
                if not existing.empty
                else int(len(combined)),
            }
        )
        stitched[symbol] = combined
        audits[symbol] = audit

    common_dates = set(stitched[SYMBOLS[0]]["date"])
    for symbol in SYMBOLS[1:]:
        common_dates &= set(stitched[symbol]["date"])
    ordered_dates = sorted(common_dates)
    if not ordered_dates or ordered_dates[-1].date().isoformat() != end:
        raise ValueError("SOXS common calendar does not reach requested as_of")

    rows: list[pd.DataFrame] = []
    for symbol, frame in stitched.items():
        common = frame.loc[frame["date"].isin(ordered_dates)].copy()
        common.insert(1, "instrument", symbol)
        common = common.rename(columns={"date": "datetime"})
        _write_symbol_history(paths.raw_dir / f"{symbol}.csv", common, symbol=symbol)
        rows.append(common)
    qlib_frame = pd.concat(rows, ignore_index=True)
    close_panel = qlib_frame.pivot(index="datetime", columns="instrument", values="close").sort_index()
    close_returns = close_panel.pct_change().dropna()
    relation_audit = {
        "soxs_soxx_close_return_correlation": float(close_returns["SOXS"].corr(close_returns["SOXX"])),
        "soxs_soxl_close_return_correlation": float(close_returns["SOXS"].corr(close_returns["SOXL"])),
        "common_rows_per_instrument": int(len(ordered_dates)),
        "common_start": ordered_dates[0].date().isoformat(),
        "common_end": ordered_dates[-1].date().isoformat(),
    }
    if relation_audit["soxs_soxx_close_return_correlation"] > -0.80:
        raise ValueError("SOXS does not preserve the expected inverse relationship to SOXX")

    source_payload = {
        "as_of": end,
        "start": start,
        "symbols": list(SYMBOLS),
        "sources": [NASDAQ_HISTORY_URL, "query1/query2.finance.yahoo.com/v8/finance/chart"],
        "adjustment_method": "initial build stitches early Yahoo OHLC to Nasdaq; incremental refresh preserves existing dates and appends Nasdaq OHLC; Yahoo adjusted-close ratio applies distributions to all OHLC",
        "symbol_audits": audits,
        "relationship_audit": relation_audit,
    }
    source_manifest = source_payload | {"hash": stable_hash(source_payload)}
    paths.source_manifest.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    qlib_manifest = build_qlib_store_from_frame(
        qlib_frame,
        provider_uri=paths.provider_uri,
        market="all",
        region="us",
        freq="day",
        instrument_col="instrument",
        datetime_col="datetime",
        field_map={f"${column}": column for column in PRICE_COLUMNS},
        as_of=end,
        source=paths.source_manifest.resolve().as_posix(),
    )
    audit = audit_qlib_provider(
        QlibProviderConfig(provider_uri=paths.provider_uri, region="us", market="all", freq="day"),
        instruments=list(SYMBOLS),
        fields=["$open", "$close"],
        start=relation_audit["common_start"],
        end=end,
    )
    return {
        "source_manifest": paths.source_manifest.resolve().as_posix(),
        "qlib_manifest": qlib_manifest,
        "qlib_audit": audit,
        "relationship_audit": relation_audit,
    }


def _read_existing_symbol(path: Path, *, symbol: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", *PRICE_COLUMNS])
    frame = pd.read_csv(path)
    date_column = "datetime" if "datetime" in frame else "date"
    required = {date_column, *PRICE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"existing {symbol} defensive raw data missing columns: {missing}")
    result = frame.loc[:, [date_column, *PRICE_COLUMNS]].rename(columns={date_column: "date"})
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in PRICE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return (
        result.dropna(subset=["date", *PRICE_COLUMNS])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _merge_overlap_preserving_history(
    existing: pd.DataFrame, update: pd.DataFrame, *, refresh_start: str
) -> pd.DataFrame:
    pd.Timestamp(refresh_start)
    existing_dates = set(pd.to_datetime(existing["date"]))
    update_dates = set(pd.to_datetime(update["date"]))
    if update.empty or not existing_dates.intersection(update_dates):
        raise ValueError("incremental SOXS refresh is missing its validation overlap")
    existing_end = pd.Timestamp(existing["date"].max())
    appended = update.loc[update["date"] > existing_end]
    return (
        pd.concat([existing, appended], ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _write_symbol_history(path: Path, frame: pd.DataFrame, *, symbol: str) -> None:
    """Append new sessions without reserializing already committed prices."""

    existing = _read_existing_symbol(path, symbol=symbol)
    if existing.empty:
        frame.to_csv(path, index=False)
        return
    existing_dates = set(existing["date"])
    frame_dates = set(pd.to_datetime(frame["datetime"]))
    if not existing_dates.issubset(frame_dates):
        raise ValueError(f"incremental {symbol} refresh would remove existing sessions")
    existing_end = pd.Timestamp(existing["date"].max())
    historical_backfills = frame.loc[
        (pd.to_datetime(frame["datetime"]) <= existing_end)
        & ~pd.to_datetime(frame["datetime"]).isin(existing_dates)
    ]
    if not historical_backfills.empty:
        raise ValueError(f"incremental {symbol} refresh requires an explicit historical rebuild")
    appended = frame.loc[pd.to_datetime(frame["datetime"]) > existing_end]
    if not appended.empty:
        appended.to_csv(path, mode="a", header=False, index=False)

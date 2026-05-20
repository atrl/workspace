#!/usr/bin/env python3
"""增量更新 SOXL/SOXX/VIX 日数据到主 CSV，并同步 raw/qlib 本地数据面。

用法:
    python scripts/update_market_data.py [--dry-run] [--force-full]
    python scripts/update_market_data.py --provider polygon
    python scripts/update_market_data.py --local-only

环境依赖:
    yfinance (已包含在 requirements/qlib.txt 中)
    POLYGON_API_KEY (可选；provider=polygon 时用于 SOXL/SOXX，VIX 来自 CBOE)
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.data import refresh_qlib_binary
from src.soxl_quant.settings import CSV_PATH, QLIB_DATA_PATH, RAW_DATA_DIR

RAW_SYMBOL_DIR = RAW_DATA_DIR / "symbols"
RAW_EXTERNAL_DIR = RAW_DATA_DIR / "external"
REMOTE_DATA_PROVIDERS = ("yfinance", "yahoo_api", "polygon")
POLYGON_API_KEY_ENV = "POLYGON_API_KEY"
CBOE_VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
DNS_ERROR_SNIPPETS = (
    "could not resolve host",
    "failed to resolve",
    "name or service not known",
    "nodename nor servname provided",
)


def _is_dns_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(snippet in message for snippet in DNS_ERROR_SNIPPETS)


def _raise_fetch_error(ticker: str, exc: Exception) -> None:
    if _is_dns_failure(exc):
        raise RuntimeError(
            f"Failed to fetch {ticker}: DNS/network unavailable in current environment. "
            "Re-run in a network-enabled environment or use --local-only to sync raw/qlib "
            "from the existing canonical panel."
        ) from exc
    raise RuntimeError(f"Failed to fetch {ticker}: {exc}") from exc


def _fetch_yfinance(ticker: str, start: str | None = None, retries: int = 5) -> pd.DataFrame:
    """从 yfinance 拉取数据，返回标准列名的 DataFrame。

    使用指数退避重试策略应对 yfinance 速率限制。
    """
    import time
    import random
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # Use Ticker.history() instead of download() to avoid batch rate limits
            tk = yf.Ticker(ticker)
            if start:
                df = tk.history(start=start, auto_adjust=True)
            else:
                df = tk.history(period="max", auto_adjust=True)
            if df.empty:
                raise RuntimeError(f"No data returned for {ticker} from yfinance")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            # yfinance Ticker.history returns 'date' or 'index' column
            if "date" not in df.columns and "datetime" in df.columns:
                df = df.rename(columns={"datetime": "date"})
            if "adj_close" in df.columns and "close" not in df.columns:
                df["close"] = df["adj_close"]
            return df[["date", "open", "high", "low", "close", "volume"]].copy()
        except Exception as exc:
            if _is_dns_failure(exc):
                _raise_fetch_error(ticker, exc)
            last_exc = exc
            if attempt < retries:
                # Exponential backoff with jitter: 10s, 20s, 40s, 80s, 160s
                sleep_sec = min(10 * (2 ** (attempt - 1)) + random.randint(0, 5), 180)
                print(f"[yfinance] {ticker} attempt {attempt}/{retries} failed: {exc}. Retrying in {sleep_sec}s...")
                time.sleep(sleep_sec)
    raise RuntimeError(f"Failed to fetch {ticker} from yfinance after {retries} attempts: {last_exc}")


def _fetch_yahoo_api(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance v8 API."""
    fetch_start = start or "2000-01-01"
    end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_ts = int(pd.Timestamp(fetch_start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    try:
        response = requests.get(
            url,
            timeout=60,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()
        data = response.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
    except Exception as exc:
        _raise_fetch_error(ticker, exc)

    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp(ts, unit="s") for ts in timestamps],
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }
    )
    if frame.empty:
        raise RuntimeError(f"No data returned for {ticker} from Yahoo Finance API")
    return frame


def _fetch_polygon(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV from Polygon aggregates."""
    api_key = os.environ.get(POLYGON_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{POLYGON_API_KEY_ENV} environment variable not set")

    fetch_start = start or "2000-01-01"
    fetch_end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{fetch_start}/{fetch_end}"
    try:
        response = requests.get(
            url,
            params={
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": api_key,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        _raise_fetch_error(ticker, exc)

    rows = data.get("results") or []
    if not rows:
        status = data.get("status", "unknown")
        message = data.get("message") or data.get("error") or "no results"
        raise RuntimeError(f"No data returned for {ticker} from Polygon: {status} {message}")

    frame = pd.DataFrame(rows)
    dates = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    result = pd.DataFrame(
        {
            "date": dates.dt.tz_localize(None).dt.normalize(),
            "open": pd.to_numeric(frame["o"], errors="coerce"),
            "high": pd.to_numeric(frame["h"], errors="coerce"),
            "low": pd.to_numeric(frame["l"], errors="coerce"),
            "close": pd.to_numeric(frame["c"], errors="coerce"),
            "volume": pd.to_numeric(frame["v"], errors="coerce").fillna(0).astype(int),
        }
    )
    result = result.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if result.empty:
        raise RuntimeError(f"No usable OHLCV rows returned for {ticker} from Polygon")
    return result


def _fetch_cboe_vix(start: str | None = None) -> pd.DataFrame:
    """Fetch VIX daily close from CBOE's public historical CSV."""
    try:
        frame = pd.read_csv(CBOE_VIX_HISTORY_URL)
    except Exception as exc:
        _raise_fetch_error("VIX", exc)

    columns = {name.upper(): name for name in frame.columns}
    if "DATE" not in columns or "CLOSE" not in columns:
        raise RuntimeError("CBOE VIX history did not include DATE/CLOSE columns")

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[columns["DATE"]]),
            "vix": pd.to_numeric(frame[columns["CLOSE"]], errors="coerce"),
        }
    ).dropna(subset=["date", "vix"])
    if start:
        result = result[result["date"] >= pd.to_datetime(start)]
    result = result.sort_values("date").reset_index(drop=True)
    if result.empty:
        raise RuntimeError("No VIX rows returned from CBOE")
    return result


def _fetch_market_frames(provider: str, start: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if provider == "polygon":
        return _fetch_polygon("SOXL", start=start), _fetch_polygon("SOXX", start=start), _fetch_cboe_vix(start=start)

    fetchers = {
        "yfinance": _fetch_yfinance,
        "yahoo_api": _fetch_yahoo_api,
    }
    if provider not in fetchers:
        raise ValueError(f"Unknown provider: {provider}")
    fetcher = fetchers[provider]
    vix = fetcher("^VIX", start=start).drop(columns=["open", "high", "low", "volume"]).rename(
        columns={"close": "vix"}
    )
    return fetcher("SOXL", start=start), fetcher("SOXX", start=start), vix


def _load_master_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Master CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _normalize_symbol_view(master: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = master[
        [
            "date",
            f"{prefix}_open",
            f"{prefix}_high",
            f"{prefix}_low",
            f"{prefix}_close",
            f"{prefix}_volume",
        ]
    ].copy()
    frame = frame.rename(
        columns={
            f"{prefix}_open": "open",
            f"{prefix}_high": "high",
            f"{prefix}_low": "low",
            f"{prefix}_close": "close",
            f"{prefix}_volume": "volume",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def build_local_surface_frames(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Derive raw/external mirror CSVs from the canonical processed panel."""
    return {
        "data/raw/symbols/SOXL.csv": _normalize_symbol_view(master, "soxl"),
        "data/raw/symbols/SOXX.csv": _normalize_symbol_view(master, "soxx"),
        "data/raw/external/VIX.csv": pd.DataFrame(
            {
                "DATE": pd.to_datetime(master["date"]).dt.strftime("%Y-%m-%d"),
                "CLOSE": master["vix"].astype(float),
            }
        ),
    }


def sync_local_data_surfaces(
    csv_path: Path = CSV_PATH,
    dry_run: bool = False,
) -> dict[str, object]:
    master = _load_master_csv(csv_path)
    frames = build_local_surface_frames(master)
    for relative_path, frame in frames.items():
        target = ROOT / relative_path
        print(f"[sync] {relative_path} rows={len(frame)} last={frame.iloc[-1, 0]}")
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
    return {
        "targets": sorted(frames),
        "rows": len(master),
        "last_date": master["date"].max().strftime("%Y-%m-%d"),
        "dry_run": dry_run,
    }


def _date_range_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"rows": 0, "start_date": None, "last_date": None}
    dates = pd.to_datetime(frame["date"])
    return {
        "rows": int(len(frame)),
        "start_date": dates.min().strftime("%Y-%m-%d"),
        "last_date": dates.max().strftime("%Y-%m-%d"),
    }


def _format_source_ranges(source_ranges: dict[str, dict[str, object]], overlap_last_date: str | None) -> str:
    legs = " ".join(
        f"{name}={summary['start_date']}..{summary['last_date']}({summary['rows']})"
        for name, summary in source_ranges.items()
    )
    return f"{legs} overlap_last_date={overlap_last_date or 'none'}"


def update_master_csv(
    csv_path: Path = CSV_PATH,
    dry_run: bool = False,
    force_full: bool = False,
    provider: str = "yfinance",
) -> dict:
    """增量更新主 CSV。

    Returns:
        {"before_rows": int, "after_rows": int, "new_rows": int, "last_date": str}
    """
    master = _load_master_csv(csv_path)
    before_rows = len(master)
    last_date = master["date"].max()

    if force_full:
        start = None
    else:
        # 拉取最后日期前 5 天起的数据，保证有足够 overlap 去重
        start_dt = last_date - timedelta(days=10)
        start = start_dt.strftime("%Y-%m-%d")

    print(f"[update] last_date={last_date.date()} fetch_start={start or 'max'} provider={provider}")

    soxl, soxx, vix = _fetch_market_frames(provider, start)
    source_ranges = {
        "SOXL": _date_range_summary(soxl),
        "SOXX": _date_range_summary(soxx),
        "VIX": _date_range_summary(vix),
    }

    soxl = soxl.rename(
        columns={
            "open": "soxl_open",
            "high": "soxl_high",
            "low": "soxl_low",
            "close": "soxl_close",
            "volume": "soxl_volume",
        }
    )
    soxx = soxx.rename(
        columns={
            "open": "soxx_open",
            "high": "soxx_high",
            "low": "soxx_low",
            "close": "soxx_close",
            "volume": "soxx_volume",
        }
    )

    merged = (
        soxl.merge(soxx, on="date", how="inner")
        .merge(vix, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    overlap_last_date = None
    if not merged.empty:
        overlap_last_date = pd.to_datetime(merged["date"]).max().strftime("%Y-%m-%d")

    print(f"[update] source_ranges {_format_source_ranges(source_ranges, overlap_last_date)}")

    if merged.empty:
        raise RuntimeError(
            "No overlapping dates for SOXL/SOXX/VIX: "
            f"{_format_source_ranges(source_ranges, overlap_last_date)}"
        )

    # 合并策略：新数据与旧数据 concat 后按 date 去重，保留新数据
    master["date"] = pd.to_datetime(master["date"])
    merged["date"] = pd.to_datetime(merged["date"])

    combined = pd.concat([master, merged], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    after_rows = len(combined)
    new_rows = after_rows - before_rows
    new_last_date = combined["date"].max()

    print(
        f"[update] before={before_rows} after={after_rows} new_rows={new_rows} "
        f"last_date={new_last_date.date()}"
    )

    if dry_run:
        print("[update] dry-run: skip write")
    else:
        combined.to_csv(csv_path, index=False)
        print(f"[update] written {csv_path}")

    return {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "new_rows": new_rows,
        "last_date": new_last_date.strftime("%Y-%m-%d"),
        "dry_run": dry_run,
        "source_ranges": source_ranges,
        "overlap_last_date": overlap_last_date,
    }


def refresh_qlib(
    csv_path: Path = CSV_PATH,
    qlib_dir: Path = QLIB_DATA_PATH,
    dry_run: bool = False,
) -> None:
    if dry_run:
        print(f"[update] dry-run: skip refresh_qlib_binary({qlib_dir})")
        return
    refresh_qlib_binary(csv_path, qlib_dir, leveraged_etf="SOXL", underlying_etf="SOXX", leverage_ratio=3.0)
    print(f"[update] refreshed {qlib_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incremental market data updater for SOXL/SOXX/VIX")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files")
    parser.add_argument("--force-full", action="store_true", help="Force full re-download (slow)")
    parser.add_argument("--skip-qlib", action="store_true", help="Skip qlib binary refresh")
    parser.add_argument(
        "--provider",
        default="yfinance",
        choices=list(REMOTE_DATA_PROVIDERS),
        help="Remote market data provider when fetching new rows",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip remote fetch and only sync raw/qlib surfaces from the existing canonical panel",
    )
    return parser.parse_args()


def main(argv: list[str] | None = None) -> dict:
    if argv is not None:
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0], *argv]
    else:
        original_argv = None
    args = parse_args()
    try:
        if args.local_only:
            state = _load_master_csv(CSV_PATH)
            result = {
                "before_rows": len(state),
                "after_rows": len(state),
                "new_rows": 0,
                "last_date": state["date"].max().strftime("%Y-%m-%d"),
                "dry_run": args.dry_run,
                "mode": "local-only",
            }
            print(
                f"[update] local-only rows={result['after_rows']} "
                f"last_date={result['last_date']}"
            )
        else:
            result = update_master_csv(
                dry_run=args.dry_run,
                force_full=args.force_full,
                provider=args.provider,
            )
        sync_local_data_surfaces(dry_run=args.dry_run)
        if not args.skip_qlib:
            refresh_qlib(dry_run=args.dry_run)
        return result
    finally:
        if original_argv is not None:
            sys.argv = original_argv


if __name__ == "__main__":
    main()

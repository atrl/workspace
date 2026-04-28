#!/usr/bin/env python3
"""Update SOXL/SOXX/VIX data using direct Yahoo Finance API (bypasses yfinance rate limits)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.data import refresh_qlib_binary
from src.soxl_quant.settings import CSV_PATH

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance v8 API."""
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&period1={start_ts}&period2={end_ts}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.json()
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": [pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d") for ts in timestamps],
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote["volume"],
    })
    return df


def main() -> None:
    print("[update_yahoo_api] Loading existing data...")
    master = pd.read_csv(CSV_PATH, parse_dates=["date"])
    last_date = master["date"].max()
    print(f"[update_yahoo_api] Last date: {last_date.date()}")

    fetch_start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
    fetch_end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[update_yahoo_api] Fetching from {fetch_start} to {fetch_end}")

    soxl = fetch_yahoo("SOXL", fetch_start, fetch_end)
    soxx = fetch_yahoo("SOXX", fetch_start, fetch_end)
    vix = fetch_yahoo("^VIX", fetch_start, fetch_end)

    soxl = soxl.rename(columns={"open": "soxl_open", "high": "soxl_high", "low": "soxl_low", "close": "soxl_close", "volume": "soxl_volume"})
    soxx = soxx.rename(columns={"open": "soxx_open", "high": "soxx_high", "low": "soxx_low", "close": "soxx_close", "volume": "soxx_volume"})
    vix = vix[["date", "close"]].rename(columns={"close": "vix"})

    merged = soxl.merge(soxx, on="date", how="inner").merge(vix, on="date", how="inner")
    merged["date"] = pd.to_datetime(merged["date"])

    # Combine with master, drop duplicates, keep new
    combined = pd.concat([master, merged], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    before_rows = len(master)
    after_rows = len(combined)
    new_rows = after_rows - before_rows

    combined.to_csv(CSV_PATH, index=False)
    print(f"[update_yahoo_api] Saved: {before_rows} -> {after_rows} rows ({new_rows} new)")
    print(f"[update_yahoo_api] New last date: {combined['date'].max().date()}")

    print("[update_yahoo_api] Refreshing qlib binary...")
    refresh_qlib_binary(str(CSV_PATH))
    print("[update_yahoo_api] Done")


if __name__ == "__main__":
    main()

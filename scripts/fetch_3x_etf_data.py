#!/usr/bin/env python3
"""Fetch daily OHLCV for 3x leveraged ETFs from multiple data sources.

Sources:
    yfinance      - free, unlimited (default)
    alphavantage  - requires ALPHA_VANTAGE_API_KEY env var, 5 calls/minute on free tier

Outputs CSVs compatible with the SOXL quant pipeline under:
    data/generalization/{source}/{ETF}_with_underlying_vix.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.settings import ETF_UNIVERSE

DATA_DIR = ROOT / "data" / "generalization"

ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")


def _sleep_for_rate_limit():
    """Alpha Vantage free tier: 5 calls per minute."""
    time.sleep(13)


def fetch_yfinance(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]
    return df[["date", "open", "high", "low", "close", "volume"]].copy()


def fetch_alphavantage(ticker: str) -> pd.DataFrame:
    if not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY environment variable not set")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}"
        f"&outputsize=full"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "Time Series (Daily)" not in data:
        raise RuntimeError(f"Alpha Vantage error for {ticker}: {json.dumps(data, indent=2)}")
    records = []
    for date_str, values in data["Time Series (Daily)"].items():
        records.append(
            {
                "date": pd.to_datetime(date_str),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"]),
            }
        )
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def fetch_vix_yfinance() -> pd.DataFrame:
    return fetch_yfinance("^VIX").drop(columns=["open", "high", "low", "volume"]).rename(columns={"close": "vix"})


def build_etf_csv(etf: str, underlying: str, source: str) -> pd.DataFrame:
    if source == "yfinance":
        edf = fetch_yfinance(etf)
        udf = fetch_yfinance(underlying)
    elif source == "alphavantage":
        print(f"Fetching {etf} from Alpha Vantage ...")
        edf = fetch_alphavantage(etf)
        _sleep_for_rate_limit()
        print(f"Fetching {underlying} from Alpha Vantage ...")
        udf = fetch_alphavantage(underlying)
        _sleep_for_rate_limit()
    else:
        raise ValueError(f"Unknown source: {source}")

    vdf = fetch_vix_yfinance()

    edf = edf.rename(
        columns={
            "open": f"{etf.lower()}_open",
            "high": f"{etf.lower()}_high",
            "low": f"{etf.lower()}_low",
            "close": f"{etf.lower()}_close",
            "volume": f"{etf.lower()}_volume",
        }
    )
    udf = udf.rename(
        columns={
            "open": f"{underlying.lower()}_open",
            "high": f"{underlying.lower()}_high",
            "low": f"{underlying.lower()}_low",
            "close": f"{underlying.lower()}_close",
            "volume": f"{underlying.lower()}_volume",
        }
    )

    merged = (
        edf.merge(udf, on="date", how="inner")
        .merge(vdf, on="date", how="inner")
        .sort_values("date")
    )
    if merged.empty:
        raise RuntimeError(f"No overlapping dates for {etf}/{underlying}/VIX")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alphavantage"], help="Data source")
    parser.add_argument("--etf", help="Single ETF to fetch (default: all)")
    args = parser.parse_args()

    out_dir = DATA_DIR / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {args.etf.upper(): ETF_UNIVERSE[args.etf.upper()]} if args.etf else ETF_UNIVERSE

    for etf, meta in targets.items():
        out_path = out_dir / f"{etf.lower()}_with_underlying_vix.csv"
        try:
            df = build_etf_csv(etf, meta["underlying"], args.source)
            df.to_csv(out_path, index=False)
            print(
                f"Saved {out_path} ({len(df)} rows, {df['date'].min().date()} ~ {df['date'].max().date()})"
            )
        except Exception as exc:
            print(f"ERROR processing {etf}: {exc}")


if __name__ == "__main__":
    main()

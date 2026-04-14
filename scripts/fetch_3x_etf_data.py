#!/usr/bin/env python3
"""Fetch daily OHLCV for 3x leveraged ETFs and their underlying indices.

Outputs CSVs compatible with the SOXL quant pipeline under:
    data/generalization/{etf}_with_underlying_vix.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "generalization"

ETF_UNIVERSE = {
    "SOXL": {"underlying": "SOXX", "leverage": 3.0},
    "TQQQ": {"underlying": "QQQ", "leverage": 3.0},
    "UPRO": {"underlying": "SPY", "leverage": 3.0},
    "SPXL": {"underlying": "SPY", "leverage": 3.0},
    "TECL": {"underlying": "XLK", "leverage": 3.0},
}


def fetch_ticker(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # yfinance with auto_adjust returns only close-like prices
    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{ticker} missing columns: {missing}")
    return df[["date", "open", "high", "low", "close", "volume"]].copy()


def build_etf_csv(etf: str, underlying: str, leverage: float) -> pd.DataFrame:
    print(f"Fetching {etf} ...")
    edf = fetch_ticker(etf)
    print(f"Fetching {underlying} ...")
    udf = fetch_ticker(underlying)
    print(f"Fetching ^VIX ...")
    vdf = fetch_ticker("^VIX")

    # Rename columns
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
    vdf = vdf.rename(columns={"close": "vix"}).drop(columns=["open", "high", "low", "volume"])

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
    parser.add_argument("--etf", help="Single ETF to fetch (default: all)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    targets = {args.etf.upper(): ETF_UNIVERSE[args.etf.upper()]} if args.etf else ETF_UNIVERSE

    for etf, meta in targets.items():
        out_path = DATA_DIR / f"{etf.lower()}_with_underlying_vix.csv"
        try:
            df = build_etf_csv(etf, meta["underlying"], meta["leverage"])
            df.to_csv(out_path, index=False)
            print(f"Saved {out_path} ({len(df)} rows, {df['date'].min().date()} ~ {df['date'].max().date()})")
        except Exception as exc:
            print(f"ERROR processing {etf}: {exc}")


if __name__ == "__main__":
    main()

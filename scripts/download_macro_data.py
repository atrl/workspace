#!/usr/bin/env python3
"""Download macro leading-indicator CSVs from FRED for SOXL feature engineering."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT / "data" / "raw" / "external"

# FRED series IDs → descriptive file names
FRED_SERIES = {
    "BAA": "baa_yield.csv",                      # Moody's Baa Corp Yield (for credit spread)
    "DGS10": "treasury_10y.csv",                 # 10-Year Treasury Constant Maturity
    "T10Y2Y": "yield_curve_10y2y.csv",           # 10Y-2Y spread
    "T10Y3M": "yield_curve_10y3m.csv",           # 10Y-3M spread
    "DTWEXBGS": "dxy_broad.csv",                 # Trade Weighted Dollar Index
    "DCOILWTICO": "oil_wti.csv",                 # WTI Crude Oil
    "GOLDAMGBD228NLBM": "gold_london.csv",       # Gold fixing price (AM)
}

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_series(series_id: str, out_path: Path) -> pd.DataFrame | None:
    url = f"{FRED_BASE}?id={series_id}"
    print(f"[fetch] {series_id} -> {out_path.name}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_text(resp.text, encoding="utf-8")
    df = pd.read_csv(out_path)
    # Rename columns for consistency
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    value_col = series_id.lower()
    if value_col not in df.columns:
        # FRED sometimes uses the series id as column name
        for c in df.columns:
            if series_id.lower() in c:
                value_col = c
                break
        else:
            value_col = df.columns[-1]
    df = df[[value_col]].rename(columns={value_col: "value"})
    # FRED uses "." for missing values
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    # Write cleaned CSV back (overwriting raw FRED response)
    df.to_csv(out_path)
    print(f"  rows={len(df)}  range={df.index.min().date()} ~ {df.index.max().date()}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    for series_id, filename in FRED_SERIES.items():
        out_path = EXTERNAL_DIR / filename
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path.name} already exists")
            ok += 1
            continue
        try:
            fetch_series(series_id, out_path)
            ok += 1
        except Exception as exc:
            print(f"[error] {series_id}: {exc}")
            fail += 1
    print(f"\nDone: {ok} succeeded, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

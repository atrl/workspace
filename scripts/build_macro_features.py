#!/usr/bin/env python3
"""Post-process FRED downloads into aligned macro feature CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT / "data" / "raw" / "external"


def _load_fred_csv(path: Path, value_col: str = "value") -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    # Find value column
    if value_col in df.columns:
        return df[value_col].astype(float)
    # FRED sometimes names column after series id
    for c in df.columns:
        if c not in ("observation_date", "date"):
            return df[c].astype(float)
    raise ValueError(f"No value column in {path}")


def build_credit_spread() -> None:
    """BAA - DGS10 as a proxy for credit spread."""
    baa = _load_fred_csv(EXTERNAL_DIR / "baa_yield.csv")
    dgs10 = _load_fred_csv(EXTERNAL_DIR / "treasury_10y.csv")
    # Align to trading days (intersection)
    aligned = pd.concat([baa, dgs10], axis=1).dropna()
    aligned.columns = ["baa", "dgs10"]
    spread = (aligned["baa"] - aligned["dgs10"]).dropna()
    out = EXTERNAL_DIR / "credit_spread_baa.csv"
    spread.to_frame(name="credit_spread").to_csv(out)
    print(f"[build] credit_spread_baa.csv  rows={len(spread)}  range={spread.index.min().date()} ~ {spread.index.max().date()}")


def inspect_all() -> None:
    files = [
        ("credit_spread_baa.csv", "credit_spread"),
        ("yield_curve_10y2y.csv", "value"),
        ("yield_curve_10y3m.csv", "value"),
        ("dxy_broad.csv", "value"),
        ("oil_wti.csv", "value"),
    ]
    for fname, col in files:
        path = EXTERNAL_DIR / fname
        if not path.exists():
            print(f"[missing] {fname}")
            continue
        s = _load_fred_csv(path, col)
        print(f"[ok] {fname:30s} rows={len(s):6d}  range={s.index.min().date()} ~ {s.index.max().date()}")


def main() -> int:
    build_credit_spread()
    inspect_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())

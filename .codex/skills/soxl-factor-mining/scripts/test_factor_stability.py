"""Rolling stability analysis for SOXL features.

Computes rolling IC statistics over time to detect regime-dependent decay.

Usage:
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/test_factor_stability.py \
      --feature RB_PANIC_SCORE_PROXY --window 63 --backfill
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from soxl_quant.data import build_feature_frame, init_qlib
from pathlib import Path

DB_PATH = ROOT / "data" / "soxl_factor_library.db"


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS feature_stability (
            feature_name TEXT,
            window_days INTEGER,
            stability_score REAL,
            ic_recent_vs_full REAL,
            max_dd_days INTEGER,
            PRIMARY KEY (feature_name, window_days)
        );
        """
    )
    return conn


def rolling_ic(factor: pd.Series, forward_ret: pd.Series, window: int = 63) -> pd.Series:
    df = pd.DataFrame({"factor": factor, "ret": forward_ret}).dropna()
    return df["factor"].rolling(window).corr(df["ret"]).dropna()


def main():
    parser = argparse.ArgumentParser(description="Rolling stability analysis")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--window", type=int, default=63)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()

    init_qlib(Path("qlib_soxl_binary"))

    print(f"Loading data ({args.start} ~ {args.end})...")
    feature_df, price = build_feature_frame(args.start, args.end, feature_packs=("rulebase_signals", "academic_signals"))
    feature_df = feature_df.droplevel("instrument")
    price = price.droplevel("instrument")

    if args.feature not in feature_df.columns:
        print(f"ERROR: {args.feature} not found")
        sys.exit(1)

    factor = feature_df[args.feature]
    forward_ret = (price.shift(-args.horizon) / price - 1.0).fillna(0.0)

    ric = rolling_ic(factor, forward_ret, args.window)

    if len(ric) == 0:
        print("No rolling IC data")
        return

    stability_score = (ric > 0).mean()
    recent = ric.tail(args.window)
    full = ric
    ic_recent_vs_full = recent.mean() / full.mean() if full.mean() != 0 else 0.0

    # Max consecutive negative IC days
    negative = (ric < 0).astype(int)
    max_dd_days = 0
    current = 0
    for v in negative:
        if v:
            current += 1
            max_dd_days = max(max_dd_days, current)
        else:
            current = 0

    result = {
        "feature_name": args.feature,
        "window_days": args.window,
        "stability_score": float(stability_score),
        "ic_recent_vs_full": float(ic_recent_vs_full),
        "max_dd_days": int(max_dd_days),
    }

    print(f"\nStability: {args.feature} (horizon={args.horizon}d, window={args.window}d)")
    print(f"  stability_score (positive ratio): {stability_score:.2%}")
    print(f"  ic_recent_vs_full: {ic_recent_vs_full:.2f}")
    print(f"  max_dd_days: {max_dd_days}")

    if args.backfill:
        conn = _ensure_db()
        conn.execute(
            """INSERT INTO feature_stability
               (feature_name, window_days, stability_score, ic_recent_vs_full, max_dd_days)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(feature_name, window_days) DO UPDATE SET
                 stability_score=excluded.stability_score,
                 ic_recent_vs_full=excluded.ic_recent_vs_full,
                 max_dd_days=excluded.max_dd_days""",
            (args.feature, args.window, stability_score, ic_recent_vs_full, max_dd_days),
        )
        conn.commit()
        conn.close()
        print("[Backfill] Wrote stability record to DB")

    out_dir = ROOT / "docs" / "factor_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"stability_{args.feature}.csv"
    pd.DataFrame([result]).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

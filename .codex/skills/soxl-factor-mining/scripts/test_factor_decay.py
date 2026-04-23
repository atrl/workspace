"""Multi-horizon IC decay analysis for SOXL features.

Computes time-series IC at horizons 1d/3d/5d/10d/20d to understand
factor persistence. Critical for setting optimal rebalance frequency.

Usage:
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/test_factor_decay.py \
      --feature RB_PANIC_SCORE_PROXY --backfill
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
from test_factor_ic import timeseries_ic_stats, compute_forward_returns, _ensure_db, backfill_db


def main():
    parser = argparse.ArgumentParser(description="IC decay analysis")
    parser.add_argument("--feature", required=True, help="Feature name")
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]

    init_qlib(Path("qlib_soxl_binary"))

    print(f"Loading data ({args.start} ~ {args.end})...")
    feature_df, price = build_feature_frame(args.start, args.end, feature_packs=("rulebase_signals",))
    feature_df = feature_df.droplevel("instrument")
    price = price.droplevel("instrument")

    if args.feature not in feature_df.columns:
        print(f"ERROR: {args.feature} not found")
        sys.exit(1)

    factor = feature_df[args.feature]
    forward_returns = compute_forward_returns(price, horizons)

    results = []
    print(f"\nIC Decay: {args.feature}")
    for h in horizons:
        stats_dict = timeseries_ic_stats(factor, forward_returns[h])
        stats_dict["feature_name"] = args.feature
        stats_dict["horizon"] = h
        results.append(stats_dict)
        print(
            f"  {h:>2}d  IC={stats_dict['ic_mean']:+.4f}  t={stats_dict['ic_t']:+.2f}  "
            f"pos_ratio={stats_dict.get('ic_positive_ratio', 0):.2%}"
        )

    # Decay ratio
    ic_1d = next((r["ic_mean"] for r in results if r["horizon"] == 1), None)
    ic_5d = next((r["ic_mean"] for r in results if r["horizon"] == 5), None)
    if ic_1d and ic_5d and ic_1d != 0:
        decay_ratio = ic_5d / ic_1d
        weekly_suitable = abs(decay_ratio) >= 0.5
        print(f"\n  5d/1d decay ratio: {decay_ratio:.2f}")
        print(f"  Weekly suitable: {weekly_suitable}")

    if args.backfill:
        conn = _ensure_db()
        try:
            backfill_db(conn, results, args.start, args.end)
        finally:
            conn.close()

    out_dir = ROOT / "docs" / "factor_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"decay_{args.feature}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

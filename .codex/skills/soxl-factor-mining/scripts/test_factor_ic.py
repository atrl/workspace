"""Time-series IC test for SOXL single-asset features.

Computes rolling time-series correlation between feature and forward returns
at multiple horizons (1d/3d/5d/10d/20d). Results are saved to CSV and
optionally written to the factor library.

Usage:
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/test_factor_ic.py \
      --feature RB_PANIC_SCORE_PROXY --horizons 1,5,10,20
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/test_factor_ic.py \
      --all-from-db --backfill
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
        CREATE TABLE IF NOT EXISTS feature_ic_tests (
            feature_name TEXT,
            horizon_days INTEGER,
            test_start TEXT,
            test_end TEXT,
            n_days INTEGER,
            ic_mean REAL,
            ic_std REAL,
            ic_t REAL,
            ic_positive_ratio REAL,
            PRIMARY KEY (feature_name, horizon_days)
        );
        """
    )
    return conn


def compute_forward_returns(price: pd.Series, horizons: list[int]) -> dict[int, pd.Series]:
    """Compute forward returns at multiple horizons."""
    returns = {}
    for h in horizons:
        returns[h] = (price.shift(-h) / price - 1.0).fillna(0.0)
    return returns


def timeseries_ic_stats(factor: pd.Series, forward_ret: pd.Series, min_obs: int = 30) -> dict:
    """Compute time-series IC statistics."""
    aligned = pd.DataFrame({"factor": factor, "ret": forward_ret}).dropna()
    if len(aligned) < min_obs:
        return {
            "n_days": len(aligned),
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_t": np.nan,
            "ic_positive_ratio": np.nan,
        }

    # Daily IC using rolling 21-day window
    rolling_ic = aligned["factor"].rolling(21).corr(aligned["ret"]).dropna()
    n = len(rolling_ic)
    if n < min_obs:
        return {
            "n_days": n,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_t": np.nan,
            "ic_positive_ratio": np.nan,
        }

    ic_mean = rolling_ic.mean()
    ic_std = rolling_ic.std()
    ic_t = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
    ic_positive_ratio = (rolling_ic > 0).mean()

    return {
        "n_days": n,
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "ic_t": float(ic_t),
        "ic_positive_ratio": float(ic_positive_ratio),
    }


def backfill_db(conn: sqlite3.Connection, results: list[dict], test_start: str, test_end: str) -> None:
    for r in results:
        conn.execute(
            """INSERT INTO feature_ic_tests
               (feature_name, horizon_days, test_start, test_end, n_days, ic_mean, ic_std, ic_t, ic_positive_ratio)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(feature_name, horizon_days) DO UPDATE SET
                 test_start=excluded.test_start,
                 test_end=excluded.test_end,
                 n_days=excluded.n_days,
                 ic_mean=excluded.ic_mean,
                 ic_std=excluded.ic_std,
                 ic_t=excluded.ic_t,
                 ic_positive_ratio=excluded.ic_positive_ratio""",
            (
                r["feature_name"],
                r["horizon"],
                test_start,
                test_end,
                r["n_days"],
                r["ic_mean"],
                r["ic_std"],
                r["ic_t"],
                r["ic_positive_ratio"],
            ),
        )
    conn.commit()
    print(f"[Backfill] Wrote {len(results)} IC test records to DB")


def test_single_feature(
    feature_name: str,
    feature_df: pd.DataFrame,
    price: pd.Series,
    horizons: list[int],
) -> list[dict]:
    if feature_name not in feature_df.columns:
        raise KeyError(f"Feature '{feature_name}' not in feature_df columns: {list(feature_df.columns)}")

    factor = feature_df[feature_name]
    forward_returns = compute_forward_returns(price, horizons)
    results = []

    print(f"\nTesting feature: {feature_name}")
    print(f"  Data points: {len(factor)}")

    for h in horizons:
        stats_dict = timeseries_ic_stats(factor, forward_returns[h])
        stats_dict["feature_name"] = feature_name
        stats_dict["horizon"] = h
        results.append(stats_dict)
        sig = "✓" if stats_dict.get("ic_positive_ratio", 0) > 0.55 and stats_dict.get("ic_t", 0) > 2.0 else "✗"
        print(
            f"  {h:>2}d  IC={stats_dict['ic_mean']:+.4f}  t={stats_dict['ic_t']:+.2f}  "
            f"pos_ratio={stats_dict.get('ic_positive_ratio', 0):.2%}  {sig}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Time-series IC test for SOXL features")
    parser.add_argument("--feature", help="Single feature name to test")
    parser.add_argument("--all-from-db", action="store_true", help="Test all Candidate features from DB")
    parser.add_argument("--horizons", default="1,5,10,20", help="Comma-separated horizons")
    parser.add_argument("--start", default="2019-01-01", help="Test start date")
    parser.add_argument("--end", default="2026-04-01", help="Test end date")
    parser.add_argument("--backfill", action="store_true", help="Write results to DB")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]

    # Init qlib
    init_qlib(Path("qlib_soxl_binary"))

    # Load data
    print(f"Loading SOXL feature frame ({args.start} ~ {args.end})...")
    feature_df, price = build_feature_frame(args.start, args.end, feature_packs=("rulebase_signals", "academic_signals"))
    feature_df = feature_df.droplevel("instrument")
    price = price.droplevel("instrument")
    print(f"  Loaded: {len(feature_df)} rows, {len(feature_df.columns)} features")

    features_to_test = []
    if args.feature:
        features_to_test = [args.feature]
    elif args.all_from_db:
        conn = _ensure_db()
        rows = conn.execute("SELECT name FROM features WHERE status = 'Candidate'").fetchall()
        features_to_test = [r[0] for r in rows]
        conn.close()
        print(f"Found {len(features_to_test)} Candidate features in DB")
    else:
        # Test all columns in feature_df that start with RB_
        features_to_test = [c for c in feature_df.columns if c.startswith("RB_")]
        print(f"Auto-detected {len(features_to_test)} RB_ features")

    all_results = []
    for fname in features_to_test:
        if fname not in feature_df.columns:
            print(f"  SKIP: {fname} not in feature frame")
            continue
        try:
            res = test_single_feature(fname, feature_df, price, horizons)
            all_results.extend(res)
        except Exception as e:
            print(f"  ERROR testing {fname}: {e}")

    # Save CSV
    out_dir = ROOT / "docs" / "factor_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "factor_ic_results.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\nSaved IC results: {out_csv}")

    # Backfill
    if args.backfill and all_results:
        conn = _ensure_db()
        try:
            backfill_db(conn, all_results, args.start, args.end)
        finally:
            conn.close()


if __name__ == "__main__":
    main()

"""End-to-end factor mining pipeline for SOXL.

Orchestrates: hypothesis → feature generation → IC test → signal backtest → correlation check → DB入库

Usage:
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/mine_factors.py \
      --feature RB_PANIC_SCORE_PROXY --full
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/mine_factors.py \
      --all-rulebase --full --backfill
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SKILL = ROOT / ".codex" / "skills" / "soxl-factor-mining" / "scripts"


def run_script(name: str, *args: str) -> int:
    cmd = [sys.executable, str(SKILL / name), *args]
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="SOXL factor mining pipeline")
    parser.add_argument("--feature", help="Single feature to mine")
    parser.add_argument("--all-rulebase", action="store_true", help="Process all rulebase-derived features")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (IC + backtest + decay + stability)")
    parser.add_argument("--backfill", action="store_true", help="Write results to DB")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-04-01")
    args = parser.parse_args()

    # First, ensure rulebase features are registered in DB
    run_script("extract_rulebase_signals.py")

    features = []
    if args.feature:
        features = [args.feature]
    elif args.all_rulebase:
        # Import to get the list
        sys.path.insert(0, str(SKILL))
        from extract_rulebase_signals import RULEBASE_HYPOTHESES  # noqa: E402
        features = [h["name"] for h in RULEBASE_HYPOTHESES]
        print(f"Will process {len(features)} rulebase features")
    else:
        print("Error: specify --feature or --all-rulebase")
        sys.exit(1)

    # Register all features in DB as Candidate
    db_path = ROOT / "data" / "soxl_factor_library.db"
    conn = sqlite3.connect(str(db_path))
    for h in RULEBASE_HYPOTHESES:
        conn.execute(
            """INSERT INTO features (name, expression, category, source, status, notes)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO NOTHING""",
            (h["name"], h["expression"], h["category"], h["source"], "Candidate", h.get("market_logic", "")),
        )
    conn.commit()
    conn.close()

    for fname in features:
        print(f"\n{'='*80}")
        print(f"PIPELINE: {fname}")
        print(f"{'='*80}")

        # 1. IC test
        run_script("test_factor_ic.py", "--feature", fname, "--start", args.start, "--end", args.end)

        if args.full:
            # 2. Signal backtest (sweep)
            run_script("backtest_signal.py", "--feature", fname, "--sweep", "--start", args.start, "--end", args.end)

            # 3. IC decay
            run_script("test_factor_decay.py", "--feature", fname, "--start", args.start, "--end", args.end)

            # 4. Stability
            run_script("test_factor_stability.py", "--feature", fname, "--start", args.start, "--end", args.end)

            # 5. Regime slice
            run_script(
                "regime_slice_analysis.py",
                "--feature", fname,
                "--start", args.start, "--end", args.end,
            )

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

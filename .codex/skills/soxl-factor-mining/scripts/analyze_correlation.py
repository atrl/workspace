"""Analyze correlation between candidate features and existing CORE features.

Usage:
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/analyze_correlation.py \
      --candidates RB_PANIC_SCORE_PROXY,RB_DAILY_DROP_SEVERITY
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from soxl_quant.data import build_feature_frame, CORE_FEATURE_COLUMNS, init_qlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Feature correlation analysis")
    parser.add_argument("--candidates", required=True, help="Comma-separated candidate feature names")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument("--threshold", type=float, default=0.80, help="Correlation alert threshold")
    args = parser.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",")]

    init_qlib(Path("qlib_soxl_binary"))

    print(f"Loading feature frame ({args.start} ~ {args.end})...")
    feature_df, _ = build_feature_frame(args.start, args.end, feature_packs=("rulebase_signals", "academic_signals"))
    feature_df = feature_df.droplevel("instrument")

    # Select candidate + core features
    core_cols = [c for c in CORE_FEATURE_COLUMNS if c in feature_df.columns]
    all_cols = list(dict.fromkeys(core_cols + candidates))
    df = feature_df[all_cols].dropna()

    corr = df.corr(method="spearman")

    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS: Candidates vs CORE_FEATURE_COLUMNS")
    print(f"{'='*80}")

    alerts = []
    for cand in candidates:
        if cand not in corr.columns:
            print(f"  SKIP: {cand} not found")
            continue
        print(f"\n  {cand}:")
        for core in core_cols:
            if core == cand:
                continue
            rho = corr.loc[cand, core]
            flag = " ⚠️ ALERT" if abs(rho) > args.threshold else ""
            if abs(rho) > args.threshold:
                alerts.append((cand, core, rho))
            print(f"    vs {core:35s}: rho={rho:+.4f}{flag}")

    if alerts:
        print(f"\n{'='*80}")
        print(f"HIGH CORRELATION ALERTS (|rho| > {args.threshold}):")
        for cand, core, rho in alerts:
            print(f"  {cand} <-> {core}: {rho:+.4f}")
    else:
        print(f"\nNo high correlations found (all |rho| ≤ {args.threshold})")

    # Save full correlation matrix
    out_dir = ROOT / "docs" / "factor_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "feature_correlation_matrix.csv"
    corr.to_csv(out_csv)
    print(f"\nSaved correlation matrix: {out_csv}")


if __name__ == "__main__":
    main()

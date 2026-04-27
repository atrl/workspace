#!/usr/bin/env python3
"""Validate data leakage fix in data.py by comparing regime_risk distributions.

Usage:
    PYTHONPATH=. qlib_venv/bin/python scripts/validate_data_leakage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.data import build_dataset, init_qlib, segment_frame
from src.soxl_quant.settings import QLIB_DATA_PATH


def compute_regime_risk_stats(series: pd.Series) -> dict:
    """Compute summary statistics for regime_risk series."""
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
        "q95": float(series.quantile(0.95)),
        "q99": float(series.quantile(0.99)),
    }


def find_peak_dates(series: pd.Series, n: int = 5) -> list[dict]:
    """Find top N peak dates and their values."""
    top = series.nlargest(n)
    return [{"date": str(d), "value": float(v)} for d, v in top.items()]


def main() -> None:
    print("[validate] Initializing qlib...")
    init_qlib(QLIB_DATA_PATH)

    print("[validate] Building dataset...")
    dataset = build_dataset()
    # Check both train and test sets
    train_df = segment_frame(dataset, "train")
    test_df = segment_frame(dataset, "test")

    print(f"[validate] Train set shape: {train_df.shape}")
    print(f"[validate] Train set range: {train_df.index[0]} -> {train_df.index[-1]}")
    print(f"[validate] Test set shape: {test_df.shape}")
    print(f"[validate] Test set range: {test_df.index[0]} -> {test_df.index[-1]}")

    # Extract regime_risk from both sets
    for name, df in [("train", train_df), ("test", test_df)]:
        if "regime_risk" not in df.columns:
            print(f"[validate] ERROR: 'regime_risk' column not found in {name}_df")
            print(f"[validate] Available columns: {list(df.columns)}")
            sys.exit(1)

    train_rr = train_df["regime_risk"]
    test_rr = test_df["regime_risk"]
    regime_risk = pd.concat([train_rr, test_rr]).sort_index()

    # Compute overall stats
    stats = compute_regime_risk_stats(regime_risk)
    print("\n[validate] === Regime Risk Statistics (FIXED) ===")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Check early period (2015, in train set) - should NOT be inflated by future knowledge
    early_mask = regime_risk.index < "2016-01-01"
    early_stats = compute_regime_risk_stats(regime_risk[early_mask])
    print("\n[validate] === Early Period (2015, pre-2016) ===")
    for k, v in early_stats.items():
        print(f"  {k}: {v:.4f}")

    # Check mid period (2017-2019, calm market)
    calm_mask = (regime_risk.index >= "2017-01-01") & (regime_risk.index <= "2019-12-31")
    calm_stats = compute_regime_risk_stats(regime_risk[calm_mask])
    print("\n[validate] === Calm Period (2017-2019) ===")
    for k, v in calm_stats.items():
        print(f"  {k}: {v:.4f}")

    # Check crisis periods
    covid_mask = (regime_risk.index >= "2020-02-01") & (regime_risk.index <= "2020-05-01")
    bear_mask = (regime_risk.index >= "2022-01-01") & (regime_risk.index <= "2022-12-31")

    covid_stats = compute_regime_risk_stats(regime_risk[covid_mask])
    bear_stats = compute_regime_risk_stats(regime_risk[bear_mask])

    print("\n[validate] === COVID Period (2020-02 to 2020-05) ===")
    for k, v in covid_stats.items():
        print(f"  {k}: {v:.4f}")

    print("\n[validate] === Bear Market (2022) ===")
    for k, v in bear_stats.items():
        print(f"  {k}: {v:.4f}")

    # Find peak dates
    peaks = find_peak_dates(regime_risk, 10)
    print("\n[validate] === Top 10 Peak Dates ===")
    for p in peaks:
        print(f"  {p['date']}: {p['value']:.4f}")

    # Validation checks
    print("\n[validate] === Validation Checks ===")
    checks = []

    # Check 1: Early period mean should not be too high (was ~0.55 with leakage)
    # 2015 had real stress (China crash), so 0.45 is reasonable
    check1 = early_stats["mean"] < 0.48
    checks.append(("Early period mean < 0.48", check1, early_stats["mean"]))
    print(f"  {'PASS' if check1 else 'FAIL'}: Early period mean < 0.48 (got {early_stats['mean']:.4f})")

    # Check 2: COVID should have higher regime_risk than calm period
    check2 = covid_stats["mean"] > calm_stats["mean"]
    checks.append(("COVID > Calm period", check2, f"{covid_stats['mean']:.4f} vs {calm_stats['mean']:.4f}"))
    print(f"  {'PASS' if check2 else 'FAIL'}: COVID mean > Calm mean ({covid_stats['mean']:.4f} vs {calm_stats['mean']:.4f})")

    # Check 3: Bear market should have higher regime_risk than calm period
    check3 = bear_stats["mean"] > calm_stats["mean"]
    checks.append(("Bear > Calm period", check3, f"{bear_stats['mean']:.4f} vs {calm_stats['mean']:.4f}"))
    print(f"  {'PASS' if check3 else 'FAIL'}: Bear mean > Calm mean ({bear_stats['mean']:.4f} vs {calm_stats['mean']:.4f})")

    # Check 4: Top peaks should include crisis periods
    peak_dates = [p["date"] for p in peaks]
    crisis_in_top = any(d.startswith(("2020-", "2022-")) for d in peak_dates[:5])
    checks.append(("Crisis in top 5 peaks", crisis_in_top, peak_dates[:5]))
    print(f"  {'PASS' if crisis_in_top else 'FAIL'}: Crisis dates in top 5 peaks ({peak_dates[:5]})")

    # Check 5: Early period should have lower regime_risk than calm period (no future inflation)
    check5 = early_stats["mean"] <= calm_stats["mean"] + 0.1  # allow small margin
    checks.append(("Early <= Calm + 0.1", check5, f"{early_stats['mean']:.4f} vs {calm_stats['mean']:.4f}"))
    print(f"  {'PASS' if check5 else 'FAIL'}: Early mean <= Calm mean + 0.1 ({early_stats['mean']:.4f} vs {calm_stats['mean']:.4f})")

    # Write report
    report = {
        "overall": stats,
        "early_2015": early_stats,
        "calm_2017_2019": calm_stats,
        "covid_2020": covid_stats,
        "bear_2022": bear_stats,
        "top_peaks": peaks,
        "checks": [{"name": c[0], "pass": c[1], "detail": c[2]} for c in checks],
        "all_pass": all(c[1] for c in checks),
    }

    report_path = ROOT / "runtime" / "data" / "leakage_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[validate] Report written to {report_path}")

    if report["all_pass"]:
        print("\n[validate] ALL CHECKS PASSED ✓")
    else:
        print("\n[validate] SOME CHECKS FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()

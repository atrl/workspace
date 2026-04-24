"""Regime-sliced signal analysis.

Tests signal performance separately in bull / base / stress regimes.

Usage:
    qlib_venv/bin/python .codex/skills/soxl-signal-backtest/scripts/regime_slice_analysis.py \
      --feature RB_PANIC_SCORE_PROXY --horizon 5
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

from soxl_quant.data import build_feature_frame, init_qlib
from pathlib import Path
from soxl_quant.policy.execution import execution_regime


def classify_regime(row: pd.Series) -> str:
    """Classify regime using the same logic as execution.py."""
    return execution_regime(
        regime_risk=row.get("regime_risk", 0.0),
        realized_vol=row.get("realized_vol_10", 0.0),
        trend_strength=abs(row.get("trend_gap_21", 0.0)),
        cum_drop_30=row.get("cum_drop_30", 0.0),
        relative_strength=row.get("relative_strength_21", 0.0),
        vix_zscore=row.get("vix_zscore", 0.0),
        downside_ratio_10=row.get("downside_ratio_10", 0.0),
    )


def analyze_regime_slice(factor: pd.Series, price: pd.Series, regime_labels: pd.Series, horizon: int) -> dict:
    forward_ret = (price.shift(-horizon) / price - 1.0).fillna(0.0)
    results = {}
    for regime in ["bull", "base", "stress"]:
        mask = regime_labels == regime
        if mask.sum() < 10:
            results[regime] = {"n": int(mask.sum()), "ic_mean": np.nan, "ic_std": np.nan, "win_rate": np.nan}
            continue
        f_reg = factor[mask]
        r_reg = forward_ret[mask]
        ic = f_reg.rolling(21).corr(r_reg).dropna()
        win_rate = (r_reg > 0).mean()
        results[regime] = {
            "n": int(mask.sum()),
            "ic_mean": float(ic.mean()) if len(ic) > 0 else np.nan,
            "ic_std": float(ic.std()) if len(ic) > 0 else np.nan,
            "win_rate": float(win_rate),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Regime slice analysis")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-04-01")
    args = parser.parse_args()

    init_qlib(Path("qlib_soxl_binary"))

    print(f"Loading data ({args.start} ~ {args.end})...")
    feature_df, price = build_feature_frame(args.start, args.end, feature_packs=("rulebase_signals",))
    feature_df = feature_df.droplevel("instrument")
    price = price.droplevel("instrument")

    if args.feature not in feature_df.columns:
        print(f"ERROR: {args.feature} not found")
        sys.exit(1)

    print("Classifying regimes...")
    regime_labels = feature_df.apply(classify_regime, axis=1)

    factor = feature_df[args.feature]
    results = analyze_regime_slice(factor, price, regime_labels, args.horizon)

    print(f"\n{'='*60}")
    print(f"Regime Slice Analysis: {args.feature} (horizon={args.horizon}d)")
    print(f"{'='*60}")
    print(f"{'Regime':<10} {'N':>6} {'IC_mean':>10} {'IC_std':>10} {'WinRate':>8}")
    print("-" * 50)
    for regime, stats in results.items():
        print(
            f"{regime:<10} {stats['n']:>6} {stats['ic_mean']:>+10.4f} "
            f"{stats['ic_std']:>10.4f} {stats['win_rate']:>8.2%}"
        )

    out_dir = ROOT / "docs" / "factor_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"regime_slice_{args.feature}.csv"
    pd.DataFrame([{"regime": k, **v} for k, v in results.items()]).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

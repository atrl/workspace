"""Compare a signal-based strategy to the rulebase baseline.

Usage:
    qlib_venv/bin/python .codex/skills/soxl-signal-backtest/scripts/compare_to_rulebase.py \
      --feature RB_PANIC_SCORE_PROXY --threshold-long 0.0 --threshold-short -0.5
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

sys.path.insert(0, str(ROOT / ".codex" / "skills" / "soxl-factor-mining" / "scripts"))
from backtest_signal import backtest_signal  # noqa: E402


def simulate_rulebase_baseline(price: pd.Series) -> pd.Series:
    """Simplified rulebase baseline: buy-and-hold with 50% min position."""
    returns = price.pct_change().fillna(0.0)
    # Approximate rulebase: always 50% long minimum
    return (0.5 * returns).cumsum()


def main():
    parser = argparse.ArgumentParser(description="Compare signal to rulebase")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--threshold-long", type=float, default=0.0)
    parser.add_argument("--threshold-short", type=float, default=-0.5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--hold-days", type=int, default=1)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-04-01")
    args = parser.parse_args()

    init_qlib(Path("qlib_soxl_binary"))

    feature_df, price = build_feature_frame(args.start, args.end, feature_packs=("rulebase_signals",))
    feature_df = feature_df.droplevel("instrument")
    price = price.droplevel("instrument")

    if args.feature not in feature_df.columns:
        print(f"ERROR: {args.feature} not found")
        sys.exit(1)

    factor = feature_df[args.feature]

    # Signal strategy
    signal_res = backtest_signal(
        factor, price, args.horizon,
        args.threshold_long, args.threshold_short, args.hold_days,
    )

    # Rulebase baseline (approximate)
    returns = price.pct_change().fillna(0.0)
    rulebase_cum = (1.0 + 0.5 * returns).cumprod()
    rulebase_ann = (rulebase_cum.iloc[-1] ** (252 / len(returns)) - 1.0)
    rulebase_dd = ((rulebase_cum - rulebase_cum.cummax()) / rulebase_cum.cummax()).min()

    print(f"\n{'='*70}")
    print(f"Comparison: {args.feature} vs Rulebase Baseline")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Signal':>12} {'Rulebase':>12}")
    print("-" * 50)
    print(f"{'Annualized Return':<25} {signal_res.get('annualized_return', 0):>12.2%} {rulebase_ann:>12.2%}")
    print(f"{'Max Drawdown':<25} {signal_res.get('max_drawdown', 0):>12.2%} {rulebase_dd:>12.2%}")
    print(f"{'Sharpe':<25} {signal_res.get('sharpe', 0):>12.2f} {'N/A':>12}")
    print(f"{'Signal Accuracy':<25} {signal_res.get('signal_accuracy', 0):>12.2%} {'N/A':>12}")
    print(f"{'Profit Factor':<25} {signal_res.get('profit_factor', 0):>12.2f} {'N/A':>12}")
    print(f"{'Win Rate':<25} {signal_res.get('win_rate', 0):>12.2%} {'N/A':>12}")

    # Save comparison
    out_dir = ROOT / "docs" / "factor_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "feature": args.feature,
        "signal": signal_res,
        "rulebase_approx": {
            "annualized_return": float(rulebase_ann),
            "max_drawdown": float(rulebase_dd),
        },
    }
    out_csv = out_dir / f"compare_{args.feature}.csv"
    pd.DataFrame([comparison]).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

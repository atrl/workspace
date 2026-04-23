"""Signal backtest main entry — wraps backtest_signal.py with report generation.

Usage:
    qlib_venv/bin/python .codex/skills/soxl-signal-backtest/scripts/signal_backtest.py \
      --feature RB_PANIC_SCORE_PROXY --report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the core backtest engine from soxl-factor-mining
sys.path.insert(0, str(ROOT / ".codex" / "skills" / "soxl-factor-mining" / "scripts"))
from backtest_signal import backtest_signal, sweep_thresholds  # noqa: E402
from soxl_quant.data import build_feature_frame, init_qlib
from pathlib import Path  # noqa: E402


def generate_report(feature_name: str, results: list[dict], output_path: Path) -> None:
    lines = [
        f"# Signal Backtest Report: {feature_name}",
        "",
        "## Parameter Sweep Results",
        "",
        "| thr_long | thr_short | accuracy | PF | avg_ret | sharpe | win_rate | MDD | n_signals |",
        "|----------|-----------|----------|----|---------|--------|----------|-----|-----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['threshold_long']:.3f} | {r['threshold_short']:.3f} | "
            f"{r['signal_accuracy']:.2%} | {r['profit_factor']:.2f} | "
            f"{r['avg_return']:.4f} | {r['sharpe']:.2f} | "
            f"{r['win_rate']:.1%} | {r['max_drawdown']:.2%} | {r['num_signals']} |"
        )
    lines.append("")
    # Best by sharpe
    valid = [r for r in results if r.get("sharpe", -999) > -999]
    if valid:
        best = max(valid, key=lambda x: x["sharpe"])
        lines.append("## Best Configuration (by Sharpe)")
        lines.append("")
        lines.append(f"- threshold_long: {best['threshold_long']:.3f}")
        lines.append(f"- threshold_short: {best['threshold_short']:.3f}")
        lines.append(f"- sharpe: {best['sharpe']:.2f}")
        lines.append(f"- signal_accuracy: {best['signal_accuracy']:.2%}")
        lines.append(f"- profit_factor: {best['profit_factor']:.2f}")
        lines.append(f"- max_drawdown: {best['max_drawdown']:.2%}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Signal backtest with report generation")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--hold-days", type=int, default=1)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    args = parser.parse_args()

    init_qlib(Path("qlib_soxl_binary"))

    print(f"Loading data ({args.start} ~ {args.end})...")
    feature_df, price = build_feature_frame(args.start, args.end, feature_packs=("rulebase_signals",))
    feature_df = feature_df.droplevel("instrument")
    price = price.droplevel("instrument")

    if args.feature not in feature_df.columns:
        print(f"ERROR: {args.feature} not found")
        sys.exit(1)

    factor = feature_df[args.feature]
    results = sweep_thresholds(factor, price, args.horizon, args.hold_days)

    print(f"\n{'='*100}")
    print(f"  SIGNAL BACKTEST: {args.feature}  horizon={args.horizon}d")
    print(f"{'='*100}")
    print(
        f"{'thr_long':>8} {'thr_short':>9} {'acc':>6} {'PF':>5} {'avg_ret':>8} "
        f"{'sharpe':>6} {'win%':>5} {'MDD':>7} {'n_sig':>6}"
    )
    for r in results:
        print(
            f"{r['threshold_long']:>8.3f} {r['threshold_short']:>9.3f} "
            f"{r['signal_accuracy']:>6.2%} {r['profit_factor']:>5.2f} "
            f"{r['avg_return']:>8.4f} {r['sharpe']:>6.2f} {r['win_rate']:>5.1%} "
            f"{r['max_drawdown']:>7.2%} {r['num_signals']:>6d}"
        )

    if args.report:
        out_dir = ROOT / "docs" / "factor_research"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"report_backtest_{args.feature}.md"
        generate_report(args.feature, results, out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compose and score a budget-usage counterfactual from two daily policy paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.budget_counterfactual import (  # noqa: E402
    BullParticipationGuardConfig,
    EquityDrawdownControlConfig,
    RecoveryReentryGuardConfig,
    build_budget_usage_counterfactual,
    write_counterfactual_artifacts,
)
from src.soxl_quant.data import build_dataset, init_qlib  # noqa: E402
from src.soxl_quant.settings import QLIB_DATA_PATH  # noqa: E402


def _default_label(path: Path) -> str:
    return path.parent.name or path.stem.replace("_soxl_daily", "")


def relative_output_label(path: Path, root: Path = ROOT) -> str:
    base = path if path.is_absolute() else root / path
    try:
        return base.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a budget-usage counterfactual from slow/fast daily CSVs")
    parser.add_argument("--slow", required=True, type=Path, help="Slow risk-mode daily CSV")
    parser.add_argument("--fast", required=True, type=Path, help="Fast usage daily CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for JSON/CSV/Markdown artifacts")
    parser.add_argument("--profile", default="", help="Output profile/card key")
    parser.add_argument("--slow-label", default="", help="Label for the slow path")
    parser.add_argument("--fast-label", default="", help="Label for the fast path")
    parser.add_argument("--composed-label", default="", help="Label for the composed path")
    parser.add_argument("--description", default="", help="Review description")
    parser.add_argument(
        "--bull-participation-guard",
        action="store_true",
        help="Floor raw target to the fast path during strong, shallow-drawdown trend windows",
    )
    parser.add_argument("--bull-ma-window", type=int, default=50, help="Moving-average window for bull guard")
    parser.add_argument("--bull-momentum-window", type=int, default=20, help="Momentum window for bull guard")
    parser.add_argument("--bull-high-window", type=int, default=50, help="Rolling-high window for bull guard drawdown")
    parser.add_argument("--bull-min-periods", type=int, default=20, help="Minimum periods for bull guard rolling stats")
    parser.add_argument("--bull-min-momentum", type=float, default=0.03, help="Minimum return for bull guard")
    parser.add_argument("--bull-max-drawdown", type=float, default=0.08, help="Maximum rolling drawdown for bull guard")
    parser.add_argument(
        "--recovery-reentry-guard",
        action="store_true",
        help="Floor low composed targets in high-volatility positive-momentum rebound states",
    )
    parser.add_argument("--reentry-min-target", type=float, default=0.35, help="Minimum target when reentry guard fires")
    parser.add_argument("--reentry-max-target", type=float, default=0.55, help="Maximum target applied by reentry guard")
    parser.add_argument(
        "--reentry-max-base-target",
        type=float,
        default=0.25,
        help="Only apply reentry guard when the raw composed target is at or below this value",
    )
    parser.add_argument("--reentry-min-mom-10", type=float, default=0.08)
    parser.add_argument("--reentry-min-mom-21", type=float, default=0.12)
    parser.add_argument("--reentry-min-relative-strength-21", type=float, default=1.0)
    parser.add_argument("--reentry-min-vol-of-vol-21", type=float, default=0.10)
    parser.add_argument("--reentry-max-vix-percentile-252", type=float, default=0.85)
    parser.add_argument("--reentry-max-below-anchor-ratio-63", type=float, default=1.0)
    parser.add_argument("--reentry-min-anchor-gap-180", type=float, default=-10.0)
    parser.add_argument(
        "--equity-drawdown-control",
        action="store_true",
        help="Cap composed targets after realized strategy equity drawdown",
    )
    parser.add_argument("--drawdown-start", type=float, default=0.10)
    parser.add_argument("--drawdown-full", type=float, default=0.18)
    parser.add_argument("--drawdown-min-target", type=float, default=0.25)
    parser.add_argument("--drawdown-sell-trade-threshold", type=float, default=0.0)
    parser.add_argument("--drawdown-sell-trade-lock-days", type=int, default=0)
    parser.add_argument("--drawdown-max-anchor-gap-180", type=float, default=None)
    parser.add_argument("--drawdown-min-below-anchor-ratio-63", type=float, default=None)
    parser.add_argument("--drawdown-min-vix-percentile-252", type=float, default=None)
    parser.add_argument("--drawdown-min-vol-of-vol-21", type=float, default=None)
    parser.add_argument("--drawdown-max-mom-21", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = args.profile or args.output_dir.name
    slow_label = args.slow_label or _default_label(args.slow)
    fast_label = args.fast_label or _default_label(args.fast)
    composed_label = args.composed_label or profile
    description = args.description or (
        f"Fixed-path budget-usage counterfactual: `{slow_label}` sets the slow risk mode "
        f"and `{fast_label}` is interpreted as usage inside that mode band."
    )

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    price = dataset.handler.price_frame.droplevel("instrument")["price"]
    feature_frame = dataset.handler.feature_frame.droplevel("instrument").sort_index()

    result = build_budget_usage_counterfactual(
        pd.read_csv(args.slow),
        pd.read_csv(args.fast),
        price,
        slow_label=slow_label,
        fast_label=fast_label,
        composed_label=composed_label,
        profile=profile,
        description=description,
        bull_guard=BullParticipationGuardConfig(
            enabled=bool(args.bull_participation_guard),
            ma_window=args.bull_ma_window,
            momentum_window=args.bull_momentum_window,
            high_window=args.bull_high_window,
            min_periods=args.bull_min_periods,
            min_momentum=args.bull_min_momentum,
            max_drawdown=args.bull_max_drawdown,
        ),
        feature_frame=feature_frame,
        recovery_guard=RecoveryReentryGuardConfig(
            enabled=bool(args.recovery_reentry_guard),
            min_target=args.reentry_min_target,
            max_target=args.reentry_max_target,
            max_base_target=args.reentry_max_base_target,
            min_mom_10=args.reentry_min_mom_10,
            min_mom_21=args.reentry_min_mom_21,
            min_relative_strength_21=args.reentry_min_relative_strength_21,
            min_vol_of_vol_21=args.reentry_min_vol_of_vol_21,
            max_vix_percentile_252=args.reentry_max_vix_percentile_252,
            max_below_anchor_ratio_63=args.reentry_max_below_anchor_ratio_63,
            min_anchor_gap_180=args.reentry_min_anchor_gap_180,
        ),
        drawdown_control=EquityDrawdownControlConfig(
            enabled=bool(args.equity_drawdown_control),
            start_drawdown=args.drawdown_start,
            full_drawdown=args.drawdown_full,
            min_target=args.drawdown_min_target,
            sell_trade_threshold=args.drawdown_sell_trade_threshold,
            sell_trade_lock_days=args.drawdown_sell_trade_lock_days,
            max_anchor_gap_180=args.drawdown_max_anchor_gap_180,
            min_below_anchor_ratio_63=args.drawdown_min_below_anchor_ratio_63,
            min_vix_percentile_252=args.drawdown_min_vix_percentile_252,
            min_vol_of_vol_21=args.drawdown_min_vol_of_vol_21,
            max_mom_21=args.drawdown_max_mom_21,
        ),
    )
    artifacts = write_counterfactual_artifacts(result, args.output_dir, price)
    composed = result.payload["variants"][composed_label]["test_proxy"]
    print(
        f"[budget_counterfactual] wrote {relative_output_label(args.output_dir)} "
        f"return={composed['return']:.1f}% sharpe={composed['sharpe']:.3f} "
        f"max_dd={composed['max_dd']:.1f}% avg_pos={composed['avg_pos']:.3f}"
    )
    for kind, path in artifacts.items():
        print(f"  {kind}: {relative_output_label(path)}")


if __name__ == "__main__":
    main()

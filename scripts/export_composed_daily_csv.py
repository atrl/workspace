#!/usr/bin/env python3
"""Export composed budget-usage daily CSV for dashboard consumption.

Usage:
    PYTHONPATH=. qlib_venv/bin/python scripts/export_composed_daily_csv.py \
        --slow runtime/model/card/balanced/balanced_soxl_daily.csv \
        --fast runtime/model/card/balanced_rule_residual_imitation_v4/balanced_rule_residual_imitation_v4_soxl_daily.csv \
        --output runtime/model/card/risk_budget_usage_v1/risk_budget_usage_v1_composed_balanced_imitation_v4_soxl_daily.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.policy.two_speed_budget import (
    TwoSpeedBudgetConfig,
    compose_budget_usage_policy_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export composed daily CSV")
    parser.add_argument("--slow", required=True, help="Slow actor daily CSV")
    parser.add_argument("--fast", required=True, help="Fast actor daily CSV")
    parser.add_argument("--output", required=True, help="Output daily CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    slow = pd.read_csv(args.slow)
    fast = pd.read_csv(args.fast)

    composed = compose_budget_usage_policy_frame(slow, fast, TwoSpeedBudgetConfig())

    # Compute NAV from composed positions and price
    prices = composed["price"].values.astype(float) if "price" in composed.columns else None
    if prices is None:
        # Try merge price from slow
        composed = composed.merge(slow[["datetime", "price"]], on="datetime", how="left")
        prices = composed["price"].values.astype(float)

    positions = composed["position"].values.astype(float)
    returns = np.diff(prices) / prices[:-1]
    strat_returns = positions[:-1] * returns

    nav = np.cumprod(1 + strat_returns)
    nav = np.concatenate([[1.0], nav])  # start at 1.0

    bnh_returns = np.diff(prices) / prices[:-1]
    bnh_nav = np.cumprod(1 + bnh_returns)
    bnh_nav = np.concatenate([[1.0], bnh_nav])

    # Scale to 1M initial NAV
    initial_nav = 1_000_000.0
    composed["strategy_nav"] = nav * initial_nav
    composed["buyhold_nav"] = bnh_nav * initial_nav
    composed["excess"] = composed["strategy_nav"] - composed["buyhold_nav"]
    composed["excess_pct"] = composed["excess"] / composed["buyhold_nav"] * 100

    # Determine action from position changes
    actions = ["HOLD"]
    for i in range(1, len(positions)):
        delta = positions[i] - positions[i - 1]
        if delta > 0.001:
            actions.append("BUY")
        elif delta < -0.001:
            actions.append("SELL")
        else:
            actions.append("HOLD")
    composed["action"] = actions

    # Rename columns to match dashboard expected format
    out = composed[["datetime", "strategy_nav", "price", "buyhold_nav", "position", "excess", "excess_pct", "action"]].copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[done] wrote {output_path} ({len(out)} rows)")
    print(f"  return={nav[-1]*100-100:.1f}%  avg_pos={np.mean(positions):.3f}  trades={sum(1 for a in actions if a in ('BUY','SELL'))}")


if __name__ == "__main__":
    main()

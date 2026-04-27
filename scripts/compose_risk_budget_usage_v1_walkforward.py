#!/usr/bin/env python3
"""Compose budget-usage counterfactual from walk-forward slow and fast daily paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.policy.two_speed_budget import (
    TwoSpeedBudgetConfig,
    compose_budget_usage_policy_frame,
)
from src.soxl_quant.data import build_dataset, init_qlib
from src.soxl_quant.settings import QLIB_DATA_PATH


WINDOWS = ["wf_2020", "wf_2022", "wf_2023_2024"]


def daily_csv_path(profile_name: str, window: str) -> Path:
    return (
        ROOT
        / "runtime"
        / "model"
        / "card"
        / profile_name
        / f"{profile_name}_{window}_soxl_daily.csv"
    )


def _align_price(frame: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime").reset_index(drop=True)
    if "price" in frame.columns:
        frame = frame.drop(columns=["price"])

    price_series = price_series.copy()
    if not isinstance(price_series.index, pd.DatetimeIndex):
        price_series.index = pd.to_datetime(price_series.index)
    price_df = price_series.rename("price").reset_index()
    price_df.columns = ["datetime", "price"]
    price_df["datetime"] = pd.to_datetime(price_df["datetime"])

    frame = frame.merge(price_df, on="datetime", how="inner")
    if frame.empty:
        raise ValueError("No overlapping dates between positions and prices")
    return frame


def compute_metrics(frame: pd.DataFrame, price_series: pd.Series) -> dict:
    frame = _align_price(frame, price_series)

    positions = frame["position"].values.astype(float)
    prices = frame["price"].values.astype(float)
    returns = np.diff(prices) / prices[:-1]
    strat_returns = positions[:-1] * returns

    nav = np.cumprod(1 + strat_returns)
    total_return = float(nav[-1] - 1) * 100

    if len(strat_returns) > 1 and np.std(strat_returns) > 0:
        sharpe = float(np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(nav)
    drawdowns = (nav - peak) / peak
    max_dd = float(np.min(drawdowns)) * 100

    avg_pos = float(np.mean(positions))
    stress_mask = drawdowns < -0.10
    stress_avg_pos = float(np.mean(positions[1:][stress_mask])) if np.any(stress_mask) else 0.0
    stress_days = int(np.sum(stress_mask))

    bnh_nav = np.cumprod(1 + returns)
    bnh_return = float(bnh_nav[-1] - 1) * 100

    changes = np.abs(np.diff(positions))
    trades = int(np.sum(changes > 0.001))

    return {
        "return": total_return,
        "sharpe": sharpe,
        "max_dd": abs(max_dd),
        "avg_pos": avg_pos,
        "stress_avg_pos": stress_avg_pos,
        "stress_days": stress_days,
        "buy_and_hold_return": bnh_return,
        "relative_return": total_return - bnh_return,
        "trades": trades,
    }


def compose_window(slow_csv: Path, fast_csv: Path, price_series: pd.Series) -> dict:
    slow = pd.read_csv(slow_csv)
    fast = pd.read_csv(fast_csv)

    slow = slow.rename(columns={"position": "slow_position"})
    fast = fast.rename(columns={"position": "fast_position"})

    composed = compose_budget_usage_policy_frame(slow, fast, TwoSpeedBudgetConfig())
    return compute_metrics(composed, price_series)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose risk_budget_usage_v1 walk-forward counterfactual")
    parser.add_argument("--slow-profile", default="balanced_rule_residual_mode_v1")
    parser.add_argument("--fast-profile", default="balanced_rule_residual_imitation_v4")
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    price = dataset.handler.price_frame.droplevel("instrument")["price"]

    window_results: dict[str, dict] = {}
    missing: list[str] = []

    for window in WINDOWS:
        slow_path = daily_csv_path(args.slow_profile, window)
        fast_path = daily_csv_path(args.fast_profile, window)
        if not slow_path.exists():
            missing.append(str(slow_path))
        if not fast_path.exists():
            missing.append(str(fast_path))
        if slow_path.exists() and fast_path.exists():
            # Slice price series to window test dates
            if window == "wf_2020":
                test_start, test_end = "2020-01-01", "2020-12-31"
            elif window == "wf_2022":
                test_start, test_end = "2022-01-01", "2022-12-31"
            else:
                test_start, test_end = "2023-01-01", "2024-12-31"
            window_price = price.loc[test_start:test_end]
            window_results[window] = compose_window(slow_path, fast_path, window_price)
            print(f"[{window}] return={window_results[window]['return']:.1f}% sharpe={window_results[window]['sharpe']:.2f} max_dd={window_results[window]['max_dd']:.1f}%")

    if missing:
        print(f"[warn] missing files: {missing}")

    payload = {
        "profile": "risk_budget_usage_v1",
        "composition": "budget_usage",
        "description": "Walk-forward counterfactual combining slow and fast actors via budget_usage.",
        "slow_profile": args.slow_profile,
        "fast_profile": args.fast_profile,
        "seed": args.seed,
        "windows": window_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()

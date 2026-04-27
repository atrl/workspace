#!/usr/bin/env python3
"""Run backtest on an arbitrary date segment and output JSON for dashboard consumption.

Usage:
    PYTHONPATH=. qlib_venv/bin/python scripts/run_segment_backtest.py \\
        --profile balanced --start-date 2024-01-01 --end-date 2024-06-01

Output: JSON to stdout with keys: summary, rows, trades, windows
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, init_qlib, segment_frame
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import (
    ACCOUNT,
    MODEL_DIR,
    QLIB_DATA_PATH,
    TEST_SEG,
    load_profile,
    model_artifact_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segment backtest and emit JSON")
    parser.add_argument("--profile", required=True, help="Profile name, e.g. balanced")
    parser.add_argument("--tag", default="", help="Artifact tag")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    return parser.parse_args()


def _compute_metrics_from_daily(rows: list[dict]) -> dict:
    """Compute zero-cost NAV-based metrics from daily backtest rows."""
    if not rows:
        return {"return": 0.0, "sharpe": 0.0, "max_dd": 0.0, "avg_pos": 0.0, "trades": 0}

    prices = np.array([r["price"] for r in rows], dtype=float)
    positions = np.array([r["position"] for r in rows], dtype=float)
    returns = np.diff(prices) / prices[:-1]
    strat_returns = positions[:-1] * returns
    nav = np.cumprod(1 + strat_returns)

    total_return = float(nav[-1] - 1) * 100
    sharpe = float(np.mean(strat_returns) / (np.std(strat_returns) + 1e-8) * np.sqrt(252))
    peak = np.maximum.accumulate(nav)
    max_dd = float(np.min((nav - peak) / peak)) * 100
    avg_pos = float(np.mean(positions))

    # Count discrete trades (position changes)
    trades = sum(1 for i in range(1, len(positions)) if abs(positions[i] - positions[i - 1]) > 0.01)

    return {
        "return": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "avg_pos": round(avg_pos, 4),
        "trades": trades,
    }


def _pair_trades(rows: list[dict]) -> list[dict]:
    """Pair BUY/SELL actions into trades."""
    trades = []
    entry = None
    for r in rows:
        action = r.get("action", "HOLD")
        if action == "BUY" and entry is None:
            entry = r
        elif action == "SELL" and entry is not None:
            entry_price = entry.get("price", entry.get("close", 0))
            exit_price = r.get("price", r.get("close", 0))
            ret_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0
            trades.append({
                "entry_date": entry["datetime"],
                "exit_date": r["datetime"],
                "days": (pd.Timestamp(r["datetime"]) - pd.Timestamp(entry["datetime"])).days,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return_pct": round(ret_pct, 2),
                "pnl": round(ret_pct / 100 * ACCOUNT, 2),
            })
            entry = None
    return trades


def main() -> None:
    args = parse_args()

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()

    # Use the full dataset but slice to requested segment
    full_df = segment_frame(dataset, "test")
    seg_df = full_df.loc[args.start_date : args.end_date]

    if seg_df.empty:
        print(json.dumps({"error": f"No data for {args.start_date} to {args.end_date}"}))
        sys.exit(1)

    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[seg_df.index]

    profile = load_profile(args.profile)
    model_path = model_artifact_path(args.profile, args.tag)
    if not model_path.exists():
        print(json.dumps({"error": f"Model not found: {model_path}"}))
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    _, actor = build_policy(
        int(ckpt["obs_dim"]),
        profile,
        use_moe=ckpt.get("use_moe", False),
        hard_regime_routing=ckpt.get("use_moe", False),
    )
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()

    result = backtest_profile(
        actor=actor,
        feature_df=seg_df,
        price_series=price,
        position_cap_config=profile.position_cap,
        position_budget_config=profile.position_budget,
        execution_config=profile.execution,
        symbol="SOXL",
        test_seg=(args.start_date, args.end_date),
        model_config=profile.model,
    )

    # Build daily rows
    pdf = result["portfolio"]
    account = pdf["account"].astype(float)
    bh_nav = [ACCOUNT]
    for i in range(1, len(account)):
        prev_price = price.iloc[i - 1]
        curr_price = price.iloc[i]
        bh_nav.append(bh_nav[-1] * (curr_price / prev_price) if prev_price > 0 else bh_nav[-1])

    rows = []
    for i, (ts, acc) in enumerate(account.items()):
        dt_str = str(ts)[:10]
        pos = result["position_history"][i] if i < len(result["position_history"]) else 0.0
        action = "HOLD"
        for sig_date, sig_price, sig_pos, sig_type in result.get("trade_signals", []):
            if str(sig_date)[:10] == dt_str:
                action = sig_type.upper()
                break

        p = price.iloc[i] if i < len(price) else 0
        excess = acc - bh_nav[i]
        excess_pct = (excess / bh_nav[i] * 100) if bh_nav[i] != 0 else 0.0

        rows.append({
            "datetime": dt_str,
            "strategy_nav": round(acc, 6),
            "price": round(p, 6),
            "buyhold_nav": round(bh_nav[i], 6),
            "position": round(pos, 6),
            "excess": round(excess, 6),
            "excess_pct": round(excess_pct, 6),
            "action": action,
        })

    # Compute metrics
    metrics = _compute_metrics_from_daily(rows)
    metrics["trades"] = result["trades"]

    # Pair trades
    paired = _pair_trades(rows)

    output = {
        "profile": args.profile,
        "tag": args.tag,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "summary": metrics,
        "rows": rows,
        "trades": paired,
        "windows": {
            "SOXL": {
                "start": args.start_date,
                "end": args.end_date,
                "rows": len(rows),
            }
        },
    }

    print(json.dumps(output, default=str))


if __name__ == "__main__":
    main()

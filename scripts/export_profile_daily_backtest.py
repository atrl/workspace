#!/usr/bin/env python3
"""Export daily SOXL backtest rows for a saved actor/profile pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
import csv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, init_qlib, segment_frame
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import ACCOUNT, MODEL_DIR, QLIB_DATA_PATH, TEST_SEG, load_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export daily SOXL backtest rows for a saved profile actor.")
    parser.add_argument("--profile", required=True, help="Profile name, e.g. balanced or balanced_transition_state")
    parser.add_argument("--output", required=True, help="CSV path to write")
    return parser.parse_args()


def export_detailed_csv(
    out_path: Path,
    portfolio_df: pd.DataFrame,
    price_series: pd.Series,
    position_history: list[float],
    trade_signals: list[tuple],
) -> None:
    account = portfolio_df["account"].astype(float)
    dates = account.index.strftime("%Y-%m-%d").tolist()
    prices = price_series.reindex(account.index).ffill().values
    initial_price = float(prices[0]) if len(prices) > 0 else 1.0
    buyhold_nav = ACCOUNT * (prices / initial_price)

    action_map: dict[str, str] = {}
    for dt, _price, _pos, stype in trade_signals:
        action_map[pd.Timestamp(dt).strftime("%Y-%m-%d")] = stype.upper()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datetime", "strategy_nav", "price", "buyhold_nav", "position", "excess", "excess_pct", "action"])
        for idx, dt in enumerate(dates):
            strat_nav = float(account.iloc[idx])
            price = float(prices[idx])
            bh_nav = float(buyhold_nav[idx])
            pos = float(position_history[idx]) if idx < len(position_history) else 0.0
            excess = strat_nav - bh_nav
            excess_pct = (excess / bh_nav * 100.0) if bh_nav != 0 else 0.0
            action = action_map.get(dt, "HOLD")
            writer.writerow([dt, strat_nav, price, bh_nav, pos, excess, excess_pct, action])


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]

    profile = load_profile(args.profile)
    ckpt = torch.load(MODEL_DIR / f"soxl_{args.profile}_actor.pt", map_location="cpu", weights_only=False)
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
        feature_df=test_df,
        price_series=price,
        position_cap_config=profile.position_cap,
        execution_config=profile.execution,
        symbol="SOXL",
        test_seg=TEST_SEG,
        model_config=profile.model,
    )
    export_detailed_csv(
        output,
        result["portfolio"],
        price,
        result["position_history"],
        result.get("trade_signals", []),
    )
    print(output)


if __name__ == "__main__":
    main()

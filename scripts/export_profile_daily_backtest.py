#!/usr/bin/env python3
"""Export daily SOXL backtest rows for a saved actor/profile pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

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
        symbol="SOXL",
        test_seg=TEST_SEG,
    )

    portfolio = result["portfolio"]
    pos_hist = result["position_history"]
    aligned_pos = pd.Series([0.0] * (len(portfolio) - len(pos_hist)) + pos_hist, index=portfolio.index)

    df = pd.DataFrame(index=portfolio.index)
    df["strategy_nav"] = portfolio["account"].astype(float)
    df["price"] = price.reindex(df.index).ffill().bfill().astype(float)
    df["buyhold_nav"] = ACCOUNT * (df["price"] / df["price"].iloc[0])
    df["position"] = aligned_pos.reindex(df.index).fillna(0.0)
    df["strategy_ret"] = df["strategy_nav"].pct_change().fillna(0.0)
    df["buyhold_ret"] = df["buyhold_nav"].pct_change().fillna(0.0)
    df["excess_ret"] = df["strategy_ret"] - df["buyhold_ret"]
    df.to_csv(output)
    print(output)


if __name__ == "__main__":
    main()

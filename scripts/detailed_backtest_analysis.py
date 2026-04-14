#!/usr/bin/env python3
"""Run a detailed qlib backtest for SOXL and output daily NAV, position, trades vs Buy&Hold."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, init_qlib, refresh_qlib_binary, segment_frame
from src.soxl_quant.rl import build_policy
from src.soxl_quant.settings import ACCOUNT, MODEL_DIR, QLIB_DATA_PATH, TEST_SEG, load_profile

RUNTIME_DIR = ROOT / "runtime" / "generalization"
CSV_PATH = RUNTIME_DIR / "soxl_detailed_trades.csv"
CHART_PATH = RUNTIME_DIR / "soxl_strategy_vs_buyhold.png"


def main():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Init data & actor
    profile = load_profile("balanced")
    ckpt = torch.load(MODEL_DIR / "soxl_balanced_actor.pt", map_location="cpu", weights_only=False)
    policy, actor = build_policy(ckpt["obs_dim"], profile)
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset(leveraged_etf="SOXL", underlying_etf="SOXX", leverage_ratio=3.0)
    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]

    # 2. Run official qlib backtest
    result = backtest_profile(
        actor=actor,
        feature_df=test_df,
        price_series=price,
        position_cap_config=profile.position_cap,
        symbol="SOXL",
        test_seg=TEST_SEG,
    )

    pdf = result["portfolio"]
    pos_hist = result["position_history"]

    # Align position history to portfolio index
    # position_history starts after LOOKBACK days; pad early days with 0.0
    aligned_pos = pd.Series([0.0] * (len(pdf) - len(pos_hist)) + pos_hist, index=pdf.index)

    # 3. Build daily detail dataframe
    df = pd.DataFrame(index=pdf.index)
    df["strategy_nav"] = pdf["account"].astype(float)
    df["price"] = price.reindex(df.index).ffill().bfill().astype(float)
    df["buyhold_nav"] = ACCOUNT * (df["price"] / df["price"].iloc[0])
    df["position"] = aligned_pos.reindex(df.index).fillna(0.0)
    df["excess"] = df["strategy_nav"] - df["buyhold_nav"]
    df["excess_pct"] = (df["strategy_nav"] / df["buyhold_nav"] - 1.0) * 100

    # 4. Mark trades
    df["action"] = "HOLD"
    pos_changes = df["position"].diff().abs()
    df.loc[pos_changes > 0.01, "action"] = df.loc[pos_changes > 0.01].apply(
        lambda row: "BUY" if row["position"] > aligned_pos.shift(1).loc[row.name] else "SELL", axis=1
    )

    # Save CSV
    df.to_csv(CSV_PATH)
    print(f"Saved detailed trades to {CSV_PATH}")

    # 5. Summary stats
    final_strategy = df["strategy_nav"].iloc[-1]
    final_buyhold = df["buyhold_nav"].iloc[-1]
    print(f"\nFinal Strategy NAV: ${final_strategy:,.2f} ({final_strategy/ACCOUNT-1:.2%})")
    print(f"Final Buy&Hold NAV: ${final_buyhold:,.2f} ({final_buyhold/ACCOUNT-1:.2%})")
    print(f"Excess return:      {final_strategy/final_buyhold-1:.2%}")
    print(f"Max outperformance: {df['excess_pct'].max():.2f}%")
    print(f"Max underperformance: {df['excess_pct'].min():.2f}%")

    trades = df[df["action"] != "HOLD"]
    print(f"\nTotal trades: {len(trades)} (BUY: {len(trades[trades['action']=='BUY'])}, SELL: {len(trades[trades['action']=='SELL'])})")

    # 6. Plot
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # NAV comparison
    axes[0].plot(df.index, df["strategy_nav"] / 1000, lw=2.2, label="Strategy", color="#1f77b4")
    axes[0].plot(df.index, df["buyhold_nav"] / 1000, lw=2.2, label="Buy & Hold", color="#ff7f0e", alpha=0.8)
    axes[0].set_ylabel("Account ($K)")
    axes[0].set_title("SOXL qlib-native PPO Strategy vs Buy & Hold", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Position + trade markers
    axes[1].plot(df.index, df["position"], lw=1.5, color="black", label="Position")
    buy_dates = df[df["action"] == "BUY"].index
    sell_dates = df[df["action"] == "SELL"].index
    axes[1].scatter(buy_dates, df.loc[buy_dates, "position"], marker="^", color="green", s=30, label="BUY", zorder=5)
    axes[1].scatter(sell_dates, df.loc[sell_dates, "position"], marker="v", color="red", s=30, label="SELL", zorder=5)
    axes[1].set_ylabel("Position")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Excess return over Buy&Hold
    axes[2].fill_between(df.index, df["excess_pct"], 0, where=(df["excess_pct"] >= 0), color="green", alpha=0.3, label="Outperform")
    axes[2].fill_between(df.index, df["excess_pct"], 0, where=(df["excess_pct"] < 0), color="red", alpha=0.3, label="Underperform")
    axes[2].plot(df.index, df["excess_pct"], lw=1.2, color="black")
    axes[2].axhline(0, color="gray", ls="--", lw=0.8)
    axes[2].set_ylabel("Excess vs B&H (%)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # Underwater (Strategy)
    dd = df["strategy_nav"] / df["strategy_nav"].cummax() - 1.0
    axes[3].plot(df.index, dd.values, lw=1.6, color="#1f77b4", label="Strategy DD")
    bh_dd = df["buyhold_nav"] / df["buyhold_nav"].cummax() - 1.0
    axes[3].plot(df.index, bh_dd.values, lw=1.6, color="#ff7f0e", alpha=0.7, label="B&H DD")
    axes[3].set_ylabel("Drawdown")
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=250, bbox_inches="tight")
    print(f"Saved chart to {CHART_PATH}")


if __name__ == "__main__":
    main()

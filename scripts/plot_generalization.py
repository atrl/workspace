#!/usr/bin/env python3
"""Plot generalization backtest comparison across 3x ETFs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "runtime" / "generalization" / "results.json"
CHART_PATH = ROOT / "runtime" / "generalization" / "comparison.png"


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    etfs = list(results.keys())
    returns = [results[e]["return"] for e in etfs]
    sharpes = [results[e]["sharpe"] for e in etfs]
    max_dds = [results[e]["max_dd"] for e in etfs]

    x = np.arange(len(etfs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, returns, width, label="Return (%)", color="#2ca02c")
    bars2 = ax.bar(x, [s * 100 for s in sharpes], width, label="Sharpe x100", color="#1f77b4")
    bars3 = ax.bar(x + width, [-dd for dd in max_dds], width, label="-MaxDD (%)", color="#d62728")

    ax.set_ylabel("Value")
    ax.set_title("Zero-Shot Generalization: SOXL-Trained Actor on 3x ETFs", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(etfs)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=250, bbox_inches="tight")
    print(f"Chart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()

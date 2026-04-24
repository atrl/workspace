#!/usr/bin/env python3
"""Test transition signal overlay on cross-ETF actors."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, init_qlib, segment_frame
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import MODEL_DIR, TEST_SEG, load_profile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True, choices=["SOXL", "TQQQ", "UPRO", "SPXL", "TECL"])
    parser.add_argument("--overlay", choices=["none", "vix_divergence"], default="none")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--boost", type=float, default=0.1)
    parser.add_argument("--regime-threshold", type=float, default=0.6)
    return parser.parse_args()


ETF_CONFIG = {
    "SOXL": (str(ROOT / "data" / "qlib" / "soxl"), "SOXX"),
    "TQQQ": (str(ROOT / "data" / "qlib" / "tqqq"), "QQQ"),
    "UPRO": (str(ROOT / "data" / "qlib" / "upro"), "SPY"),
    "SPXL": (str(ROOT / "data" / "qlib" / "spxl"), "SPY"),
    "TECL": (str(ROOT / "data" / "qlib" / "tecl"), "XLK"),
}


def main():
    args = parse_args()
    provider_uri, underlying = ETF_CONFIG[args.etf]
    init_qlib(provider_uri)

    # Use SOXL actor (zero-shot transfer test)
    profile = load_profile("balanced_rule_residual_v19")
    dataset = build_dataset(leveraged_etf=args.etf, underlying_etf=underlying, feature_packs=("transition_signals",))
    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]

    ckpt = torch.load(MODEL_DIR / "soxl_balanced_rule_residual_v19_actor.pt", map_location="cpu", weights_only=False)
    _, actor = build_policy(
        int(ckpt["obs_dim"]),
        profile,
        use_moe=ckpt.get("use_moe", False),
        hard_regime_routing=ckpt.get("use_moe", False),
    )
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()

    overlay_config = None
    if args.overlay != "none":
        overlay_config = {
            "type": args.overlay,
            "threshold": args.threshold,
            "boost": args.boost,
            "regime_threshold": args.regime_threshold,
        }

    result = backtest_profile(
        actor=actor,
        feature_df=test_df,
        price_series=price,
        position_cap_config=profile.position_cap,
        position_budget_config=profile.position_budget,
        execution_config=profile.execution,
        symbol=args.etf,
        test_seg=TEST_SEG,
        model_config=profile.model,
        overlay_config=overlay_config,
    )
    
    label = args.overlay if args.overlay != "none" else "baseline"
    print(f"\n[{args.etf} {label}] return={result['return']:.1f}% sharpe={result['sharpe']:.2f} maxdd={result['max_dd']:.1f}% trades={result['trades']}")


if __name__ == "__main__":
    main()

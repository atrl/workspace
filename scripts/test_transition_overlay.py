#!/usr/bin/env python3
"""Test transition signal overlay on fixed v19 actor."""

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
from src.soxl_quant.settings import MODEL_DIR, QLIB_DATA_PATH, TEST_SEG, load_profile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", choices=["none", "vix_divergence", "spring_compression", "panic_exhaustion"], default="none")
    parser.add_argument("--threshold", type=float, default=0.3, help="Transition signal threshold")
    parser.add_argument("--boost", type=float, default=0.1, help="Position boost when signal fires")
    parser.add_argument("--regime-threshold", type=float, default=0.6, help="Minimum regime_risk to activate overlay")
    parser.add_argument("--feature-packs", default="", help="Comma-separated feature packs (e.g. transition_signals)")
    return parser.parse_args()


def main():
    args = parse_args()
    init_qlib(QLIB_DATA_PATH)

    profile = load_profile("balanced_rule_residual_v19")
    
    # Parse feature packs
    packs = tuple(args.feature_packs.split(",")) if args.feature_packs else ()
    
    dataset = build_dataset(feature_packs=packs)
    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]

    # Load v19 actor
    ckpt = torch.load(MODEL_DIR / "soxl_balanced_rule_residual_v19_actor.pt", map_location="cpu", weights_only=False)
    _, actor = build_policy(
        int(ckpt["obs_dim"]),
        profile,
        use_moe=ckpt.get("use_moe", False),
        hard_regime_routing=ckpt.get("use_moe", False),
    )
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()

    # Build overlay config
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
        symbol="SOXL",
        test_seg=TEST_SEG,
        model_config=profile.model,
        overlay_config=overlay_config,
    )
    
    label = args.overlay if args.overlay != "none" else "baseline"
    print(f"\n[{label}] return={result['return']:.1f}% sharpe={result['sharpe']:.2f} maxdd={result['max_dd']:.1f}% trades={result['trades']}")

    # Window analysis
    pdf = result["portfolio"]
    account = pdf["account"].astype(float)
    
    # 2020-04
    apr_2020 = account.loc["2020-04-01":"2020-04-30"]
    if len(apr_2020) > 1:
        ret_2020_04 = (apr_2020.iloc[-1] / apr_2020.iloc[0] - 1) * 100
        print(f"  2020-04 return: {ret_2020_04:.1f}%")
    
    # 2022
    yr_2022 = account.loc["2022-01-01":"2022-12-31"]
    if len(yr_2022) > 1:
        ret_2022 = (yr_2022.iloc[-1] / yr_2022.iloc[0] - 1) * 100
        peak = yr_2022.cummax()
        dd_2022 = ((peak - yr_2022) / peak).max() * 100
        print(f"  2022 return: {ret_2022:.1f}%  maxdd: {dd_2022:.1f}%")
    
    # 2023-2024
    yr_2324 = account.loc["2023-01-01":"2024-12-31"]
    if len(yr_2324) > 1:
        ret_2324 = (yr_2324.iloc[-1] / yr_2324.iloc[0] - 1) * 100
        print(f"  2023-24 return: {ret_2324:.1f}%")


if __name__ == "__main__":
    main()

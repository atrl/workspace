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
    parser.add_argument("--overlay", choices=[
        "none", "vix_divergence", "spring_compression", "panic_exhaustion",
        "vix_zscore_delta", "regime_risk_delta", "combined_vix", "fake_breakout",
        "vol_stress", "downside_bull", "dual_regime", "vol_vix_combo", "rv63_stress"
    ], default="none")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--boost", type=float, default=0.1)
    parser.add_argument("--regime-threshold", type=float, default=0.6)
    parser.add_argument("--bull-regime-threshold", type=float, default=0.4)
    parser.add_argument("--regime-mode", choices=["stress", "bull", "dual"], default="stress")
    parser.add_argument("--feature-packs", default="", help="Comma-separated feature packs")
    return parser.parse_args()


def add_derived_features(df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Add derived delta features for overlay testing."""
    df = df.copy()
    df["VIX_ZSCORE_DELTA_5"] = df["vix_zscore"].diff(5).fillna(0.0)
    df["REGIME_RISK_DELTA_5"] = df["regime_risk"].diff(5).fillna(0.0)
    # Fake breakout feature
    ma21 = price.rolling(21).mean()
    df["CLOSE_VS_MA21"] = (price / ma21 - 1.0).fillna(0.0)
    deep_dd = (df["drawdown_63"] < -0.15).astype(float)
    df["CLOSE_VS_MA21_AFTER_DD"] = (df["CLOSE_VS_MA21"] * deep_dd).fillna(0.0)
    return df


def main():
    args = parse_args()
    init_qlib(QLIB_DATA_PATH)

    profile = load_profile("balanced_rule_residual_v19")
    
    packs = tuple(args.feature_packs.split(",")) if args.feature_packs else ()
    dataset = build_dataset(feature_packs=packs)
    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]
    test_df = add_derived_features(test_df, price)

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
            "regime_mode": args.regime_mode,
        }
        if args.overlay in ("dual_regime",):
            overlay_config["bull_regime_threshold"] = args.bull_regime_threshold

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

    pdf = result["portfolio"]
    account = pdf["account"].astype(float)
    
    apr_2020 = account.loc["2020-04-01":"2020-04-30"]
    if len(apr_2020) > 1:
        ret_2020_04 = (apr_2020.iloc[-1] / apr_2020.iloc[0] - 1) * 100
        print(f"  2020-04 return: {ret_2020_04:.1f}%")
    
    yr_2022 = account.loc["2022-01-01":"2022-12-31"]
    if len(yr_2022) > 1:
        ret_2022 = (yr_2022.iloc[-1] / yr_2022.iloc[0] - 1) * 100
        peak = yr_2022.cummax()
        dd_2022 = ((peak - yr_2022) / peak).max() * 100
        print(f"  2022 return: {ret_2022:.1f}%  maxdd: {dd_2022:.1f}%")
    
    yr_2324 = account.loc["2023-01-01":"2024-12-31"]
    if len(yr_2324) > 1:
        ret_2324 = (yr_2324.iloc[-1] / yr_2324.iloc[0] - 1) * 100
        print(f"  2023-24 return: {ret_2324:.1f}%")


if __name__ == "__main__":
    main()

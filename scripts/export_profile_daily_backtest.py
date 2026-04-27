#!/usr/bin/env python3
"""Export daily SOXL backtest rows for a saved actor/profile pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.app.dashboard import write_review_daily_csv
from src.soxl_quant.data import build_dataset, init_qlib, segment_frame
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import MODEL_DIR, QLIB_DATA_PATH, TEST_SEG, load_profile, model_artifact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export daily SOXL backtest rows for a saved profile actor.")
    parser.add_argument("--profile", required=True, help="Profile name, e.g. balanced or balanced_transition_state")
    parser.add_argument("--tag", default="", help="Artifact tag, e.g. wf_2020")
    parser.add_argument("--output", required=True, help="CSV path to write")
    parser.add_argument("--test-start", default="", help="Override test start date (for walk-forward windows)")
    parser.add_argument("--test-end", default="", help="Override test end date (for walk-forward windows)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    test_df = segment_frame(dataset, "test")

    if args.test_start and args.test_end:
        test_df = test_df.loc[args.test_start:args.test_end]
        test_seg = (args.test_start, args.test_end)
    else:
        test_seg = TEST_SEG

    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]

    profile = load_profile(args.profile)
    ckpt = torch.load(model_artifact_path(args.profile, args.tag), map_location="cpu", weights_only=False)
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
        position_budget_config=profile.position_budget,
        execution_config=profile.execution,
        symbol="SOXL",
        test_seg=test_seg,
        model_config=profile.model,
    )
    write_review_daily_csv(
        output,
        result["portfolio"],
        price,
        result["position_history"],
        result.get("trade_signals", []),
    )
    print(output)


if __name__ == "__main__":
    main()

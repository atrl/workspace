#!/usr/bin/env python3
"""Zero-shot generalization backtest for the SOXL-trained actor on other 3x ETFs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Silence warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import (
    build_dataset,
    flatten_feature_frame,
    init_qlib,
    refresh_qlib_binary,
    segment_frame,
)
from src.soxl_quant.rl import build_policy
from src.soxl_quant.settings import (
    ACCOUNT,
    MODEL_DIR,
    TEST_SEG,
    TRAIN_SEG,
    VALID_SEG,
    PositionCapConfig,
    StrategyConfig,
    load_profile,
)

DATA_DIR = ROOT / "data" / "generalization"
RUNTIME_DIR = ROOT / "runtime" / "generalization"
QLIB_RUNTIME = RUNTIME_DIR / "qlib_stores"

ETF_UNIVERSE = {
    "SOXL": {"underlying": "SOXX", "leverage": 3.0},
    "TQQQ": {"underlying": "QQQ", "leverage": 3.0},
    "UPRO": {"underlying": "SPY", "leverage": 3.0},
    "SPXL": {"underlying": "SPY", "leverage": 3.0},
    "TECL": {"underlying": "XLK", "leverage": 3.0},
}


def load_actor(checkpoint_path: Path, profile: StrategyConfig):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    obs_dim = int(ckpt["obs_dim"])
    policy, actor = build_policy(obs_dim, profile)
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor


def run_etf_backtest(
    etf: str,
    underlying: str,
    leverage: float,
    actor,
    position_cap_config: PositionCapConfig,
    test_seg: tuple[str, str],
) -> dict:
    csv_path = DATA_DIR / f"{etf.lower()}_with_underlying_vix.csv"
    qlib_dir = QLIB_RUNTIME / etf.lower()

    print(f"\n=== {etf} ({underlying}) ===")
    print(f"Refreshing qlib binary at {qlib_dir} ...")
    refresh_qlib_binary(
        csv_path=csv_path,
        qlib_dir=qlib_dir,
        leveraged_etf=etf,
        underlying_etf=underlying,
        leverage_ratio=leverage,
    )

    print(f"Initializing qlib ...")
    init_qlib(qlib_dir)

    print(f"Building dataset ...")
    # Use the same train/valid/test splits for comparability.
    dataset = build_dataset(
        leveraged_etf=etf,
        underlying_etf=underlying,
        leverage_ratio=leverage,
        train_seg=TRAIN_SEG,
        valid_seg=VALID_SEG,
        test_seg=test_seg,
    )

    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]

    asset_idx = list(ETF_UNIVERSE.keys()).index(etf)
    print(f"Running backtest on {len(test_df)} rows ({test_df.index[0].date()} ~ {test_df.index[-1].date()}) ...")
    result = backtest_profile(
        actor=actor,
        feature_df=test_df,
        price_series=price,
        position_cap_config=position_cap_config,
        symbol=etf,
        asset_idx=asset_idx,
        test_seg=test_seg,
    )

    return {
        "return": float(result["return"]),
        "sharpe": float(result["sharpe"]),
        "max_dd": float(result["max_dd"]),
        "trades": int(result["trades"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="balanced", help="Profile name in configs/soxl/")
    parser.add_argument("--model", default="soxl_balanced_actor.pt", help="Model checkpoint filename")
    parser.add_argument("--etf", help="Run a single ETF (default: all)")
    args = parser.parse_args()

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    profile = load_profile(args.profile)
    model_path = MODEL_DIR / args.model
    print(f"Loading actor from {model_path} ...")
    actor = load_actor(model_path, profile)

    targets = {args.etf.upper(): ETF_UNIVERSE[args.etf.upper()]} if args.etf else ETF_UNIVERSE

    results = {}
    for etf, meta in targets.items():
        try:
            result = run_etf_backtest(
                etf=etf,
                underlying=meta["underlying"],
                leverage=meta["leverage"],
                actor=actor,
                position_cap_config=profile.position_cap,
                test_seg=TEST_SEG,
            )
            results[etf] = result
            print(f"Result: return={result['return']:.2f}% sharpe={result['sharpe']:.3f} max_dd={result['max_dd']:.2f}% trades={result['trades']}")
        except Exception as exc:
            print(f"ERROR running {etf}: {exc}")
            import traceback
            traceback.print_exc()

    results_path = RUNTIME_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Text report
    lines = ["Generalization Backtest Report", "=" * 60, ""]
    lines.append(f"{'ETF':<8} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8}")
    lines.append("-" * 60)
    for etf, r in results.items():
        lines.append(
            f"{etf:<8} {r['return']:>10.2f}% {r['sharpe']:>8.3f} {r['max_dd']:>10.2f}% {r['trades']:>8d}"
        )
    report_path = RUNTIME_DIR / "report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()

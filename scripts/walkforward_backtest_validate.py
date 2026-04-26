#!/usr/bin/env python3
"""Run walk-forward evaluation using main backtest engine.

Usage:
    PYTHONPATH=. qlib_venv/bin/python scripts/walkforward_backtest_validate.py --profile balanced_rule_residual_v19

This produces corrected walk-forward metrics that align with the main backtest,
unlike the worker eval which has systematic gaps of 80-110pp.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, init_qlib, segment_frame
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import MODEL_DIR, QLIB_DATA_PATH, load_profile
from src.soxl_quant.app.workers import WALK_FORWARD_WINDOWS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, help="Profile name to evaluate")
    parser.add_argument("--output-dir", default="runtime/soxl/wf_backtest", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(args.profile)
    dataset = build_dataset(feature_packs=tuple(profile.feature.packs))
    price_full = dataset.handler.price_frame.droplevel("instrument")["price"]
    test_df_full = segment_frame(dataset, "test").copy()
    test_df_full["VIX_ZSCORE_DELTA_5"] = test_df_full["vix_zscore"].diff(5).fillna(0.0)
    test_df_full["REGIME_RISK_DELTA_5"] = test_df_full["regime_risk"].diff(5).fillna(0.0)

    ckpt = torch.load(MODEL_DIR / f"soxl_{args.profile}_actor.pt", map_location="cpu", weights_only=False)
    _, actor = build_policy(
        int(ckpt["obs_dim"]),
        profile,
        use_moe=ckpt.get("use_moe", False),
        hard_regime_routing=ckpt.get("use_moe", False),
    )
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()

    pos_cap = profile.position_cap
    exec_cfg = profile.execution if hasattr(profile, 'execution') and profile.execution else None
    pos_budget = profile.position_budget if hasattr(profile, 'position_budget') and profile.position_budget else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for wf_name, train_seg, valid_seg, test_seg in WALK_FORWARD_WINDOWS:
        price = price_full.loc[test_df_full.index]
        ma21 = price.rolling(21).mean()
        test_df = test_df_full.copy()
        test_df["CLOSE_VS_MA21"] = (price / ma21 - 1.0).fillna(0.0)
        deep_dd = (test_df["drawdown_63"] < -0.15).astype(float)
        test_df["CLOSE_VS_MA21_AFTER_DD"] = (test_df["CLOSE_VS_MA21"] * deep_dd).fillna(0.0)

        result = backtest_profile(
            actor=actor,
            feature_df=test_df,
            price_series=price,
            position_cap_config=pos_cap,
            position_budget_config=pos_budget,
            execution_config=exec_cfg,
            test_seg=test_seg,
        )

        # Load worker eval for comparison
        worker_path = Path(f"runtime/soxl/wf_workers/{args.profile}-{wf_name}-seed42.json")
        worker = json.loads(worker_path.read_text()) if worker_path.exists() else {}

        summary = {
            "profile": args.profile,
            "window": wf_name,
            "main_backtest": {
                "return": result["return"],
                "sharpe": result["sharpe"],
                "max_dd": result["max_dd"],
                "trades": result["trades"],
            },
            "worker_eval": {
                "return": worker.get("return"),
                "sharpe": worker.get("sharpe"),
            } if worker else None,
            "gap": {
                "return": result["return"] - worker.get("return", 0) if worker else None,
            },
        }
        results[wf_name] = summary

        out_path = output_dir / f"{args.profile}-{wf_name}.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[{wf_name}] Main: Return={result['return']:.1f}% Sharpe={result['sharpe']:.2f} | "
              f"Worker: Return={worker.get('return', 'N/A')}{'%' if worker else ''} | "
              f"Gap: {summary['gap']['return']:+.1f}pp" if summary['gap']['return'] is not None else f"[{wf_name}] Main: Return={result['return']:.1f}% Sharpe={result['sharpe']:.2f}")

    # Save combined report
    report_path = output_dir / f"{args.profile}_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()

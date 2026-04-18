from __future__ import annotations

import argparse
import os
import warnings

from pathlib import Path
from typing import Dict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

from qlib.data import D

from ..backtest import save_results
from ..compat import patch_gym_compat
from ..config import list_profile_names
from ..data import FEATURE_COLUMNS, build_dataset, init_qlib, refresh_qlib_binary, segment_frame
from ..settings import CSV_PATH, QLIB_DATA_PATH, RESULT_PNG, RESULT_TXT, TEST_SEG, load_profile
from .promote import build_promotion_report, persist_promotion_report
from .train import _build_multi_asset_sources, run_profile, set_global_seed
from .workers import run_walk_forward_isolated, run_worker_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOXL qlib-native PPO strategy")
    available_profiles = list_profile_names()
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["balanced", "aggressive"],
        choices=available_profiles,
        help="profiles to train and backtest",
    )
    parser.add_argument("--smoke", action="store_true", help="force short smoke run")
    parser.add_argument("--skip-data-refresh", action="store_true", help="reuse existing qlib binary files")
    parser.add_argument("--walk-forward", action="store_true", help="run expanding-window walk-forward evaluation")
    parser.add_argument("--multi-asset", action="store_true", help="train on pooled episodes from SOXL/TQQQ/UPRO/SPXL/TECL")
    parser.add_argument("--moe", action="store_true", help="use Mixture-of-Experts network architecture")
    parser.add_argument("--ablate-vix", action="store_true", help="zero out VIX features and use regime_risk_no_vix")
    parser.add_argument("--seed", type=int, default=20260411, help="global random seed for Python, NumPy, and Torch")
    parser.add_argument("--worker-run-profile", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-window-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-train-start", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-train-end", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-valid-start", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-valid-end", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-test-start", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-test-end", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_run_profile:
        run_worker_profile(args)
        return
    patch_gym_compat()
    set_global_seed(args.seed)

    print("=" * 72)
    print("SOXL Strategy v2 - qlib native package")
    print("=" * 72)

    if not args.skip_data_refresh:
        csv_df = refresh_qlib_binary(CSV_PATH, QLIB_DATA_PATH)
        print(f"[data] qlib binary refreshed: {len(csv_df)} rows, end={csv_df.index[-1].date()}")
    else:
        print("[data] skip binary refresh")

    init_qlib(QLIB_DATA_PATH)
    probe = D.features(["SOXL", "SOXX"], ["$close"], start_time="2026-03-30", end_time="2026-04-02")
    print(f"[qlib] instruments ok: {sorted(probe.index.get_level_values('instrument').unique().tolist())}")

    dataset = build_dataset(ablate_vix=args.ablate_vix)
    train_df = segment_frame(dataset, "train")
    valid_df = segment_frame(dataset, "valid")
    test_df = segment_frame(dataset, "test")
    price_series = dataset.handler.price_frame.droplevel("instrument")["price"].loc[TEST_SEG[0] : TEST_SEG[1]]

    print(f"[dataset] train={len(train_df)} valid={len(valid_df)} test={len(test_df)} features={len(FEATURE_COLUMNS)}")
    if args.ablate_vix:
        print("[ablation] VIX features zeroed out, regime_risk -> regime_risk_no_vix")

    multi_asset_sources = None
    if args.multi_asset:
        multi_asset_sources = _build_multi_asset_sources(Path(QLIB_DATA_PATH).parent / "multi_asset", verbose=True)

    results: Dict[str, Dict[str, object]] = {}
    walk_forward_results: Dict[str, Dict[str, Dict[str, object]]] = {}

    for name in args.profiles:
        profile = load_profile("smoke" if name == "smoke" else name)
        bt_result = run_profile(
            profile=profile,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            price_series=price_series,
            smoke=args.smoke or name == "smoke",
            seed=args.seed,
            multi_asset_sources=multi_asset_sources,
            use_moe=args.moe,
            ablate_vix=args.ablate_vix,
        )
        results[profile.name] = bt_result
        print(
            f"[test:{profile.name}] return={bt_result['return']:.1f}% "
            f"sharpe={bt_result['sharpe']:.2f} maxdd={bt_result['max_dd']:.1f}% trades={bt_result['trades']} "
            f"model={Path(bt_result['model_path']).name}"
        )
        if args.walk_forward:
            walk_forward_results[profile.name] = run_walk_forward_isolated(
                profile=profile,
                smoke=args.smoke or name == "smoke",
                seed=args.seed,
                skip_data_refresh=True,
                multi_asset=args.multi_asset,
                use_moe=args.moe,
                ablate_vix=args.ablate_vix,
            )
            promotion_report = build_promotion_report(
                profile,
                bt_result["slices"],
                bt_result["test_proxy"],
                walk_forward_results=walk_forward_results[profile.name],
            )
            bt_result["promotion"] = promotion_report
            persist_promotion_report(Path(bt_result["summary_path"]), promotion_report)

    save_results(results)
    print(f"\n[done] chart={RESULT_PNG.name} summary={RESULT_TXT.name}")
    for name, result in results.items():
        print(
            f"  - {name}: {result['return']:.1f}% | Sharpe {result['sharpe']:.2f} | "
            f"MaxDD {result['max_dd']:.1f}% | trades {result['trades']}"
        )
        for wf_name, wf_result in walk_forward_results.get(name, {}).items():
            print(
                f"    {wf_name}: {wf_result['return']:.1f}% | Sharpe {wf_result['sharpe']:.2f} | "
                f"MaxDD {wf_result['max_dd']:.1f}% | trades {wf_result['trades']}"
            )
        promotion = result.get("promotion", {})
        if promotion:
            status = "eligible" if promotion.get("eligible") else "blocked"
            print(f"    promotion: {status}")


__all__ = ["main", "parse_args", "run_walk_forward_isolated", "run_worker_profile"]

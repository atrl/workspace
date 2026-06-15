from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from ..config import list_profile_names
from ..settings import load_profile
from .artifacts import final_model_retention_payload, persist_final_model_retention_decision
from .bootstrap import bootstrap_runtime
from .promote import build_promotion_report, persist_promotion_report
from .reporting import persist_and_print_results, print_profile_result
from .train import run_profile
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
    _dataset, train_df, valid_df, test_df, price_series, multi_asset_sources = bootstrap_runtime(args)

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
        print_profile_result(profile.name, bt_result)
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
            final_retention = final_model_retention_payload(
                profile=profile,
                model_path=Path(str(bt_result["model_path"])),
                promotion_report=promotion_report,
            )
            bt_result["final_model_retention_decision"] = final_retention
            persist_final_model_retention_decision(Path(bt_result["summary_path"]), final_retention)

    persist_and_print_results(results, walk_forward_results)


__all__ = ["main", "parse_args", "run_walk_forward_isolated", "run_worker_profile"]

#!/usr/bin/env python3
"""Run fixed-actor execution-overlay backtests across all supported 3x ETFs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, init_qlib, refresh_qlib_binary, segment_frame
from src.soxl_quant.policy.evaluation import evaluate_split
from src.soxl_quant.runtime import bundle_position_cap_config, load_actor_bundle
from src.soxl_quant.settings import ETF_UNIVERSE, ExecutionConfig, TEST_SEG, TRAIN_SEG, VALID_SEG, load_profile

DATA_DIR = ROOT / "data" / "generalization"
RUNTIME_DIR = ROOT / "runtime" / "generalization"
QLIB_RUNTIME = RUNTIME_DIR / "qlib_stores"
SCENARIOS = {
    "crash_rebound_2020": ("2020-02-01", "2020-06-30"),
    "bull_trend_2020_2021": ("2020-07-01", "2021-12-31"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "recovery_2023_2024": ("2023-01-01", "2024-12-31"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generalization backtest with execution overlays")
    parser.add_argument("--profile", required=True, help="Base profile name for dataset/slice settings")
    parser.add_argument("--model", required=True, help="Path to saved actor bundle")
    parser.add_argument("--variants-json", required=True, help="JSON file mapping overlay name -> execution config payload")
    parser.add_argument("--output", required=True, help="Path to write the JSON result")
    return parser.parse_args()


def load_execution_variants(path: Path) -> dict[str, ExecutionConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("variants-json must contain a non-empty object")
    variants: dict[str, ExecutionConfig] = {}
    for name, overrides in payload.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"variant {name} must map to an object")
        variants[name] = ExecutionConfig(**overrides)
    return variants


def build_etf_dataset(etf: str, underlying: str, leverage: float):
    csv_path = DATA_DIR / f"{etf.lower()}_with_underlying_vix.csv"
    qlib_dir = QLIB_RUNTIME / etf.lower()
    refresh_qlib_binary(
        csv_path=csv_path,
        qlib_dir=qlib_dir,
        leveraged_etf=etf,
        underlying_etf=underlying,
        leverage_ratio=leverage,
    )
    init_qlib(qlib_dir)
    dataset = build_dataset(
        leveraged_etf=etf,
        underlying_etf=underlying,
        leverage_ratio=leverage,
        train_seg=TRAIN_SEG,
        valid_seg=VALID_SEG,
        test_seg=TEST_SEG,
    )
    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]
    return test_df, price


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    actor, bundle_path, bundle = load_actor_bundle(args.profile, model_path=Path(args.model))
    position_cap = bundle_position_cap_config(bundle, args.profile)
    execution_variants = load_execution_variants(Path(args.variants_json))

    results: dict[str, dict[str, object]] = {}
    for etf, meta in ETF_UNIVERSE.items():
        test_df, price = build_etf_dataset(etf, meta["underlying"], meta["leverage"])
        asset_idx = list(ETF_UNIVERSE.keys()).index(etf)
        results[etf] = {}
        for variant_name, execution_cfg in execution_variants.items():
            bt = backtest_profile(
                actor=actor,
                feature_df=test_df,
                price_series=price,
                position_cap_config=position_cap,
                execution_config=execution_cfg,
                symbol=etf,
                asset_idx=asset_idx,
                test_seg=TEST_SEG,
            )
            slices = {
                name: evaluate_split(
                    actor,
                    test_df.loc[start:end],
                    position_cap_config=position_cap,
                    execution_config=execution_cfg,
                    asset_idx=asset_idx,
                    stress_threshold=profile.selection_score.stress_threshold,
                )
                for name, (start, end) in SCENARIOS.items()
            }
            results[etf][variant_name] = {
                "execution": execution_cfg.__dict__,
                "test_proxy": {k: bt[k] for k in ("return", "sharpe", "max_dd", "trades")},
                "scenarios": {
                    key: {metric: value[metric] for metric in ("return", "sharpe", "max_dd", "avg_pos")}
                    for key, value in slices.items()
                },
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "profile": args.profile,
                "model": str(bundle_path),
                "scenarios": SCENARIOS,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[execution-generalization] profile={args.profile} model={bundle_path.name}")
    for etf, variants in results.items():
        print(f"  - {etf}")
        for variant_name, payload in variants.items():
            proxy = payload["test_proxy"]
            bear = payload["scenarios"]["bear_2022"]
            print(
                f"    {variant_name}: proxy={proxy['return']:.1f}% sharpe={proxy['sharpe']:.3f} "
                f"maxdd={proxy['max_dd']:.2f}% trades={proxy['trades']} "
                f"| bear_2022={bear['return']:.1f}% maxdd={bear['max_dd']:.2f}% avg_pos={bear['avg_pos']:.3f}"
            )
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()

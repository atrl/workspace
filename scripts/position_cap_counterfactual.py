#!/usr/bin/env python3
"""Run fixed-actor counterfactual backtests across multiple position-cap variants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, init_qlib, segment_frame
from src.soxl_quant.policy.evaluation import evaluate_split
from src.soxl_quant.runtime import bundle_position_cap_config, load_actor_bundle
from src.soxl_quant.settings import QLIB_DATA_PATH, TEST_SEG, PositionCapConfig, load_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-actor counterfactual position-cap backtests")
    parser.add_argument("--profile", required=True, help="Profile name used for dataset + default config")
    parser.add_argument("--model", required=True, help="Path to saved actor bundle")
    parser.add_argument("--variants-json", required=True, help="JSON file mapping variant name -> partial position_cap payload")
    parser.add_argument("--output", required=True, help="Path to write the JSON result")
    return parser.parse_args()


def build_config(base: PositionCapConfig, overrides: dict[str, object]) -> PositionCapConfig:
    payload = dict(base.__dict__)
    payload.update(overrides)
    return PositionCapConfig(**payload)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    variants_path = Path(args.variants_json)

    profile = load_profile(args.profile)
    actor, bundle_path, bundle = load_actor_bundle(args.profile, model_path=Path(args.model))
    base_cfg = bundle_position_cap_config(bundle, args.profile)

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    test_df = segment_frame(dataset, "test")
    price_series = dataset.handler.price_frame.droplevel("instrument")["price"].loc[TEST_SEG[0] : TEST_SEG[1]]

    variant_payload = json.loads(variants_path.read_text(encoding="utf-8"))
    if not isinstance(variant_payload, dict) or not variant_payload:
        raise ValueError("variants-json must contain a non-empty object")

    results: dict[str, object] = {}
    for name, overrides in variant_payload.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"variant {name} must map to an object")
        cfg = build_config(base_cfg, overrides)
        bt = backtest_profile(actor, test_df, price_series, position_cap_config=cfg, asset_idx=0)
        slices = {
            key: evaluate_split(
                actor,
                test_df.loc[start:end],
                position_cap_config=cfg,
                asset_idx=0,
                stress_threshold=profile.selection_score.stress_threshold,
            )
            for key, (start, end) in {
                "test_2020": ("2020-01-01", "2020-12-31"),
                "test_2022": ("2022-01-01", "2022-12-31"),
                "test_2023_2024": ("2023-01-01", "2024-12-31"),
            }.items()
        }
        results[name] = {
            "position_cap": cfg.__dict__,
            "test_proxy": {k: bt[k] for k in ("return", "sharpe", "max_dd", "trades")},
            "slices": {
                key: {metric: value[metric] for metric in ("return", "sharpe", "max_dd", "avg_pos")}
                for key, value in slices.items()
            },
        }

    payload = {
        "profile": args.profile,
        "model": str(bundle_path),
        "variants": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[counterfactual] profile={args.profile} model={bundle_path.name}")
    for name, row in results.items():
        proxy = row["test_proxy"]
        test_2022 = row["slices"]["test_2022"]
        print(
            f"  - {name}: proxy={proxy['return']:.1f}% sharpe={proxy['sharpe']:.3f} maxdd={proxy['max_dd']:.2f}% "
            f"| 2022={test_2022['return']:.1f}% maxdd={test_2022['max_dd']:.2f}% avg_pos={test_2022['avg_pos']:.3f}"
        )
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
